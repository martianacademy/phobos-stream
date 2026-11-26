"""Per-block `.safetensors` mmap & hybrid Unified-Memory streamer for Apple Silicon MLX."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from phobos_stream.storage.mmap_loader import MMapTensorLoader


class MLXBlockStreamer:
    """Streams MLX transformer blocks on Apple Silicon with optional Hybrid Residency & FBCache.

    Capabilities:
    1. Pure `mmap` streaming (`memory_budget_mb=None`): forces `mx.eval` at each block boundary
       and evicts every block's weights back to `(0,)` placeholders (`~1.2-1.6 GB` Peak Metal).
    2. Hybrid Unified-Memory Residency (`memory_budget_mb > 0`): keeps the first `K` blocks
       permanently resident in Metal memory up to the budget (reserving 1-block scratch headroom)
       and `mmap`-streams only the remaining `N - K` overflow blocks.
    3. Zero-MMap First-Block Residual Cache (`cache_rel_l1_thresh > 0`): probes `block[0]`
       residual `delta_0 = out_0 - x` and skips `blocks[1 .. N-1]` (both Metal compute and SSD
       `mmap` reads) when relative L1 change stays below `cache_rel_l1_thresh`.
    """

    def __init__(
        self,
        blocks: Sequence[Any],
        safetensors_paths: Sequence[Union[str, Path]],
        block_prefix_template: str = "layers.{i}",
        memory_budget_mb: Optional[float] = None,
        cache_rel_l1_thresh: float = 0.0,
        cache_warmup_steps: int = 2,
        cache_max_consecutive_skips: int = 2,
    ) -> None:
        if not safetensors_paths:
            raise ValueError("MLXBlockStreamer requires at least one `.safetensors` path.")
        if cache_rel_l1_thresh < 0.0:
            raise ValueError(f"cache_rel_l1_thresh must be >= 0.0, got {cache_rel_l1_thresh}.")

        self.blocks = list(blocks)
        self.safetensors_paths = [Path(p) for p in safetensors_paths]
        self.block_prefix_template = block_prefix_template
        self.memory_budget_mb = memory_budget_mb
        self.cache_rel_l1_thresh = float(cache_rel_l1_thresh)
        self.cache_warmup_steps = max(1, int(cache_warmup_steps))
        self.cache_max_consecutive_skips = max(1, int(cache_max_consecutive_skips))

        self.bytes_streamed: int = 0
        self.blocks_executed: int = 0
        self.total_steps: int = 0
        self.skipped_steps: int = 0
        self.computed_steps: int = 0

        self._loaders: List[MMapTensorLoader] = []
        self._block_specs: Dict[int, List[Tuple[str, MMapTensorLoader, str, Any]]] = {}
        self._block_bytes: Dict[int, int] = {}
        self.resident_indices: Set[int] = set()
        self._closed = False

        # First-Block Residual Cache state across `run_sequence` steps
        self._prev_delta_0: Optional[Any] = None
        self._cached_residual_rest: Optional[Any] = None
        self._accumulated_rel_diff: float = 0.0
        self._consecutive_skips: int = 0
        self._steps_seen: int = 0
        self._last_shape: Optional[Tuple[int, ...]] = None

        self._index_and_partition_blocks()

    @property
    def tier_summary(self) -> Dict[str, int]:
        """Return count of Metal-resident vs NVMe-mmap streamed blocks."""
        n_res = len(self.resident_indices)
        return {"metal_resident": n_res, "nvme_mmap": len(self.blocks) - n_res}

    @property
    def hit_rate(self) -> float:
        return (self.skipped_steps / self.total_steps) if self.total_steps > 0 else 0.0

    def reset_cache_state(self) -> None:
        """Clear First-Block Residual Cache tensors between independent generation trajectories."""
        self._prev_delta_0 = None
        self._cached_residual_rest = None
        self._accumulated_rel_diff = 0.0
        self._consecutive_skips = 0
        self._steps_seen = 0
        self._last_shape = None

    def _build_index(self) -> Dict[str, Tuple[MMapTensorLoader, str]]:
        index: Dict[str, Tuple[MMapTensorLoader, str]] = {}
        for path in self.safetensors_paths:
            loader = MMapTensorLoader(path)
            self._loaders.append(loader)
            for key in loader.get_tensor_names():
                index[key] = (loader, key)
                for pfx in ("model.diffusion_model.", "diffusion_model.", "transformer.", "model."):
                    if key.startswith(pfx):
                        index.setdefault(key[len(pfx) :], (loader, key))
        return index

    def _index_and_partition_blocks(self) -> None:
        import mlx.core as mx
        from mlx.utils import tree_flatten, tree_unflatten

        try:
            st_index = self._build_index()
            for idx, blk in enumerate(self.blocks):
                prefix = self.block_prefix_template.format(i=idx)
                flat_params = tree_flatten(blk.parameters())
                specs: List[Tuple[str, MMapTensorLoader, str, Any]] = []
                blk_bytes = 0

                for rel_key, arr in flat_params:
                    full_key = f"{prefix}.{rel_key}" if prefix else rel_key
                    if full_key not in st_index:
                        raise KeyError(
                            f"Missing MLX block weight '{full_key}' in {[p.name for p in self.safetensors_paths]}."
                        )
                    loader, real_key = st_index[full_key]
                    specs.append((rel_key, loader, real_key, arr.dtype))
                    blk_bytes += loader.get_tensor_byte_size(real_key)

                self._block_specs[idx] = specs
                self._block_bytes[idx] = blk_bytes

            # Decide which blocks stay resident in Metal memory vs evicted to `(0,)`
            if self.memory_budget_mb is not None and self.memory_budget_mb > 0.0 and self.blocks:
                budget_bytes = int(self.memory_budget_mb * 1024.0 * 1024.0)
                total_bytes = sum(self._block_bytes.values())
                max_blk = max(self._block_bytes.values()) if self._block_bytes else 0
                if total_bytes <= budget_bytes:
                    self.resident_indices = set(range(len(self.blocks)))
                else:
                    usable = max(0, budget_bytes - max_blk)
                    used = 0
                    for idx in range(len(self.blocks)):
                        b_sz = self._block_bytes[idx]
                        if used + b_sz <= usable:
                            self.resident_indices.add(idx)
                            used += b_sz
                        else:
                            break

            for idx, blk in enumerate(self.blocks):
                specs = self._block_specs[idx]
                if idx in self.resident_indices:
                    loaded = [
                        (rel_key, loader.read_mlx_array(real_key, target_dtype=orig_dtype))
                        for rel_key, loader, real_key, orig_dtype in specs
                    ]
                    blk.update(tree_unflatten(loaded))
                    mx.eval(blk.parameters())
                else:
                    empty = [(rel_key, mx.zeros((0,), dtype=orig_dtype)) for rel_key, _, _, orig_dtype in specs]
                    blk.update(tree_unflatten(empty))
                    mx.eval(blk.parameters())

            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
        except Exception:
            self.close()
            raise

    def evict_block(self, block_idx: int) -> None:
        """Replace a non-resident block's active MLX parameter arrays with 0-element placeholders."""
        if block_idx in self.resident_indices:
            return
        import mlx.core as mx
        from mlx.utils import tree_unflatten

        specs = self._block_specs[block_idx]
        blk = self.blocks[block_idx]
        empty_leaves = [(rel_key, mx.zeros((0,), dtype=orig_dtype)) for rel_key, _, _, orig_dtype in specs]
        blk.update(tree_unflatten(empty_leaves))
        mx.eval(blk.parameters())
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

    def stream_block_call(self, block_idx: int, *args: Any, **kwargs: Any) -> Any:
        """Execute `block_idx` (directly if Metal-resident, or materialize -> eval -> evict if mmap)."""
        import mlx.core as mx
        from mlx.utils import tree_flatten, tree_unflatten

        if self._closed:
            raise RuntimeError("MLXBlockStreamer is already closed.")

        blk = self.blocks[block_idx]
        if block_idx in self.resident_indices:
            out = blk(*args, **kwargs)
            out_arrays = [v for _, v in tree_flatten(out) if isinstance(v, mx.array)]
            if out_arrays:
                mx.eval(*out_arrays)
            self.blocks_executed += 1
            return out

        specs = self._block_specs[block_idx]
        loaded_leaves: List[Tuple[str, Any]] = []
        bytes_this_block = 0
        for rel_key, loader, real_key, orig_dtype in specs:
            arr = loader.read_mlx_array(real_key, target_dtype=orig_dtype)
            loaded_leaves.append((rel_key, arr))
            bytes_this_block += loader.get_tensor_byte_size(real_key)

        blk.update(tree_unflatten(loaded_leaves))
        mx.eval(blk.parameters())
        del loaded_leaves

        out = blk(*args, **kwargs)
        out_arrays = [v for _, v in tree_flatten(out) if isinstance(v, mx.array)]
        if out_arrays:
            mx.eval(*out_arrays)

        self.bytes_streamed += bytes_this_block
        self.blocks_executed += 1
        self.evict_block(block_idx)
        return out

    def run_sequence(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute all blocks sequentially, applying First-Block Residual Cache when enabled."""
        import mlx.core as mx

        n_blocks = len(self.blocks)
        if n_blocks == 0:
            return x
        if self.cache_rel_l1_thresh <= 0.0 or n_blocks < 2 or not isinstance(x, mx.array):
            cur = x
            for idx in range(n_blocks):
                cur = self.stream_block_call(idx, cur, *args, **kwargs)
            return cur

        cur_shape = tuple(int(d) for d in x.shape)
        if self._last_shape is not None and cur_shape != self._last_shape:
            self.reset_cache_state()
        self._last_shape = cur_shape

        out_0 = self.stream_block_call(0, x, *args, **kwargs)
        if not isinstance(out_0, mx.array) or out_0.shape != x.shape:
            cur = out_0
            for idx in range(1, n_blocks):
                cur = self.stream_block_call(idx, cur, *args, **kwargs)
            return cur

        delta_0 = out_0 - x
        mx.eval(delta_0)
        self._steps_seen += 1
        self.total_steps += 1

        can_skip = (
            self._steps_seen > self.cache_warmup_steps
            and self._prev_delta_0 is not None
            and self._cached_residual_rest is not None
            and self._consecutive_skips < self.cache_max_consecutive_skips
        )
        if can_skip:
            d_cur = delta_0.astype(mx.float32)
            d_prev = self._prev_delta_0.astype(mx.float32)
            denom = float(mx.mean(mx.abs(d_prev)).item()) + 1e-6
            rel_diff = float(mx.mean(mx.abs(d_cur - d_prev)).item()) / denom
            if (self._accumulated_rel_diff + rel_diff) < self.cache_rel_l1_thresh:
                self._accumulated_rel_diff += rel_diff
                self._consecutive_skips += 1
                self.skipped_steps += 1
                approx = (out_0 + self._cached_residual_rest).astype(out_0.dtype)
                mx.eval(approx)
                return approx

        cur = out_0
        for idx in range(1, n_blocks):
            cur = self.stream_block_call(idx, cur, *args, **kwargs)

        res_rest = cur - out_0
        mx.eval(res_rest)
        self._prev_delta_0 = delta_0
        self._cached_residual_rest = res_rest
        self._accumulated_rel_diff = 0.0
        self._consecutive_skips = 0
        self.computed_steps += 1
        return cur

    def restore_all(self) -> None:
        """Reload full weights into all MLX blocks from `.safetensors`."""
        import mlx.core as mx
        from mlx.utils import tree_unflatten

        for idx, blk in enumerate(self.blocks):
            if idx in self.resident_indices:
                continue
            leaves = [
                (rel_key, loader.read_mlx_array(real_key, target_dtype=orig_dtype))
                for rel_key, loader, real_key, orig_dtype in self._block_specs[idx]
            ]
            blk.update(tree_unflatten(leaves))
            mx.eval(blk.parameters())

    def close(self) -> None:
        """Close all underlying `MMapTensorLoader` file handles."""
        for loader in self._loaders:
            loader.close()
        self._loaders.clear()
        self._closed = True

    def __enter__(self) -> "MLXBlockStreamer":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

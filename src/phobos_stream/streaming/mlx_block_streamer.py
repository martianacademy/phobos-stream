"""Per-block `.safetensors` mmap weight streamer for Apple Silicon MLX models."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from phobos_stream.storage.mmap_loader import MMapTensorLoader


class MLXBlockStreamer:
    """Streams MLX transformer blocks one-by-one from `.safetensors` mmap on Apple Silicon.

    On Unified Memory Macs, MLX evaluates computation graphs lazily by default; if all
    blocks run without evaluation boundaries, MLX holds every block's weights in Metal
    memory simultaneously. `MLXBlockStreamer` forces `mx.eval` at each block boundary
    and immediately evicts the finished block's weight arrays back to `(0,)` placeholders.
    """

    def __init__(
        self,
        blocks: Sequence[Any],
        safetensors_paths: Sequence[Union[str, Path]],
        block_prefix_template: str = "layers.{i}",
    ) -> None:
        if not safetensors_paths:
            raise ValueError("MLXBlockStreamer requires at least one `.safetensors` path.")

        self.blocks = list(blocks)
        self.safetensors_paths = [Path(p) for p in safetensors_paths]
        self.block_prefix_template = block_prefix_template
        self.bytes_streamed: int = 0
        self.blocks_executed: int = 0

        self._loaders: List[MMapTensorLoader] = []
        self._block_specs: Dict[int, List[Tuple[str, MMapTensorLoader, str, Any]]] = {}
        self._closed = False

        self._index_and_evict_all()

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

    def _index_and_evict_all(self) -> None:
        import mlx.core as mx
        from mlx.utils import tree_flatten, tree_unflatten

        try:
            st_index = self._build_index()
            for idx, blk in enumerate(self.blocks):
                prefix = self.block_prefix_template.format(i=idx)
                flat_params = tree_flatten(blk.parameters())
                specs: List[Tuple[str, MMapTensorLoader, str, Any]] = []
                empty_leaves: List[Tuple[str, Any]] = []

                for rel_key, arr in flat_params:
                    full_key = f"{prefix}.{rel_key}" if prefix else rel_key
                    if full_key not in st_index:
                        raise KeyError(
                            f"Missing MLX block weight '{full_key}' in {[p.name for p in self.safetensors_paths]}."
                        )
                    loader, real_key = st_index[full_key]
                    specs.append((rel_key, loader, real_key, arr.dtype))
                    empty_leaves.append((rel_key, mx.zeros((0,), dtype=arr.dtype)))

                self._block_specs[idx] = specs
                blk.update(tree_unflatten(empty_leaves))
                mx.eval(blk.parameters())

            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
        except Exception:
            self.close()
            raise

    def evict_block(self, block_idx: int) -> None:
        """Replace a block's active MLX parameter arrays with 0-element placeholders."""
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
        """Materialize `block_idx` weights, execute its forward pass, evaluate, and evict."""
        import mlx.core as mx
        from mlx.utils import tree_flatten, tree_unflatten

        if self._closed:
            raise RuntimeError("MLXBlockStreamer is already closed.")

        specs = self._block_specs[block_idx]
        blk = self.blocks[block_idx]

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
        """Stream and execute all indexed blocks sequentially on input `x`."""
        cur = x
        for idx in range(len(self.blocks)):
            cur = self.stream_block_call(idx, cur, *args, **kwargs)
        return cur

    def restore_all(self) -> None:
        """Reload full weights into all MLX blocks from `.safetensors`."""
        import mlx.core as mx
        from mlx.utils import tree_unflatten

        for idx, blk in enumerate(self.blocks):
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

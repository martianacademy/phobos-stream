"""Universal hook-based block streamer for PyTorch Diffusion Transformers, T5/CLIP, & UNets."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple, Union

from phobos_stream.storage.mmap_loader import MMapTensorLoader
from phobos_stream.streaming.block_discovery import (
    KNOWN_BLOCK_CONTAINERS,
    STRIP_MODEL_PREFIXES,
    assign_tensor,
    build_safetensors_index,
    discover_block_modules,
    iter_persistent_module_tensors,
    lookup_safetensors_key,
    move_non_persistent_buffers,
)
from phobos_stream.streaming.budget_planner import BlockTier, HybridBudgetPlan, MemoryBudgetPlanner
from phobos_stream.streaming.first_block_cache import FirstBlockResidualCache


class BlockStreamMode(str, Enum):
    """Execution mode for transformer block memory management."""

    STAGE_ONLY = "stage_only"
    CPU_OFFLOAD = "cpu_offload"
    MMAP_STREAM = "mmap_stream"
    HYBRID_BUDGET = "hybrid_budget"


class TorchBlockStreamer:
    """Streams PyTorch blocks across 3 tiers (`GPU_RESIDENT`, `CPU_PINNED`, `NVME_MMAP`)."""

    def __init__(
        self,
        model: Any,
        compute_device: Union[str, Any] = "auto",
        mode: BlockStreamMode = BlockStreamMode.CPU_OFFLOAD,
        safetensors_paths: Optional[Sequence[Union[str, Path]]] = None,
        block_attr_names: Optional[Sequence[str]] = None,
        pin_memory: bool = True,
        prefetch: bool = True,
        vram_budget_mb: Optional[float] = None,
        ram_budget_mb: Optional[float] = None,
        activation_reserve_mb: int = 1536,
        cache_rel_l1_thresh: float = 0.0,
        cache_warmup_steps: int = 2,
        cache_max_consecutive_skips: int = 3,
    ) -> None:
        import torch

        self.model = model
        self._auto_device = str(compute_device) == "auto"
        if self._auto_device:
            if torch.cuda.is_available():
                self.compute_device = torch.device("cuda:0")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.compute_device = torch.device("mps")
            else:
                self.compute_device = torch.device("cpu")
        else:
            self.compute_device = torch.device(compute_device)

        self.mode = BlockStreamMode(mode)
        self.safetensors_paths = [Path(p) for p in (safetensors_paths or [])]
        self.block_attr_names = block_attr_names
        self.pin_memory = pin_memory
        self.prefetch = prefetch
        self.vram_budget_mb = vram_budget_mb
        self.ram_budget_mb = ram_budget_mb
        self.activation_reserve_mb = activation_reserve_mb
        self.residual_cache: Optional[FirstBlockResidualCache] = (
            FirstBlockResidualCache(
                rel_l1_thresh=cache_rel_l1_thresh,
                warmup_steps=cache_warmup_steps,
                max_consecutive_skips=cache_max_consecutive_skips,
            )
            if cache_rel_l1_thresh > 0.0
            else None
        )

        self.blocks: List[Tuple[str, Any]] = []
        self.budget_plan: Optional[HybridBudgetPlan] = None
        self._effective_tiers: List[BlockTier] = []
        self.bytes_streamed: int = 0
        self.blocks_executed: int = 0

        self._hooks: List[Any] = []
        self._loaders: List[MMapTensorLoader] = []
        self._cpu_shadow: Dict[int, List[Tuple[str, bool, Any]]] = {}
        self._prefetch_cache: Dict[int, Dict[str, Any]] = {}
        self._prefetch_stream: Optional[Any] = None
        self._inflight_events: Deque[Any] = deque()
        self._mmap_map: Dict[int, List[Tuple[str, bool, MMapTensorLoader, str, Any]]] = {}
        self._attached = False

    @property
    def tier_summary(self) -> Dict[str, int]:
        """Return block residency distribution across GPU, CPU, and NVMe."""
        tiers = self._effective_tiers
        if not tiers and self.budget_plan is not None:
            return self.budget_plan.summary()
        if not tiers:
            if self.mode == BlockStreamMode.CPU_OFFLOAD:
                return {BlockTier.GPU_RESIDENT.value: 0, BlockTier.CPU_PINNED.value: len(self.blocks), BlockTier.NVME_MMAP.value: 0}
            if self.mode == BlockStreamMode.MMAP_STREAM:
                return {BlockTier.GPU_RESIDENT.value: 0, BlockTier.CPU_PINNED.value: 0, BlockTier.NVME_MMAP.value: len(self.blocks)}
            return {BlockTier.GPU_RESIDENT.value: len(self.blocks), BlockTier.CPU_PINNED.value: 0, BlockTier.NVME_MMAP.value: 0}
        return {
            BlockTier.GPU_RESIDENT.value: sum(1 for t in tiers if t == BlockTier.GPU_RESIDENT),
            BlockTier.CPU_PINNED.value: sum(1 for t in tiers if t == BlockTier.CPU_PINNED),
            BlockTier.NVME_MMAP.value: sum(1 for t in tiers if t == BlockTier.NVME_MMAP),
        }

    def _resolve_active_device(self, inputs: Any, kwargs: Optional[Dict[str, Any]] = None) -> Any:
        import torch

        if not self._auto_device:
            return self.compute_device
        candidates: List[Any] = list(inputs) if isinstance(inputs, (tuple, list)) else [inputs]
        if kwargs:
            candidates.extend(kwargs.values())
        for item in candidates:
            if isinstance(item, torch.Tensor):
                return item.device
            if isinstance(item, (tuple, list)):
                for sub in item:
                    if isinstance(sub, torch.Tensor):
                        return sub.device
        return self.compute_device

    def attach(self) -> "TorchBlockStreamer":
        """Attach per-block hooks according to `CPU_OFFLOAD`, `MMAP_STREAM`, or `HYBRID_BUDGET`."""
        if self._attached or self.mode == BlockStreamMode.STAGE_ONLY:
            return self

        self.blocks = discover_block_modules(self.model, self.block_attr_names)
        if not self.blocks:
            raise ValueError("No repeating transformer/UNet block ModuleList found on model.")

        block_prefixes = tuple(f"{pfx}." for pfx, _ in self.blocks)

        try:
            if self.safetensors_paths:
                self._loaders, st_index = build_safetensors_index(self.safetensors_paths)
            else:
                st_index = {}

            self._place_non_block_stem_and_head(block_prefixes, st_index)
            move_non_persistent_buffers(self.model, self.compute_device)

            if self.mode == BlockStreamMode.CPU_OFFLOAD:
                tiers = [BlockTier.CPU_PINNED] * len(self.blocks)
            elif self.mode == BlockStreamMode.MMAP_STREAM:
                if not st_index:
                    raise ValueError("MMAP_STREAM mode requires at least one `.safetensors` file path.")
                tiers = [BlockTier.NVME_MMAP] * len(self.blocks)
            else:
                block_sizes = [self._estimate_block_bytes(pfx, blk, st_index) for pfx, blk in self.blocks]
                self.budget_plan = MemoryBudgetPlanner.plan_blocks(
                    block_sizes_bytes=block_sizes,
                    vram_budget_mb=self.vram_budget_mb,
                    ram_budget_mb=self.ram_budget_mb,
                    compute_device=self.compute_device,
                    has_mmap_backing=bool(st_index),
                    activation_reserve_mb=self.activation_reserve_mb,
                )
                tiers = list(self.budget_plan.tiers)

            self._configure_block_tiers(tiers, st_index)
            if self.residual_cache is not None and self.residual_cache.enabled:
                first_post0_cpu = next(
                    (i for i, t in enumerate(self._effective_tiers) if i >= 1 and t == BlockTier.CPU_PINNED),
                    None,
                )
                miss_cb = (
                    (lambda: self._prefetch_cpu_block(first_post0_cpu))
                    if (first_post0_cpu is not None and self._prefetch_stream is not None)
                    else None
                )
                self.residual_cache.attach(self.model, blocks=self.blocks, on_miss_callback=miss_cb)
        except Exception:
            for loader in self._loaders:
                loader.close()
            self._loaders.clear()
            raise

        self._attached = True
        return self

    def _estimate_block_bytes(
        self,
        blk_prefix: str,
        blk: Any,
        st_index: Dict[str, Tuple[MMapTensorLoader, str]],
    ) -> int:
        total = 0
        for rel_name, _, tensor_obj in iter_persistent_module_tensors(blk):
            if tensor_obj.numel() > 0 and not getattr(tensor_obj, "is_meta", False):
                total += int(tensor_obj.numel() * tensor_obj.element_size())
            else:
                resolved = lookup_safetensors_key(st_index, f"{blk_prefix}.{rel_name}")
                if resolved is not None:
                    loader, real_key = resolved
                    total += loader.get_tensor_byte_size(real_key)
        return max(1, total)

    def _place_non_block_stem_and_head(
        self,
        block_prefixes: Tuple[str, ...],
        st_index: Dict[str, Tuple[MMapTensorLoader, str]],
    ) -> None:
        for full_name, is_param, tensor_obj in iter_persistent_module_tensors(self.model):
            if full_name.startswith(block_prefixes):
                continue
            if getattr(tensor_obj, "is_meta", False) or tensor_obj.numel() == 0:
                resolved = lookup_safetensors_key(st_index, full_name)
                if resolved is None:
                    raise KeyError(f"Missing non-block tensor '{full_name}' in safetensors index.")
                loader, real_key = resolved
                loaded = loader.read_torch_tensor(real_key, device=str(self.compute_device), target_dtype=tensor_obj.dtype)
                assign_tensor(self.model, full_name, loaded, is_param)
            elif tensor_obj.device != self.compute_device:
                assign_tensor(self.model, full_name, tensor_obj.data.to(self.compute_device), is_param)

    def _can_mmap_block(
        self,
        blk_prefix: str,
        blk: Any,
        st_index: Dict[str, Tuple[MMapTensorLoader, str]],
    ) -> bool:
        if not st_index:
            return False
        for rel_name, _, tensor_obj in iter_persistent_module_tensors(blk):
            resolved = lookup_safetensors_key(st_index, f"{blk_prefix}.{rel_name}")
            if resolved is None:
                return False
            loader, real_key = resolved
            if not getattr(tensor_obj, "is_meta", False) and tensor_obj.numel() > 0:
                if tuple(loader.get_tensor_metadata(real_key)["shape"]) != tuple(tensor_obj.shape):
                    return False
        return True

    def _configure_block_tiers(
        self,
        tiers: Sequence[BlockTier],
        st_index: Dict[str, Tuple[MMapTensorLoader, str]],
    ) -> None:
        import torch

        effective: List[BlockTier] = []
        for (blk_prefix, blk), tier in zip(self.blocks, tiers):
            if tier == BlockTier.NVME_MMAP and self.mode == BlockStreamMode.HYBRID_BUDGET:
                if not self._can_mmap_block(blk_prefix, blk, st_index):
                    effective.append(BlockTier.CPU_PINNED)
                    continue
            effective.append(tier)
        self._effective_tiers = effective

        has_cpu_tier = any(t == BlockTier.CPU_PINNED for t in effective)
        use_pin = has_cpu_tier and self.pin_memory and self.compute_device.type == "cuda" and torch.cuda.is_available()
        if has_cpu_tier and self.prefetch and self.compute_device.type == "cuda" and torch.cuda.is_available():
            self._prefetch_stream = torch.cuda.Stream(device=self.compute_device)

        cpu_tier_indices = [i for i, t in enumerate(effective) if t == BlockTier.CPU_PINNED]
        next_cpu_peer: Dict[int, Optional[int]] = {
            cpu_tier_indices[pos]: (cpu_tier_indices[pos + 1] if pos + 1 < len(cpu_tier_indices) else None)
            for pos in range(len(cpu_tier_indices))
        }

        for idx, ((blk_prefix, blk), tier) in enumerate(zip(self.blocks, effective)):
            if tier == BlockTier.GPU_RESIDENT:
                for rel_name, is_param, tensor_obj in iter_persistent_module_tensors(blk):
                    if getattr(tensor_obj, "is_meta", False) or tensor_obj.numel() == 0:
                        resolved = lookup_safetensors_key(st_index, f"{blk_prefix}.{rel_name}")
                        if resolved is None:
                            raise KeyError(f"Missing GPU-resident tensor '{blk_prefix}.{rel_name}'.")
                        loader, real_key = resolved
                        dev_t = loader.read_torch_tensor(real_key, device=str(self.compute_device), target_dtype=tensor_obj.dtype)
                        assign_tensor(blk, rel_name, dev_t, is_param)
                    elif tensor_obj.device != self.compute_device:
                        assign_tensor(blk, rel_name, tensor_obj.data.to(self.compute_device), is_param)
                self._hooks.append(blk.register_forward_hook(self._make_gpu_resident_hook(idx)))

            elif tier == BlockTier.CPU_PINNED:
                raw_items: List[Tuple[str, bool, Any]] = []
                for rel_name, is_param, tensor_obj in iter_persistent_module_tensors(blk):
                    if getattr(tensor_obj, "is_meta", False) or tensor_obj.numel() == 0:
                        resolved = lookup_safetensors_key(st_index, f"{blk_prefix}.{rel_name}")
                        if resolved is None:
                            raise KeyError(f"Missing CPU-pinned tensor '{blk_prefix}.{rel_name}'.")
                        loader, real_key = resolved
                        cpu_t = loader.read_torch_tensor(real_key, device="cpu", target_dtype=tensor_obj.dtype)
                    else:
                        cpu_t = tensor_obj.data.detach().to("cpu")
                    raw_items.append((rel_name, is_param, cpu_t))

                if use_pin and len(raw_items) > 1:
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        pinned_tensors = list(
                            pool.map(lambda t: t.pin_memory() if not t.is_pinned() else t, [it[2] for it in raw_items])
                        )
                    shadow = [(it[0], it[1], pt) for it, pt in zip(raw_items, pinned_tensors)]
                else:
                    shadow = [
                        (rel_name, is_param, cpu_t.pin_memory() if (use_pin and not cpu_t.is_pinned()) else cpu_t)
                        for rel_name, is_param, cpu_t in raw_items
                    ]
                for rel_name, is_param, cpu_t in shadow:
                    assign_tensor(blk, rel_name, cpu_t, is_param)
                self._cpu_shadow[idx] = shadow
                self._register_cpu_hooks(idx, blk, next_cpu_peer.get(idx))

            else:
                blk_entries: List[Tuple[str, bool, MMapTensorLoader, str, Any]] = []
                for rel_name, is_param, tensor_obj in iter_persistent_module_tensors(blk):
                    candidate = f"{blk_prefix}.{rel_name}"
                    resolved = lookup_safetensors_key(st_index, candidate)
                    if resolved is None:
                        raise KeyError(
                            f"Missing tensor '{candidate}' in safetensors files {[p.name for p in self.safetensors_paths]}."
                        )
                    loader, real_key = resolved
                    orig_dtype = tensor_obj.dtype
                    blk_entries.append((rel_name, is_param, loader, real_key, orig_dtype))
                    assign_tensor(blk, rel_name, torch.empty(0, dtype=orig_dtype, device="cpu"), is_param)
                self._mmap_map[idx] = blk_entries
                self._register_mmap_hooks(idx, blk)

    def _make_gpu_resident_hook(self, idx: int = 0) -> Any:
        def _post(_mod: Any, _inp: Any, _out: Any) -> None:
            if idx > 0 and self.residual_cache is not None and self.residual_cache.skip_active:
                return
            self.blocks_executed += 1

        return _post

    def _prefetch_cpu_block(self, idx: Optional[int]) -> None:
        import torch

        if idx is None or idx not in self._cpu_shadow or idx in self._prefetch_cache or self._prefetch_stream is None:
            return
        with torch.cuda.stream(self._prefetch_stream):
            self._prefetch_cache[idx] = {
                rel_name: cpu_t.to(self.compute_device, non_blocking=True)
                for rel_name, _, cpu_t in self._cpu_shadow[idx]
            }

    def _register_cpu_hooks(self, idx: int, blk: Any, next_peer_idx: Optional[int]) -> None:
        import torch

        def _pre_hook(module: Any, inputs: Any, kwargs: Optional[Dict[str, Any]] = None) -> None:
            if idx > 0 and self.residual_cache is not None and self.residual_cache.skip_active:
                return
            target_dev = self._resolve_active_device(inputs, kwargs)
            if self._auto_device:
                move_non_persistent_buffers(module, target_dev)

            if self._prefetch_stream is not None and target_dev == self.compute_device:
                while len(self._inflight_events) >= 2:
                    self._inflight_events.popleft().synchronize()

            if self._prefetch_stream is not None and target_dev == self.compute_device and idx in self._prefetch_cache:
                compute_stream = torch.cuda.current_stream(self.compute_device)
                compute_stream.wait_stream(self._prefetch_stream)
                dev_map = self._prefetch_cache.pop(idx)
                for dev_t in dev_map.values():
                    dev_t.record_stream(compute_stream)
            else:
                dev_map = {
                    rel_name: cpu_t.to(target_dev, non_blocking=(target_dev.type == "cuda"))
                    for rel_name, _, cpu_t in self._cpu_shadow[idx]
                }

            for rel_name, is_param, _ in self._cpu_shadow[idx]:
                dev_t = dev_map[rel_name]
                assign_tensor(module, rel_name, dev_t, is_param)
                self.bytes_streamed += dev_t.numel() * dev_t.element_size()

            if self._prefetch_stream is not None and target_dev == self.compute_device:
                # Hold block-0 prefetch when residual cache is active until block-0 declares a miss.
                if not (idx == 0 and self.residual_cache is not None and self.residual_cache.enabled):
                    self._prefetch_cpu_block(next_peer_idx)

        def _post_hook(module: Any, _inputs: Any, _output: Any) -> None:
            if idx > 0 and self.residual_cache is not None and self.residual_cache.skip_active:
                return
            for rel_name, is_param, cpu_t in self._cpu_shadow[idx]:
                assign_tensor(module, rel_name, cpu_t, is_param)
            if self._prefetch_stream is not None and self.compute_device.type == "cuda":
                ev = torch.cuda.Event()
                ev.record(torch.cuda.current_stream(self.compute_device))
                self._inflight_events.append(ev)
            self.blocks_executed += 1

        try:
            self._hooks.append(blk.register_forward_pre_hook(_pre_hook, with_kwargs=True))
        except TypeError:
            self._hooks.append(blk.register_forward_pre_hook(lambda m, inp: _pre_hook(m, inp, None)))
        self._hooks.append(blk.register_forward_hook(_post_hook))

    def _register_mmap_hooks(self, idx: int, blk: Any) -> None:
        import torch

        def _pre_hook(module: Any, inputs: Any, kwargs: Optional[Dict[str, Any]] = None) -> None:
            if idx > 0 and self.residual_cache is not None and self.residual_cache.skip_active:
                return
            target_dev = self._resolve_active_device(inputs, kwargs)
            if self._auto_device:
                move_non_persistent_buffers(module, target_dev)
            for rel_name, is_param, loader, real_key, orig_dtype in self._mmap_map[idx]:
                loaded = loader.read_torch_tensor(real_key, device=str(target_dev), target_dtype=orig_dtype)
                assign_tensor(module, rel_name, loaded, is_param)
                self.bytes_streamed += loader.get_tensor_byte_size(real_key)

        def _post_hook(module: Any, _inputs: Any, _output: Any) -> None:
            if idx > 0 and self.residual_cache is not None and self.residual_cache.skip_active:
                return
            for rel_name, is_param, _, _, orig_dtype in self._mmap_map[idx]:
                assign_tensor(module, rel_name, torch.empty(0, dtype=orig_dtype, device="cpu"), is_param)
            self.blocks_executed += 1

        try:
            self._hooks.append(blk.register_forward_pre_hook(_pre_hook, with_kwargs=True))
        except TypeError:
            self._hooks.append(blk.register_forward_pre_hook(lambda m, inp: _pre_hook(m, inp, None)))
        self._hooks.append(blk.register_forward_hook(_post_hook))

    def detach(self, restore_cpu_weights: bool = True) -> None:
        """Remove all forward hooks, restore CPU shadow weights if applicable, and close mmaps."""
        if self.residual_cache is not None:
            self.residual_cache.detach()
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._prefetch_cache.clear()
        self._inflight_events.clear()

        if restore_cpu_weights and self._cpu_shadow:
            for idx, (_, blk) in enumerate(self.blocks):
                for rel_name, is_param, cpu_t in self._cpu_shadow.get(idx, []):
                    assign_tensor(blk, rel_name, cpu_t, is_param)

        self._cpu_shadow.clear()
        self._mmap_map.clear()
        self._effective_tiers.clear()
        for loader in self._loaders:
            loader.close()
        self._loaders.clear()
        self._attached = False

    def __enter__(self) -> "TorchBlockStreamer":
        return self.attach()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.detach()


__all__ = [
    "BlockStreamMode",
    "KNOWN_BLOCK_CONTAINERS",
    "STRIP_MODEL_PREFIXES",
    "TorchBlockStreamer",
    "discover_block_modules",
]

"""2-Slot Pinned Async `mmap` Ring Prefetcher, Disjoint `MADV_DONTNEED` Paging & Compressed GGUF Shadow.

Ports DeimosStream's zero-shadow pinned staging ring + disjoint `<24 GiB` page-cache eviction to
PhobosStream's Diffusion Transformer streamer:
1. `PackedGGUFShadow`: Keeps `.gguf` (`Q8_0`/`Q4_K`/`Q5_K`/`Q6_K`) tensors compressed as `uint8`
   in pinned Host RAM (`47%-71%` less RAM & PCIe DMA) and unpacks on-GPU via `phobos_cuda_gguf.cu`.
2. `MMapRingPrefetcher`: Overlaps `mmap -> 2-slot pinned slab -> async CUDA H2D (+ nvcc GGUF unpack)`
   of block `next_idx` on a background thread + dedicated CUDA stream while block `idx` executes,
   and calls `MADV_DONTNEED` when Host RAM cannot hold the full checkpoint in OS page cache.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple
import numpy as np

from phobos_stream.storage.gguf_mmap_loader import GGUFMMapLoader, dequantize_ggml_on_device
from phobos_stream.streaming.block_discovery import (
    assign_tensor,
    iter_persistent_module_tensors,
    lookup_safetensors_key,
    move_non_persistent_buffers,
)
from phobos_stream.streaming.budget_planner import detect_host_available_ram_bytes


@dataclass
class PackedGGUFShadow:
    """Compressed `uint8` GGUF tensor held in pinned CPU RAM for 1.8x-3.4x smaller PCIe DMA."""

    raw_u8: Any
    ggml_type: str
    shape: Tuple[int, ...]
    out_dtype: Any

    @property
    def nbytes(self) -> int:
        return int(self.raw_u8.numel())

    def to_device(self, device: Any, non_blocking: bool = True) -> Any:
        dev_u8 = self.raw_u8.to(device, non_blocking=non_blocking)
        return dequantize_ggml_on_device(dev_u8, self.ggml_type, self.shape, self.out_dtype)


def build_cpu_pinned_shadow(
    blk_prefix: str,
    blk: Any,
    st_index: Dict[str, Tuple[Any, str]],
    use_pin: bool,
) -> List[Tuple[str, bool, Any]]:
    """Build pinned CPU shadow weights for `blk`, keeping `.gguf` tensors compressed as `PackedGGUFShadow`."""
    import torch

    raw_items: List[Tuple[str, bool, Any]] = []
    for rel_name, is_param, tensor_obj in iter_persistent_module_tensors(blk):
        if getattr(tensor_obj, "is_meta", False) or tensor_obj.numel() == 0:
            resolved = lookup_safetensors_key(st_index, f"{blk_prefix}.{rel_name}")
            if resolved is None:
                raise KeyError(f"Missing CPU-pinned tensor '{blk_prefix}.{rel_name}'.")
            loader, real_key = resolved
            if isinstance(loader, GGUFMMapLoader):
                view, dt_str, shape = loader.read_raw_slice(real_key)
                u8_t = torch.from_numpy(np.frombuffer(view, dtype=np.uint8).copy())
                del view
                cpu_t: Any = PackedGGUFShadow(
                    u8_t.pin_memory() if (use_pin and not u8_t.is_pinned()) else u8_t,
                    dt_str,
                    shape,
                    tensor_obj.dtype,
                )
            else:
                cpu_t = loader.read_torch_tensor(real_key, device="cpu", target_dtype=tensor_obj.dtype)
        else:
            cpu_t = tensor_obj.data.detach().to("cpu")
        raw_items.append((rel_name, is_param, cpu_t))

    def _pin_item(obj: Any) -> Any:
        if isinstance(obj, PackedGGUFShadow) or not use_pin:
            return obj
        return obj.pin_memory() if not obj.is_pinned() else obj

    if use_pin and len(raw_items) > 1:
        with ThreadPoolExecutor(max_workers=4) as pool:
            pinned_objs = list(pool.map(_pin_item, [it[2] for it in raw_items]))
        shadow = [(it[0], it[1], po) for it, po in zip(raw_items, pinned_objs)]
    else:
        shadow = [(rel_name, is_param, _pin_item(cpu_t)) for rel_name, is_param, cpu_t in raw_items]

    for rel_name, is_param, sh_obj in shadow:
        placeholder = (
            torch.empty(0, dtype=sh_obj.out_dtype, device="cpu")
            if isinstance(sh_obj, PackedGGUFShadow)
            else sh_obj
        )
        assign_tensor(blk, rel_name, placeholder, is_param)
    return shadow


def materialize_raw_bytes_to_device(
    raw_u8_pinned: Any,
    loader: Any,
    dtype_str: str,
    shape: Tuple[int, ...],
    target_dev: Any,
    orig_dtype: Any,
    non_blocking: bool,
) -> Any:
    """Transfer a pinned `uint8` slice to `target_dev` and reinterpret/unpack without CPU FP32 expansion."""
    import torch

    if isinstance(loader, GGUFMMapLoader):
        dev_u8 = raw_u8_pinned.to(device=target_dev, non_blocking=non_blocking)
        return dequantize_ggml_on_device(dev_u8, dtype_str, shape, orig_dtype)

    dev_u8 = raw_u8_pinned.to(device=target_dev, non_blocking=non_blocking)
    if dtype_str == "BF16":
        t = dev_u8.view(torch.bfloat16).reshape(shape)
    elif dtype_str == "F16":
        t = dev_u8.view(torch.float16).reshape(shape)
    elif dtype_str == "F32":
        t = dev_u8.view(torch.float32).reshape(shape)
    elif dtype_str == "F8_E4M3" and hasattr(torch, "float8_e4m3fn"):
        t = dev_u8.view(torch.float8_e4m3fn).reshape(shape)
    elif dtype_str == "F8_E5M2" and hasattr(torch, "float8_e5m2"):
        t = dev_u8.view(torch.float8_e5m2).reshape(shape)
    else:
        return loader.read_torch_tensor("__fallback__", device=str(target_dev), target_dtype=orig_dtype)

    if orig_dtype is not None and t.dtype != orig_dtype:
        t = t.to(dtype=orig_dtype)
    return t


class MMapRingPrefetcher:
    """Asynchronous 2-slot pinned ring prefetcher + disjoint `MADV_DONTNEED` evictor for `NVME_MMAP`."""

    def __init__(
        self,
        compute_device: Any,
        mmap_map: Dict[int, List[Tuple[str, bool, Any, str, Any]]],
        loaders: Sequence[Any],
        pin_memory: bool = True,
        prefetch: bool = True,
        ram_budget_mb: Optional[float] = None,
    ) -> None:
        import torch

        self.compute_device = compute_device
        self.mmap_map = mmap_map
        self.is_cuda = compute_device.type == "cuda" and torch.cuda.is_available()
        self.enabled = self.is_cuda and prefetch and len(mmap_map) > 0

        # Enable disjoint MADV_DONTNEED when total backing file size exceeds available/budgeted host RAM
        total_file_bytes = 0
        seen_paths = set()
        for ldr in loaders:
            fp = getattr(ldr, "file_path", None)
            if fp is not None and fp not in seen_paths and fp.exists():
                seen_paths.add(fp)
                total_file_bytes += int(fp.stat().st_size)
        avail_ram = detect_host_available_ram_bytes(safety_fraction=0.85)
        if ram_budget_mb is not None and ram_budget_mb > 0:
            avail_ram = min(avail_ram, int(ram_budget_mb * 1024 * 1024))
        self.disjoint_evict = total_file_bytes > 0 and total_file_bytes > avail_ram

        self._stream: Optional[Any] = None
        self._pool: Optional[ThreadPoolExecutor] = None
        self._slabs: List[Any] = []
        self._slab_h2d_events: List[Optional[Any]] = [None, None]
        self._futures: Dict[int, Future[Tuple[Dict[str, Any], int, Any]]] = {}
        self._inflight_compute_events: Deque[Any] = deque()
        self._slot_cursor: int = 0

        if self.enabled:
            max_blk_bytes = max(
                sum(ldr.get_tensor_byte_size(rk) for _, _, ldr, rk, _ in entries)
                for entries in mmap_map.values()
            )
            if max_blk_bytes > 0:
                use_pin = pin_memory and avail_ram > max_blk_bytes * 3
                self._slabs = [
                    torch.empty((max_blk_bytes,), dtype=torch.uint8, device="cpu", pin_memory=use_pin)
                    for _ in range(2)
                ]
                self._stream = torch.cuda.Stream(device=compute_device)
                self._pool = ThreadPoolExecutor(max_workers=1)
            else:
                self.enabled = False

    def _stage_block_worker(self, idx: int, slot: int) -> Tuple[Dict[str, Any], int, Any]:
        import torch

        assert self._stream is not None
        prev_ev = self._slab_h2d_events[slot]
        if prev_ev is not None:
            prev_ev.synchronize()

        slab = self._slabs[slot]
        slab_np = slab.numpy()
        cursor = 0
        staged_meta: List[Tuple[str, Any, str, Tuple[int, ...], Any, int, int]] = []
        evict_targets: List[Tuple[Any, str]] = []

        for rel_name, _, loader, real_key, orig_dtype in self.mmap_map[idx]:
            if hasattr(loader, "read_raw_slice"):
                view, dt_str, shape = loader.read_raw_slice(real_key)
                if dt_str in ("BF16", "F16", "F32", "F8_E4M3", "F8_E5M2", "Q8_0", "Q4_0", "Q4_1", "Q4_K", "Q5_K", "Q6_K"):
                    nb = len(view)
                    slab_np[cursor : cursor + nb] = np.frombuffer(view, dtype=np.uint8)
                    del view
                    staged_meta.append((rel_name, loader, dt_str, shape, orig_dtype, cursor, cursor + nb))
                    cursor += nb
                    if self.disjoint_evict and hasattr(loader, "evict_page_cache"):
                        evict_targets.append((loader, real_key))
                    continue
            staged_meta.append((rel_name, loader, f"__fallback__:{real_key}", (), orig_dtype, 0, 0))

        dev_map: Dict[str, Any] = {}
        bytes_read = 0
        with torch.cuda.stream(self._stream):
            for rel_name, loader, dt_str, shape, orig_dtype, s_off, e_off in staged_meta:
                if dt_str.startswith("__fallback__:"):
                    rk_fb = dt_str.split(":", 1)[1]
                    dev_map[rel_name] = loader.read_torch_tensor(rk_fb, device=str(self.compute_device), target_dtype=orig_dtype)
                    bytes_read += loader.get_tensor_byte_size(rk_fb)
                else:
                    dev_map[rel_name] = materialize_raw_bytes_to_device(
                        slab[s_off:e_off],
                        loader,
                        dt_str,
                        shape,
                        self.compute_device,
                        orig_dtype,
                        non_blocking=True,
                    )
                    bytes_read += e_off - s_off
            done_ev = torch.cuda.Event()
            done_ev.record(self._stream)
            self._slab_h2d_events[slot] = done_ev

        for loader, rk in evict_targets:
            loader.evict_page_cache(rk)

        return dev_map, bytes_read, done_ev

    def submit_prefetch(self, idx: Optional[int]) -> None:
        if not self.enabled or self._pool is None or idx is None or idx not in self.mmap_map or idx in self._futures:
            return
        slot = self._slot_cursor
        self._slot_cursor = (self._slot_cursor + 1) & 1
        self._futures[idx] = self._pool.submit(self._stage_block_worker, idx, slot)

    def register_hooks(
        self,
        streamer: Any,
        idx: int,
        blk: Any,
        next_mmap_idx: Optional[int],
        resolve_device_fn: Callable[[Any, Optional[Dict[str, Any]]], Any],
    ) -> List[Any]:
        import torch

        hooks: List[Any] = []

        def _pre_hook(module: Any, inputs: Any, kwargs: Optional[Dict[str, Any]] = None) -> None:
            if streamer._is_skipped_by_cache(idx):
                return
            target_dev = resolve_device_fn(inputs, kwargs)
            if streamer._auto_device:
                move_non_persistent_buffers(module, target_dev)

            if self.enabled and target_dev == self.compute_device:
                while len(self._inflight_compute_events) >= 2:
                    self._inflight_compute_events.popleft().synchronize()

                if idx not in self._futures:
                    self.submit_prefetch(idx)
                fut = self._futures.pop(idx)
                dev_map, b_read, h2d_ev = fut.result()
                compute_stream = torch.cuda.current_stream(self.compute_device)
                compute_stream.wait_event(h2d_ev)
                for rel_name, is_param, _, _, _ in self.mmap_map[idx]:
                    dev_t = dev_map[rel_name]
                    dev_t.record_stream(compute_stream)
                    assign_tensor(module, rel_name, dev_t, is_param)
                streamer.bytes_streamed += b_read

                is_probe = (
                    streamer.residual_cache is not None
                    and streamer.residual_cache.enabled
                    and idx == streamer.residual_cache.probe_block_index
                )
                if not is_probe:
                    self.submit_prefetch(next_mmap_idx)
                return

            # Synchronous fallback (`cpu` / `mps` / prefetch=False)
            for rel_name, is_param, loader, real_key, orig_dtype in self.mmap_map[idx]:
                loaded = loader.read_torch_tensor(real_key, device=str(target_dev), target_dtype=orig_dtype)
                assign_tensor(module, rel_name, loaded, is_param)
                streamer.bytes_streamed += loader.get_tensor_byte_size(real_key)
                if self.disjoint_evict and hasattr(loader, "evict_page_cache"):
                    loader.evict_page_cache(real_key)

        def _post_hook(module: Any, _inputs: Any, _output: Any) -> None:
            if streamer._is_skipped_by_cache(idx):
                return
            for rel_name, is_param, _, _, orig_dtype in self.mmap_map[idx]:
                assign_tensor(module, rel_name, torch.empty(0, dtype=orig_dtype, device="cpu"), is_param)
            if self.enabled and self.compute_device.type == "cuda":
                ev = torch.cuda.Event()
                ev.record(torch.cuda.current_stream(self.compute_device))
                self._inflight_compute_events.append(ev)
            streamer.blocks_executed += 1

        try:
            hooks.append(blk.register_forward_pre_hook(_pre_hook, with_kwargs=True))
        except TypeError:
            hooks.append(blk.register_forward_pre_hook(lambda m, inp: _pre_hook(m, inp, None)))
        hooks.append(blk.register_forward_hook(_post_hook))
        return hooks

    def close(self) -> None:
        if self._pool is not None:
            for fut in list(self._futures.values()):
                try:
                    fut.result()
                except Exception:
                    pass
            self._futures.clear()
            self._pool.shutdown(wait=True)
            self._pool = None
        self._slabs.clear()
        self._inflight_compute_events.clear()

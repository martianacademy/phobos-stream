"""First-Block Residual Cache (TeaCache / FBCache-style dynamic step skipping) for DiTs.

Accelerates Diffusion Transformer denoising by 1.5x-2.5x with zero extra GPU VRAM:
1. Always executes `blocks[0]` (~2.5% of compute on a 40-block DiT) to measure the actual
   first-block hidden-state residual `delta_0(t) = h_out_0(t) - h_in(t)`.
2. Tracks accumulated relative L1 change `||delta_0(t) - delta_0(t_prev)||_1 / ||delta_0(t_prev)||_1`
   independently per CFG branch (`cond` vs `uncond`) and automatically resets when a new
   generation starts (detected via non-decreasing timestep/sigma or latent shape change).
3. On a cache hit (`accumulated_rel_diff < rel_l1_thresh`), sets `skip_active = True` so
   `TorchBlockStreamer` skips both PCIe DMA transfers and block forward passes for
   `blocks[1 .. N-1]`, reconstructing `h_final(t) = h_out_0(t) + R_rest_cached`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from phobos_stream.streaming.block_discovery import discover_block_modules


@dataclass
class _BranchCacheSlot:
    """Per-branch (`cond` / `uncond`) residual state across denoising steps."""

    prev_delta_0: Optional[Any] = None
    residual_rest: Optional[Any] = None
    accumulated_rel_diff: float = 0.0
    consecutive_skips: int = 0
    steps_seen: int = 0


def _extract_primary_tensor(obj: Any) -> Optional[Any]:
    """Return the first `torch.Tensor` inside `obj` (tensor, tuple, list, or dict)."""
    import torch

    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (tuple, list)):
        for item in obj:
            found = _extract_primary_tensor(item)
            if found is not None:
                return found
    elif isinstance(obj, dict):
        for item in obj.values():
            found = _extract_primary_tensor(item)
            if found is not None:
                return found
    return None


def _extract_scalar_timestep(args: Tuple[Any, ...], kwargs: Optional[Dict[str, Any]]) -> Optional[float]:
    """Extract scalar timestep/sigma from root `diffusion_model.forward` arguments if present."""
    import torch

    candidates: List[Any] = []
    if kwargs:
        for key in ("timestep", "timesteps", "t", "sigma", "sigmas"):
            if key in kwargs and kwargs[key] is not None:
                candidates.append(kwargs[key])
    if len(args) >= 2:
        candidates.append(args[1])

    for cand in candidates:
        if isinstance(cand, (int, float)):
            return float(cand)
        if isinstance(cand, torch.Tensor) and cand.numel() > 0 and cand.dtype.is_floating_point or (
            isinstance(cand, torch.Tensor) and cand.numel() > 0 and cand.ndim <= 2
        ):
            try:
                return float(cand.detach().reshape(-1)[0].item())
            except Exception:
                continue
    return None


class FirstBlockResidualCache:
    """Hook-compatible First-Block Residual Cache (`TeaCache` / `FBCache`) for PyTorch DiTs."""

    def __init__(
        self,
        rel_l1_thresh: float = 0.12,
        warmup_steps: int = 2,
        max_consecutive_skips: int = 3,
        offload_cache_to_cpu: bool = True,
    ) -> None:
        if rel_l1_thresh < 0.0:
            raise ValueError(f"rel_l1_thresh must be >= 0.0, got {rel_l1_thresh}.")
        self.rel_l1_thresh = float(rel_l1_thresh)
        self.warmup_steps = max(1, int(warmup_steps))
        self.max_consecutive_skips = max(1, int(max_consecutive_skips))
        self.offload_cache_to_cpu = bool(offload_cache_to_cpu)

        self.skip_active: bool = False
        self.total_steps: int = 0
        self.skipped_steps: int = 0
        self.computed_steps: int = 0

        self._slots: Dict[int, _BranchCacheSlot] = {}
        self._active_branch: int = 0
        self._last_timestep: Optional[float] = None
        self._last_shape: Optional[Tuple[int, ...]] = None
        self._cur_block0_in: Optional[Any] = None
        self._cur_block0_out: Optional[Any] = None
        self._cur_delta_0: Optional[Any] = None

        self._root_hook: Optional[Any] = None
        self._orig_forwards: List[Tuple[Any, Callable[..., Any]]] = []
        self._on_miss_callback: Optional[Callable[[], None]] = None
        self._attached: bool = False

    @property
    def enabled(self) -> bool:
        return self.rel_l1_thresh > 0.0

    @property
    def hit_rate(self) -> float:
        return (self.skipped_steps / self.total_steps) if self.total_steps > 0 else 0.0

    def reset_state(self) -> None:
        """Clear cached residuals between independent generations while keeping stats or hooks."""
        self._slots.clear()
        self._active_branch = 0
        self._last_timestep = None
        self._last_shape = None
        self._cur_block0_in = None
        self._cur_block0_out = None
        self._cur_delta_0 = None
        self.skip_active = False

    def reset_metrics(self) -> None:
        """Reset step hit/miss counters."""
        self.total_steps = 0
        self.skipped_steps = 0
        self.computed_steps = 0

    def _store_tensor(self, tensor: Any) -> Any:
        import torch

        t = tensor.detach()
        if self.offload_cache_to_cpu and t.device.type != "cpu":
            cpu_t = t.to("cpu", non_blocking=False)
            if torch.cuda.is_available():
                cpu_t = cpu_t.pin_memory()
            return cpu_t
        return t.clone()

    def attach(
        self,
        model: Any,
        blocks: Optional[Sequence[Tuple[str, Any]]] = None,
        on_miss_callback: Optional[Callable[[], None]] = None,
    ) -> "FirstBlockResidualCache":
        """Attach residual caching wrappers onto `blocks` (requires >= 2 blocks when enabled)."""
        if self._attached or not self.enabled:
            return self

        resolved_blocks = list(blocks) if blocks is not None else discover_block_modules(model)
        if len(resolved_blocks) < 2:
            return self

        self._on_miss_callback = on_miss_callback

        def _root_pre_hook(_mod: Any, args: Tuple[Any, ...], kwargs: Optional[Dict[str, Any]] = None) -> None:
            self.skip_active = False
            x_t = _extract_primary_tensor(args)
            if x_t is None:
                x_t = _extract_primary_tensor(kwargs)
            if x_t is not None:
                cur_shape = tuple(int(d) for d in x_t.shape)
                if self._last_shape is not None and cur_shape != self._last_shape:
                    self.reset_state()
                self._last_shape = cur_shape

            t_val = _extract_scalar_timestep(args, kwargs)
            if t_val is not None:
                if self._last_timestep is not None:
                    # Timestep jumping upward indicates a brand-new denoising trajectory.
                    if t_val > self._last_timestep + 1e-5:
                        self.reset_state()
                        self._active_branch = 0
                    elif abs(t_val - self._last_timestep) <= 1e-5:
                        # Second forward pass at identical sigma (e.g. uncond pass of CFG > 1).
                        self._active_branch += 1
                    else:
                        self._active_branch = 0
                self._last_timestep = t_val
            else:
                self._active_branch = 0

        try:
            self._root_hook = model.register_forward_pre_hook(_root_pre_hook, with_kwargs=True)
        except TypeError:
            self._root_hook = model.register_forward_pre_hook(lambda m, a: _root_pre_hook(m, a, None))

        num_blocks = len(resolved_blocks)
        for idx, (_, blk) in enumerate(resolved_blocks):
            orig_fwd = blk.forward
            self._orig_forwards.append((blk, orig_fwd))
            if idx == 0:
                blk.forward = self._make_first_block_forward(orig_fwd)
            elif idx < num_blocks - 1:
                blk.forward = self._make_middle_block_forward(orig_fwd)
            else:
                blk.forward = self._make_last_block_forward(orig_fwd)

        self._attached = True
        return self

    def _make_first_block_forward(self, orig_fwd: Callable[..., Any]) -> Callable[..., Any]:
        import torch

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            h_in = _extract_primary_tensor(args)
            if h_in is None:
                h_in = _extract_primary_tensor(kwargs)
            out = orig_fwd(*args, **kwargs)
            if not isinstance(out, torch.Tensor):
                self.skip_active = False
                if self._on_miss_callback is not None:
                    self._on_miss_callback()
                return out
            h_out = out

            if h_in is None or h_out is None or h_in.shape != h_out.shape:
                self.skip_active = False
                if self._on_miss_callback is not None:
                    self._on_miss_callback()
                return out

            slot = self._slots.setdefault(self._active_branch, _BranchCacheSlot())
            slot.steps_seen += 1
            self.total_steps += 1

            with torch.no_grad():
                delta_0 = h_out.detach() - h_in.detach()
                can_skip = (
                    slot.steps_seen > self.warmup_steps
                    and slot.prev_delta_0 is not None
                    and slot.residual_rest is not None
                    and slot.consecutive_skips < self.max_consecutive_skips
                )
                if can_skip:
                    prev_d0 = slot.prev_delta_0.to(device=delta_0.device, dtype=torch.float32)
                    cur_d0 = delta_0.float()
                    denom = float(prev_d0.abs().mean().item()) + 1e-6
                    rel_diff = float((cur_d0 - prev_d0).abs().mean().item()) / denom
                    if (slot.accumulated_rel_diff + rel_diff) < self.rel_l1_thresh:
                        slot.accumulated_rel_diff += rel_diff
                        slot.consecutive_skips += 1
                        self.skip_active = True
                        self.skipped_steps += 1
                        self._cur_block0_out = h_out.detach()
                        return out

                # Cache miss: run remaining blocks and refresh cached residual at block N-1.
                self.skip_active = False
                self.computed_steps += 1
                self._cur_delta_0 = delta_0
                self._cur_block0_out = h_out.detach()
                if self._on_miss_callback is not None:
                    self._on_miss_callback()
            return out

        return _wrapped

    def _make_middle_block_forward(self, orig_fwd: Callable[..., Any]) -> Callable[..., Any]:
        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            if self.skip_active:
                if len(args) == 1:
                    return args[0]
                if len(args) > 1:
                    return args[0]
                return next(iter(kwargs.values()))
            return orig_fwd(*args, **kwargs)

        return _wrapped

    def _make_last_block_forward(self, orig_fwd: Callable[..., Any]) -> Callable[..., Any]:
        import torch

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            slot = self._slots.setdefault(self._active_branch, _BranchCacheSlot())
            if self.skip_active and slot.residual_rest is not None:
                x_cur = args[0] if args else next(iter(kwargs.values()))
                if isinstance(x_cur, torch.Tensor):
                    res = slot.residual_rest.to(device=x_cur.device, dtype=x_cur.dtype)
                    return x_cur + res
                if isinstance(x_cur, tuple) and len(x_cur) > 0 and isinstance(x_cur[0], torch.Tensor):
                    res = slot.residual_rest.to(device=x_cur[0].device, dtype=x_cur[0].dtype)
                    return (x_cur[0] + res, *x_cur[1:])

            out = orig_fwd(*args, **kwargs)
            h_final = _extract_primary_tensor(out)
            if (
                h_final is not None
                and self._cur_block0_out is not None
                and h_final.shape == self._cur_block0_out.shape
                and self._cur_delta_0 is not None
            ):
                with torch.no_grad():
                    res_rest = h_final.detach() - self._cur_block0_out
                    slot.prev_delta_0 = self._store_tensor(self._cur_delta_0)
                    slot.residual_rest = self._store_tensor(res_rest)
                    slot.accumulated_rel_diff = 0.0
                    slot.consecutive_skips = 0
            self._cur_block0_out = None
            self._cur_delta_0 = None
            return out

        return _wrapped

    def detach(self) -> None:
        """Restore original `block.forward` methods and remove root pre-hook."""
        if self._root_hook is not None:
            self._root_hook.remove()
            self._root_hook = None
        for blk, orig_fwd in self._orig_forwards:
            blk.forward = orig_fwd
        self._orig_forwards.clear()
        self.reset_state()
        self._attached = False

"""Multi-Group First-Block Residual Cache (TeaCache / FBCache) for Single- & Dual-Stream DiTs.

Supports both single-stream DiTs (`Wan2.1`, `CogVideoX`, `LTX-Video`) and multi-group /
dual-stream MMDiT architectures (`FLUX.1` `double_blocks` + `single_blocks`, `HunyuanVideo`,
`MiniMax-H3` `refiner_blocks` + `blocks`, `Lumina2` / `Z-Image-Turbo`):
1. Groups discovered blocks by their container prefix (e.g. `double_blocks`, `single_blocks`).
2. Always executes Group 0's `block[0]` (~1.7%-2.5% of compute) to measure the relative L1
   change in first-block residual `delta_0(t) = img_out_0(t) - img_in(t)`.
3. On a cache hit, sets `skip_active = True` so `TorchBlockStreamer` skips both GPU compute
   and PCIe DMA across ALL remaining blocks in Group 0 AND 100% of blocks in Groups `1..G-1`,
   applying each group's cached residual at the group boundary so inter-group parent ops
   (like `FLUX`'s `torch.cat((txt, img), dim=1)`) execute naturally without model surgery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from phobos_stream.streaming.block_discovery import discover_block_modules


@dataclass
class _BranchCacheSlot:
    """Per-branch (`cond` / `uncond`) multi-group residual state across denoising steps."""

    prev_delta_0: Optional[Any] = None
    group_residuals: Dict[int, Any] = field(default_factory=dict)
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
        if isinstance(cand, torch.Tensor) and cand.numel() > 0 and cand.ndim <= 2:
            try:
                return float(cand.detach().reshape(-1)[0].item())
            except Exception:
                continue
    return None


def _is_valid_state(state: Any) -> bool:
    """Check whether `state` is a single `Tensor` or a 2-tuple `(Tensor, Tensor)`."""
    import torch

    if isinstance(state, torch.Tensor):
        return True
    return (
        isinstance(state, tuple)
        and len(state) == 2
        and isinstance(state[0], torch.Tensor)
        and isinstance(state[1], torch.Tensor)
    )


def _extract_call_state(args: Tuple[Any, ...], kwargs: Dict[str, Any], is_pair: Optional[bool]) -> Optional[Any]:
    """Extract single-stream `x` or dual-stream `(a, b)` input matching block return layout."""
    import torch

    if is_pair is True or (is_pair is None and ("txt" in kwargs or "encoder_hidden_states" in kwargs)):
        if "img" in kwargs and "txt" in kwargs:
            return (kwargs["img"], kwargs["txt"])
        if "encoder_hidden_states" in kwargs and "hidden_states" in kwargs:
            return (kwargs["encoder_hidden_states"], kwargs["hidden_states"])
        if len(args) >= 2 and isinstance(args[0], torch.Tensor) and isinstance(args[1], torch.Tensor):
            return (args[0], args[1])

    if args and isinstance(args[0], torch.Tensor):
        return args[0]
    for key in ("img", "x", "hidden_states"):
        if key in kwargs and isinstance(kwargs[key], torch.Tensor):
            return kwargs[key]
    return _extract_primary_tensor(args) if args else _extract_primary_tensor(kwargs)


def _states_compatible(a: Any, b: Any) -> bool:
    import torch

    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a.shape == b.shape
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == 2 and len(b) == 2:
        return a[0].shape == b[0].shape and a[1].shape == b[1].shape
    return False


def _clone_state(state: Any) -> Optional[Any]:
    """Deep-copy tensor(s) so downstream in-place `+=` blocks (`FLUX` / `HunyuanVideo`) cannot corrupt anchors."""
    import torch

    if isinstance(state, torch.Tensor):
        return state.detach().clone()
    if (
        isinstance(state, tuple)
        and len(state) == 2
        and isinstance(state[0], torch.Tensor)
        and isinstance(state[1], torch.Tensor)
    ):
        return (state[0].detach().clone(), state[1].detach().clone())
    return None


def _snapshot_pre_call(args: Tuple[Any, ...], kwargs: Dict[str, Any], known_pair: Optional[bool]) -> Tuple[Optional[Any], Optional[Any]]:
    """Snapshot pre-forward input tensors before blocks mutate `img += ...` / `x += ...` in-place."""
    if known_pair is True:
        return None, _clone_state(_extract_call_state(args, kwargs, True))
    if known_pair is False:
        return _clone_state(_extract_call_state(args, kwargs, False)), None
    return (
        _clone_state(_extract_call_state(args, kwargs, False)),
        _clone_state(_extract_call_state(args, kwargs, True)),
    )


def _sub_state(out_s: Any, base_s: Any) -> Any:
    import torch

    if isinstance(out_s, torch.Tensor):
        return out_s.detach() - base_s.detach()
    return (out_s[0].detach() - base_s[0].detach(), out_s[1].detach() - base_s[1].detach())


def _add_state(cur_s: Any, res_s: Any) -> Any:
    import torch

    if isinstance(cur_s, torch.Tensor):
        return cur_s + res_s.to(device=cur_s.device, dtype=cur_s.dtype)
    return (
        cur_s[0] + res_s[0].to(device=cur_s[0].device, dtype=cur_s[0].dtype),
        cur_s[1] + res_s[1].to(device=cur_s[1].device, dtype=cur_s[1].dtype),
    )


def _group_blocks_by_container(blocks: Sequence[Tuple[str, Any]]) -> List[List[Tuple[str, Any]]]:
    """Partition `[(name, mod), ...]` into contiguous container groups (`double_blocks`, `single_blocks`)."""
    groups: List[List[Tuple[str, Any]]] = []
    cur_prefix: Optional[str] = None
    for name, mod in blocks:
        container = name.rsplit(".", 1)[0] if "." in name else ""
        if not groups or container != cur_prefix:
            groups.append([(name, mod)])
            cur_prefix = container
        else:
            groups[-1].append((name, mod))
    return groups


class FirstBlockResidualCache:
    """Hook-compatible Multi-Group First-Block Residual Cache (`TeaCache` / `FBCache`)."""

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

        self._num_groups: int = 0
        self.probe_block_index: int = 0
        self._group_is_pair: Dict[int, Optional[bool]] = {}
        self._group_anchors: Dict[int, Any] = {}
        self._slots: Dict[int, _BranchCacheSlot] = {}
        self._active_branch: int = 0
        self._last_timestep: Optional[float] = None
        self._last_shape: Optional[Tuple[int, ...]] = None
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
        """Clear cached residuals between independent generations."""
        self._slots.clear()
        self._group_anchors.clear()
        self._active_branch = 0
        self._last_timestep = None
        self._last_shape = None
        self._cur_delta_0 = None
        self.skip_active = False

    def reset_metrics(self) -> None:
        self.total_steps = 0
        self.skipped_steps = 0
        self.computed_steps = 0

    def _store_state(self, state: Any) -> Any:
        import torch

        def _one(t: Any) -> Any:
            d = t.detach()
            if self.offload_cache_to_cpu and d.device.type != "cpu":
                c = d.to("cpu", non_blocking=False)
                return c.pin_memory() if torch.cuda.is_available() else c
            return d.clone()

        if isinstance(state, tuple):
            return (_one(state[0]), _one(state[1]))
        return _one(state)

    def attach(
        self,
        model: Any,
        blocks: Optional[Sequence[Tuple[str, Any]]] = None,
        on_miss_callback: Optional[Callable[[], None]] = None,
    ) -> "FirstBlockResidualCache":
        """Attach multi-group residual caching wrappers onto `blocks`."""
        if self._attached or not self.enabled:
            return self

        resolved_blocks = list(blocks) if blocks is not None else discover_block_modules(model)
        if len(resolved_blocks) < 2:
            return self

        raw_groups = _group_blocks_by_container(resolved_blocks)
        # Skip leading pure-text refiner groups (`txt_in`, `context_refiner`) whose outputs are
        # invariant across denoising timesteps `t` so `delta_0` probes actual noisy latent blocks.
        skipped_leading_blocks = 0
        while len(raw_groups) > 1:
            first_pfx = raw_groups[0][0][0]
            if any(tok in first_pfx for tok in ("txt_in", "context_refiner")):
                skipped_leading_blocks += len(raw_groups.pop(0))
            else:
                break

        groups = raw_groups
        if sum(len(g) for g in groups) < 2:
            return self

        self.probe_block_index = skipped_leading_blocks
        self._num_groups = len(groups)
        self._group_is_pair = {g_idx: None for g_idx in range(self._num_groups)}
        self._on_miss_callback = on_miss_callback

        def _root_pre_hook(_mod: Any, args: Tuple[Any, ...], kwargs: Optional[Dict[str, Any]] = None) -> None:
            self.skip_active = False
            self._group_anchors.clear()
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
                    if t_val > self._last_timestep + 1e-5:
                        self.reset_state()
                        self._active_branch = 0
                    elif abs(t_val - self._last_timestep) <= 1e-5:
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

        for g_idx, g_blocks in enumerate(groups):
            g_len = len(g_blocks)
            for local_idx, (_, blk) in enumerate(g_blocks):
                orig_fwd = blk.forward
                self._orig_forwards.append((blk, orig_fwd))
                is_first_overall = g_idx == 0 and local_idx == 0
                is_group_start = g_idx >= 1 and local_idx == 0
                is_group_end = local_idx == (g_len - 1)
                blk.forward = self._make_block_wrapper(
                    orig_fwd=orig_fwd,
                    g_idx=g_idx,
                    is_first_overall=is_first_overall,
                    is_group_start=is_group_start,
                    is_group_end=is_group_end,
                )

        self._attached = True
        return self

    def _make_block_wrapper(
        self,
        orig_fwd: Callable[..., Any],
        g_idx: int,
        is_first_overall: bool,
        is_group_start: bool,
        is_group_end: bool,
    ) -> Callable[..., Any]:
        import torch

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            slot = self._slots.setdefault(self._active_branch, _BranchCacheSlot())

            # 1. Very first block (`group 0, block 0`) ALWAYS runs to probe `delta_0`.
            if is_first_overall:
                snap_single, snap_pair = _snapshot_pre_call(args, kwargs, self._group_is_pair[0])
                out = orig_fwd(*args, **kwargs)
                if not _is_valid_state(out):
                    self.skip_active = False
                    if self._on_miss_callback is not None:
                        self._on_miss_callback()
                    return out

                is_pair = isinstance(out, tuple)
                self._group_is_pair[0] = is_pair
                in_state = snap_pair if is_pair else snap_single
                if not _states_compatible(in_state, out):
                    self.skip_active = False
                    if self._on_miss_callback is not None:
                        self._on_miss_callback()
                    return out

                slot.steps_seen += 1
                self.total_steps += 1
                with torch.no_grad():
                    out_primary = out[0] if is_pair else out
                    in_primary = in_state[0] if is_pair else in_state
                    delta_0 = out_primary.detach() - in_primary.detach()

                    all_groups_ready = len(slot.group_residuals) == self._num_groups
                    can_skip = (
                        slot.steps_seen > self.warmup_steps
                        and slot.prev_delta_0 is not None
                        and all_groups_ready
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
                            if is_group_end and 0 in slot.group_residuals:
                                return _add_state(out, slot.group_residuals[0])
                            return out

                    self.skip_active = False
                    self.computed_steps += 1
                    self._cur_delta_0 = delta_0.clone()
                    if is_group_end:
                        # Single-block group 0 has zero intra-group residual after block 0.
                        zero_r = (
                            (torch.zeros_like(out[0]), torch.zeros_like(out[1]))
                            if is_pair
                            else torch.zeros_like(out)
                        )
                        slot.group_residuals[0] = self._store_state(zero_r)
                    else:
                        self._group_anchors[0] = _clone_state(out)
                    if self._on_miss_callback is not None:
                        self._on_miss_callback()
                return out

            # 2. Fast skip path for all blocks after `group 0, block 0`.
            if self.skip_active:
                cur_in = _extract_call_state(args, kwargs, self._group_is_pair.get(g_idx))
                if is_group_end and g_idx in slot.group_residuals:
                    return _add_state(cur_in, slot.group_residuals[g_idx])
                return cur_in

            # 3. Full-compute path: capture cloned group entry anchor & group exit residual.
            snap_s, snap_p = _snapshot_pre_call(args, kwargs, self._group_is_pair.get(g_idx)) if is_group_start else (None, None)
            out = orig_fwd(*args, **kwargs)
            if _is_valid_state(out):
                is_pair = isinstance(out, tuple)
                self._group_is_pair[g_idx] = is_pair
                if is_group_start:
                    anch = snap_p if is_pair else snap_s
                    if _states_compatible(anch, out):
                        self._group_anchors[g_idx] = anch

                if is_group_end and g_idx in self._group_anchors:
                    anch_val = self._group_anchors.pop(g_idx)
                    if _states_compatible(out, anch_val):
                        with torch.no_grad():
                            res_g = _sub_state(out, anch_val)
                            slot.group_residuals[g_idx] = self._store_state(res_g)
                            if g_idx == self._num_groups - 1 and self._cur_delta_0 is not None:
                                slot.prev_delta_0 = self._store_state(self._cur_delta_0)
                                slot.accumulated_rel_diff = 0.0
                                slot.consecutive_skips = 0
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

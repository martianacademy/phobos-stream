"""ComfyUI `CLIP -> CLIP` block-streaming patcher for T5-XXL, UMT5, Llama-3, Qwen-VL & DualCLIP."""

from __future__ import annotations

import gc
from typing import Any, Callable, Dict, List, Set, Tuple

from phobos_stream.streaming import BlockStreamMode, TorchBlockStreamer, discover_block_modules


def _move_non_block_clip_tensors(
    cond_stage_model: Any,
    block_prefixes: Set[str],
    large_embed_modules: Set[int],
    target_device: Any,
) -> None:
    """Move small non-block norms/projections to `target_device`, keeping giant vocab tables on CPU."""
    for mod_name, sub_mod in cond_stage_model.named_modules():
        if any(mod_name == p or mod_name.startswith(f"{p}.") for p in block_prefixes):
            continue
        if id(sub_mod) in large_embed_modules:
            for k, p in list(sub_mod._parameters.items()):
                if p is not None and not getattr(p, "is_meta", False) and p.device.type != "cpu":
                    sub_mod._parameters[k] = p.to("cpu")
            continue

        for k, p in list(sub_mod._parameters.items()):
            if p is not None and not getattr(p, "is_meta", False) and p.numel() > 0:
                if p.device != target_device:
                    sub_mod._parameters[k] = torch_param_to(p, target_device)
        for k, b in list(sub_mod._buffers.items()):
            if b is not None and not getattr(b, "is_meta", False) and b.numel() > 0:
                if b.device != target_device:
                    sub_mod._buffers[k] = b.to(target_device)


def torch_param_to(param: Any, target_device: Any) -> Any:
    import torch.nn as nn

    return nn.Parameter(param.detach().to(target_device), requires_grad=False)


def _install_cpu_vocab_embedding_wrappers(
    cond_stage_model: Any,
    block_prefixes: Set[str],
    min_bytes_threshold: int = 32 * 1024 * 1024,
) -> Tuple[Set[int], List[Tuple[Any, Callable[..., Any]]]]:
    """Keep multi-hundred-MB vocabulary embedding tables (`UMT5` 2.1 GB, `Llama` 1.1 GB) on CPU.

    Looks up the handful of prompt token rows on CPU (~0.05 ms) and moves only the small
    `[B, Seq, Dim]` result (~2-4 MB) onto the active accelerator (`0 MB` GPU VRAM!).
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    wrapped_ids: Set[int] = set()
    orig_forwards: List[Tuple[Any, Callable[..., Any]]] = []

    for mod_name, sub_mod in cond_stage_model.named_modules():
        if any(mod_name == p or mod_name.startswith(f"{p}.") for p in block_prefixes):
            continue
        if not isinstance(sub_mod, nn.Embedding):
            continue
        w = getattr(sub_mod, "weight", None)
        if w is None or getattr(w, "is_meta", False):
            continue
        if w.numel() * w.element_size() < min_bytes_threshold:
            continue

        if w.device.type != "cpu":
            sub_mod.weight = nn.Parameter(w.data.to("cpu"), requires_grad=False)
        wrapped_ids.add(id(sub_mod))
        orig_fwd = sub_mod.forward
        orig_forwards.append((sub_mod, orig_fwd))

        def _make_emb_fwd(emb_mod: Any, fallback_fwd: Callable[..., Any]) -> Callable[..., Any]:
            def _cpu_lookup(input_ids: Any, *args: Any, **kwargs: Any) -> Any:
                wt = emb_mod.weight
                if wt is None or wt.device.type != "cpu":
                    return fallback_fwd(input_ids, *args, **kwargs)
                out_dtype = kwargs.get("out_dtype", None)
                ids_cpu = input_ids.to("cpu")
                # Support FP8/quantized or standard float embedding weights
                w_cpu = wt.float() if wt.dtype not in (torch.float16, torch.bfloat16, torch.float32) else wt
                looked_up = F.embedding(
                    ids_cpu,
                    w_cpu,
                    emb_mod.padding_idx,
                    emb_mod.max_norm,
                    emb_mod.norm_type,
                    emb_mod.scale_grad_by_freq,
                    emb_mod.sparse,
                )
                target_dt = out_dtype if out_dtype is not None else looked_up.dtype
                return looked_up.to(device=input_ids.device, dtype=target_dt)

            return _cpu_lookup

        sub_mod.forward = _make_emb_fwd(sub_mod, orig_fwd)

    return wrapped_ids, orig_forwards


def patch_comfy_clip_for_streaming(
    clip: Any,
    mode: str = "cpu_offload",
    vram_budget_mb: float = 0.0,
    evict_after_encode: bool = True,
) -> Any:
    """Clone and patch a ComfyUI `CLIP` instance with per-block streaming + CPU vocab lookup."""
    import torch

    cloned = clip.clone() if hasattr(clip, "clone") else clip
    prev = getattr(cloned, "_phobos_clip_streamer", None)
    if prev is not None and hasattr(prev, "detach"):
        prev.detach()

    cond_mod = getattr(cloned, "cond_stage_model", None)
    if cond_mod is None:
        return cloned

    patcher = getattr(cloned, "patcher", None)
    if patcher is not None and hasattr(patcher, "patch_model") and getattr(patcher, "patches", None):
        try:
            patcher.patch_model(device_to=torch.device("cpu"))
        except Exception:
            pass

    blocks = discover_block_modules(cond_mod)
    block_prefixes = {b_name for b_name, _ in blocks}
    large_emb_ids, _ = _install_cpu_vocab_embedding_wrappers(cond_mod, block_prefixes)

    streamer: Any = None
    if blocks:
        enum_mode = BlockStreamMode(mode)
        budget = vram_budget_mb if (vram_budget_mb and vram_budget_mb > 0.0) else None
        streamer = TorchBlockStreamer(
            model=cond_mod,
            compute_device="auto",
            mode=enum_mode,
            vram_budget_mb=budget,
            attn_mode="auto",
        ).attach()
        cloned._phobos_clip_streamer = streamer

    # Resolve target accelerator device from ComfyUI patcher or PyTorch runtime
    def _resolve_exec_dev() -> Any:
        if patcher is not None and getattr(patcher, "load_device", None) is not None:
            return torch.device(patcher.load_device)
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # Neutralize `clip.load_model` and `patcher.load` so ComfyUI never pulls all blocks onto GPU!
    if patcher is not None:
        if hasattr(patcher, "load"):
            patcher.load = lambda *a, **kw: None
        if hasattr(patcher, "patch_model"):
            patcher.patch_model = lambda *a, **kw: cond_mod
        if hasattr(patcher, "model_size"):
            patcher.model_size = lambda: 0

    if hasattr(cloned, "load_model"):
        cloned.load_model = lambda *a, **kw: patcher

    def _wrap_encode_fn(orig_fn: Callable[..., Any]) -> Callable[..., Any]:
        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            exec_dev = _resolve_exec_dev()
            _move_non_block_clip_tensors(cond_mod, block_prefixes, large_emb_ids, exec_dev)
            try:
                return orig_fn(*args, **kwargs)
            finally:
                if evict_after_encode and exec_dev.type != "cpu":
                    _move_non_block_clip_tensors(cond_mod, block_prefixes, large_emb_ids, torch.device("cpu"))
                    gc.collect()
                    if exec_dev.type == "cuda" and torch.cuda.is_available():
                        torch.cuda.empty_cache()

        return _wrapped

    if hasattr(cloned, "encode_from_tokens"):
        cloned.encode_from_tokens = _wrap_encode_fn(cloned.encode_from_tokens)
    if hasattr(cloned, "encode_from_tokens_scheduled"):
        cloned.encode_from_tokens_scheduled = _wrap_encode_fn(cloned.encode_from_tokens_scheduled)

    return cloned


class PhobosCLIPStreamPatcherNode:
    """ComfyUI Node: Streams `CLIP` / `T5-XXL` / `UMT5` / `Llama-3` / `Qwen-VL` in `< 1 GB` VRAM."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "clip": ("CLIP",),
                "mode": (["cpu_offload", "hybrid_budget"], {"default": "cpu_offload"}),
            },
            "optional": {
                "vram_budget_mb": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 65536.0, "step": 256.0},
                ),
                "evict_after_encode": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "patch_clip"
    CATEGORY = "PhobosStream"

    def patch_clip(
        self,
        clip: Any,
        mode: str = "cpu_offload",
        vram_budget_mb: float = 0.0,
        evict_after_encode: bool = True,
    ) -> Tuple[Any]:
        patched = patch_comfy_clip_for_streaming(
            clip=clip,
            mode=mode,
            vram_budget_mb=vram_budget_mb,
            evict_after_encode=evict_after_encode,
        )
        return (patched,)

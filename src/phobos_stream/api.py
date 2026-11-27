"""One-line universal Python SDK (`phobos_stream.stream`) for any PyTorch `nn.Module`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Union

from phobos_stream.streaming import BlockStreamMode, TorchBlockStreamer


def stream(
    model: Any,
    *,
    mode: Union[str, BlockStreamMode] = "hybrid_budget",
    compute_device: str = "auto",
    vram_budget_mb: Optional[float] = None,
    ram_budget_mb: Optional[float] = None,
    safetensors_paths: Optional[Sequence[Union[str, Path]]] = None,
    cache_thresh: float = 0.0,
    cache_warmup_steps: int = 2,
    cache_max_consecutive_skips: int = 2,
    attention_mode: str = "auto",
) -> Any:
    """Attach zero-VRAM block streaming, Zero-DMA TeaCache, and lossless SDPA to `model` in 1 line.

    Example:
        ```python
        import phobos_stream

        model = phobos_stream.stream(model, mode="cpu_offload", cache_thresh=0.12)
        out = model(latents, timestep, context)
        # Inspect `model._phobos_streamer.bytes_streamed` or call `phobos_stream.unstream(model)`
        ```
    """
    prev = getattr(model, "_phobos_streamer", None)
    if prev is not None and hasattr(prev, "detach"):
        prev.detach()

    enum_mode = BlockStreamMode(mode) if isinstance(mode, str) else mode
    resolved_paths = [Path(p) for p in safetensors_paths] if safetensors_paths else None

    # Auto-downgrade `hybrid_budget` to `cpu_offload` when weights are already materialized in RAM
    # and no external `.safetensors` / `.gguf` disk paths were supplied.
    if enum_mode == BlockStreamMode.HYBRID_BUDGET and not resolved_paths:
        has_meta = any(getattr(p, "is_meta", False) for p in model.parameters())
        if not has_meta and vram_budget_mb is None:
            enum_mode = BlockStreamMode.CPU_OFFLOAD

    streamer = TorchBlockStreamer(
        model=model,
        compute_device=compute_device,
        mode=enum_mode,
        safetensors_paths=resolved_paths,
        vram_budget_mb=vram_budget_mb,
        ram_budget_mb=ram_budget_mb,
        cache_rel_l1_thresh=cache_thresh,
        cache_warmup_steps=cache_warmup_steps,
        cache_max_consecutive_skips=cache_max_consecutive_skips,
        attn_mode=attention_mode,
    ).attach()
    model._phobos_streamer = streamer
    return model


def unstream(model: Any, restore_cpu_weights: bool = True) -> Any:
    """Detach `TorchBlockStreamer` hooks from `model` and optionally restore full CPU parameters."""
    streamer = getattr(model, "_phobos_streamer", None)
    if streamer is not None and hasattr(streamer, "detach"):
        streamer.detach(restore_cpu_weights=restore_cpu_weights)
        delattr(model, "_phobos_streamer")
    return model

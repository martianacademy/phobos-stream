"""Standalone ComfyUI node for PhobosStream First-Block Residual Cache (TeaCache / FBCache)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from phobos_stream.integrations.comfyui.comfy_runtime import attach_streamer_to_comfy_model


class PhobosFirstBlockCacheNode:
    """ComfyUI Node: Accelerates any DiT `MODEL` via First-Block Residual Caching (1.5x-2.5x faster).

    Works standalone (`stage_only`, zero streaming) OR chained with `PhobosBlockStreamPatcher`
    (if a `TorchBlockStreamer` is already attached on `model`, attaches the residual cache directly
    onto that active streamer so skipped steps also skip all PCIe weight DMA!).
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "rel_l1_thresh": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "warmup_steps": ("INT", {"default": 2, "min": 1, "max": 20}),
                "max_consecutive_skips": ("INT", {"default": 3, "min": 1, "max": 10}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_cache"
    CATEGORY = "PhobosStream"

    def apply_cache(
        self,
        model: Any,
        rel_l1_thresh: float = 0.12,
        warmup_steps: int = 2,
        max_consecutive_skips: int = 3,
    ) -> Tuple[Any]:
        patched = model.clone() if hasattr(model, "clone") else model
        existing_streamer = getattr(patched, "_phobos_block_streamer", None)
        if existing_streamer is not None and getattr(existing_streamer, "_attached", False):
            from phobos_stream.streaming.budget_planner import BlockTier
            from phobos_stream.streaming.first_block_cache import FirstBlockResidualCache

            if existing_streamer.residual_cache is not None:
                existing_streamer.residual_cache.detach()
            cache = FirstBlockResidualCache(
                rel_l1_thresh=rel_l1_thresh,
                warmup_steps=warmup_steps,
                max_consecutive_skips=max_consecutive_skips,
            )
            existing_streamer.residual_cache = cache if cache.enabled else None
            if cache.enabled:
                cache.attach(existing_streamer.model, blocks=existing_streamer.blocks)
                p_idx = cache.probe_block_index
                first_post_probe_cpu = next(
                    (
                        i
                        for i, t in enumerate(existing_streamer._effective_tiers)
                        if i > p_idx and t == BlockTier.CPU_PINNED
                    ),
                    None,
                )
                if first_post_probe_cpu is not None and existing_streamer._prefetch_stream is not None:
                    cache._on_miss_callback = lambda: existing_streamer._prefetch_cpu_block(first_post_probe_cpu)
            return (patched,)

        attach_streamer_to_comfy_model(
            comfy_model=patched,
            block_stream_mode="stage_only",
            cache_rel_l1_thresh=rel_l1_thresh,
            cache_warmup_steps=warmup_steps,
            cache_max_consecutive_skips=max_consecutive_skips,
        )
        return (patched,)

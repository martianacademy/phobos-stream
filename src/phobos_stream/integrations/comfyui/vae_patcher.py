"""ComfyUI VAE zero-VRAM block-streaming patcher & overlapped 1D Audio VAE chunk decoder.

Solves out-of-memory crashes on large Video VAEs (such as MiniMax-H3's 4.85 GB 36-block
ViT3D Video VAE and Wan2.1/Hunyuan 3D VAEs) and 1D Audio VAEs (such as 32 kHz Stereo
BigVGAN) by:
1. Optionally dropping unused `encoder` and `quant_conv` weights before decode.
2. Attaching `TorchBlockStreamer` (`hybrid_budget`, `cpu_offload`, or `mmap_stream`) to
   repeating decoder blocks while preventing ComfyUI's `ModelPatcher` from reloading
   offloaded blocks back onto GPU.
3. Automatically routing 3D audio latents `[B, C, T]` through smooth linear-crossfade
   overlap-add chunked decoding so long stereo waveforms decode in <700 MB VRAM without
   triggering ComfyUI's 2D spatial tiler.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from phobos_stream.streaming import BlockStreamMode, TorchBlockStreamer
from phobos_stream.streaming.block_discovery import discover_block_modules

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

try:
    import folder_paths  # type: ignore
except Exception:
    folder_paths = None


def _flush_accelerator_cache() -> None:
    """Release fragmented GPU/MPS memory and Python garbage after dropping encoder weights."""
    gc.collect()
    if not TORCH_AVAILABLE or torch is None:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def decode_audio_latent_chunked(
    decode_fn: Callable[[Any], Any],
    latent: Any,
    chunk_steps: int = 64,
    overlap_steps: int = 8,
    device: Optional[Any] = None,
    dtype: Optional[Any] = None,
) -> Any:
    """Decode a 1D audio latent `[B, C, T]` in overlapped chunks with linear crossfade.

    Parameters
    ----------
    decode_fn:
        Callable mapping a latent chunk `[B, C, t]` to waveform `[B, Ch, t * hop]`.
    latent:
        Input audio latent tensor of shape `[B, C, T]` (or `[C, T]`).
    chunk_steps:
        Maximum latent timesteps decoded per chunk (e.g., 64 ~= 1.6s on 40 Hz latents).
    overlap_steps:
        Latent overlap on each seam for smooth linear overlap-add crossfading.
    device / dtype:
        Optional compute device and dtype to move each chunk to before calling `decode_fn`.

    Returns
    -------
    torch.Tensor
        Full reconstructed CPU float32 waveform with exact `[B, Ch, T * hop]` length.
    """
    if not TORCH_AVAILABLE or torch is None:
        raise RuntimeError("PyTorch is required for chunked audio VAE decoding.")
    if chunk_steps < 1:
        raise ValueError(f"chunk_steps must be >= 1, got {chunk_steps}.")
    if overlap_steps < 0 or overlap_steps >= chunk_steps:
        raise ValueError(f"overlap_steps ({overlap_steps}) must be in [0, chunk_steps={chunk_steps}).")

    squeezed = False
    z = latent
    if z.ndim == 2:
        z = z.unsqueeze(0)
        squeezed = True
    if z.ndim not in (3, 4):
        raise ValueError(f"Expected 3D [B, C, T] or 4D [B, C, Ch, T] audio latent, got shape {tuple(z.shape)}.")

    total_steps = int(z.shape[-1])
    if total_steps <= chunk_steps:
        z_in = z.to(device=device if device is not None else z.device, dtype=dtype if dtype is not None else z.dtype)
        with torch.no_grad():
            out = decode_fn(z_in).detach().float().cpu()
        return out.squeeze(0) if squeezed else out

    stride = chunk_steps - overlap_steps
    acc: Optional[Any] = None
    weights: Optional[Any] = None
    hop: Optional[int] = None

    start = 0
    while start < total_steps:
        end = min(start + chunk_steps, total_steps)
        chunk = z[..., start:end]
        if device is not None or dtype is not None:
            chunk = chunk.to(
                device=device if device is not None else chunk.device,
                dtype=dtype if dtype is not None else chunk.dtype,
            )

        with torch.no_grad():
            out_chunk = decode_fn(chunk).detach().float().cpu()

        chunk_len = end - start
        out_len = int(out_chunk.shape[-1])
        if hop is None:
            if out_len % chunk_len != 0:
                raise ValueError(
                    f"Audio decoder output length ({out_len}) is not an integer multiple of latent chunk ({chunk_len})."
                )
            hop = out_len // chunk_len
            acc = torch.zeros((*out_chunk.shape[:-1], total_steps * hop), dtype=torch.float32)
            weights = torch.zeros(total_steps * hop, dtype=torch.float32)

        assert acc is not None and weights is not None and hop is not None
        win = torch.ones(out_len, dtype=torch.float32)
        ov_samples = min(overlap_steps * hop, out_len // 2)
        if ov_samples > 0:
            ramp = torch.linspace(0.0, 1.0, ov_samples + 2, dtype=torch.float32)[1:-1]
            if start > 0:
                win[:ov_samples] = ramp
            if end < total_steps:
                win[-ov_samples:] = ramp.flip(0)

        out_start = start * hop
        out_end = out_start + out_len
        acc[..., out_start:out_end] += out_chunk * win
        weights[out_start:out_end] += win

        del chunk, out_chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if end >= total_steps:
            break
        start += stride

    assert acc is not None and weights is not None
    result = acc / weights.clamp_min(1e-8)
    return result.squeeze(0) if squeezed else result


def _resolve_vae_safetensors(vae_path: str) -> List[Path]:
    """Resolve an optional `.safetensors` path directly or via ComfyUI `folder_paths`."""
    cleaned = (vae_path or "").strip()
    if not cleaned:
        return []
    direct = Path(cleaned)
    if direct.exists():
        return [direct]
    if folder_paths is not None:
        try:
            resolved = folder_paths.get_full_path("vae", cleaned)
            if resolved and Path(resolved).exists():
                return [Path(resolved)]
        except Exception:
            pass
    return []


def _drop_unused_encoder_modules(first_stage_model: Any) -> List[str]:
    """Release `encoder` and `quant_conv` on decode-only VAEs to reclaim ~50% of weights."""
    dropped: List[str] = []
    if not hasattr(first_stage_model, "decoder"):
        return dropped
    for attr in ("encoder", "quant_conv"):
        sub = getattr(first_stage_model, attr, None)
        if sub is not None:
            setattr(first_stage_model, attr, None)
            dropped.append(attr)
    if dropped:
        _flush_accelerator_cache()
    return dropped


def _infer_vae_compute_device(vae: Any, fallback: str = "auto") -> str:
    """Resolve the compute device used by a ComfyUI `VAE` or PyTorch module."""
    if fallback != "auto":
        return fallback
    for attr in ("device", "load_device"):
        dev = getattr(vae, attr, None)
        if dev is not None:
            return str(dev)
    if TORCH_AVAILABLE and torch is not None:
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def attach_streamer_to_comfy_vae(
    comfy_vae: Any,
    block_stream_mode: str = "hybrid_budget",
    vram_budget_mb: Optional[float] = None,
    drop_encoder: bool = True,
    audio_chunk_steps: int = 64,
    audio_overlap_steps: int = 8,
    safetensors_path: str = "",
    compute_device: str = "auto",
) -> Dict[str, Any]:
    """Patch a ComfyUI `VAE` (or PyTorch VAE module) with block streaming & audio chunking.

    Returns a metadata dict containing the active `streamer` (if repeating blocks were
    discovered), `dropped_modules`, `mode`, and `audio_chunk_steps`.
    """
    mode = BlockStreamMode(block_stream_mode)
    prev_streamer = getattr(comfy_vae, "_phobos_vae_streamer", None)
    if prev_streamer is not None and hasattr(prev_streamer, "detach"):
        prev_streamer.detach()

    first_stage = getattr(comfy_vae, "first_stage_model", comfy_vae)
    dropped = _drop_unused_encoder_modules(first_stage) if drop_encoder else []

    target_module = getattr(first_stage, "decoder", None) or first_stage
    target_device = _infer_vae_compute_device(comfy_vae, fallback=compute_device)
    st_paths = _resolve_vae_safetensors(safetensors_path)

    streamer: Optional[TorchBlockStreamer] = None
    discovered_blocks = discover_block_modules(target_module) if mode != BlockStreamMode.STAGE_ONLY else []
    vae_dtype = getattr(comfy_vae, "vae_dtype", None)
    if discovered_blocks:
        if vae_dtype is not None and hasattr(first_stage, "to"):
            first_stage.to(dtype=vae_dtype)

        streamer = TorchBlockStreamer(
            model=target_module,
            compute_device=target_device,
            mode=mode,
            safetensors_paths=st_paths,
            vram_budget_mb=vram_budget_mb if (vram_budget_mb and vram_budget_mb > 0) else None,
        )
        streamer.attach()

        # Keep small decode-path siblings (e.g. `post_quant_conv`, audio stems) & buffers on compute_device.
        for child_name, child_mod in first_stage.named_children():
            if child_mod is None or child_mod is target_module or child_name in ("encoder", "quant_conv"):
                continue
            if hasattr(child_mod, "to"):
                child_mod.to(device=streamer.compute_device, dtype=vae_dtype if vae_dtype is not None else None)
        if TORCH_AVAILABLE and torch is not None:
            for buf_name in ("latents_mean", "latents_std", "pixel_mean", "pixel_std"):
                buf_val = getattr(first_stage, buf_name, None)
                if isinstance(buf_val, torch.Tensor):
                    setattr(first_stage, buf_name, buf_val.to(streamer.compute_device))

        # MiniMaxH3 ViT3D VAE: force 1-tile streaming to avoid multi-tile GPU buffer OOMs.
        if (
            hasattr(first_stage, "_decode_tile_row")
            and hasattr(first_stage, "_decode_pixels")
            and hasattr(first_stage, "vae_ratio")
        ):
            def _single_tile_row(z_row: Any, x_idx: Any, x_len: Any) -> Any:
                ratio = first_stage.vae_ratio
                for j_pos, j_len in zip(x_idx, x_len):
                    sl = z_row[..., j_pos // ratio : (j_pos + j_len) // ratio]
                    yield from first_stage._decode_pixels(sl).chunk(1)

            first_stage._decode_tile_row = _single_tile_row

        # Prevent ComfyUI's ModelPatcher from hauling offloaded decoder blocks back onto GPU.
        patcher = getattr(comfy_vae, "patcher", None)
        if patcher is not None:
            patcher._phobos_orig_load = getattr(patcher, "_phobos_orig_load", getattr(patcher, "load", None))
            patcher.load = lambda *args, **kwargs: None
            if hasattr(patcher, "patch_model"):
                patcher._phobos_orig_patch_model = getattr(patcher, "_phobos_orig_patch_model", patcher.patch_model)
                patcher.patch_model = lambda *args, **kwargs: getattr(patcher, "model", first_stage)
            if hasattr(patcher, "model_size"):
                patcher.model_size = lambda: 0

    def _wrap_decode_method(attr_name: str) -> None:
        if not hasattr(comfy_vae, attr_name):
            return
        orig_key = f"_phobos_orig_{attr_name}"
        orig_fn = getattr(comfy_vae, orig_key, getattr(comfy_vae, attr_name))
        setattr(comfy_vae, orig_key, orig_fn)

        def _phobos_decode_wrapper(samples_in: Any, *args: Any, **kwargs: Any) -> Any:
            if (
                TORCH_AVAILABLE
                and torch is not None
                and isinstance(samples_in, torch.Tensor)
                and samples_in.ndim == 5
                and streamer is not None
                and hasattr(first_stage, "decode_output_shape")
                and callable(getattr(first_stage, "decode", None))
            ):
                out_shape = first_stage.decode_output_shape(samples_in.shape)
                out_cpu = torch.empty(out_shape, dtype=torch.float32, device="cpu")
                with torch.no_grad():
                    z_dev = samples_in.to(
                        device=streamer.compute_device,
                        dtype=vae_dtype if vae_dtype is not None else samples_in.dtype,
                    )
                    first_stage.decode(z_dev, output_buffer=out_cpu)
                    del z_dev
                # Normalize `(B, C, T, H, W)` -> ComfyUI `(B, T, H, W, C)` float32 in [0, 1].
                if out_cpu.ndim == 5 and out_cpu.shape[1] in (3, 4):
                    return out_cpu.permute(0, 2, 3, 4, 1).clamp(0.0, 1.0)
                return out_cpu

            is_audio_latent = False
            if TORCH_AVAILABLE and torch is not None and isinstance(samples_in, torch.Tensor):
                if samples_in.ndim == 3:
                    is_audio_latent = True
                elif samples_in.ndim == 4:
                    cls_hint = f"{type(comfy_vae).__name__} {type(first_stage).__name__}".lower()
                    is_audio_cls = any(k in cls_hint for k in ("audio", "bigvgan", "oobleck", "dac", "encodec"))
                    is_stereo_4d = int(samples_in.shape[2]) in (1, 2) and int(samples_in.shape[3]) > int(samples_in.shape[2]) * 4
                    is_audio_latent = is_audio_cls or is_stereo_4d

            if (
                is_audio_latent
                and audio_chunk_steps > 0
                and int(samples_in.shape[-1]) > audio_chunk_steps
            ):
                decode_callable = getattr(first_stage, "decode", None)
                if callable(decode_callable):
                    dev = streamer.compute_device if streamer is not None else torch.device(target_device)
                    if streamer is None and hasattr(first_stage, "to"):
                        first_stage.to(device=dev, dtype=vae_dtype if vae_dtype is not None else None)
                    return decode_audio_latent_chunked(
                        decode_fn=decode_callable,
                        latent=samples_in,
                        chunk_steps=audio_chunk_steps,
                        overlap_steps=audio_overlap_steps,
                        device=dev,
                        dtype=vae_dtype,
                    )
                return decode_audio_latent_chunked(
                    decode_fn=lambda chunk: orig_fn(chunk, *args, **kwargs),
                    latent=samples_in,
                    chunk_steps=audio_chunk_steps,
                    overlap_steps=audio_overlap_steps,
                )
            return orig_fn(samples_in, *args, **kwargs)

        setattr(comfy_vae, attr_name, _phobos_decode_wrapper)

    # Wrap `decode` and `decode_tiled` so 3D audio latents `[B, C, T]` use overlap-add chunking safely.
    _wrap_decode_method("decode")
    _wrap_decode_method("decode_tiled")

    meta: Dict[str, Any] = {
        "streamer": streamer,
        "dropped_modules": dropped,
        "mode": mode.value,
        "num_streamed_blocks": len(discovered_blocks),
        "audio_chunk_steps": audio_chunk_steps,
        "audio_overlap_steps": audio_overlap_steps,
        "tier_summary": streamer.tier_summary if streamer is not None else {},
    }
    setattr(comfy_vae, "_phobos_vae_streamer", streamer)
    setattr(comfy_vae, "_phobos_vae_meta", meta)
    return meta


class PhobosVAEStreamPatcherNode:
    """ComfyUI Node: Patches any Video or Audio `VAE` with zero-VRAM block streaming & chunking."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "vae": ("VAE",),
                "mode": (["hybrid_budget", "cpu_offload", "mmap_stream"], {"default": "hybrid_budget"}),
            },
            "optional": {
                "vram_budget_mb": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 65536.0, "step": 256.0}),
                "drop_encoder": ("BOOLEAN", {"default": True}),
                "audio_chunk_steps": ("INT", {"default": 64, "min": 8, "max": 512, "step": 8}),
                "audio_overlap_steps": ("INT", {"default": 8, "min": 0, "max": 64, "step": 2}),
                "safetensors_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "patch_vae"
    CATEGORY = "PhobosStream"

    def patch_vae(
        self,
        vae: Any,
        mode: str = "hybrid_budget",
        vram_budget_mb: float = 0.0,
        drop_encoder: bool = True,
        audio_chunk_steps: int = 64,
        audio_overlap_steps: int = 8,
        safetensors_path: str = "",
    ) -> Tuple[Any]:
        attach_streamer_to_comfy_vae(
            comfy_vae=vae,
            block_stream_mode=mode,
            vram_budget_mb=vram_budget_mb if vram_budget_mb > 0 else None,
            drop_encoder=drop_encoder,
            audio_chunk_steps=audio_chunk_steps,
            audio_overlap_steps=audio_overlap_steps,
            safetensors_path=safetensors_path,
        )
        return (vae,)

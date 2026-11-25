"""ComfyUI runtime bridge: 3-stage sequential offloading, dynamic 4D/5D latents, DualCLIP, and block streaming."""

import gc
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from phobos_stream.streaming import BlockStreamMode, TorchBlockStreamer

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

COMFY_NATIVE_AVAILABLE = False
try:
    import folder_paths  # type: ignore
    import comfy.model_management as comfy_mm  # type: ignore
    import nodes as comfy_nodes  # type: ignore

    # Optional user-configured shared model folder via environment variable (no hardcoded personal paths)
    _extra_models_env = os.environ.get("PHOBOS_EXTRA_MODELS_DIR", "").strip()
    if _extra_models_env:
        _extra_root = Path(_extra_models_env).expanduser()
        if _extra_root.exists():
            for _cat in ("diffusion_models", "text_encoders", "vae", "checkpoints", "loras"):
                _cat_dir = _extra_root / _cat
                if _cat_dir.exists():
                    folder_paths.add_model_folder_path(_cat, str(_cat_dir))
    COMFY_NATIVE_AVAILABLE = True
except Exception:
    folder_paths = None
    comfy_mm = None
    comfy_nodes = None


DUAL_CLIP_TYPES = {
    "flux",
    "sd3",
    "sdxl",
    "hunyuan_video",
    "hunyuan_image",
    "hunyuan_video_15",
    "hidream",
    "ltxv",
}


def free_comfy_memory() -> None:
    """Unload all active ComfyUI models and flush CUDA/MPS/MLX/CPU caches between stages."""
    if COMFY_NATIVE_AVAILABLE and comfy_mm is not None:
        try:
            comfy_mm.unload_all_models()
            comfy_mm.soft_empty_cache()
        except Exception:
            pass
    gc.collect()
    if TORCH_AVAILABLE and torch is not None:
        if hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_comfy_filenames(category: str, fallback: List[str]) -> List[str]:
    """Query ComfyUI `folder_paths` safely with a deterministic fallback list."""
    if COMFY_NATIVE_AVAILABLE and folder_paths is not None:
        try:
            found = folder_paths.get_filename_list(category)
            if found:
                return list(found)
        except Exception:
            pass
    return list(fallback)


def to_comfy_image_tensor(rgb_frames: Any) -> Any:
    """Normalize `(T, H, W, 3)` or 5D `(B, T, H, W, 3)` output into ComfyUI 4D float32 `[0, 1]` IMAGE."""
    if TORCH_AVAILABLE and torch is not None and isinstance(rgb_frames, torch.Tensor):
        t = rgb_frames
        if t.ndim == 5:
            b, f, h, w, c = t.shape
            t = t.reshape(b * f, h, w, c)
        elif t.ndim == 3:
            t = t.unsqueeze(0)
        return t.float()

    arr = np.asarray(rgb_frames)
    if arr.dtype == np.uint8:
        img_np = arr.astype(np.float32) / 255.0
    else:
        img_np = np.clip(arr.astype(np.float32), 0.0, 1.0)

    if img_np.ndim == 5:
        b, f, h, w, c = img_np.shape
        img_np = img_np.reshape(b * f, h, w, c)
    elif img_np.ndim == 3:
        img_np = np.expand_dims(img_np, axis=0)
    elif img_np.ndim == 2:
        img_np = np.repeat(img_np[None, :, :, None], 3, axis=-1)

    if TORCH_AVAILABLE and torch is not None:
        return torch.from_numpy(img_np)
    return img_np


def build_empty_comfy_latent(
    active_model: Any,
    width: int,
    height: int,
    num_frames: int = 1,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """Build an exact 4D image or 5D video latent dict matching `active_model`'s `latent_format`."""
    if not TORCH_AVAILABLE or torch is None:
        raise RuntimeError("PyTorch is required to allocate ComfyUI latent tensors.")

    latent_fmt = None
    if hasattr(active_model, "get_model_object"):
        try:
            latent_fmt = active_model.get_model_object("latent_format")
        except Exception:
            latent_fmt = None

    channels = int(getattr(latent_fmt, "latent_channels", 16))
    latent_dims = int(getattr(latent_fmt, "latent_dimensions", 3 if num_frames > 1 else 2))
    sp_ratio = int(getattr(latent_fmt, "spacial_downscale_ratio", 8))
    tp_ratio = int(getattr(latent_fmt, "temporal_downscale_ratio", 4 if latent_dims == 3 else 1))

    h_lat = max(1, int(height) // max(1, sp_ratio))
    w_lat = max(1, int(width) // max(1, sp_ratio))

    if latent_dims == 3 or int(num_frames) > 1:
        t_lat = max(1, (max(1, int(num_frames)) - 1) // max(1, tp_ratio) + 1)
        samples = torch.zeros([batch_size, channels, t_lat, h_lat, w_lat], device="cpu", dtype=torch.float32)
    else:
        samples = torch.zeros([batch_size, channels, h_lat, w_lat], device="cpu", dtype=torch.float32)

    return {"samples": samples, "downscale_ratio_spacial": sp_ratio}


def load_comfy_clip(clip_name: str, clip_type: str, clip_name2: str = "") -> Any:
    """Load single CLIP/T5/Qwen via `CLIPLoader` or dual encoders via `DualCLIPLoader`."""
    if not COMFY_NATIVE_AVAILABLE or comfy_nodes is None:
        raise RuntimeError("ComfyUI runtime is not loaded.")

    second = (clip_name2 or "").strip()
    if second and second != "none" and hasattr(comfy_nodes, "DualCLIPLoader"):
        dual_type = clip_type if clip_type in DUAL_CLIP_TYPES else "flux"
        clip, = comfy_nodes.DualCLIPLoader().load_clip(clip_name, second, dual_type)
        return clip

    clip, = comfy_nodes.CLIPLoader().load_clip(clip_name, clip_type)
    return clip


def decode_comfy_vae_smart(
    vae: Any,
    sampled_latent: Dict[str, Any],
    use_tiled_vae: bool = True,
    tile_size: int = 512,
    overlap: int = 64,
) -> Any:
    """Decode ComfyUI latents using tiled VAE when beneficial to avoid 2K/4K/Video OOMs."""
    if not COMFY_NATIVE_AVAILABLE or comfy_nodes is None:
        raise RuntimeError("ComfyUI runtime is not loaded.")

    samples = sampled_latent.get("samples")
    is_large_or_video = False
    if samples is not None and hasattr(samples, "shape"):
        is_large_or_video = len(samples.shape) == 5 or max(int(samples.shape[-1]), int(samples.shape[-2])) >= 192

    if (use_tiled_vae or is_large_or_video) and hasattr(comfy_nodes, "VAEDecodeTiled"):
        try:
            decoded, = comfy_nodes.VAEDecodeTiled().decode(
                vae=vae,
                samples=sampled_latent,
                tile_size=tile_size,
                overlap=overlap,
                temporal_size=64,
                temporal_overlap=8,
            )
            return to_comfy_image_tensor(decoded)
        except Exception:
            pass

    decoded, = comfy_nodes.VAEDecode().decode(vae, sampled_latent)
    return to_comfy_image_tensor(decoded)


def attach_streamer_to_comfy_model(
    comfy_model: Any,
    block_stream_mode: str = "hybrid_budget",
    unet_name: str = "",
    vram_budget_mb: Optional[float] = None,
) -> Optional[TorchBlockStreamer]:
    """Attach `TorchBlockStreamer` onto a ComfyUI `ModelPatcher`'s underlying `diffusion_model`."""
    mode = BlockStreamMode(block_stream_mode)
    if mode == BlockStreamMode.STAGE_ONLY:
        return None

    diff_mod = None
    if hasattr(comfy_model, "get_model_object"):
        try:
            diff_mod = comfy_model.get_model_object("diffusion_model")
        except Exception:
            diff_mod = None
    if diff_mod is None and hasattr(comfy_model, "model"):
        diff_mod = getattr(comfy_model.model, "diffusion_model", comfy_model.model)
    if diff_mod is None:
        diff_mod = comfy_model

    st_paths: List[Path] = []
    if mode in (BlockStreamMode.MMAP_STREAM, BlockStreamMode.HYBRID_BUDGET):
        if unet_name and Path(unet_name).exists():
            st_paths.append(Path(unet_name))
        elif unet_name and COMFY_NATIVE_AVAILABLE and folder_paths is not None:
            resolved = folder_paths.get_full_path("diffusion_models", unet_name)
            if resolved:
                st_paths.append(Path(resolved))

    streamer = TorchBlockStreamer(
        model=diff_mod,
        compute_device="auto",
        mode=mode,
        safetensors_paths=st_paths,
        vram_budget_mb=vram_budget_mb if (vram_budget_mb and vram_budget_mb > 0) else None,
    )
    streamer.attach()
    return streamer


def run_comfy_stage_offloaded_pipeline(
    unet_name: str,
    clip_name: str,
    clip_type: str,
    vae_name: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    seed: int,
    num_frames: int = 1,
    sampler_name: str = "res_multistep",
    scheduler: str = "simple",
    aura_shift: float = 3.0,
    clip_name2: str = "",
    weight_dtype: str = "default",
    block_stream_mode: str = "stage_only",
    use_tiled_vae: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """Execute strict 3-stage sequential offloading in ComfyUI (`Text -> DiT -> VAE`)."""
    if not COMFY_NATIVE_AVAILABLE or comfy_nodes is None:
        raise RuntimeError("ComfyUI runtime is not loaded.")

    free_comfy_memory()

    # Stage 1: Load Text Encoder(s), encode conditioning, and immediately unload
    clip = load_comfy_clip(clip_name=clip_name, clip_type=clip_type, clip_name2=clip_name2)
    positive, = comfy_nodes.CLIPTextEncode().encode(clip, prompt)
    if negative_prompt and negative_prompt.strip() and cfg > 1.0:
        negative, = comfy_nodes.CLIPTextEncode().encode(clip, negative_prompt)
    else:
        negative, = comfy_nodes.ConditioningZeroOut().zero_out(positive)
    del clip
    free_comfy_memory()

    # Stage 2: Load Diffusion Transformer / UNet, optionally attach block streamer, sample, unload
    model, = comfy_nodes.UNETLoader().load_unet(unet_name, weight_dtype)
    if aura_shift > 0:
        try:
            from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow  # type: ignore

            active_model, = ModelSamplingAuraFlow().patch_aura(model, aura_shift)
        except Exception:
            active_model = model
    else:
        active_model = model

    latent = build_empty_comfy_latent(
        active_model=active_model,
        width=width,
        height=height,
        num_frames=num_frames,
    )
    streamer: Optional[TorchBlockStreamer] = None
    try:
        if block_stream_mode != BlockStreamMode.STAGE_ONLY.value:
            streamer = attach_streamer_to_comfy_model(
                active_model,
                block_stream_mode=block_stream_mode,
                unet_name=unet_name,
            )
        sampled_latent, = comfy_nodes.KSampler().sample(
            model=active_model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent,
            denoise=1.0,
        )
    finally:
        if streamer is not None:
            streamer.detach(restore_cpu_weights=False)
        del active_model, model, positive, negative
        free_comfy_memory()

    # Stage 3: Load VAE, decode (tiled when appropriate), and immediately unload
    vae, = comfy_nodes.VAELoader().load_vae(vae_name)
    try:
        decoded_image = decode_comfy_vae_smart(vae, sampled_latent, use_tiled_vae=use_tiled_vae)
    finally:
        del vae
        free_comfy_memory()

    latent_bundle = {
        "samples": sampled_latent["samples"],
        "comfy_latent": sampled_latent,
        "decoded_image": decoded_image,
        "vae_name": vae_name,
        "width": width,
        "height": height,
        "num_frames": num_frames,
    }
    return decoded_image, latent_bundle

"""ComfyUI custom node implementations for PhobosStream zero-VRAM streaming & stage offloading."""

import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from phobos_stream.pipeline.stream_engine import PhobosStreamEngine
from phobos_stream.storage.layout import ModelShardsManifest
from phobos_stream.types import BackendType, GenerationConfig

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Detect whether we are running inside an active ComfyUI runtime
COMFY_NATIVE_AVAILABLE = False
try:
    import folder_paths  # type: ignore
    import comfy.model_management as comfy_mm  # type: ignore
    import nodes as comfy_nodes  # type: ignore

    # Automatically register shared model directories if present on Mac
    _SHARED_MODELS = Path("/Users/suru.martian/ComfyUI-Shared/models")
    if _SHARED_MODELS.exists():
        for _cat in ("diffusion_models", "text_encoders", "vae", "checkpoints", "loras"):
            _p = _SHARED_MODELS / _cat
            if _p.exists():
                folder_paths.add_model_folder_path(_cat, str(_p))
    COMFY_NATIVE_AVAILABLE = True
except Exception:
    folder_paths = None
    comfy_mm = None
    comfy_nodes = None


def _free_comfy_memory() -> None:
    """Strictly unload all active ComfyUI models and clear Metal/CUDA/CPU caches between stages."""
    if COMFY_NATIVE_AVAILABLE and comfy_mm is not None:
        try:
            comfy_mm.unload_all_models()
            comfy_mm.soft_empty_cache()
        except Exception:
            pass
    gc.collect()
    if TORCH_AVAILABLE and torch is not None:
        if hasattr(torch, "mps") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run_comfy_stage_offloaded_pipeline(
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
    sampler_name: str = "res_multistep",
    scheduler: str = "simple",
    aura_shift: float = 3.0,
) -> Tuple[Any, Dict[str, Any]]:
    """Execute strict 3-stage sequential offloading in ComfyUI (Text -> DiT -> VAE)."""
    if not COMFY_NATIVE_AVAILABLE:
        raise RuntimeError("ComfyUI runtime is not loaded.")

    _free_comfy_memory()

    # Stage 1: Load ONLY Text Encoder (e.g. Qwen 3.4B), encode prompt, then immediately unload
    clip, = comfy_nodes.CLIPLoader().load_clip(clip_name, clip_type)
    positive, = comfy_nodes.CLIPTextEncode().encode(clip, prompt)
    if negative_prompt and negative_prompt.strip() and cfg > 1.0:
        negative, = comfy_nodes.CLIPTextEncode().encode(clip, negative_prompt)
    else:
        negative, = comfy_nodes.ConditioningZeroOut().zero_out(positive)
    del clip
    _free_comfy_memory()

    # Stage 2: Load ONLY Diffusion Transformer (e.g. Z-Image-Turbo 11.74 GB), denoise, then immediately unload
    model, = comfy_nodes.UNETLoader().load_unet(unet_name, "default")
    if aura_shift > 0:
        try:
            from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow  # type: ignore
            active_model, = ModelSamplingAuraFlow().patch_aura(model, aura_shift)
        except Exception:
            active_model = model
    else:
        active_model = model

    latent = {
        "samples": torch.zeros([1, 16, height // 8, width // 8], device="cpu", dtype=torch.float32),
        "downscale_ratio_spacial": 8,
    }
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
    del active_model, model, positive, negative
    _free_comfy_memory()

    # Stage 3: Load ONLY VAE (e.g. ae.safetensors 320 MB), decode to RGB tensor, then immediately unload
    vae, = comfy_nodes.VAELoader().load_vae(vae_name)
    decoded_image, = comfy_nodes.VAEDecode().decode(vae, sampled_latent)
    del vae
    _free_comfy_memory()

    latent_bundle = {
        "samples": sampled_latent["samples"],
        "comfy_latent": sampled_latent,
        "decoded_image": decoded_image,
        "vae_name": vae_name,
        "width": width,
        "height": height,
        "num_frames": 1,
    }
    return decoded_image, latent_bundle


def _to_comfy_image_tensor(rgb_frames: np.ndarray) -> Any:
    """Convert uint8 or float32 (T, H, W, 3) frames into ComfyUI 4D float32 [0.0, 1.0] tensor."""
    if TORCH_AVAILABLE and torch is not None and isinstance(rgb_frames, torch.Tensor):
        return rgb_frames

    if rgb_frames.dtype == np.uint8:
        img_np = rgb_frames.astype(np.float32) / 255.0
    else:
        img_np = np.clip(rgb_frames.astype(np.float32), 0.0, 1.0)

    if img_np.ndim == 3:
        img_np = np.expand_dims(img_np, axis=0)
    elif img_np.ndim == 2:
        img_np = np.expand_dims(img_np, axis=(0, -1))
        img_np = np.repeat(img_np, 3, axis=-1)

    if TORCH_AVAILABLE:
        return torch.from_numpy(img_np)
    return img_np


class PhobosStreamLoaderNode:
    """ComfyUI Node: Loads and initializes zero-VRAM PhobosStream engine & stage offloader."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        unet_list: List[str] = ["z_image_turbo_bf16.safetensors"]
        clip_list: List[str] = ["qwen_3_4b.safetensors"]
        vae_list: List[str] = ["ae.safetensors"]
        if COMFY_NATIVE_AVAILABLE and folder_paths is not None:
            u_found = folder_paths.get_filename_list("diffusion_models")
            c_found = folder_paths.get_filename_list("text_encoders")
            v_found = folder_paths.get_filename_list("vae")
            if u_found:
                unet_list = u_found
            if c_found:
                clip_list = c_found
            if v_found:
                vae_list = v_found

        return {
            "required": {
                "model_name": ("STRING", {"default": "wan2.1-14b"}),
                "architecture": (["wan2.1", "flux", "hunyuan", "cogvideox", "lumina2"], {"default": "wan2.1"}),
                "backend": (["auto", "mlx", "cuda", "gguf", "cpu"], {"default": "auto"}),
                "num_blocks": ("INT", {"default": 30, "min": 1, "max": 64}),
                "hidden_size": ("INT", {"default": 3840, "min": 64, "max": 8192, "step": 64}),
            },
            "optional": {
                "manifest_path": ("STRING", {"default": ""}),
                "unet_name": (unet_list, {"default": unet_list[0]}),
                "clip_name": (clip_list, {"default": clip_list[0]}),
                "clip_type": (["lumina2", "flux", "sd3", "wan", "stable_diffusion"], {"default": "lumina2"}),
                "vae_name": (vae_list, {"default": vae_list[0]}),
                "aura_shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "use_comfy_stage_offload": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("PHOBOS_MODEL",)
    RETURN_NAMES = ("phobos_model",)
    FUNCTION = "load_model"
    CATEGORY = "PhobosStream"

    def load_model(
        self,
        model_name: str,
        architecture: str,
        backend: str,
        num_blocks: int,
        hidden_size: int,
        manifest_path: str = "",
        unet_name: str = "z_image_turbo_bf16.safetensors",
        clip_name: str = "qwen_3_4b.safetensors",
        clip_type: str = "lumina2",
        vae_name: str = "ae.safetensors",
        aura_shift: float = 3.0,
        use_comfy_stage_offload: bool = False,
    ) -> Tuple[Dict[str, Any]]:
        if manifest_path and Path(manifest_path).exists():
            manifest = ModelShardsManifest.load(Path(manifest_path))
        else:
            manifest = ModelShardsManifest(
                model_name=model_name,
                architecture=architecture,
                num_blocks=num_blocks,
                hidden_size=hidden_size,
            )

        engine = PhobosStreamEngine(manifest=manifest, backend_type=BackendType(backend))
        return (
            {
                "engine": engine,
                "manifest": manifest,
                "unet_name": unet_name,
                "clip_name": clip_name,
                "clip_type": clip_type,
                "vae_name": vae_name,
                "aura_shift": aura_shift,
                "use_comfy_stage_offload": use_comfy_stage_offload and COMFY_NATIVE_AVAILABLE,
            },
        )


class PhobosStreamSamplerNode:
    """ComfyUI Node: Executes zero-VRAM stage-offloaded or block-streamed diffusion loop."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "phobos_model": ("PHOBOS_MODEL",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A hyper-detailed futuristic Martian base on red crater landscape, glowing domes, starry galaxy background, 8k photorealistic",
                }),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 2048, "min": 128, "max": 3840, "step": 64}),
                "height": ("INT", {"default": 1152, "min": 128, "max": 2160, "step": 64}),
                "num_frames": ("INT", {"default": 1, "min": 1, "max": 240, "step": 1}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 20.0, "step": 0.5}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "sampler_name": (["res_multistep", "euler", "euler_ancestral", "dpmpp_2m"], {"default": "res_multistep"}),
                "scheduler": (["simple", "normal", "karras", "sgm_uniform"], {"default": "simple"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "PHOBOS_LATENT")
    RETURN_NAMES = ("images", "latents")
    FUNCTION = "sample"
    CATEGORY = "PhobosStream"

    def sample(
        self,
        phobos_model: Optional[Dict[str, Any]] = None,
        prompt: str = "",
        negative_prompt: str = "",
        width: int = 2048,
        height: int = 1152,
        num_frames: int = 1,
        steps: int = 8,
        guidance_scale: float = 1.0,
        seed: int = 42,
        sampler_name: str = "res_multistep",
        scheduler: str = "simple",
        aether_model: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        model_bundle = phobos_model if phobos_model is not None else aether_model
        if model_bundle is None:
            raise ValueError("A valid PHOBOS_MODEL dict must be provided.")

        # If running inside ComfyUI with stage offloading enabled, execute real 3-stage offloaded pipeline
        if model_bundle.get("use_comfy_stage_offload") and COMFY_NATIVE_AVAILABLE:
            decoded_image, latent_bundle = _run_comfy_stage_offloaded_pipeline(
                unet_name=model_bundle.get("unet_name", "z_image_turbo_bf16.safetensors"),
                clip_name=model_bundle.get("clip_name", "qwen_3_4b.safetensors"),
                clip_type=model_bundle.get("clip_type", "lumina2"),
                vae_name=model_bundle.get("vae_name", "ae.safetensors"),
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                cfg=guidance_scale,
                seed=seed,
                sampler_name=sampler_name,
                scheduler=scheduler,
                aura_shift=float(model_bundle.get("aura_shift", 3.0)),
            )
            return (decoded_image, latent_bundle)

        engine: PhobosStreamEngine = model_bundle["engine"]
        config = GenerationConfig(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

        rgb_frames, raw_latents, _ = engine.generate_with_latents(config)
        image_out = _to_comfy_image_tensor(rgb_frames)

        latent_dict = {
            "samples": raw_latents,
            "rgb_frames": rgb_frames,
            "width": width,
            "height": height,
            "num_frames": num_frames,
        }
        return (image_out, latent_dict)


class PhobosStreamVAEDecodeNode:
    """ComfyUI Node: Decodes PhobosStream latents to images/video frames with immediate VAE offload."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "latents": ("PHOBOS_LATENT",),
                "phobos_model": ("PHOBOS_MODEL",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "PhobosStream"

    def decode(
        self,
        latents: Dict[str, Any],
        phobos_model: Optional[Dict[str, Any]] = None,
        aether_model: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any]:
        if "decoded_image" in latents and latents["decoded_image"] is not None:
            return (latents["decoded_image"],)

        if "comfy_latent" in latents and COMFY_NATIVE_AVAILABLE:
            model_bundle = phobos_model if phobos_model is not None else aether_model or {}
            vae_name = latents.get("vae_name") or model_bundle.get("vae_name", "ae.safetensors")
            _free_comfy_memory()
            vae, = comfy_nodes.VAELoader().load_vae(vae_name)
            decoded_image, = comfy_nodes.VAEDecode().decode(vae, latents["comfy_latent"])
            del vae
            _free_comfy_memory()
            return (decoded_image,)

        model_bundle = phobos_model if phobos_model is not None else aether_model
        raw_samples = latents["samples"]

        if (
            model_bundle is not None
            and "engine" in model_bundle
            and hasattr(raw_samples, "ndim")
            and raw_samples.ndim == 3
            and raw_samples.shape[-1] != 3
        ):
            engine: PhobosStreamEngine = model_bundle["engine"]
            latent_tensor = engine.backend.convert_from_numpy(raw_samples)
            rgb_frames, _ = engine.offloader.decode_vae_stage(
                latents=latent_tensor,
                height=latents.get("height", 480),
                width=latents.get("width", 720),
                num_frames=latents.get("num_frames", 1),
            )
        elif "rgb_frames" in latents:
            rgb_frames = latents["rgb_frames"]
        else:
            rgb_frames = raw_samples

        image_out = _to_comfy_image_tensor(rgb_frames)
        return (image_out,)


class PhobosStageOffloadPipelineNode:
    """All-in-One ComfyUI Node: Executes 3-stage offloaded generation (Text -> DiT -> VAE) for 2K/4K."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        unet_list: List[str] = ["z_image_turbo_bf16.safetensors"]
        clip_list: List[str] = ["qwen_3_4b.safetensors"]
        vae_list: List[str] = ["ae.safetensors"]
        if COMFY_NATIVE_AVAILABLE and folder_paths is not None:
            u_found = folder_paths.get_filename_list("diffusion_models")
            c_found = folder_paths.get_filename_list("text_encoders")
            v_found = folder_paths.get_filename_list("vae")
            if u_found:
                unet_list = u_found
            if c_found:
                clip_list = c_found
            if v_found:
                vae_list = v_found

        return {
            "required": {
                "unet_name": (unet_list, {"default": unet_list[0]}),
                "clip_name": (clip_list, {"default": clip_list[0]}),
                "clip_type": (["lumina2", "flux", "sd3", "wan", "stable_diffusion"], {"default": "lumina2"}),
                "vae_name": (vae_list, {"default": vae_list[0]}),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A hyper-detailed futuristic Martian base on red crater landscape, glowing domes, starry galaxy background, 8k photorealistic",
                }),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 2048, "min": 128, "max": 3840, "step": 64}),
                "height": ("INT", {"default": 1152, "min": 128, "max": 2160, "step": 64}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 20.0, "step": 0.5}),
                "aura_shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "sampler_name": (["res_multistep", "euler", "euler_ancestral", "dpmpp_2m"], {"default": "res_multistep"}),
                "scheduler": (["simple", "normal", "karras", "sgm_uniform"], {"default": "simple"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate_offloaded"
    CATEGORY = "PhobosStream"

    def generate_offloaded(
        self,
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
        aura_shift: float,
        sampler_name: str,
        scheduler: str,
        seed: int,
    ) -> Tuple[Any]:
        decoded_image, _ = _run_comfy_stage_offloaded_pipeline(
            unet_name=unet_name,
            clip_name=clip_name,
            clip_type=clip_type,
            vae_name=vae_name,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            seed=seed,
            sampler_name=sampler_name,
            scheduler=scheduler,
            aura_shift=aura_shift,
        )
        return (decoded_image,)

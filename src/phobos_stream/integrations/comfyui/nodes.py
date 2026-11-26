"""ComfyUI custom node definitions for PhobosStream zero-VRAM streaming & stage offloading."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from phobos_stream.integrations.comfyui.comfy_runtime import (
    COMFY_NATIVE_AVAILABLE,
    comfy_nodes,
    decode_comfy_vae_smart,
    free_comfy_memory,
    get_comfy_filenames,
    run_comfy_stage_offloaded_pipeline,
    to_comfy_image_tensor,
    attach_streamer_to_comfy_model,
)
from phobos_stream.pipeline.stream_engine import PhobosStreamEngine
from phobos_stream.storage.layout import ModelShardsManifest
from phobos_stream.types import BackendType, GenerationConfig


CLIP_TYPE_CHOICES = [
    "lumina2",
    "wan",
    "flux",
    "sd3",
    "sdxl",
    "hunyuan_video",
    "cogvideox",
    "ltxv",
    "qwen_image",
    "stable_diffusion",
]


class PhobosStreamLoaderNode:
    """ComfyUI Node: Configures PhobosStream zero-VRAM stage offloader & block-streaming engine."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        unet_list = get_comfy_filenames("diffusion_models", ["z_image_turbo_bf16.safetensors"])
        clip_list = get_comfy_filenames("text_encoders", ["qwen_3_4b.safetensors"])
        clip2_list = ["none"] + clip_list
        vae_list = get_comfy_filenames("vae", ["ae.safetensors"])

        return {
            "required": {
                "model_name": ("STRING", {"default": "wan2.1-14b"}),
                "architecture": (["wan2.1", "flux", "hunyuan", "cogvideox", "lumina2"], {"default": "wan2.1"}),
                "backend": (["auto", "mlx", "cuda", "gguf", "cpu"], {"default": "auto"}),
                "num_blocks": ("INT", {"default": 30, "min": 1, "max": 80}),
                "hidden_size": ("INT", {"default": 3840, "min": 64, "max": 8192, "step": 64}),
            },
            "optional": {
                "manifest_path": ("STRING", {"default": ""}),
                "unet_name": (unet_list, {"default": unet_list[0]}),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {"default": "default"}),
                "clip_name": (clip_list, {"default": clip_list[0]}),
                "clip_name2": (clip2_list, {"default": "none"}),
                "clip_type": (CLIP_TYPE_CHOICES, {"default": "lumina2"}),
                "vae_name": (vae_list, {"default": vae_list[0]}),
                "aura_shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 12.0, "step": 0.1}),
                "block_stream_mode": (["stage_only", "hybrid_budget", "cpu_offload", "mmap_stream"], {"default": "stage_only"}),
                "use_comfy_stage_offload": ("BOOLEAN", {"default": True}),
                "allow_synthetic": ("BOOLEAN", {"default": False}),
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
        weight_dtype: str = "default",
        clip_name: str = "qwen_3_4b.safetensors",
        clip_name2: str = "none",
        clip_type: str = "lumina2",
        vae_name: str = "ae.safetensors",
        aura_shift: float = 3.0,
        block_stream_mode: str = "stage_only",
        use_comfy_stage_offload: bool = True,
        allow_synthetic: bool = False,
    ) -> Tuple[Dict[str, Any]]:
        if manifest_path and Path(manifest_path).exists():
            manifest = ModelShardsManifest.load(Path(manifest_path))
            if allow_synthetic:
                manifest.allow_synthetic = True
        else:
            manifest = ModelShardsManifest(
                model_name=model_name,
                architecture=architecture,
                num_blocks=num_blocks,
                hidden_size=hidden_size,
                allow_synthetic=allow_synthetic,
            )

        stage_offload_active = bool(use_comfy_stage_offload and COMFY_NATIVE_AVAILABLE)
        engine: Optional[PhobosStreamEngine] = None
        if not stage_offload_active:
            engine = PhobosStreamEngine(manifest=manifest, backend_type=BackendType(backend))

        return (
            {
                "engine": engine,
                "manifest": manifest,
                "backend": backend,
                "unet_name": unet_name,
                "weight_dtype": weight_dtype,
                "clip_name": clip_name,
                "clip_name2": "" if clip_name2 == "none" else clip_name2,
                "clip_type": clip_type,
                "vae_name": vae_name,
                "aura_shift": aura_shift,
                "block_stream_mode": block_stream_mode,
                "use_comfy_stage_offload": stage_offload_active,
            },
        )


class PhobosBlockStreamPatcherNode:
    """ComfyUI Node: Attaches per-block CPU-offload or zero-RAM mmap streaming hooks to ANY ComfyUI MODEL."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (["hybrid_budget", "cpu_offload", "mmap_stream"], {"default": "hybrid_budget"}),
            },
            "optional": {
                "vram_budget_mb": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 65536.0, "step": 256.0}),
                "safetensors_path": ("STRING", {"default": ""}),
                "cache_rel_l1_thresh": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cache_warmup_steps": ("INT", {"default": 2, "min": 1, "max": 20}),
                "cache_max_consecutive_skips": ("INT", {"default": 3, "min": 1, "max": 10}),
                "attn_mode": (["auto", "chunked_sdpa", "stock"], {"default": "auto"}),
                "attn_chunk_size": ("INT", {"default": 8192, "min": 1024, "max": 65536, "step": 1024}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch_model"
    CATEGORY = "PhobosStream"

    def patch_model(
        self,
        model: Any,
        mode: str = "hybrid_budget",
        vram_budget_mb: float = 0.0,
        safetensors_path: str = "",
        cache_rel_l1_thresh: float = 0.0,
        cache_warmup_steps: int = 2,
        cache_max_consecutive_skips: int = 3,
        attn_mode: str = "auto",
        attn_chunk_size: int = 8192,
    ) -> Tuple[Any]:
        patched = model.clone() if hasattr(model, "clone") else model
        streamer = attach_streamer_to_comfy_model(
            comfy_model=patched,
            block_stream_mode=mode,
            unet_name=safetensors_path.strip(),
            vram_budget_mb=vram_budget_mb if vram_budget_mb > 0 else None,
            cache_rel_l1_thresh=cache_rel_l1_thresh,
            cache_warmup_steps=cache_warmup_steps,
            cache_max_consecutive_skips=cache_max_consecutive_skips,
            attn_mode=attn_mode,
            attn_chunk_size=attn_chunk_size,
        )
        setattr(patched, "_phobos_block_streamer", streamer)
        return (patched,)


class PhobosStreamSamplerNode:
    """ComfyUI Node: Executes 3-stage offloaded or block-streamed image & video diffusion."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "phobos_model": ("PHOBOS_MODEL",),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "A hyper-detailed futuristic Martian base on red crater landscape, glowing domes, starry galaxy background, 8k photorealistic",
                    },
                ),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 1280, "min": 128, "max": 3840, "step": 16}),
                "height": ("INT", {"default": 720, "min": 128, "max": 2160, "step": 16}),
                "num_frames": ("INT", {"default": 1, "min": 1, "max": 241, "step": 1}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 20.0, "step": 0.5}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "sampler_name": (["res_multistep", "euler", "euler_ancestral", "uni_pc", "dpmpp_2m"], {"default": "res_multistep"}),
                "scheduler": (["simple", "normal", "karras", "sgm_uniform", "beta"], {"default": "simple"}),
                "use_tiled_vae": ("BOOLEAN", {"default": True}),
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
        width: int = 1280,
        height: int = 720,
        num_frames: int = 1,
        steps: int = 8,
        guidance_scale: float = 1.0,
        seed: int = 42,
        sampler_name: str = "res_multistep",
        scheduler: str = "simple",
        use_tiled_vae: bool = True,
        aether_model: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        model_bundle = phobos_model if phobos_model is not None else aether_model
        if model_bundle is None:
            raise ValueError("A valid PHOBOS_MODEL dict must be provided.")

        if model_bundle.get("use_comfy_stage_offload") and COMFY_NATIVE_AVAILABLE:
            decoded_image, latent_bundle = run_comfy_stage_offloaded_pipeline(
                unet_name=model_bundle.get("unet_name", "z_image_turbo_bf16.safetensors"),
                clip_name=model_bundle.get("clip_name", "qwen_3_4b.safetensors"),
                clip_type=model_bundle.get("clip_type", "lumina2"),
                vae_name=model_bundle.get("vae_name", "ae.safetensors"),
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                steps=steps,
                cfg=guidance_scale,
                seed=seed,
                sampler_name=sampler_name,
                scheduler=scheduler,
                aura_shift=float(model_bundle.get("aura_shift", 3.0)),
                clip_name2=model_bundle.get("clip_name2", ""),
                weight_dtype=model_bundle.get("weight_dtype", "default"),
                block_stream_mode=model_bundle.get("block_stream_mode", "stage_only"),
                use_tiled_vae=use_tiled_vae,
            )
            return (decoded_image, latent_bundle)

        engine: Optional[PhobosStreamEngine] = model_bundle.get("engine")
        if engine is None:
            manifest = model_bundle["manifest"]
            backend = model_bundle.get("backend", "auto")
            engine = PhobosStreamEngine(manifest=manifest, backend_type=BackendType(backend))
            model_bundle["engine"] = engine

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
        image_out = to_comfy_image_tensor(rgb_frames)
        latent_dict = {
            "samples": raw_latents,
            "rgb_frames": rgb_frames,
            "width": width,
            "height": height,
            "num_frames": num_frames,
        }
        return (image_out, latent_dict)


class PhobosStreamVAEDecodeNode:
    """ComfyUI Node: Decodes PhobosStream image/video latents with tiled VAE & immediate offload."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "latents": ("PHOBOS_LATENT",),
                "phobos_model": ("PHOBOS_MODEL",),
            },
            "optional": {
                "use_tiled_vae": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "PhobosStream"

    def decode(
        self,
        latents: Dict[str, Any],
        phobos_model: Optional[Dict[str, Any]] = None,
        use_tiled_vae: bool = True,
        aether_model: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any]:
        if "decoded_image" in latents and latents["decoded_image"] is not None:
            return (latents["decoded_image"],)

        if "comfy_latent" in latents and COMFY_NATIVE_AVAILABLE and comfy_nodes is not None:
            model_bundle = phobos_model if phobos_model is not None else aether_model or {}
            vae_name = latents.get("vae_name") or model_bundle.get("vae_name", "ae.safetensors")
            free_comfy_memory()
            vae, = comfy_nodes.VAELoader().load_vae(vae_name)
            try:
                decoded_image = decode_comfy_vae_smart(vae, latents["comfy_latent"], use_tiled_vae=use_tiled_vae)
            finally:
                del vae
                free_comfy_memory()
            return (decoded_image,)

        model_bundle = phobos_model if phobos_model is not None else aether_model
        raw_samples = latents["samples"]
        if (
            model_bundle is not None
            and model_bundle.get("engine") is not None
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

        return (to_comfy_image_tensor(rgb_frames),)


class PhobosStageOffloadPipelineNode:
    """All-in-One ComfyUI Node: 3-stage sequential offloading (`Text -> DiT -> VAE`) for 2K/4K & Video."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        unet_list = get_comfy_filenames("diffusion_models", ["z_image_turbo_bf16.safetensors"])
        clip_list = get_comfy_filenames("text_encoders", ["qwen_3_4b.safetensors"])
        clip2_list = ["none"] + clip_list
        vae_list = get_comfy_filenames("vae", ["ae.safetensors"])

        return {
            "required": {
                "unet_name": (unet_list, {"default": unet_list[0]}),
                "clip_name": (clip_list, {"default": clip_list[0]}),
                "clip_type": (CLIP_TYPE_CHOICES, {"default": "lumina2"}),
                "vae_name": (vae_list, {"default": vae_list[0]}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "A hyper-detailed futuristic Martian base on red crater landscape, glowing domes, starry galaxy background, 8k photorealistic",
                    },
                ),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 2048, "min": 128, "max": 3840, "step": 16}),
                "height": ("INT", {"default": 1152, "min": 128, "max": 2160, "step": 16}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 20.0, "step": 0.5}),
                "aura_shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 12.0, "step": 0.1}),
                "sampler_name": (["res_multistep", "euler", "euler_ancestral", "uni_pc", "dpmpp_2m"], {"default": "res_multistep"}),
                "scheduler": (["simple", "normal", "karras", "sgm_uniform", "beta"], {"default": "simple"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "num_frames": ("INT", {"default": 1, "min": 1, "max": 241, "step": 1}),
                "clip_name2": (clip2_list, {"default": "none"}),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {"default": "default"}),
                "block_stream_mode": (["stage_only", "hybrid_budget", "cpu_offload", "mmap_stream"], {"default": "stage_only"}),
                "use_tiled_vae": ("BOOLEAN", {"default": True}),
            },
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
        num_frames: int = 1,
        clip_name2: str = "none",
        weight_dtype: str = "default",
        block_stream_mode: str = "stage_only",
        use_tiled_vae: bool = True,
    ) -> Tuple[Any]:
        decoded_image, _ = run_comfy_stage_offloaded_pipeline(
            unet_name=unet_name,
            clip_name=clip_name,
            clip_type=clip_type,
            vae_name=vae_name,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            steps=steps,
            cfg=cfg,
            seed=seed,
            sampler_name=sampler_name,
            scheduler=scheduler,
            aura_shift=aura_shift,
            clip_name2="" if clip_name2 == "none" else clip_name2,
            weight_dtype=weight_dtype,
            block_stream_mode=block_stream_mode,
            use_tiled_vae=use_tiled_vae,
        )
        return (decoded_image,)

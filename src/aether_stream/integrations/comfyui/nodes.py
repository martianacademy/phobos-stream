"""ComfyUI custom node implementations for AetherStream zero-VRAM streaming."""

from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np

from aether_stream.pipeline.stream_engine import AetherStreamEngine
from aether_stream.storage.layout import ModelShardsManifest
from aether_stream.types import BackendType, GenerationConfig

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


class AetherStreamLoaderNode:
    """ComfyUI Node: Loads and initializes zero-VRAM AetherStream engine."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model_name": ("STRING", {"default": "wan2.1-14b"}),
                "architecture": (["wan2.1", "flux", "hunyuan", "cogvideox"], {"default": "wan2.1"}),
                "backend": (["auto", "mlx", "cuda", "gguf", "cpu"], {"default": "auto"}),
                "num_blocks": ("INT", {"default": 40, "min": 1, "max": 64}),
                "hidden_size": ("INT", {"default": 4096, "min": 64, "max": 8192, "step": 64}),
            },
            "optional": {
                "manifest_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("AETHER_MODEL",)
    RETURN_NAMES = ("aether_model",)
    FUNCTION = "load_model"
    CATEGORY = "AetherStream"

    def load_model(
        self,
        model_name: str,
        architecture: str,
        backend: str,
        num_blocks: int,
        hidden_size: int,
        manifest_path: str = "",
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

        engine = AetherStreamEngine(manifest=manifest, backend_type=BackendType(backend))
        return ({"engine": engine, "manifest": manifest},)


class AetherStreamSamplerNode:
    """ComfyUI Node: Executes zero-VRAM diffusion denoising loop."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "aether_model": ("AETHER_MODEL",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Cinematic shot of an ancient mountain temple in morning mist, 8k photorealistic",
                }),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 720, "min": 128, "max": 3840, "step": 64}),
                "height": ("INT", {"default": 480, "min": 128, "max": 2160, "step": 64}),
                "num_frames": ("INT", {"default": 25, "min": 1, "max": 240, "step": 4}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 20.0, "step": 0.5}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AETHER_LATENT")
    RETURN_NAMES = ("images", "latents")
    FUNCTION = "sample"
    CATEGORY = "AetherStream"

    def sample(
        self,
        aether_model: Dict[str, Any],
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_frames: int,
        steps: int,
        guidance_scale: float,
        seed: int,
    ) -> Tuple[Any, Dict[str, Any]]:
        engine: AetherStreamEngine = aether_model["engine"]
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

        rgb_frames, metrics = engine.generate(config)

        # Normalize uint8 (T, H, W, 3) to ComfyUI standard float32 [0.0, 1.0]
        # In ComfyUI, IMAGE tensor is (Batch, H, W, C)
        img_np = (rgb_frames.astype(np.float32) / 255.0)

        # Reshape to (Batch, H, W, C)
        if img_np.ndim == 3:
            img_np = np.expand_dims(img_np, axis=0)
        elif img_np.ndim == 2:
            img_np = np.expand_dims(img_np, axis=(0, -1))
            img_np = np.repeat(img_np, 3, axis=-1)

        # Convert to Torch tensor if torch is available (native ComfyUI format)
        if TORCH_AVAILABLE:
            image_out = torch.from_numpy(img_np)
        else:
            image_out = img_np

        latent_dict = {
            "samples": rgb_frames,
            "width": width,
            "height": height,
            "num_frames": num_frames,
        }
        return (image_out, latent_dict)


class AetherStreamVAEDecodeNode:
    """ComfyUI Node: Decodes AetherStream latents to images/video frames."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "latents": ("AETHER_LATENT",),
                "aether_model": ("AETHER_MODEL",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "AetherStream"

    def decode(self, latents: Dict[str, Any], aether_model: Dict[str, Any]) -> Tuple[Any]:
        raw_samples = latents["samples"]
        img_np = (raw_samples.astype(np.float32) / 255.0)
        if img_np.ndim == 3:
            img_np = np.expand_dims(img_np, axis=0)

        if TORCH_AVAILABLE:
            image_out = torch.from_numpy(img_np)
        else:
            image_out = img_np

        return (image_out,)

"""ComfyUI integration node mappings for AetherStream."""

from aether_stream.integrations.comfyui.nodes import (
    AetherStreamLoaderNode,
    AetherStreamSamplerNode,
    AetherStreamVAEDecodeNode,
)

NODE_CLASS_MAPPINGS = {
    "AetherStreamLoader": AetherStreamLoaderNode,
    "AetherStreamSampler": AetherStreamSamplerNode,
    "AetherStreamVAEDecode": AetherStreamVAEDecodeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AetherStreamLoader": "🌌 AetherStream Model Loader (Zero-VRAM)",
    "AetherStreamSampler": "⚡ AetherStream Diffusion Sampler (Disk-Stream)",
    "AetherStreamVAEDecode": "🎞️ AetherStream VAE Decoder",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

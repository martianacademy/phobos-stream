"""ComfyUI integration node mappings for PhobosStream."""

from phobos_stream.integrations.comfyui.nodes import (
    PhobosStageOffloadPipelineNode,
    PhobosStreamLoaderNode,
    PhobosStreamSamplerNode,
    PhobosStreamVAEDecodeNode,
)

NODE_CLASS_MAPPINGS = {
    "PhobosStreamLoader": PhobosStreamLoaderNode,
    "PhobosStreamSampler": PhobosStreamSamplerNode,
    "PhobosStreamVAEDecode": PhobosStreamVAEDecodeNode,
    "PhobosStageOffloadPipeline": PhobosStageOffloadPipelineNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhobosStreamLoader": "🌌 PhobosStream Model Loader (Zero-VRAM)",
    "PhobosStreamSampler": "⚡ PhobosStream Diffusion Sampler (Stage-Offload)",
    "PhobosStreamVAEDecode": "🎞️ PhobosStream VAE Decoder",
    "PhobosStageOffloadPipeline": "🚀 Phobos Zero-VRAM 2K/4K Pipeline (Stage-Offload)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

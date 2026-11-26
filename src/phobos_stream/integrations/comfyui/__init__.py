"""ComfyUI Node Mappings for PhobosStream."""

from phobos_stream.integrations.comfyui.cache_node import PhobosFirstBlockCacheNode
from phobos_stream.integrations.comfyui.nodes import (
    PhobosBlockStreamPatcherNode,
    PhobosStageOffloadPipelineNode,
    PhobosStreamLoaderNode,
    PhobosStreamSamplerNode,
    PhobosStreamVAEDecodeNode,
)
from phobos_stream.integrations.comfyui.vae_patcher import PhobosVAEStreamPatcherNode

NODE_CLASS_MAPPINGS = {
    "PhobosStreamLoader": PhobosStreamLoaderNode,
    "PhobosBlockStreamPatcher": PhobosBlockStreamPatcherNode,
    "PhobosFirstBlockCache": PhobosFirstBlockCacheNode,
    "PhobosVAEStreamPatcher": PhobosVAEStreamPatcherNode,
    "PhobosStreamSampler": PhobosStreamSamplerNode,
    "PhobosStreamVAEDecode": PhobosStreamVAEDecodeNode,
    "PhobosStageOffloadPipeline": PhobosStageOffloadPipelineNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhobosStreamLoader": "🌌 PhobosStream Model Loader (Zero-VRAM)",
    "PhobosBlockStreamPatcher": "🔌 PhobosStream Block-Streamer (Any MODEL)",
    "PhobosFirstBlockCache": "⚡ PhobosStream TeaCache / First-Block Cache",
    "PhobosVAEStreamPatcher": "🎬 PhobosStream VAE Streamer (Video & Audio VAE)",
    "PhobosStreamSampler": "⚡ PhobosStream Diffusion Sampler (Stage-Offload)",
    "PhobosStreamVAEDecode": "🎞️ PhobosStream Tiled VAE Decoder",
    "PhobosStageOffloadPipeline": "🚀 Phobos Zero-VRAM 2K/4K & Video Pipeline",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "PhobosBlockStreamPatcherNode",
    "PhobosFirstBlockCacheNode",
    "PhobosStageOffloadPipelineNode",
    "PhobosStreamLoaderNode",
    "PhobosStreamSamplerNode",
    "PhobosStreamVAEDecodeNode",
    "PhobosVAEStreamPatcherNode",
]

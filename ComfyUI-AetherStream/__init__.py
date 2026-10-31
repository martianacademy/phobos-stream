"""ComfyUI-AetherStream: Zero-VRAM Video & Image Diffusion Streaming Extension.

Symlink or copy this directory directly into ComfyUI/custom_nodes/
"""

import sys
from pathlib import Path

# Add aether_stream source path to sys.path if not installed globally
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from aether_stream.integrations.comfyui import (
        NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS,
    )
except ImportError:
    # Fallback to local import if inside standalone node folder
    from nodes import (
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

"""PhobosStream: Universal Zero-VRAM Video & Image Diffusion Streaming Engine.

Supports Apple Silicon (MLX), NVIDIA CUDA, AMD ROCm, Vulkan, and GGUF quantization.
"""

__version__ = "0.1.0"
__author__ = "Martian Academy"

from phobos_stream.types import (
    BackendType,
    GenerationConfig,
    ModelArchitecture,
    PipelineMetrics,
    QuantPrecision,
)

__all__ = [
    "BackendType",
    "GenerationConfig",
    "ModelArchitecture",
    "PipelineMetrics",
    "QuantPrecision",
    "__version__",
]

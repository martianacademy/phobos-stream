"""PhobosStream: Universal Zero-VRAM Video & Image Diffusion Streaming Engine.

Supports Apple Silicon (MLX), NVIDIA CUDA, AMD ROCm, Vulkan, and GGUF quantization.
"""

__version__ = "1.1.0"
__author__ = "Martian Academy"

from phobos_stream.api import stream, unstream
from phobos_stream.storage.gguf_mmap_loader import GGUFMMapLoader
from phobos_stream.storage.mmap_loader import MMapTensorLoader
from phobos_stream.streaming import (
    BlockStreamMode,
    FirstBlockResidualCache,
    PhobosAttentionFastpath,
    TorchBlockStreamer,
)
from phobos_stream.types import (
    BackendType,
    GenerationConfig,
    ModelArchitecture,
    PipelineMetrics,
    QuantPrecision,
)

__all__ = [
    "BackendType",
    "BlockStreamMode",
    "FirstBlockResidualCache",
    "GGUFMMapLoader",
    "GenerationConfig",
    "MMapTensorLoader",
    "ModelArchitecture",
    "PhobosAttentionFastpath",
    "PipelineMetrics",
    "QuantPrecision",
    "TorchBlockStreamer",
    "__version__",
    "stream",
    "unstream",
]

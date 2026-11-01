"""Model architecture adapters mapping safetensors weights to universal DiT blocks."""

from phobos_stream.adapters.base import BaseModelAdapter
from phobos_stream.adapters.cogvideo_adapter import CogVideoXAdapter
from phobos_stream.adapters.factory import get_adapter
from phobos_stream.adapters.flux_adapter import FluxAdapter
from phobos_stream.adapters.hunyuan_adapter import HunyuanAdapter
from phobos_stream.adapters.wan_adapter import WanAdapter

__all__ = [
    "BaseModelAdapter",
    "CogVideoXAdapter",
    "FluxAdapter",
    "HunyuanAdapter",
    "WanAdapter",
    "get_adapter",
]

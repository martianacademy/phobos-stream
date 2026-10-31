"""Architecture adapters for Wan2.1, FLUX.1, and HunyuanVideo."""

from aether_stream.adapters.base import BaseModelAdapter
from aether_stream.adapters.factory import get_adapter
from aether_stream.adapters.flux_adapter import FluxAdapter
from aether_stream.adapters.hunyuan_adapter import HunyuanAdapter
from aether_stream.adapters.wan_adapter import WanAdapter

__all__ = [
    "BaseModelAdapter",
    "FluxAdapter",
    "HunyuanAdapter",
    "WanAdapter",
    "get_adapter",
]

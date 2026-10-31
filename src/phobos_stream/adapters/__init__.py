"""Architecture adapters for Wan2.1, FLUX.1, and HunyuanVideo."""

from phobos_stream.adapters.base import BaseModelAdapter
from phobos_stream.adapters.factory import get_adapter
from phobos_stream.adapters.flux_adapter import FluxAdapter
from phobos_stream.adapters.hunyuan_adapter import HunyuanAdapter
from phobos_stream.adapters.wan_adapter import WanAdapter

__all__ = [
    "BaseModelAdapter",
    "FluxAdapter",
    "HunyuanAdapter",
    "WanAdapter",
    "get_adapter",
]

"""Legacy v0.1 synthetic shard benchmark adapters (isolated to `allow_synthetic=True` tests).

For production models, use `phobos_stream.stream(model)` (`TorchBlockStreamer`),
`HFDiffusersStreamEngine`, ComfyUI nodes, or `MLXWanVideoStreamer` / `MLXBlockStreamer`.
"""

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

"""Adapter registry and auto-detection factory."""

from phobos_stream.adapters.base import BaseModelAdapter
from phobos_stream.adapters.cogvideo_adapter import CogVideoXAdapter
from phobos_stream.adapters.flux_adapter import FluxAdapter
from phobos_stream.adapters.hunyuan_adapter import HunyuanAdapter
from phobos_stream.adapters.wan_adapter import WanAdapter
from phobos_stream.backends.base import BaseBackend


def get_adapter(
    architecture: str,
    backend: BaseBackend,
    hidden_size: int = 4096,
) -> BaseModelAdapter:
    """Resolve and return model adapter for the given architecture."""
    arch = architecture.lower()
    if "wan" in arch:
        return WanAdapter(backend, hidden_size=hidden_size)
    if "flux" in arch:
        return FluxAdapter(backend, hidden_size=hidden_size)
    if "hunyuan" in arch:
        return HunyuanAdapter(backend, hidden_size=hidden_size)
    if "cogvideo" in arch:
        return CogVideoXAdapter(backend, hidden_size=hidden_size)
    return WanAdapter(backend, hidden_size=hidden_size)

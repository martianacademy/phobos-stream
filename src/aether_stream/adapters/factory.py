"""Adapter registry and auto-detection factory."""

from aether_stream.adapters.base import BaseModelAdapter
from aether_stream.adapters.flux_adapter import FluxAdapter
from aether_stream.adapters.hunyuan_adapter import HunyuanAdapter
from aether_stream.adapters.wan_adapter import WanAdapter
from aether_stream.backends.base import BaseBackend


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
    # Default to Wan2.1 adapter
    return WanAdapter(backend, hidden_size=hidden_size)

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
    allow_synthetic: bool = False,
) -> BaseModelAdapter:
    """Resolve and return model adapter for the given architecture."""
    arch = architecture.lower()
    if "wan" in arch or "lumina" in arch or "z_image" in arch or "z-image" in arch:
        return WanAdapter(backend, hidden_size=hidden_size, allow_synthetic=allow_synthetic)
    if "flux" in arch:
        return FluxAdapter(backend, hidden_size=hidden_size, allow_synthetic=allow_synthetic)
    if "hunyuan" in arch:
        return HunyuanAdapter(backend, hidden_size=hidden_size, allow_synthetic=allow_synthetic)
    if "cogvideo" in arch:
        return CogVideoXAdapter(backend, hidden_size=hidden_size, allow_synthetic=allow_synthetic)
    return WanAdapter(backend, hidden_size=hidden_size, allow_synthetic=allow_synthetic)

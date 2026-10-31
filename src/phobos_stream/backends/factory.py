"""Factory to dynamically instantiate the best available compute backend."""

import platform
from phobos_stream.backends.base import BaseBackend
from phobos_stream.backends.gguf_backend import GGUFBackend
from phobos_stream.types import BackendType


def get_backend(backend_type: BackendType = BackendType.AUTO, device: str = "auto") -> BaseBackend:
    """Instantiate and return the appropriate execution backend."""
    if backend_type == BackendType.MLX:
        from phobos_stream.backends.mlx_backend import MLXBackend
        return MLXBackend(device="gpu" if device == "auto" else device)

    if backend_type in (BackendType.CUDA, BackendType.ROCM):
        from phobos_stream.backends.cuda_backend import CUDABackend
        return CUDABackend(device="cuda:0" if device == "auto" else device)

    if backend_type in (BackendType.GGUF, BackendType.CPU):
        return GGUFBackend(device="cpu" if device == "auto" else device)

    # AUTO Detection
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            from phobos_stream.backends.mlx_backend import MLX_AVAILABLE, MLXBackend
            if MLX_AVAILABLE:
                return MLXBackend(device="gpu")
        except Exception:
            pass

    try:
        import torch
        if torch.cuda.is_available():
            from phobos_stream.backends.cuda_backend import CUDABackend
            return CUDABackend(device="cuda:0")
    except Exception:
        pass

    return GGUFBackend(device="cpu")

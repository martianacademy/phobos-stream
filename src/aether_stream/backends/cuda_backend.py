"""NVIDIA CUDA and AMD ROCm PyTorch execution backend."""

from typing import Any, Optional, Tuple
import numpy as np
from aether_stream.backends.base import BaseBackend

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    F = None


class CUDABackend(BaseBackend):
    """PyTorch CUDA and ROCm GPU backend."""

    def __init__(self, device: str = "cuda:0") -> None:
        super().__init__(device=device)
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed. Run: pip install torch")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this machine.")
        self.torch_device = torch.device(device)

    @property
    def name(self) -> str:
        return "cuda"

    def zeros(self, shape: Tuple[int, ...], dtype: str = "float32") -> Any:
        torch_dtype = getattr(torch, dtype, torch.float32)
        return torch.zeros(shape, dtype=torch_dtype, device=self.torch_device)

    def randn(self, shape: Tuple[int, ...], seed: Optional[int] = None, dtype: str = "float32") -> Any:
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.torch_device).manual_seed(seed)
        torch_dtype = getattr(torch, dtype, torch.float32)
        return torch.randn(shape, generator=generator, dtype=torch_dtype, device=self.torch_device)

    def matmul(self, a: Any, b: Any) -> Any:
        return torch.matmul(a, b)

    def layer_norm(self, x: Any, weight: Any, bias: Optional[Any] = None, eps: float = 1e-5) -> Any:
        return F.layer_norm(x, normalized_shape=(x.shape[-1],), weight=weight, bias=bias, eps=eps)

    def silu(self, x: Any) -> Any:
        return F.silu(x)

    def scaled_dot_product_attention(
        self,
        q: Any,
        k: Any,
        v: Any,
        scale: Optional[float] = None,
        mask: Optional[Any] = None,
    ) -> Any:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)

    def convert_from_numpy(self, arr: np.ndarray, dtype: Optional[str] = None) -> Any:
        t = torch.from_numpy(arr).to(self.torch_device)
        if dtype is not None and hasattr(torch, dtype):
            t = t.to(getattr(torch, dtype))
        return t

    def to_numpy(self, tensor: Any) -> np.ndarray:
        return tensor.detach().cpu().numpy()

    def synchronize(self) -> None:
        torch.cuda.synchronize(self.torch_device)

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def get_memory_allocated_mb(self) -> float:
        return float(torch.cuda.memory_allocated(self.torch_device)) / (1024.0 * 1024.0)

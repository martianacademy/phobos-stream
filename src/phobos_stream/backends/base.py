"""Abstract base interface for execution backends (MLX, CUDA, ROCm, GGUF, CPU)."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple
import numpy as np


class BaseBackend(ABC):
    """Universal compute backend interface."""

    def __init__(self, device: str = "auto") -> None:
        self.device = device

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the backend."""
        pass

    @abstractmethod
    def zeros(self, shape: Tuple[int, ...], dtype: str = "float32") -> Any:
        """Create a zero tensor."""
        pass

    @abstractmethod
    def randn(self, shape: Tuple[int, ...], seed: Optional[int] = None, dtype: str = "float32") -> Any:
        """Create a normally distributed random tensor."""
        pass

    @abstractmethod
    def matmul(self, a: Any, b: Any) -> Any:
        """Perform matrix multiplication a @ b."""
        pass

    def linear(self, x: Any, weight: Any, bias: Optional[Any] = None) -> Any:
        """Apply linear projection handling both (out, in) and (in, out) weight layouts."""
        w_shape = weight.shape
        x_last = x.shape[-1]
        if len(w_shape) == 2:
            # Standard PyTorch/Safetensors linear weight is (out_features, in_features).
            # If weight is (in_features, out_features) and non-square, use directly;
            # otherwise transpose last two axes.
            if w_shape[0] == x_last and w_shape[1] != x_last:
                out = self.matmul(x, weight)
            else:
                out = self.matmul(x, self.swapaxes(weight, -1, -2))
        else:
            out = self.matmul(x, weight)

        if bias is not None:
            out = out + bias
        return out

    def reshape(self, x: Any, shape: Tuple[int, ...]) -> Any:
        """Reshape a tensor."""
        return x.reshape(*shape) if hasattr(x, "reshape") else np.reshape(x, shape)

    def swapaxes(self, x: Any, axis1: int, axis2: int) -> Any:
        """Swap two axes of a tensor."""
        if hasattr(x, "swapaxes"):
            return x.swapaxes(axis1, axis2)
        return np.swapaxes(x, axis1, axis2)

    @abstractmethod
    def layer_norm(self, x: Any, weight: Any, bias: Optional[Any] = None, eps: float = 1e-5) -> Any:
        """Perform LayerNorm or RMSNorm."""
        pass

    @abstractmethod
    def silu(self, x: Any) -> Any:
        """SiLU (Swish) activation function."""
        pass

    @abstractmethod
    def scaled_dot_product_attention(
        self,
        q: Any,
        k: Any,
        v: Any,
        scale: Optional[float] = None,
        mask: Optional[Any] = None,
    ) -> Any:
        """Compute multi-head scaled dot product attention."""
        pass

    @abstractmethod
    def convert_from_numpy(self, arr: np.ndarray, dtype: Optional[str] = None) -> Any:
        """Convert a numpy array to the backend's native tensor format."""
        pass

    @abstractmethod
    def to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert a backend native tensor to numpy."""
        pass

    @abstractmethod
    def synchronize(self) -> None:
        """Wait for all pending GPU/Metal work to finish."""
        pass

    @abstractmethod
    def empty_cache(self) -> None:
        """Reclaim unused memory caches."""
        pass

    @abstractmethod
    def get_memory_allocated_mb(self) -> float:
        """Current active allocated memory in Megabytes."""
        pass

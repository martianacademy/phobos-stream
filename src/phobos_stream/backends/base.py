"""Abstract base interface for execution backends (MLX, CUDA, ROCm, GGUF, CPU)."""

from abc import ABC, abstractmethod
import contextlib
from typing import Any, Iterator, Optional, Sequence, Tuple
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

    def concat(self, tensors: Sequence[Any], axis: int = 0) -> Any:
        """Concatenate tensors along `axis` on their native backend."""
        if not tensors:
            raise ValueError("concat requires at least one tensor.")
        mod_name = type(tensors[0]).__module__
        if "mlx" in mod_name:
            import mlx.core as mx

            return mx.concatenate(list(tensors), axis=axis)
        if "torch" in mod_name:
            import torch

            return torch.cat(list(tensors), dim=axis)
        return np.concatenate(list(tensors), axis=axis)

    def split_two(self, tensor: Any, axis: int = 0) -> Tuple[Any, Any]:
        """Split an even-sized batch tensor into `(first_half, second_half)` along axis 0."""
        if axis != 0:
            raise ValueError("split_two expects axis=0 for batched conditioning.")
        half = tensor.shape[0] // 2
        return tensor[:half], tensor[half:]

    def async_eval(self, *tensors: Any) -> None:
        """Schedule asynchronous evaluation of tensors (mlx_lm style pipeline)."""
        pass

    @contextlib.contextmanager
    def wired_limit(self) -> Iterator[None]:
        """Context manager to lock recommended GPU working set size in unified memory."""
        yield

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

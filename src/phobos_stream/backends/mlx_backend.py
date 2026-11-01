"""Apple Silicon Metal (MLX) backend implementation."""

from typing import Any, Optional, Tuple
import numpy as np
from phobos_stream.backends.base import BaseBackend

try:
    import mlx.core as mx
    import mlx.core.fast as mxf
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None
    mxf = None


class MLXBackend(BaseBackend):
    """High-performance Apple Silicon unified memory backend."""

    def __init__(self, device: str = "gpu") -> None:
        super().__init__(device=device)
        if not MLX_AVAILABLE:
            raise RuntimeError("MLX is not installed or platform is not Apple Silicon.")

    @property
    def name(self) -> str:
        return "mlx"

    def zeros(self, shape: Tuple[int, ...], dtype: str = "float32") -> Any:
        mlx_dtype = getattr(mx, dtype, mx.float32)
        return mx.zeros(shape, dtype=mlx_dtype)

    def randn(self, shape: Tuple[int, ...], seed: Optional[int] = None, dtype: str = "float32") -> Any:
        if seed is not None:
            mx.random.seed(seed)
        mlx_dtype = getattr(mx, dtype, mx.float32)
        return mx.random.normal(shape=shape, dtype=mlx_dtype)

    def matmul(self, a: Any, b: Any) -> Any:
        return mx.matmul(a, b)

    def reshape(self, x: Any, shape: Tuple[int, ...]) -> Any:
        return mx.reshape(x, shape)

    def swapaxes(self, x: Any, axis1: int, axis2: int) -> Any:
        return mx.swapaxes(x, axis1, axis2)

    def layer_norm(self, x: Any, weight: Any, bias: Optional[Any] = None, eps: float = 1e-5) -> Any:
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        normed = (x - mean) / mx.sqrt(var + eps)
        out = normed * weight
        if bias is not None:
            out = out + bias
        return out

    def silu(self, x: Any) -> Any:
        return x * mx.sigmoid(x)

    def scaled_dot_product_attention(
        self,
        q: Any,
        k: Any,
        v: Any,
        scale: Optional[float] = None,
        mask: Optional[Any] = None,
    ) -> Any:
        scale_val = scale if scale is not None else 1.0 / (q.shape[-1] ** 0.5)
        if q.ndim == 4 and hasattr(mxf, "scaled_dot_product_attention"):
            try:
                return mxf.scaled_dot_product_attention(q, k, v, scale=scale_val, mask=mask)
            except Exception:
                pass
        scores = mx.matmul(q, mx.swapaxes(k, -1, -2)) * scale_val
        if mask is not None:
            scores = scores + mask
        weights = mx.softmax(scores, axis=-1)
        return mx.matmul(weights, v)

    def convert_from_numpy(self, arr: np.ndarray, dtype: Optional[str] = None) -> Any:
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        mlx_arr = mx.array(arr)
        if dtype is not None and hasattr(mx, dtype):
            mlx_arr = mlx_arr.astype(getattr(mx, dtype))
        return mlx_arr

    def to_numpy(self, tensor: Any) -> np.ndarray:
        mx.eval(tensor)
        return np.array(tensor)

    def synchronize(self) -> None:
        mx.eval()

    def empty_cache(self) -> None:
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()

    def get_memory_allocated_mb(self) -> float:
        if hasattr(mx, "get_active_memory"):
            return float(mx.get_active_memory()) / (1024.0 * 1024.0)
        elif hasattr(mx, "metal") and hasattr(mx.metal, "get_active_memory"):
            return float(mx.metal.get_active_memory()) / (1024.0 * 1024.0)
        return 0.0

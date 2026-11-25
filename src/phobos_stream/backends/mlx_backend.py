"""Apple Silicon Metal (MLX) backend with mlx_lm optimizations (wired_limit, async_eval, quantized_matmul)."""

import contextlib
from typing import Any, Iterator, Optional, Tuple
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
    """High-performance Apple Silicon unified memory backend incorporating mlx_lm primitives."""

    def __init__(
        self,
        device: str = "gpu",
        quantize_bits: Optional[int] = None,
        quantize_group_size: int = 64,
    ) -> None:
        super().__init__(device=device)
        if not MLX_AVAILABLE:
            raise RuntimeError("MLX is not installed or platform is not Apple Silicon.")
        self.quantize_bits = quantize_bits
        self.quantize_group_size = quantize_group_size
        self.generation_stream = (
            mx.new_thread_local_stream(mx.default_device())
            if hasattr(mx, "new_thread_local_stream")
            else mx.default_stream(mx.default_device())
        )

    @property
    def name(self) -> str:
        return "mlx"

    @contextlib.contextmanager
    def wired_limit(self) -> Iterator[None]:
        """Pin Metal working set size using mlx_lm's maybe_set_recommended_wired_limit strategy."""
        old_limit = None
        if hasattr(mx, "device_info") and hasattr(mx, "set_wired_limit"):
            try:
                max_rec = mx.device_info().get("max_recommended_working_set_size")
                if max_rec is not None:
                    old_limit = mx.set_wired_limit(max_rec)
            except Exception:
                old_limit = None
        try:
            yield
        finally:
            if old_limit is not None:
                try:
                    mx.synchronize(self.generation_stream)
                    mx.set_wired_limit(old_limit)
                except Exception:
                    pass

    def async_eval(self, *tensors: Any) -> None:
        """Asynchronously evaluate MLX computation graph without blocking CPU (mlx_lm style)."""
        if hasattr(mx, "async_eval"):
            mx.async_eval(*tensors)
        else:
            mx.eval(*tensors)

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

    def quantize_weight(self, weight: Any, bits: int = 8, group_size: int = 64) -> Tuple[Any, Any, Any]:
        """Pack a 2D weight matrix (out_features, in_features) into MLX quantized format."""
        q_res = mx.quantize(weight, group_size=group_size, bits=bits)
        return (q_res[0], q_res[1], q_res[2])

    def linear(self, x: Any, weight: Any, bias: Optional[Any] = None) -> Any:
        """Execute linear layer supporting both dense and mlx_lm-style quantized (qweight, scales, biases) tuples/lists."""
        if isinstance(weight, (tuple, list)) and len(weight) == 3:
            qweight, scales, biases = weight
            out = mx.quantized_matmul(
                x,
                qweight,
                scales,
                biases,
                transpose=True,
                group_size=self.quantize_group_size,
                bits=self.quantize_bits or 8,
            )
            if bias is not None:
                out = out + bias
            return out
        return super().linear(x, weight, bias)

    def reshape(self, x: Any, shape: Tuple[int, ...]) -> Any:
        return mx.reshape(x, shape)

    def swapaxes(self, x: Any, axis1: int, axis2: int) -> Any:
        return mx.swapaxes(x, axis1, axis2)

    def layer_norm(self, x: Any, weight: Any, bias: Optional[Any] = None, eps: float = 1e-5) -> Any:
        if hasattr(mxf, "layer_norm"):
            try:
                return mxf.layer_norm(x, weight, bias, eps=eps)
            except Exception:
                pass
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
        if (
            self.quantize_bits in (4, 8)
            and mlx_arr.ndim == 2
            and mlx_arr.shape[-1] % self.quantize_group_size == 0
        ):
            return self.quantize_weight(mlx_arr, bits=self.quantize_bits, group_size=self.quantize_group_size)
        return mlx_arr

    def to_numpy(self, tensor: Any) -> np.ndarray:
        mx.eval(tensor)
        if tensor.dtype == mx.bfloat16:
            tensor = tensor.astype(mx.float32)
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

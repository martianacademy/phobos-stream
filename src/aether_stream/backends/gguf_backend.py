"""Universal GGUF quantized tensor and CPU fallback backend."""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from aether_stream.backends.base import BaseBackend


class GGUFBackend(BaseBackend):
    """CPU and quantized GGUF execution backend."""

    def __init__(self, device: str = "cpu") -> None:
        super().__init__(device=device)

    @property
    def name(self) -> str:
        return "gguf"

    def zeros(self, shape: Tuple[int, ...], dtype: str = "float32") -> Any:
        np_dtype = getattr(np, dtype, np.float32)
        return np.zeros(shape, dtype=np_dtype)

    def randn(self, shape: Tuple[int, ...], seed: Optional[int] = None, dtype: str = "float32") -> Any:
        rng = np.random.default_rng(seed)
        np_dtype = getattr(np, dtype, np.float32)
        return rng.standard_normal(size=shape).astype(np_dtype)

    def matmul(self, a: Any, b: Any) -> Any:
        return np.matmul(a, b)

    def layer_norm(self, x: Any, weight: Any, bias: Optional[Any] = None, eps: float = 1e-5) -> Any:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        normed = (x - mean) / np.sqrt(var + eps)
        out = normed * weight
        if bias is not None:
            out = out + bias
        return out

    def silu(self, x: Any) -> Any:
        return x / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))

    def scaled_dot_product_attention(
        self,
        q: Any,
        k: Any,
        v: Any,
        scale: Optional[float] = None,
        mask: Optional[Any] = None,
    ) -> Any:
        scale_val = scale if scale is not None else 1.0 / np.sqrt(q.shape[-1])
        scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale_val
        if mask is not None:
            scores = scores + mask
        # numerically stable softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        return np.matmul(weights, v)

    def dequantize_q8_0(self, data: bytes, n_elements: int) -> np.ndarray:
        """Dequantize Q8_0 block format (32 int8 + 1 f16 scale)."""
        block_size = 34  # 2 bytes fp16 delta + 32 bytes int8
        n_blocks = n_elements // 32
        out = np.zeros(n_elements, dtype=np.float32)
        for b in range(n_blocks):
            offset = b * block_size
            d = np.frombuffer(data[offset : offset + 2], dtype=np.float16).astype(np.float32)[0]
            qs = np.frombuffer(data[offset + 2 : offset + 34], dtype=np.int8)
            out[b * 32 : (b + 1) * 32] = qs * d
        return out

    def convert_from_numpy(self, arr: np.ndarray, dtype: Optional[str] = None) -> Any:
        if dtype is not None and hasattr(np, dtype):
            return arr.astype(getattr(np, dtype))
        return arr

    def to_numpy(self, tensor: Any) -> np.ndarray:
        return np.asarray(tensor)

    def synchronize(self) -> None:
        pass

    def empty_cache(self) -> None:
        pass

    def get_memory_allocated_mb(self) -> float:
        return 0.0

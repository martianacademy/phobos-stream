"""Universal GGUF quantized tensor and CPU fallback backend."""

from pathlib import Path
import platform
import resource
from typing import Any, Dict, Optional, Tuple
import numpy as np
from phobos_stream.backends.base import BaseBackend

try:
    import gguf  # type: ignore
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False
    gguf = None


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

    def reshape(self, x: Any, shape: Tuple[int, ...]) -> Any:
        return np.reshape(x, shape)

    def swapaxes(self, x: Any, axis1: int, axis2: int) -> Any:
        return np.swapaxes(x, axis1, axis2)

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
        seq_q = q.shape[-2]
        k_t = np.swapaxes(k, -1, -2)

        # Flash-style query chunking for high-resolution (2K/4K) sequences to avoid O(N^2) memory spikes
        chunk_size = 512
        if seq_q > chunk_size and mask is None:
            out = np.empty_like(q)
            for start in range(0, seq_q, chunk_size):
                end = min(start + chunk_size, seq_q)
                q_chunk = q[..., start:end, :]
                scores = np.matmul(q_chunk, k_t) * scale_val
                exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
                weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
                out[..., start:end, :] = np.matmul(weights, v)
            return out

        scores = np.matmul(q, k_t) * scale_val
        if mask is not None:
            scores = scores + mask
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        return np.matmul(weights, v)

    def quantize_q8_0(self, arr: np.ndarray) -> bytes:
        """Quantize float32 tensor into GGUF Q8_0 block format (32 int8 + 1 fp16 scale)."""
        flat = arr.astype(np.float32).ravel()
        pad = (-len(flat)) % 32
        if pad:
            flat = np.pad(flat, (0, pad))
        blocks = flat.reshape(-1, 32)
        amax = np.max(np.abs(blocks), axis=1, keepdims=True)
        d = (amax / 127.0).astype(np.float16)
        d_f32 = d.astype(np.float32)
        inv_d = np.where(d_f32 == 0, 0.0, 1.0 / d_f32)
        qs = np.clip(np.round(blocks * inv_d), -127, 127).astype(np.int8)

        n_blocks = blocks.shape[0]
        raw = np.empty((n_blocks, 34), dtype=np.uint8)
        raw[:, :2] = d.view(np.uint8).reshape(n_blocks, 2)
        raw[:, 2:] = qs.view(np.uint8)
        return raw.tobytes()

    def dequantize_q8_0(self, data: bytes, n_elements: int) -> np.ndarray:
        """Vectorized dequantization of Q8_0 block format (32 int8 + 1 f16 scale)."""
        block_size = 34  # 2 bytes fp16 delta + 32 bytes int8
        n_blocks = n_elements // 32
        if n_blocks == 0:
            return np.zeros(n_elements, dtype=np.float32)

        raw = np.frombuffer(data, dtype=np.uint8, count=n_blocks * block_size).reshape(n_blocks, block_size)
        scales = raw[:, :2].copy().view(np.float16).astype(np.float32)  # (n_blocks, 1)
        qs = raw[:, 2:].view(np.int8).astype(np.float32)  # (n_blocks, 32)
        return (qs * scales).reshape(-1)[:n_elements]

    def dequantize_q4_0(self, data: bytes, n_elements: int) -> np.ndarray:
        """Vectorized dequantization of Q4_0 block format (16 packed uint8 + 1 f16 scale)."""
        block_size = 18  # 2 bytes fp16 delta + 16 bytes nibbles (32 x 4-bit)
        n_blocks = n_elements // 32
        if n_blocks == 0:
            return np.zeros(n_elements, dtype=np.float32)

        raw = np.frombuffer(data, dtype=np.uint8, count=n_blocks * block_size).reshape(n_blocks, block_size)
        scales = raw[:, :2].copy().view(np.float16).astype(np.float32)  # (n_blocks, 1)
        nibbles = raw[:, 2:]  # (n_blocks, 16)
        low = (nibbles & 0x0F).astype(np.int8) - 8
        high = (nibbles >> 4).astype(np.int8) - 8
        qs = np.concatenate([low, high], axis=1).astype(np.float32)
        return (qs * scales).reshape(-1)[:n_elements]

    def load_gguf_tensors(self, file_path: Path) -> Dict[str, np.ndarray]:
        """Load and dequantize tensors from a .gguf file when gguf package is available."""
        if not GGUF_AVAILABLE:
            raise RuntimeError("The 'gguf' package is required to read .gguf files.")
        reader = gguf.GGUFReader(str(file_path))
        tensors: Dict[str, np.ndarray] = {}
        for tensor in reader.tensors:
            shape = tuple(int(s) for s in reversed(tensor.shape))
            n_elems = int(np.prod(shape))
            t_type = getattr(tensor.tensor_type, "name", "")
            if t_type == "Q8_0":
                arr = self.dequantize_q8_0(bytes(tensor.data), n_elems).reshape(shape)
            elif t_type == "Q4_0":
                arr = self.dequantize_q4_0(bytes(tensor.data), n_elems).reshape(shape)
            else:
                arr = np.array(tensor.data, dtype=np.float32)
                if arr.size == n_elems:
                    arr = arr.reshape(shape)
            tensors[tensor.name] = arr
        return tensors

    def convert_from_numpy(self, arr: np.ndarray, dtype: Optional[str] = None) -> Any:
        if dtype is not None and hasattr(np, dtype):
            return arr.astype(getattr(np, dtype))
        return np.asarray(arr, dtype=np.float32) if arr.dtype != np.float32 else arr

    def to_numpy(self, tensor: Any) -> np.ndarray:
        return np.asarray(tensor)

    def synchronize(self) -> None:
        pass

    def empty_cache(self) -> None:
        pass

    def get_memory_allocated_mb(self) -> float:
        """Report current process resident memory in MB."""
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            return float(usage) / (1024.0 * 1024.0)
        return float(usage) / 1024.0

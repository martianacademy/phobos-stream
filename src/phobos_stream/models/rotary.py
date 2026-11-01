"""3D Spatio-Temporal Rotary Position Embedding (3D-RoPE) for Video/Image DiT."""

from typing import Any, Optional, Tuple
import numpy as np
from phobos_stream.backends.base import BaseBackend


def rotate_half_pairs(x: Any) -> Any:
    """Rotate interleaved 2D coordinate pairs (x0, x1) -> (-x1, x0) across any backend."""
    orig_shape = x.shape
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    # Check backend tensor type dynamically
    mod_name = type(x).__module__
    if "mlx" in mod_name:
        import mlx.core as mx
        stacked = mx.stack([-x_odd, x_even], axis=-1)
        return mx.reshape(stacked, orig_shape)
    if "torch" in mod_name:
        import torch
        stacked = torch.stack([-x_odd, x_even], dim=-1)
        return stacked.reshape(*orig_shape)

    stacked = np.stack([-x_odd, x_even], axis=-1)
    return stacked.reshape(orig_shape)


class RotaryEmbedding3D:
    """3D RoPE encoding temporal (T), height (H), and width (W) coordinates."""

    def __init__(
        self,
        dim: int,
        theta: float = 10000.0,
        backend: Optional[BaseBackend] = None,
    ) -> None:
        self.dim = dim
        self.theta = theta
        self.backend = backend

    def get_freqs_1d(self, length: int, dim_part: int) -> np.ndarray:
        """Compute 1D sinusoidal frequency grid."""
        even_dim = max(2, (dim_part // 2) * 2)
        freqs = 1.0 / (self.theta ** (np.arange(0, even_dim, 2, dtype=np.float32) / even_dim))
        t = np.arange(length, dtype=np.float32)
        grid = np.outer(t, freqs)
        return grid

    def compute_frequencies(self, t_len: int, h_len: int, w_len: int) -> Tuple[Any, Any]:
        """Compute 3D cos and sin tables for spatio-temporal video latents."""
        # Divide dimension across T (1/4), H (3/8), W (3/8) while keeping each part even
        d_t = max(2, ((self.dim // 4) // 2) * 2)
        d_h = max(2, (((self.dim - d_t) // 2) // 2) * 2)
        d_w = max(2, self.dim - d_t - d_h)

        freq_t = self.get_freqs_1d(t_len, d_t)
        freq_h = self.get_freqs_1d(h_len, d_h)
        freq_w = self.get_freqs_1d(w_len, d_w)

        grid_t = np.broadcast_to(freq_t[:, None, None, :], (t_len, h_len, w_len, freq_t.shape[-1]))
        grid_h = np.broadcast_to(freq_h[None, :, None, :], (t_len, h_len, w_len, freq_h.shape[-1]))
        grid_w = np.broadcast_to(freq_w[None, None, :, :], (t_len, h_len, w_len, freq_w.shape[-1]))

        freqs_3d = np.concatenate([grid_t, grid_h, grid_w], axis=-1)
        freqs_flat = freqs_3d.reshape(-1, freqs_3d.shape[-1])

        cos = np.repeat(np.cos(freqs_flat), 2, axis=-1)[:, : self.dim].astype(np.float32)
        sin = np.repeat(np.sin(freqs_flat), 2, axis=-1)[:, : self.dim].astype(np.float32)

        if self.backend is not None:
            return self.backend.convert_from_numpy(cos), self.backend.convert_from_numpy(sin)
        return cos, sin

    def apply_rotary_emb(self, x: Any, cos: Any, sin: Any) -> Any:
        """Apply rotary embedding to query or key tensor."""
        return (x * cos) + ( self._rotate_half(x) * sin )

    def _rotate_half(self, x: Any) -> Any:
        """Rotate interleaved 2D pairs (-x_odd, x_even) for standard rotary multiplication."""
        return rotate_half_pairs(x)

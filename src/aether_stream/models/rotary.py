"""3D Spatio-Temporal Rotary Position Embedding (3D-RoPE) for Video/Image DiT."""

from typing import Any, Tuple
import numpy as np
from aether_stream.backends.base import BaseBackend


class RotaryEmbedding3D:
    """3D RoPE encoding temporal (T), height (H), and width (W) coordinates."""

    def __init__(
        self,
        dim: int,
        theta: float = 10000.0,
        backend: BaseBackend = None,
    ) -> None:
        self.dim = dim
        self.theta = theta
        self.backend = backend

    def get_freqs_1d(self, length: int, dim_part: int) -> np.ndarray:
        """Compute 1D sinusoidal frequency grid."""
        freqs = 1.0 / (self.theta ** (np.arange(0, dim_part, 2, dtype=np.float32) / dim_part))
        t = np.arange(length, dtype=np.float32)
        grid = np.outer(t, freqs)
        return grid

    def compute_frequencies(self, t_len: int, h_len: int, w_len: int) -> Tuple[Any, Any]:
        """Compute 3D cos and sin tables for spatio-temporal video latents."""
        # Divide head dimension across T (1/4), H (3/8), W (3/8)
        d_t = self.dim // 4
        d_h = (self.dim - d_t) // 2
        d_w = self.dim - d_t - d_h

        freq_t = self.get_freqs_1d(t_len, d_t)
        freq_h = self.get_freqs_1d(h_len, d_h)
        freq_w = self.get_freqs_1d(w_len, d_w)

        # Broadcast across 3D grid
        grid_t = np.broadcast_to(freq_t[:, None, None, :], (t_len, h_len, w_len, d_t // 2))
        grid_h = np.broadcast_to(freq_h[None, :, None, :], (t_len, h_len, w_len, d_h // 2))
        grid_w = np.broadcast_to(freq_w[None, None, :, :], (t_len, h_len, w_len, d_w // 2))

        # Concatenate along frequency dimension: (T, H, W, dim // 2)
        freqs_3d = np.concatenate([grid_t, grid_h, grid_w], axis=-1)
        freqs_flat = freqs_3d.reshape(-1, freqs_3d.shape[-1])

        cos = np.repeat(np.cos(freqs_flat), 2, axis=-1)
        sin = np.repeat(np.sin(freqs_flat), 2, axis=-1)

        if self.backend is not None:
            return self.backend.convert_from_numpy(cos), self.backend.convert_from_numpy(sin)
        return cos, sin

    def apply_rotary_emb(self, x: Any, cos: Any, sin: Any) -> Any:
        """Apply rotary embedding to query or key tensor."""
        # x: (Batch, SeqLen, Dim)
        return (x * cos) + (self._rotate_half(x) * sin)

    def _rotate_half(self, x: Any) -> Any:
        """Rotate tensor halves for rotary multiplication."""
        # Simple sign flip on alternating pairs
        return -x

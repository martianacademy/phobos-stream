"""Universal Self-Attention and Cross-Attention for Diffusion Transformers."""

from dataclasses import dataclass
from typing import Any, Optional
from phobos_stream.backends.base import BaseBackend
from phobos_stream.models.rotary import rotate_half_pairs


@dataclass
class AttentionWeights:
    """Weight tensors for a single attention layer."""

    q_proj: Any
    k_proj: Any
    v_proj: Any
    out_proj: Any
    q_bias: Optional[Any] = None
    k_bias: Optional[Any] = None
    v_bias: Optional[Any] = None
    out_bias: Optional[Any] = None


class DiTAttention:
    """Multi-Head Self-Attention and Cross-Attention module."""

    def __init__(self, hidden_size: int, num_heads: int, backend: BaseBackend) -> None:
        self.hidden_size = hidden_size
        self.num_heads = max(1, num_heads)
        self.head_dim = hidden_size // self.num_heads
        self.backend = backend

    def forward(
        self,
        x: Any,
        weights: AttentionWeights,
        context: Optional[Any] = None,
        cos: Optional[Any] = None,
        sin: Optional[Any] = None,
    ) -> Any:
        """Execute multi-head attention forward pass.

        If context is provided, performs Cross-Attention (Q from x, K/V from context).
        Otherwise performs Self-Attention with 3D-RoPE.
        """
        batch_size, seq_q, _ = x.shape

        # Linear projection for Q (handles both (out, in) and (in, out) weight layouts)
        q = self.backend.linear(x, weights.q_proj, weights.q_bias)

        # Determine K, V source
        kv_src = context if context is not None else x
        seq_kv = kv_src.shape[1]
        k = self.backend.linear(kv_src, weights.k_proj, weights.k_bias)
        v = self.backend.linear(kv_src, weights.v_proj, weights.v_bias)

        # Apply 3D-RoPE when self-attending
        if cos is not None and sin is not None and context is None:
            q = (q * cos) + (rotate_half_pairs(q) * sin)
            k = (k * cos) + (rotate_half_pairs(k) * sin)

        # Split heads: (B, Seq, Hidden) -> (B, NumHeads, Seq, HeadDim)
        q = self.backend.swapaxes(
            self.backend.reshape(q, (batch_size, seq_q, self.num_heads, self.head_dim)),
            1,
            2,
        )
        k = self.backend.swapaxes(
            self.backend.reshape(k, (batch_size, seq_kv, self.num_heads, self.head_dim)),
            1,
            2,
        )
        v = self.backend.swapaxes(
            self.backend.reshape(v, (batch_size, seq_kv, self.num_heads, self.head_dim)),
            1,
            2,
        )

        # Multi-head scaled dot-product attention
        scale = 1.0 / (self.head_dim ** 0.5)
        attn_out = self.backend.scaled_dot_product_attention(q, k, v, scale=scale)

        # Merge heads: (B, NumHeads, Seq_Q, HeadDim) -> (B, Seq_Q, Hidden)
        attn_merged = self.backend.reshape(
            self.backend.swapaxes(attn_out, 1, 2),
            (batch_size, seq_q, self.hidden_size),
        )

        # Output projection
        out = self.backend.linear(attn_merged, weights.out_proj, weights.out_bias)
        return out

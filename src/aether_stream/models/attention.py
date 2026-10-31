"""Universal Self-Attention and Cross-Attention for Diffusion Transformers."""

from dataclasses import dataclass
from typing import Any, Optional
from aether_stream.backends.base import BaseBackend


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
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.backend = backend

    def forward(
        self,
        x: Any,
        weights: AttentionWeights,
        context: Optional[Any] = None,
        cos: Optional[Any] = None,
        sin: Optional[Any] = None,
    ) -> Any:
        """Execute attention forward pass.

        If context is provided, performs Cross-Attention (Q from x, K/V from context).
        Otherwise performs Self-Attention.
        """
        # Linear projection for Q
        q = self.backend.matmul(x, weights.q_proj)
        if weights.q_bias is not None:
            q = q + weights.q_bias

        # Determine K, V source
        kv_src = context if context is not None else x
        k = self.backend.matmul(kv_src, weights.k_proj)
        v = self.backend.matmul(kv_src, weights.v_proj)

        if weights.k_bias is not None:
            k = k + weights.k_bias
        if weights.v_bias is not None:
            v = v + weights.v_bias

        # Apply RoPE if provided and self-attending
        if cos is not None and sin is not None and context is None:
            q = (q * cos) - (q * sin)
            k = (k * cos) - (k * sin)

        # Scaled dot-product attention
        scale = 1.0 / (self.head_dim ** 0.5)
        attn_out = self.backend.scaled_dot_product_attention(q, k, v, scale=scale)

        # Output projection
        out = self.backend.matmul(attn_out, weights.out_proj)
        if weights.out_bias is not None:
            out = out + weights.out_bias
        return out

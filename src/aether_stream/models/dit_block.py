"""Universal Diffusion Transformer (DiT) Block with SwiGLU / MoE support."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from aether_stream.backends.base import BaseBackend
from aether_stream.models.attention import AttentionWeights, DiTAttention


@dataclass
class MLPWeights:
    """Feed-Forward Network weights (Standard or SwiGLU)."""

    gate_proj: Any
    up_proj: Any
    down_proj: Any


@dataclass
class DiTBlockWeights:
    """All weights required to execute a single DiT block."""

    norm1_weight: Any
    self_attn: AttentionWeights
    norm2_weight: Any
    cross_attn: Optional[AttentionWeights]
    norm3_weight: Any
    mlp: MLPWeights
    ada_scale: Optional[Any] = None


class DiTBlock:
    """Universal Diffusion Transformer layer block."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        backend: BaseBackend,
        has_cross_attn: bool = True,
    ) -> None:
        self.hidden_size = hidden_size
        self.backend = backend
        self.self_attn_module = DiTAttention(hidden_size, num_heads, backend)
        self.has_cross_attn = has_cross_attn
        if has_cross_attn:
            self.cross_attn_module = DiTAttention(hidden_size, num_heads, backend)

    def forward(
        self,
        x: Any,
        weights: DiTBlockWeights,
        context: Optional[Any] = None,
        timestep_emb: Optional[Any] = None,
        cos: Optional[Any] = None,
        sin: Optional[Any] = None,
    ) -> Any:
        """Forward pass through Self-Attention, Cross-Attention, and MLP."""
        # 1. Pre-norm Self-Attention
        norm1 = self.backend.layer_norm(x, weights.norm1_weight)
        attn_out = self.self_attn_module.forward(
            norm1,
            weights.self_attn,
            context=None,
            cos=cos,
            sin=sin,
        )
        x = x + attn_out

        # 2. Cross-Attention (Conditioning on prompt text embeddings)
        if self.has_cross_attn and weights.cross_attn is not None and context is not None:
            norm2 = self.backend.layer_norm(x, weights.norm2_weight)
            cross_out = self.cross_attn_module.forward(norm2, weights.cross_attn, context=context)
            x = x + cross_out

        # 3. Feed-Forward MLP (SwiGLU)
        norm3 = self.backend.layer_norm(x, weights.norm3_weight)
        gate = self.backend.silu(self.backend.matmul(norm3, weights.mlp.gate_proj))
        up = self.backend.matmul(norm3, weights.mlp.up_proj)
        mlp_intermediate = gate * up
        mlp_out = self.backend.matmul(mlp_intermediate, weights.mlp.down_proj)

        x = x + mlp_out
        return x

"""Universal Diffusion Transformer (DiT) Block with SwiGLU & AdaLN support."""

from dataclasses import dataclass
from typing import Any, Optional
from phobos_stream.backends.base import BaseBackend
from phobos_stream.models.attention import AttentionWeights, DiTAttention


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

    def _apply_timestep_modulation(
        self,
        normed_x: Any,
        timestep_emb: Optional[Any],
        ada_scale: Optional[Any],
    ) -> Any:
        """Modulate normalized activations with timestep embedding (AdaLN)."""
        if timestep_emb is None:
            return normed_x

        t_cond = self.backend.silu(timestep_emb)
        if ada_scale is not None:
            t_cond = self.backend.linear(t_cond, ada_scale)

        if len(t_cond.shape) == 1:
            t_cond = self.backend.reshape(t_cond, (1, 1, -1))
        elif len(t_cond.shape) == 2:
            t_cond = self.backend.reshape(t_cond, (t_cond.shape[0], 1, -1))

        # Scale & shift modulation when dimensions align
        if t_cond.shape[-1] == self.hidden_size:
            return normed_x * (1.0 + 0.1 * t_cond) + (0.05 * t_cond)
        return normed_x

    def forward(
        self,
        x: Any,
        weights: DiTBlockWeights,
        context: Optional[Any] = None,
        timestep_emb: Optional[Any] = None,
        cos: Optional[Any] = None,
        sin: Optional[Any] = None,
    ) -> Any:
        """Forward pass through AdaLN Self-Attention, Cross-Attention, and SwiGLU MLP."""
        # 1. Pre-norm + AdaLN Self-Attention
        norm1 = self.backend.layer_norm(x, weights.norm1_weight)
        norm1 = self._apply_timestep_modulation(norm1, timestep_emb, weights.ada_scale)
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

        # 3. Feed-Forward MLP (SwiGLU) with AdaLN modulation
        norm3 = self.backend.layer_norm(x, weights.norm3_weight)
        norm3 = self._apply_timestep_modulation(norm3, timestep_emb, None)
        gate = self.backend.silu(self.backend.linear(norm3, weights.mlp.gate_proj))
        up = self.backend.linear(norm3, weights.mlp.up_proj)
        mlp_intermediate = gate * up
        mlp_out = self.backend.linear(mlp_intermediate, weights.mlp.down_proj)

        x = x + mlp_out
        return x

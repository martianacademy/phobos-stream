"""Adapter for CogVideoX (5B / 2B) 3D Diffusion Transformer."""

from typing import Any, Dict
from phobos_stream.adapters.base import BaseModelAdapter
from phobos_stream.models.attention import AttentionWeights
from phobos_stream.models.dit_block import DiTBlockWeights, MLPWeights


class CogVideoXAdapter(BaseModelAdapter):
    """Adapter for Tsinghua/Zhipu CogVideoX DiT models."""

    @property
    def architecture_name(self) -> str:
        return "cogvideox"

    def extract_block_weights(
        self,
        block_id: int,
        raw_tensors: Dict[str, Any],
    ) -> DiTBlockWeights:
        """Map CogVideoX transformer_blocks tensors into normalized DiTBlockWeights."""
        prefix = f"transformer_blocks.{block_id}."
        dim = self.hidden_size

        def get_t(suffix: str, shape):
            return self.resolve_tensor(raw_tensors, block_id, prefix, suffix, shape)

        self_attn = AttentionWeights(
            q_proj=get_t("attn1.to_q.weight", (dim, dim)),
            k_proj=get_t("attn1.to_k.weight", (dim, dim)),
            v_proj=get_t("attn1.to_v.weight", (dim, dim)),
            out_proj=get_t("attn1.to_out.0.weight", (dim, dim)),
        )

        cross_attn = AttentionWeights(
            q_proj=get_t("attn2.to_q.weight", (dim, dim)),
            k_proj=get_t("attn2.to_k.weight", (dim, dim)),
            v_proj=get_t("attn2.to_v.weight", (dim, dim)),
            out_proj=get_t("attn2.to_out.0.weight", (dim, dim)),
        )

        mlp = MLPWeights(
            gate_proj=get_t("ff.net.0.proj.weight", (dim, dim * 4)),
            up_proj=get_t("ff.net.1.proj.weight", (dim, dim * 4)),
            down_proj=get_t("ff.net.2.weight", (dim * 4, dim)),
        )

        return DiTBlockWeights(
            norm1_weight=get_t("norm1.weight", (dim,)),
            self_attn=self_attn,
            norm2_weight=get_t("norm2.weight", (dim,)),
            cross_attn=cross_attn,
            norm3_weight=get_t("norm3.weight", (dim,)),
            mlp=mlp,
        )

"""Adapter for Tencent HunyuanVideo 13B Diffusion Transformer."""

from typing import Any, Dict
from phobos_stream.adapters.base import BaseModelAdapter
from phobos_stream.models.attention import AttentionWeights
from phobos_stream.models.dit_block import DiTBlockWeights, MLPWeights


class HunyuanAdapter(BaseModelAdapter):
    """Adapter for HunyuanVideo 13B models."""

    @property
    def architecture_name(self) -> str:
        return "hunyuan"

    def extract_block_weights(
        self,
        block_id: int,
        raw_tensors: Dict[str, Any],
    ) -> DiTBlockWeights:
        """Map HunyuanVideo double/single block tensors into normalized DiTBlockWeights."""
        is_single = any("single_blocks." in k for k in raw_tensors)
        prefix = f"single_blocks.{block_id}." if is_single else f"double_blocks.{block_id}."
        dim = self.hidden_size

        def get_t(suffix: str, shape):
            return self.resolve_tensor(raw_tensors, block_id, prefix, suffix, shape)

        self_attn = AttentionWeights(
            q_proj=get_t("img_attn.q_proj.weight", (dim, dim)),
            k_proj=get_t("img_attn.k_proj.weight", (dim, dim)),
            v_proj=get_t("img_attn.v_proj.weight", (dim, dim)),
            out_proj=get_t("img_attn.proj.weight", (dim, dim)),
        )

        cross_attn = None if is_single else AttentionWeights(
            q_proj=get_t("txt_attn.q_proj.weight", (dim, dim)),
            k_proj=get_t("txt_attn.k_proj.weight", (dim, dim)),
            v_proj=get_t("txt_attn.v_proj.weight", (dim, dim)),
            out_proj=get_t("txt_attn.proj.weight", (dim, dim)),
        )

        mlp = MLPWeights(
            gate_proj=get_t("img_mlp.gate.weight", (dim, dim * 4)),
            up_proj=get_t("img_mlp.up.weight", (dim, dim * 4)),
            down_proj=get_t("img_mlp.down.weight", (dim * 4, dim)),
        )

        return DiTBlockWeights(
            norm1_weight=get_t("img_norm1.weight", (dim,)),
            self_attn=self_attn,
            norm2_weight=get_t("txt_norm1.weight", (dim,)),
            cross_attn=cross_attn,
            norm3_weight=get_t("img_norm2.weight", (dim,)),
            mlp=mlp,
        )

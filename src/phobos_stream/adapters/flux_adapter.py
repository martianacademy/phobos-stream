"""Adapter for Black Forest Labs FLUX.1 (schnell / dev) DiT models."""

from typing import Any, Dict
from phobos_stream.adapters.base import BaseModelAdapter
from phobos_stream.models.attention import AttentionWeights
from phobos_stream.models.dit_block import DiTBlockWeights, MLPWeights


class FluxAdapter(BaseModelAdapter):
    """Adapter for FLUX.1 12B DiT models (double and single stream blocks)."""

    @property
    def architecture_name(self) -> str:
        return "flux"

    def extract_block_weights(
        self,
        block_id: int,
        raw_tensors: Dict[str, Any],
    ) -> DiTBlockWeights:
        """Map FLUX.1 double/single block tensors into normalized DiTBlockWeights."""
        is_single = any("single_blocks." in k for k in raw_tensors)
        prefix = f"single_blocks.{block_id}." if is_single else f"double_blocks.{block_id}."
        dim = self.hidden_size

        def get_t(suffix: str, shape):
            return self.resolve_tensor(raw_tensors, block_id, prefix, suffix, shape)

        if is_single:
            self_attn = AttentionWeights(
                q_proj=get_t("linear1.q.weight", (dim, dim)),
                k_proj=get_t("linear1.k.weight", (dim, dim)),
                v_proj=get_t("linear1.v.weight", (dim, dim)),
                out_proj=get_t("linear2.proj.weight", (dim, dim)),
            )
            cross_attn = None
            mlp = MLPWeights(
                gate_proj=get_t("mlp.gate.weight", (dim, dim * 4)),
                up_proj=get_t("mlp.up.weight", (dim, dim * 4)),
                down_proj=get_t("linear2.mlp.weight", (dim * 4, dim)),
            )
            return DiTBlockWeights(
                norm1_weight=get_t("modulation.lin.weight", (dim,)),
                self_attn=self_attn,
                norm2_weight=get_t("modulation.lin.weight", (dim,)),
                cross_attn=cross_attn,
                norm3_weight=get_t("modulation.lin.weight", (dim,)),
                mlp=mlp,
            )

        self_attn = AttentionWeights(
            q_proj=get_t("img_attn.q_proj.weight", (dim, dim)),
            k_proj=get_t("img_attn.k_proj.weight", (dim, dim)),
            v_proj=get_t("img_attn.v_proj.weight", (dim, dim)),
            out_proj=get_t("img_attn.proj.weight", (dim, dim)),
        )

        cross_attn = AttentionWeights(
            q_proj=get_t("txt_attn.q_proj.weight", (dim, dim)),
            k_proj=get_t("txt_attn.k_proj.weight", (dim, dim)),
            v_proj=get_t("txt_attn.v_proj.weight", (dim, dim)),
            out_proj=get_t("txt_attn.proj.weight", (dim, dim)),
        )

        mlp = MLPWeights(
            gate_proj=get_t("img_mlp.0.weight", (dim, dim * 4)),
            up_proj=get_t("img_mlp.1.weight", (dim, dim * 4)),
            down_proj=get_t("img_mlp.2.weight", (dim * 4, dim)),
        )

        return DiTBlockWeights(
            norm1_weight=get_t("img_mod.lin.weight", (dim,)),
            self_attn=self_attn,
            norm2_weight=get_t("txt_mod.lin.weight", (dim,)),
            cross_attn=cross_attn,
            norm3_weight=get_t("img_norm2.weight", (dim,)),
            mlp=mlp,
        )

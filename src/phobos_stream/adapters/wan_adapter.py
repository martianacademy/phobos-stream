"""Adapter for Alibaba Wan2.1 (14B / 1.3B) and Lumina2 / Z-Image-Turbo DiT models."""

from typing import Any, Dict
from phobos_stream.adapters.base import BaseModelAdapter
from phobos_stream.models.attention import AttentionWeights
from phobos_stream.models.dit_block import DiTBlockWeights, MLPWeights


class WanAdapter(BaseModelAdapter):
    """Adapter for Wan2.1 and Lumina2 / Z-Image-Turbo DiT models."""

    @property
    def architecture_name(self) -> str:
        return "wan2.1"

    def extract_block_weights(
        self,
        block_id: int,
        raw_tensors: Dict[str, Any],
    ) -> DiTBlockWeights:
        """Map Wan2.1 or Lumina2/Z-Image-Turbo layer tensors into normalized DiTBlockWeights."""
        dim = self.hidden_size

        # Detect Lumina2 / Z-Image-Turbo fused QKV format ('layers.{block_id}.attention.qkv.weight')
        lumina_qkv_key = f"layers.{block_id}.attention.qkv.weight"
        if lumina_qkv_key in raw_tensors:
            prefix = f"layers.{block_id}."
            qkv = raw_tensors[lumina_qkv_key]
            q_np, k_np, v_np = qkv[:dim], qkv[dim : 2 * dim], qkv[2 * dim : 3 * dim]
            self_attn = AttentionWeights(
                q_proj=self.backend.convert_from_numpy(q_np),
                k_proj=self.backend.convert_from_numpy(k_np),
                v_proj=self.backend.convert_from_numpy(v_np),
                out_proj=self.resolve_tensor(raw_tensors, block_id, prefix, "attention.out.weight", (dim, dim)),
            )
            mlp = MLPWeights(
                gate_proj=self.resolve_tensor(raw_tensors, block_id, prefix, "feed_forward.w1.weight", (dim * 2, dim)),
                up_proj=self.resolve_tensor(raw_tensors, block_id, prefix, "feed_forward.w3.weight", (dim * 2, dim)),
                down_proj=self.resolve_tensor(raw_tensors, block_id, prefix, "feed_forward.w2.weight", (dim, dim * 2)),
            )
            return DiTBlockWeights(
                norm1_weight=self.resolve_tensor(raw_tensors, block_id, prefix, "attention_norm1.weight", (dim,)),
                self_attn=self_attn,
                norm2_weight=self.resolve_tensor(raw_tensors, block_id, prefix, "attention_norm2.weight", (dim,)),
                cross_attn=None,
                norm3_weight=self.resolve_tensor(raw_tensors, block_id, prefix, "ffn_norm1.weight", (dim,)),
                mlp=mlp,
            )

        prefix = f"blocks.{block_id}."

        def get_t(suffix: str, shape):
            return self.resolve_tensor(raw_tensors, block_id, prefix, suffix, shape)

        self_attn = AttentionWeights(
            q_proj=get_t("self_attn.q_proj.weight", (dim, dim)),
            k_proj=get_t("self_attn.k_proj.weight", (dim, dim)),
            v_proj=get_t("self_attn.v_proj.weight", (dim, dim)),
            out_proj=get_t("self_attn.o_proj.weight", (dim, dim)),
        )

        cross_attn = AttentionWeights(
            q_proj=get_t("cross_attn.q_proj.weight", (dim, dim)),
            k_proj=get_t("cross_attn.k_proj.weight", (dim, dim)),
            v_proj=get_t("cross_attn.v_proj.weight", (dim, dim)),
            out_proj=get_t("cross_attn.o_proj.weight", (dim, dim)),
        )

        mlp = MLPWeights(
            gate_proj=get_t("ffn.gate_proj.weight", (dim, dim * 2)),
            up_proj=get_t("ffn.up_proj.weight", (dim, dim * 2)),
            down_proj=get_t("ffn.down_proj.weight", (dim * 2, dim)),
        )

        return DiTBlockWeights(
            norm1_weight=get_t("norm1.weight", (dim,)),
            self_attn=self_attn,
            norm2_weight=get_t("norm2.weight", (dim,)),
            cross_attn=cross_attn,
            norm3_weight=get_t("norm3.weight", (dim,)),
            mlp=mlp,
        )

"""Adapter for Tencent HunyuanVideo 13B Diffusion Transformer."""

from typing import Any, Dict
import numpy as np
from aether_stream.adapters.base import BaseModelAdapter
from aether_stream.models.attention import AttentionWeights
from aether_stream.models.dit_block import DiTBlockWeights, MLPWeights


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
        """Map HunyuanVideo layer tensors into normalized DiTBlockWeights."""
        prefix = f"double_blocks.{block_id}."
        dim = self.hidden_size

        def get_tensor(suffix: str, default_shape=(64, 64)) -> Any:
            key = f"{prefix}{suffix}"
            if key in raw_tensors:
                return self.backend.convert_from_numpy(raw_tensors[key])
            for k in raw_tensors:
                if suffix in k:
                    return self.backend.convert_from_numpy(raw_tensors[k])
            return self.backend.zeros(default_shape)

        self_attn = AttentionWeights(
            q_proj=get_tensor("img_attn.q_proj.weight", (dim, dim)),
            k_proj=get_tensor("img_attn.k_proj.weight", (dim, dim)),
            v_proj=get_tensor("img_attn.v_proj.weight", (dim, dim)),
            out_proj=get_tensor("img_attn.proj.weight", (dim, dim)),
        )

        cross_attn = AttentionWeights(
            q_proj=get_tensor("txt_attn.q_proj.weight", (dim, dim)),
            k_proj=get_tensor("txt_attn.k_proj.weight", (dim, dim)),
            v_proj=get_tensor("txt_attn.v_proj.weight", (dim, dim)),
            out_proj=get_tensor("txt_attn.proj.weight", (dim, dim)),
        )

        mlp = MLPWeights(
            gate_proj=get_tensor("img_mlp.gate.weight", (dim, dim * 4)),
            up_proj=get_tensor("img_mlp.up.weight", (dim, dim * 4)),
            down_proj=get_tensor("img_mlp.down.weight", (dim * 4, dim)),
        )

        return DiTBlockWeights(
            norm1_weight=get_tensor("img_norm1.weight", (dim,)),
            self_attn=self_attn,
            norm2_weight=get_tensor("txt_norm1.weight", (dim,)),
            cross_attn=cross_attn,
            norm3_weight=get_tensor("img_norm2.weight", (dim,)),
            mlp=mlp,
        )

    def format_prompt_embeddings(self, text_tokens: Any) -> Any:
        return text_tokens

    def compute_timestep_embedding(self, timestep: float, dim: int) -> Any:
        half = dim // 2
        freqs = np.exp(-np.log(10000.0) * np.arange(0, half, dtype=np.float32) / half)
        args = timestep * freqs
        embedding = np.concatenate([np.cos(args), np.sin(args)], axis=-1)
        return self.backend.convert_from_numpy(embedding)

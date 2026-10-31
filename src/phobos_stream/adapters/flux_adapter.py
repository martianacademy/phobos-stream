"""Adapter for Black Forest Labs FLUX.1 (schnell / dev) DiT models."""

from typing import Any, Dict
import numpy as np
from phobos_stream.adapters.base import BaseModelAdapter
from phobos_stream.models.attention import AttentionWeights
from phobos_stream.models.dit_block import DiTBlockWeights, MLPWeights


class FluxAdapter(BaseModelAdapter):
    """Adapter for FLUX.1 12B DiT models."""

    @property
    def architecture_name(self) -> str:
        return "flux"

    def extract_block_weights(
        self,
        block_id: int,
        raw_tensors: Dict[str, Any],
    ) -> DiTBlockWeights:
        """Map FLUX.1 double/single block tensors into normalized DiTBlockWeights."""
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
            gate_proj=get_tensor("img_mlp.0.weight", (dim, dim * 4)),
            up_proj=get_tensor("img_mlp.0.weight", (dim, dim * 4)),
            down_proj=get_tensor("img_mlp.2.weight", (dim * 4, dim)),
        )

        return DiTBlockWeights(
            norm1_weight=get_tensor("img_mod.lin.weight", (dim,)),
            self_attn=self_attn,
            norm2_weight=get_tensor("txt_mod.lin.weight", (dim,)),
            cross_attn=cross_attn,
            norm3_weight=get_tensor("img_mod.lin.weight", (dim,)),
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

"""Base model adapter interface for Diffusion Transformer models."""

from abc import ABC, abstractmethod
import hashlib
from typing import Any, Dict, Tuple
import numpy as np
from phobos_stream.backends.base import BaseBackend
from phobos_stream.models.dit_block import DiTBlockWeights


class BaseModelAdapter(ABC):
    """Abstract adapter mapping model-specific safetensors weights to DiTBlockWeights."""

    def __init__(
        self,
        backend: BaseBackend,
        hidden_size: int = 4096,
        allow_synthetic: bool = False,
    ) -> None:
        self.backend = backend
        self.hidden_size = hidden_size
        self.allow_synthetic = allow_synthetic
        self._synthetic_cache: Dict[Tuple[int, str, Tuple[int, ...]], Any] = {}

    @property
    @abstractmethod
    def architecture_name(self) -> str:
        """Name of the model architecture."""
        pass

    def _synthesize_weight(self, block_id: int, suffix: str, shape: Tuple[int, ...]) -> Any:
        """Synthesize deterministic Xavier-scaled weight ONLY when `allow_synthetic=True`."""
        cache_key = (block_id, suffix, shape)
        if cache_key in self._synthetic_cache:
            return self._synthetic_cache[cache_key]

        if len(shape) == 1:
            arr = np.ones(shape, dtype=np.float32)
        else:
            digest = hashlib.sha256(f"{self.architecture_name}:{block_id}:{suffix}".encode("utf-8")).digest()
            seed = int.from_bytes(digest[:4], "little")
            rng = np.random.default_rng(seed)
            fan_in = shape[0]
            fan_out = shape[1] if len(shape) > 1 else shape[0]
            scale = float(np.sqrt(2.0 / max(1, fan_in + fan_out))) * 0.25
            arr = (rng.standard_normal(size=shape) * scale).astype(np.float32)

        tensor = self.backend.convert_from_numpy(arr)
        self._synthetic_cache[cache_key] = tensor
        return tensor

    def resolve_tensor(
        self,
        raw_tensors: Dict[str, Any],
        block_id: int,
        prefix: str,
        suffix: str,
        default_shape: Tuple[int, ...],
    ) -> Any:
        """Look up a tensor by exact key or suffix match; raise KeyError if missing in strict mode."""
        key = f"{prefix}{suffix}"
        if key in raw_tensors:
            return self.backend.convert_from_numpy(raw_tensors[key])
        for k, v in raw_tensors.items():
            if k.endswith(suffix) or suffix in k:
                return self.backend.convert_from_numpy(v)

        if not self.allow_synthetic:
            raise KeyError(
                f"Missing required weight tensor '{key}' for block {block_id} on architecture "
                f"'{self.architecture_name}' (allow_synthetic=False)."
            )
        return self._synthesize_weight(block_id, suffix, default_shape)

    @abstractmethod
    def extract_block_weights(
        self,
        block_id: int,
        raw_tensors: Dict[str, Any],
    ) -> DiTBlockWeights:
        """Translate raw model state dict tensors into normalized DiTBlockWeights."""
        pass

    def format_prompt_embeddings(self, text_tokens: Any) -> Any:
        """Format or project text encoder embeddings for cross-attention."""
        return text_tokens

    def compute_timestep_embedding(self, timestep: float, dim: int) -> Any:
        """Compute sinusoidal timestep embedding."""
        half = max(1, dim // 2)
        freqs = np.exp(-np.log(10000.0) * np.arange(0, half, dtype=np.float32) / half)
        args = float(timestep) * 1000.0 * freqs
        embedding = np.concatenate([np.cos(args), np.sin(args)], axis=-1)
        if embedding.shape[0] < dim:
            embedding = np.pad(embedding, (0, dim - embedding.shape[0]))
        return self.backend.convert_from_numpy(embedding.astype(np.float32))

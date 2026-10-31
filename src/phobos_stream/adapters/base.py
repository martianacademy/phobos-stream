"""Base model adapter interface for Diffusion Transformer models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from phobos_stream.backends.base import BaseBackend
from phobos_stream.models.dit_block import DiTBlockWeights, MLPWeights
from phobos_stream.models.attention import AttentionWeights


class BaseModelAdapter(ABC):
    """Abstract adapter mapping model-specific safetensors weights to DiTBlockWeights."""

    def __init__(self, backend: BaseBackend, hidden_size: int = 4096) -> None:
        self.backend = backend
        self.hidden_size = hidden_size

    @property
    @abstractmethod
    def architecture_name(self) -> str:
        """Name of the model architecture."""
        pass

    @abstractmethod
    def extract_block_weights(
        self,
        block_id: int,
        raw_tensors: Dict[str, Any],
    ) -> DiTBlockWeights:
        """Translate raw model state dict tensors into normalized DiTBlockWeights."""
        pass

    @abstractmethod
    def format_prompt_embeddings(self, text_tokens: Any) -> Any:
        """Format or project text encoder embeddings for cross-attention."""
        pass

    @abstractmethod
    def compute_timestep_embedding(self, timestep: float, dim: int) -> Any:
        """Compute sinusoidal or learned timestep embedding."""
        pass

"""Storage layout and file path resolution for Diffusion models."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json


@dataclass
class BlockMetadata:
    """Metadata for an individual DiT transformer block.

    Attributes:
        block_id: Sequential index of the transformer block in execution order.
        filename: Path to the `.safetensors` shard containing this block's tensors.
        tensor_names: Exact safetensors keys belonging to this block.
        byte_size: Total raw byte footprint of all tensors in this block.
    """

    block_id: int
    filename: str
    tensor_names: List[str] = field(default_factory=list)
    byte_size: int = 0


@dataclass
class ModelShardsManifest:
    """Manifest describing partitioned or virtually indexed weights of a diffusion pipeline.

    Attributes:
        model_name: Human-readable model identifier.
        architecture: Architecture family (e.g. 'wan2.1', 'flux', 'hunyuan', 'cogvideox').
        num_blocks: Number of transformer blocks to execute per denoising step.
        hidden_size: Hidden feature dimension of the transformer backbone.
        text_encoder_path: Optional path to a `.safetensors` text encoder checkpoint.
        vae_path: Optional path to a `.safetensors` VAE checkpoint.
        blocks: Mapping from integer block_id to its BlockMetadata.
        allow_synthetic: Explicit opt-in for synthetic benchmark/test weights when real
            checkpoint files are absent. Defaults to False so production calls never
            silently fabricate weights.
    """

    model_name: str
    architecture: str
    num_blocks: int
    hidden_size: int
    text_encoder_path: Optional[str] = None
    vae_path: Optional[str] = None
    blocks: Dict[int, BlockMetadata] = field(default_factory=dict)
    allow_synthetic: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize manifest metadata to a JSON-compatible dictionary."""
        return {
            "model_name": self.model_name,
            "architecture": self.architecture,
            "num_blocks": self.num_blocks,
            "hidden_size": self.hidden_size,
            "text_encoder_path": self.text_encoder_path,
            "vae_path": self.vae_path,
            "allow_synthetic": self.allow_synthetic,
            "blocks": {
                str(k): {
                    "block_id": v.block_id,
                    "filename": v.filename,
                    "tensor_names": v.tensor_names,
                    "byte_size": v.byte_size,
                }
                for k, v in self.blocks.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelShardsManifest":
        """Reconstruct a ModelShardsManifest from a parsed JSON dictionary."""
        blocks: Dict[int, BlockMetadata] = {}
        for k, v in data.get("blocks", {}).items():
            blocks[int(k)] = BlockMetadata(
                block_id=int(v["block_id"]),
                filename=str(v["filename"]),
                tensor_names=list(v.get("tensor_names", [])),
                byte_size=int(v.get("byte_size", 0)),
            )
        return cls(
            model_name=str(data["model_name"]),
            architecture=str(data["architecture"]),
            num_blocks=int(data["num_blocks"]),
            hidden_size=int(data["hidden_size"]),
            text_encoder_path=data.get("text_encoder_path"),
            vae_path=data.get("vae_path"),
            blocks=blocks,
            allow_synthetic=bool(data.get("allow_synthetic", False)),
        )

    def save(self, path: Path) -> None:
        """Write manifest JSON to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ModelShardsManifest":
        """Load manifest JSON from disk."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

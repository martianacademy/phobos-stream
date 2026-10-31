"""Storage layout and file path resolution for Diffusion models."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import json


@dataclass
class BlockMetadata:
    """Metadata for an individual DiT transformer block."""

    block_id: int
    filename: str
    tensor_names: List[str] = field(default_factory=list)
    byte_size: int = 0


@dataclass
class ModelShardsManifest:
    """Manifest describing partitioned weights of a diffusion pipeline."""

    model_name: str
    architecture: str
    num_blocks: int
    hidden_size: int
    text_encoder_path: Optional[str] = None
    vae_path: Optional[str] = None
    blocks: Dict[int, BlockMetadata] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "architecture": self.architecture,
            "num_blocks": self.num_blocks,
            "hidden_size": self.hidden_size,
            "text_encoder_path": self.text_encoder_path,
            "vae_path": self.vae_path,
            "blocks": {
                k: {
                    "block_id": v.block_id,
                    "filename": v.filename,
                    "tensor_names": v.tensor_names,
                    "byte_size": v.byte_size,
                }
                for k, v in self.blocks.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelShardsManifest":
        blocks = {}
        for k, v in data.get("blocks", {}).items():
            blocks[int(k)] = BlockMetadata(
                block_id=v["block_id"],
                filename=v["filename"],
                tensor_names=v.get("tensor_names", []),
                byte_size=v.get("byte_size", 0),
            )
        return cls(
            model_name=data["model_name"],
            architecture=data["architecture"],
            num_blocks=data["num_blocks"],
            hidden_size=data["hidden_size"],
            text_encoder_path=data.get("text_encoder_path"),
            vae_path=data.get("vae_path"),
            blocks=blocks,
        )

    def save(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ModelShardsManifest":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

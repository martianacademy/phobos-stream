"""Model sharder and virtual byte-range indexer for zero-copy streaming."""

from pathlib import Path
from typing import Dict, List, Optional
import os
import safetensors.numpy
from aether_stream.storage.layout import BlockMetadata, ModelShardsManifest
from aether_stream.storage.mmap_loader import MMapTensorLoader


class DiffusionSharder:
    """Partitions or virtual-indexes Diffusion Transformer weights."""

    def __init__(self, model_dir: Path, output_dir: Optional[Path] = None) -> None:
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir) if output_dir else self.model_dir / "shards"

    def build_virtual_manifest(
        self,
        architecture: str = "generic_dit",
        hidden_size: int = 4096,
    ) -> ModelShardsManifest:
        """Create a zero-copy virtual manifest from existing monolithic safetensors."""
        safetensor_files = list(self.model_dir.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No safetensors found in {self.model_dir}")

        blocks: Dict[int, BlockMetadata] = {}
        text_encoder_path: Optional[str] = None
        vae_path: Optional[str] = None

        for sf in safetensor_files:
            fname = sf.name.lower()
            if "text_encoder" in fname or "t5" in fname or "clip" in fname:
                text_encoder_path = str(sf.resolve())
            elif "vae" in fname:
                vae_path = str(sf.resolve())

            loader = MMapTensorLoader(sf)
            tensor_names = loader.get_tensor_names()
            loader.close()

            # Group by transformer block index (e.g., 'blocks.0.', 'transformer_blocks.0.')
            for t_name in tensor_names:
                for prefix in ("blocks.", "transformer_blocks.", "double_blocks.", "single_blocks."):
                    if prefix in t_name:
                        parts = t_name.split(prefix)[1].split(".")
                        if parts[0].isdigit():
                            b_id = int(parts[0])
                            if b_id not in blocks:
                                blocks[b_id] = BlockMetadata(
                                    block_id=b_id,
                                    filename=str(sf.resolve()),
                                    tensor_names=[],
                                    byte_size=0,
                                )
                            blocks[b_id].tensor_names.append(t_name)
                        break

        manifest = ModelShardsManifest(
            model_name=self.model_dir.name,
            architecture=architecture,
            num_blocks=len(blocks) if blocks else 1,
            hidden_size=hidden_size,
            text_encoder_path=text_encoder_path,
            vae_path=vae_path,
            blocks=blocks,
        )
        return manifest

    def partition_to_disk(self, manifest: ModelShardsManifest) -> Path:
        """Physical sharding helper when offline partitioning is explicitly requested."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        blocks_dir = self.output_dir / "blocks"
        blocks_dir.mkdir(exist_ok=True)

        new_blocks: Dict[int, BlockMetadata] = {}
        for b_id, meta in manifest.blocks.items():
            loader = MMapTensorLoader(Path(meta.filename))
            block_tensors = {t_name: loader.read_tensor(t_name) for t_name in meta.tensor_names}
            loader.close()

            shard_file = blocks_dir / f"block_{b_id:03d}.safetensors"
            safetensors.numpy.save_file(block_tensors, str(shard_file))

            new_blocks[b_id] = BlockMetadata(
                block_id=b_id,
                filename=str(shard_file.resolve()),
                tensor_names=meta.tensor_names,
                byte_size=os.path.getsize(shard_file),
            )

        partitioned_manifest = ModelShardsManifest(
            model_name=manifest.model_name,
            architecture=manifest.architecture,
            num_blocks=manifest.num_blocks,
            hidden_size=manifest.hidden_size,
            text_encoder_path=manifest.text_encoder_path,
            vae_path=manifest.vae_path,
            blocks=new_blocks,
        )
        manifest_path = self.output_dir / "manifest.json"
        partitioned_manifest.save(manifest_path)
        return manifest_path

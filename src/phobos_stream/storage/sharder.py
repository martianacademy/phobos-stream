"""Model sharder and virtual byte-range indexer for zero-copy streaming."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
import safetensors.numpy
from phobos_stream.storage.layout import BlockMetadata, ModelShardsManifest
from phobos_stream.storage.mmap_loader import MMapTensorLoader


BLOCK_PREFIXES: Tuple[str, ...] = (
    "transformer_blocks.",
    "double_blocks.",
    "single_blocks.",
    "blocks.",
    "layers.",
)


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
        if self.model_dir.is_file() and self.model_dir.suffix == ".safetensors":
            safetensor_files = [self.model_dir]
        else:
            safetensor_files = sorted(self.model_dir.glob("*.safetensors"))

        if not safetensor_files:
            raise FileNotFoundError(f"No safetensors found in {self.model_dir}")

        blocks: Dict[int, BlockMetadata] = {}
        single_blocks_temp: Dict[int, Tuple[str, List[str], int]] = {}
        text_encoder_path: Optional[str] = None
        vae_path: Optional[str] = None

        for sf in safetensor_files:
            fname = sf.name.lower()
            if "text_encoder" in fname or "t5" in fname or "clip" in fname or "qwen" in fname:
                text_encoder_path = str(sf.resolve())
            elif "vae" in fname or fname == "ae.safetensors":
                vae_path = str(sf.resolve())

            with MMapTensorLoader(sf) as loader:
                tensor_names = loader.get_tensor_names()
                for t_name in tensor_names:
                    for prefix in BLOCK_PREFIXES:
                        if prefix in t_name:
                            parts = t_name.split(prefix, 1)[1].split(".")
                            if parts[0].isdigit():
                                raw_idx = int(parts[0])
                                t_bytes = loader.get_tensor_byte_size(t_name)
                                if prefix == "single_blocks.":
                                    if raw_idx not in single_blocks_temp:
                                        single_blocks_temp[raw_idx] = (str(sf.resolve()), [], 0)
                                    fn, t_list, b_sz = single_blocks_temp[raw_idx]
                                    t_list.append(t_name)
                                    single_blocks_temp[raw_idx] = (fn, t_list, b_sz + t_bytes)
                                else:
                                    if raw_idx not in blocks:
                                        blocks[raw_idx] = BlockMetadata(
                                            block_id=raw_idx,
                                            filename=str(sf.resolve()),
                                            tensor_names=[],
                                            byte_size=0,
                                        )
                                    blocks[raw_idx].tensor_names.append(t_name)
                                    blocks[raw_idx].byte_size += t_bytes
                            break

        # Append single_blocks after double_blocks so they don't overwrite double_blocks[0..N]
        if single_blocks_temp:
            offset = (max(blocks.keys()) + 1) if blocks else 0
            for raw_idx in sorted(single_blocks_temp.keys()):
                fn, t_list, b_sz = single_blocks_temp[raw_idx]
                b_id = offset + raw_idx
                blocks[b_id] = BlockMetadata(
                    block_id=b_id,
                    filename=fn,
                    tensor_names=t_list,
                    byte_size=b_sz,
                )

        manifest = ModelShardsManifest(
            model_name=self.model_dir.stem if self.model_dir.is_file() else self.model_dir.name,
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
            with MMapTensorLoader(Path(meta.filename)) as loader:
                block_tensors = {t_name: loader.read_tensor(t_name, copy=True) for t_name in meta.tensor_names}

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

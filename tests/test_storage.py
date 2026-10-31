"""Tests for mmap tensor loader and storage sharding."""

from pathlib import Path
import numpy as np
import pytest
import safetensors.numpy
from phobos_stream.storage.layout import BlockMetadata, ModelShardsManifest
from phobos_stream.storage.mmap_loader import MMapTensorLoader
from phobos_stream.storage.sharder import DiffusionSharder


def test_mmap_tensor_loader_safetensors(tmp_path: Path) -> None:
    test_file = tmp_path / "test_block.safetensors"
    tensors = {
        "weight_a": np.random.randn(8, 16).astype(np.float32),
        "weight_b": np.random.randn(16, 8).astype(np.float32),
    }
    safetensors.numpy.save_file(tensors, str(test_file))

    with MMapTensorLoader(test_file) as loader:
        names = loader.get_tensor_names()
        assert "weight_a" in names
        assert "weight_b" in names

        arr_a = loader.read_tensor("weight_a")
        assert arr_a.shape == (8, 16)
        assert np.allclose(arr_a, tensors["weight_a"])


def test_manifest_serialization(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = ModelShardsManifest(
        model_name="test-dit",
        architecture="wan2.1",
        num_blocks=2,
        hidden_size=1024,
        blocks={
            0: BlockMetadata(block_id=0, filename="block_00.safetensors", tensor_names=["a"], byte_size=100),
            1: BlockMetadata(block_id=1, filename="block_01.safetensors", tensor_names=["b"], byte_size=100),
        },
    )
    manifest.save(manifest_path)
    loaded = ModelShardsManifest.load(manifest_path)

    assert loaded.model_name == "test-dit"
    assert loaded.num_blocks == 2
    assert 0 in loaded.blocks
    assert loaded.blocks[0].filename == "block_00.safetensors"

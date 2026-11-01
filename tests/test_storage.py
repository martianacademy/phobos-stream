"""Tests for mmap tensor loader, FP8/BF16 decoding, and storage sharding."""

from pathlib import Path
import struct
import json
import numpy as np
import pytest
import safetensors.numpy
from phobos_stream.storage.layout import BlockMetadata, ModelShardsManifest
from phobos_stream.storage.mmap_loader import MMapTensorLoader
from phobos_stream.storage.sharder import DiffusionSharder


def _write_raw_safetensors(path: Path, dtype_str: str, shape: list, raw_bytes: bytes) -> None:
    header = {
        "tensor_x": {
            "dtype": dtype_str,
            "shape": shape,
            "data_offsets": [0, len(raw_bytes)],
        }
    }
    header_json = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        f.write(raw_bytes)


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
        assert loader.get_tensor_byte_size("weight_a") == 8 * 16 * 4

        arr_a = loader.read_tensor("weight_a")
        assert arr_a.shape == (8, 16)
        assert np.allclose(arr_a, tensors["weight_a"])


def test_mmap_fp8_and_bf16_decoding(tmp_path: Path) -> None:
    # BF16 test: 1.0f has IEEE754 float32 bits 0x3F800000 -> BF16 uint16 is 0x3F80
    bf16_file = tmp_path / "bf16.safetensors"
    bf16_raw = np.array([0x3F80, 0xC000], dtype=np.uint16).tobytes()  # [1.0, -2.0]
    _write_raw_safetensors(bf16_file, "BF16", [2], bf16_raw)

    with MMapTensorLoader(bf16_file) as loader:
        arr = loader.read_tensor("tensor_x")
        assert arr.dtype == np.float32
        assert np.allclose(arr, np.array([1.0, -2.0], dtype=np.float32))

    # F8_E4M3 test: 0x38 is sign=0, exp=7 (2^0), mant=0 -> 1.0
    fp8_file = tmp_path / "fp8.safetensors"
    fp8_raw = np.array([0x38, 0xB8], dtype=np.uint8).tobytes()  # [1.0, -1.0]
    _write_raw_safetensors(fp8_file, "F8_E4M3", [2], fp8_raw)

    with MMapTensorLoader(fp8_file) as loader:
        arr = loader.read_tensor("tensor_x")
        assert arr.dtype == np.float32
        assert np.allclose(arr, np.array([1.0, -1.0], dtype=np.float32))


def test_sharder_double_and_single_blocks(tmp_path: Path) -> None:
    model_file = tmp_path / "flux_mock.safetensors"
    tensors = {
        "double_blocks.0.img_attn.q_proj.weight": np.ones((16, 16), dtype=np.float32),
        "double_blocks.1.img_attn.q_proj.weight": np.ones((16, 16), dtype=np.float32),
        "single_blocks.0.linear1.q.weight": np.ones((16, 16), dtype=np.float32),
    }
    safetensors.numpy.save_file(tensors, str(model_file))

    sharder = DiffusionSharder(model_dir=tmp_path, output_dir=tmp_path / "shards")
    manifest = sharder.build_virtual_manifest(architecture="flux", hidden_size=16)
    # 2 double_blocks + 1 single_block = 3 distinct blocks
    assert manifest.num_blocks == 3
    assert manifest.blocks[0].byte_size > 0
    assert manifest.blocks[2].tensor_names == ["single_blocks.0.linear1.q.weight"]


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

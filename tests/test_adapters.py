"""Tests for architecture adapters and strict non-synthetic weight enforcement."""

import numpy as np
import pytest
from phobos_stream.adapters.cogvideo_adapter import CogVideoXAdapter
from phobos_stream.adapters.factory import get_adapter
from phobos_stream.adapters.flux_adapter import FluxAdapter
from phobos_stream.adapters.hunyuan_adapter import HunyuanAdapter
from phobos_stream.adapters.wan_adapter import WanAdapter
from phobos_stream.backends.gguf_backend import GGUFBackend


def test_adapter_factory() -> None:
    backend = GGUFBackend(device="cpu")
    wan = get_adapter("wan2.1", backend)
    flux = get_adapter("flux", backend)
    hunyuan = get_adapter("hunyuan", backend)
    cogvideo = get_adapter("cogvideox", backend)

    assert isinstance(wan, WanAdapter)
    assert isinstance(flux, FluxAdapter)
    assert isinstance(hunyuan, HunyuanAdapter)
    assert isinstance(cogvideo, CogVideoXAdapter)


def test_strict_adapter_rejects_missing_weights_by_default() -> None:
    backend = GGUFBackend(device="cpu")
    strict_adapter = WanAdapter(backend, hidden_size=32)
    with pytest.raises(KeyError, match="allow_synthetic=False"):
        strict_adapter.extract_block_weights(0, {})


def test_wan_adapter_extract_weights_with_explicit_synthetic_opt_in() -> None:
    backend = GGUFBackend(device="cpu")
    adapter = WanAdapter(backend, hidden_size=64, allow_synthetic=True)

    raw_tensors = {
        "blocks.0.self_attn.q_proj.weight": np.random.randn(64, 64).astype(np.float32),
        "blocks.0.ffn.gate_proj.weight": np.random.randn(64, 128).astype(np.float32),
    }

    block_weights = adapter.extract_block_weights(0, raw_tensors)
    assert block_weights.self_attn.q_proj.shape == (64, 64)
    assert block_weights.mlp.gate_proj.shape == (64, 128)
    assert np.any(block_weights.self_attn.k_proj != 0.0)

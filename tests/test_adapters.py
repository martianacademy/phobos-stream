"""Tests for architecture adapters (Wan2.1, FLUX.1, HunyuanVideo)."""

import numpy as np
import pytest
from aether_stream.adapters.factory import get_adapter
from aether_stream.adapters.flux_adapter import FluxAdapter
from aether_stream.adapters.hunyuan_adapter import HunyuanAdapter
from aether_stream.adapters.wan_adapter import WanAdapter
from aether_stream.backends.gguf_backend import GGUFBackend


def test_adapter_factory() -> None:
    backend = GGUFBackend(device="cpu")
    wan = get_adapter("wan2.1", backend)
    flux = get_adapter("flux", backend)
    hunyuan = get_adapter("hunyuan", backend)

    assert isinstance(wan, WanAdapter)
    assert isinstance(flux, FluxAdapter)
    assert isinstance(hunyuan, HunyuanAdapter)


def test_wan_adapter_extract_weights() -> None:
    backend = GGUFBackend(device="cpu")
    adapter = WanAdapter(backend)

    raw_tensors = {
        "blocks.0.self_attn.q_proj.weight": np.random.randn(64, 64).astype(np.float32),
        "blocks.0.ffn.gate_proj.weight": np.random.randn(64, 128).astype(np.float32),
    }

    block_weights = adapter.extract_block_weights(0, raw_tensors)
    assert block_weights.self_attn.q_proj.shape == (64, 64)
    assert block_weights.mlp.gate_proj.shape == (64, 128)

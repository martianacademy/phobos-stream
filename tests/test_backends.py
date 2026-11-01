"""Tests for universal execution backends."""

import numpy as np
import pytest
from phobos_stream.backends.factory import get_backend
from phobos_stream.backends.gguf_backend import GGUFBackend
from phobos_stream.types import BackendType


def test_gguf_backend_basic_math() -> None:
    backend = GGUFBackend(device="cpu")
    assert backend.name == "gguf"

    # Zeros
    z = backend.zeros((2, 4), dtype="float32")
    assert z.shape == (2, 4)
    assert np.all(z == 0)

    # Random
    r = backend.randn((3, 3), seed=42)
    assert r.shape == (3, 3)

    # Matmul
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    c = backend.matmul(a, b)
    expected = np.matmul(a, b)
    assert np.allclose(c, expected)

    # Linear with PyTorch-style (out_features, in_features) weight
    x = np.random.randn(2, 4, 8).astype(np.float32)
    w_pt = np.random.randn(16, 8).astype(np.float32)  # (out=16, in=8)
    out_lin = backend.linear(x, w_pt)
    assert out_lin.shape == (2, 4, 16)
    assert np.allclose(out_lin, x @ w_pt.T)

    # LayerNorm
    x_ln = np.random.randn(2, 4).astype(np.float32)
    w_ln = np.ones(4, dtype=np.float32)
    normed = backend.layer_norm(x_ln, w_ln)
    assert normed.shape == (2, 4)

    # SiLU
    silu_val = backend.silu(np.array([0.0, 2.0], dtype=np.float32))
    assert np.isclose(silu_val[0], 0.0)

    # Q8_0 roundtrip quantization/dequantization
    orig = np.linspace(-3.0, 3.0, 64, dtype=np.float32)
    q8_bytes = backend.quantize_q8_0(orig)
    deq = backend.dequantize_q8_0(q8_bytes, 64)
    assert deq.shape == (64,)
    assert np.allclose(orig, deq, atol=0.06)

    # Real RSS memory telemetry (> 0 MB)
    assert backend.get_memory_allocated_mb() > 0.0


def test_attention_scaled_dot_product() -> None:
    backend = GGUFBackend(device="cpu")
    q = np.random.randn(1, 2, 4, 8).astype(np.float32)
    k = np.random.randn(1, 2, 4, 8).astype(np.float32)
    v = np.random.randn(1, 2, 4, 8).astype(np.float32)

    attn_out = backend.scaled_dot_product_attention(q, k, v)
    assert attn_out.shape == (1, 2, 4, 8)


def test_backend_factory_auto() -> None:
    backend = get_backend(BackendType.AUTO)
    assert backend is not None
    assert backend.name in ("mlx", "cuda", "gguf")

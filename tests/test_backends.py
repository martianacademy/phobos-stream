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

    # LayerNorm
    x = np.random.randn(2, 4).astype(np.float32)
    w = np.ones(4, dtype=np.float32)
    normed = backend.layer_norm(x, w)
    assert normed.shape == (2, 4)

    # SiLU
    silu_val = backend.silu(np.array([0.0, 2.0], dtype=np.float32))
    assert np.isclose(silu_val[0], 0.0)


def test_attention_scaled_dot_product() -> None:
    backend = GGUFBackend(device="cpu")
    q = np.random.randn(1, 4, 8).astype(np.float32)
    k = np.random.randn(1, 4, 8).astype(np.float32)
    v = np.random.randn(1, 4, 8).astype(np.float32)

    attn_out = backend.scaled_dot_product_attention(q, k, v)
    assert attn_out.shape == (1, 4, 8)


def test_backend_factory_auto() -> None:
    backend = get_backend(BackendType.AUTO)
    assert backend is not None
    assert backend.name in ("mlx", "cuda", "gguf")

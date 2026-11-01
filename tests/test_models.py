"""Tests for 3D RoPE, multi-head attention, and DiT transformer blocks."""

import numpy as np
import pytest
from phobos_stream.backends.gguf_backend import GGUFBackend
from phobos_stream.models.attention import AttentionWeights, DiTAttention
from phobos_stream.models.dit_block import DiTBlock, DiTBlockWeights, MLPWeights
from phobos_stream.models.rotary import RotaryEmbedding3D


def test_rotary_embedding_3d() -> None:
    backend = GGUFBackend(device="cpu")
    rotary = RotaryEmbedding3D(dim=64, backend=backend)
    cos, sin = rotary.compute_frequencies(t_len=2, h_len=4, w_len=4)

    # 2 * 4 * 4 = 32 tokens
    assert cos.shape == (32, 64)
    assert sin.shape == (32, 64)

    # Verify _rotate_half rotates 2D pairs: (x0, x1) -> (-x1, x0), preserving norm
    vec = np.array([[[3.0, 4.0, 1.0, -2.0]]], dtype=np.float32)
    rot = rotary._rotate_half(vec)
    assert np.allclose(rot, np.array([[[-4.0, 3.0, 2.0, 1.0]]], dtype=np.float32))


def test_dit_block_forward_and_timestep_conditioning() -> None:
    backend = GGUFBackend(device="cpu")
    dim = 64
    num_heads = 4

    block = DiTBlock(hidden_size=dim, num_heads=num_heads, backend=backend)

    self_attn = AttentionWeights(
        q_proj=np.eye(dim, dtype=np.float32),
        k_proj=np.eye(dim, dtype=np.float32),
        v_proj=np.eye(dim, dtype=np.float32),
        out_proj=np.eye(dim, dtype=np.float32),
    )
    cross_attn = AttentionWeights(
        q_proj=np.eye(dim, dtype=np.float32),
        k_proj=np.eye(dim, dtype=np.float32),
        v_proj=np.eye(dim, dtype=np.float32),
        out_proj=np.eye(dim, dtype=np.float32),
    )
    # Test PyTorch-style (out_features, in_features) non-square MLP weights
    mlp = MLPWeights(
        gate_proj=np.ones((dim * 2, dim), dtype=np.float32) * 0.05,
        up_proj=np.ones((dim * 2, dim), dtype=np.float32) * 0.05,
        down_proj=np.ones((dim, dim * 2), dtype=np.float32) * 0.05,
    )

    weights = DiTBlockWeights(
        norm1_weight=np.ones(dim, dtype=np.float32),
        self_attn=self_attn,
        norm2_weight=np.ones(dim, dtype=np.float32),
        cross_attn=cross_attn,
        norm3_weight=np.ones(dim, dtype=np.float32),
        mlp=mlp,
    )

    x = np.random.randn(1, 16, dim).astype(np.float32)
    context = np.random.randn(1, 8, dim).astype(np.float32)
    t_emb_1 = np.ones((dim,), dtype=np.float32) * 0.1
    t_emb_2 = np.ones((dim,), dtype=np.float32) * 2.5

    out1 = block.forward(x, weights, context=context, timestep_emb=t_emb_1)
    out2 = block.forward(x, weights, context=context, timestep_emb=t_emb_2)
    assert out1.shape == (1, 16, dim)
    assert not np.allclose(out1, out2)

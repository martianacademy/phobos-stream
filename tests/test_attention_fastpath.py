"""Unit tests for PhobosAttentionFastpath and query_chunked_sdpa."""

from __future__ import annotations

import pytest

from phobos_stream.streaming.attention_fastpath import (
    PhobosAttentionFastpath,
    _is_sage2_version,
    query_chunked_sdpa,
)


def test_query_chunked_sdpa_is_mathematically_exact() -> None:
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    torch.manual_seed(42)
    q = torch.randn(2, 4, 600, 64)
    k = torch.randn(2, 4, 600, 64)
    v = torch.randn(2, 4, 600, 64)

    expected = F.scaled_dot_product_attention(q, k, v)
    actual = query_chunked_sdpa(F.scaled_dot_product_attention, q, k, v, chunk_size=128)
    assert actual.shape == expected.shape
    assert torch.allclose(expected, actual, atol=1e-6)

    # Verify per-query mask slicing [B, 1, Q, K]
    mask = torch.zeros(2, 1, 600, 600)
    mask[:, :, :, 500:] = -1e4
    expected_m = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    actual_m = query_chunked_sdpa(F.scaled_dot_product_attention, q, k, v, attn_mask=mask, chunk_size=150)
    assert torch.allclose(expected_m, actual_m, atol=1e-5)


def test_attention_fastpath_context_installs_and_cleanly_restores_sdpa() -> None:
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    orig_fn = F.scaled_dot_product_attention
    q = torch.randn(1, 2, 300, 32)
    k = torch.randn(1, 2, 300, 32)
    v = torch.randn(1, 2, 300, 32)
    expected = orig_fn(q, k, v)

    with PhobosAttentionFastpath(mode="chunked_sdpa", query_chunk_size=100) as fp:
        assert F.scaled_dot_product_attention is not orig_fn
        out = F.scaled_dot_product_attention(q, k, v)
        assert fp.chunked_calls == 1
        assert torch.allclose(expected, out, atol=1e-6)

    assert F.scaled_dot_product_attention is orig_fn


def test_sageattention_version_gate_rejects_v1_triton_and_accepts_v2_cuda() -> None:
    assert not _is_sage2_version("1.0.6")
    assert not _is_sage2_version("0.9.0")
    assert not _is_sage2_version("")
    assert _is_sage2_version("2.2.0")
    assert _is_sage2_version("2.2.0+cu128")
    fp = PhobosAttentionFastpath(mode="sage2")
    assert fp.enabled

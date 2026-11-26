"""Unit tests for Apple Silicon MLXBlockStreamer Hybrid Residency & First-Block Residual Cache."""

from pathlib import Path
from typing import Any
import numpy as np
import pytest

from phobos_stream.streaming import MLXBlockStreamer


def test_mlx_block_streamer_hybrid_budget_and_first_block_cache(tmp_path: Path) -> None:
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    from mlx.utils import tree_flatten

    class SmoothMLXBlock(nn.Module):
        def __init__(self, dim: int = 64) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.w1 = nn.Linear(dim, dim * 2, bias=False)
            self.w2 = nn.Linear(dim * 2, dim, bias=False)

        def __call__(self, x: Any, t_scale: Any) -> Any:
            h = self.norm(x) * t_scale
            return x + 0.08 * self.w2(nn.silu(self.w1(h)))

    mx.random.seed(21)
    dim = 64
    num_blocks = 6
    blocks_ref = [SmoothMLXBlock(dim) for _ in range(num_blocks)]
    flat_weights = {}
    for idx, blk in enumerate(blocks_ref):
        blk.w1.weight = mx.random.normal(blk.w1.weight.shape).astype(mx.bfloat16) * 0.04
        blk.w2.weight = mx.random.normal(blk.w2.weight.shape).astype(mx.bfloat16) * 0.04
        blk.set_dtype(mx.bfloat16)
        mx.eval(blk.parameters())
        for k, v in tree_flatten(blk.parameters()):
            flat_weights[f"layers.{idx}.{k}"] = v

    st_path = tmp_path / "mlx_six_blocks_bf16.safetensors"
    mx.save_safetensors(str(st_path), flat_weights)

    # Compute exact single-block size in MB (~32 KB = 0.0315 MB)
    one_blk_bytes = sum(
        v.nbytes for k, v in flat_weights.items() if k.startswith("layers.0.")
    )
    # Budget for 2 resident blocks + 1 scratch block = 3 * one_blk_bytes
    budget_mb = (3.1 * one_blk_bytes) / (1024.0 * 1024.0)

    x0 = mx.random.normal((1, 12, dim)).astype(mx.bfloat16)
    mx.eval(x0)

    # 1. Reference trajectory
    x_ref = x0
    sigmas = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    for s in sigmas:
        t_sc = mx.array(1.0 + 0.05 * s, dtype=mx.bfloat16)
        cur = x_ref
        for blk in blocks_ref:
            cur = blk(cur, t_sc)
        x_ref = x_ref - 0.02 * cur
        mx.eval(x_ref)

    # 2. Hybrid Residency (`memory_budget_mb` keeps 2 resident + streams 4 from mmap) — bit-exact!
    blocks_hyb = [SmoothMLXBlock(dim) for _ in range(num_blocks)]
    for blk in blocks_hyb:
        blk.set_dtype(mx.bfloat16)

    with MLXBlockStreamer(
        blocks_hyb,
        [st_path],
        block_prefix_template="layers.{i}",
        memory_budget_mb=budget_mb,
        cache_rel_l1_thresh=0.0,
    ) as s_hyb:
        assert s_hyb.tier_summary == {"metal_resident": 2, "nvme_mmap": 4}
        # Resident blocks (0, 1) stay materialized; overflow blocks (2..5) start evicted!
        for _, arr in tree_flatten(blocks_hyb[0].parameters()):
            assert arr.size > 0
        for _, arr in tree_flatten(blocks_hyb[3].parameters()):
            assert arr.size == 0

        x_hyb = x0
        for s in sigmas:
            t_sc = mx.array(1.0 + 0.05 * s, dtype=mx.bfloat16)
            vel = s_hyb.run_sequence(x_hyb, t_sc)
            x_hyb = x_hyb - 0.02 * vel
            mx.eval(x_hyb)

        assert s_hyb.blocks_executed == len(sigmas) * num_blocks
        # Only 4 of 6 blocks streamed per step!
        assert s_hyb.bytes_streamed == len(sigmas) * 4 * one_blk_bytes

    assert np.allclose(
        np.array(x_ref.astype(mx.float32)),
        np.array(x_hyb.astype(mx.float32)),
        atol=1e-4,
    )

    # 3. Hybrid Residency + Zero-MMap First-Block Residual Cache (`cache_rel_l1_thresh=0.18`)
    blocks_cached = [SmoothMLXBlock(dim) for _ in range(num_blocks)]
    for blk in blocks_cached:
        blk.set_dtype(mx.bfloat16)

    with MLXBlockStreamer(
        blocks_cached,
        [st_path],
        block_prefix_template="layers.{i}",
        memory_budget_mb=budget_mb,
        cache_rel_l1_thresh=0.18,
        cache_warmup_steps=2,
        cache_max_consecutive_skips=2,
    ) as s_cache:
        x_fast = x0
        for s in sigmas:
            t_sc = mx.array(1.0 + 0.05 * s, dtype=mx.bfloat16)
            vel = s_cache.run_sequence(x_fast, t_sc)
            x_fast = x_fast - 0.02 * vel
            mx.eval(x_fast)

        assert s_cache.skipped_steps >= 3
        # Because block 0 is Metal-resident, skipped steps perform ZERO mmap bytes!
        assert s_cache.bytes_streamed == s_cache.computed_steps * 4 * one_blk_bytes

    a = np.array(x_ref.astype(mx.float32)).ravel()
    b = np.array(x_fast.astype(mx.float32)).ravel()
    cos_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    assert cos_sim > 0.999, f"Expected >0.999 cosine similarity on MLX FBCache, got {cos_sim}"

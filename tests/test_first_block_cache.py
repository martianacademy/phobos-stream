"""Unit tests for FirstBlockResidualCache (TeaCache / FBCache) and PhobosFirstBlockCacheNode."""

from __future__ import annotations

import pytest

from phobos_stream.integrations.comfyui.cache_node import PhobosFirstBlockCacheNode
from phobos_stream.integrations.comfyui.nodes import PhobosBlockStreamPatcherNode
from phobos_stream.streaming import BlockStreamMode, FirstBlockResidualCache, TorchBlockStreamer


def _make_smooth_dit(dim: int = 32, num_blocks: int = 8):
    import torch
    import torch.nn as nn

    torch.manual_seed(7)

    class _SmoothDiTBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(dim, dim)
            nn.init.normal_(self.lin.weight, mean=0.0, std=0.04)
            nn.init.zeros_(self.lin.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + 0.1 * torch.tanh(self.lin(x))

    class _SmoothDiT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Linear(dim, dim)
            self.blocks = nn.ModuleList([_SmoothDiTBlock() for _ in range(num_blocks)])
            self.head = nn.Linear(dim, dim)

        def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            t_scale = 1.0 + 0.05 * timestep.view(-1, 1, 1)
            h = self.stem(x) * t_scale
            for blk in self.blocks:
                h = blk(h)
            return self.head(h)

    return _SmoothDiT().eval()


def test_first_block_cache_skips_blocks_and_dma_with_high_fidelity() -> None:
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    dit_ref = _make_smooth_dit(dim=32, num_blocks=8)
    dit_cached = _make_smooth_dit(dim=32, num_blocks=8)
    dit_cached.load_state_dict(dit_ref.state_dict())

    sigmas = torch.linspace(1.0, 0.1, steps=10)
    torch.manual_seed(123)
    x0 = torch.randn(1, 16, 32)

    # 1. Uncached streamed baseline
    x_ref = x0.clone()
    with TorchBlockStreamer(dit_ref, compute_device="cpu", mode=BlockStreamMode.CPU_OFFLOAD) as ref_s:
        with torch.no_grad():
            for s in sigmas:
                vel = dit_ref(x_ref, s.unsqueeze(0))
                x_ref = x_ref - 0.02 * vel
        ref_blocks = ref_s.blocks_executed
        ref_bytes = ref_s.bytes_streamed

    assert ref_blocks == 10 * 8

    # 2. Streamed + First-Block Residual Cache
    x_fast = x0.clone()
    with TorchBlockStreamer(
        dit_cached,
        compute_device="cpu",
        mode=BlockStreamMode.CPU_OFFLOAD,
        cache_rel_l1_thresh=0.15,
        cache_warmup_steps=2,
        cache_max_consecutive_skips=2,
    ) as fast_s:
        with torch.no_grad():
            for s in sigmas:
                vel = dit_cached(x_fast, s.unsqueeze(0))
                x_fast = x_fast - 0.02 * vel
        assert fast_s.residual_cache is not None
        skipped = fast_s.residual_cache.skipped_steps
        fast_blocks = fast_s.blocks_executed
        fast_bytes = fast_s.bytes_streamed

    assert skipped >= 4, f"Expected >=4 skipped steps out of 10, got {skipped}"
    # Each skipped step executes ONLY block 0 (saving 7 of 8 blocks AND their DMA transfers!)
    assert fast_blocks == (10 - skipped) * 8 + skipped * 1
    assert fast_bytes < ref_bytes * 0.70

    cos_sim = float(F.cosine_similarity(x_ref.flatten(), x_fast.flatten(), dim=0).item())
    assert cos_sim > 0.999, f"Expected >0.999 cosine similarity, got {cos_sim}"


def test_first_block_cache_auto_resets_on_new_trajectory_and_isolates_cfg_branches() -> None:
    torch = pytest.importorskip("torch")

    dit = _make_smooth_dit(dim=16, num_blocks=4)
    x_a = torch.randn(1, 8, 16)
    x_b = torch.randn(1, 8, 16) * 3.0

    with torch.no_grad():
        exact_b_step0 = dit(x_b, torch.tensor([1.0]))

    cache = FirstBlockResidualCache(rel_l1_thresh=0.20, warmup_steps=1, max_consecutive_skips=3).attach(dit)
    try:
        with torch.no_grad():
            # Trajectory 1 with CFG (cond + uncond at identical sigma)
            for s_val in (1.0, 0.8, 0.6):
                t = torch.tensor([s_val])
                _ = dit(x_a, t)          # branch 0 (cond)
                _ = dit(-x_a, t)         # branch 1 (uncond at same sigma!)
            assert len(cache._slots) == 2
            assert cache.skipped_steps >= 2

            # Trajectory 2 starts at t=1.0 (> 0.6) -> must auto-reset and compute exact step 0!
            out_b_step0 = dit(x_b, torch.tensor([1.0]))
            assert torch.allclose(exact_b_step0, out_b_step0, atol=1e-6)
    finally:
        cache.detach()


def test_comfy_first_block_cache_node_chained_and_standalone() -> None:
    torch = pytest.importorskip("torch")

    class _MockPatcher:
        def __init__(self, mod) -> None:
            self.model = mod

        def clone(self) -> "_MockPatcher":
            cloned = _MockPatcher(self.model)
            if hasattr(self, "_phobos_block_streamer"):
                cloned._phobos_block_streamer = self._phobos_block_streamer
            return cloned

        def get_model_object(self, name: str):
            return self.model if name == "diffusion_model" else None

    dit = _make_smooth_dit(dim=16, num_blocks=4)
    p0 = _MockPatcher(dit)

    # Chained: PhobosBlockStreamPatcher -> PhobosFirstBlockCache
    p1, = PhobosBlockStreamPatcherNode().patch_model(p0, mode="cpu_offload")
    p2, = PhobosFirstBlockCacheNode().apply_cache(p1, rel_l1_thresh=0.15, warmup_steps=1)
    streamer = getattr(p2, "_phobos_block_streamer")
    assert streamer.residual_cache is not None

    dev = streamer.compute_device
    x = torch.randn(1, 4, 16, device=dev)
    try:
        with torch.no_grad():
            for s in (1.0, 0.9, 0.8):
                _ = dit(x, torch.tensor([s], device=dev))
        assert streamer.residual_cache.skipped_steps >= 1
    finally:
        streamer.detach()

    # Standalone (no block streamer)
    dit2 = _make_smooth_dit(dim=16, num_blocks=4)
    p_solo, = PhobosFirstBlockCacheNode().apply_cache(_MockPatcher(dit2), rel_l1_thresh=0.15, warmup_steps=1)
    solo_cache = getattr(p_solo, "_phobos_residual_cache")
    assert solo_cache is not None
    x_cpu = torch.randn(1, 4, 16)
    try:
        with torch.no_grad():
            for s in (1.0, 0.9, 0.8):
                _ = dit2(x_cpu, torch.tensor([s]))
        assert solo_cache.skipped_steps >= 1
    finally:
        solo_cache.detach()


def test_flux_mmdit_dual_stream_and_single_stream_multi_group_cache() -> None:
    """Verify FLUX/Hunyuan MMDiT (`double_blocks` kwargs `(img, txt)` + `cat` + `single_blocks`)."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(11)

    class _DoubleBlock(nn.Module):
        def __init__(self, dim: int = 24) -> None:
            super().__init__()
            self.img_lin = nn.Linear(dim, dim)
            self.txt_lin = nn.Linear(dim, dim)
            nn.init.normal_(self.img_lin.weight, 0.0, 0.03)
            nn.init.normal_(self.txt_lin.weight, 0.0, 0.03)

        def forward(self, *, img: torch.Tensor, txt: torch.Tensor, vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # Mutate in-place (`+=`) exactly like ComfyUI `comfy/ldm/flux/layers.py` `DoubleStreamBlock`!
            img += 0.1 * torch.tanh(self.img_lin(img) + vec)
            txt += 0.1 * torch.tanh(self.txt_lin(txt))
            return img, txt

    class _SingleBlock(nn.Module):
        def __init__(self, dim: int = 24) -> None:
            super().__init__()
            self.lin = nn.Linear(dim, dim)
            nn.init.normal_(self.lin.weight, 0.0, 0.03)

        def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
            # Mutate in-place (`+=`) exactly like ComfyUI `SingleStreamBlock`!
            x += 0.1 * torch.tanh(self.lin(x) + vec)
            return x

    class _MiniFluxMMDiT(nn.Module):
        def __init__(self, dim: int = 24) -> None:
            super().__init__()
            self.img_in = nn.Linear(dim, dim)
            self.txt_in = nn.Linear(dim, dim)
            self.double_blocks = nn.ModuleList([_DoubleBlock(dim) for _ in range(4)])
            self.single_blocks = nn.ModuleList([_SingleBlock(dim) for _ in range(6)])
            self.final_layer = nn.Linear(dim, dim)

        def forward(self, x: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
            vec = (1.0 + 0.04 * timestep.view(-1, 1, 1)).expand(-1, 1, x.shape[-1])
            img = self.img_in(x)
            txt = self.txt_in(context)
            for blk in self.double_blocks:
                img, txt = blk(img=img, txt=txt, vec=vec)
            cat_x = torch.cat((txt, img), dim=1)
            for blk in self.single_blocks:
                cat_x = blk(cat_x, vec=vec)
            img_out = cat_x[:, txt.shape[1] :, :]
            return self.final_layer(img_out)

    flux_ref = _MiniFluxMMDiT(dim=24).eval()
    flux_fast = _MiniFluxMMDiT(dim=24).eval()
    flux_fast.load_state_dict(flux_ref.state_dict())

    sigmas = torch.linspace(1.0, 0.1, steps=8)
    x0 = torch.randn(1, 12, 24)
    ctx = torch.randn(1, 6, 24)

    x_ref = x0.clone()
    with TorchBlockStreamer(flux_ref, compute_device="cpu", mode=BlockStreamMode.CPU_OFFLOAD) as ref_s:
        with torch.no_grad():
            for s in sigmas:
                x_ref = x_ref - 0.02 * flux_ref(x_ref, s.unsqueeze(0), ctx)
        assert ref_s.blocks_executed == 8 * 10  # 4 double + 6 single = 10 blocks/step

    x_fast = x0.clone()
    with TorchBlockStreamer(
        flux_fast,
        compute_device="cpu",
        mode=BlockStreamMode.CPU_OFFLOAD,
        cache_rel_l1_thresh=0.18,
        cache_warmup_steps=2,
        cache_max_consecutive_skips=2,
    ) as fast_s:
        with torch.no_grad():
            for s in sigmas:
                x_fast = x_fast - 0.02 * flux_fast(x_fast, s.unsqueeze(0), ctx)
        assert fast_s.residual_cache is not None
        skipped = fast_s.residual_cache.skipped_steps
        # On every skipped step, ONLY double_blocks[0] executes (9 of 10 blocks + DMA skipped!)
        assert skipped >= 3
        assert fast_s.blocks_executed == (8 - skipped) * 10 + skipped * 1
        assert fast_s.bytes_streamed < ref_s.bytes_streamed * 0.72

    cos = float(F.cosine_similarity(x_ref.flatten(), x_fast.flatten(), dim=0).item())
    assert cos > 0.999, f"Expected >0.999 cosine similarity on MMDiT, got {cos}"

"""Unit tests for Phase 1 (`PhobosCLIPStreamPatcherNode` + `PhobosZeroRAMLoader` meta helpers)."""

from __future__ import annotations

from pathlib import Path
import pytest

from phobos_stream.integrations.comfyui.clip_patcher import PhobosCLIPStreamPatcherNode
from phobos_stream.integrations.comfyui.zero_ram_loader import (
    build_header_only_meta_state_dict,
    materialize_non_block_stem_from_mmap,
)
from phobos_stream.streaming import BlockStreamMode, TorchBlockStreamer


def test_clip_stream_patcher_dual_encoder_and_cpu_vocab_wrapper() -> None:
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    target_dev = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"

    class _MiniSubEncoder(nn.Module):
        def __init__(self, vocab: int, dim: int, n_layers: int) -> None:
            super().__init__()
            self.shared = nn.Embedding(vocab, dim)
            self.block = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
            self.final_norm = nn.LayerNorm(dim)

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            h = self.shared(ids)
            for blk in self.block:
                h = h + 0.1 * torch.tanh(blk(h))
            return self.final_norm(h)

    class _DualCondStage(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # 16,384 * 64 * 4 bytes = 4 MB (> 1 MB test threshold when triggered via patcher)
            self.clip_l = _MiniSubEncoder(vocab=512, dim=32, n_layers=3)
            self.t5xxl = _MiniSubEncoder(vocab=1024, dim=32, n_layers=4)

        def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return self.clip_l(ids), self.t5xxl(ids)

    class _MockPatcher:
        def __init__(self, mod: nn.Module) -> None:
            self.model = mod
            self.load_device = torch.device(target_dev)
            self.patches = {}

        def clone(self) -> "_MockPatcher":
            return _MockPatcher(self.model)

        def load(self, *a, **kw) -> None:
            self.model.to(self.load_device)

        def patch_model(self, *a, **kw):
            return self.model

        def model_size(self) -> int:
            return 1024

    class _MockCLIP:
        def __init__(self, cond: _DualCondStage) -> None:
            self.cond_stage_model = cond
            self.patcher = _MockPatcher(cond)

        def clone(self) -> "_MockCLIP":
            c = _MockCLIP(self.cond_stage_model)
            c.patcher = self.patcher.clone()
            return c

        def load_model(self, *a, **kw):
            self.patcher.load()
            return self.patcher

        def encode_from_tokens(self, tokens: torch.Tensor):
            self.load_model(tokens)
            dev = self.patcher.load_device
            return self.cond_stage_model(tokens.to(dev))

    torch.manual_seed(17)
    cond_ref = _DualCondStage().eval()
    cond_stream = _DualCondStage().eval()
    cond_stream.load_state_dict(cond_ref.state_dict())

    tokens = torch.tensor([[3, 19, 42, 7, 101]], dtype=torch.long)
    with torch.no_grad():
        ref_l, ref_t5 = cond_ref.to(target_dev)(tokens.to(target_dev))

    patched_clip, = PhobosCLIPStreamPatcherNode().patch_clip(_MockCLIP(cond_stream), mode="cpu_offload")
    try:
        with torch.no_grad():
            out_l, out_t5 = patched_clip.encode_from_tokens(tokens)
        streamer = getattr(patched_clip, "_phobos_clip_streamer")
        # Both sub-encoders (3 + 4 = 7 blocks) must stream cleanly!
        assert streamer.blocks_executed == 7
        assert torch.allclose(ref_l.cpu(), out_l.cpu(), atol=1e-5)
        assert torch.allclose(ref_t5.cpu(), out_t5.cpu(), atol=1e-5)
        # After encode_from_tokens with evict_after_encode=True, norms move back to CPU!
        assert cond_stream.t5xxl.final_norm.weight.device.type == "cpu"
    finally:
        getattr(patched_clip, "_phobos_clip_streamer").detach()


def test_zero_ram_header_meta_state_dict_and_stem_materialization(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    st_torch = pytest.importorskip("safetensors.torch")
    import torch.nn as nn

    class _ZeroRAMTestDiT(nn.Module):
        def __init__(self, dim: int = 48) -> None:
            super().__init__()
            self.img_in = nn.Linear(16, dim)
            self.blocks = nn.ModuleList([nn.Linear(dim, dim) for _ in range(5)])
            self.final_layer = nn.Linear(dim, 16)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.img_in(x)
            for b in self.blocks:
                h = h + 0.1 * torch.nn.functional.silu(b(h))
            return self.final_layer(h)

    torch.manual_seed(33)
    ref_dit = _ZeroRAMTestDiT(dim=48).to(torch.bfloat16).eval()
    ckpt_path = tmp_path / "zero_ram_dit_bf16.safetensors"
    st_torch.save_file(ref_dit.state_dict(), str(ckpt_path), metadata={"format": "pt"})

    x = torch.randn(2, 8, 16, dtype=torch.bfloat16)
    with torch.no_grad():
        expected = ref_dit(x)

    # 1. Header-only meta state dict must keep large matrices on `meta` (0 bytes RAM!)
    meta_sd, meta_hdr = build_header_only_meta_state_dict(ckpt_path, scalar_threshold_elems=8)
    assert meta_hdr.get("format") == "pt"
    assert meta_sd["blocks.0.weight"].is_meta

    # 2. Instantiate on `meta`, materialize ONLY stem/head, and stream blocks via `MMAP_STREAM`
    with torch.device("meta"):
        meta_dit = _ZeroRAMTestDiT(dim=48).to(torch.bfloat16).eval()

    stem_bytes = materialize_non_block_stem_from_mmap(meta_dit, ckpt_path, target_device="cpu")
    assert stem_bytes > 0
    assert not meta_dit.img_in.weight.is_meta
    assert meta_dit.blocks[0].weight.is_meta

    with TorchBlockStreamer(
        meta_dit,
        compute_device="cpu",
        mode=BlockStreamMode.MMAP_STREAM,
        safetensors_paths=[ckpt_path],
    ) as streamer:
        assert meta_dit.blocks[0].weight.numel() == 0
        with torch.no_grad():
            actual = meta_dit(x)
        assert streamer.blocks_executed == 5
        assert meta_dit.blocks[0].weight.numel() == 0

    assert torch.allclose(expected, actual, atol=1e-4)


def test_hunyuan_repackaged_prefix_and_scale_key_remap(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    st_torch = pytest.importorskip("safetensors.torch")
    import torch.nn as nn

    class _HunyuanStyleModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.img_in = nn.Linear(8, 8)
            self.double_blocks = nn.ModuleList([nn.LayerNorm(8), nn.LayerNorm(8)])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.img_in(x)
            for b in self.double_blocks:
                h = b(h)
            return h

    ref = _HunyuanStyleModel().to(torch.bfloat16).eval()
    repackaged_sd = {}
    for k, v in ref.state_dict().items():
        rk = f"model.model.{k}"
        if rk.endswith(".weight") and "double_blocks" in rk:
            rk = f"{rk[:-7]}.scale"
        repackaged_sd[rk] = v

    ckpt_path = tmp_path / "hunyuan_repackaged.safetensors"
    st_torch.save_file(repackaged_sd, str(ckpt_path))

    x = torch.randn(1, 4, 8, dtype=torch.bfloat16)
    with torch.no_grad():
        expected = ref(x)

    with torch.device("meta"):
        meta_m = _HunyuanStyleModel().to(torch.bfloat16).eval()

    materialize_non_block_stem_from_mmap(meta_m, ckpt_path, target_device="cpu")
    with TorchBlockStreamer(
        meta_m,
        compute_device="cpu",
        mode=BlockStreamMode.MMAP_STREAM,
        safetensors_paths=[ckpt_path],
    ) as streamer:
        with torch.no_grad():
            actual = meta_m(x)
        assert streamer.blocks_executed == 2

    assert torch.equal(expected, actual)

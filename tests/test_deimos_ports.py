"""Unit tests for DeimosStream -> PhobosStream cross-pollination optimizations:
1. Compressed `PackedGGUFShadow` (`uint8`) in `CPU_OFFLOAD` & `HYBRID_BUDGET` RAM planning.
2. Page-aligned `advise_dontneed_mmap_range` (`MADV_DONTNEED`) + `MMapRingPrefetcher` disjoint eviction.
3. `@mx.compile` fused Apple Metal kernels + static Cross-Attention KV precomputation in `MLXWanVideoStreamer`.
"""

from __future__ import annotations

import mmap
from pathlib import Path
import struct
import pytest

from phobos_stream.backends.gguf_backend import GGUFBackend
from phobos_stream.streaming import BlockStreamMode, BlockTier, MemoryBudgetPlanner, TorchBlockStreamer
from phobos_stream.streaming.budget_planner import advise_dontneed_mmap_range
from phobos_stream.streaming.mmap_ring_prefetcher import PackedGGUFShadow


def _write_gguf_v3(path: Path, tensors: dict[str, tuple[int, tuple[int, ...], bytes]]) -> None:
    buf = bytearray(b"GGUF")
    buf.extend(struct.pack("<IQQ", 3, len(tensors), 0))
    blobs = bytearray()
    for name, (ggml_type, pt_shape, raw_bytes) in tensors.items():
        nb = name.encode("utf-8")
        buf.extend(struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(pt_shape)))
        for d in reversed(pt_shape):
            buf.extend(struct.pack("<Q", int(d)))
        buf.extend(struct.pack("<IQ", ggml_type, len(blobs)))
        blobs.extend(raw_bytes)
        if len(blobs) % 32:
            blobs.extend(b"\x00" * (32 - len(blobs) % 32))
    if len(buf) % 32:
        buf.extend(b"\x00" * (32 - len(buf) % 32))
    buf.extend(blobs)
    path.write_bytes(bytes(buf))


def test_advise_dontneed_mmap_range_safe_on_all_offsets(tmp_path: Path) -> None:
    p = tmp_path / "weights.bin"
    p.write_bytes(b"\xAB" * (64 * 1024))
    with open(p, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            _ = mm[100:40000]
            # Unaligned range spanning multiple pages + sub-page range (no-op, never raises)
            advise_dontneed_mmap_range(mm, 123, 49152)
            advise_dontneed_mmap_range(mm, 10, 200)
            assert mm[0] == 0xAB
        finally:
            mm.close()


def test_packed_gguf_shadow_cpu_offload_and_hybrid_compressed_budget(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    import torch.nn.functional as F

    dev = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"

    class _MiniDiT(nn.Module):
        def __init__(self, dim: int = 128) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(4)])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for blk in self.blocks:
                x = x + 0.15 * F.silu(blk(x))
            return x

    torch.manual_seed(7)
    ref = _MiniDiT(128).to(torch.bfloat16).eval()
    gguf = GGUFBackend()
    tensors = {}
    q8_total = 0
    bf16_total = 0
    for k, p in ref.state_dict().items():
        raw_q8 = gguf.quantize_q8_0(p.detach().float().cpu().numpy())
        tensors[k] = (8, tuple(p.shape), raw_q8)
        q8_total += len(raw_q8)
        bf16_total += p.numel() * 2

    gguf_path = tmp_path / "mini_q8_0.gguf"
    _write_gguf_v3(gguf_path, tensors)

    with torch.device("meta"):
        meta_m = _MiniDiT(128).to(torch.bfloat16).eval()

    x = torch.randn(2, 8, 128, dtype=torch.bfloat16, device=dev)
    with TorchBlockStreamer(
        meta_m,
        compute_device=dev,
        mode=BlockStreamMode.CPU_OFFLOAD,
        safetensors_paths=[gguf_path],
    ) as streamer:
        # Verify CPU shadow holds compressed `PackedGGUFShadow` (`~53%` of BF16!) instead of expanded BF16
        sh0 = streamer._cpu_shadow[0][0][2]
        assert isinstance(sh0, PackedGGUFShadow)
        assert sh0.nbytes < (128 * 128 * 2) * 0.60
        assert meta_m.blocks[0].weight.numel() == 0
        with torch.no_grad():
            out_cpu = meta_m(x)
        assert streamer.bytes_streamed == q8_total
        assert streamer.bytes_streamed < bf16_total * 0.60

    # Verify MMAP_STREAM with tight `ram_budget_mb` activates `disjoint_evict` and matches bit-for-bit!
    with torch.device("meta"):
        meta_m2 = _MiniDiT(128).to(torch.bfloat16).eval()
    with TorchBlockStreamer(
        meta_m2,
        compute_device=dev,
        mode=BlockStreamMode.MMAP_STREAM,
        safetensors_paths=[gguf_path],
        ram_budget_mb=0.01,  # 10 KB budget < file size -> forces disjoint MADV_DONTNEED eviction
    ) as mmap_streamer:
        assert mmap_streamer._mmap_ring is not None
        assert mmap_streamer._mmap_ring.disjoint_evict is True
        with torch.no_grad():
            out_mmap = meta_m2(x)
        assert torch.equal(out_cpu, out_mmap)

    # Verify `MemoryBudgetPlanner` fits all 4 compressed Q8_0 blocks into RAM budget that fits <4 BF16 blocks!
    plan = MemoryBudgetPlanner.plan_blocks(
        block_sizes_bytes=[100 * 1024 * 1024] * 4,      # 400 MB BF16 on GPU
        cpu_block_sizes_bytes=[53 * 1024 * 1024] * 4,   # 212 MB Q8_0 in CPU RAM
        vram_budget_mb=0.0,
        ram_budget_mb=250.0,                            # 250 MB fits all 4 Q8_0 blocks (212 MB), only 2 BF16 blocks!
        has_mmap_backing=True,
    )
    assert plan.tiers == [BlockTier.CPU_PINNED] * 4


def test_gguf_string_kv_metadata_forwarded_to_zero_ram_header(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from phobos_stream.integrations.comfyui.zero_ram_loader import build_header_only_meta_state_dict

    buf = bytearray(b"GGUF")
    buf.extend(struct.pack("<IQQ", 3, 1, 1))  # 1 tensor, 1 string KV ("config")
    k_b = b"config"
    v_b = b'{"transformer": {"ff_bias": false, "num_layers": 48}}'
    buf.extend(struct.pack("<Q", len(k_b)) + k_b + struct.pack("<I", 8) + struct.pack("<Q", len(v_b)) + v_b)
    t_b = b"transformer_blocks.0.ff.net.0.proj.weight"
    buf.extend(struct.pack("<Q", len(t_b)) + t_b + struct.pack("<I", 2) + struct.pack("<QQ", 32, 2) + struct.pack("<IQ", 0, 0))
    if len(buf) % 32:
        buf.extend(b"\x00" * (32 - len(buf) % 32))
    buf.extend(b"\x00" * (64 * 4))
    p = tmp_path / "ltx25_cfg.gguf"
    p.write_bytes(bytes(buf))

    meta_sd, raw_meta = build_header_only_meta_state_dict(p, scalar_threshold_elems=16)
    assert "transformer_blocks.0.ff.net.0.proj.weight" in meta_sd
    assert '"ff_bias": false' in raw_meta.get("config", "")


def test_quantized_tensor_wrapper_subclass_streaming() -> None:
    """Ensure `__torch_dispatch__` `QuantizedTensor` wrapper subclasses preserve `_qdata` across block swaps."""
    import torch
    import torch.nn as nn
    from phobos_stream.streaming import BlockStreamMode, TorchBlockStreamer

    class _MockQuantizedTensor(torch.Tensor):
        @staticmethod
        def __new__(cls, qdata: torch.Tensor, scale: torch.Tensor) -> "_MockQuantizedTensor":
            return torch.Tensor._make_wrapper_subclass(
                cls,
                qdata.shape,
                dtype=torch.float32,
                device=qdata.device,
                requires_grad=False,
            )

        def __init__(self, qdata: torch.Tensor, scale: torch.Tensor) -> None:
            self._qdata = qdata
            self._scale = scale

        def __repr__(self) -> str:
            return f"_MockQuantizedTensor(device={self._qdata.device})"

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func in (torch.ops.aten._to_copy.default, torch.ops.aten.to.dtype_layout):
                src = args[0]
                dev = kwargs.get("device", src._qdata.device)
                return _MockQuantizedTensor(src._qdata.to(device=dev), src._scale.to(device=dev))
            if func is torch.ops.aten.detach.default:
                src = args[0]
                return _MockQuantizedTensor(src._qdata.detach(), src._scale.detach())
            if func is torch.ops.aten.t.default:
                src = args[0]
                return _MockQuantizedTensor(src._qdata.t(), src._scale)
            if func is torch.ops.aten.mm.default:
                inp, wt = args[0], args[1]
                assert inp.device == wt._qdata.device
                return torch.mm(inp, wt._qdata.float() * wt._scale)
            if func is torch.ops.aten.linear.default:
                inp, wt = args[0], args[1]
                bias = args[2] if len(args) > 2 else None
                assert inp.device == wt._qdata.device
                return torch.nn.functional.linear(inp, wt._qdata.float() * wt._scale, bias)
            raise NotImplementedError(f"Unhandled op {func}")

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(16, 16, bias=False)
            qt = _MockQuantizedTensor(torch.ones((16, 16), dtype=torch.int8), torch.tensor(0.25))
            self.proj.weight = nn.Parameter(qt, requires_grad=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x)

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([_Block(), _Block()])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for blk in self.blocks:
                x = blk(x)
            return x

    m = _Model()
    x = torch.ones((2, 16), dtype=torch.float32)
    ref = m(x)
    dev = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
    with TorchBlockStreamer(m, compute_device=dev, mode=BlockStreamMode.CPU_OFFLOAD, pin_memory=False) as st:
        out = m(x.to(dev)).cpu()
        assert st.blocks_executed == 2
        assert m.blocks[0].proj.weight._qdata.device.type == "cpu"
    assert torch.allclose(ref, out)


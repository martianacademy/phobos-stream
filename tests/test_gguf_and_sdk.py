"""Unit tests for On-GPU `.gguf` (`Q8_0` / `Q4_0` / `Q4_K`) block streaming, I2V image loading & 1-line SDK."""

from __future__ import annotations

import base64
import io
from pathlib import Path
import struct
import numpy as np
import pytest

import phobos_stream
from phobos_stream.backends.gguf_backend import GGUFBackend
from phobos_stream.pipelines.hf_diffusers_universal import load_pil_conditioning_image
from phobos_stream.storage.gguf_mmap_loader import GGUFMMapLoader, dequantize_ggml_on_device
from phobos_stream.streaming import BlockStreamMode, TorchBlockStreamer


def _write_minimal_gguf_v3(path: Path, tensors: dict[str, tuple[int, tuple[int, ...], bytes]]) -> None:
    """Write a spec-compliant GGUF v3 binary file (`magic='GGUF'`, alignment=32) for testing."""
    buf = bytearray()
    buf.extend(b"GGUF")
    buf.extend(struct.pack("<IQQ", 3, len(tensors), 0))  # version=3, n_tensors, n_kv=0

    data_blobs = bytearray()
    for name, (ggml_type, pt_shape, raw_bytes) in tensors.items():
        name_b = name.encode("utf-8")
        buf.extend(struct.pack("<Q", len(name_b)))
        buf.extend(name_b)
        buf.extend(struct.pack("<I", len(pt_shape)))
        # GGUF stores dimensions reversed relative to PyTorch
        for dim in reversed(pt_shape):
            buf.extend(struct.pack("<Q", int(dim)))
        rel_offset = len(data_blobs)
        buf.extend(struct.pack("<IQ", ggml_type, rel_offset))
        data_blobs.extend(raw_bytes)
        rem = len(data_blobs) % 32
        if rem:
            data_blobs.extend(b"\x00" * (32 - rem))

    rem_hdr = len(buf) % 32
    if rem_hdr:
        buf.extend(b"\x00" * (32 - rem_hdr))
    buf.extend(data_blobs)
    path.write_bytes(bytes(buf))


def test_gguf_mmap_loader_on_gpu_q8_0_and_torch_block_streamer(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    import torch.nn.functional as F

    target_dev = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"

    class _GGUFDiT(nn.Module):
        def __init__(self, dim: int = 64) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(3)])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = x
            for blk in self.blocks:
                h = h + 0.2 * F.silu(blk(h))
            return h

    torch.manual_seed(42)
    ref_model = _GGUFDiT(dim=64).to(torch.bfloat16).eval()
    gguf_helper = GGUFBackend()

    # Quantize all 3 blocks into real GGUF `Q8_0` (ggml_type = 8) -> 47% smaller than BF16!
    gguf_tensors = {}
    bf16_bytes = 0
    q8_bytes = 0
    for k, param in ref_model.state_dict().items():
        arr_f32 = param.detach().float().cpu().numpy()
        raw_q8 = gguf_helper.quantize_q8_0(arr_f32)
        gguf_tensors[k] = (8, tuple(param.shape), raw_q8)
        bf16_bytes += param.numel() * 2
        q8_bytes += len(raw_q8)

    gguf_path = tmp_path / "tiny_dit_q8_0.gguf"
    _write_minimal_gguf_v3(gguf_path, gguf_tensors)

    x = torch.randn(2, 10, 64, dtype=torch.bfloat16, device=target_dev)
    with torch.no_grad():
        expected = ref_model.to(target_dev)(x)

    # Stream directly from `.gguf` onto zero-RAM `meta` skeleton!
    with torch.device("meta"):
        meta_dit = _GGUFDiT(dim=64).to(torch.bfloat16).eval()

    with TorchBlockStreamer(
        meta_dit,
        compute_device=target_dev,
        mode=BlockStreamMode.MMAP_STREAM,
        safetensors_paths=[gguf_path],
    ) as streamer:
        assert meta_dit.blocks[0].weight.numel() == 0
        with torch.no_grad():
            actual = meta_dit(x)
        assert streamer.blocks_executed == 3
        # Streamed bytes must equal compressed Q8_0 bytes (~53% of BF16!), NOT expanded FP32!
        assert streamer.bytes_streamed == q8_bytes
        assert streamer.bytes_streamed < bf16_bytes * 0.56

    cos = float(F.cosine_similarity(expected.float().flatten(), actual.float().flatten(), dim=0).item())
    assert cos > 0.9999, f"Expected >0.9999 Q8_0 cosine similarity, got {cos}"


def test_q4_k_superblock_dequantization_runs_vectorized() -> None:
    torch = pytest.importorskip("torch")

    # 2 super-blocks = 512 elements -> 2 * 144 = 288 bytes
    raw = torch.zeros((2, 144), dtype=torch.uint8)
    # Set fp16 d = 0.5 (0x3800 little-endian -> [0x00, 0x38]), dmin = 0.25 (0x3400 -> [0x00, 0x34])
    raw[:, 0] = 0x00
    raw[:, 1] = 0x38
    raw[:, 2] = 0x00
    raw[:, 3] = 0x34
    raw[:, 4:8] = 2      # sub-blocks 0..3 scales = 2
    raw[:, 8:12] = 4     # sub-blocks 0..3 mins = 4 -> m_sub = 0.25 * 4 = 1.0
    raw[:, 12:16] = 0x42 # sub-blocks 4..7: low nibble scale = 2, high nibble min = 4
    raw[:, 16:144] = 0x33  # low nibble=3, high nibble=3 -> expected = 3 * (0.5 * 2) - (0.25 * 4) = 2.0
    out = dequantize_ggml_on_device(raw.reshape(-1), "Q4_K", (16, 32), torch.float32)
    assert out.shape == (16, 32)
    assert torch.allclose(out, torch.full((16, 32), 2.0), atol=1e-3)

    # Q5_K (176B per 256 elems): d=0.5, dmin=0, scales=2, qh=0xFF (high bit=1 -> +16), qs=0x11 (low=1 -> 17) -> 17.0
    raw5 = torch.zeros((2, 176), dtype=torch.uint8)
    raw5[:, 1] = 0x38     # fp16 0.5
    raw5[:, 4:8] = 2
    raw5[:, 12:16] = 0x02
    raw5[:, 16:48] = 0xFF
    raw5[:, 48:176] = 0x11
    out5 = dequantize_ggml_on_device(raw5.reshape(-1), "Q5_K", (16, 32), torch.float32)
    assert torch.allclose(out5, torch.full((16, 32), 17.0), atol=1e-3)

    # Q6_K (210B per 256 elems): ql=0x55 (5), qh=0xFF (bits=3 -> 48 => 53 - 32 = 21), scales=2, d=0.5 -> 21.0
    raw6 = torch.zeros((2, 210), dtype=torch.uint8)
    raw6[:, 0:128] = 0x55
    raw6[:, 128:192] = 0xFF
    raw6[:, 192:208] = 2
    raw6[:, 209] = 0x38   # fp16 0.5
    out6 = dequantize_ggml_on_device(raw6.reshape(-1), "Q6_K", (16, 32), torch.float32)
    assert torch.allclose(out6, torch.full((16, 32), 21.0), atol=1e-3)


def test_one_line_sdk_stream_and_unstream_plus_i2v_image_loader(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class _SimpleDiT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([nn.Linear(32, 32) for _ in range(4)])

        def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            h = x * (1.0 + 0.02 * timestep.view(-1, 1, 1))
            for b in self.blocks:
                h = h + 0.05 * torch.tanh(b(h))
            return h

    model = _SimpleDiT().eval()
    phobos_stream.stream(model, mode="cpu_offload", compute_device="cpu", cache_thresh=0.18)
    assert hasattr(model, "_phobos_streamer")

    x = torch.randn(1, 6, 32)
    with torch.no_grad():
        for s in (1.0, 0.9, 0.8, 0.7):
            _ = model(x, torch.tensor([s]))
    assert model._phobos_streamer.residual_cache.skipped_steps >= 1
    phobos_stream.unstream(model)
    assert not hasattr(model, "_phobos_streamer")

    # Test I2V conditioning image loader (data URI + local PNG resize)
    Image = pytest.importorskip("PIL.Image")
    img_arr = np.zeros((40, 60, 3), dtype=np.uint8)
    img_arr[:, :, 0] = 220
    buf = io.BytesIO()
    Image.fromarray(img_arr).save(buf, format="PNG")
    data_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    loaded = load_pil_conditioning_image(data_uri, width=128, height=96)
    assert loaded.size == (128, 96)

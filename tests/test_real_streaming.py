"""Lightweight (<5 MB RAM) verification tests for real PyTorch & MLX block streaming."""

from pathlib import Path
from typing import Any
import numpy as np
import pytest
import safetensors.numpy

from phobos_stream.media import save_rgb_mp4
from phobos_stream.pipeline.stage_runner import StageSequentialRunner, tiled_spatial_decode
from phobos_stream.pipeline.stream_engine import PhobosStreamEngine
from phobos_stream.storage.layout import BlockMetadata, ModelShardsManifest
from phobos_stream.streaming import BlockStreamMode, MLXBlockStreamer, TorchBlockStreamer
from phobos_stream.types import BackendType, GenerationConfig


def test_torch_block_streamer_cpu_offload_and_mmap_bf16(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    st_torch = pytest.importorskip("safetensors.torch")
    import torch.nn as nn

    class TinyBlock(nn.Module):
        def __init__(self, dim: int = 32) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.qkv = nn.Linear(dim, dim * 3, bias=False)
            self.proj = nn.Linear(dim, dim, bias=False)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp_in = nn.Linear(dim, dim * 2)
            self.mlp_out = nn.Linear(dim * 2, dim)
            # Inner ModuleList to ensure discover_block_modules does NOT misclassify it
            self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False)])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.norm1(x)
            q, k, v = self.qkv(h).chunk(3, dim=-1)
            scores = torch.matmul(q, k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
            attn = torch.matmul(torch.softmax(scores, dim=-1), v)
            x = x + self.to_out[0](self.proj(attn))
            x = x + self.mlp_out(torch.nn.functional.silu(self.mlp_in(self.norm2(x))))
            return x

    class TinyDiT(nn.Module):
        def __init__(self, dim: int = 32, num_blocks: int = 4) -> None:
            super().__init__()
            self.x_embedder = nn.Linear(8, dim)
            self.layers = nn.ModuleList([TinyBlock(dim) for _ in range(num_blocks)])
            self.final_layer = nn.Linear(dim, 8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.x_embedder(x)
            for blk in self.layers:
                h = blk(h)
            return self.final_layer(h)

    torch.manual_seed(7)
    model_ref = TinyDiT(dim=32, num_blocks=4).to(dtype=torch.bfloat16).eval()
    st_path = tmp_path / "tiny_dit_bf16.safetensors"
    st_torch.save_file(model_ref.state_dict(), str(st_path))

    x = torch.randn(2, 12, 8, dtype=torch.bfloat16)
    with torch.no_grad():
        expected = model_ref(x)

    # 1. Verify CPU_OFFLOAD exact equivalence
    model_cpu_off = TinyDiT(dim=32, num_blocks=4).to(dtype=torch.bfloat16).eval()
    model_cpu_off.load_state_dict(model_ref.state_dict())
    with TorchBlockStreamer(model_cpu_off, compute_device="cpu", mode=BlockStreamMode.CPU_OFFLOAD) as s_cpu:
        with torch.no_grad():
            out_cpu_off = model_cpu_off(x)
        assert s_cpu.blocks_executed == 4
        assert s_cpu.bytes_streamed > 0
    assert torch.allclose(expected, out_cpu_off, atol=1e-4)

    # 2. Verify MMAP_STREAM on zero-RAM `meta`-device skeleton + 0-element inactive eviction
    with torch.device("meta"):
        model_mmap = TinyDiT(dim=32, num_blocks=4).to(dtype=torch.bfloat16)
    model_mmap.eval()
    with TorchBlockStreamer(
        model_mmap,
        compute_device="cpu",
        mode=BlockStreamMode.MMAP_STREAM,
        safetensors_paths=[st_path],
    ) as s_mmap:
        # Before forward: every block parameter must be evicted to 0 elements
        for blk in model_mmap.layers:
            for p in blk.parameters():
                assert p.numel() == 0

        with torch.no_grad():
            out_mmap = model_mmap(x)

        # Immediately after forward: every block parameter must be evicted back to 0 elements
        for blk in model_mmap.layers:
            for p in blk.parameters():
                assert p.numel() == 0

        assert s_mmap.blocks_executed == 4
        assert s_mmap.bytes_streamed > 0

    assert torch.allclose(expected, out_mmap, atol=1e-4)

    # 3. Verify missing safetensors key raises KeyError (zero silent fake weights)
    incomplete_path = tmp_path / "incomplete.safetensors"
    st_torch.save_file({"x_embedder.weight": model_ref.x_embedder.weight}, str(incomplete_path))
    with pytest.raises(KeyError):
        TorchBlockStreamer(
            TinyDiT(dim=32, num_blocks=2),
            compute_device="cpu",
            mode=BlockStreamMode.MMAP_STREAM,
            safetensors_paths=[incomplete_path],
        ).attach()


def test_mlx_block_streamer_exact_equivalence_and_eviction(tmp_path: Path) -> None:
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    from mlx.utils import tree_flatten

    class TinyMLXBlock(nn.Module):
        def __init__(self, dim: int = 32) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.w1 = nn.Linear(dim, dim * 2)
            self.w2 = nn.Linear(dim * 2, dim)

        def __call__(self, x: Any) -> Any:
            h = self.norm(x)
            return x + self.w2(nn.silu(self.w1(h)))

    mx.random.seed(11)
    blocks_ref = [TinyMLXBlock(32) for _ in range(3)]
    flat_weights = {}
    for idx, blk in enumerate(blocks_ref):
        blk.set_dtype(mx.bfloat16)
        mx.eval(blk.parameters())
        for k, v in tree_flatten(blk.parameters()):
            flat_weights[f"layers.{idx}.{k}"] = v

    st_path = tmp_path / "tiny_mlx_bf16.safetensors"
    mx.save_safetensors(str(st_path), flat_weights)

    x = mx.random.normal((1, 8, 32)).astype(mx.bfloat16)
    expected = x
    for blk in blocks_ref:
        expected = blk(expected)
    mx.eval(expected)

    blocks_stream = [TinyMLXBlock(32) for _ in range(3)]
    for blk in blocks_stream:
        blk.set_dtype(mx.bfloat16)

    with MLXBlockStreamer(blocks_stream, [st_path], block_prefix_template="layers.{i}") as streamer:
        for blk in blocks_stream:
            for _, arr in tree_flatten(blk.parameters()):
                assert arr.size == 0

        actual = streamer.run_sequence(x)
        assert streamer.blocks_executed == 3
        assert streamer.bytes_streamed > 0

        for blk in blocks_stream:
            for _, arr in tree_flatten(blk.parameters()):
                assert arr.size == 0

    assert np.allclose(
        np.array(expected.astype(mx.float32)),
        np.array(actual.astype(mx.float32)),
        atol=1e-4,
    )


def test_stage_sequential_runner_and_tiled_vae() -> None:
    def _decode_tile(lat_tile: np.ndarray) -> np.ndarray:
        # Deterministic linear upsample mock: (B, C, H, W) -> (1, H*8, W*8, 3)
        _, _, h, w = lat_tile.shape
        base = np.mean(lat_tile[0, :3], axis=0, keepdims=False)
        up = np.repeat(np.repeat(base, 8, axis=0), 8, axis=1)
        return np.stack([up, up * 0.5, up * 0.25], axis=-1)[None, ...].astype(np.float32)

    grad_y = np.linspace(0.0, 1.0, 24, dtype=np.float32)[None, None, :, None]
    grad_x = np.linspace(0.0, 1.0, 24, dtype=np.float32)[None, None, None, :]
    latents = np.broadcast_to(grad_y + grad_x, (1, 4, 24, 24)).copy()

    full_decoded = _decode_tile(latents)
    tiled_decoded = tiled_spatial_decode(
        _decode_tile,
        latents,
        tile_latent_h=12,
        tile_latent_w=12,
        overlap_latent=4,
        spatial_upscale=8,
    )
    assert tiled_decoded.shape == (1, 192, 192, 3)
    assert np.allclose(full_decoded, tiled_decoded, atol=1e-5)

    runner = StageSequentialRunner()
    frames, metrics = runner.run(
        encode_fn=lambda: {"cond": np.ones((1, 4, 8), dtype=np.float32)},
        denoise_fn=lambda cond: (latents * float(cond["cond"].mean()), 4, 4096),
        decode_fn=lambda lat: np.clip(tiled_spatial_decode(_decode_tile, lat, 12, 12, 4, 8) * 127.5, 0, 255).astype(np.uint8),
        total_steps=4,
    )
    assert frames.shape == (1, 192, 192, 3)
    assert metrics.steps_completed == 4
    assert metrics.disk_bytes_read_mb > 0


def test_batched_cfg_reads_shards_once_per_step_and_exports_mp4(tmp_path: Path) -> None:
    dim = 32
    shard_file = tmp_path / "block0.safetensors"
    tensors = {
        "blocks.0.norm1.weight": np.ones((dim,), dtype=np.float32),
        "blocks.0.self_attn.q_proj.weight": np.eye(dim, dtype=np.float32) * 0.1,
        "blocks.0.self_attn.k_proj.weight": np.eye(dim, dtype=np.float32) * 0.1,
        "blocks.0.self_attn.v_proj.weight": np.eye(dim, dtype=np.float32) * 0.1,
        "blocks.0.self_attn.o_proj.weight": np.eye(dim, dtype=np.float32) * 0.1,
        "blocks.0.norm2.weight": np.ones((dim,), dtype=np.float32),
        "blocks.0.cross_attn.q_proj.weight": np.eye(dim, dtype=np.float32) * 0.1,
        "blocks.0.cross_attn.k_proj.weight": np.eye(dim, dtype=np.float32) * 0.1,
        "blocks.0.cross_attn.v_proj.weight": np.eye(dim, dtype=np.float32) * 0.1,
        "blocks.0.cross_attn.o_proj.weight": np.eye(dim, dtype=np.float32) * 0.1,
        "blocks.0.norm3.weight": np.ones((dim,), dtype=np.float32),
        "blocks.0.ffn.gate_proj.weight": np.ones((dim * 2, dim), dtype=np.float32) * 0.02,
        "blocks.0.ffn.up_proj.weight": np.ones((dim * 2, dim), dtype=np.float32) * 0.02,
        "blocks.0.ffn.down_proj.weight": np.ones((dim, dim * 2), dtype=np.float32) * 0.02,
    }
    safetensors.numpy.save_file(tensors, str(shard_file))
    expected_block_bytes = sum(arr.nbytes for arr in tensors.values())

    manifest = ModelShardsManifest(
        model_name="tiny-real-shard",
        architecture="wan2.1",
        num_blocks=1,
        hidden_size=dim,
        blocks={
            0: BlockMetadata(
                block_id=0,
                filename=str(shard_file),
                tensor_names=list(tensors.keys()),
                byte_size=expected_block_bytes,
            )
        },
        allow_synthetic=True,  # For text/vae stages while testing real DiT shard IO
    )
    engine = PhobosStreamEngine(manifest=manifest, backend_type=BackendType.CPU)
    cfg = GenerationConfig(
        prompt="Mars canyon",
        negative_prompt="blurry",
        height=128,
        width=128,
        num_frames=5,
        steps=3,
        guidance_scale=6.0,
    )
    frames, metrics = engine.generate(cfg)
    # 3 steps * 1x shard read per step (NOT 2x!)
    expected_mb = (3 * expected_block_bytes) / (1024.0 * 1024.0)
    assert np.isclose(metrics.disk_bytes_read_mb, expected_mb, rtol=1e-4)

    mp4_path = save_rgb_mp4(frames, tmp_path / "out.mp4", fps=12)
    assert mp4_path.exists()
    assert b"ftyp" in mp4_path.read_bytes()[:32]

"""Unit tests for PhobosVAEStreamPatcherNode (Video VAE block streaming & 1D Audio VAE chunking)."""

from __future__ import annotations

import pytest

from phobos_stream.integrations.comfyui.vae_patcher import (
    PhobosVAEStreamPatcherNode,
    attach_streamer_to_comfy_vae,
    decode_audio_latent_chunked,
)


def test_video_vae_patcher_drops_encoder_and_matches_monolithic() -> None:
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class _MockViT3DDecoder(nn.Module):
        def __init__(self, dim: int = 16, num_blocks: int = 6) -> None:
            super().__init__()
            self.stem = nn.Linear(dim, dim)
            self.blocks = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_blocks)])
            self.head = nn.Linear(dim, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.stem(x)
            for blk in self.blocks:
                h = h + 0.1 * torch.tanh(blk(h))
            return self.head(h)

    class _MockFirstStageVideoVAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Linear(16, 16)
            self.quant_conv = nn.Linear(16, 16)
            self.post_quant_conv = nn.Linear(16, 16)
            self.decoder = _MockViT3DDecoder(dim=16, num_blocks=6)

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            return self.decoder(self.post_quant_conv(z))

    class _MockPatcher:
        def __init__(self, model: nn.Module) -> None:
            self.model = model
            self.load_called = False

        def load(self, *args, **kwargs) -> None:
            self.load_called = True

        def patch_model(self, *args, **kwargs):
            self.load_called = True
            return self.model

        def model_size(self) -> int:
            return 4_850_000_000

    class _MockComfyVAE:
        def __init__(self) -> None:
            self.first_stage_model = _MockFirstStageVideoVAE().eval()
            self.patcher = _MockPatcher(self.first_stage_model)
            self.device = torch.device("cpu")
            self.vae_dtype = torch.float32

        def decode(self, samples: torch.Tensor) -> torch.Tensor:
            self.patcher.load()
            self.patcher.patch_model()
            return self.first_stage_model.decode(samples)

    vae = _MockComfyVAE()
    z = torch.randn(1, 4, 16)
    with torch.no_grad():
        expected = vae.first_stage_model.decode(z)

    patcher_node = PhobosVAEStreamPatcherNode()
    patched_vae, = patcher_node.patch_vae(
        vae=vae,
        mode="cpu_offload",
        drop_encoder=True,
    )
    meta = getattr(patched_vae, "_phobos_vae_meta")
    streamer = meta["streamer"]
    assert streamer is not None
    assert meta["num_streamed_blocks"] == 6
    assert set(meta["dropped_modules"]) == {"encoder", "quant_conv"}
    assert patched_vae.first_stage_model.encoder is None
    assert patched_vae.first_stage_model.quant_conv is None
    assert patched_vae.patcher.model_size() == 0

    try:
        with torch.no_grad():
            actual = patched_vae.decode(z)
        assert not patched_vae.patcher.load_called
        assert streamer.blocks_executed == 6
        assert torch.allclose(expected, actual, atol=1e-5)

        # Verify idempotent re-patch switches cleanly to hybrid_budget without stacking hooks.
        attach_streamer_to_comfy_vae(patched_vae, block_stream_mode="hybrid_budget", drop_encoder=False)
        new_streamer = getattr(patched_vae, "_phobos_vae_streamer")
        assert new_streamer is not streamer
        with torch.no_grad():
            actual_hybrid = patched_vae.decode(z)
        assert new_streamer.blocks_executed == 6
        assert torch.allclose(expected, actual_hybrid, atol=1e-5)
    finally:
        active = getattr(patched_vae, "_phobos_vae_streamer", None)
        if active is not None:
            active.detach()


def test_audio_vae_chunked_overlap_add_is_exact_and_wraps_comfy_vae() -> None:
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    hop = 16

    class _MockStereoAudioFirstStage(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Conv1d(8, 2, kernel_size=1)
            self.calls = 0

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            # Pointwise projection + deterministic repeat upsample so overlap-add is exact.
            stereo = self.proj(z)
            return stereo.repeat_interleave(hop, dim=-1)

    class _MockAudioVAE:
        def __init__(self) -> None:
            self.first_stage_model = _MockStereoAudioFirstStage().eval()
            self.device = torch.device("cpu")
            self.vae_dtype = torch.float32

        def decode(self, samples: torch.Tensor) -> torch.Tensor:
            return self.first_stage_model.decode(samples)

        def decode_tiled(self, samples: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            raise IndexError("2D spatial tiler should never run on 3D audio latents")

    audio_vae = _MockAudioVAE()
    latent = torch.randn(1, 8, 160)
    with torch.no_grad():
        expected = audio_vae.first_stage_model.decode(latent)

    audio_vae.first_stage_model.calls = 0
    patched_vae, = PhobosVAEStreamPatcherNode().patch_vae(
        vae=audio_vae,
        mode="hybrid_budget",
        drop_encoder=True,
        audio_chunk_steps=64,
        audio_overlap_steps=8,
    )

    out_decode = patched_vae.decode(latent)
    assert tuple(out_decode.shape) == (1, 2, 160 * hop)
    assert patched_vae.first_stage_model.calls == 3  # [0:64], [56:120], [112:160]
    assert torch.allclose(expected, out_decode, atol=1e-5)

    # Even `decode_tiled` must intercept 3D audio latents cleanly instead of crashing.
    out_tiled = patched_vae.decode_tiled(latent)
    assert torch.allclose(expected, out_tiled, atol=1e-5)


def test_decode_audio_latent_chunked_edge_cases() -> None:
    torch = pytest.importorskip("torch")

    # Short latent (T <= chunk_steps) + 2D input [C, T]
    z_short = torch.ones(4, 16)
    out = decode_audio_latent_chunked(
        decode_fn=lambda x: x[:, :2, :].repeat_interleave(8, dim=-1),
        latent=z_short,
        chunk_steps=64,
        overlap_steps=8,
    )
    assert tuple(out.shape) == (2, 128)
    assert torch.allclose(out, torch.ones(2, 128))

    with pytest.raises(ValueError, match="overlap_steps"):
        decode_audio_latent_chunked(lambda x: x, torch.ones(1, 4, 32), chunk_steps=16, overlap_steps=16)


def test_audio_vae_4d_stereo_latent_chunking_minimax_h3_shape() -> None:
    """Regression test for MiniMax-H3 4D stereo audio latents `[B, C, 2, T]`."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    hop = 8

    class _MockMiniMaxAudioFirstStage(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(32, 1)
            self.calls = 0

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            # z: [B, 32, 2, t] -> [B, 2, t * hop]
            reduced = self.proj(z.permute(0, 2, 3, 1)).squeeze(-1)
            return reduced.repeat_interleave(hop, dim=-1)

    class _MockMiniMaxAudioVAE:
        def __init__(self) -> None:
            self.first_stage_model = _MockMiniMaxAudioFirstStage().eval()
            self.device = torch.device("cpu")
            self.vae_dtype = torch.float32

        def decode(self, samples: torch.Tensor) -> torch.Tensor:
            return self.first_stage_model.decode(samples)

    avae = _MockMiniMaxAudioVAE()
    latent_4d = torch.randn(1, 32, 2, 235)
    with torch.no_grad():
        expected = avae.first_stage_model.decode(latent_4d)

    avae.first_stage_model.calls = 0
    patched, = PhobosVAEStreamPatcherNode().patch_vae(
        vae=avae,
        mode="cpu_offload",
        drop_encoder=True,
        audio_chunk_steps=64,
        audio_overlap_steps=8,
    )
    out = patched.decode(latent_4d)
    assert tuple(out.shape) == (1, 2, 235 * hop)
    assert patched.first_stage_model.calls == 5
    assert torch.allclose(expected, out, atol=1e-5)

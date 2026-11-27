"""Cross-framework numerical parity & streaming tests for `MLXWanVideoStreamer` (`Wan2.1` 3D DiT)."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest


def test_mlx_wan_video_matches_diffusers_wan_transformer_3d_and_streams_blocks(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    mx = pytest.importorskip("mlx.core")
    st_torch = pytest.importorskip("safetensors.torch")
    diffusers = pytest.importorskip("diffusers")

    from phobos_stream.pipelines.mlx_wan_video import MLXWanVideoStreamer

    torch.manual_seed(42)
    pt_wan = diffusers.WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=4,
        attention_head_dim=24,  # divisible by 6 for 3D RoPE (8, 8, 8)
        in_channels=16,
        out_channels=16,
        text_dim=64,
        freq_dim=32,
        ffn_dim=192,
        num_layers=4,
    ).to(torch.bfloat16).eval()

    ckpt_path = tmp_path / "wan21_tiny_bf16.safetensors"
    st_torch.save_file(pt_wan.state_dict(), str(ckpt_path))

    # 1-step Euler flow-matching parity check: sigma 1.0 -> 0.0 => lat_out = lat_in - 1.0 * v_pred
    lat_np = (np.random.default_rng(7).standard_normal((1, 16, 3, 8, 8)) * 0.5).astype(np.float32)
    txt_np = (np.random.default_rng(9).standard_normal((1, 12, 64)) * 0.3).astype(np.float32)

    with torch.no_grad():
        v_pt = pt_wan(
            hidden_states=torch.from_numpy(lat_np).to(torch.bfloat16),
            timestep=torch.tensor([999.0]),
            encoder_hidden_states=torch.from_numpy(txt_np).to(torch.bfloat16),
            return_dict=False,
        )[0].float().numpy()
    expected_lat = lat_np - v_pt

    # 1. Pure `block_stream` (0 resident blocks)
    streamer_mmap = MLXWanVideoStreamer(
        ckpt_path,
        num_heads=4,
        patch_size=(1, 2, 2),
        freq_dim=32,
        mode="block_stream",
    )
    out_mmap, m_mmap = streamer_mmap.denoise_latents(
        mx.array(lat_np),
        mx.array(txt_np),
        timesteps=[999.0],
        sigmas=[1.0, 0.0],
    )
    actual_mmap = np.array(out_mmap)
    assert m_mmap.resident_blocks == 0
    cos = float(
        np.dot(expected_lat.flatten(), actual_mmap.flatten())
        / (np.linalg.norm(expected_lat) * np.linalg.norm(actual_mmap))
    )
    assert cos > 0.9995, f"Expected >0.9995 parity with PyTorch WanTransformer3DModel, got {cos}"

    # 2. `hybrid_stream` + `First-Block Cache` across 5 Euler steps
    streamer_hyb = MLXWanVideoStreamer(
        ckpt_path,
        num_heads=4,
        patch_size=(1, 2, 2),
        freq_dim=32,
        mode="hybrid_stream",
        memory_budget_mb=400.0,
        activation_reserve_mb=0.0,
        cache_rel_l1_thresh=0.50,
        cache_warmup_steps=2,
    )
    _, m_hyb = streamer_hyb.denoise_latents(
        mx.array(lat_np),
        mx.array(txt_np),
        timesteps=[999.0, 800.0, 600.0, 400.0, 200.0],
        sigmas=[1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
    )
    assert m_hyb.skipped_steps >= 1

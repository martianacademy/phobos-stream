"""Tests for ComfyUI custom nodes, 4D/5D latent formats, block patcher, and standalone install."""

from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pytest

from phobos_stream.integrations.comfyui.comfy_runtime import build_empty_comfy_latent
from phobos_stream.integrations.comfyui.nodes import (
    PhobosBlockStreamPatcherNode,
    PhobosStreamLoaderNode,
    PhobosStreamSamplerNode,
    PhobosStreamVAEDecodeNode,
)


def test_comfyui_loader_sampler_and_vae_decode() -> None:
    loader = PhobosStreamLoaderNode()
    model_dict = loader.load_model(
        model_name="test-wan2.1",
        architecture="wan2.1",
        backend="cpu",
        num_blocks=2,
        hidden_size=64,
        use_comfy_stage_offload=False,
        allow_synthetic=True,
    )[0]
    assert "manifest" in model_dict
    assert model_dict["engine"] is not None

    sampler = PhobosStreamSamplerNode()
    images, latents = sampler.sample(
        phobos_model=model_dict,
        prompt="A serene mountain valley with crystal streams",
        negative_prompt="blurry",
        width=128,
        height=128,
        num_frames=5,
        steps=2,
        guidance_scale=5.0,
        seed=42,
    )

    assert tuple(images.shape) == (5, 128, 128, 3)
    assert "samples" in latents

    decoder = PhobosStreamVAEDecodeNode()
    decoded_images = decoder.decode(latents, model_dict)[0]
    assert tuple(decoded_images.shape) == (5, 128, 128, 3)
    assert float(decoded_images.max()) > 0.1


def test_dynamic_comfy_latent_shapes_4ch_16ch_and_5d_video() -> None:
    pytest.importorskip("torch")

    class _MockLatentFormat:
        def __init__(self, channels: int, dims: int, sp: int = 8, tp: int = 1) -> None:
            self.latent_channels = channels
            self.latent_dimensions = dims
            self.spacial_downscale_ratio = sp
            self.temporal_downscale_ratio = tp

    class _MockPatcher:
        def __init__(self, lf: _MockLatentFormat) -> None:
            self._lf = lf

        def get_model_object(self, name: str):
            return self._lf if name == "latent_format" else None

    # 1. SDXL 4-channel 2D image latent
    sdxl_lat = build_empty_comfy_latent(_MockPatcher(_MockLatentFormat(4, 2, 8, 1)), width=1024, height=768, num_frames=1)
    assert tuple(sdxl_lat["samples"].shape) == (1, 4, 96, 128)

    # 2. FLUX / Lumina2 16-channel 2D image latent
    flux_lat = build_empty_comfy_latent(_MockPatcher(_MockLatentFormat(16, 2, 8, 1)), width=1280, height=720, num_frames=1)
    assert tuple(flux_lat["samples"].shape) == (1, 16, 90, 160)

    # 3. Wan2.1 / HunyuanVideo 16-channel 5D video latent: (49 - 1)//4 + 1 = 13 temporal frames
    wan_lat = build_empty_comfy_latent(_MockPatcher(_MockLatentFormat(16, 3, 8, 4)), width=1280, height=720, num_frames=49)
    assert tuple(wan_lat["samples"].shape) == (1, 16, 13, 90, 160)


def test_phobos_block_stream_patcher_node_on_mock_comfy_model() -> None:
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class _MiniDiffModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([nn.Linear(16, 16), nn.Linear(16, 16)])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for blk in self.blocks:
                x = blk(x)
            return x

    class _MockComfyModelPatcher:
        def __init__(self, diff_mod: nn.Module) -> None:
            self._diff_mod = diff_mod

        def clone(self) -> "_MockComfyModelPatcher":
            return _MockComfyModelPatcher(self._diff_mod)

        def get_model_object(self, name: str):
            return self._diff_mod if name == "diffusion_model" else None

    diff_mod = _MiniDiffModel().eval()
    x = torch.randn(1, 4, 16)
    with torch.no_grad():
        expected = diff_mod(x)

    patcher_node = PhobosBlockStreamPatcherNode()
    patched_model, = patcher_node.patch_model(_MockComfyModelPatcher(diff_mod), mode="cpu_offload")
    streamer = getattr(patched_model, "_phobos_block_streamer", None)
    assert streamer is not None

    try:
        with torch.no_grad():
            actual = diff_mod(x)
        assert streamer.blocks_executed == 2
        assert torch.allclose(expected, actual, atol=1e-6)
    finally:
        streamer.detach()


def test_standalone_comfyui_manager_git_clone_import(tmp_path: Path) -> None:
    """Simulate an external user cloning `ComfyUI-PhobosStream` into `ComfyUI/custom_nodes/`."""
    repo_root = Path(__file__).resolve().parent.parent
    ext_src = repo_root / "ComfyUI-PhobosStream"
    assert (ext_src / "phobos_stream").is_dir(), "ComfyUI-PhobosStream must bundle phobos_stream/"

    fake_comfy_root = tmp_path / "ComfyUI"
    fake_custom_nodes = fake_comfy_root / "custom_nodes"
    fake_custom_nodes.mkdir(parents=True)

    # Simulate ComfyUI's own global root `nodes.py` that previously caused a collision!
    (fake_comfy_root / "nodes.py").write_text("NODE_CLASS_MAPPINGS = {}\n", encoding="utf-8")

    cloned_ext = fake_custom_nodes / "ComfyUI-PhobosStream"
    shutil.copytree(ext_src, cloned_ext, ignore=shutil.ignore_patterns(".git", "__pycache__", ".DS_Store"))

    check_code = (
        "import importlib.util, sys\n"
        f"sys.path.insert(0, {str(fake_comfy_root)!r})\n"
        f"spec = importlib.util.spec_from_file_location('ComfyUI_PhobosStream', {str(cloned_ext / '__init__.py')!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "assert len(mod.NODE_CLASS_MAPPINGS) == 5, mod.NODE_CLASS_MAPPINGS\n"
        "print('STANDALONE_OK:', sorted(mod.NODE_CLASS_MAPPINGS.keys()))\n"
    )
    env = {"PATH": ""}
    proc = subprocess.run(
        [sys.executable, "-S", "-c", f"import site; site.main();\n{check_code}"],
        cwd=str(fake_comfy_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Standalone import failed:\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
    assert "STANDALONE_OK:" in proc.stdout

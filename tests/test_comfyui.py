"""Tests for ComfyUI custom node integration."""

import numpy as np
import pytest
from aether_stream.integrations.comfyui.nodes import (
    AetherStreamLoaderNode,
    AetherStreamSamplerNode,
    AetherStreamVAEDecodeNode,
)


def test_comfyui_loader_node() -> None:
    loader = AetherStreamLoaderNode()
    result = loader.load_model(
        model_name="test-wan2.1",
        architecture="wan2.1",
        backend="cpu",
        num_blocks=2,
        hidden_size=64,
    )
    assert len(result) == 1
    model_dict = result[0]
    assert "engine" in model_dict
    assert "manifest" in model_dict


def test_comfyui_sampler_and_vae_decode() -> None:
    loader = AetherStreamLoaderNode()
    model_dict = loader.load_model(
        model_name="test-wan2.1",
        architecture="wan2.1",
        backend="cpu",
        num_blocks=2,
        hidden_size=64,
    )[0]

    sampler = AetherStreamSamplerNode()
    images, latents = sampler.sample(
        aether_model=model_dict,
        prompt="A serene mountain valley with crystal streams",
        negative_prompt="",
        width=128,
        height=128,
        num_frames=2,
        steps=2,
        guidance_scale=5.0,
        seed=42,
    )

    assert images is not None
    assert "samples" in latents

    # Test VAE Decode
    decoder = AetherStreamVAEDecodeNode()
    decoded_images = decoder.decode(latents, model_dict)[0]
    assert decoded_images is not None

"""Tests for ComfyUI custom node integration."""

import numpy as np
import pytest
from phobos_stream.integrations.comfyui.nodes import (
    PhobosStreamLoaderNode,
    PhobosStreamSamplerNode,
    PhobosStreamVAEDecodeNode,
)


def test_comfyui_loader_node() -> None:
    loader = PhobosStreamLoaderNode()
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
    loader = PhobosStreamLoaderNode()
    model_dict = loader.load_model(
        model_name="test-wan2.1",
        architecture="wan2.1",
        backend="cpu",
        num_blocks=2,
        hidden_size=64,
    )[0]

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

    assert images is not None
    # ComfyUI IMAGE tensor must be 4D (Batch/Frames, Height, Width, 3) in [0.0, 1.0]
    assert tuple(images.shape) == (5, 128, 128, 3)
    assert "samples" in latents

    # Test VAE Decode produces identical 4D (5, 128, 128, 3) shape in [0.0, 1.0] without double division
    decoder = PhobosStreamVAEDecodeNode()
    decoded_images = decoder.decode(latents, model_dict)[0]
    assert decoded_images is not None
    assert tuple(decoded_images.shape) == (5, 128, 128, 3)
    assert float(decoded_images.max()) > 0.1

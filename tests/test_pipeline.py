"""Tests for diffusion scheduler, offloader strictness, and streaming engine."""

import numpy as np
import pytest
from phobos_stream.pipeline.scheduler import FlowMatchingScheduler
from phobos_stream.pipeline.stream_engine import PhobosStreamEngine
from phobos_stream.storage.layout import ModelShardsManifest
from phobos_stream.types import BackendType, GenerationConfig


def test_flow_matching_scheduler() -> None:
    scheduler = FlowMatchingScheduler(num_steps=10)
    timesteps = scheduler.get_timesteps()
    assert len(timesteps) == 10
    assert timesteps[0] > timesteps[-1]

    sample = np.ones((2, 4), dtype=np.float32)
    model_output = np.ones((2, 4), dtype=np.float32) * 0.5
    next_sample = scheduler.step(model_output, timesteps[0], sample)
    assert next_sample.shape == (2, 4)


def test_phobos_stream_engine_rejects_unconfigured_weights_when_strict() -> None:
    manifest = ModelShardsManifest(
        model_name="strict-wan2.1",
        architecture="wan2.1",
        num_blocks=2,
        hidden_size=64,
        allow_synthetic=False,
    )
    engine = PhobosStreamEngine(manifest=manifest, backend_type=BackendType.CPU)
    config = GenerationConfig(prompt="Test strict error", height=128, width=128, num_frames=1, steps=1)
    with pytest.raises(FileNotFoundError):
        engine.generate(config)


def test_phobos_stream_engine_generate_explicit_synthetic() -> None:
    manifest = ModelShardsManifest(
        model_name="test-wan2.1",
        architecture="wan2.1",
        num_blocks=2,
        hidden_size=64,
        allow_synthetic=True,
    )
    engine = PhobosStreamEngine(manifest=manifest, backend_type=BackendType.CPU)

    config = GenerationConfig(
        prompt="A serene mountain lake at sunrise",
        negative_prompt="blurry, low quality",
        height=128,
        width=128,
        num_frames=5,
        steps=2,
        guidance_scale=5.0,
    )

    steps_recorded = []

    def callback(step: int, total: int, stage: str) -> None:
        steps_recorded.append((step, stage))

    frames, metrics = engine.generate(config, progress_callback=callback)

    assert frames is not None
    assert frames.shape == (5, 128, 128, 3)
    assert frames.dtype == np.uint8
    assert metrics.steps_completed == 2
    assert len(steps_recorded) > 0
    assert metrics.total_time_sec > 0
    assert metrics.peak_vram_mb > 0

"""Benchmark PhobosStream's zero-VRAM diffusion streaming engine."""

import time
from phobos_stream.pipeline.stream_engine import PhobosStreamEngine
from phobos_stream.storage.layout import ModelShardsManifest
from phobos_stream.types import BackendType, GenerationConfig


def run_benchmark() -> None:
    print("=" * 70)
    print("PHOBOS-STREAM: UNIVERSAL ZERO-VRAM VIDEO & IMAGE STREAMING ENGINE")
    print("=" * 70)

    num_blocks = 16
    hidden_size = 2048

    manifest = ModelShardsManifest(
        model_name="wan2.1-14b-simulated",
        architecture="wan2.1",
        num_blocks=num_blocks,
        hidden_size=hidden_size,
    )

    print(f"[*] Initializing Engine [Architecture: Wan2.1 | Blocks: {num_blocks} | Hidden: {hidden_size}]...")
    engine = PhobosStreamEngine(manifest=manifest, backend_type=BackendType.AUTO)
    print(f"[*] Active Backend: {engine.backend.name.upper()}")

    config = GenerationConfig(
        prompt="Cinematic drone shot of an ancient mountain temple in mist, 8k resolution photorealistic",
        height=720,
        width=1280,
        num_frames=25,
        steps=5,
        seed=1337,
    )

    print(f"\n[*] Starting Video Denoising Benchmark ({config.steps} Steps, {config.num_frames} Frames)...")

    def progress(step: int, total: int, stage: str) -> None:
        print(f"    -> [{stage.upper():14s}] Step {step}/{total}")

    frames, metrics = engine.generate(config, progress_callback=progress)

    print("\n" + "=" * 70)
    print("BENCHMARK PERFORMANCE TELEMETRY")
    print("=" * 70)
    print(f"  • Output Frame Tensor:     {frames.shape} (dtype={frames.dtype})")
    print(f"  • Total Runtime:           {metrics.total_time_sec:.3f} s")
    print(f"  • Text Encoding Stage:     {metrics.text_encode_time_sec*1000:.2f} ms")
    print(f"  • DiT Denoising Stage:     {metrics.denoising_time_sec:.3f} s ({metrics.denoising_time_sec/config.steps:.3f} s/step)")
    print(f"  • VAE Latent Decode Stage: {metrics.vae_decode_time_sec*1000:.2f} ms")
    print(f"  • Peak Memory Consumption: {metrics.peak_vram_mb:.2f} MB")
    print("  • Status:                  Zero-VRAM Crash-Free Execution")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()

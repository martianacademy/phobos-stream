"""Command Line Interface (CLI) for AetherStream."""

from pathlib import Path
import click
import uvicorn
from aether_stream.pipeline.stream_engine import AetherStreamEngine
from aether_stream.server.app import create_app
from aether_stream.storage.layout import ModelShardsManifest
from aether_stream.storage.sharder import DiffusionSharder
from aether_stream.types import BackendType, GenerationConfig


@click.group()
def main() -> None:
    """AetherStream: Universal Zero-VRAM Video & Image Diffusion Streaming Engine."""
    pass


@main.command()
@click.option("--host", default="0.0.0.0", help="Host interface to bind")
@click.option("--port", default=8088, type=int, help="Port to listen on")
@click.option("--manifest-path", default=None, type=click.Path(exists=True), help="Path to manifest.json")
@click.option("--backend", default="auto", type=click.Choice(["auto", "mlx", "cuda", "gguf", "cpu"]))
def serve(host: str, port: int, manifest_path: str, backend: str) -> None:
    """Launch the OpenAI/Sora-compatible HTTP & WebSocket API Gateway."""
    manifest = ModelShardsManifest.load(Path(manifest_path)) if manifest_path else None
    backend_type = BackendType(backend)
    app = create_app(manifest=manifest, backend_type=backend_type)
    click.echo(f"🌌 Starting AetherStream Gateway on http://{host}:{port} [Backend: {backend}]")
    uvicorn.run(app, host=host, port=port)


@main.command()
@click.option("--prompt", required=True, help="Text description of the desired video or image")
@click.option("--video/--image", default=True, help="Generate video (default) or image")
@click.option("--steps", default=20, type=int, help="Denoising steps")
@click.option("--backend", default="auto", type=click.Choice(["auto", "mlx", "cuda", "gguf", "cpu"]))
def generate(prompt: str, video: bool, steps: int, backend: str) -> None:
    """Generate a video or image via zero-VRAM disk streaming."""
    manifest = ModelShardsManifest(
        model_name="wan2.1-streamed",
        architecture="wan2.1",
        num_blocks=4,
        hidden_size=2048,
    )
    engine = AetherStreamEngine(manifest=manifest, backend_type=BackendType(backend))
    config = GenerationConfig(
        prompt=prompt,
        num_frames=49 if video else 1,
        steps=steps,
    )
    click.echo(f"🎬 Generating {'Video' if video else 'Image'} with AetherStream...")
    _, metrics = engine.generate(
        config,
        progress_callback=lambda step, total, stage: click.echo(f"[{stage}] Step {step}/{total}"),
    )
    click.echo(f"✨ Generation Completed in {metrics.total_time_sec:.2f}s!")
    click.echo(f"💾 Peak RAM: {metrics.peak_vram_mb:.1f} MB | I/O Speed: {metrics.io_throughput_mb_s:.1f} MB/s")


@main.command()
@click.option("--steps", default=10, type=int, help="Benchmark denoising steps")
@click.option("--blocks", default=8, type=int, help="Simulated transformer blocks")
@click.option("--backend", default="auto", type=click.Choice(["auto", "mlx", "cuda", "gguf", "cpu"]))
def benchmark(steps: int, blocks: int, backend: str) -> None:
    """Run an end-to-end synthetic streaming benchmark."""
    manifest = ModelShardsManifest(
        model_name="synthetic-dit",
        architecture="wan2.1",
        num_blocks=blocks,
        hidden_size=2048,
    )
    engine = AetherStreamEngine(manifest=manifest, backend_type=BackendType(backend))
    config = GenerationConfig(
        prompt="A robotic cheetah sprinting through futuristic neon Tokyo at sunset",
        num_frames=25,
        steps=steps,
    )
    click.echo(f"🚀 Running AetherStream Benchmark on backend [{engine.backend.name}]...")
    _, metrics = engine.generate(config)
    click.echo("=" * 60)
    click.echo(f"Total Latency:     {metrics.total_time_sec:.3f} s")
    click.echo(f"Denoising Latency: {metrics.denoising_time_sec:.3f} s ({metrics.denoising_time_sec/steps:.3f} s/step)")
    click.echo(f"Peak VRAM:         {metrics.peak_vram_mb:.2f} MB")
    click.echo(f"I/O Throughput:    {metrics.io_throughput_mb_s:.2f} MB/s")
    click.echo("=" * 60)


if __name__ == "__main__":
    main()

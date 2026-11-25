"""Command Line Interface (CLI) for PhobosStream."""

from pathlib import Path
from typing import Optional
import click
import uvicorn

from phobos_stream.media import save_rgb_mp4, save_rgb_png
from phobos_stream.pipeline.stream_engine import PhobosStreamEngine
from phobos_stream.server.app import create_app
from phobos_stream.storage.layout import ModelShardsManifest
from phobos_stream.storage.sharder import DiffusionSharder
from phobos_stream.types import BackendType, GenerationConfig


@click.group()
def main() -> None:
    """PhobosStream: Universal Zero-VRAM Video & Image Diffusion Streaming Engine."""
    pass


@main.command()
@click.option("--host", default="0.0.0.0", help="Host interface to bind")
@click.option("--port", default=8088, type=int, help="Port to listen on")
@click.option("--manifest-path", default=None, type=click.Path(exists=True), help="Path to manifest.json")
@click.option("--backend", default="auto", type=click.Choice(["auto", "mlx", "cuda", "gguf", "cpu"]))
@click.option("--allow-synthetic", is_flag=True, default=False, help="Allow synthetic benchmark weights when no manifest is set")
def serve(
    host: str,
    port: int,
    manifest_path: Optional[str],
    backend: str,
    allow_synthetic: bool,
) -> None:
    """Launch the OpenAI/Sora-compatible HTTP & WebSocket API Gateway."""
    manifest = ModelShardsManifest.load(Path(manifest_path)) if manifest_path else None
    backend_type = BackendType(backend)
    app = create_app(manifest=manifest, backend_type=backend_type, allow_synthetic=allow_synthetic)
    click.echo(f"[*] Starting PhobosStream Gateway on http://{host}:{port} [Backend: {backend}]")
    uvicorn.run(app, host=host, port=port)


@main.command()
@click.option("--prompt", required=True, help="Text description of the desired video or image")
@click.option("--negative-prompt", default="", help="Negative prompt for guidance")
@click.option("--video/--image", default=True, help="Generate video (default) or image")
@click.option("--height", default=480, type=int, help="Output height in pixels")
@click.option("--width", default=720, type=int, help="Output width in pixels")
@click.option("--frames", default=17, type=int, help="Number of video frames when --video is set")
@click.option("--fps", default=24, type=int, help="Video frame rate when --video is set")
@click.option("--steps", default=20, type=int, help="Denoising steps")
@click.option("--guidance-scale", default=6.0, type=float, help="Classifier-free guidance scale")
@click.option("--manifest-path", default=None, type=click.Path(exists=True), help="Path to shard manifest.json")
@click.option("--allow-synthetic", is_flag=True, default=False, help="Explicitly run synthetic benchmark mode without real weights")
@click.option("--output", "-o", default="outputs/generated.mp4", help="Output file path (.mp4 or .png)")
@click.option("--backend", default="auto", type=click.Choice(["auto", "mlx", "cuda", "gguf", "cpu"]))
def generate(
    prompt: str,
    negative_prompt: str,
    video: bool,
    height: int,
    width: int,
    frames: int,
    fps: int,
    steps: int,
    guidance_scale: float,
    manifest_path: Optional[str],
    allow_synthetic: bool,
    output: str,
    backend: str,
) -> None:
    """Generate a video or image via zero-VRAM streaming and save to disk."""
    if manifest_path:
        manifest = ModelShardsManifest.load(Path(manifest_path))
        if allow_synthetic:
            manifest.allow_synthetic = True
    elif allow_synthetic:
        manifest = ModelShardsManifest(
            model_name="wan2.1-synthetic-benchmark",
            architecture="wan2.1",
            num_blocks=4,
            hidden_size=2048,
            allow_synthetic=True,
        )
    else:
        raise click.UsageError(
            "Real generation requires `--manifest-path <path/to/manifest.json>` (or use ComfyUI / "
            "`StageSequentialRunner`). Pass `--allow-synthetic` only for synthetic benchmarking."
        )

    engine = PhobosStreamEngine(manifest=manifest, backend_type=BackendType(backend))
    config = GenerationConfig(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=frames if video else 1,
        fps=fps,
        steps=steps,
        guidance_scale=guidance_scale,
    )
    click.echo(f"[*] Generating {'Video' if video else 'Image'} with PhobosStream...")
    rgb_frames, metrics = engine.generate(
        config,
        progress_callback=lambda step, total, stage: click.echo(f"[{stage}] Step {step}/{total}"),
    )

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if video and rgb_frames.shape[0] > 1:
        mp4_path = save_rgb_mp4(rgb_frames, out_path.with_suffix(".mp4"), fps=fps)
        preview_path = save_rgb_png(rgb_frames[0], out_path.with_suffix(".png"))
        click.echo(f"[+] Saved MP4 video {rgb_frames.shape} to {mp4_path} and preview to {preview_path}")
    else:
        png_path = save_rgb_png(rgb_frames[0], out_path.with_suffix(".png"))
        click.echo(f"[+] Saved PNG image {rgb_frames[0].shape} to {png_path}")

    click.echo(f"[+] Generation Completed in {metrics.total_time_sec:.2f}s!")
    click.echo(f"[+] Peak Memory: {metrics.peak_vram_mb:.1f} MB | I/O Speed: {metrics.io_throughput_mb_s:.1f} MB/s")


@main.command()
@click.option("--model-dir", required=True, type=click.Path(exists=True), help="Directory or .safetensors file")
@click.option("--output-dir", default=None, type=click.Path(), help="Output directory for shards and manifest.json")
@click.option("--architecture", default="wan2.1", help="Model architecture (wan2.1, flux, hunyuan, cogvideox)")
@click.option("--hidden-size", default=4096, type=int, help="Transformer hidden dimension")
@click.option("--partition/--virtual", default=False, help="Physically split block files or write virtual manifest")
def shard(
    model_dir: str,
    output_dir: Optional[str],
    architecture: str,
    hidden_size: int,
    partition: bool,
) -> None:
    """Index or physically shard safetensors weights for zero-copy block streaming."""
    model_path = Path(model_dir)
    out_path = (
        Path(output_dir)
        if output_dir
        else (model_path.parent / "shards" if model_path.is_file() else model_path / "shards")
    )
    sharder = DiffusionSharder(model_dir=model_path, output_dir=out_path)
    manifest = sharder.build_virtual_manifest(architecture=architecture, hidden_size=hidden_size)
    if partition:
        manifest_file = sharder.partition_to_disk(manifest)
    else:
        out_path.mkdir(parents=True, exist_ok=True)
        manifest_file = out_path / "manifest.json"
        manifest.save(manifest_file)
    click.echo(f"[+] Indexed {manifest.num_blocks} blocks -> {manifest_file}")


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
        allow_synthetic=True,
    )
    engine = PhobosStreamEngine(manifest=manifest, backend_type=BackendType(backend))
    config = GenerationConfig(
        prompt="A robotic cheetah sprinting through futuristic neon Tokyo at sunset",
        num_frames=25,
        steps=steps,
    )
    click.echo(f"[*] Running PhobosStream Synthetic Benchmark on backend [{engine.backend.name}]...")
    rgb_frames, metrics = engine.generate(config)
    click.echo("=" * 60)
    click.echo(f"Output Shape:      {rgb_frames.shape} (uint8 RGB)")
    click.echo(f"Total Latency:     {metrics.total_time_sec:.3f} s")
    click.echo(f"Denoising Latency: {metrics.denoising_time_sec:.3f} s ({metrics.denoising_time_sec / steps:.3f} s/step)")
    click.echo(f"Peak VRAM/RAM:     {metrics.peak_vram_mb:.2f} MB")
    click.echo(f"I/O Throughput:    {metrics.io_throughput_mb_s:.2f} MB/s")
    click.echo("=" * 60)


if __name__ == "__main__":
    main()

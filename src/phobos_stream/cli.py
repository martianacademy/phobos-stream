"""Command Line Interface (CLI) for PhobosStream."""

from pathlib import Path
import struct
import zlib
import click
import numpy as np
import uvicorn
from phobos_stream.pipeline.stream_engine import PhobosStreamEngine
from phobos_stream.server.app import create_app
from phobos_stream.storage.layout import ModelShardsManifest
from phobos_stream.storage.sharder import DiffusionSharder
from phobos_stream.types import BackendType, GenerationConfig


def save_rgb_png(rgb_frame: np.ndarray, path: Path) -> None:
    """Save a (H, W, 3) uint8 RGB array as a valid PNG file without external dependencies."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
    height, width, channels = arr.shape
    if channels != 3:
        raise ValueError(f"Expected 3 RGB channels, got shape {arr.shape}")

    # Prepend filter byte 0 to each scanline
    scanlines = np.hstack([np.zeros((height, 1), dtype=np.uint8), arr.reshape(height, width * 3)])
    compressed = zlib.compress(scanlines.tobytes(), level=6)

    def make_chunk(chunk_type: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        + make_chunk(b"IHDR", ihdr)
        + make_chunk(b"IDAT", compressed)
        + make_chunk(b"IEND", b"")
    )
    path.write_bytes(png_bytes)


@click.group()
def main() -> None:
    """PhobosStream: Universal Zero-VRAM Video & Image Diffusion Streaming Engine."""
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
    click.echo(f"[*] Starting PhobosStream Gateway on http://{host}:{port} [Backend: {backend}]")
    uvicorn.run(app, host=host, port=port)


@main.command()
@click.option("--prompt", required=True, help="Text description of the desired video or image")
@click.option("--negative-prompt", default="", help="Negative prompt for guidance")
@click.option("--video/--image", default=True, help="Generate video (default) or image")
@click.option("--height", default=480, type=int, help="Output height in pixels")
@click.option("--width", default=720, type=int, help="Output width in pixels")
@click.option("--frames", default=17, type=int, help="Number of video frames when --video is set")
@click.option("--steps", default=20, type=int, help="Denoising steps")
@click.option("--guidance-scale", default=6.0, type=float, help="Classifier-free guidance scale")
@click.option("--manifest-path", default=None, type=click.Path(exists=True), help="Optional path to manifest.json")
@click.option("--output", "-o", default="outputs/generated.png", help="Output file path")
@click.option("--backend", default="auto", type=click.Choice(["auto", "mlx", "cuda", "gguf", "cpu"]))
def generate(
    prompt: str,
    negative_prompt: str,
    video: bool,
    height: int,
    width: int,
    frames: int,
    steps: int,
    guidance_scale: float,
    manifest_path: str,
    output: str,
    backend: str,
) -> None:
    """Generate a video or image via zero-VRAM disk streaming and save to disk."""
    if manifest_path:
        manifest = ModelShardsManifest.load(Path(manifest_path))
    else:
        manifest = ModelShardsManifest(
            model_name="wan2.1-streamed",
            architecture="wan2.1",
            num_blocks=4,
            hidden_size=2048,
        )
    engine = PhobosStreamEngine(manifest=manifest, backend_type=BackendType(backend))
    config = GenerationConfig(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=frames if video else 1,
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
    # Save first/key frame as PNG and full video tensor if multi-frame
    png_path = out_path if out_path.suffix.lower() == ".png" else out_path.with_suffix(".png")
    save_rgb_png(rgb_frames[0], png_path)
    if video and rgb_frames.shape[0] > 1:
        npy_path = out_path.with_suffix(".npy")
        np.save(npy_path, rgb_frames)
        click.echo(f"[+] Saved video frames tensor {rgb_frames.shape} to {npy_path} and preview to {png_path}")
    else:
        click.echo(f"[+] Saved image {rgb_frames[0].shape} to {png_path}")

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
    output_dir: str,
    architecture: str,
    hidden_size: int,
    partition: bool,
) -> None:
    """Index or physically shard safetensors weights for zero-copy block streaming."""
    model_path = Path(model_dir)
    out_path = Path(output_dir) if output_dir else (model_path.parent / "shards" if model_path.is_file() else model_path / "shards")
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
    )
    engine = PhobosStreamEngine(manifest=manifest, backend_type=BackendType(backend))
    config = GenerationConfig(
        prompt="A robotic cheetah sprinting through futuristic neon Tokyo at sunset",
        num_frames=25,
        steps=steps,
    )
    click.echo(f"[*] Running PhobosStream Benchmark on backend [{engine.backend.name}]...")
    rgb_frames, metrics = engine.generate(config)
    click.echo("=" * 60)
    click.echo(f"Output Shape:      {rgb_frames.shape} (uint8 RGB)")
    click.echo(f"Total Latency:     {metrics.total_time_sec:.3f} s")
    click.echo(f"Denoising Latency: {metrics.denoising_time_sec:.3f} s ({metrics.denoising_time_sec/steps:.3f} s/step)")
    click.echo(f"Peak VRAM/RAM:     {metrics.peak_vram_mb:.2f} MB")
    click.echo(f"I/O Throughput:    {metrics.io_throughput_mb_s:.2f} MB/s")
    click.echo("=" * 60)


if __name__ == "__main__":
    main()

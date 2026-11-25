"""CLI entry script for Apple Metal MLX Z-Image-Turbo generation using `phobos_stream.pipelines`."""

import argparse
from pathlib import Path
from PIL import Image

from phobos_stream.pipelines import generate_zimage_mlx

_HOME = Path.home()
DEFAULT_QWEN = str(_HOME / "ComfyUI-Shared/models/text_encoders/qwen_3_4b.safetensors")
DEFAULT_DIT = str(_HOME / "ComfyUI-Shared/models/diffusion_models/z_image_turbo_bf16.safetensors")
DEFAULT_VAE = str(_HOME / "ComfyUI-Shared/models/vae/ae.safetensors")
DEFAULT_TOK = str(_HOME / "ComfyUI-Installs/ComfyUI/ComfyUI/comfy/text_encoders/qwen25_tokenizer")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate images with Z-Image-Turbo via PhobosStream MLX.")
    parser.add_argument(
        "--prompt",
        default=(
            "A hyper-detailed futuristic Martian base on red crater landscape, domes with glowing "
            "warm atmospheric lights, starry galaxy background, photorealistic, 8k resolution, cinematic masterpiece"
        ),
    )
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=["stage_offload", "q4_stage_offload", "block_stream", "monolithic"],
        default="block_stream",
    )
    parser.add_argument("--qwen-path", default=DEFAULT_QWEN)
    parser.add_argument("--dit-path", default=DEFAULT_DIT)
    parser.add_argument("--vae-path", default=DEFAULT_VAE)
    parser.add_argument("--tokenizer-dir", default=DEFAULT_TOK)
    parser.add_argument("--output", default="outputs/z_image_mlx_phobos.png")
    args = parser.parse_args()

    res = generate_zimage_mlx(
        prompt=args.prompt,
        qwen_path=args.qwen_path,
        dit_path=args.dit_path,
        vae_path=args.vae_path,
        tokenizer_dir=args.tokenizer_dir,
        width=args.width,
        height=args.height,
        steps=args.steps,
        seed=args.seed,
        mode=args.mode,
    )
    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(res.image).save(out_file)
    print(
        f"[+] Saved {out_file} | Mode: {res.mode} | Wall: {res.wall_sec}s "
        f"| Peak Metal: {res.peak_metal_mb} MB | Streamed: {res.streamed_gb} GB"
    )


if __name__ == "__main__":
    main()

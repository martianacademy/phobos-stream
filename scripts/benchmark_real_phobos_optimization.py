"""Apples-to-Apples Real-Model Benchmark on 19.5 GB Z-Image-Turbo Suite (Apple Metal GPU).

Runs each mode in an isolated fresh subprocess so `mx.get_peak_memory()` and macOS `ru_maxrss`
are uncontaminated across modes:
  1. `monolithic`       : Baseline (Qwen 3.4B + 30-Block DiT + VAE kept in Metal RAM simultaneously)
  2. `stage_offload`    : PhobosStream 3-Stage Sequential Offload + Fused Metal Kernels
  3. `q4_stage_offload` : PhobosStream 3-Stage Offload + 4-Bit MLX Quantized Linear
  4. `block_stream`     : PhobosStream Zero-VRAM 3-Stage + Per-Block MMap Streaming
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from PIL import Image
import numpy as np

from phobos_stream.pipelines import generate_zimage_mlx

_HOME = Path.home()
QWEN_PATH = str(_HOME / "ComfyUI-Shared/models/text_encoders/qwen_3_4b.safetensors")
DIT_PATH = str(_HOME / "ComfyUI-Shared/models/diffusion_models/z_image_turbo_bf16.safetensors")
VAE_PATH = str(_HOME / "ComfyUI-Shared/models/vae/ae.safetensors")
TOKENIZER_DIR = str(_HOME / "ComfyUI-Installs/ComfyUI/ComfyUI/comfy/text_encoders/qwen25_tokenizer")

DEFAULT_PROMPT = (
    "A hyper-detailed futuristic Martian base on red crater landscape, "
    "domes with glowing warm atmospheric lights, starry galaxy background, "
    "photorealistic, 8k resolution, cinematic masterpiece"
)


def _psnr(img_a: np.ndarray, img_b: np.ndarray) -> float:
    mse = float(np.mean((img_a.astype(np.float32) - img_b.astype(np.float32)) ** 2))
    if mse < 1e-10:
        return 99.9
    return round(20.0 * float(np.log10(255.0 / np.sqrt(mse))), 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--single-mode",
        choices=["monolithic", "stage_offload", "q4_stage_offload", "block_stream"],
        default=None,
    )
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "real_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.single_mode:
        out_img = out_dir / f"{args.single_mode}_{args.width}x{args.height}.png"
        print(f"\n>>> STARTING [{args.single_mode}] ({args.width}x{args.height}, {args.steps} steps) <<<", flush=True)
        res = generate_zimage_mlx(
            prompt=DEFAULT_PROMPT,
            qwen_path=QWEN_PATH,
            dit_path=DIT_PATH,
            vae_path=VAE_PATH,
            tokenizer_dir=TOKENIZER_DIR,
            width=args.width,
            height=args.height,
            steps=args.steps,
            seed=args.seed,
            mode=args.single_mode,
        )
        Image.fromarray(res.image).save(out_img)
        payload = {k: v for k, v in asdict(res).items() if k != "image"}
        payload["output"] = str(out_img)
        print(f">>> FINISHED [{args.single_mode}]: {json.dumps(payload)}", flush=True)
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return

    modes = ["monolithic", "stage_offload", "q4_stage_offload", "block_stream"]
    results: dict[str, dict] = {}
    for m in modes:
        j_file = out_dir / f"{m}.json"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--single-mode",
            m,
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--steps",
            str(args.steps),
            "--seed",
            str(args.seed),
            "--json-out",
            str(j_file),
        ]
        subprocess.run(cmd, check=True)
        results[m] = json.loads(j_file.read_text(encoding="utf-8"))

    base_img = np.asarray(Image.open(results["monolithic"]["output"]))
    base_peak = results["monolithic"]["peak_metal_mb"]
    base_den = results["monolithic"]["denoise_sec"]

    print("\n" + "=" * 95)
    print("REAL 19.5 GB MODEL BENCHMARK SUMMARY (Qwen 3.4B + Z-Image-Turbo BF16 + VAE @ 1024x1024, 8 Steps)")
    print("=" * 95)
    for m in modes:
        r = results[m]
        img_m = np.asarray(Image.open(r["output"]))
        psnr_val = _psnr(base_img, img_m)
        mae_val = round(float(np.mean(np.abs(base_img.astype(np.float32) - img_m.astype(np.float32)))), 2)
        mem_saved_pct = round((1.0 - r["peak_metal_mb"] / base_peak) * 100.0, 1)
        speedup = round(base_den / r["denoise_sec"], 2) if r["denoise_sec"] > 0 else 0.0
        r["psnr_vs_baseline_db"] = psnr_val
        r["mae_vs_baseline_px"] = mae_val
        r["metal_vram_saved_pct"] = mem_saved_pct
        r["denoise_speedup_vs_base"] = speedup
        print(
            f"• {m:17s} | Peak Metal: {r['peak_metal_mb']:8.1f} MB ({mem_saved_pct:+5.1f}% saved) "
            f"| Denoise: {r['denoise_sec']:5.2f}s ({r['step_sec']:4.2f}s/step, {speedup:.2f}x) "
            f"| Wall: {r['wall_sec']:5.2f}s | PSNR: {psnr_val:5.2f} dB"
        )
    print("=" * 95)
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

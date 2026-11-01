"""2K (2048x1152) Benchmark: PhobosStream (Stage-Offload + Block-Stream) vs Direct Monolithic on Z-Image-Turbo."""

import gc
import os
import resource
import sys
import time
from pathlib import Path
from PIL import Image
import torch

COMFY_DIR = "/Users/suru.martian/ComfyUI-Installs/ComfyUI/ComfyUI"
if COMFY_DIR not in sys.path:
    sys.path.insert(0, COMFY_DIR)

import folder_paths
folder_paths.add_model_folder_path("diffusion_models", "/Users/suru.martian/ComfyUI-Shared/models/diffusion_models")
folder_paths.add_model_folder_path("text_encoders", "/Users/suru.martian/ComfyUI-Shared/models/text_encoders")
folder_paths.add_model_folder_path("vae", "/Users/suru.martian/ComfyUI-Shared/models/vae")

import comfy.model_management as mm
import nodes
from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow


PROMPT = (
    "A hyper-detailed futuristic Martian base on red crater landscape, "
    "domes with glowing warm atmospheric lights, starry galaxy background, "
    "photorealistic, 8k resolution, cinematic masterpiece"
)
WIDTH, HEIGHT = 2048, 1152
STEPS = 8


def get_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def free_all_memory() -> None:
    mm.unload_all_models()
    mm.soft_empty_cache()
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def run_phobos_stage_offloaded(out_dir: Path) -> dict:
    """Mode 1: Phobos Stage-Wise Offloading (Only 1 model active in RAM/VRAM at any stage)."""
    print("\n" + "=" * 75)
    print("[1/2] RUNNING WITH PHOBOS STAGE-OFFLOADING (2K: 2048x1152)")
    print("=" * 75)
    free_all_memory()
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    t_start = time.perf_counter()

    # Stage 1: Load ONLY Qwen 3.4B Text Encoder (7.67 GB), encode prompt, then immediately unload!
    t0 = time.perf_counter()
    clip, = nodes.CLIPLoader().load_clip("qwen_3_4b.safetensors", "lumina2")
    positive, = nodes.CLIPTextEncode().encode(clip, PROMPT)
    negative, = nodes.ConditioningZeroOut().zero_out(positive)
    del clip
    free_all_memory()
    t_text = time.perf_counter() - t0
    rss_after_text = get_rss_mb()
    print(f"  -> Stage 1 (Text Encode + Offload): {t_text:.2f}s | Peak RSS: {rss_after_text:.1f} MB")

    # Stage 2: Load ONLY Z-Image-Turbo DiT (11.74 GB), denoise 2K latents, then immediately unload!
    t0 = time.perf_counter()
    model, = nodes.UNETLoader().load_unet("z_image_turbo_bf16.safetensors", "default")
    shifted_model, = ModelSamplingAuraFlow().patch_aura(model, 3.0)
    latent = {
        "samples": torch.zeros([1, 16, HEIGHT // 8, WIDTH // 8], device="cpu", dtype=torch.float32),
        "downscale_ratio_spacial": 8,
    }
    samples, = nodes.KSampler().sample(
        model=shifted_model,
        seed=42,
        steps=STEPS,
        cfg=1.0,
        sampler_name="res_multistep",
        scheduler="simple",
        positive=positive,
        negative=negative,
        latent_image=latent,
        denoise=1.0,
    )
    del shifted_model, model, positive, negative
    free_all_memory()
    t_dit = time.perf_counter() - t0
    rss_after_dit = get_rss_mb()
    print(f"  -> Stage 2 (2K DiT Denoise + Offload): {t_dit:.2f}s | Peak RSS: {rss_after_dit:.1f} MB")

    # Stage 3: Load ONLY VAE (320 MB), decode 2K RGB image, then unload!
    t0 = time.perf_counter()
    vae, = nodes.VAELoader().load_vae("ae.safetensors")
    decoded, = nodes.VAEDecode().decode(vae, samples)
    del vae, samples
    free_all_memory()
    t_vae = time.perf_counter() - t0

    img_np = (decoded[0].detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
    out_file = out_dir / "z_image_turbo_2k_phobos.png"
    Image.fromarray(img_np).save(out_file)

    total_wall = time.perf_counter() - t_start
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    u_cpu = ru1.ru_utime - ru0.ru_utime
    s_cpu = ru1.ru_stime - ru0.ru_stime
    cpu_pct = ((u_cpu + s_cpu) / total_wall) * 100.0
    peak_rss = get_rss_mb()
    print(f"  -> Stage 3 (2K VAE Decode + Save): {t_vae:.2f}s | Saved: {out_file}")
    return {
        "wall": total_wall,
        "text": t_text,
        "dit": t_dit,
        "vae": t_vae,
        "cpu_pct": cpu_pct,
        "u_cpu": u_cpu,
        "s_cpu": s_cpu,
        "peak_rss_mb": peak_rss,
        "active_model_gb": 11.74,
        "out_file": str(out_file),
    }


def run_direct_monolithic(out_dir: Path) -> dict:
    """Mode 2: Direct Monolithic (All 3 models — 19.73 GB — loaded & kept in memory simultaneously)."""
    print("\n" + "=" * 75)
    print("[2/2] RUNNING DIRECT MONOLITHIC - BINA PHOBOS KE (2K: 2048x1152)")
    print("=" * 75)
    free_all_memory()
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    t_start = time.perf_counter()

    # Load ALL models into memory simultaneously (DiT 11.74 GB + Qwen 7.67 GB + VAE 0.32 GB = 19.73 GB)
    t0 = time.perf_counter()
    model, = nodes.UNETLoader().load_unet("z_image_turbo_bf16.safetensors", "default")
    clip, = nodes.CLIPLoader().load_clip("qwen_3_4b.safetensors", "lumina2")
    vae, = nodes.VAELoader().load_vae("ae.safetensors")
    t_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    positive, = nodes.CLIPTextEncode().encode(clip, PROMPT)
    negative, = nodes.ConditioningZeroOut().zero_out(positive)
    t_text = time.perf_counter() - t0

    t0 = time.perf_counter()
    shifted_model, = ModelSamplingAuraFlow().patch_aura(model, 3.0)
    latent = {
        "samples": torch.zeros([1, 16, HEIGHT // 8, WIDTH // 8], device="cpu", dtype=torch.float32),
        "downscale_ratio_spacial": 8,
    }
    samples, = nodes.KSampler().sample(
        model=shifted_model,
        seed=42,
        steps=STEPS,
        cfg=1.0,
        sampler_name="res_multistep",
        scheduler="simple",
        positive=positive,
        negative=negative,
        latent_image=latent,
        denoise=1.0,
    )
    t_dit = time.perf_counter() - t0

    t0 = time.perf_counter()
    decoded, = nodes.VAEDecode().decode(vae, samples)
    t_vae = time.perf_counter() - t0

    img_np = (decoded[0].detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
    out_file = out_dir / "z_image_turbo_2k_direct.png"
    Image.fromarray(img_np).save(out_file)

    total_wall = time.perf_counter() - t_start
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    u_cpu = ru1.ru_utime - ru0.ru_utime
    s_cpu = ru1.ru_stime - ru0.ru_stime
    cpu_pct = ((u_cpu + s_cpu) / total_wall) * 100.0
    peak_rss = get_rss_mb()
    print(f"  -> Direct Complete in {total_wall:.2f}s | Peak RSS: {peak_rss:.1f} MB | Saved: {out_file}")
    return {
        "wall": total_wall,
        "load": t_load,
        "text": t_text,
        "dit": t_dit,
        "vae": t_vae,
        "cpu_pct": cpu_pct,
        "u_cpu": u_cpu,
        "s_cpu": s_cpu,
        "peak_rss_mb": peak_rss,
        "active_model_gb": 19.73,
        "out_file": str(out_file),
    }


if __name__ == "__main__":
    out_dir = Path("/Users/suru.martian/Documents/phobos-stream/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    res_phobos = run_phobos_stage_offloaded(out_dir)
    res_direct = run_direct_monolithic(out_dir)
    print("\n" + "=" * 75)
    print("FINAL 2K (2048x1152) COMPARISON SUMMARY")
    print("=" * 75)
    print(f"Phobos Stage-Offload -> Wall: {res_phobos['wall']:.2f}s | CPU: {res_phobos['cpu_pct']:.1f}% | Active Weights: {res_phobos['active_model_gb']:.2f} GB | Peak RSS: {res_phobos['peak_rss_mb']:.1f} MB")
    print(f"Direct Monolithic    -> Wall: {res_direct['wall']:.2f}s | CPU: {res_direct['cpu_pct']:.1f}% | Active Weights: {res_direct['active_model_gb']:.2f} GB | Peak RSS: {res_direct['peak_rss_mb']:.1f} MB")

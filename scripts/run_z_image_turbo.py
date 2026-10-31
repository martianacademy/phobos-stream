"""Benchmark runner for Z-Image-Turbo on Apple Silicon Metal GPU."""

import os
import sys
import time
from pathlib import Path

# Add ComfyUI directory to sys.path
COMFY_DIR = "/Users/suru.martian/ComfyUI-Installs/ComfyUI/ComfyUI"
if COMFY_DIR not in sys.path:
    sys.path.insert(0, COMFY_DIR)

import folder_paths

# Register model search directories
folder_paths.add_model_folder_path("diffusion_models", "/Users/suru.martian/ComfyUI-Shared/models/diffusion_models")
folder_paths.add_model_folder_path("text_encoders", "/Users/suru.martian/ComfyUI-Shared/models/text_encoders")
folder_paths.add_model_folder_path("vae", "/Users/suru.martian/ComfyUI-Shared/models/vae")

import nodes
from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow
import torch
from PIL import Image


def benchmark_z_image_turbo() -> None:
    print("=== Z-Image-Turbo Apple Silicon Metal (MPS) Benchmark ===")
    out_dir = Path("/Users/suru.martian/Documents/phobos-stream/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load UNET (DiT)
    print("\n[1/6] Loading DiT model (z_image_turbo_bf16.safetensors 11 GB)...")
    t0 = time.time()
    unet_loader = nodes.UNETLoader()
    model, = unet_loader.load_unet("z_image_turbo_bf16.safetensors", "default")
    t_unet = time.time() - t0
    print(f"  -> DiT loaded in {t_unet:.2f}s")

    # 2. Load Text Encoder (Qwen 3.4B)
    print("\n[2/6] Loading Text Encoder (qwen_3_4b.safetensors 7.5 GB)...")
    t0 = time.time()
    clip_loader = nodes.CLIPLoader()
    clip, = clip_loader.load_clip("qwen_3_4b.safetensors", "lumina2")
    t_clip = time.time() - t0
    print(f"  -> Text Encoder loaded in {t_clip:.2f}s")

    # 3. Load VAE
    print("\n[3/6] Loading VAE (ae.safetensors 320 MB)...")
    t0 = time.time()
    vae_loader = nodes.VAELoader()
    vae, = vae_loader.load_vae("ae.safetensors")
    t_vae = time.time() - t0
    print(f"  -> VAE loaded in {t_vae:.2f}s")

    # 4. Text Conditioning
    prompt = (
        "A hyper-detailed futuristic Martian base on red crater landscape, "
        "domes with glowing warm atmospheric lights, starry galaxy background, "
        "photorealistic, 8k resolution, cinematic masterpiece"
    )
    print(f"\n[4/6] Encoding prompt: '{prompt}'...")
    t0 = time.time()
    clip_encode = nodes.CLIPTextEncode()
    positive, = clip_encode.encode(clip, prompt)
    zero_out = nodes.ConditioningZeroOut()
    negative, = zero_out.zero_out(positive)
    t_encode = time.time() - t0
    print(f"  -> Text encoded in {t_encode:.2f}s")

    # 5. Model Sampling setup & Empty Latent
    print("\n[5/6] Setting up AuraFlow shift & 1024x1024 empty latent...")
    aura_flow = ModelSamplingAuraFlow()
    shifted_model, = aura_flow.patch(model, 3.0)

    latent = {
        "samples": torch.zeros([1, 16, 1024 // 8, 1024 // 8], device="cpu", dtype=torch.float32),
        "downscale_ratio_spacial": 8,
    }

    # 6. Denoise Sampling (8 steps)
    print("\n[6/6] Sampling 8 steps with KSampler (res_multistep / simple)...")
    ksampler = nodes.KSampler()
    t_sample_start = time.time()
    samples, = ksampler.sample(
        model=shifted_model,
        seed=42,
        steps=8,
        cfg=1.0,
        sampler_name="res_multistep",
        scheduler="simple",
        positive=positive,
        negative=negative,
        latent_image=latent,
        denoise=1.0,
    )
    t_sample = time.time() - t_sample_start
    print(f"  -> 8 DiT steps finished in {t_sample:.2f}s ({t_sample / 8:.2f}s / step)!")

    # 7. VAE Decode
    print("\n[7/7] Decoding latents with VAE to 1024x1024 RGB...")
    t0 = time.time()
    vae_decode = nodes.VAEDecode()
    decoded, = vae_decode.decode(vae, samples)
    t_decode = time.time() - t0
    print(f"  -> VAE decoded in {t_decode:.2f}s")

    # Save output image
    img_tensor = decoded[0]
    img_np = (img_tensor.detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
    img = Image.fromarray(img_np)
    out_file = out_dir / "z_image_turbo_martian_1024.png"
    img.save(out_file)
    print(f"\nSUCCESS: Output saved to {out_file}")

    total_time = t_unet + t_clip + t_vae + t_encode + t_sample + t_decode
    print(f"\nTotal Pipeline Time: {total_time:.2f}s")
    print(f"Pure Sampling Time: {t_sample:.2f}s (8 steps @ 1024x1024)")


if __name__ == "__main__":
    benchmark_z_image_turbo()

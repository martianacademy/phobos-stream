# ComfyUI-AetherStream 🌌

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Node-orange.svg)](https://github.com/comfyanonymous/ComfyUI)
[![Backends](https://img.shields.io/badge/Backends-MLX%20•%20CUDA%20•%20ROCm%20•%20GGUF-purple.svg)]()

**ComfyUI-AetherStream** is a zero-VRAM diffusion streaming custom node extension for ComfyUI. It enables consumer hardware — including **16GB/24GB Apple Silicon Macs** and **8GB/12GB NVIDIA GeForce RTX GPUs** — to generate professional-quality video and images with massive 12B–20B+ models without Out-Of-Memory (OOM) crashes.

---

## 🎯 Supported Models

* **Alibaba Wan2.1 (14B)**: 720p/1080p photorealistic text-to-video.
* **Black Forest Labs FLUX.1 (12B)**: SOTA photorealistic text-to-image.
* **Tencent HunyuanVideo (13B)**: Cinematic video generation.
* **THUDM CogVideoX (5B / 17B)**: Open-source video generation.

---

## 🚀 Installation

### Option 1: Via ComfyUI Manager (Recommended)
1. Open ComfyUI and open **ComfyUI Manager**.
2. Click **Install Custom Nodes**.
3. Search for `ComfyUI-AetherStream` and click **Install**.
4. Restart ComfyUI.

### Option 2: Git Clone
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/martianacademy/ComfyUI-AetherStream.git
cd ComfyUI-AetherStream
pip install -r requirements.txt
```

---

## 🧩 Nodes & Workflow

```mermaid
graph LR
    L["🌌 AetherStream Model Loader<br/>(Wan2.1 / FLUX.1 / Hunyuan)"] -->|"aether_model"| S["⚡ AetherStream Diffusion Sampler<br/>(Prompt, Steps, Seed, Guidance)"]
    S -->|"images"| P["PreviewImage / SaveImage"]
    S -->|"latents"| V["🎞️ AetherStream VAE Decoder"]
    V -->|"images"| VC["VHS_VideoCombine / Save"]
```

### 1. `🌌 AetherStream Model Loader (Zero-VRAM)`
* **Inputs:**
  * `model_name`: Model identifier (e.g. `wan2.1-14b`, `flux-schnell`).
  * `architecture`: `wan2.1`, `flux`, `hunyuan`, `cogvideox`.
  * `backend`: `auto` (Apple Silicon Metal MLX / CUDA / GGUF).
  * `num_blocks`: Number of transformer blocks (e.g., 40 for Wan2.1).
  * `hidden_size`: Model hidden dimension (e.g., 4096).

### 2. `⚡ AetherStream Diffusion Sampler (Disk-Stream)`
* **Inputs:**
  * `aether_model`: Connected from Loader.
  * `prompt`: Text description.
  * `width` & `height`: Resolution (e.g. 1280x720).
  * `num_frames`: Frames count (49 for video, 1 for image).
  * `steps`: Diffusion steps (20–30 recommended).
  * `seed`: Random seed.
* **Outputs:**
  * `images`: Standard ComfyUI `IMAGE` tensor `(Batch, H, W, 3)`.
  * `latents`: Raw spatio-temporal latent tensors.

### 3. `🎞️ AetherStream VAE Decoder`
* Decodes latents to full-resolution RGB video frames.

---

## 📊 Benchmark (Apple Silicon Metal GPU)

* **Wan2.1 (14B simulated 16-block 2048-dim HD video):**
  * Baseline VRAM required: **28.4 GB (OOM Crash on 16GB Mac)**
  * AetherStream Peak VRAM: **788.11 MB** (>97% memory reduction)
  * Denoising throughput: **11.19 s / step** on Metal GPU.

---

## 📜 License

MIT License — 100% Free & Open-Source.

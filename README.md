# <img src="https://raw.githubusercontent.com/martianacademy/phobos-stream/main/assets/icon.png" width="42" align="top" alt="PhobosStream Icon" /> PhobosStream

[![PyPI Version](https://img.shields.io/pypi/v/phobos-stream?color=EE6A3E&label=PyPI)](https://pypi.org/project/phobos-stream/)
[![Comfy Registry](https://img.shields.io/badge/Comfy_Registry-v0.2.3-EE6A3E)](https://registry.comfy.org/publishers/martianacademy/nodes/comfyui-phobosstream)
[![CI Tests](https://github.com/martianacademy/phobos-stream/actions/workflows/ci.yml/badge.svg)](https://github.com/martianacademy/phobos-stream/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/martianacademy/phobos-stream/blob/main/LICENSE)

**PhobosStream** is a universal zero-VRAM diffusion streaming engine built to run massive **12B–20B+ Video and Image Diffusion Transformers (DiTs)** — including **Alibaba Wan2.1 (14B)**, **Black Forest Labs FLUX.1 (12B)**, **Lumina2 / Z-Image-Turbo**, and **Tencent HunyuanVideo (13B)** — on consumer hardware without Out-Of-Memory (OOM) crashes.

By combining **3-Stage Sequential Offloading** (`Text Encoder -> DiT -> Tiled VAE`) with **Hook-Based Zero-Copy `mmap` Block-by-Block Streaming** (`TorchBlockStreamer` & `MLXBlockStreamer`), PhobosStream cuts Peak GPU Memory on a real **19.5 GB BF16 pipeline from 26.45 GB down to 1.25–1.60 GB (93.9%–95.3% VRAM reduction)** with **zero quantization quality loss**.

---

## 📊 Verified Empirical Benchmarks

### A. NVIDIA CUDA GPU — 8.0 GiB Hardware VRAM Cap & 3-Tier `HYBRID_BUDGET` (Tesla T4)

*1. **Real `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` (~14.5 GB BF16: `UMT5-XXL` + 30-block `WanDiT` + 3D `WanVAE`)** under an enforced **`8.00 GiB` CUDA VRAM limit** (`torch.cuda.set_per_process_memory_fraction`, `480×320`, `17 frames`, `12 steps` via standalone FastAPI `/v1/videos/generations`):*

| Pipeline Configuration | Enforced CUDA Cap | Peak CUDA VRAM | Result |
| :--- | :---: | :---: | :--- |
| **Standard Monolithic `WanPipeline.to("cuda")`** | `8,192.0 MB` (`8.0 GiB`) | `> 8,192 MB` (**OOM**) | ❌ Crashes (`CUDA out of memory`) |
| **`phobos-stream serve` (`HYBRID_BUDGET`: 27 GPU + 3 CPU-Pinned)** | `8,192.0 MB` (`8.0 GiB`) | **`4,373.1 MB` (`4.27 GiB`)** | ✅ Valid 17-frame H.264 `.mp4` (`260.3 s` on SM75 T4) |

*2. **3-Tier `TorchBlockStreamer` Benchmark on NVIDIA Tesla T4** (24-block 2.3 GB FP16 DiT, `4 steps`, double-buffered CUDA `record_stream` prefetch):*

| Streaming Mode | Block Residency (`GPU / CPU / NVMe`) | Peak CUDA VRAM | VRAM Saved | 4-Step Latency | Max Abs Diff vs Monolithic |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Monolithic FP16** | `24 / 0 / 0` | `4,610.0 MB` | `0.0%` | `1.024 s` | `0.0` (Reference) |
| **`cpu_offload` (Pinned + Async CUDA Stream)** | `0 / 24 / 0` | **`322.2 MB`** | **`93.0%`** | **`1.074 s`** *(only +4.9%!)* | **`0.0` (Bit-Exact)** |
| **`hybrid_budget` (3-Tier Smart Split)** | **`7 / 9 / 8`** | **`994.3 MB`** | **`78.4%`** | **`2.418 s`** *(2.34× faster than mmap)* | **`0.0` (Bit-Exact)** |
| **`mmap_stream` (Zero-RAM `meta` + NVMe)** | `0 / 0 / 24` | **`226.1 MB`** | **`95.1%`** | `5.653 s` | **`0.0` (Bit-Exact)** |

### B. Apple Silicon Metal GPU — Real 19.5 GB Model Suite (`1024×1024`, 8 Steps)

*Measured end-to-end across isolated fresh subprocesses (`scripts/benchmark_real_phobos_optimization.py`) using `qwen_3_4b.safetensors` (7.5 GB, 36 layers) + `z_image_turbo_bf16.safetensors` (11.7 GB, 34 blocks) + `ae.safetensors` (320 MB):*

| Execution Mode | Peak Metal GPU Memory | VRAM Saved | Active Resident Weights | Denoising (`Per Step`) | Total Wall Time | Visual Quality |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Monolithic Baseline** *(All 3 in GPU RAM)* | **26,445.0 MB** | `0.0%` | 19,637.5 MB | `36.80 s` (`4.60 s`) | `41.97 s` | Reference BF16 |
| **2. Phobos 3-Stage Offload (BF16)** | **12,906.4 MB** | **`51.2%`** | 11,742.5 MB | `44.14 s` (`5.52 s`) | `46.99 s` | Lossless BF16 (`32.68 dB` PSNR vs non-strip VAE) |
| **3. Phobos 3-Stage + Q4 Quantized** | **4,814.7 MB** | **`81.8%`** | 3,310.4 MB | `66.89 s` (`8.36 s`) | `69.72 s` | 4-bit Quantized (`19.25 dB` PSNR) |
| **4. Phobos Zero-VRAM Block-Stream (Full BF16)** | **1,601.0 MB** *(DiT: **1,248.2 MB**)* | **`93.9%`** *(DiT: **95.3%**)* | **23.2 MB** (`91.6 MB` active) | `71.45 s` (`8.93 s`) | `75.04 s` | **Bit-Exact BF16 (`0.0000 px` diff vs Stage-Offload)** |

### Key Architectural Mechanisms
1. **Native BF16/FP16 Zero-Copy `mmap` (`MMapTensorLoader`)**: Slices tensor byte-ranges directly from `.safetensors` on NVMe SSD into native `torch.bfloat16` (`read_torch_tensor`) and `mlx.core.bfloat16` (`read_mlx_array`) without expanding weights to FP32 in host RAM.
2. **Hook-Based Per-Block Streaming (`TorchBlockStreamer` & `MLXBlockStreamer`)**:
   - **PyTorch (`CUDA` / `MPS` / `ROCm` / `CPU`)**: Supports `cpu_offload` (read-only pinned CPU shadow + async CUDA prefetch stream, zero GPU→CPU writeback) and `mmap_stream` (zero-RAM `torch.device("meta")` skeleton support + immediate 0-element parameter eviction after each block).
   - **Apple Silicon (`MLX`)**: Enforces `mx.eval` boundaries per transformer block so MLX lazy graphs never accumulate multi-GB block weights simultaneously in unified memory.
3. **3-Stage Sequential Offloading & Tiled VAE (`StageSequentialRunner`)**: Isolates Text Encoder, DiT, and spatial-feather-blended Tiled/Strip VAE stages with explicit accelerator cache reclamation between stages, plus single-pass batched Classifier-Free Guidance (cutting disk reads per CFG step by 50%).

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph STAGE1["Stage 1: Text Encoding"]
        P["User Prompt"] --> TE["Qwen / T5-XXL / DualCLIP"]
        TE --> EMB["Conditioning Tensor (~2–5 MB)"]
        TE -.->|"Immediate Unload + Cache Flush"| FREE1["Text Encoder Freed"]
    end

    subgraph STAGE2["Stage 2: Hook-Streamed DiT Denoising"]
        EMB --> DIT["TorchBlockStreamer / MLXBlockStreamer"]
        NVME[("NVMe SSD (.safetensors mmap)")] ==>|"Stream Active Block N (~360 MB)"| BLK["Native DiT Block Forward"]
        BLK -.->|"Eval + Evict Block N to 0 Bytes"| NVME
        BLK --> ODE["Batched CFG + Flow Matching Step"]
    end

    subgraph STAGE3["Stage 3: Tiled / Strip VAE Reconstruction"]
        ODE --> VAE["Feather-Blended Tiled / Strip VAE"]
        VAE --> OUT["PNG Image / H.264 MP4 Video"]
    end
```

---

## ⚡ Quick Start

### 1. Install from PyPI

```bash
pip install phobos-stream
# With HuggingFace Diffusers (Wan2.1 / FLUX / SDXL) or GGUF extras:
pip install "phobos-stream[diffusers,gguf]"
```

Or install from source for development:
```bash
git clone https://github.com/martianacademy/phobos-stream.git
cd phobos-stream
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest -v
```

### 2. Python API: Stream Any Real PyTorch DiT in 4 Lines

```python
import torch
from phobos_stream.streaming import BlockStreamMode, TorchBlockStreamer

with torch.device("meta"):
    model = MyDiffusionTransformer().to(torch.bfloat16)

with TorchBlockStreamer(
    model,
    compute_device="auto",
    mode=BlockStreamMode.MMAP_STREAM,
    safetensors_paths=["model_bf16.safetensors"],
):
    output = model(latents, timestep, context)
```

### 3. CLI & OpenAI / Sora-Compatible Video & Image API Gateway

```bash
# Launch standalone Diffusers video/image server under a strict 3-tier VRAM/RAM budget (runs Wan2.1 in ~4.3 GiB VRAM!):
phobos-stream serve --port 8088 \
  --hf-model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --stream-mode hybrid_budget \
  --vram-budget-mb 3200

# Or index a local .safetensors checkpoint directory into a zero-copy virtual manifest:
phobos-stream shard --model-dir /path/to/models --architecture wan2.1
phobos-stream serve --port 8088 --manifest-path /path/to/models/shards/manifest.json --backend auto
```

---

## 🎨 ComfyUI Custom Node Extension (`ComfyUI-PhobosStream`)

Published on both the [**Official Comfy Registry**](https://registry.comfy.org/publishers/martianacademy/nodes/comfyui-phobosstream) and **ComfyUI-Manager** ([repository](https://github.com/martianacademy/ComfyUI-PhobosStream)):

```bash
comfy node install comfyui-phobosstream
```

* **`🚀 Phobos Zero-VRAM 2K/4K & Video Pipeline`**: 3-stage sequential offloader supporting single & `DualCLIPLoader`, dynamic 4-ch/16-ch 4D image and 5D video latents (`Wan2.1`, `HunyuanVideo`, `FLUX`, `Lumina2`, `SDXL`), `VAEDecodeTiled`, and per-block streaming.
* **`🔌 PhobosStream Block-Streamer (Any MODEL)`**: Drop-in `MODEL -> MODEL` hook patcher (`cpu_offload` / `mmap_stream`) compatible with standard ComfyUI workflows (LoRAs, ControlNet, WanVideo, FLUX).
* **`🌌 PhobosStream Model Loader`**, **`⚡ PhobosStream Diffusion Sampler`**, and **`🎞️ PhobosStream Tiled VAE Decoder`**.

---

## 📜 License

[MIT License](https://github.com/martianacademy/phobos-stream/blob/main/LICENSE) — free for commercial and personal use.

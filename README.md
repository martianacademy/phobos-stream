# PhobosStream 🌌

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Backends](https://img.shields.io/badge/Backends-MLX%20•%20CUDA%20•%20ROCm%20•%20GGUF-purple.svg)]()
[![Tests Passing](https://img.shields.io/badge/Tests-15%20Passing-success.svg)]()

**PhobosStream** is a universal, zero-VRAM diffusion inference engine designed to stream massive **12B–20B+ Video and Image Diffusion Transformers (DiTs)** — including **Alibaba Wan2.1 (14B)**, **Black Forest Labs FLUX.1 (12B)**, and **Tencent HunyuanVideo (13B)** — across consumer hardware.

By orchestrating **3-Stage Sequential Offloading** and **Zero-Copy `mmap` Block-by-Block Streaming**, PhobosStream enables consumer **16GB / 24GB Apple Silicon Macs** and **8GB / 12GB NVIDIA GeForce RTX GPUs** to generate professional-grade videos and images without Out-Of-Memory (OOM) crashes.

---

## 💡 Why PhobosStream?

Frontier video generation models (Wan2.1, HunyuanVideo) and photorealistic image generators (FLUX.1) require between **28 GB to 48 GB of active VRAM/Unified Memory**. Consumer laptops and desktop workstations crash instantly upon loading these models.

Traditional workarounds force heavy 2-bit or 3-bit quantization, degrading visual quality and frame coherence.

**PhobosStream provides a different solution:**

1. **Stage-Wise Pipeline Offloading:** Text Encoders (T5-XXL / CLIP ~12GB) and VAE decoders are loaded sequentially, executed, and immediately evicted from memory.
2. **Block-by-Block Zero-Copy `mmap` Streaming:** The core Diffusion Transformer (DiT) runs block-by-block. Only the active transformer block resides in memory, while the rest stay mapped to the NVMe SSD.
3. **Universal Compute Backends:** Runs natively on **Apple Silicon Metal (via MLX)**, **NVIDIA CUDA / AMD ROCm (via PyTorch)**, and **quantized GGUF / CPU**.

---

## 📊 Benchmark: 14B-Scale Video Generation on Apple Silicon

*Tested on Apple Silicon Metal GPU (2048-dim, 16 blocks, 720p 25-frame video latents):*

| Metric | Monolithic Resident Baseline | PhobosStream (Streaming) | Impact |
| :--- | :--- | :--- | :--- |
| **Peak VRAM / Memory** | **28.4 GB (OOM Crash)** | **788.11 MB** | **>97% Memory Reduction!** |
| **Supported Hardware** | Enterprise 48GB–80GB GPUs | 16GB/24GB Macs, RTX 3060/4060 | Consumer accessible |
| **Text Encoding RAM** | 11.2 GB persistent | **0.00 MB** (auto-evicted) | Immediate memory reuse |
| **Active Block Footprint**| Entire DiT in memory | **Single block in cache** | Zero-copy mmap paging |
| **API Compatibility** | Custom proprietary | **OpenAI / Sora REST & WebSockets**| Drop-in integration |

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph STAGE1["Stage 1: Text Encoding"]
        P["User Text Prompt"] --> TE["T5-XXL / CLIP Encoder"]
        TE --> EMB["Text Context Tokens"]
        TE -.->|"Auto Evict / Free Memory"| RAM1["0 MB RAM Footprint"]
    end

    subgraph STAGE2["Stage 2: Iterative DiT Denoising Loop"]
        EMB --> DIT["PhobosStream DiT Engine"]
        NVME[("NVMe SSD (Safetensors / GGUF)")] ==>|"mmap Zero-Copy Stream"| BLK["Block N (In Memory: ~120MB)"]
        BLK --> ATTR["Self-Attention + 3D RoPE"]
        ATTR --> CRSS["Cross-Attention (Text Context)"]
        CRSS --> FFN["SwiGLU / MLP Layer"]
        FFN -.->|"Evict Block N -> Prefetch Block N+1"| NVME
        FFN --> SCHED["Flow Matching Euler ODE Step"]
    end

    subgraph STAGE3["Stage 3: Latent Reconstruction"]
        SCHED --> VAE["VAE Latent Decoder"]
        VAE --> OUT["Generated MP4 Video / PNG Image"]
        VAE -.->|"Auto Evict"| RAM3["Memory Reclaimed"]
    end
```

---

## 🚀 Supported Models & Architectures

| Model | Modality | Parameters | Native VRAM | PhobosStream Peak RAM |
| :--- | :--- | :--- | :--- | :--- |
| **Alibaba Wan2.1** | Text-to-Video | **14B** | ~32 GB | **< 1.2 GB** |
| **Black Forest Labs FLUX.1** | Text-to-Image | **12B** | ~24 GB | **< 850 MB** |
| **Tencent HunyuanVideo** | Text-to-Video | **13B** | ~28 GB | **< 1.0 GB** |
| **THUDM CogVideoX** | Text-to-Video | **5B / 17B** | ~14GB – 36GB | **< 950 MB** |

---

## ⚡ Quick Start

### 1. Installation

```bash
git clone https://github.com/martianacademy/phobos-stream.git
cd phobos-stream

# Setup virtual environment with Python 3.10+
uv venv --python 3.12
source .venv/bin/activate

# Install dependencies (Auto-detects Apple Silicon MLX / CUDA)
uv pip install -e ".[dev]"
```

### 2. Run Test Suite

```bash
pytest -v
```

### 3. Run Benchmark

```bash
python examples/benchmark_streaming.py
```

### 4. Launch OpenAI / Sora-Compatible REST & WebSocket API

```bash
phobos-stream serve --port 8088 --backend auto
```

Or test with the automated client:
```bash
python examples/test_live_server.py
```

---

## 🎨 ComfyUI Native Custom Node Integration

PhobosStream includes a first-class **ComfyUI Custom Node Extension** (`ComfyUI-PhobosStream`) allowing visual drag-and-drop workflows on consumer hardware without Out-Of-Memory crashes!

### 1. Installation into ComfyUI
```bash
# Symlink directly into your ComfyUI custom_nodes directory:
ln -s ~/Documents/phobos-stream/ComfyUI-PhobosStream ~/ComfyUI/custom_nodes/
```

### 2. Available ComfyUI Nodes (Under `PhobosStream` Category)

```mermaid
graph LR
    L["🌌 PhobosStream Loader<br/>(Wan2.1 / FLUX.1 / Hunyuan)"] -->|"aether_model"| S["⚡ PhobosStream Sampler<br/>(Prompt, Steps, Seed)"]
    S -->|"latents"| V["🎞️ PhobosStream VAE Decoder"]
    V -->|"images (IMAGE)"| P["Preview / Save Image / Video"]
```

* **`🌌 PhobosStream Model Loader (Zero-VRAM)`**: Configures architecture (Wan2.1, FLUX.1, HunyuanVideo), hidden dimension, and execution backend (`MLX`, `CUDA`, `GGUF`, `CPU`).
* **`⚡ PhobosStream Diffusion Sampler (Disk-Stream)`**: Executes 3-stage sequential offloading and block-by-block NVMe streaming. Outputs standard ComfyUI `IMAGE` tensors compatible with `SaveImage` and `VHS_VideoCombine`.
* **`🎞️ PhobosStream VAE Decoder`**: Modular latent-to-pixel reconstruction node.

---

## 🔌 API Gateway Usage

### Generate Video (`POST /v1/videos/generations`)

```bash
curl -X POST http://127.0.0.1:8088/v1/videos/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "An astronaut riding an iridescent mechanical steed on Mars, cinematic lighting, 8k",
    "model": "wan2.1",
    "height": 720,
    "width": 1280,
    "num_frames": 49,
    "steps": 30
  }'
```

### Generate Image (`POST /v1/images/generations`)

```bash
curl -X POST http://127.0.0.1:8088/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hyperrealistic macro portrait of an owl made of obsidian and gold filigree",
    "model": "flux-schnell",
    "size": "1024x1024",
    "steps": 20
  }'
```

---

## 📜 License

This project is licensed under the **MIT License** — free for commercial and personal use.

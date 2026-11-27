<div align="center">

<img src="https://raw.githubusercontent.com/martianacademy/phobos-stream/main/assets/icon.png" width="64" alt="PhobosStream" />

# PhobosStream

**Zero-RAM & Zero-VRAM Execution Engine for 22B–33B+ Video, Audio-Video & Image Diffusion Models**

[![PyPI](https://img.shields.io/pypi/v/phobos-stream?style=flat-square&color=0EA5E9&label=PyPI)](https://pypi.org/project/phobos-stream/)
[![Comfy Registry](https://img.shields.io/badge/Comfy_Registry-ComfyUI--PhobosStream-10B981?style=flat-square)](https://registry.comfy.org/publishers/martianacademy/nodes/comfyui-phobosstream)
[![Colab Demo](https://img.shields.io/badge/Colab-MiniMax--H3_Free_T4-1E293B?style=flat-square&logo=googlecolab&logoColor=38BDF8)](https://colab.research.google.com/github/martianacademy/phobos-stream/blob/main/notebooks/MiniMax_H3_Free_T4_PhobosStream.ipynb)
[![CI](https://img.shields.io/github/actions/workflow/status/martianacademy/phobos-stream/ci.yml?style=flat-square&label=Tests)](https://github.com/martianacademy/phobos-stream/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-334155?style=flat-square)](LICENSE)

</div>

---

**PhobosStream** runs **`22B–33B+` (`14 GB–68 GB+`)** Diffusion Transformers — **`FLUX.2 [dev]` (`32B`)**, **`LTX-2.5` (`22B` Audio+Video + `Gemma-4 12B`)**, **`Wan2.2-A14B` (`57.2 GB` Dual-Expert MoE)**, and **`MiniMax-H3` (`33B` Video+Audio)** — inside **`2.3–6.6 GiB` Peak GPU VRAM** and **`~1.0 GiB` Host Heap RAM** on **NVIDIA CUDA**, **Apple Silicon Metal (`MLX`)**, and **ROCm**.

<p align="center">
  <img src="https://raw.githubusercontent.com/martianacademy/phobos-stream/main/assets/architecture.png" width="100%" alt="PhobosStream Execution Architecture and Memory Benchmarks" />
</p>

| Subsystem | Key Mechanism | Verified Gain |
| :--- | :--- | :--- |
| **Zero-RAM `meta` + Pinned Ring** | Header-only `.safetensors` / `.gguf` load (`torch.device("meta")`) + 2-slot async H2D ring + `MADV_DONTNEED` eviction | **`-97%` Host Heap** · **`5.15×` faster** BF16 `mmap` |
| **Single-Pass `nvcc` GGUF** | Zero-workspace `phobos_cuda_gguf.cu` (`Q4`–`Q8`) + compressed `uint8` `PackedGGUFShadow` CPU staging | **`7.29×` faster** (`546 GB/s`, `0.0` diff) · **`-71%` DMA** |
| **Zero-DMA Residual Cache** | Multi-group `Δ₀ < τ` gate across `(x)`, `(img, txt)`, and `(vx, ax)` streams skipping compute & PCIe DMA | **`1.50×–1.94×` E2E speedup** |
| **Fused Attention & Metal** | Compiled CUDA `SageAttention 2.2` (`SM80+`) + Apple Silicon `@mx.compile` static Cross-Attention KV | **`88.5 TFLOP/s`** · **`633 MB` Peak Metal** |

---

## Benchmarks

### 1. Frontier Heavyweight Models (Enforced `24.0 GiB` Consumer GPU Ceiling)

*End-to-end ComfyUI benchmarks (`torch 2.15 + cu132`, strict `24.0 GiB` VRAM cap via `set_per_process_memory_fraction`):*

| Model (`Checkpoint Footprint`) | Stock ComfyUI (`24.0 GiB` Cap) | PhobosStream Peak VRAM | Host Heap (`VmData`) | Latency (`sec/step`) | Fidelity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`FLUX.2 [dev]` `32B`** (`35.5 GB` FP8 / `35.0 GB` `Q8_0.gguf`, `1024×1024`) | `CUDA OOM` (`24.5 GB`) + `34.9 GB` Heap | **`4,063.8 MB` (`3.97 GiB`)** | **`1,023.7 MB` (`-97.1%`)** | **`17.11 s` (`4.28 s`)** | `Q8_0` Ref (`64.5 GB` BF16 class) |
| **`FLUX.2 [dev]` `32B`** (`19.3 GB` `Q4_K_S.gguf`, `PackedGGUFShadow`) | `CUDA OOM` | **`5,935.8 MB`** *(CPU)* / **`4,063.8 MB`** *(mmap)* | **`1,069.9 MB` (`-96.9%`)** | **`6.06 s` (`1.51 s`, `2.82×`)** | **`44.04 dB` PSNR (`0.9974` cos)** |
| **`LTX-2.5` `Gemma-4 12B` Encoder** (`26.26 GB` BF16, 48 blocks) | `CUDA OOM` (`23.6 GB + 1.44 GiB`) | **`6,578.0 MB`** (`2,226 MB` post-encode) | Streamed | **`8.81 s`** | Lossless BF16 |
| **`LTX-2.5` `22B` Audio+Video** (`42.02 GB` BF16, Joint AV `NestedTensor`) | `CUDA OOM` (`23.9 GB`) + `41.1 GB` Heap | **`6,797.8 MB` (`6.64 GiB`)** | **`1,048.4 MB` (`-97.4%`)** | **`29.03 s` (`7.26 s`)** | Joint Video + Audio BF16 Ref |
| **`LTX-2.5` `22B` Audio+Video** (`22.8 GB` `Q8_0` / `13.6 GB` `Q4_K_M`) | `CUDA OOM` | **`6,796.7 MB`** / **`7,537.2 MB`** | **`1,075–1,088 MB`** | **`25.85 s`** / **`5.75 s` (`1.44 s`, `5.05×`)** | `41.61 dB` / `34.72 dB` Audio PSNR |
| **`Wan2.2-T2V-A14B` `27B` MoE** (`57.16 GB` Dual `14B` High+Low FP16) | `CUDA OOM` (`24.5 GB`) + `31.1 GB+` Heap | **`2,314.7 MB` (`2.26 GiB`)** | **`1,032.8 MB` (`-96.7%`)** | **`34.15 s` (`5.69 s`, `2.14 s` load)** | Lossless Dual-Expert Handoff |
| **`MiniMax-H3` `33B` Video+Audio VAE** (`4.85 GB` `ViT3D` + `32 kHz` Stereo, `141f`) | `CUDA OOM` on `14.5 GiB` T4 | **`466.8 MB`** *(Video)* / **`259.6 MB`** *(Audio)* | Fits `12 GiB` VM | `38.37 s` (`hybrid_budget`) | **`0.0` Bit-Exact Diff (`96.8%` saved)** |

<p align="center">
  <img src="https://raw.githubusercontent.com/martianacademy/phobos-stream/main/assets/minimax_h3_demo.gif" width="560" alt="MiniMax-H3 141-frame video + 32kHz stereo audio streamed via PhobosStream" />
  <br/>
  <sub><b>MiniMax-H3 (141 frames @ 24fps + 32 kHz Stereo Audio)</b> — 36-block <code>4.85 GB</code> ViT3D Video VAE streamed in <b>466.8 MB Peak VRAM</b> on a 12 GiB RAM T4 VM.</sub>
</p>

### 2. Native `nvcc` GGUF Kernel & Async Ring Throughput (`NVIDIA A100`)

*Isolated `[5120, 5120]` DiT weight unpack & same-card `Wan2.1-14B` (`29.05 GB`, 6 steps @ `832×480 × 17f`) comparison:*

| Workload / Format | Baseline (PyTorch / Sync `mmap`) | PhobosStream (`nvcc` + Pinned Ring / `PackedGGUFShadow`) | Speedup | Bandwidth / Fidelity |
| :--- | :---: | :---: | :---: | :---: |
| **`phobos_cuda_gguf.cu` (`Q8_0` / `Q4_K` / `Q6_K`)** | `0.503 ms` / `1.030 ms` / `0.990 ms` | **`0.147 ms` / `0.147 ms` / `0.136 ms`** | **`3.42×` / `6.99×` / `7.29×`** | **`545.6 GB/s` HBM · `0.0` Max Diff** |
| **`Wan2.1-14B` `BF16.gguf` (`29.05 GB`)** | `86.14 s` (`14.36 s/step`) | **`16.74 s` (`2.79 s/step`, `1,867.6 MB` VRAM)** | **`5.15×` faster** | **`99.99 dB` PSNR (`0.0` Diff)** |
| **`Wan2.1-14B` `Q8_0.gguf` (`15.88 GB`)** | `34.48 s` (`5.75 s/step`) | **`10.27 s`** *(mmap)* / **`9.16 s`** *(`PackedGGUFShadow`)* | **`3.76×` faster** | **`54.19 dB` PSNR (`-46.8%` DMA)** |
| **`Wan2.1-14B` `Q4_K_S.gguf` (`9.22 GB`)** | `24.78 s` (`4.13 s/step`) | **`9.69 s`** *(mmap)* / **`8.67 s`** *(`1.45 s/step`)* | **`2.86×` faster** | **`33.60 dB` PSNR (`-70.5%` DMA)** |

### 3. Apple Silicon Metal (`M4 Max` Unified Memory)

*Native `MLX` `@mx.compile` + static Cross-Attention KV precomputation:*

| Pipeline & Mode | Peak Metal Memory | Memory Saved vs Monolithic | Wall Time (`sec/step`) | Fidelity |
| :--- | :---: | :---: | :---: | :---: |
| **`Wan2.1-1.3B` 3D Video (`block_stream`, `512×320 × 17f`)** | **`633.2 MB`** *(vs `3,184.7 MB` PyTorch MPS)* | **`78.6%`** *(vs `2,965.3 MB` MLX mono)* | **`7.355 s` (`0.919 s`)** | **`62.49 dB` PSNR (`0.99998` cos)** |
| **`Wan2.1-1.3B` 3D Video (`hybrid_stream`, `1.2 GB` budget)** | **`1,431.3 MB`** (`10/30` resident) | **`51.7%`** | **`7.625 s` (`0.953 s`)** | **`62.49 dB` PSNR** |
| **`Z-Image-Turbo + Qwen-3.4B` (`19.5 GB` BF16, `1024×1024`)** | **`1,601.0 MB`** *(DiT: `1,248.2 MB`)* | **`93.9%`** *(vs `26,445.0 MB` mono)* | `75.04 s` *(or **`54.05 s`** `hybrid`)* | **`0.0000 px` BF16 Diff** |

---

## Quickstart

### 1. Install

```bash
pip install phobos-stream
# Optional HuggingFace Diffusers (T2V / I2V / API server) & GGUF extras:
pip install "phobos-stream[diffusers,gguf]"
```

### 2. Python SDK — Stream Any `nn.Module`, `.safetensors`, or `.gguf`

```python
import torch
import phobos_stream
from phobos_stream import BlockStreamMode, TorchBlockStreamer

# A. One-line streaming + Zero-DMA Residual Cache + auto SageAttention 2.2 on an existing model:
model = phobos_stream.stream(model, mode="cpu_offload", cache_thresh=0.12)
out = model(latents, timestep, context)
phobos_stream.unstream(model)

# B. Zero-RAM `meta` skeleton streaming directly from a 35 GB–64 GB `.gguf` or `.safetensors` file:
with torch.device("meta"):
    dit = MyDiffusionTransformer().to(torch.bfloat16)

with TorchBlockStreamer(dit, mode=BlockStreamMode.MMAP_STREAM, safetensors_paths=["flux2-dev-Q8_0.gguf"]):
    out = dit(latents, timestep, context)
```

### 3. ComfyUI Extension (`ComfyUI-PhobosStream`)

Published on the [**Official Comfy Registry**](https://registry.comfy.org/publishers/martianacademy/nodes/comfyui-phobosstream) and **ComfyUI-Manager** ([source](https://github.com/martianacademy/ComfyUI-PhobosStream)):

```bash
comfy node install comfyui-phobosstream
```

Ships **9 drop-in nodes** under `PhobosStream`:
- **`PhobosZeroRAMLoader`** — Loads `14 GB–68 GB+` `.safetensors` & `.gguf` (`FLUX.2`, `LTX-2.5`, `Wan2.2`, `HunyuanVideo`) onto `meta` in `~1.0 GB` Host Heap (`mmap_stream`, `hybrid_budget`, or compressed `cpu_offload`).
- **`PhobosCLIPStreamPatcher`** — `CLIP -> CLIP` streamer for `Gemma-4 12B`, `Mistral-3`, `UMT5-XXL`, `T5-XXL`, and `Qwen-VL` with `0 MB`-VRAM CPU vocabulary lookup.
- **`PhobosBlockStreamPatcher`**, **`PhobosFirstBlockCache`**, **`PhobosVAEStreamPatcher`** (`ViT3D` Video + `32 kHz` Stereo Audio), and **`PhobosStageOffloadPipeline`**.
- **Ready-to-Run Workflows (`workflows/`):** [`flux2_32b_zeroram.json`](https://github.com/martianacademy/ComfyUI-PhobosStream/blob/main/workflows/flux2_32b_zeroram.json) · [`ltx25_22b_audiovideo_zeroram.json`](https://github.com/martianacademy/ComfyUI-PhobosStream/blob/main/workflows/ltx25_22b_audiovideo_zeroram.json) · [`wan22_dual14b_zeroram.json`](https://github.com/martianacademy/ComfyUI-PhobosStream/blob/main/workflows/wan22_dual14b_zeroram.json) · [`minimax_h3_lowvram.json`](https://github.com/martianacademy/ComfyUI-PhobosStream/blob/main/workflows/minimax_h3_lowvram.json)

### 4. CLI & OpenAI / Sora-Compatible Video (`T2V` + `I2V`) Server

```bash
# Start OpenAI-compatible `/v1/videos/generations` & `/v1/images/generations` gateway:
phobos-stream serve --port 8088 --hf-model Lightricks/LTX-Video --stream-mode cpu_offload --cache-thresh 0.12

# One-shot Image-to-Video (I2V) from CLI:
phobos-stream generate --hf-model Lightricks/LTX-Video --input-image keyframe.png \
  --prompt "Cinematic tracking shot across basalt ridges at sunset" \
  --width 512 --height 320 --frames 17 --steps 12 --output out_i2v.mp4
```

---

## Supported Model Families

| Modality | Verified Architectures |
| :--- | :--- |
| **Native Audio + Video** | `LTX-2.5 (22B)` (`AVTransformer3DModel` + `Gemma-4 12B`), `MiniMax-H3 (33B / 20B)` (`ViT3D` + `BigVGAN` `32 kHz` Stereo) |
| **Text / Image-to-Video** | `Wan2.2 (27B Dual-Expert MoE & 5B)`, `Wan2.1 (14B & 1.3B)`, `HunyuanVideo (13B)`, `LTX-Video`, `CogVideoX` |
| **High-Res Image & Edit** | `FLUX.2 [dev] (32B & Klein)`, `FLUX.1`, `Qwen-Image 2.1`, `Lumina2 / Z-Image-Turbo`, `SD3.5 / SDXL` |
| **Quantized Formats** | Native `.safetensors` (`BF16`, `FP16`, `FP8_E4M3`, `FP8_E5M2`) & `.gguf` (`Q8_0`, `Q6_K`, `Q5_K`, `Q4_K`, `Q4_1`, `Q4_0`, `BF16`, `F16`) |

---

## License

Released under the [MIT License](LICENSE).

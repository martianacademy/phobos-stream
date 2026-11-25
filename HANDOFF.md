# <img src="https://raw.githubusercontent.com/martianacademy/phobos-stream/main/assets/icon.png" width="38" align="top" alt="PhobosStream Icon" /> Project Handoff & Architecture Guide

**Project:** `PhobosStream` (`v0.2.1`)  
**GitHub Repository:** [https://github.com/martianacademy/phobos-stream](https://github.com/martianacademy/phobos-stream)  
**PyPI Package:** [https://pypi.org/project/phobos-stream/](https://pypi.org/project/phobos-stream/)  
**Comfy Registry:** [https://registry.comfy.org/publishers/martianacademy/nodes/comfyui-phobosstream](https://registry.comfy.org/publishers/martianacademy/nodes/comfyui-phobosstream)  
**ComfyUI Extension Repo:** [https://github.com/martianacademy/ComfyUI-PhobosStream](https://github.com/martianacademy/ComfyUI-PhobosStream)  
**License:** MIT

---

## 🎯 Executive Summary

`PhobosStream` is a universal zero-VRAM diffusion streaming engine designed to run **12B–20B+ Video and Image Diffusion Transformers (DiTs)** on consumer Apple Silicon Macs and 8GB/12GB NVIDIA/AMD GPUs without Out-Of-Memory (OOM) crashes.

By combining **3-Stage Sequential Offloading** (`Text Encoder -> DiT -> Tiled/Strip VAE`) with **Hook-Based Per-Block Zero-Copy `mmap` Streaming** (`TorchBlockStreamer` & `MLXBlockStreamer`), PhobosStream reduces Peak Metal GPU memory on a real **19.5 GB BF16 pipeline (`Qwen 3.4B` + `Z-Image-Turbo` + `VAE`) from 26.45 GB down to 1.25–1.60 GB (93.9%–95.3% reduction)** while preserving **bit-exact BF16 output**.

---

## 🏗️ Codebase Directory Structure

```text
phobos-stream/
├── pyproject.toml                       # Build config, PyPI metadata, and CLI entrypoint
├── README.md                            # Main user & developer documentation
├── HANDOFF.md                           # Architecture & release handoff guide
├── assets/icon.png                      # Martian brand mark (#181110, #EE6A3E, #FFF7F0)
├── ComfyUI-PhobosStream/                # Self-contained ComfyUI custom node repository
├── src/phobos_stream/
│   ├── types.py                         # GenerationConfig, BackendType, PipelineMetrics
│   ├── cli.py                           # CLI (`serve`, `generate`, `shard`, `benchmark`)
│   ├── storage/
│   │   ├── layout.py                    # ModelShardsManifest & BlockMetadata
│   │   ├── mmap_loader.py               # Zero-copy safetensors mmap (NumPy, PyTorch BF16, MLX BF16)
│   │   └── sharder.py                   # Virtual indexer & physical block partitioner
│   ├── streaming/
│   │   ├── torch_block_streamer.py      # Hook-based PyTorch CPU-offload & meta-device mmap streamer
│   │   └── mlx_block_streamer.py        # Per-block MLX mmap streamer with mx.eval boundaries
│   ├── pipeline/
│   │   ├── stage_runner.py              # StageSequentialRunner & feather-blended tiled_spatial_decode
│   │   ├── scheduler.py                 # Flow Matching Euler ODE scheduler
│   │   ├── offloader.py                 # Shard streaming coordinator
│   │   └── stream_engine.py             # Batched single-pass CFG denoising engine
│   ├── pipelines/
│   │   ├── mlx_zimage.py                # End-to-end Apple Metal MLX Z-Image-Turbo pipeline
│   │   ├── mlx_zimage_blocks.py         # Fused QKV/W13 kernels & per-block mmap streaming
│   │   └── mlx_zimage_vae.py            # Low-memory MLX Strip-VAE weight mapper
│   ├── media/
│   │   └── video_writer.py              # PNG & H.264/MPEG-4 MP4 video encoder
│   ├── integrations/comfyui/
│   │   ├── comfy_runtime.py             # 3-stage offload, 4D/5D latents, DualCLIP, Tiled VAE
│   │   └── nodes.py                     # 5 registered ComfyUI custom nodes
│   ├── backends/                        # MLX, CUDA/ROCm, and GGUF/CPU compute backends
│   ├── models/                          # Standalone attention, 3D-RoPE, and DiT block primitives
│   ├── adapters/                        # Wan2.1, FLUX, HunyuanVideo, CogVideoX weight mappers
│   └── server/                          # FastAPI HTTP + WebSocket OpenAI/Sora-compatible gateway
├── scripts/
│   └── benchmark_real_phobos_optimization.py # Isolated 4-mode empirical benchmark on 19.5 GB suite
└── tests/                               # 27 passing unit & integration tests (<3s runtime)
```

---

## 📊 Verified Empirical Results (`1024×1024`, 8 Steps, Real 19.5 GB Suite)

| Mode | Peak Metal GPU | VRAM Saved | Active Resident | Wall Time | Fidelity |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `monolithic` | `26,445.0 MB` | `0.0%` | `19,637.5 MB` | `41.97 s` | Reference |
| `stage_offload` | `12,906.4 MB` | `51.2%` | `11,742.5 MB` | `46.99 s` | `32.68 dB` PSNR (Strip-VAE) |
| `q4_stage_offload` | `4,814.7 MB` | `81.8%` | `3,310.4 MB` | `69.72 s` | `19.25 dB` PSNR (4-bit) |
| `block_stream` | `1,601.0 MB` (`1,248.2 MB` DiT) | `93.9%` (`95.3%` DiT) | `23.2 MB` | `75.04 s` | Bit-exact BF16 (`0.0 px` diff vs `stage_offload`) |

---

## 🚀 Release & CI Automation

- **GitHub CI (`.github/workflows/ci.yml`):** Runs all 27 `pytest` checks on Ubuntu 24.04 on every push/PR.
- **PyPI Auto-Publish (`.github/workflows/publish-pypi.yml`):** Publishes to `pypi.org/project/phobos-stream/` via GitHub OIDC Trusted Publisher on release or manual `workflow_dispatch` (bump `version` in `pyproject.toml` first).
- **Comfy Registry Auto-Publish (`ComfyUI-PhobosStream/.github/workflows/publish.yml`):** Publishes to `registry.comfy.org` via `Comfy-Org/publish-node-action` whenever `ComfyUI-PhobosStream/pyproject.toml` version is bumped on `main`.

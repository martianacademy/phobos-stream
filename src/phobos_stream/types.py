"""Data models, enums, and schemas for PhobosStream."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BackendType(str, Enum):
    """Supported execution backends."""

    AUTO = "auto"
    MLX = "mlx"
    CUDA = "cuda"
    ROCM = "rocm"
    GGUF = "gguf"
    CPU = "cpu"


class ModelArchitecture(str, Enum):
    """Supported Diffusion Transformer architectures."""

    WAN2_1 = "wan2.1"
    FLUX = "flux"
    HUNYUAN = "hunyuan"
    COGVIDEOX = "cogvideox"
    GENERIC_DIT = "generic_dit"


class QuantPrecision(str, Enum):
    """Supported precision and quantization formats."""

    FP16 = "fp16"
    BF16 = "bf16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    Q8_0 = "q8_0"
    Q4_K_M = "q4_k_m"
    Q4_0 = "q4_0"


class PipelineStage(str, Enum):
    """Current stage in the generation pipeline."""

    IDLE = "idle"
    TEXT_ENCODING = "text_encoding"
    LATENT_INIT = "latent_init"
    DENOISING = "denoising"
    VAE_DECODING = "vae_decoding"
    COMPLETED = "completed"
    FAILED = "failed"


class GenerationConfig(BaseModel):
    """Configuration for an image or video generation request."""

    prompt: str = Field(..., description="Text prompt describing the desired video or image")
    negative_prompt: Optional[str] = Field(default="", description="Negative prompt guidance")
    height: int = Field(default=720, ge=128, le=2160, description="Height in pixels")
    width: int = Field(default=1280, ge=128, le=3840, description="Width in pixels")
    num_frames: int = Field(default=49, ge=1, le=240, description="Number of video frames (1 for image)")
    fps: int = Field(default=24, ge=8, le=60, description="Frames per second for output video")
    steps: int = Field(default=30, ge=1, le=100, description="Diffusion denoising steps")
    guidance_scale: float = Field(default=6.0, ge=1.0, le=20.0, description="CFG guidance scale")
    seed: int = Field(default=42, description="Random seed for reproducible noise")
    block_stream: bool = Field(default=True, description="Enable block-by-block disk streaming")
    precision: QuantPrecision = Field(default=QuantPrecision.BF16, description="Working precision")


class PipelineMetrics(BaseModel):
    """Performance telemetry and telemetry tracking."""

    total_time_sec: float = 0.0
    text_encode_time_sec: float = 0.0
    denoising_time_sec: float = 0.0
    vae_decode_time_sec: float = 0.0
    peak_vram_mb: float = 0.0
    disk_bytes_read_mb: float = 0.0
    io_throughput_mb_s: float = 0.0
    steps_completed: int = 0

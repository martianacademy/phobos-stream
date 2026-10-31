"""FastAPI Pydantic schemas for OpenAI-compatible Image & Video generation."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ImageGenerationRequest(BaseModel):
    """OpenAI-compatible image generation request."""

    prompt: str = Field(..., description="Text description of the desired image")
    model: str = Field(default="flux", description="Model name (flux, wan2.1, hunyuan)")
    n: int = Field(default=1, ge=1, le=4)
    size: str = Field(default="1024x1024", description="Image resolution (e.g. 1024x1024, 768x1280)")
    steps: int = Field(default=20, ge=1, le=100)
    seed: Optional[int] = Field(default=42)


class VideoGenerationRequest(BaseModel):
    """Video generation request."""

    prompt: str = Field(..., description="Text description of the desired video")
    model: str = Field(default="wan2.1", description="Model name (wan2.1, hunyuan, cogvideo)")
    height: int = Field(default=720, ge=128, le=2160)
    width: int = Field(default=1280, ge=128, le=3840)
    num_frames: int = Field(default=49, ge=1, le=240)
    fps: int = Field(default=24, ge=8, le=60)
    steps: int = Field(default=30, ge=1, le=100)
    seed: Optional[int] = Field(default=42)


class GenerationData(BaseModel):
    """Output image/video data URL or base64 representation."""

    url: Optional[str] = None
    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    """OpenAI-compatible image response."""

    created: int
    data: List[GenerationData]


class VideoGenerationResponse(BaseModel):
    """Video generation response."""

    id: str
    status: str
    created: int
    video_url: str
    total_time_sec: float
    peak_vram_mb: float
    io_throughput_mb_s: float

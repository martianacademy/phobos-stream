"""FastAPI server and schemas for image/video generation."""

from phobos_stream.server.app import create_app
from phobos_stream.server.schemas import (
    ImageGenerationRequest,
    ImageGenerationResponse,
    VideoGenerationRequest,
    VideoGenerationResponse,
)

__all__ = [
    "ImageGenerationRequest",
    "ImageGenerationResponse",
    "VideoGenerationRequest",
    "VideoGenerationResponse",
    "create_app",
]

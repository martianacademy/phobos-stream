"""FastAPI gateway exposing OpenAI/Sora compatible image & video generation APIs."""

from typing import Any, Dict, Optional
import time
import uuid
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from phobos_stream.pipeline.stream_engine import PhobosStreamEngine
from phobos_stream.server.schemas import (
    GenerationData,
    ImageGenerationRequest,
    ImageGenerationResponse,
    VideoGenerationRequest,
    VideoGenerationResponse,
)
from phobos_stream.server.ws_progress import ProgressManager
from phobos_stream.storage.layout import ModelShardsManifest
from phobos_stream.types import BackendType, GenerationConfig


def create_app(
    manifest: Optional[ModelShardsManifest] = None,
    backend_type: BackendType = BackendType.AUTO,
) -> FastAPI:
    """Factory to create configured FastAPI service."""
    app = FastAPI(
        title="PhobosStream API Gateway",
        description="Universal Zero-VRAM Video & Image Diffusion Streaming Engine",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Fallback default manifest if none provided
    if manifest is None:
        manifest = ModelShardsManifest(
            model_name="wan2.1-streamed",
            architecture="wan2.1",
            num_blocks=4,
            hidden_size=2048,
        )

    engine = PhobosStreamEngine(manifest=manifest, backend_type=backend_type)
    progress_manager = ProgressManager()

    @app.get("/health")
    def health_check() -> Dict[str, Any]:
        return {
            "status": "healthy",
            "backend": engine.backend.name,
            "architecture": manifest.architecture,
            "num_blocks": str(manifest.num_blocks),
        }

    @app.get("/v1/models")
    def list_models() -> Dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {"id": "wan2.1", "object": "model", "type": "video", "owned_by": "phobos-stream"},
                {"id": "flux-schnell", "object": "model", "type": "image", "owned_by": "phobos-stream"},
                {"id": "hunyuan-video", "object": "model", "type": "video", "owned_by": "phobos-stream"},
            ],
        }

    @app.post("/v1/images/generations", response_model=ImageGenerationResponse)
    def generate_image(req: ImageGenerationRequest) -> ImageGenerationResponse:
        try:
            dims = req.size.split("x")
            width, height = int(dims[0]), int(dims[1])
        except Exception:
            width, height = 1024, 1024

        config = GenerationConfig(
            prompt=req.prompt,
            width=width,
            height=height,
            num_frames=1,
            steps=req.steps,
            seed=req.seed or 42,
        )
        _, metrics = engine.generate(config)
        return ImageGenerationResponse(
            created=int(time.time()),
            data=[GenerationData(url="/outputs/sample_generated.png", revised_prompt=req.prompt)],
        )

    @app.post("/v1/videos/generations", response_model=VideoGenerationResponse)
    def generate_video(req: VideoGenerationRequest) -> VideoGenerationResponse:
        config = GenerationConfig(
            prompt=req.prompt,
            width=req.width,
            height=req.height,
            num_frames=req.num_frames,
            fps=req.fps,
            steps=req.steps,
            seed=req.seed or 42,
        )
        _, metrics = engine.generate(config)
        gen_id = f"video_{uuid.uuid4().hex[:10]}"
        return VideoGenerationResponse(
            id=gen_id,
            status="completed",
            created=int(time.time()),
            video_url=f"/outputs/{gen_id}.mp4",
            total_time_sec=round(metrics.total_time_sec, 2),
            peak_vram_mb=round(metrics.peak_vram_mb, 2),
            io_throughput_mb_s=round(metrics.io_throughput_mb_s, 2),
        )

    @app.websocket("/ws/progress")
    async def websocket_progress(websocket: WebSocket) -> None:
        await progress_manager.connect(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            progress_manager.disconnect(websocket)

    return app

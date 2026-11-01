"""FastAPI gateway exposing OpenAI/Sora compatible image & video generation APIs."""

import asyncio
import base64
from pathlib import Path
import struct
import time
from typing import Any, Dict, List, Optional
import uuid
import zlib
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np

from phobos_stream.adapters.factory import get_adapter
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


def _encode_png_bytes(rgb_frame: np.ndarray) -> bytes:
    """Encode a (H, W, 3) uint8 RGB image into PNG bytes."""
    arr = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
    height, width, _ = arr.shape
    scanlines = np.hstack([np.zeros((height, 1), dtype=np.uint8), arr.reshape(height, width * 3)])
    compressed = zlib.compress(scanlines.tobytes(), level=6)

    def make_chunk(chunk_type: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + make_chunk(b"IHDR", ihdr)
        + make_chunk(b"IDAT", compressed)
        + make_chunk(b"IEND", b"")
    )


def create_app(
    manifest: Optional[ModelShardsManifest] = None,
    backend_type: BackendType = BackendType.AUTO,
    output_dir: Optional[Path] = None,
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

    if manifest is None:
        manifest = ModelShardsManifest(
            model_name="wan2.1-streamed",
            architecture="wan2.1",
            num_blocks=4,
            hidden_size=2048,
        )

    out_dir = Path(output_dir) if output_dir else Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/outputs", StaticFiles(directory=str(out_dir)), name="outputs")

    engine = PhobosStreamEngine(manifest=manifest, backend_type=backend_type)
    progress_manager = ProgressManager()

    def _make_progress_cb() -> Any:
        def _cb(step: int, total: int, stage: str) -> None:
            if not progress_manager.active_connections:
                return
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(progress_manager.broadcast_progress(step, total, stage))
            except RuntimeError:
                asyncio.run(progress_manager.broadcast_progress(step, total, stage))
        return _cb

    def _select_model_architecture(model_name: str) -> None:
        if model_name and model_name.lower() != engine.manifest.architecture.lower():
            engine.manifest.architecture = model_name.lower()
            engine.adapter = get_adapter(model_name, engine.backend, hidden_size=engine.manifest.hidden_size)
            engine.offloader.adapter = engine.adapter

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
                {"id": "cogvideox", "object": "model", "type": "video", "owned_by": "phobos-stream"},
            ],
        }

    @app.post("/v1/images/generations", response_model=ImageGenerationResponse)
    def generate_image(req: ImageGenerationRequest) -> ImageGenerationResponse:
        try:
            dims = req.size.lower().split("x")
            width, height = int(dims[0]), int(dims[1])
        except Exception:
            width, height = 1024, 1024

        _select_model_architecture(req.model)
        base_seed = req.seed if req.seed is not None else 42
        results: List[GenerationData] = []

        for idx in range(req.n):
            config = GenerationConfig(
                prompt=req.prompt,
                width=width,
                height=height,
                num_frames=1,
                steps=req.steps,
                seed=base_seed + idx,
            )
            rgb_frames, _ = engine.generate(config, progress_callback=_make_progress_cb())
            png_bytes = _encode_png_bytes(rgb_frames[0])
            file_id = f"img_{uuid.uuid4().hex[:10]}.png"
            file_path = out_dir / file_id
            file_path.write_bytes(png_bytes)

            results.append(
                GenerationData(
                    url=f"/outputs/{file_id}",
                    b64_json=base64.b64encode(png_bytes).decode("ascii"),
                    revised_prompt=req.prompt,
                )
            )

        return ImageGenerationResponse(
            created=int(time.time()),
            data=results,
        )

    @app.post("/v1/videos/generations", response_model=VideoGenerationResponse)
    def generate_video(req: VideoGenerationRequest) -> VideoGenerationResponse:
        _select_model_architecture(req.model)
        config = GenerationConfig(
            prompt=req.prompt,
            width=req.width,
            height=req.height,
            num_frames=req.num_frames,
            fps=req.fps,
            steps=req.steps,
            seed=req.seed if req.seed is not None else 42,
        )
        rgb_frames, metrics = engine.generate(config, progress_callback=_make_progress_cb())
        gen_id = f"video_{uuid.uuid4().hex[:10]}"

        # Save video frame tensor (.npy) and preview keyframe (.png) to output_dir
        np.save(out_dir / f"{gen_id}.npy", rgb_frames)
        (out_dir / f"{gen_id}.png").write_bytes(_encode_png_bytes(rgb_frames[0]))

        return VideoGenerationResponse(
            id=gen_id,
            status="completed",
            created=int(time.time()),
            video_url=f"/outputs/{gen_id}.npy",
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

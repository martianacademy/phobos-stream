"""FastAPI gateway exposing OpenAI/Sora compatible image & video generation APIs."""

import asyncio
import base64
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import uuid
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from phobos_stream.adapters.factory import get_adapter
from phobos_stream.media import encode_png_bytes, save_rgb_mp4, save_rgb_png
from phobos_stream.pipeline.stream_engine import PhobosStreamEngine
from phobos_stream.pipelines.hf_diffusers import HFDiffusersStreamEngine
from phobos_stream.server.schemas import (
    GenerationData,
    ImageGenerationRequest,
    ImageGenerationResponse,
    VideoGenerationRequest,
    VideoGenerationResponse,
)
from phobos_stream.server.ws_progress import ProgressManager
from phobos_stream.storage.layout import ModelShardsManifest
from phobos_stream.streaming import BlockStreamMode
from phobos_stream.types import BackendType, GenerationConfig


def create_app(
    manifest: Optional[ModelShardsManifest] = None,
    backend_type: BackendType = BackendType.AUTO,
    output_dir: Optional[Path] = None,
    allow_synthetic: bool = False,
    hf_model_id: Optional[str] = None,
    stream_mode: BlockStreamMode = BlockStreamMode.HYBRID_BUDGET,
    vram_budget_mb: Optional[float] = None,
    ram_budget_mb: Optional[float] = None,
    cache_rel_l1_thresh: float = 0.0,
) -> FastAPI:
    """Factory to create configured FastAPI service supporting shard manifests and real HF models."""
    app = FastAPI(
        title="PhobosStream API Gateway",
        description="Universal Zero-VRAM Video & Image Diffusion Streaming Engine",
        version="0.5.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    out_dir = Path(output_dir) if output_dir else Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/outputs", StaticFiles(directory=str(out_dir)), name="outputs")

    hf_engine: Optional[HFDiffusersStreamEngine] = None
    engine: Optional[PhobosStreamEngine] = None

    if hf_model_id:
        dev = "cpu" if backend_type == BackendType.CPU else "auto"
        hf_engine = HFDiffusersStreamEngine(
            model_id_or_path=hf_model_id,
            device=dev,
            stream_mode=stream_mode,
            vram_budget_mb=vram_budget_mb,
            ram_budget_mb=ram_budget_mb,
            cache_rel_l1_thresh=cache_rel_l1_thresh,
        )
    else:
        if manifest is None:
            manifest = ModelShardsManifest(
                model_name="wan2.1-streamed",
                architecture="wan2.1",
                num_blocks=4,
                hidden_size=2048,
                allow_synthetic=allow_synthetic,
            )
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
        if engine is not None and manifest is not None:
            if model_name and model_name.lower() != engine.manifest.architecture.lower():
                engine.manifest.architecture = model_name.lower()
                engine.adapter = get_adapter(
                    model_name,
                    engine.backend,
                    hidden_size=engine.manifest.hidden_size,
                    allow_synthetic=engine.manifest.allow_synthetic,
                )
                engine.offloader.adapter = engine.adapter

    @app.get("/health")
    def health_check() -> Dict[str, Any]:
        if hf_engine is not None:
            return {
                "status": "healthy",
                "backend": hf_engine.backend_name,
                "architecture": hf_engine.model_id,
                "stream_mode": hf_engine.stream_mode.value,
                "cache_rel_l1_thresh": hf_engine.cache_rel_l1_thresh,
                "tier_summary": hf_engine.last_tier_summary,
            }
        assert engine is not None and manifest is not None
        return {
            "status": "healthy",
            "backend": engine.backend.name,
            "architecture": manifest.architecture,
            "num_blocks": str(manifest.num_blocks),
            "allow_synthetic": manifest.allow_synthetic,
        }

    @app.get("/v1/models")
    def list_models() -> Dict[str, Any]:
        models = [
            {"id": "wan2.1", "object": "model", "type": "video", "owned_by": "phobos-stream"},
            {"id": "flux-schnell", "object": "model", "type": "image", "owned_by": "phobos-stream"},
            {"id": "hunyuan-video", "object": "model", "type": "video", "owned_by": "phobos-stream"},
            {"id": "cogvideox", "object": "model", "type": "video", "owned_by": "phobos-stream"},
        ]
        if hf_engine is not None:
            models.insert(0, {"id": hf_engine.model_id, "object": "model", "type": "diffusers", "owned_by": "huggingface"})
        return {"object": "list", "data": models}

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
        active_runner: Any = hf_engine if hf_engine is not None else engine

        try:
            for idx in range(req.n):
                config = GenerationConfig(
                    prompt=req.prompt,
                    negative_prompt=req.negative_prompt or "",
                    width=width,
                    height=height,
                    num_frames=1,
                    steps=req.steps,
                    guidance_scale=req.guidance_scale,
                    seed=base_seed + idx,
                    cache_rel_l1_thresh=max(cache_rel_l1_thresh, req.cache_rel_l1_thresh),
                )
                rgb_frames, _ = active_runner.generate(config, progress_callback=_make_progress_cb())
                png_bytes = encode_png_bytes(rgb_frames[0])
                file_id = f"img_{uuid.uuid4().hex[:10]}.png"
                (out_dir / file_id).write_bytes(png_bytes)

                results.append(
                    GenerationData(
                        url=f"/outputs/{file_id}",
                        b64_json=base64.b64encode(png_bytes).decode("ascii"),
                        revised_prompt=req.prompt,
                    )
                )
        except (FileNotFoundError, KeyError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return ImageGenerationResponse(created=int(time.time()), data=results)

    @app.post("/v1/videos/generations", response_model=VideoGenerationResponse)
    def generate_video(req: VideoGenerationRequest) -> VideoGenerationResponse:
        _select_model_architecture(req.model)
        config = GenerationConfig(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt or "",
            width=req.width,
            height=req.height,
            num_frames=req.num_frames,
            fps=req.fps,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed if req.seed is not None else 42,
            cache_rel_l1_thresh=max(cache_rel_l1_thresh, req.cache_rel_l1_thresh),
        )
        active_runner: Any = hf_engine if hf_engine is not None else engine
        try:
            rgb_frames, metrics = active_runner.generate(config, progress_callback=_make_progress_cb())
        except (FileNotFoundError, KeyError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        gen_id = f"video_{uuid.uuid4().hex[:10]}"
        save_rgb_mp4(rgb_frames, out_dir / f"{gen_id}.mp4", fps=req.fps)
        save_rgb_png(rgb_frames[0], out_dir / f"{gen_id}.png")

        return VideoGenerationResponse(
            id=gen_id,
            status="completed",
            created=int(time.time()),
            video_url=f"/outputs/{gen_id}.mp4",
            total_time_sec=round(metrics.total_time_sec, 2),
            peak_vram_mb=round(metrics.peak_vram_mb, 2),
            io_throughput_mb_s=round(metrics.io_throughput_mb_s, 2),
            skipped_steps=int(metrics.skipped_steps),
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

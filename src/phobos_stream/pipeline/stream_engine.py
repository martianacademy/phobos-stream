"""Core diffusion streaming generation engine."""

from typing import Any, Callable, Optional, Tuple
import time
import numpy as np

from phobos_stream.adapters.factory import get_adapter
from phobos_stream.backends.base import BaseBackend
from phobos_stream.backends.factory import get_backend
from phobos_stream.models.rotary import RotaryEmbedding3D
from phobos_stream.pipeline.offloader import PipelineOffloader
from phobos_stream.pipeline.scheduler import FlowMatchingScheduler
from phobos_stream.storage.layout import ModelShardsManifest
from phobos_stream.types import BackendType, GenerationConfig, PipelineMetrics, PipelineStage


class PhobosStreamEngine:
    """Universal Zero-VRAM Video and Image Generation Engine."""

    def __init__(
        self,
        manifest: ModelShardsManifest,
        backend_type: BackendType = BackendType.AUTO,
    ) -> None:
        self.manifest = manifest
        self.backend = get_backend(backend_type)
        self.adapter = get_adapter(manifest.architecture, self.backend, hidden_size=manifest.hidden_size)
        self.offloader = PipelineOffloader(manifest, self.adapter, self.backend)
        self.rotary = RotaryEmbedding3D(dim=manifest.hidden_size, backend=self.backend)

    def generate(
        self,
        config: GenerationConfig,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[np.ndarray, PipelineMetrics]:
        """Execute full video or image generation pipeline."""
        metrics = PipelineMetrics()
        start_total = time.perf_counter()

        # 1. Text Encoding Stage
        if progress_callback:
            progress_callback(0, config.steps, PipelineStage.TEXT_ENCODING.value)
        prompt_emb, t_enc_time = self.offloader.encode_text_stage(config.prompt)
        metrics.text_encode_time_sec = t_enc_time

        # 2. Latent Noise Initialization
        # DiT patch tokens: spatial downsampling 8x or 16x
        t_len = max(1, (config.num_frames - 1) // 4 + 1) if config.num_frames > 1 else 1
        h_len = config.height // 16
        w_len = config.width // 16
        seq_len = t_len * h_len * w_len

        # Initial random Gaussian latent noise
        latents = self.backend.randn(
            (1, seq_len, self.manifest.hidden_size),
            seed=config.seed,
        )

        # 3D RoPE position embeddings
        cos, sin = self.rotary.compute_frequencies(t_len=t_len, h_len=h_len, w_len=w_len)

        # 3. Iterative Denoising Loop
        scheduler = FlowMatchingScheduler(num_steps=config.steps)
        timesteps = scheduler.get_timesteps()

        start_denoise = time.perf_counter()
        total_io_bytes = 0

        for step_idx, t in enumerate(timesteps):
            if progress_callback:
                progress_callback(step_idx + 1, config.steps, PipelineStage.DENOISING.value)

            model_out, io_bytes = self.offloader.stream_dit_step(
                latents=latents,
                context=prompt_emb,
                timestep=t,
                cos=cos,
                sin=sin,
            )
            total_io_bytes += io_bytes

            # ODE update step
            lat_np = self.backend.to_numpy(latents)
            out_np = self.backend.to_numpy(model_out)
            prev_np = scheduler.step(model_output=out_np, timestep=t, sample=lat_np)
            latents = self.backend.convert_from_numpy(prev_np)

            metrics.steps_completed = step_idx + 1

        metrics.denoising_time_sec = time.perf_counter() - start_denoise

        # 4. VAE Latent Decode Stage
        if progress_callback:
            progress_callback(config.steps, config.steps, PipelineStage.VAE_DECODING.value)
        rgb_frames, vae_time = self.offloader.decode_vae_stage(
            latents=latents,
            height=config.height,
            width=config.width,
            num_frames=config.num_frames,
        )
        metrics.vae_decode_time_sec = vae_time

        # Final Telemetry
        metrics.total_time_sec = time.perf_counter() - start_total
        metrics.disk_bytes_read_mb = total_io_bytes / (1024.0 * 1024.0)
        metrics.peak_vram_mb = self.backend.get_memory_allocated_mb()
        if metrics.denoising_time_sec > 0:
            metrics.io_throughput_mb_s = metrics.disk_bytes_read_mb / metrics.denoising_time_sec

        if progress_callback:
            progress_callback(config.steps, config.steps, PipelineStage.COMPLETED.value)

        return rgb_frames, metrics

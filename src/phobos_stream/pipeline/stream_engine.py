"""Legacy v0.1 synthetic shard benchmark engine (`allow_synthetic=True` isolated).

For real HuggingFace Diffusers T2V/I2V/T2I pipelines, use `HFDiffusersStreamEngine`
(`phobos_stream.pipelines.hf_diffusers`).
"""

from typing import Callable, Optional, Tuple
import time
import numpy as np

from phobos_stream.adapters.factory import get_adapter
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
        self.adapter = get_adapter(
            manifest.architecture,
            self.backend,
            hidden_size=manifest.hidden_size,
            allow_synthetic=manifest.allow_synthetic,
        )
        self.offloader = PipelineOffloader(manifest, self.adapter, self.backend)
        self.rotary = RotaryEmbedding3D(dim=manifest.hidden_size, backend=self.backend)

    def generate(
        self,
        config: GenerationConfig,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[np.ndarray, PipelineMetrics]:
        """Execute full video or image generation pipeline returning `(T, H, W, 3)` RGB frames."""
        rgb_frames, _, metrics = self.generate_with_latents(config, progress_callback=progress_callback)
        return rgb_frames, metrics

    def generate_with_latents(
        self,
        config: GenerationConfig,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, PipelineMetrics]:
        """Execute full pipeline and return decoded `(T, H, W, 3)` RGB frames and raw latents."""
        metrics = PipelineMetrics()
        start_total = time.perf_counter()

        # 1. Text Encoding Stage
        if progress_callback:
            progress_callback(0, config.steps, PipelineStage.TEXT_ENCODING.value)
        prompt_emb, t_enc_time = self.offloader.encode_text_stage(config.prompt)
        neg_emb = None
        if config.negative_prompt and config.guidance_scale > 1.0:
            neg_emb, neg_time = self.offloader.encode_text_stage(config.negative_prompt)
            t_enc_time += neg_time
        metrics.text_encode_time_sec = t_enc_time

        # 2. Latent Noise Initialization
        if progress_callback:
            progress_callback(0, config.steps, PipelineStage.LATENT_INIT.value)
        t_len = max(1, (config.num_frames - 1) // 4 + 1) if config.num_frames > 1 else 1
        h_len = max(1, config.height // 16)
        w_len = max(1, config.width // 16)
        seq_len = t_len * h_len * w_len

        latents = self.backend.randn(
            (1, seq_len, self.manifest.hidden_size),
            seed=config.seed,
        )
        cos, sin = self.rotary.compute_frequencies(t_len=t_len, h_len=h_len, w_len=w_len)

        # 3. Iterative Denoising Loop (Single-pass batched CFG halves shard disk I/O)
        scheduler = FlowMatchingScheduler(num_steps=config.steps)
        timesteps = scheduler.get_timesteps()

        start_denoise = time.perf_counter()
        total_io_bytes = 0
        use_cfg = neg_emb is not None and config.guidance_scale > 1.0
        batched_ctx = self.backend.concat([prompt_emb, neg_emb], axis=0) if use_cfg else prompt_emb

        with self.backend.wired_limit():
            for step_idx, t in enumerate(timesteps):
                if progress_callback:
                    progress_callback(step_idx + 1, config.steps, PipelineStage.DENOISING.value)

                if use_cfg:
                    batched_latents = self.backend.concat([latents, latents], axis=0)
                    batched_out, io_bytes = self.offloader.stream_dit_step(
                        latents=batched_latents,
                        context=batched_ctx,
                        timestep=t,
                        cos=cos,
                        sin=sin,
                        block_stream=config.block_stream,
                    )
                    total_io_bytes += io_bytes
                    cond_out, uncond_out = self.backend.split_two(batched_out, axis=0)
                    guided_out = uncond_out + float(config.guidance_scale) * (cond_out - uncond_out)
                else:
                    guided_out, io_bytes = self.offloader.stream_dit_step(
                        latents=latents,
                        context=batched_ctx,
                        timestep=t,
                        cos=cos,
                        sin=sin,
                        block_stream=config.block_stream,
                    )
                    total_io_bytes += io_bytes

                latents = scheduler.step(model_output=guided_out, timestep=t, sample=latents)
                self.backend.async_eval(latents)
                metrics.steps_completed = step_idx + 1

        self.backend.synchronize()
        metrics.denoising_time_sec = time.perf_counter() - start_denoise
        final_latents_np = self.backend.to_numpy(latents)

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

        return rgb_frames, final_latents_np, metrics

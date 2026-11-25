"""Real 3-stage sequential pipeline runner and feather-blended spatial VAE tiler."""

import gc
import platform
import resource
import time
from typing import Any, Callable, Optional, Tuple
import numpy as np

from phobos_stream.types import PipelineMetrics, PipelineStage


def flush_accelerator_memory() -> None:
    """Reclaim Python, PyTorch (CUDA/MPS), and Apple MLX memory caches between stages."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass

    try:
        import mlx.core as mx

        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
    except ImportError:
        pass


def _process_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return float(usage) / (1024.0 * 1024.0)
    return float(usage) / 1024.0


def _build_1d_feather(length: int, left_overlap: int, right_overlap: int) -> np.ndarray:
    """Construct a 1D linear feather ramp equal to 1.0 in interior and tapering at overlaps."""
    w = np.ones(length, dtype=np.float32)
    if left_overlap > 0:
        ramp_len = min(left_overlap, length)
        w[:ramp_len] = np.linspace(0.0, 1.0, ramp_len + 2, dtype=np.float32)[1:-1]
    if right_overlap > 0:
        ramp_len = min(right_overlap, length)
        w[-ramp_len:] = np.minimum(
            w[-ramp_len:],
            np.linspace(1.0, 0.0, ramp_len + 2, dtype=np.float32)[1:-1],
        )
    return w


def tiled_spatial_decode(
    decode_tile_fn: Callable[[Any], np.ndarray],
    latents: Any,
    tile_latent_h: int = 64,
    tile_latent_w: int = 64,
    overlap_latent: int = 8,
    spatial_upscale: int = 8,
) -> np.ndarray:
    """Decode 4D `(B, C, H, W)` or 5D `(B, C, T, H, W)` latents in overlapping spatial tiles.

    Prevents Out-Of-Memory spikes during 2K/4K image or HD video VAE decoding by decoding
    bounded spatial patches and blending overlaps with smooth 2D linear feather weights.

    Args:
        decode_tile_fn: Callable mapping a latent tile slice to `(F, h_px, w_px, 3)` float/uint8 RGB.
        latents: 4D `(B, C, H_lat, W_lat)` or 5D `(B, C, T_lat, H_lat, W_lat)` array/tensor.
        tile_latent_h: Tile height in latent space.
        tile_latent_w: Tile width in latent space.
        overlap_latent: Overlap margin in latent pixels between adjacent tiles.
        spatial_upscale: VAE spatial upsampling factor (typically 8 or 16).

    Returns:
        Blended `(F, H_px, W_px, 3)` float32 RGB array.
    """
    lat_shape = tuple(int(s) for s in latents.shape)
    if len(lat_shape) not in (4, 5):
        raise ValueError(f"Expected 4D or 5D latents for tiled_spatial_decode, got shape {lat_shape}")

    h_lat, w_lat = lat_shape[-2], lat_shape[-1]
    if h_lat <= tile_latent_h and w_lat <= tile_latent_w:
        return np.asarray(decode_tile_fn(latents), dtype=np.float32)

    stride_h = max(1, tile_latent_h - overlap_latent)
    stride_w = max(1, tile_latent_w - overlap_latent)
    out_h = h_lat * spatial_upscale
    out_w = w_lat * spatial_upscale

    accum: Optional[np.ndarray] = None
    weight_sum: Optional[np.ndarray] = None

    y_starts = list(range(0, max(1, h_lat - tile_latent_h + 1), stride_h))
    if not y_starts or y_starts[-1] + tile_latent_h < h_lat:
        y_starts.append(max(0, h_lat - tile_latent_h))

    x_starts = list(range(0, max(1, w_lat - tile_latent_w + 1), stride_w))
    if not x_starts or x_starts[-1] + tile_latent_w < w_lat:
        x_starts.append(max(0, w_lat - tile_latent_w))

    for y0 in y_starts:
        y1 = min(h_lat, y0 + tile_latent_h)
        for x0 in x_starts:
            x1 = min(w_lat, x0 + tile_latent_w)
            tile_slice = latents[..., y0:y1, x0:x1]
            decoded_tile = np.asarray(decode_tile_fn(tile_slice), dtype=np.float32)
            if decoded_tile.ndim == 3:
                decoded_tile = decoded_tile[None, ...]

            num_frames, th_px, tw_px, channels = decoded_tile.shape
            if accum is None:
                accum = np.zeros((num_frames, out_h, out_w, channels), dtype=np.float32)
                weight_sum = np.zeros((1, out_h, out_w, 1), dtype=np.float32)

            py0, px0 = y0 * spatial_upscale, x0 * spatial_upscale
            py1, px1 = py0 + th_px, px0 + tw_px

            ov_top = overlap_latent * spatial_upscale if y0 > 0 else 0
            ov_bot = overlap_latent * spatial_upscale if y1 < h_lat else 0
            ov_left = overlap_latent * spatial_upscale if x0 > 0 else 0
            ov_right = overlap_latent * spatial_upscale if x1 < w_lat else 0

            wy = _build_1d_feather(th_px, ov_top, ov_bot)[:, None]
            wx = _build_1d_feather(tw_px, ov_left, ov_right)[None, :]
            mask_2d = (wy * wx)[None, :, :, None]

            accum[:, py0:py1, px0:px1, :] += decoded_tile * mask_2d
            assert weight_sum is not None
            weight_sum[:, py0:py1, px0:px1, :] += mask_2d

    assert accum is not None and weight_sum is not None
    return accum / np.maximum(weight_sum, 1e-6)


class StageSequentialRunner:
    """Executes strict 3-stage sequential offloading (`Encode -> Denoise -> Decode`)."""

    def run(
        self,
        encode_fn: Callable[[], Any],
        denoise_fn: Callable[[Any], Tuple[Any, int, int]],
        decode_fn: Callable[[Any], np.ndarray],
        total_steps: int = 1,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[np.ndarray, PipelineMetrics]:
        """Run isolated stages, flushing model memory completely between every transition.

        Args:
            encode_fn: Loads text encoder, encodes prompt(s), drops references, returns conditioning.
            denoise_fn: Receives conditioning, runs denoising, returns `(latents, steps_done, bytes_streamed)`.
            decode_fn: Receives latents, loads VAE (optionally tiled), returns RGB array.
            total_steps: Total diffusion steps for progress reporting.
            progress_callback: Optional `(step, total, stage_name)` listener.
        """
        metrics = PipelineMetrics()
        t_total = time.perf_counter()
        flush_accelerator_memory()

        # Stage 1: Text Encoding
        if progress_callback:
            progress_callback(0, total_steps, PipelineStage.TEXT_ENCODING.value)
        t0 = time.perf_counter()
        conditioning = encode_fn()
        flush_accelerator_memory()
        metrics.text_encode_time_sec = time.perf_counter() - t0

        # Stage 2: Denoising
        if progress_callback:
            progress_callback(0, total_steps, PipelineStage.DENOISING.value)
        t0 = time.perf_counter()
        latents, steps_done, bytes_streamed = denoise_fn(conditioning)
        del conditioning
        flush_accelerator_memory()
        metrics.denoising_time_sec = time.perf_counter() - t0
        metrics.steps_completed = steps_done
        metrics.disk_bytes_read_mb = float(bytes_streamed) / (1024.0 * 1024.0)
        if metrics.denoising_time_sec > 0 and metrics.disk_bytes_read_mb > 0:
            metrics.io_throughput_mb_s = metrics.disk_bytes_read_mb / metrics.denoising_time_sec

        # Stage 3: VAE Decode
        if progress_callback:
            progress_callback(steps_done, total_steps, PipelineStage.VAE_DECODING.value)
        t0 = time.perf_counter()
        frames = decode_fn(latents)
        del latents
        flush_accelerator_memory()
        metrics.vae_decode_time_sec = time.perf_counter() - t0

        metrics.total_time_sec = time.perf_counter() - t_total
        metrics.peak_vram_mb = _process_rss_mb()
        if progress_callback:
            progress_callback(steps_done, total_steps, PipelineStage.COMPLETED.value)
        return frames, metrics

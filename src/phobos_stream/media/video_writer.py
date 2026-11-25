"""Image (PNG) and Video (MP4) encoding utilities for PhobosStream."""

from pathlib import Path
import shutil
import struct
import subprocess
import zlib
import numpy as np


def encode_png_bytes(rgb_frame: np.ndarray) -> bytes:
    """Encode a `(H, W, 3)` uint8 RGB image array into standard PNG bytes.

    Args:
        rgb_frame: 3D `(H, W, 3)` uint8 RGB array.

    Returns:
        Valid PNG binary payload.
    """
    arr = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) RGB array for PNG encoding, got shape {arr.shape}")

    height, width, _ = arr.shape
    scanlines = np.hstack([np.zeros((height, 1), dtype=np.uint8), arr.reshape(height, width * 3)])
    compressed = zlib.compress(scanlines.tobytes(), level=6)

    def _make_chunk(chunk_type: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _make_chunk(b"IHDR", ihdr)
        + _make_chunk(b"IDAT", compressed)
        + _make_chunk(b"IEND", b"")
    )


def save_rgb_png(rgb_frame: np.ndarray, path: Path) -> Path:
    """Write a `(H, W, 3)` uint8 RGB image to disk as a PNG file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(encode_png_bytes(rgb_frame))
    return out_path


def save_rgb_mp4(rgb_frames: np.ndarray, path: Path, fps: int = 24) -> Path:
    """Encode `(T, H, W, 3)` uint8 RGB frames into a standards-compliant MP4 video.

    Uses `ffmpeg` (`libx264`, `yuv420p`, `+faststart`) when installed on PATH,
    with automatic fallback to OpenCV (`cv2.VideoWriter` with `mp4v`).

    Args:
        rgb_frames: `(T, H, W, 3)` or `(H, W, 3)` RGB array (`uint8` or float `[0, 1]`).
        path: Target `.mp4` output path.
        fps: Playback frame rate in frames per second.

    Returns:
        Resolved output `.mp4` path.
    """
    frames = np.asarray(rgb_frames)
    if frames.ndim == 3:
        frames = frames[None, ...]
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected (T, H, W, 3) RGB frames for MP4 encoding, got shape {frames.shape}")

    if frames.dtype != np.uint8:
        if np.issubdtype(frames.dtype, np.floating) and float(frames.max()) <= 1.0:
            frames = np.clip(frames * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            frames = np.clip(frames, 0, 255).astype(np.uint8)
    frames = np.ascontiguousarray(frames)

    out_path = Path(path)
    if out_path.suffix.lower() != ".mp4":
        out_path = out_path.with_suffix(".mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _, height, width, _ = frames.shape
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(max(1, int(fps))),
            "-i",
            "-",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        proc = subprocess.run(cmd, input=frames.tobytes(), capture_output=True, check=False)
        if proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
            return out_path

    # Fallback to OpenCV VideoWriter if ffmpeg is absent or lacks libx264
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Neither a working 'ffmpeg' binary with libx264 nor 'opencv-python' is available to encode MP4."
        ) from exc

    even_w = width + (width % 2)
    even_h = height + (height % 2)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(max(1, int(fps))), (even_w, even_h))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV VideoWriter failed to open output path: {out_path}")

    try:
        for frame in frames:
            if even_h != height or even_w != width:
                frame = np.pad(frame, ((0, even_h - height), (0, even_w - width), (0, 0)), mode="edge")
            bgr = frame[:, :, ::-1].copy()
            writer.write(bgr)
    finally:
        writer.release()

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to write non-empty MP4 video to {out_path}")
    return out_path

"""Zero-copy memory-mapped tensor loader for Diffusion Transformer weights."""

import json
import mmap
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


DTYPE_MAP: Dict[str, Any] = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.uint16,
    "F8_E4M3": np.uint8,
    "F8_E5M2": np.uint8,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U64": np.uint64,
    "U32": np.uint32,
    "U16": np.uint16,
    "U8": np.uint8,
    "BOOL": np.bool_,
}


def _decode_fp8_e4m3(raw_u8: np.ndarray) -> np.ndarray:
    """Vectorized conversion of 8-bit float (E4M3FN) bit patterns to float32."""
    u = raw_u8.astype(np.uint32)
    sign = (u >> 7) & 0x1
    exp = (u >> 3) & 0xF
    mant = u & 0x7

    # Normal numbers: (-1)^sign * 2^(exp - 7) * (1 + mant / 8)
    # Subnormal numbers (exp == 0): (-1)^sign * 2^(-6) * (mant / 8)
    val = np.where(
        exp == 0,
        (mant.astype(np.float32) / 8.0) * (2.0 ** -6),
        (1.0 + mant.astype(np.float32) / 8.0) * np.exp2(exp.astype(np.float32) - 7.0),
    )
    return np.where(sign == 1, -val, val).astype(np.float32)


def _decode_fp8_e5m2(raw_u8: np.ndarray) -> np.ndarray:
    """Vectorized conversion of 8-bit float (E5M2) bit patterns to float32."""
    u = raw_u8.astype(np.uint16)
    fp16_bits = (u << 8).astype(np.uint16)
    return fp16_bits.view(np.float16).astype(np.float32)


class MMapTensorLoader:
    """Zero-copy mmap tensor reader supporting NumPy, PyTorch, and Apple MLX."""

    def __init__(self, file_path: Path) -> None:
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Weight file not found: {self.file_path}")

        self.file_obj = open(self.file_path, "rb")
        self.fileno = self.file_obj.fileno()
        self.mmap_obj = mmap.mmap(self.fileno, 0, access=mmap.ACCESS_READ)
        self.header, self.data_offset = self._read_safetensors_header()

    def _read_safetensors_header(self) -> Tuple[Dict[str, Any], int]:
        """Parse safetensors 8-byte little-endian header and JSON metadata."""
        self.mmap_obj.seek(0)
        header_size_bytes = self.mmap_obj.read(8)
        if len(header_size_bytes) < 8:
            return {}, 0
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        header_json = self.mmap_obj.read(header_size).decode("utf-8")
        header = json.loads(header_json)
        data_offset = 8 + header_size
        return header, data_offset

    def get_tensor_names(self) -> List[str]:
        """List all tensor names present in this safetensors file."""
        return [k for k in self.header.keys() if k != "__metadata__"]

    def get_tensor_byte_size(self, name: str) -> int:
        """Return the raw byte size of a tensor from header metadata."""
        if name not in self.header:
            raise KeyError(f"Tensor '{name}' not found in {self.file_path.name}")
        start_rel, end_rel = self.header[name]["data_offsets"]
        return int(end_rel - start_rel)

    def get_tensor_metadata(self, name: str) -> Dict[str, Any]:
        """Return header metadata dictionary for `name`."""
        if name not in self.header:
            raise KeyError(f"Tensor '{name}' not found in {self.file_path.name}")
        return self.header[name]

    def evict_page_cache(self, name: str) -> None:
        """Advise OS kernel (`MADV_DONTNEED`) to drop clean `mmap` pages for `name`."""
        from phobos_stream.streaming.budget_planner import advise_dontneed_mmap_range

        meta = self.get_tensor_metadata(name)
        start_rel, end_rel = meta["data_offsets"]
        advise_dontneed_mmap_range(self.mmap_obj, self.data_offset + int(start_rel), self.data_offset + int(end_rel))

    def read_raw_slice(self, name: str) -> Tuple[memoryview, str, Tuple[int, ...]]:
        """Return zero-copy `(memoryview, safetensors_dtype, shape)` for `name`."""
        if name not in self.header:
            raise KeyError(f"Tensor '{name}' not found in {self.file_path.name}")

        meta = self.header[name]
        dtype_str = str(meta["dtype"])
        if dtype_str not in DTYPE_MAP:
            raise ValueError(f"Unsupported safetensors dtype '{dtype_str}' in {self.file_path.name}")

        shape = tuple(int(dim) for dim in meta["shape"])
        start_rel, end_rel = meta["data_offsets"]
        abs_start = self.data_offset + int(start_rel)
        abs_end = self.data_offset + int(end_rel)
        return memoryview(self.mmap_obj)[abs_start:abs_end], dtype_str, shape

    def read_tensor(self, name: str, copy: bool = False) -> np.ndarray:
        """Slice tensor from mmap buffer as a NumPy array (decoding BF16/FP8 to float32)."""
        tensor_bytes, dtype_str, shape = self.read_raw_slice(name)
        dtype = DTYPE_MAP[dtype_str]
        arr = np.frombuffer(tensor_bytes, dtype=dtype)

        if dtype_str == "BF16":
            arr = (arr.astype(np.uint32) << 16).view(np.float32)
        elif dtype_str == "F8_E4M3":
            arr = _decode_fp8_e4m3(arr)
        elif dtype_str == "F8_E5M2":
            arr = _decode_fp8_e5m2(arr)
        elif copy:
            arr = arr.copy()

        return arr.reshape(shape)

    def read_torch_tensor(
        self,
        name: str,
        device: str = "cpu",
        target_dtype: Optional[Any] = None,
    ) -> Any:
        """Materialize a PyTorch tensor in native precision without BF16->FP32 expansion."""
        import torch

        tensor_bytes, dtype_str, shape = self.read_raw_slice(name)
        if dtype_str == "BF16":
            raw_i16 = np.frombuffer(tensor_bytes, dtype=np.int16).copy().reshape(shape)
            tensor = torch.from_numpy(raw_i16).view(torch.bfloat16)
        elif dtype_str == "F16":
            raw_f16 = np.frombuffer(tensor_bytes, dtype=np.float16).copy().reshape(shape)
            tensor = torch.from_numpy(raw_f16)
        elif dtype_str == "F32":
            raw_f32 = np.frombuffer(tensor_bytes, dtype=np.float32).copy().reshape(shape)
            tensor = torch.from_numpy(raw_f32)
        elif dtype_str == "F8_E4M3" and hasattr(torch, "float8_e4m3fn"):
            raw_u8 = np.frombuffer(tensor_bytes, dtype=np.uint8).copy().reshape(shape)
            tensor = torch.from_numpy(raw_u8).view(torch.float8_e4m3fn)
        elif dtype_str == "F8_E5M2" and hasattr(torch, "float8_e5m2"):
            raw_u8 = np.frombuffer(tensor_bytes, dtype=np.uint8).copy().reshape(shape)
            tensor = torch.from_numpy(raw_u8).view(torch.float8_e5m2)
        elif dtype_str in ("F8_E4M3", "F8_E5M2"):
            decoded = self.read_tensor(name, copy=False)
            tensor = torch.from_numpy(decoded)
        else:
            tensor = torch.from_numpy(self.read_tensor(name, copy=True))

        if target_dtype is not None and tensor.dtype != target_dtype:
            tensor = tensor.to(dtype=target_dtype)

        target_dev = torch.device(device)
        if tensor.device != target_dev:
            non_blocking = target_dev.type == "cuda"
            tensor = tensor.to(device=target_dev, non_blocking=non_blocking)
        return tensor

    def read_mlx_array(self, name: str, target_dtype: Optional[Any] = None) -> Any:
        """Materialize an Apple MLX array preserving native BF16/FP16 unified memory layout."""
        import mlx.core as mx

        tensor_bytes, dtype_str, shape = self.read_raw_slice(name)
        if dtype_str == "BF16":
            raw_u16 = np.frombuffer(tensor_bytes, dtype=np.uint16).reshape(shape)
            arr = mx.array(raw_u16).view(mx.bfloat16)
        elif dtype_str == "F16":
            arr = mx.array(np.frombuffer(tensor_bytes, dtype=np.float16).reshape(shape))
        elif dtype_str == "F32":
            arr = mx.array(np.frombuffer(tensor_bytes, dtype=np.float32).reshape(shape))
        else:
            arr = mx.array(self.read_tensor(name, copy=False))

        if target_dtype is not None and arr.dtype != target_dtype:
            arr = arr.astype(target_dtype)
        return arr

    def read_all_tensors(self, copy: bool = False) -> Dict[str, np.ndarray]:
        """Read all tensors in this file into a dictionary."""
        return {name: self.read_tensor(name, copy=copy) for name in self.get_tensor_names()}

    def close(self) -> None:
        """Close mmap and file descriptors safely even if zero-copy views exist."""
        if not self.mmap_obj.closed:
            try:
                self.mmap_obj.close()
            except BufferError:
                pass
        if not self.file_obj.closed:
            self.file_obj.close()

    def __enter__(self) -> "MMapTensorLoader":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

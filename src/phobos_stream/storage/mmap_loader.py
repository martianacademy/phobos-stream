"""Zero-copy memory-mapped tensor loader for Diffusion Transformer weights."""

import json
import mmap
import os
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


DTYPE_MAP = {
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.uint16,  # Treated as uint16 for raw buffer, converted as needed
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}


class MMapTensorLoader:
    """Zero-copy mmap tensor reader for fast NVMe streaming."""

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

    def read_tensor(self, name: str) -> np.ndarray:
        """Slice tensor directly from mmap buffer without full file read."""
        if name not in self.header:
            raise KeyError(f"Tensor '{name}' not found in {self.file_path.name}")

        meta = self.header[name]
        dtype = DTYPE_MAP.get(meta["dtype"], np.float32)
        shape = tuple(meta["shape"])
        start_rel, end_rel = meta["data_offsets"]

        abs_start = self.data_offset + start_rel
        abs_end = self.data_offset + end_rel

        # Sliced view copied so mmap handles can be cleanly closed
        tensor_bytes = memoryview(self.mmap_obj)[abs_start:abs_end]
        arr = np.frombuffer(tensor_bytes, dtype=dtype).copy()
        if dtype == np.uint16 and meta["dtype"] == "BF16":
            # Convert bfloat16 bit pattern to float32
            arr = (arr.astype(np.uint32) << 16).view(np.float32)
        return arr.reshape(shape)

    def read_all_tensors(self) -> Dict[str, np.ndarray]:
        """Read all tensors in this file into a dictionary."""
        return {name: self.read_tensor(name) for name in self.get_tensor_names()}

    def close(self) -> None:
        """Close mmap and file descriptors."""
        if not self.mmap_obj.closed:
            self.mmap_obj.close()
        if not self.file_obj.closed:
            self.file_obj.close()

    def __enter__(self) -> "MMapTensorLoader":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

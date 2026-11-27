"""Zero-copy `.gguf` v2/v3 `mmap` reader with On-GPU quantized block unpacking (`Q8_0`, `Q4_0`, `Q4_1`, `Q4_K`).

Transfers compressed `uint8` GGUF blocks across PCIe (`1.88x` smaller for `Q8_0`, `3.56x` smaller
for `Q4_0` / `Q4_K`) and unpacks them directly on the target GPU (`cuda` / `mps`) in vectorized
PyTorch tensor ops into `bfloat16` / `float16`, slashing PCIe & SSD bandwidth without custom C++ builds.
"""

from __future__ import annotations

import mmap
from pathlib import Path
import struct
from typing import Any, BinaryIO, Dict, List, Optional, Tuple


# GGML tensor type IDs -> (name, block_elements, block_bytes)
GGML_TYPES: Dict[int, Tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    8: ("Q8_0", 32, 34),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    30: ("BF16", 1, 2),
}

# GGUF metadata value type readers
_SCALAR_FMT: Dict[int, Tuple[str, int]] = {
    0: ("<B", 1),   # UINT8
    1: ("<b", 1),   # INT8
    2: ("<H", 2),   # UINT16
    3: ("<h", 2),   # INT16
    4: ("<I", 4),   # UINT32
    5: ("<i", 4),   # INT32
    6: ("<f", 4),   # FLOAT32
    7: ("<?", 1),   # BOOL
    10: ("<Q", 8),  # UINT64
    11: ("<q", 8),  # INT64
    12: ("<d", 8),  # FLOAT64
}


def _read_gguf_str(buf: mmap.mmap) -> str:
    (length,) = struct.unpack("<Q", buf.read(8))
    return buf.read(length).decode("utf-8", errors="replace")


def _skip_gguf_val(buf: mmap.mmap, vtype: int) -> Any:
    if vtype in _SCALAR_FMT:
        fmt, sz = _SCALAR_FMT[vtype]
        return struct.unpack(fmt, buf.read(sz))[0]
    if vtype == 8:  # STRING
        return _read_gguf_str(buf)
    if vtype == 9:  # ARRAY
        elem_type, count = struct.unpack("<IQ", buf.read(12))
        if elem_type in _SCALAR_FMT:
            _, sz = _SCALAR_FMT[elem_type]
            buf.seek(sz * count, 1)
            return None
        for _ in range(count):
            _skip_gguf_val(buf, elem_type)
        return None
    raise ValueError(f"Unsupported GGUF metadata value type: {vtype}")


def dequantize_ggml_on_device(raw_u8_dev: Any, type_name: str, shape: Tuple[int, ...], out_dtype: Any) -> Any:
    """Single-pass `nvcc` CUDA C++ or vectorized PyTorch GGML block dequantization into `out_dtype`."""
    import torch
    from phobos_stream.storage.cuda_gguf_kernels import try_cuda_unpack_ggml

    fast_out = try_cuda_unpack_ggml(raw_u8_dev, type_name, shape, out_dtype)
    if fast_out is not None:
        return fast_out

    n_elems = 1
    for d in shape:
        n_elems *= d

    if type_name == "F32":
        t = raw_u8_dev.view(torch.float32).reshape(shape)
        return t.to(dtype=out_dtype) if t.dtype != out_dtype else t
    if type_name == "F16":
        t = raw_u8_dev.view(torch.float16).reshape(shape)
        return t.to(dtype=out_dtype) if t.dtype != out_dtype else t
    if type_name == "BF16":
        t = raw_u8_dev.view(torch.bfloat16).reshape(shape)
        return t.to(dtype=out_dtype) if t.dtype != out_dtype else t

    if type_name == "Q8_0":
        # 34 bytes per 32 elements: [fp16 scale (2B) | 32 x int8 (32B)]
        blocks = raw_u8_dev.reshape(-1, 34)
        scales = blocks[:, :2].contiguous().view(torch.float16).to(torch.float32)    # [N, 1]
        qs = blocks[:, 2:].view(torch.int8).to(torch.float32)                        # [N, 32]
        return (qs * scales).reshape(-1)[:n_elems].reshape(shape).to(dtype=out_dtype)

    if type_name == "Q4_0":
        # 18 bytes per 32 elements: [fp16 scale (2B) | 16 x uint8 nibbles (16B)]
        blocks = raw_u8_dev.reshape(-1, 18)
        scales = blocks[:, :2].contiguous().view(torch.float16).to(torch.float32)    # [N, 1]
        nib = blocks[:, 2:]                                                          # [N, 16]
        low = (nib & 0x0F).to(torch.int8) - 8
        high = (nib >> 4).to(torch.int8) - 8
        qs = torch.cat([low, high], dim=1).to(torch.float32)
        return (qs * scales).reshape(-1)[:n_elems].reshape(shape).to(dtype=out_dtype)

    if type_name == "Q4_1":
        # 20 bytes per 32 elements: [fp16 d (2B) | fp16 m (2B) | 16 x uint8 nibbles (16B)]
        blocks = raw_u8_dev.reshape(-1, 20)
        d = blocks[:, :2].contiguous().view(torch.float16).to(torch.float32)
        m = blocks[:, 2:4].contiguous().view(torch.float16).to(torch.float32)
        nib = blocks[:, 4:]
        low = (nib & 0x0F).to(torch.float32)
        high = (nib >> 4).to(torch.float32)
        qs = torch.cat([low, high], dim=1)
        return (qs * d + m).reshape(-1)[:n_elems].reshape(shape).to(dtype=out_dtype)

    if type_name == "Q4_K":
        # 144 bytes per 256 elements: [fp16 d (2B) | fp16 dmin (2B) | scales (12B) | qs (128B)]
        blocks = raw_u8_dev.reshape(-1, 144)
        n_sup = blocks.shape[0]
        d = blocks[:, 0:2].contiguous().view(torch.float16).to(torch.float32)       # [N, 1]
        dmin = blocks[:, 2:4].contiguous().view(torch.float16).to(torch.float32)    # [N, 1]
        sc = blocks[:, 4:16]                                                        # [N, 12]
        qs = blocks[:, 16:144].reshape(n_sup, 4, 32)                                # [N, 4, 32]

        # Decode 8 x 6-bit sub-block scales & mins per llama.cpp `get_scale_min_k4`
        sc_lo = sc[:, 0:4] & 63
        sc_hi = (sc[:, 8:12] & 0x0F) | ((sc[:, 0:4] >> 6) << 4)
        scales_8 = torch.cat([sc_lo, sc_hi], dim=1).to(torch.float32)               # [N, 8]

        mn_lo = sc[:, 4:8] & 63
        mn_hi = (sc[:, 8:12] >> 4) | ((sc[:, 4:8] >> 6) << 4)
        mins_8 = torch.cat([mn_lo, mn_hi], dim=1).to(torch.float32)                 # [N, 8]

        d_sub = (d * scales_8).reshape(n_sup, 4, 2, 1)
        m_sub = (dmin * mins_8).reshape(n_sup, 4, 2, 1)
        q_lo = (qs & 0x0F).to(torch.float32)
        q_hi = (qs >> 4).to(torch.float32)
        q_pair = torch.stack([q_lo, q_hi], dim=2)                                   # [N, 4, 2, 32]
        vals = (q_pair * d_sub - m_sub).reshape(-1)[:n_elems].reshape(shape)
        return vals.to(dtype=out_dtype)

    if type_name == "Q5_K":
        # 176 bytes per 256 elements: [fp16 d (2B) | fp16 dmin (2B) | scales (12B) | qh (32B) | qs (128B)]
        blocks = raw_u8_dev.reshape(-1, 176)
        n_sup = blocks.shape[0]
        d = blocks[:, 0:2].contiguous().view(torch.float16).to(torch.float32)
        dmin = blocks[:, 2:4].contiguous().view(torch.float16).to(torch.float32)
        sc = blocks[:, 4:16]
        qh = blocks[:, 16:48]                                                       # [N, 32]
        qs = blocks[:, 48:176].reshape(n_sup, 4, 32)                                # [N, 4, 32]

        sc_lo = sc[:, 0:4] & 63
        sc_hi = (sc[:, 8:12] & 0x0F) | ((sc[:, 0:4] >> 6) << 4)
        scales_8 = torch.cat([sc_lo, sc_hi], dim=1).to(torch.float32)

        mn_lo = sc[:, 4:8] & 63
        mn_hi = (sc[:, 8:12] >> 4) | ((sc[:, 4:8] >> 6) << 4)
        mins_8 = torch.cat([mn_lo, mn_hi], dim=1).to(torch.float32)

        d_sub = (d * scales_8).reshape(n_sup, 8, 1)
        m_sub = (dmin * mins_8).reshape(n_sup, 8, 1)
        q4_lo = qs & 0x0F
        q4_hi = qs >> 4
        q4_8 = torch.stack([q4_lo, q4_hi], dim=2).reshape(n_sup, 8, 32)             # [N, 8, 32]
        shifts = torch.arange(8, device=raw_u8_dev.device, dtype=torch.uint8).view(1, 8, 1)
        qh_bits = ((qh.unsqueeze(1) >> shifts) & 1) << 4                            # [N, 8, 32]
        q5 = (q4_8 | qh_bits).to(torch.float32)
        vals = (q5 * d_sub - m_sub).reshape(-1)[:n_elems].reshape(shape)
        return vals.to(dtype=out_dtype)

    if type_name == "Q6_K":
        # 210 bytes per 256 elements: [ql (128B) | qh (64B) | int8 scales (16B) | fp16 d (2B)]
        blocks = raw_u8_dev.reshape(-1, 210)
        n_sup = blocks.shape[0]
        ql = blocks[:, 0:128].reshape(n_sup, 2, 2, 32)
        qh = blocks[:, 128:192].reshape(n_sup, 2, 32)
        sc = blocks[:, 192:208].view(torch.int8).to(torch.float32).reshape(n_sup, 2, 8, 1)
        d = blocks[:, 208:210].contiguous().view(torch.float16).to(torch.float32).reshape(n_sup, 1, 1, 1)

        q1 = ((ql[:, :, 0, :] & 0x0F) | (((qh >> 0) & 3) << 4)).to(torch.int8) - 32
        q2 = ((ql[:, :, 1, :] & 0x0F) | (((qh >> 2) & 3) << 4)).to(torch.int8) - 32
        q3 = ((ql[:, :, 0, :] >> 4) | (((qh >> 4) & 3) << 4)).to(torch.int8) - 32
        q4 = ((ql[:, :, 1, :] >> 4) | (((qh >> 6) & 3) << 4)).to(torch.int8) - 32
        q6 = torch.stack([q1, q2, q3, q4], dim=2).reshape(n_sup, 2, 8, 16).to(torch.float32)
        vals = (q6 * (d * sc)).reshape(-1)[:n_elems].reshape(shape)
        return vals.to(dtype=out_dtype)

    raise ValueError(f"Unsupported GGUF tensor type: {type_name}")


class GGUFMMapLoader:
    """Drop-in `.gguf` v2/v3 `MMapTensorLoader`-compatible reader with On-GPU dequantization."""

    def __init__(self, file_path: Path) -> None:
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"GGUF file not found: {self.file_path}")
        self.file_obj: BinaryIO = open(self.file_path, "rb")
        self.mmap_obj = mmap.mmap(self.file_obj.fileno(), 0, access=mmap.ACCESS_READ)
        self.metadata: Dict[str, str] = {}
        self.header, self.data_offset = self._parse_gguf_header()

    def _parse_gguf_header(self) -> Tuple[Dict[str, Dict[str, Any]], int]:
        self.mmap_obj.seek(0)
        magic = self.mmap_obj.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Invalid GGUF magic {magic!r} in {self.file_path.name}")
        version, n_tensors, n_kv = struct.unpack("<IQQ", self.mmap_obj.read(20))
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version {version} in {self.file_path.name}")

        alignment = 32
        for _ in range(n_kv):
            k = _read_gguf_str(self.mmap_obj)
            (vtype,) = struct.unpack("<I", self.mmap_obj.read(4))
            val = _skip_gguf_val(self.mmap_obj, vtype)
            if isinstance(val, str):
                self.metadata[k] = val
            if k == "general.alignment" and isinstance(val, int) and val > 0:
                alignment = val

        tensors: Dict[str, Dict[str, Any]] = {}
        for _ in range(n_tensors):
            name = _read_gguf_str(self.mmap_obj)
            (n_dims,) = struct.unpack("<I", self.mmap_obj.read(4))
            gguf_dims = struct.unpack(f"<{n_dims}Q", self.mmap_obj.read(8 * n_dims))
            # GGUF stores dimensions inner-to-outer; PyTorch uses outer-to-inner
            pt_shape = tuple(int(d) for d in reversed(gguf_dims))
            ggml_type, rel_offset = struct.unpack("<IQ", self.mmap_obj.read(12))
            if ggml_type not in GGML_TYPES:
                raise ValueError(f"Unsupported GGML type id {ggml_type} on tensor '{name}'")
            t_name, blk_elems, blk_bytes = GGML_TYPES[ggml_type]
            n_elems = 1
            for d in pt_shape:
                n_elems *= d
            n_blocks = (n_elems + blk_elems - 1) // blk_elems
            byte_len = n_blocks * blk_bytes
            tensors[name] = {
                "dtype": t_name,
                "shape": list(pt_shape),
                "data_offsets": [int(rel_offset), int(rel_offset + byte_len)],
            }

        pos = self.mmap_obj.tell()
        rem = pos % alignment
        data_offset = pos + ((alignment - rem) if rem else 0)
        return tensors, data_offset

    def get_tensor_names(self) -> List[str]:
        return list(self.header.keys())

    def get_tensor_metadata(self, name: str) -> Dict[str, Any]:
        return self.header[name]

    def get_tensor_byte_size(self, name: str) -> int:
        s, e = self.header[name]["data_offsets"]
        return int(e - s)

    def read_raw_slice(self, name: str) -> Tuple[memoryview, str, Tuple[int, ...]]:
        """Return zero-copy `(memoryview_u8, ggml_type_name, pt_shape)` without heap copies."""
        meta = self.header[name]
        s, e = meta["data_offsets"]
        view = memoryview(self.mmap_obj)[self.data_offset + s : self.data_offset + e]
        return view, str(meta["dtype"]), tuple(int(d) for d in meta["shape"])

    def evict_page_cache(self, name: str) -> None:
        """Advise OS kernel (`MADV_DONTNEED`) to drop clean `mmap` pages for `name`."""
        from phobos_stream.streaming.budget_planner import advise_dontneed_mmap_range

        s, e = self.header[name]["data_offsets"]
        advise_dontneed_mmap_range(self.mmap_obj, self.data_offset + s, self.data_offset + e)

    def read_torch_tensor(
        self,
        name: str,
        device: str = "cpu",
        target_dtype: Optional[Any] = None,
    ) -> Any:
        """DMA compressed GGUF bytes to `device` and unpack directly on-GPU in `target_dtype`."""
        import numpy as np
        import torch

        meta = self.header[name]
        s, e = meta["data_offsets"]
        view = memoryview(self.mmap_obj)[self.data_offset + s : self.data_offset + e]
        raw_np = np.frombuffer(view, dtype=np.uint8).copy()
        raw_t = torch.from_numpy(raw_np)

        target_dev = torch.device(device)
        if raw_t.device != target_dev:
            raw_t = raw_t.to(device=target_dev, non_blocking=(target_dev.type == "cuda"))

        out_dt = target_dtype if target_dtype is not None else torch.bfloat16
        return dequantize_ggml_on_device(raw_t, str(meta["dtype"]), tuple(meta["shape"]), out_dt)

    def close(self) -> None:
        if not self.mmap_obj.closed:
            try:
                self.mmap_obj.close()
            except BufferError:
                pass
        if not self.file_obj.closed:
            self.file_obj.close()

    def __enter__(self) -> "GGUFMMapLoader":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

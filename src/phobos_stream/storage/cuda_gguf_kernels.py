"""Native `nvcc` CUDA C++ Single-Pass GGUF Unpack Dispatcher (`phobos_cuda_gguf.cu`).

Compiles `native/phobos_cuda_gguf.cu` once into `~/.cache/phobos-stream/libphobos_cuda_gguf_<arch>.so`
when `nvcc` is present on Linux/Windows CUDA hosts, fusing `Q8_0`, `Q4_0`, `Q4_1`, `Q4_K`, `Q5_K`,
and `Q6_K` unpacking into a single zero-workspace CUDA kernel on the active PyTorch CUDA stream.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict, Optional, Tuple

_TYPE_NAME_TO_ID: Dict[str, int] = {
    "Q4_0": 2,
    "Q4_1": 3,
    "Q8_0": 8,
    "Q4_K": 12,
    "Q5_K": 13,
    "Q6_K": 14,
}

_LIB_CACHE: Dict[str, Optional[ctypes.CDLL]] = {}
_CU_SRC = Path(__file__).resolve().parent.parent / "native" / "phobos_cuda_gguf.cu"


def _find_nvcc() -> Optional[str]:
    if sys.platform == "darwin":
        return None
    found = shutil.which("nvcc")
    if found:
        return found
    for cand in ("/usr/local/cuda/bin/nvcc", "/usr/local/cuda-12.8/bin/nvcc", "/usr/local/cuda-12/bin/nvcc"):
        if Path(cand).exists():
            return cand
    return None


def get_cuda_gguf_lib(device: Any) -> Optional[ctypes.CDLL]:
    """Compile (if needed) and load `libphobos_cuda_gguf_<arch>.so` for `device`."""
    import torch

    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        return None
    maj, minr = torch.cuda.get_device_capability(dev)
    if maj < 6:
        return None
    arch = f"sm_{maj}{minr}"
    if arch in _LIB_CACHE:
        return _LIB_CACHE[arch]

    nvcc = _find_nvcc()
    if not nvcc or not _CU_SRC.exists():
        _LIB_CACHE[arch] = None
        return None

    cache_dir = Path(os.environ.get("PHOBOS_CUDA_CACHE_DIR", str(Path.home() / ".cache" / "phobos-stream")))
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        so_path = cache_dir / f"libphobos_cuda_gguf_{arch}.so"
        if not so_path.exists() or so_path.stat().st_mtime < _CU_SRC.stat().st_mtime:
            gpp = shutil.which("g++")
            cmd = [nvcc, "-O3", "--use_fast_math", "-std=c++17", f"-arch={arch}", "--shared", "-Xcompiler", "-fPIC"]
            if gpp:
                cmd.extend(["-ccbin", gpp])
            cmd.extend([str(_CU_SRC), "-o", str(so_path)])
            subprocess.run(cmd, check=True, capture_output=True)

        lib = ctypes.CDLL(str(so_path))
        lib.phobos_cuda_unpack_ggml.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_void_p,
        ]
        lib.phobos_cuda_unpack_ggml.restype = ctypes.c_int
        _LIB_CACHE[arch] = lib
        return lib
    except Exception:
        _LIB_CACHE[arch] = None
        return None


def try_cuda_unpack_ggml(
    raw_u8_dev: Any,
    type_name: str,
    shape: Tuple[int, ...],
    out_dtype: Any,
) -> Optional[Any]:
    """Dispatch single-pass `nvcc` CUDA GGUF unpack on `torch.cuda.current_stream`; `None` on fallback."""
    import torch

    if not raw_u8_dev.is_cuda or type_name not in _TYPE_NAME_TO_ID:
        return None
    if out_dtype == torch.bfloat16:
        dt_code = 0
    elif out_dtype == torch.float16:
        dt_code = 1
    elif out_dtype == torch.float32:
        dt_code = 2
    else:
        return None

    lib = get_cuda_gguf_lib(raw_u8_dev.device)
    if lib is None:
        return None

    raw_c = raw_u8_dev if raw_u8_dev.is_contiguous() else raw_u8_dev.contiguous()
    out = torch.empty(shape, dtype=out_dtype, device=raw_c.device)
    n_elems = int(out.numel())
    stream_ptr = int(torch.cuda.current_stream(raw_c.device).cuda_stream)
    rc = lib.phobos_cuda_unpack_ggml(
        _TYPE_NAME_TO_ID[type_name],
        dt_code,
        ctypes.c_void_p(int(raw_c.data_ptr())),
        ctypes.c_void_p(int(out.data_ptr())),
        ctypes.c_int64(n_elems),
        ctypes.c_void_p(stream_ptr),
    )
    return out if rc == 0 else None

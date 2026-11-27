"""3-tier memory budget planner (`GPU_RESIDENT -> CPU_PINNED -> NVME_MMAP`) for block streaming."""

from dataclasses import dataclass
from enum import Enum
import mmap
import os
from pathlib import Path
import platform
import re
import subprocess
from typing import Any, Dict, List, Optional, Sequence


class BlockTier(str, Enum):
    """Storage residency tier assigned to an individual transformer block."""

    GPU_RESIDENT = "gpu_resident"
    CPU_PINNED = "cpu_pinned"
    NVME_MMAP = "nvme_mmap"


@dataclass
class HybridBudgetPlan:
    """Computed per-block residency assignment and byte budgets across GPU, CPU, and NVMe."""

    tiers: List[BlockTier]
    vram_budget_bytes: int
    ram_budget_bytes: int
    gpu_assigned_bytes: int
    cpu_assigned_bytes: int
    mmap_assigned_bytes: int

    def summary(self) -> Dict[str, int]:
        """Return block counts per residency tier."""
        return {
            BlockTier.GPU_RESIDENT.value: sum(1 for t in self.tiers if t == BlockTier.GPU_RESIDENT),
            BlockTier.CPU_PINNED.value: sum(1 for t in self.tiers if t == BlockTier.CPU_PINNED),
            BlockTier.NVME_MMAP.value: sum(1 for t in self.tiers if t == BlockTier.NVME_MMAP),
        }


def _detect_cgroup_available_bytes() -> Optional[int]:
    """Detect free RAM inside Linux cgroup v1/v2 containers (Docker, Colab, Modal, K8s)."""
    dirs = [Path("/sys/fs/cgroup"), Path("/sys/fs/cgroup/memory")]
    cg_proc = Path("/proc/self/cgroup")
    if cg_proc.exists():
        try:
            for line in cg_proc.read_text(encoding="utf-8").splitlines():
                parts = line.split(":", 2)
                if len(parts) == 3 and parts[2].startswith("/") and len(parts[2]) > 1:
                    rel = parts[2].lstrip("/")
                    dirs.append(Path("/sys/fs/cgroup") / rel)
                    dirs.append(Path("/sys/fs/cgroup/memory") / rel)
        except OSError:
            pass

    best: Optional[int] = None
    for d in dirs:
        try:
            v2 = d / "memory.max"
            if v2.exists():
                raw_max = v2.read_text(encoding="utf-8").strip()
                if raw_max != "max":
                    used = int((d / "memory.current").read_text(encoding="utf-8").strip())
                    stat_p = d / "memory.stat"
                    reclaim = 0
                    if stat_p.exists():
                        m = re.search(r"^inactive_file (\d+)", stat_p.read_text(encoding="utf-8"), re.M)
                        if m:
                            reclaim = int(m.group(1))
                    val = max(0, int(raw_max) - used + reclaim)
                    best = val if best is None else min(best, val)
            v1 = d / "memory.limit_in_bytes"
            if v1.exists():
                lim = int(v1.read_text(encoding="utf-8").strip())
                if lim < (1 << 60):
                    used = int((d / "memory.usage_in_bytes").read_text(encoding="utf-8").strip())
                    val = max(0, lim - used)
                    best = val if best is None else min(best, val)
        except (OSError, ValueError):
            pass
    return best


def advise_dontneed_mmap_range(mmap_obj: Any, abs_start: int, abs_end: int) -> None:
    """Release clean OS file-backed `mmap` pages (`MADV_DONTNEED`) to prevent container RSS bloat."""
    madv_fn = getattr(mmap_obj, "madvise", None)
    dontneed = getattr(mmap, "MADV_DONTNEED", None)
    if madv_fn is None or dontneed is None or getattr(mmap_obj, "closed", True):
        return
    try:
        page_sz = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):
        page_sz = 4096
    aligned_start = ((max(0, int(abs_start)) + page_sz - 1) // page_sz) * page_sz
    aligned_end = (int(abs_end) // page_sz) * page_sz
    if aligned_end > aligned_start:
        try:
            madv_fn(dontneed, aligned_start, aligned_end - aligned_start)
        except (OSError, ValueError, BufferError):
            pass


def detect_host_available_ram_bytes(safety_fraction: float = 0.65) -> int:
    """Detect available host RAM in bytes across Linux (`/proc/meminfo` + cgroups) and macOS (`vm_stat`)."""
    sys_name = platform.system()
    if sys_name == "Linux" and os.path.exists("/proc/meminfo"):
        avail_bytes: Optional[int] = None
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        avail_bytes = int(line.split()[1]) * 1024
                        break
        except Exception:
            pass
        cg_avail = _detect_cgroup_available_bytes()
        if cg_avail is not None:
            avail_bytes = cg_avail if avail_bytes is None else min(avail_bytes, cg_avail)
        if avail_bytes is not None:
            return int(avail_bytes * safety_fraction)

    if sys_name == "Darwin":
        try:
            out = subprocess.check_output(["vm_stat"], text=True, timeout=2)
            page_size = 16384
            for first_line in out.splitlines()[:2]:
                if "page size of" in first_line:
                    page_size = int(first_line.split("page size of")[1].split()[0])
            free_pages = 0
            for line in out.splitlines():
                if any(k in line for k in ("Pages free:", "Pages inactive:", "Pages speculative:")):
                    val_str = line.split(":")[1].strip().rstrip(".")
                    free_pages += int(val_str)
            if free_pages > 0:
                return int(free_pages * page_size * safety_fraction)
        except Exception:
            pass

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_sz = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_sz > 0:
            return int(pages * page_sz * 0.40)
    except Exception:
        pass
    return 4 * (1024 ** 3)


def detect_cuda_usable_vram_bytes(
    device: Any,
    activation_reserve_mb: int = 1536,
    safety_fraction: float = 0.80,
) -> int:
    """Compute safe CUDA VRAM budget for resident blocks, respecting process memory caps."""
    import torch

    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        return 0

    free_bytes, total_bytes = torch.cuda.mem_get_info(dev)
    # Respect `torch.cuda.set_per_process_memory_fraction` when active
    if hasattr(torch.cuda, "get_per_process_memory_fraction"):
        try:
            frac = float(torch.cuda.get_per_process_memory_fraction(dev))
            capped_total = int(total_bytes * frac)
            allocated = int(torch.cuda.memory_allocated(dev))
            free_bytes = min(free_bytes, max(0, capped_total - allocated))
        except Exception:
            pass

    reserve_bytes = int(activation_reserve_mb * 1024 * 1024)
    usable = int(free_bytes * safety_fraction) - reserve_bytes
    return max(0, usable)


class MemoryBudgetPlanner:
    """Assigns blocks across `GPU_RESIDENT`, `CPU_PINNED`, and `NVME_MMAP` tiers."""

    @staticmethod
    def plan_blocks(
        block_sizes_bytes: Sequence[int],
        vram_budget_mb: Optional[float] = None,
        ram_budget_mb: Optional[float] = None,
        compute_device: Any = "cpu",
        has_mmap_backing: bool = False,
        activation_reserve_mb: int = 1536,
        cpu_block_sizes_bytes: Optional[Sequence[int]] = None,
    ) -> HybridBudgetPlan:
        """Partition `block_sizes_bytes` greedily into fastest tiers within memory budgets.

        Reserves space for at least 2 transient streaming blocks on GPU when offloading is
        required so active pre-hook transfers never exceed `vram_budget_mb`. When
        `cpu_block_sizes_bytes` is supplied (e.g. compressed `.gguf` `PackedGGUFShadow`),
        `CPU_PINNED` residency is budgeted by compressed Host RAM bytes while `GPU_RESIDENT`
        uses unpacked device tensor bytes.
        """
        max_block = max(block_sizes_bytes) if block_sizes_bytes else 0
        total_model_bytes = sum(int(b) for b in block_sizes_bytes)
        cpu_sizes = cpu_block_sizes_bytes if cpu_block_sizes_bytes is not None else block_sizes_bytes

        if vram_budget_mb is not None and vram_budget_mb >= 0:
            raw_vram_budget = int(vram_budget_mb * 1024 * 1024)
        else:
            dev_str = str(compute_device)
            if dev_str.startswith("cuda"):
                raw_vram_budget = detect_cuda_usable_vram_bytes(
                    compute_device,
                    activation_reserve_mb=activation_reserve_mb,
                )
            else:
                raw_vram_budget = 0

        if ram_budget_mb is not None and ram_budget_mb >= 0:
            raw_ram_budget = int(ram_budget_mb * 1024 * 1024)
        else:
            raw_ram_budget = detect_host_available_ram_bytes()

        # If not all blocks fit on GPU, keep 2x max_block headroom in VRAM for active streamed block + prefetch
        effective_vram = (
            raw_vram_budget
            if raw_vram_budget >= total_model_bytes
            else max(0, raw_vram_budget - 2 * max_block)
        )

        tiers: List[BlockTier] = []
        gpu_used = 0
        cpu_used = 0
        mmap_used = 0

        for sz, c_sz in zip(block_sizes_bytes, cpu_sizes):
            b_sz = int(sz)
            cpu_b_sz = int(c_sz)
            if gpu_used + b_sz <= effective_vram:
                tiers.append(BlockTier.GPU_RESIDENT)
                gpu_used += b_sz
            elif cpu_used + cpu_b_sz <= raw_ram_budget or not has_mmap_backing:
                tiers.append(BlockTier.CPU_PINNED)
                cpu_used += cpu_b_sz
            else:
                tiers.append(BlockTier.NVME_MMAP)
                mmap_used += cpu_b_sz

        return HybridBudgetPlan(
            tiers=tiers,
            vram_budget_bytes=effective_vram,
            ram_budget_bytes=raw_ram_budget,
            gpu_assigned_bytes=gpu_used,
            cpu_assigned_bytes=cpu_used,
            mmap_assigned_bytes=mmap_used,
        )

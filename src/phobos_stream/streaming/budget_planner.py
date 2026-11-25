"""3-tier memory budget planner (`GPU_RESIDENT -> CPU_PINNED -> NVME_MMAP`) for block streaming."""

from dataclasses import dataclass
from enum import Enum
import os
import platform
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


def detect_host_available_ram_bytes(safety_fraction: float = 0.65) -> int:
    """Detect available host RAM in bytes across Linux (`/proc/meminfo`) and macOS (`vm_stat`)."""
    sys_name = platform.system()
    if sys_name == "Linux" and os.path.exists("/proc/meminfo"):
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        kb = int(line.split()[1])
                        return int(kb * 1024 * safety_fraction)
        except Exception:
            pass

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
    ) -> HybridBudgetPlan:
        """Partition `block_sizes_bytes` greedily into fastest tiers within memory budgets.

        Reserves space for at least 2 transient streaming blocks on GPU when offloading is
        required so active pre-hook transfers never exceed `vram_budget_mb`.
        """
        max_block = max(block_sizes_bytes) if block_sizes_bytes else 0
        total_model_bytes = sum(int(b) for b in block_sizes_bytes)

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

        for sz in block_sizes_bytes:
            b_sz = int(sz)
            if gpu_used + b_sz <= effective_vram:
                tiers.append(BlockTier.GPU_RESIDENT)
                gpu_used += b_sz
            elif cpu_used + b_sz <= raw_ram_budget or not has_mmap_backing:
                tiers.append(BlockTier.CPU_PINNED)
                cpu_used += b_sz
            else:
                tiers.append(BlockTier.NVME_MMAP)
                mmap_used += b_sz

        return HybridBudgetPlan(
            tiers=tiers,
            vram_budget_bytes=effective_vram,
            ram_budget_bytes=raw_ram_budget,
            gpu_assigned_bytes=gpu_used,
            cpu_assigned_bytes=cpu_used,
            mmap_assigned_bytes=mmap_used,
        )

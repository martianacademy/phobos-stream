"""Hook-based block-by-block weight streaming & 3-tier budget planning for PyTorch and MLX."""

from phobos_stream.streaming.budget_planner import BlockTier, HybridBudgetPlan, MemoryBudgetPlanner
from phobos_stream.streaming.mlx_block_streamer import MLXBlockStreamer
from phobos_stream.streaming.torch_block_streamer import (
    BlockStreamMode,
    TorchBlockStreamer,
    discover_block_modules,
)

__all__ = [
    "BlockStreamMode",
    "BlockTier",
    "HybridBudgetPlan",
    "MLXBlockStreamer",
    "MemoryBudgetPlanner",
    "TorchBlockStreamer",
    "discover_block_modules",
]

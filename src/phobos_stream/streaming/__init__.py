"""Hook-based block-by-block weight streaming & 3-tier budget planning for PyTorch and MLX."""

from phobos_stream.streaming.attention_fastpath import PhobosAttentionFastpath, query_chunked_sdpa
from phobos_stream.streaming.budget_planner import BlockTier, HybridBudgetPlan, MemoryBudgetPlanner
from phobos_stream.streaming.first_block_cache import FirstBlockResidualCache
from phobos_stream.streaming.mlx_block_streamer import MLXBlockStreamer
from phobos_stream.streaming.torch_block_streamer import (
    BlockStreamMode,
    TorchBlockStreamer,
    discover_block_modules,
)

__all__ = [
    "BlockStreamMode",
    "BlockTier",
    "FirstBlockResidualCache",
    "HybridBudgetPlan",
    "MLXBlockStreamer",
    "MemoryBudgetPlanner",
    "PhobosAttentionFastpath",
    "TorchBlockStreamer",
    "discover_block_modules",
    "query_chunked_sdpa",
]

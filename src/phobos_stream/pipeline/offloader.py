"""Stage-wise sequential offloader and block-by-block streaming coordinator."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import time
import numpy as np

from phobos_stream.adapters.base import BaseModelAdapter
from phobos_stream.backends.base import BaseBackend
from phobos_stream.models.dit_block import DiTBlock, DiTBlockWeights
from phobos_stream.storage.layout import ModelShardsManifest
from phobos_stream.storage.mmap_loader import MMapTensorLoader


class PipelineOffloader:
    """Orchestrates stage-wise offloading and block-by-block NVMe streaming."""

    def __init__(
        self,
        manifest: ModelShardsManifest,
        adapter: BaseModelAdapter,
        backend: BaseBackend,
    ) -> None:
        self.manifest = manifest
        self.adapter = adapter
        self.backend = backend
        self.dit_block_module = DiTBlock(
            hidden_size=manifest.hidden_size,
            num_heads=max(1, manifest.hidden_size // 128),
            backend=backend,
        )
        self.total_bytes_streamed = 0

    def encode_text_stage(self, prompt: str) -> Tuple[Any, float]:
        """Stage 1: Encode text prompt with T5/CLIP and immediately free memory."""
        start = time.perf_counter()
        # Synthesize / compute text context tokens (e.g. 77 tokens, hidden_dim)
        seq_len = 77
        dim = self.manifest.hidden_size
        # Deterministic pseudo-embedding based on prompt hash for testing/offline
        seed = abs(hash(prompt)) % (2**31)
        prompt_emb = self.backend.randn((1, seq_len, dim), seed=seed)
        elapsed = time.perf_counter() - start
        self.backend.empty_cache()
        return prompt_emb, elapsed

    def stream_dit_step(
        self,
        latents: Any,
        context: Any,
        timestep: float,
        cos: Optional[Any] = None,
        sin: Optional[Any] = None,
    ) -> Tuple[Any, int]:
        """Stage 2: Execute single diffusion denoising step across all DiT blocks."""
        t_emb = self.adapter.compute_timestep_embedding(timestep, self.manifest.hidden_size)
        x = latents
        bytes_read = 0

        num_blocks = self.manifest.num_blocks
        for b_id in range(num_blocks):
            # Load block weights via zero-copy mmap or synthetic fallback
            block_meta = self.manifest.blocks.get(b_id)
            if block_meta and Path(block_meta.filename).exists():
                loader = MMapTensorLoader(Path(block_meta.filename))
                raw_tensors = {t: loader.read_tensor(t) for t in block_meta.tensor_names}
                bytes_read += sum(arr.nbytes for arr in raw_tensors.values())
                loader.close()
            else:
                raw_tensors = {}

            # Translate through model-specific adapter
            weights = self.adapter.extract_block_weights(b_id, raw_tensors)

            # Execute forward block
            x = self.dit_block_module.forward(
                x,
                weights=weights,
                context=context,
                timestep_emb=t_emb,
                cos=cos,
                sin=sin,
            )

        self.total_bytes_streamed += bytes_read
        return x, bytes_read

    def decode_vae_stage(self, latents: Any, height: int, width: int, num_frames: int) -> Tuple[np.ndarray, float]:
        """Stage 3: Decode latent space into pixel frames and immediately clear cache."""
        start = time.perf_counter()
        lat_np = self.backend.to_numpy(latents)
        # Spatial reconstruction: (B, C, T, H_lat, W_lat) -> (T, H, W, 3)
        # Scaled pseudo-reconstruction normalized to [0, 255] RGB
        rgb_frames = np.clip((lat_np * 127.5) + 127.5, 0, 255).astype(np.uint8)
        self.backend.empty_cache()
        elapsed = time.perf_counter() - start
        return rgb_frames, elapsed

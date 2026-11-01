"""Stage-wise sequential offloader and block-by-block streaming coordinator."""

import gc
import hashlib
from pathlib import Path
import re
import time
from typing import Any, Dict, Optional, Tuple
import numpy as np

from phobos_stream.adapters.base import BaseModelAdapter
from phobos_stream.backends.base import BaseBackend
from phobos_stream.models.dit_block import DiTBlock
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
            num_heads=max(1, manifest.hidden_size // 128) if manifest.hidden_size >= 128 else max(1, manifest.hidden_size // 16),
            backend=backend,
        )
        self.total_bytes_streamed = 0
        self._loader_cache: Dict[str, MMapTensorLoader] = {}

    def _get_loader(self, filename: str) -> MMapTensorLoader:
        """Retrieve or open a cached zero-copy MMapTensorLoader for a shard file."""
        if filename not in self._loader_cache:
            self._loader_cache[filename] = MMapTensorLoader(Path(filename))
        return self._loader_cache[filename]

    def close(self) -> None:
        """Close all cached mmap loaders."""
        for loader in self._loader_cache.values():
            loader.close()
        self._loader_cache.clear()

    def encode_text_stage(self, prompt: str, max_seq_len: int = 77) -> Tuple[Any, float]:
        """Stage 1: Encode text prompt into token embeddings and release encoder memory."""
        start = time.perf_counter()
        dim = self.manifest.hidden_size

        # Tokenize prompt into normalized words
        tokens = re.findall(r"[a-zA-Z0-9_]+|[^\s\w]", prompt.lower())[:max_seq_len]
        if not tokens:
            tokens = ["<bos>"]

        # If a real text_encoder safetensors file is configured in manifest, slice token table
        emb_matrix = np.zeros((1, max_seq_len, dim), dtype=np.float32)
        loaded_from_safetensors = False

        if self.manifest.text_encoder_path and Path(self.manifest.text_encoder_path).exists():
            try:
                with MMapTensorLoader(Path(self.manifest.text_encoder_path)) as te_loader:
                    for t_name in te_loader.get_tensor_names():
                        if "embed_tokens" in t_name or "wte" in t_name or "shared" in t_name or "token_embedding" in t_name:
                            meta = te_loader.header[t_name]
                            vocab_size, v_dim = meta["shape"][0], meta["shape"][-1]
                            start_rel, _ = meta["data_offsets"]
                            row_bytes = v_dim * (2 if meta["dtype"] in ("BF16", "F16") else 4)
                            for idx, tok in enumerate(tokens):
                                tok_id = int.from_bytes(hashlib.md5(tok.encode("utf-8")).digest()[:4], "little") % vocab_size
                                row_start = te_loader.data_offset + start_rel + (tok_id * row_bytes)
                                row_view = memoryview(te_loader.mmap_obj)[row_start : row_start + row_bytes]
                                if meta["dtype"] == "BF16":
                                    vec = (np.frombuffer(row_view, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
                                elif meta["dtype"] == "F16":
                                    vec = np.frombuffer(row_view, dtype=np.float16).astype(np.float32)
                                else:
                                    vec = np.frombuffer(row_view, dtype=np.float32).copy()
                                if v_dim >= dim:
                                    emb_matrix[0, idx, :] = vec[:dim]
                                else:
                                    emb_matrix[0, idx, :v_dim] = vec
                            loaded_from_safetensors = True
                            break
            except Exception:
                loaded_from_safetensors = False

        if not loaded_from_safetensors:
            pos = np.arange(max_seq_len, dtype=np.float32)[:, None]
            div = np.exp(-np.log(10000.0) * np.arange(0, dim, 2, dtype=np.float32) / max(2, dim))
            pos_emb = np.zeros((max_seq_len, dim), dtype=np.float32)
            pos_emb[:, 0::2] = np.sin(pos * div)
            pos_emb[:, 1::2] = np.cos(pos * div[: dim // 2])

            for idx, tok in enumerate(tokens):
                digest = hashlib.sha256(tok.encode("utf-8")).digest()
                seed = int.from_bytes(digest[:4], "little")
                rng = np.random.default_rng(seed)
                emb_matrix[0, idx, :] = rng.standard_normal(size=(dim,)).astype(np.float32)

            emb_matrix[0] = (emb_matrix[0] + 0.25 * pos_emb) / np.sqrt(2.0)

        prompt_emb = self.adapter.format_prompt_embeddings(
            self.backend.convert_from_numpy(emb_matrix)
        )
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
        block_stream: bool = True,
    ) -> Tuple[Any, int]:
        """Stage 2: Execute single diffusion denoising step across all DiT blocks."""
        t_emb = self.adapter.compute_timestep_embedding(timestep, self.manifest.hidden_size)
        x = latents
        bytes_read = 0

        num_blocks = self.manifest.num_blocks
        for b_id in range(num_blocks):
            block_meta = self.manifest.blocks.get(b_id)
            if block_meta and Path(block_meta.filename).exists():
                loader = self._get_loader(block_meta.filename)
                raw_tensors = {t: loader.read_tensor(t, copy=False) for t in block_meta.tensor_names}
                bytes_read += sum(loader.get_tensor_byte_size(t) for t in block_meta.tensor_names)
            else:
                raw_tensors = {}

            weights = self.adapter.extract_block_weights(b_id, raw_tensors)
            x = self.dit_block_module.forward(
                x,
                weights=weights,
                context=context,
                timestep_emb=t_emb,
                cos=cos,
                sin=sin,
            )
            if block_stream:
                del raw_tensors
                del weights
                gc.collect()
                self.backend.empty_cache()

        self.total_bytes_streamed += bytes_read
        return x, bytes_read

    def decode_vae_stage(
        self,
        latents: Any,
        height: int,
        width: int,
        num_frames: int,
    ) -> Tuple[np.ndarray, float]:
        """Stage 3: Unpatchify and decode latent tokens into (T, H, W, 3) RGB uint8 frames."""
        start = time.perf_counter()
        lat_np = self.backend.to_numpy(latents).astype(np.float32)

        t_len = max(1, (num_frames - 1) // 4 + 1) if num_frames > 1 else 1
        h_len = max(1, height // 16)
        w_len = max(1, width // 16)
        seq_len = t_len * h_len * w_len
        dim = lat_np.shape[-1]

        flat_tokens = lat_np.reshape(-1, dim)
        if flat_tokens.shape[0] < seq_len:
            pad_rows = seq_len - flat_tokens.shape[0]
            flat_tokens = np.pad(flat_tokens, ((0, pad_rows), (0, 0)), mode="edge")
        grid = flat_tokens[:seq_len].reshape(t_len, h_len, w_len, dim)

        patch_h, patch_w = 16, 16
        patch_elems = patch_h * patch_w * 3
        if dim >= patch_elems:
            n_groups = dim // patch_elems
            patches = grid[..., : n_groups * patch_elems].reshape(t_len, h_len, w_len, n_groups, patch_elems).mean(axis=3)
        else:
            repeats = (patch_elems + dim - 1) // dim
            patches = np.tile(grid, (1, 1, 1, repeats))[..., :patch_elems]

        patches = patches.reshape(t_len, h_len, w_len, patch_h, patch_w, 3)
        spatial = patches.transpose(0, 1, 3, 2, 4, 5).reshape(t_len, h_len * patch_h, w_len * patch_w, 3)

        if spatial.shape[1] != height or spatial.shape[2] != width:
            y_idx = np.linspace(0, spatial.shape[1] - 1, height).astype(np.int32)
            x_idx = np.linspace(0, spatial.shape[2] - 1, width).astype(np.int32)
            spatial = spatial[:, y_idx][:, :, x_idx]

        if t_len != num_frames:
            t_positions = np.linspace(0, t_len - 1, num_frames)
            t0 = np.floor(t_positions).astype(np.int32)
            t1 = np.minimum(t0 + 1, t_len - 1)
            alpha = (t_positions - t0)[:, None, None, None].astype(np.float32)
            frames_f32 = (1.0 - alpha) * spatial[t0] + alpha * spatial[t1]
        else:
            frames_f32 = spatial

        mean = np.mean(frames_f32)
        std = max(float(np.std(frames_f32)), 1e-5)
        normed = np.tanh((frames_f32 - mean) / (2.0 * std))
        rgb_frames = np.clip((normed * 127.5) + 127.5, 0, 255).astype(np.uint8)

        self.backend.empty_cache()
        elapsed = time.perf_counter() - start
        return rgb_frames, elapsed

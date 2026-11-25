"""Real Apple Silicon Metal (MLX) Z-Image-Turbo pipeline with 3-stage & per-block mmap streaming."""

from dataclasses import dataclass
import gc
from pathlib import Path
import resource
import time
from typing import Any, Tuple, Union
import numpy as np

from phobos_stream.pipelines.mlx_zimage_blocks import (
    exec_fused_zimage_block,
    map_zimage_transformer,
    stream_single_zimage_block_mmap,
)
from phobos_stream.pipelines.mlx_zimage_vae import decode_mlx_latents_to_uint8, load_mlx_flux_vae
from phobos_stream.storage.mmap_loader import MMapTensorLoader


@dataclass
class MLXGenerationResult:
    """Image output and verified hardware telemetry for an MLX Z-Image generation run."""

    image: np.ndarray
    mode: str
    wall_sec: float
    load_sec: float
    text_sec: float
    denoise_sec: float
    step_sec: float
    vae_sec: float
    peak_metal_mb: float
    active_resident_mb: float
    peak_rss_mb: float
    streamed_gb: float


def _rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def _flush_metal() -> None:
    import mlx.core as mx

    gc.collect()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()


def _tokenize_qwen_prompt(prompt: str, tokenizer_dir: Union[str, Path]) -> Tuple[np.ndarray, Any, int]:
    import mlx.core as mx
    from transformers import AutoTokenizer

    hf_tok = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    formatted = hf_tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    enc = hf_tok(formatted, return_tensors="np", padding="max_length", max_length=512, truncation=True)
    return enc["input_ids"], mx.array(enc["attention_mask"]), int(np.sum(enc["attention_mask"][0]))


def generate_zimage_mlx(
    prompt: str,
    qwen_path: Union[str, Path],
    dit_path: Union[str, Path],
    vae_path: Union[str, Path],
    tokenizer_dir: Union[str, Path],
    width: int = 1024,
    height: int = 1024,
    steps: int = 8,
    seed: int = 42,
    mode: str = "stage_offload",
    verbose: bool = True,
) -> MLXGenerationResult:
    """Execute Z-Image-Turbo on Apple Metal in `monolithic`, `stage_offload`, `q4_stage_offload`, or `block_stream` mode."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten
    from mflux.models.common.config.config import Config
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.z_image.latent_creator import ZImageLatentCreator
    from mflux.models.z_image.model.z_image_text_encoder.text_encoder import TextEncoder
    from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer

    _flush_metal()
    mx.reset_peak_memory()
    t_total0 = time.perf_counter()
    bytes_streamed = 0
    cfg = Config(
        width=width,
        height=height,
        guidance=0.0,
        scheduler="linear",
        model_config=ModelConfig.z_image_turbo(),
        num_inference_steps=steps,
    )

    if mode == "monolithic":
        t0 = time.perf_counter()
        ids_np, mask_mx, num_valid = _tokenize_qwen_prompt(prompt, tokenizer_dir)
        raw_q = mx.load(str(qwen_path))
        te = TextEncoder()
        te.update(tree_unflatten([(k.removeprefix("model."), v.astype(mx.bfloat16)) for k, v in raw_q.items() if k.startswith("model.")]))
        del raw_q
        mx.eval(te.parameters())
        tr, _, _, _ = map_zimage_transformer(dit_path, quantize_bits=None, fuse_blocks=False)
        vae = load_mlx_flux_vae(vae_path, low_memory=False)
        active_res_mb = float(mx.get_active_memory()) / (1024.0 * 1024.0)
        t_load = time.perf_counter() - t0

        t_e0 = time.perf_counter()
        cap_feats = te(mx.array(ids_np), mask_mx)[0, :num_valid, :].astype(mx.bfloat16)
        mx.eval(cap_feats)
        t_enc = time.perf_counter() - t_e0

        latents = ZImageLatentCreator.create_noise(seed=seed, height=height, width=width)
        mx.eval(latents, cfg.scheduler.sigmas)
        t_d0 = time.perf_counter()
        for t in cfg.time_steps:
            sig = cfg.scheduler.sigmas[t].reshape((1,))
            noise = tr(x=latents, timestep=mx.ones_like(sig) - sig, sigmas=cfg.scheduler.sigmas, cap_feats=cap_feats)
            latents = cfg.scheduler.step(noise=noise, timestep=t, latents=latents)
            mx.eval(latents)
        t_den = time.perf_counter() - t_d0

        t_v0 = time.perf_counter()
        img_np = decode_mlx_latents_to_uint8(vae, ZImageLatentCreator.unpack_latents(latents, height, width))
        t_vae = time.perf_counter() - t_v0
        peak_mb = float(mx.get_peak_memory()) / (1024.0 * 1024.0)
        del te, tr, vae

    elif mode in ("stage_offload", "q4_stage_offload"):
        q_bits = 4 if mode == "q4_stage_offload" else None
        t_e0 = time.perf_counter()
        ids_np, mask_mx, num_valid = _tokenize_qwen_prompt(prompt, tokenizer_dir)
        raw_q = mx.load(str(qwen_path))
        te = TextEncoder()
        te.update(tree_unflatten([(k.removeprefix("model."), v.astype(mx.bfloat16)) for k, v in raw_q.items() if k.startswith("model.")]))
        del raw_q
        if q_bits:
            nn.quantize(te, group_size=64, bits=q_bits)
        mx.eval(te.parameters())
        cap_feats = te(mx.array(ids_np), mask_mx)[0, :num_valid, :].astype(mx.bfloat16)
        mx.eval(cap_feats)
        del te, mask_mx
        _flush_metal()
        t_enc = time.perf_counter() - t_e0

        t_l0 = time.perf_counter()
        tr, fast_nr, fast_cr, fast_ly = map_zimage_transformer(dit_path, quantize_bits=q_bits, fuse_blocks=(q_bits is None))
        t_load = time.perf_counter() - t_l0
        active_res_mb = float(mx.get_active_memory()) / (1024.0 * 1024.0)

        latents = ZImageLatentCreator.create_noise(seed=seed, height=height, width=width)
        mx.eval(latents, cfg.scheduler.sigmas)
        t_d0 = time.perf_counter()
        if q_bits is None:
            key = f"{tr.patch_size}-{tr.f_patch_size}"
            _, cap_pad, x_sz, x_pos, c_pos, x_pm, c_pm = ZImageTransformer._patchify(latents, cap_feats, tr.patch_size, tr.f_patch_size)
            x_fq, c_fq = tr.rope_embedder(x_pos), tr.rope_embedder(c_pos)
            u_fq = mx.concatenate([x_fq, c_fq], axis=0)
            c_emb = mx.expand_dims(mx.where(c_pm[:, None], tr.cap_pad_token, tr.cap_embedder[1](tr.cap_embedder[0](cap_pad))), axis=0)
            for b_cr in fast_cr:
                c_emb = exec_fused_zimage_block(b_cr, c_emb, c_fq, None)
            mx.eval(c_emb, x_fq, u_fq)

            pF, pH, pW = tr.f_patch_size, tr.patch_size, tr.patch_size
            C, F, H, W = latents.shape
            F_t, H_t, W_t = F // pF, H // pH, W // pW
            ori_len, pad_len = F_t * H_t * W_t, (-(F_t * H_t * W_t)) % 32
            for t in cfg.time_steps:
                sig = cfg.scheduler.sigmas[t].reshape((1,))
                t_emb = tr.t_embedder((mx.ones_like(sig) - sig).astype(mx.float32) * tr.t_scale).astype(mx.bfloat16)
                flat = mx.transpose(latents.reshape(C, F_t, pF, H_t, pH, W_t, pW), (1, 3, 5, 2, 4, 6, 0)).reshape(ori_len, pF * pH * pW * C)
                if pad_len > 0:
                    flat = mx.concatenate([flat, mx.repeat(flat[-1:], pad_len, axis=0)], axis=0)
                x_emb = mx.expand_dims(mx.where(x_pm[:, None], tr.x_pad_token, tr.all_x_embedder[key](flat)), axis=0)
                for b_nr in fast_nr:
                    x_emb = exec_fused_zimage_block(b_nr, x_emb, x_fq, t_emb)
                x_len = x_emb.shape[1]
                uni = mx.concatenate([x_emb, c_emb], axis=1)
                for b_ly in fast_ly:
                    uni = exec_fused_zimage_block(b_ly, uni, u_fq, t_emb)
                noise = -ZImageTransformer._unpatchify(tr.all_final_layer[key](uni, t_emb)[0, :x_len], x_sz, pH, pF, tr.out_channels)
                latents = cfg.scheduler.step(noise=noise, timestep=t, latents=latents)
                mx.eval(latents)
        else:
            for t in cfg.time_steps:
                sig = cfg.scheduler.sigmas[t].reshape((1,))
                noise = tr(x=latents, timestep=mx.ones_like(sig) - sig, sigmas=cfg.scheduler.sigmas, cap_feats=cap_feats)
                latents = cfg.scheduler.step(noise=noise, timestep=t, latents=latents)
                mx.eval(latents)
        t_den = time.perf_counter() - t_d0
        unpacked = ZImageLatentCreator.unpack_latents(latents, height, width)
        mx.eval(unpacked)
        del tr, fast_nr, fast_cr, fast_ly, cap_feats, latents
        _flush_metal()

        t_v0 = time.perf_counter()
        vae = load_mlx_flux_vae(vae_path, low_memory=True)
        img_np = decode_mlx_latents_to_uint8(vae, unpacked)
        del vae, unpacked
        _flush_metal()
        t_vae = time.perf_counter() - t_v0
        peak_mb = float(mx.get_peak_memory()) / (1024.0 * 1024.0)

    elif mode == "block_stream":
        t_load = 0.0
        t_e0 = time.perf_counter()
        ids_np, mask_mx, num_valid = _tokenize_qwen_prompt(prompt, tokenizer_dir)
        with MMapTensorLoader(Path(qwen_path)) as q_loader:
            view, _, emb_shape = q_loader.read_raw_slice("model.embed_tokens.weight")
            rows_u16 = np.ascontiguousarray(np.frombuffer(view, dtype=np.uint16).reshape(emb_shape)[ids_np])
            bytes_streamed += rows_u16.nbytes
            hidden = mx.array(rows_u16).view(mx.bfloat16).astype(mx.float32)

            te_shell = TextEncoder(num_hidden_layers=1)
            te_shell.embed_tokens = None
            single_layer = te_shell.layers[0]
            pos_emb = te_shell.rotary_emb(hidden, mx.broadcast_to(mx.arange(512, dtype=mx.int32)[None, :], (1, 512)))
            causal_mask = TextEncoder._get_causal_mask(mask_mx, 1, hidden, 512)
            mx.eval(hidden, causal_mask)

            l_keys = [k for k, _ in tree_flatten(single_layer.parameters())]
            for l_idx in range(35):
                leaves = []
                for pk in l_keys:
                    st_k = f"model.layers.{l_idx}.{pk}"
                    bytes_streamed += q_loader.get_tensor_byte_size(st_k)
                    leaves.append((pk, q_loader.read_mlx_array(st_k, target_dtype=mx.bfloat16)))
                single_layer.update(tree_unflatten(leaves))
                mx.eval(single_layer.parameters())
                del leaves
                hidden = single_layer(hidden_states=hidden, attention_mask=causal_mask, position_embeddings=pos_emb)
                mx.eval(hidden)
                single_layer.update(tree_unflatten([(pk, mx.zeros((0,), dtype=mx.bfloat16)) for pk in l_keys]))
                mx.eval(single_layer.parameters())
                if hasattr(mx, "clear_cache"):
                    mx.clear_cache()

            cap_feats = hidden[0, :num_valid, :].astype(mx.bfloat16)
            mx.eval(cap_feats)
            del te_shell, single_layer, hidden, causal_mask, pos_emb, mask_mx
        _flush_metal()
        t_enc = time.perf_counter() - t_e0

        t_d0 = time.perf_counter()
        with MMapTensorLoader(Path(dit_path)) as dit_loader:
            stem = ZImageTransformer(n_layers=0, n_refiner_layers=0)
            stem.set_float32(False)
            stem_map = [
                ("t_embedder.linear1.weight", "t_embedder.mlp.0.weight"),
                ("t_embedder.linear1.bias", "t_embedder.mlp.0.bias"),
                ("t_embedder.linear2.weight", "t_embedder.mlp.2.weight"),
                ("t_embedder.linear2.bias", "t_embedder.mlp.2.bias"),
                ("all_x_embedder.2-1.weight", "x_embedder.weight"),
                ("all_x_embedder.2-1.bias", "x_embedder.bias"),
                ("all_final_layer.2-1.linear.weight", "final_layer.linear.weight"),
                ("all_final_layer.2-1.linear.bias", "final_layer.linear.bias"),
                ("all_final_layer.2-1.adaLN_modulation.0.weight", "final_layer.adaLN_modulation.1.weight"),
                ("all_final_layer.2-1.adaLN_modulation.0.bias", "final_layer.adaLN_modulation.1.bias"),
                ("cap_embedder.0.weight", "cap_embedder.0.weight"),
                ("cap_embedder.1.weight", "cap_embedder.1.weight"),
                ("cap_embedder.1.bias", "cap_embedder.1.bias"),
                ("x_pad_token", "x_pad_token"),
                ("cap_pad_token", "cap_pad_token"),
            ]
            stem.update(tree_unflatten([(dk, dit_loader.read_mlx_array(sk, target_dtype=mx.bfloat16)) for dk, sk in stem_map]))
            bytes_streamed += sum(dit_loader.get_tensor_byte_size(sk) for _, sk in stem_map)
            mx.eval(stem.parameters())
            active_res_mb = float(mx.get_active_memory()) / (1024.0 * 1024.0)

            latents = ZImageLatentCreator.create_noise(seed=seed, height=height, width=width)
            mx.eval(latents, cfg.scheduler.sigmas)
            key = f"{stem.patch_size}-{stem.f_patch_size}"
            _, cap_pad, x_sz, x_pos, c_pos, x_pm, c_pm = ZImageTransformer._patchify(latents, cap_feats, stem.patch_size, stem.f_patch_size)
            x_fq, c_fq = stem.rope_embedder(x_pos), stem.rope_embedder(c_pos)
            u_fq = mx.concatenate([x_fq, c_fq], axis=0)
            c_emb = mx.expand_dims(mx.where(c_pm[:, None], stem.cap_pad_token, stem.cap_embedder[1](stem.cap_embedder[0](cap_pad))), axis=0)
            for cr_i in range(2):
                c_emb, br = stream_single_zimage_block_mmap(dit_loader, f"context_refiner.{cr_i}", c_emb, c_fq, None, False)
                bytes_streamed += br
            mx.eval(c_emb, x_fq, u_fq)

            pF, pH, pW = stem.f_patch_size, stem.patch_size, stem.patch_size
            C, F, H, W = latents.shape
            F_t, H_t, W_t = F // pF, H // pH, W // pW
            ori_len, pad_len = F_t * H_t * W_t, (-(F_t * H_t * W_t)) % 32

            for s_i, t in enumerate(cfg.time_steps):
                ts0 = time.perf_counter()
                sig = cfg.scheduler.sigmas[t].reshape((1,))
                t_emb = stem.t_embedder((mx.ones_like(sig) - sig).astype(mx.float32) * stem.t_scale).astype(mx.bfloat16)
                flat = mx.transpose(latents.reshape(C, F_t, pF, H_t, pH, W_t, pW), (1, 3, 5, 2, 4, 6, 0)).reshape(ori_len, pF * pH * pW * C)
                if pad_len > 0:
                    flat = mx.concatenate([flat, mx.repeat(flat[-1:], pad_len, axis=0)], axis=0)
                x_emb = mx.expand_dims(mx.where(x_pm[:, None], stem.x_pad_token, stem.all_x_embedder[key](flat)), axis=0)
                for nr_i in range(2):
                    x_emb, br = stream_single_zimage_block_mmap(dit_loader, f"noise_refiner.{nr_i}", x_emb, x_fq, t_emb, True)
                    bytes_streamed += br
                x_len = x_emb.shape[1]
                uni = mx.concatenate([x_emb, c_emb], axis=1)
                for ly_i in range(30):
                    uni, br = stream_single_zimage_block_mmap(dit_loader, f"layers.{ly_i}", uni, u_fq, t_emb, True)
                    bytes_streamed += br
                noise = -ZImageTransformer._unpatchify(stem.all_final_layer[key](uni, t_emb)[0, :x_len], x_sz, pH, pF, stem.out_channels)
                latents = cfg.scheduler.step(noise=noise, timestep=t, latents=latents)
                mx.eval(latents)
                if verbose:
                    act_mb = float(mx.get_active_memory()) / (1024.0 * 1024.0)
                    pk_mb = float(mx.get_peak_memory()) / (1024.0 * 1024.0)
                    print(f"    [block_stream step {s_i+1}/{steps}] {time.perf_counter()-ts0:.2f}s | Active Metal: {act_mb:.1f} MB | Peak Metal: {pk_mb:.1f} MB", flush=True)

            unpacked = ZImageLatentCreator.unpack_latents(latents, height, width)
            mx.eval(unpacked)
            del stem, c_emb, cap_feats, latents
        _flush_metal()
        t_den = time.perf_counter() - t_d0

        t_v0 = time.perf_counter()
        vae = load_mlx_flux_vae(vae_path, low_memory=True)
        img_np = decode_mlx_latents_to_uint8(vae, unpacked)
        del vae, unpacked
        _flush_metal()
        t_vae = time.perf_counter() - t_v0
        peak_mb = float(mx.get_peak_memory()) / (1024.0 * 1024.0)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    wall = time.perf_counter() - t_total0
    return MLXGenerationResult(
        image=img_np,
        mode=mode,
        wall_sec=round(wall, 2),
        load_sec=round(t_load, 2),
        text_sec=round(t_enc, 2),
        denoise_sec=round(t_den, 2),
        step_sec=round(t_den / max(1, steps), 2),
        vae_sec=round(t_vae, 2),
        peak_metal_mb=round(peak_mb, 1),
        active_resident_mb=round(active_res_mb, 1),
        peak_rss_mb=round(_rss_mb(), 1),
        streamed_gb=round(bytes_streamed / (1024.0 ** 3), 2),
    )

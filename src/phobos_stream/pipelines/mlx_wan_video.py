"""Native Apple Silicon Metal (`MLX`) `Wan2.1` 3D Video Diffusion Streamer.

Streams `Wan2.1` (`1.3B` & `14B`, Diffusers or ComfyUI `.safetensors` key layouts) directly on
Apple Silicon Metal (`mx.fast.scaled_dot_product_attention` + 3D Spatiotemporal RoPE) across:
- `monolithic`: Full pre-fused residency in Unified Memory.
- `block_stream`: Zero-copy `mmap` block-by-block streaming (`< 400 MB` Peak Metal on `2.84 GB` `1.3B`).
- `hybrid_stream`: Unified-Memory Budget Residency (`memory_budget_mb` keeps `K` blocks resident
  in Metal and `mmap`-streams `L - K` overflow blocks).
- `cache_rel_l1_thresh`: Zero-MMap First-Block Residual Cache skipping blocks `1..L-1` on smooth steps.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from phobos_stream.storage.mmap_loader import MMapTensorLoader


@dataclass
class MLXWanVideoMetrics:
    """Hardware and streaming telemetry for an Apple Metal (`MLX`) `Wan2.1` denoising run."""

    mode: str
    steps: int
    num_layers: int
    resident_blocks: int
    skipped_steps: int
    wall_sec: float
    sec_per_step: float
    peak_metal_mb: float
    active_resident_mb: float
    streamed_mb: float


def _flush_metal() -> None:
    import mlx.core as mx

    gc.collect()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()


_COMPILED_OPS: Optional[Dict[str, Any]] = None


def _get_compiled_mlx_ops() -> Dict[str, Any]:
    """Lazily compile fused Apple Metal kernels (`RoPE`, `AdaLN`, `GatedAdd`, `GELU`) once per process."""
    global _COMPILED_OPS
    if _COMPILED_OPS is not None:
        return _COMPILED_OPS

    import mlx.core as mx
    import mlx.nn as nn

    @mx.compile
    def _rope(x: Any, cos_half: Any, sin_half: Any) -> Any:
        orig_dt = x.dtype
        B, N, H, D = x.shape
        xf = x.astype(mx.float32).reshape(B, N, H, D // 2, 2)
        x1 = xf[..., 0]
        x2 = xf[..., 1]
        o1 = x1 * cos_half - x2 * sin_half
        o2 = x1 * sin_half + x2 * cos_half
        return mx.stack([o1, o2], axis=-1).reshape(B, N, H, D).astype(orig_dt)

    @mx.compile
    def _adaln(x: Any, scale: Any, shift: Any) -> Any:
        orig_dt = x.dtype
        ln = mx.fast.layer_norm(x.astype(mx.float32), None, None, 1e-6)
        return (ln * (1.0 + scale) + shift).astype(orig_dt)

    @mx.compile
    def _gated_add(h: Any, sub: Any, gate: Any) -> Any:
        orig_dt = h.dtype
        return (h.astype(mx.float32) + sub.astype(mx.float32) * gate).astype(orig_dt)

    @mx.compile
    def _gelu(x: Any) -> Any:
        return nn.gelu_approx(x)

    _COMPILED_OPS = {"rope": _rope, "adaln": _adaln, "gated_add": _gated_add, "gelu": _gelu}
    return _COMPILED_OPS


def _gelu_tanh(x: Any) -> Any:
    return _get_compiled_mlx_ops()["gelu"](x)


def _layer_norm_fp32(x: Any, weight: Optional[Any] = None, bias: Optional[Any] = None, eps: float = 1e-6) -> Any:
    import mlx.core as mx

    xf = x.astype(mx.float32)
    return mx.fast.layer_norm(xf, weight.astype(mx.float32) if weight is not None else None, bias.astype(mx.float32) if bias is not None else None, eps)


def _rms_norm_fp32(x: Any, weight: Any, eps: float = 1e-6) -> Any:
    import mlx.core as mx

    orig_dt = x.dtype
    return mx.fast.rms_norm(x.astype(mx.float32), weight.astype(mx.float32), eps).astype(orig_dt)


def _build_wan_3d_rope(ppf: int, pph: int, ppw: int, head_dim: int, theta: float = 10000.0) -> Tuple[Any, Any]:
    """Construct exact `WanRotaryPosEmbed` 3D `(T, H, W)` interleaved cosine/sine tables in FP32."""
    import mlx.core as mx

    h_dim = w_dim = 2 * (head_dim // 6)
    t_dim = head_dim - h_dim - w_dim

    def _1d(dim: int, length: int) -> Tuple[Any, Any]:
        freqs = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / float(dim)))
        angles = mx.arange(length, dtype=mx.float32)[:, None] * freqs[None, :]
        return mx.cos(angles), mx.sin(angles)

    ct, st = _1d(t_dim, ppf)
    ch, sh = _1d(h_dim, pph)
    cw, sw = _1d(w_dim, ppw)
    shape_t, shape_h, shape_w = (ppf, pph, ppw, t_dim // 2), (ppf, pph, ppw, h_dim // 2), (ppf, pph, ppw, w_dim // 2)
    cos_half = mx.concatenate([mx.broadcast_to(ct[:, None, None, :], shape_t), mx.broadcast_to(ch[None, :, None, :], shape_h), mx.broadcast_to(cw[None, None, :, :], shape_w)], axis=-1)
    sin_half = mx.concatenate([mx.broadcast_to(st[:, None, None, :], shape_t), mx.broadcast_to(sh[None, :, None, :], shape_h), mx.broadcast_to(sw[None, None, :, :], shape_w)], axis=-1)
    return cos_half.reshape(1, ppf * pph * ppw, 1, head_dim // 2), sin_half.reshape(1, ppf * pph * ppw, 1, head_dim // 2)


def _apply_wan_rope(x: Any, cos_half: Any, sin_half: Any) -> Any:
    """Apply fused `@mx.compile` interleaved complex 3D RoPE (`[B, N, H, D]`)."""
    return _get_compiled_mlx_ops()["rope"](x, cos_half, sin_half)


def _timestep_sinusoidal(timesteps: Any, freq_dim: int) -> Any:
    import mlx.core as mx

    half = freq_dim // 2
    exp_arg = -math.log(10000.0) * mx.arange(half, dtype=mx.float32) / float(half)
    emb = timesteps.astype(mx.float32)[:, None] * mx.exp(exp_arg)[None, :]
    return mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)


class _KeyMap:
    """Resolves both HuggingFace Diffusers `WanTransformer3DModel` and ComfyUI `Wan2.1` keys."""

    def __init__(self, loader: MMapTensorLoader) -> None:
        self.loader = loader
        names = set(loader.get_tensor_names())
        self.is_comfy = "blocks.0.self_attn.q.weight" in names or "model.diffusion_model.blocks.0.self_attn.q.weight" in names
        self.pfx = "model.diffusion_model." if any(n.startswith("model.diffusion_model.") for n in names) else ""
        self.num_layers = 0
        while f"{self.pfx}blocks.{self.num_layers}.ffn.net.2.weight" in names or f"{self.pfx}blocks.{self.num_layers}.ffn.2.weight" in names:
            self.num_layers += 1

    def read(self, diff_key: str, comfy_key: Optional[str] = None, dtype: Optional[Any] = None) -> Tuple[Any, int]:
        k = f"{self.pfx}{comfy_key if (self.is_comfy and comfy_key) else diff_key}"
        nbytes = self.loader.get_tensor_byte_size(k)
        arr = self.loader.read_mlx_array(k)
        if dtype is not None and arr.dtype != dtype:
            arr = arr.astype(dtype)
        return arr, nbytes


def _load_fused_wan_block(
    km: _KeyMap,
    idx: int,
    compute_dt: Any,
    static_enc: Optional[Any] = None,
    num_heads: int = 12,
) -> Tuple[Dict[str, Any], int]:
    """Read and QKV-fuse one `Wan2.1` transformer block into Metal arrays."""
    import mlx.core as mx

    p = f"blocks.{idx}"
    total_b = 0

    def _r(dk: str, ck: str, dt: Any = compute_dt) -> Any:
        nonlocal total_b
        a, nb = km.read(f"{p}.{dk}", f"{p}.{ck}", dtype=dt)
        total_b += nb
        return a

    sst = _r("scale_shift_table", "modulation", mx.float32)
    sq_w, sq_b = _r("attn1.to_q.weight", "self_attn.q.weight"), _r("attn1.to_q.bias", "self_attn.q.bias")
    sk_w, sk_b = _r("attn1.to_k.weight", "self_attn.k.weight"), _r("attn1.to_k.bias", "self_attn.k.bias")
    sv_w, sv_b = _r("attn1.to_v.weight", "self_attn.v.weight"), _r("attn1.to_v.bias", "self_attn.v.bias")
    so_w, so_b = _r("attn1.to_out.0.weight", "self_attn.o.weight"), _r("attn1.to_out.0.bias", "self_attn.o.bias")
    snq = _r("attn1.norm_q.weight", "self_attn.norm_q.weight", mx.float32)
    snk = _r("attn1.norm_k.weight", "self_attn.norm_k.weight", mx.float32)

    cq_w, cq_b = _r("attn2.to_q.weight", "cross_attn.q.weight"), _r("attn2.to_q.bias", "cross_attn.q.bias")
    ck_w, ck_b = _r("attn2.to_k.weight", "cross_attn.k.weight"), _r("attn2.to_k.bias", "cross_attn.k.bias")
    cv_w, cv_b = _r("attn2.to_v.weight", "cross_attn.v.weight"), _r("attn2.to_v.bias", "cross_attn.v.bias")
    co_w, co_b = _r("attn2.to_out.0.weight", "cross_attn.o.weight"), _r("attn2.to_out.0.bias", "cross_attn.o.bias")
    cnq = _r("attn2.norm_q.weight", "cross_attn.norm_q.weight", mx.float32)
    cnk = _r("attn2.norm_k.weight", "cross_attn.norm_k.weight", mx.float32)

    n2_w = _r("norm2.weight", "norm3.weight", mx.float32)
    n2_b = _r("norm2.bias", "norm3.bias", mx.float32)

    f1_w, f1_b = _r("ffn.net.0.proj.weight", "ffn.0.weight"), _r("ffn.net.0.proj.bias", "ffn.0.bias")
    f2_w, f2_b = _r("ffn.net.2.weight", "ffn.2.weight"), _r("ffn.net.2.bias", "ffn.2.bias")

    blk: Dict[str, Any] = {
        "sst": sst,
        "s_qkv_wT": mx.concatenate([sq_w, sk_w, sv_w], axis=0).T,
        "s_qkv_b": mx.concatenate([sq_b, sk_b, sv_b], axis=0),
        "so_wT": so_w.T,
        "so_b": so_b,
        "snq": snq,
        "snk": snk,
        "cq_wT": cq_w.T,
        "cq_b": cq_b,
        "co_wT": co_w.T,
        "co_b": co_b,
        "cnq": cnq,
        "n2_w": n2_w,
        "n2_b": n2_b,
        "f1_wT": f1_w.T,
        "f1_b": f1_b,
        "f2_wT": f2_w.T,
        "f2_b": f2_b,
    }
    if static_enc is not None:
        B, S, D = static_enc.shape
        head_dim = D // num_heads
        ckv = static_enc @ mx.concatenate([ck_w, cv_w], axis=0).T + mx.concatenate([ck_b, cv_b], axis=0)
        ck, cv = mx.split(ckv, 2, axis=-1)
        blk["static_ck"] = _rms_norm_fp32(ck, cnk).reshape(B, S, num_heads, head_dim).transpose(0, 2, 1, 3)
        blk["static_cv"] = cv.reshape(B, S, num_heads, head_dim).transpose(0, 2, 1, 3)
    else:
        blk["c_kv_wT"] = mx.concatenate([ck_w, cv_w], axis=0).T
        blk["c_kv_b"] = mx.concatenate([ck_b, cv_b], axis=0)
        blk["cnk"] = cnk
    mx.eval(*blk.values())
    return blk, total_b


def _exec_wan_block(
    h: Any,
    enc_h: Any,
    temb_6: Any,
    cos_half: Any,
    sin_half: Any,
    blk: Dict[str, Any],
    num_heads: int,
) -> Any:
    """Execute one fused `Wan2.1` 3D spatiotemporal transformer block on Apple Metal."""
    import mlx.core as mx

    ops = _get_compiled_mlx_ops()
    dt = h.dtype
    B, N, D = h.shape
    head_dim = D // num_heads
    scale_attn = head_dim ** -0.5

    mods = blk["sst"] + temb_6  # [B, 6, D] in fp32
    shift_msa = mods[:, 0:1, :]
    scale_msa = mods[:, 1:2, :]
    gate_msa = mods[:, 2:3, :]
    c_shift = mods[:, 3:4, :]
    c_scale = mods[:, 4:5, :]
    c_gate = mods[:, 5:6, :]

    # 1. 3D Self-Attention + Fused `@mx.compile` RoPE & AdaLN
    n1 = ops["adaln"](h, scale_msa, shift_msa)
    qkv = n1 @ blk["s_qkv_wT"] + blk["s_qkv_b"]
    q, k, v = mx.split(qkv, 3, axis=-1)
    q = _rms_norm_fp32(q, blk["snq"]).reshape(B, N, num_heads, head_dim)
    k = _rms_norm_fp32(k, blk["snk"]).reshape(B, N, num_heads, head_dim)
    v = v.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
    q = ops["rope"](q, cos_half, sin_half).transpose(0, 2, 1, 3)
    k = ops["rope"](k, cos_half, sin_half).transpose(0, 2, 1, 3)
    sa = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale_attn).transpose(0, 2, 1, 3).reshape(B, N, D)
    sa = sa @ blk["so_wT"] + blk["so_b"]
    h = ops["gated_add"](h, sa, gate_msa)

    # 2. Text Cross-Attention (uses precomputed static `(ck, cv)` on resident blocks!)
    n2 = _layer_norm_fp32(h, blk["n2_w"], blk["n2_b"]).astype(dt)
    cq = _rms_norm_fp32(n2 @ blk["cq_wT"] + blk["cq_b"], blk["cnq"]).reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
    if "static_ck" in blk:
        ck, cv = blk["static_ck"], blk["static_cv"]
    else:
        ckv = enc_h @ blk["c_kv_wT"] + blk["c_kv_b"]
        ck, cv = mx.split(ckv, 2, axis=-1)
        S = enc_h.shape[1]
        ck = _rms_norm_fp32(ck, blk["cnk"]).reshape(B, S, num_heads, head_dim).transpose(0, 2, 1, 3)
        cv = cv.reshape(B, S, num_heads, head_dim).transpose(0, 2, 1, 3)
    ca = mx.fast.scaled_dot_product_attention(cq, ck, cv, scale=scale_attn).transpose(0, 2, 1, 3).reshape(B, N, D)
    ca = ca @ blk["co_wT"] + blk["co_b"]
    h = h + ca

    # 3. Feed-Forward (`@mx.compile` `gelu_tanh` + `adaln` + `gated_add`)
    n3 = ops["adaln"](h, c_scale, c_shift)
    ff = ops["gelu"](n3 @ blk["f1_wT"] + blk["f1_b"]) @ blk["f2_wT"] + blk["f2_b"]
    h = ops["gated_add"](h, ff, c_gate)
    return h


class MLXWanVideoStreamer:
    """Streams `Wan2.1` 3D Video DiT on Apple Silicon Metal (`monolithic`, `block_stream`, `hybrid_stream`)."""

    def __init__(
        self,
        safetensors_path: Union[str, Path],
        num_heads: int = 12,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        freq_dim: int = 256,
        mode: str = "hybrid_stream",
        memory_budget_mb: float = 1200.0,
        activation_reserve_mb: float = 350.0,
        cache_rel_l1_thresh: float = 0.0,
        cache_warmup_steps: int = 2,
        cache_max_consecutive_skips: int = 2,
    ) -> None:
        self.safetensors_path = Path(safetensors_path)
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.freq_dim = freq_dim
        self.mode = mode
        self.memory_budget_mb = memory_budget_mb
        self.activation_reserve_mb = activation_reserve_mb
        self.cache_rel_l1_thresh = cache_rel_l1_thresh
        self.cache_warmup_steps = cache_warmup_steps
        self.cache_max_consecutive_skips = cache_max_consecutive_skips

    def denoise_latents(
        self,
        initial_latents: Any,
        encoder_hidden_states: Any,
        timesteps: Sequence[float],
        sigmas: Sequence[float],
    ) -> Tuple[Any, MLXWanVideoMetrics]:
        """Run multi-step Euler flow-matching video denoising `[B, C, F, H, W]` on Apple Metal."""
        import time
        import mlx.core as mx
        import mlx.nn as nn

        _flush_metal()
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
        t_start = time.perf_counter()
        bytes_streamed = 0

        with MMapTensorLoader(self.safetensors_path) as loader:
            km = _KeyMap(loader)
            L = km.num_layers
            dt = mx.bfloat16

            def _rs(dk: str, ck: str, t_dt: Any = dt) -> Any:
                nonlocal bytes_streamed
                a, nb_s = km.read(dk, ck, t_dt)
                bytes_streamed += nb_s
                return a

            p_w, p_b = _rs("patch_embedding.weight", "patch_embedding.weight"), _rs("patch_embedding.bias", "patch_embedding.bias")
            t1_w = _rs("condition_embedder.time_embedder.linear_1.weight", "time_embedding.0.weight")
            t1_b = _rs("condition_embedder.time_embedder.linear_1.bias", "time_embedding.0.bias")
            t2_w = _rs("condition_embedder.time_embedder.linear_2.weight", "time_embedding.2.weight")
            t2_b = _rs("condition_embedder.time_embedder.linear_2.bias", "time_embedding.2.bias")
            tp_w = _rs("condition_embedder.time_proj.weight", "time_projection.1.weight")
            tp_b = _rs("condition_embedder.time_proj.bias", "time_projection.1.bias")
            tx1_w = _rs("condition_embedder.text_embedder.linear_1.weight", "text_embedding.0.weight")
            tx1_b = _rs("condition_embedder.text_embedder.linear_1.bias", "text_embedding.0.bias")
            tx2_w = _rs("condition_embedder.text_embedder.linear_2.weight", "text_embedding.2.weight")
            tx2_b = _rs("condition_embedder.text_embedder.linear_2.bias", "text_embedding.2.bias")
            out_sst = _rs("scale_shift_table", "head.modulation", mx.float32)
            out_w, out_b = _rs("proj_out.weight", "head.head.weight"), _rs("proj_out.bias", "head.head.bias")

            dim = int(p_w.shape[0])
            p_t, p_h, p_w_sz = self.patch_size
            patch_wT = p_w.reshape(dim, -1).T

            lat = initial_latents.astype(mx.float32)
            B, C, F_len, H_len, W_len = lat.shape
            ppf, pph, ppw = F_len // p_t, H_len // p_h, W_len // p_w_sz
            cos_half, sin_half = _build_wan_3d_rope(ppf, pph, ppw, dim // self.num_heads)
            enc_proj = _gelu_tanh(encoder_hidden_states.astype(dt) @ tx1_w.T + tx1_b) @ tx2_w.T + tx2_b
            mx.eval(cos_half, sin_half, enc_proj)

            resident: Dict[int, Dict[str, Any]] = {}
            if self.mode == "monolithic":
                target_res = L
            elif self.mode == "block_stream":
                target_res = 0
            else:
                resident[0], blk0_bytes = _load_fused_wan_block(km, 0, dt, enc_proj, self.num_heads)
                bytes_streamed += blk0_bytes
                usable_b = max(0.0, self.memory_budget_mb - self.activation_reserve_mb) * 1024.0 * 1024.0
                target_res = min(L, max(1, int(usable_b // max(1, blk0_bytes))))

            for bi in range(len(resident), target_res):
                resident[bi], nb_blk = _load_fused_wan_block(km, bi, dt, enc_proj, self.num_heads)
                bytes_streamed += nb_blk

            active_res_mb = float(mx.get_active_memory()) / (1024.0 * 1024.0) if hasattr(mx, "get_active_memory") else 0.0

            prev_d0: Optional[Any] = None
            cached_res: Optional[Any] = None
            consec_skips = 0
            skipped_steps = 0

            for step_i, t_val in enumerate(timesteps):
                # Patchify `[B, C, F, H, W]` -> `[B, ppf*pph*ppw, C*p_t*p_h*p_w]` -> `[B, N, dim]`
                x_in = lat.astype(dt).reshape(B, C, ppf, p_t, pph, p_h, ppw, p_w_sz)
                x_patches = x_in.transpose(0, 2, 4, 6, 1, 3, 5, 7).reshape(B, ppf * pph * ppw, -1)
                h = x_patches @ patch_wT + p_b

                t_sin = _timestep_sinusoidal(mx.array([float(t_val)], dtype=mx.float32), self.freq_dim).astype(dt)
                temb = nn.silu(t_sin @ t1_w.T + t1_b) @ t2_w.T + t2_b
                temb_6 = (nn.silu(temb) @ tp_w.T + tp_b).reshape(B, 6, dim).astype(mx.float32)
                mx.eval(h, temb, temb_6)

                # Execute block 0 (probe block for First-Block Cache!)
                if 0 in resident:
                    b0 = resident[0]
                else:
                    b0, nb0 = _load_fused_wan_block(km, 0, dt)
                    bytes_streamed += nb0
                h_after_0 = _exec_wan_block(h, enc_proj, temb_6, cos_half, sin_half, b0, self.num_heads)
                if 0 not in resident:
                    del b0
                mx.eval(h_after_0)

                d0 = (h_after_0.astype(mx.float32) - h.astype(mx.float32))
                can_skip = (
                    self.cache_rel_l1_thresh > 0.0
                    and step_i >= self.cache_warmup_steps
                    and step_i < len(timesteps) - 1
                    and prev_d0 is not None
                    and cached_res is not None
                    and consec_skips < self.cache_max_consecutive_skips
                )
                did_skip = False
                if can_skip:
                    rel_l1 = float((mx.mean(mx.abs(d0 - prev_d0)) / (mx.mean(mx.abs(prev_d0)) + 1e-6)).item())
                    if rel_l1 < self.cache_rel_l1_thresh:
                        h = (h_after_0.astype(mx.float32) + cached_res).astype(dt)
                        mx.eval(h)
                        consec_skips += 1
                        skipped_steps += 1
                        did_skip = True

                if not did_skip:
                    prev_d0 = d0
                    consec_skips = 0
                    h_cur = h_after_0
                    for bi in range(1, L):
                        if bi in resident:
                            h_cur = _exec_wan_block(h_cur, enc_proj, temb_6, cos_half, sin_half, resident[bi], self.num_heads)
                        else:
                            b_tmp, nb_i = _load_fused_wan_block(km, bi, dt)
                            bytes_streamed += nb_i
                            h_cur = _exec_wan_block(h_cur, enc_proj, temb_6, cos_half, sin_half, b_tmp, self.num_heads)
                            del b_tmp
                        mx.eval(h_cur)
                    if self.cache_rel_l1_thresh > 0.0:
                        cached_res = h_cur.astype(mx.float32) - h_after_0.astype(mx.float32)
                        mx.eval(cached_res)
                    h = h_cur

                out_mod = out_sst + temb.astype(mx.float32)[:, None, :]
                h_out = _get_compiled_mlx_ops()["adaln"](h, out_mod[:, 1:2, :], out_mod[:, 0:1, :])
                v_patches = h_out @ out_w.T + out_b
                v_pred = v_patches.reshape(B, ppf, pph, ppw, p_t, p_h, p_w_sz, C).transpose(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, C, F_len, H_len, W_len).astype(mx.float32)
                lat = lat + (float(sigmas[step_i + 1]) - float(sigmas[step_i])) * v_pred
                mx.eval(lat)

            wall = time.perf_counter() - t_start
            peak_mb = float(mx.get_peak_memory()) / (1024.0 * 1024.0) if hasattr(mx, "get_peak_memory") else 0.0
            del resident

        _flush_metal()
        return lat, MLXWanVideoMetrics(
            mode=self.mode,
            steps=len(timesteps),
            num_layers=L,
            resident_blocks=target_res,
            skipped_steps=skipped_steps,
            wall_sec=round(wall, 3),
            sec_per_step=round(wall / max(1, len(timesteps)), 3),
            peak_metal_mb=round(peak_mb, 1),
            active_resident_mb=round(active_res_mb, 1),
            streamed_mb=round(bytes_streamed / (1024.0 * 1024.0), 1),
        )

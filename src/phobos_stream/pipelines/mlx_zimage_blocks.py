"""Block-level MLX weight mapping, fused kernels, and per-block mmap execution for Z-Image."""

from typing import Any, List, Optional, Tuple, Union

from phobos_stream.storage.mmap_loader import MMapTensorLoader


def apply_mlx_rope_3d(x: Any, freqs_cis: Any) -> Any:
    """Apply 3D Rotary Position Embedding across `(B, Seq, Heads, HeadDim)` in native dtype."""
    import mlx.core as mx

    dtype = x.dtype
    b, seq_len, n_heads, head_dim = x.shape
    x_pairs = x.reshape(b, seq_len, n_heads, head_dim // 2, 2)
    fc = freqs_cis[None, :, None, :, :]
    out_r = x_pairs[..., 0] * fc[..., 0] - x_pairs[..., 1] * fc[..., 1]
    out_i = x_pairs[..., 0] * fc[..., 1] + x_pairs[..., 1] * fc[..., 0]
    return mx.stack([out_r, out_i], axis=-1).reshape(b, seq_len, n_heads, head_dim).astype(dtype)


def map_zimage_transformer(
    dit_path: Union[str, Any],
    quantize_bits: Optional[int],
    fuse_blocks: bool,
) -> Tuple[Any, Any, Any, Any]:
    """Load `z_image_turbo_bf16.safetensors` into MLX `ZImageTransformer` with optional QKV/W13 fusion."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_unflatten
    from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer

    raw_weights = mx.load(str(dit_path))
    mapped: List[Tuple[str, Any]] = []
    fused_qkv = {}
    w1_map = {}
    w3_map = {}

    for k, v in raw_weights.items():
        vb = v.astype(mx.bfloat16)
        if k == "t_embedder.mlp.0.weight":
            mapped.append(("t_embedder.linear1.weight", vb))
        elif k == "t_embedder.mlp.0.bias":
            mapped.append(("t_embedder.linear1.bias", vb))
        elif k == "t_embedder.mlp.2.weight":
            mapped.append(("t_embedder.linear2.weight", vb))
        elif k == "t_embedder.mlp.2.bias":
            mapped.append(("t_embedder.linear2.bias", vb))
        elif k in ("x_embedder.weight", "x_embedder.bias"):
            mapped.append((k.replace("x_embedder", "all_x_embedder.2-1"), vb))
        elif k in ("final_layer.linear.weight", "final_layer.linear.bias"):
            mapped.append((k.replace("final_layer", "all_final_layer.2-1"), vb))
        elif k in ("final_layer.adaLN_modulation.1.weight", "final_layer.adaLN_modulation.1.bias"):
            mapped.append((k.replace("final_layer.adaLN_modulation.1", "all_final_layer.2-1.adaLN_modulation.0"), vb))
        elif k.endswith(".attention.qkv.weight"):
            pfx = k[: -len(".attention.qkv.weight")]
            if fuse_blocks and not quantize_bits:
                fused_qkv[pfx] = vb
            else:
                dim = vb.shape[1]
                mapped.extend([
                    (f"{pfx}.attention.to_q.weight", vb[:dim]),
                    (f"{pfx}.attention.to_k.weight", vb[dim : 2 * dim]),
                    (f"{pfx}.attention.to_v.weight", vb[2 * dim : 3 * dim]),
                ])
        elif k.endswith(".feed_forward.w1.weight") and fuse_blocks and not quantize_bits:
            w1_map[k[: -len(".feed_forward.w1.weight")]] = vb
        elif k.endswith(".feed_forward.w3.weight") and fuse_blocks and not quantize_bits:
            w3_map[k[: -len(".feed_forward.w3.weight")]] = vb
        elif k.endswith(".attention.out.weight"):
            mapped.append((k.replace(".attention.out.weight", ".attention.to_out.0.weight"), vb))
        elif k.endswith(".attention.q_norm.weight"):
            mapped.append((k.replace(".attention.q_norm.weight", ".attention.norm_q.weight"), vb))
        elif k.endswith(".attention.k_norm.weight"):
            mapped.append((k.replace(".attention.k_norm.weight", ".attention.norm_k.weight"), vb))
        else:
            mapped.append((k, vb))
    del raw_weights

    tr = ZImageTransformer()
    tr.set_float32(False)
    tr.update(tree_unflatten(mapped))
    del mapped

    if quantize_bits in (4, 8):
        nn.quantize(tr, group_size=64, bits=quantize_bits)
        mx.eval(tr.parameters())
        return tr, None, None, None
    if not fuse_blocks:
        mx.eval(tr.parameters())
        return tr, None, None, None

    def _pack(blk_mod: Any, pfx: str, has_adaln: bool) -> Tuple[Any, Any, Any, bool]:
        qkv_t = mx.contiguous(fused_qkv.pop(pfx).T)
        w13_t = mx.contiguous(mx.concatenate([w1_map.pop(pfx), w3_map.pop(pfx)], axis=0).T)
        blk_mod.attention.to_q = blk_mod.attention.to_k = blk_mod.attention.to_v = None
        blk_mod.feed_forward.w1 = blk_mod.feed_forward.w3 = None
        return (blk_mod, qkv_t, w13_t, has_adaln)

    nr = [_pack(tr.noise_refiner[i], f"noise_refiner.{i}", True) for i in range(2)]
    cr = [_pack(tr.context_refiner[i], f"context_refiner.{i}", False) for i in range(2)]
    ly = [_pack(tr.layers[i], f"layers.{i}", True) for i in range(30)]
    mx.eval(tr.parameters(), *[b[1] for b in nr + cr + ly], *[b[2] for b in nr + cr + ly])
    return tr, nr, cr, ly


def exec_fused_zimage_block(
    blk_tuple: Tuple[Any, Any, Any, bool],
    x: Any,
    freqs: Any,
    t_emb: Optional[Any],
) -> Any:
    """Execute a pre-fused QKV + W13 Z-Image block using unmasked Metal FlashAttention."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.core.fast import scaled_dot_product_attention

    mod, qkv_t, w13_t, has_adaln = blk_tuple
    b, seq_len, _ = x.shape
    n_heads, head_dim, dim = mod.attention.n_heads, mod.attention.head_dim, mod.attention.dim
    if has_adaln and t_emb is not None:
        scale_msa, gate_msa, scale_mlp, gate_mlp = mx.split(
            mx.expand_dims(mod.adaLN_modulation[0](t_emb), axis=1), 4, axis=2
        )
        norm1_x = mod.attention_norm1(x) * (1.0 + scale_msa)
    else:
        norm1_x = mod.attention_norm1(x)

    q, k, v = mx.split(mx.matmul(norm1_x, qkv_t), 3, axis=-1)
    q = apply_mlx_rope_3d(mod.attention.norm_q(q.reshape(b, seq_len, n_heads, head_dim)), freqs)
    k = apply_mlx_rope_3d(mod.attention.norm_k(k.reshape(b, seq_len, n_heads, head_dim)), freqs)
    v = (v * 0.25).reshape(b, seq_len, n_heads, head_dim)
    attn = scaled_dot_product_attention(
        mx.transpose(q, (0, 2, 1, 3)),
        mx.transpose(k, (0, 2, 1, 3)),
        mx.transpose(v, (0, 2, 1, 3)),
        scale=mod.attention.scale,
        mask=None,
    )
    attn = mod.attention.to_out[0](mx.transpose(attn, (0, 2, 1, 3)).reshape(b, seq_len, dim)) * 4.0

    if has_adaln and t_emb is not None:
        x = x + mx.tanh(gate_msa) * mod.attention_norm2(attn)
        gate, up = mx.split(mx.matmul(mod.ffn_norm1(x) * (1.0 + scale_mlp), w13_t), 2, axis=-1)
        ffn = mod.feed_forward.w2(nn.silu(gate) * up)
        return x + mx.tanh(gate_mlp) * mod.ffn_norm2(ffn)
    x = x + mod.attention_norm2(attn)
    gate, up = mx.split(mx.matmul(mod.ffn_norm1(x), w13_t), 2, axis=-1)
    ffn = mod.feed_forward.w2(nn.silu(gate) * up)
    return x + mod.ffn_norm2(ffn)


def stream_single_zimage_block_mmap(
    loader: MMapTensorLoader,
    prefix: str,
    x: Any,
    freqs_cis: Any,
    t_emb: Optional[Any],
    has_adaln: bool,
) -> Tuple[Any, int]:
    """Stream 1 Z-Image DiT block via `MMapTensorLoader`, execute on Metal, evaluate, and evict."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.core.fast import rms_norm, scaled_dot_product_attention

    b, seq_len, dim = x.shape
    n_heads, head_dim = 30, 128
    bytes_read = 0

    def rd(suffix: str) -> Any:
        nonlocal bytes_read
        key = f"{prefix}.{suffix}"
        bytes_read += loader.get_tensor_byte_size(key)
        return loader.read_mlx_array(key, target_dtype=mx.bfloat16)

    norm1_w = rd("attention_norm1.weight")
    if has_adaln and t_emb is not None:
        ada_w, ada_b = rd("adaLN_modulation.0.weight"), rd("adaLN_modulation.0.bias")
        scale_msa, gate_msa, scale_mlp, gate_mlp = mx.split(
            mx.expand_dims(mx.matmul(t_emb, ada_w.T) + ada_b, axis=1), 4, axis=2
        )
        del ada_w, ada_b
        norm1_x = rms_norm(x, norm1_w, 1e-5) * (1.0 + scale_msa)
    else:
        norm1_x = rms_norm(x, norm1_w, 1e-5)
    del norm1_w

    qkv_w = rd("attention.qkv.weight")
    q, k, v = mx.split(mx.matmul(norm1_x, qkv_w.T), 3, axis=-1)
    del qkv_w, norm1_x

    qn_w, kn_w = rd("attention.q_norm.weight"), rd("attention.k_norm.weight")
    q = apply_mlx_rope_3d(rms_norm(q.reshape(b, seq_len, n_heads, head_dim), qn_w, 1e-5), freqs_cis)
    k = apply_mlx_rope_3d(rms_norm(k.reshape(b, seq_len, n_heads, head_dim), kn_w, 1e-5), freqs_cis)
    del qn_w, kn_w
    v = (v * 0.25).reshape(b, seq_len, n_heads, head_dim)

    attn = scaled_dot_product_attention(
        mx.transpose(q, (0, 2, 1, 3)),
        mx.transpose(k, (0, 2, 1, 3)),
        mx.transpose(v, (0, 2, 1, 3)),
        scale=head_dim ** -0.5,
        mask=None,
    )
    del q, k, v
    out_w = rd("attention.out.weight")
    attn = mx.matmul(mx.transpose(attn, (0, 2, 1, 3)).reshape(b, seq_len, dim), out_w.T) * 4.0
    del out_w

    norm2_w, ffn1_w, ffn2_w = rd("attention_norm2.weight"), rd("ffn_norm1.weight"), rd("ffn_norm2.weight")
    if has_adaln and t_emb is not None:
        x = x + mx.tanh(gate_msa) * rms_norm(attn, norm2_w, 1e-5)
        ffn_in = rms_norm(x, ffn1_w, 1e-5) * (1.0 + scale_mlp)
    else:
        x = x + rms_norm(attn, norm2_w, 1e-5)
        ffn_in = rms_norm(x, ffn1_w, 1e-5)
    del attn, norm2_w, ffn1_w

    w1, w3 = rd("feed_forward.w1.weight"), rd("feed_forward.w3.weight")
    hidden = nn.silu(mx.matmul(ffn_in, w1.T)) * mx.matmul(ffn_in, w3.T)
    del w1, w3, ffn_in
    w2 = rd("feed_forward.w2.weight")
    ffn_out = mx.matmul(hidden, w2.T)
    del w2, hidden

    x = (
        x + mx.tanh(gate_mlp) * rms_norm(ffn_out, ffn2_w, 1e-5)
        if (has_adaln and t_emb is not None)
        else x + rms_norm(ffn_out, ffn2_w, 1e-5)
    )
    del ffn_out, ffn2_w

    mx.eval(x)
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    return x, bytes_read

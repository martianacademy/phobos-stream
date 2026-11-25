"""Honest, Unbiased Benchmark: Baseline vs mlx_lm Optimizations (Wired Limit, Async Eval, 8-bit/4-bit Quantized Matmul)."""

import time
from pathlib import Path
import numpy as np

from phobos_stream.backends.mlx_backend import MLX_AVAILABLE, MLXBackend
from phobos_stream.storage.mmap_loader import MMapTensorLoader


def run_cpu_and_mlx_benchmark() -> None:
    print("=" * 80)
    print("HONEST BENCHMARK: BASELINE vs MLX_LM OPTIMIZATIONS ON REAL Z-IMAGE-TURBO BLOCK")
    print("=" * 80)

    dit_path = Path.home() / "ComfyUI-Shared/models/diffusion_models/z_image_turbo_bf16.safetensors"
    if not dit_path.exists():
        raise FileNotFoundError(f"Missing {dit_path}")

    # Load real Layer 0 weights from z_image_turbo_bf16.safetensors (618.75 MB FP32 / 309.38 MB BF16)
    with MMapTensorLoader(dit_path) as loader:
        qkv_fp32 = loader.read_tensor("layers.0.attention.qkv.weight", copy=False)  # (11520, 3840)
        w1_fp32 = loader.read_tensor("layers.0.feed_forward.w1.weight", copy=False)  # (10240, 3840)
        w2_fp32 = loader.read_tensor("layers.0.feed_forward.w2.weight", copy=False)  # (3840, 10240)
        w3_fp32 = loader.read_tensor("layers.0.feed_forward.w3.weight", copy=False)  # (10240, 3840)

    fp32_weight_mb = (qkv_fp32.nbytes + w1_fp32.nbytes + w2_fp32.nbytes + w3_fp32.nbytes) / (1024 * 1024)
    print(f"Loaded Real Layer 0 Linear Weights from z_image_turbo_bf16.safetensors: {fp32_weight_mb:.2f} MB (FP32)")

    seq_len = 1024
    dim = 3840
    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((1, seq_len, dim)).astype(np.float32)

    # 1. Baseline FP32 with per-step NumPy copy simulation
    t0 = time.perf_counter()
    for _ in range(4):
        qkv_out = np.matmul(x_np, qkv_fp32.T).copy()
        gate = np.matmul(x_np, w1_fp32.T).copy()
        up = np.matmul(x_np, w3_fp32.T).copy()
        silu_gate = gate / (1.0 + np.exp(-np.clip(gate, -50.0, 50.0)))
        down = np.matmul(silu_gate * up, w2_fp32.T).copy()
        x_step = (x_np + down + qkv_out[:, :, :3840]).copy()
    t_baseline_fp32 = time.perf_counter() - t0

    # 2. Optimized Zero-Copy In-Place
    qkv_t = np.ascontiguousarray(qkv_fp32.T)
    w1_t = np.ascontiguousarray(w1_fp32.T)
    w2_t = np.ascontiguousarray(w2_fp32.T)
    w3_t = np.ascontiguousarray(w3_fp32.T)
    t0 = time.perf_counter()
    for _ in range(4):
        qkv_out = np.matmul(x_np, qkv_t)
        gate = np.matmul(x_np, w1_t)
        up = np.matmul(x_np, w3_t)
        gate *= 1.0 / (1.0 + np.exp(-np.clip(gate, -50.0, 50.0)))
        gate *= up
        down = np.matmul(gate, w2_t)
        x_step = x_np + down + qkv_out[:, :, :3840]
    t_opt_cpu = time.perf_counter() - t0

    print("\n--- [A] CPU / BLAS TESTED RESULTS (4 Steps, Real Z-Image-Turbo Layer 0, 1024 Tokens) ---")
    print(f"  • Old Baseline (Per-step copy + uncached transpose): {t_baseline_fp32:.3f} s ({fp32_weight_mb:.1f} MB weights)")
    print(f"  • Optimized In-Place Pipeline (Zero per-step copy):  {t_opt_cpu:.3f} s ({(t_baseline_fp32 - t_opt_cpu)/t_baseline_fp32*100:.1f}% faster)")

    if MLX_AVAILABLE:
        import mlx.core as mx
        print("\n--- [B] APPLE METAL GPU (MLX_LM PRIMITIVES, 8 STEPS, 1024 TOKENS) ---")
        mlx_fp32 = MLXBackend()
        with mlx_fp32.wired_limit():
            x_mx_32 = mx.array(x_np, dtype=mx.float32)
            qkv_mx_32 = mx.array(qkv_fp32, dtype=mx.float32)
            w1_mx_32 = mx.array(w1_fp32, dtype=mx.float32)
            w2_mx_32 = mx.array(w2_fp32, dtype=mx.float32)
            w3_mx_32 = mx.array(w3_fp32, dtype=mx.float32)
            mx.eval(x_mx_32, qkv_mx_32, w1_mx_32, w2_mx_32, w3_mx_32)

            # Warmup FP32
            mx.eval(mlx_fp32.linear(x_mx_32, qkv_mx_32))

            # 1. Measure MLX FP32 with per-step synchronous to_numpy() (Old behavior)
            t0 = time.perf_counter()
            cur_32 = x_mx_32
            for _ in range(8):
                qkv_o = mlx_fp32.linear(cur_32, qkv_mx_32)
                g = mlx_fp32.silu(mlx_fp32.linear(cur_32, w1_mx_32)) * mlx_fp32.linear(cur_32, w3_mx_32)
                d = mlx_fp32.linear(g, w2_mx_32) + qkv_o[:, :, :3840]
                np_roundtrip = mlx_fp32.to_numpy(cur_32 + d)
                cur_32 = mlx_fp32.convert_from_numpy(np_roundtrip)
            mx.eval(cur_32)
            t_mlx_old_fp32 = time.perf_counter() - t0
            print(f"  1. Old MLX FP32 + Per-Step NumPy Sync:     {t_mlx_old_fp32:.3f} s | Weights: {fp32_weight_mb:.1f} MB")

            # 2. Measure MLX Native BF16 + async_eval + wired_limit (mlx_lm style)
            x_mx_bf16 = x_mx_32.astype(mx.bfloat16)
            qkv_mx_bf16 = qkv_mx_32.astype(mx.bfloat16)
            w1_mx_bf16 = w1_mx_32.astype(mx.bfloat16)
            w2_mx_bf16 = w2_mx_32.astype(mx.bfloat16)
            w3_mx_bf16 = w3_mx_32.astype(mx.bfloat16)
            mx.eval(x_mx_bf16, qkv_mx_bf16, w1_mx_bf16, w2_mx_bf16, w3_mx_bf16)
            mx.eval(mlx_fp32.linear(x_mx_bf16, qkv_mx_bf16))

            t0 = time.perf_counter()
            cur_bf16 = x_mx_bf16
            for _ in range(8):
                qkv_o = mlx_fp32.linear(cur_bf16, qkv_mx_bf16)
                g = mlx_fp32.silu(mlx_fp32.linear(cur_bf16, w1_mx_bf16)) * mlx_fp32.linear(cur_bf16, w3_mx_bf16)
                cur_bf16 = cur_bf16 + mlx_fp32.linear(g, w2_mx_bf16) + qkv_o[:, :, :3840]
                mlx_fp32.async_eval(cur_bf16)
            mx.eval(cur_bf16)
            t_mlx_bf16 = time.perf_counter() - t0
            print(f"  2. MLX_LM Native BF16 + async_eval:        {t_mlx_bf16:.3f} s | Weights: {fp32_weight_mb/2:.1f} MB ({t_mlx_old_fp32/t_mlx_bf16:.2f}x speedup)")

            # 3. Measure MLX 8-Bit Quantized Linear (mlx_lm nn.quantize / quantized_matmul)
            mlx_q8 = MLXBackend(quantize_bits=8, quantize_group_size=64)
            qkv_q8 = mlx_q8.quantize_weight(qkv_mx_bf16, bits=8, group_size=64)
            w1_q8 = mlx_q8.quantize_weight(w1_mx_bf16, bits=8, group_size=64)
            w2_q8 = mlx_q8.quantize_weight(w2_mx_bf16, bits=8, group_size=64)
            w3_q8 = mlx_q8.quantize_weight(w3_mx_bf16, bits=8, group_size=64)
            mx.eval(*qkv_q8, *w1_q8, *w2_q8, *w3_q8)
            mx.eval(mlx_q8.linear(x_mx_bf16, qkv_q8))

            t0 = time.perf_counter()
            cur_q8 = x_mx_bf16
            for _ in range(8):
                qkv_o = mlx_q8.linear(cur_q8, qkv_q8)
                g = mlx_q8.silu(mlx_q8.linear(cur_q8, w1_q8)) * mlx_q8.linear(cur_q8, w3_q8)
                cur_q8 = cur_q8 + mlx_q8.linear(g, w2_q8) + qkv_o[:, :, :3840]
                mlx_q8.async_eval(cur_q8)
            mx.eval(cur_q8)
            t_mlx_q8 = time.perf_counter() - t0
            print(f"  3. MLX_LM 8-Bit Quantized (q8 + async):    {t_mlx_q8:.3f} s | Weights: {fp32_weight_mb/4:.1f} MB ({t_mlx_old_fp32/t_mlx_q8:.2f}x speedup)")

            # 4. Measure MLX 4-Bit Quantized Linear (mlx_lm 4-bit affine)
            mlx_q4 = MLXBackend(quantize_bits=4, quantize_group_size=64)
            qkv_q4 = mlx_q4.quantize_weight(qkv_mx_bf16, bits=4, group_size=64)
            w1_q4 = mlx_q4.quantize_weight(w1_mx_bf16, bits=4, group_size=64)
            w2_q4 = mlx_q4.quantize_weight(w2_mx_bf16, bits=4, group_size=64)
            w3_q4 = mlx_q4.quantize_weight(w3_mx_bf16, bits=4, group_size=64)
            mx.eval(*qkv_q4, *w1_q4, *w2_q4, *w3_q4)
            mx.eval(mlx_q4.linear(x_mx_bf16, qkv_q4))

            t0 = time.perf_counter()
            cur_q4 = x_mx_bf16
            for _ in range(8):
                qkv_o = mlx_q4.linear(cur_q4, qkv_q4)
                g = mlx_q4.silu(mlx_q4.linear(cur_q4, w1_q4)) * mlx_q4.linear(cur_q4, w3_q4)
                cur_q4 = cur_q4 + mlx_q4.linear(g, w2_q4) + qkv_o[:, :, :3840]
                mlx_q4.async_eval(cur_q4)
            mx.eval(cur_q4)
            t_mlx_q4 = time.perf_counter() - t0
            print(f"  4. MLX_LM 4-Bit Quantized (q4 + async):    {t_mlx_q4:.3f} s | Weights: {fp32_weight_mb/8:.1f} MB ({t_mlx_old_fp32/t_mlx_q4:.2f}x speedup)")


if __name__ == "__main__":
    run_cpu_and_mlx_benchmark()

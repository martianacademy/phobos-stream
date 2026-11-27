/*
 * PhobosStream Native CUDA C++ Single-Pass GGUF Block Unpacker (`sm_60+`)
 * =======================================================================
 * Unpacks compressed `uint8` GGUF tensors (`Q8_0`, `Q4_0`, `Q4_1`, `Q4_K`, `Q5_K`, `Q6_K`)
 * directly into `__nv_bfloat16` (`out_dtype_code == 0`) or `__half` (`out_dtype_code == 1`)
 * on a caller-provided `cudaStream_t` (`_prefetch_stream` or `compute_stream`) in a single
 * warp-coalesced pass with zero intermediate HBM workspace allocations.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ float load_fp16_unaligned(const uint8_t* p) {
    uint16_t bits = static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
    __half h = *reinterpret_cast<const __half*>(&bits);
    return __half2float(h);
}

template <typename OutT>
__device__ __forceinline__ void store_scalar(OutT* out, int64_t idx, float v);

template <>
__device__ __forceinline__ void store_scalar<__nv_bfloat16>(__nv_bfloat16* out, int64_t idx, float v) {
    out[idx] = __float2bfloat16(v);
}

template <>
__device__ __forceinline__ void store_scalar<__half>(__half* out, int64_t idx, float v) {
    out[idx] = __float2half(v);
}

template <>
__device__ __forceinline__ void store_scalar<float>(float* out, int64_t idx, float v) {
    out[idx] = v;
}

template <typename OutT>
__global__ void unpack_q8_0_kernel(const uint8_t* __restrict__ raw, OutT* __restrict__ out, int64_t n_elems) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= n_elems) return;
    int64_t blk_idx = tid >> 5;
    int lane = static_cast<int>(tid & 31);
    const uint8_t* blk = raw + blk_idx * 34;
    float d = load_fp16_unaligned(blk);
    int8_t q = static_cast<int8_t>(blk[2 + lane]);
    store_scalar<OutT>(out, tid, d * static_cast<float>(q));
}

template <typename OutT>
__global__ void unpack_q4_0_kernel(const uint8_t* __restrict__ raw, OutT* __restrict__ out, int64_t n_elems) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= n_elems) return;
    int64_t blk_idx = tid >> 5;
    int lane = static_cast<int>(tid & 31);
    const uint8_t* blk = raw + blk_idx * 18;
    float d = load_fp16_unaligned(blk);
    uint8_t byte_val = blk[2 + (lane & 15)];
    int q = ((lane < 16) ? (byte_val & 0x0F) : (byte_val >> 4)) - 8;
    store_scalar<OutT>(out, tid, d * static_cast<float>(q));
}

template <typename OutT>
__global__ void unpack_q4_1_kernel(const uint8_t* __restrict__ raw, OutT* __restrict__ out, int64_t n_elems) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= n_elems) return;
    int64_t blk_idx = tid >> 5;
    int lane = static_cast<int>(tid & 31);
    const uint8_t* blk = raw + blk_idx * 20;
    float d = load_fp16_unaligned(blk);
    float m = load_fp16_unaligned(blk + 2);
    uint8_t byte_val = blk[4 + (lane & 15)];
    int q = (lane < 16) ? (byte_val & 0x0F) : (byte_val >> 4);
    store_scalar<OutT>(out, tid, d * static_cast<float>(q) + m);
}

__device__ __forceinline__ void decode_k4_scale_min(const uint8_t* sc, int sub, uint8_t& sc_u, uint8_t& mn_u) {
    if (sub < 4) {
        sc_u = sc[sub] & 63;
        mn_u = sc[sub + 4] & 63;
    } else {
        sc_u = (sc[sub + 4] & 0x0F) | ((sc[sub - 4] >> 6) << 4);
        mn_u = (sc[sub + 4] >> 4) | ((sc[sub] >> 6) << 4);
    }
}

template <typename OutT>
__global__ void unpack_q4_k_kernel(const uint8_t* __restrict__ raw, OutT* __restrict__ out, int64_t n_elems) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= n_elems) return;
    int64_t sup_idx = tid >> 8;
    int in_blk = static_cast<int>(tid & 255);
    const uint8_t* blk = raw + sup_idx * 144;

    float d = load_fp16_unaligned(blk);
    float dmin = load_fp16_unaligned(blk + 2);
    int pair = in_blk >> 6;
    int half_idx = (in_blk >> 5) & 1;
    int lane = in_blk & 31;
    int sub = (pair << 1) | half_idx;

    uint8_t sc_u, mn_u;
    decode_k4_scale_min(blk + 4, sub, sc_u, mn_u);
    uint8_t qb = blk[16 + pair * 32 + lane];
    uint8_t q = half_idx ? (qb >> 4) : (qb & 0x0F);
    float val = (d * static_cast<float>(sc_u)) * static_cast<float>(q) - (dmin * static_cast<float>(mn_u));
    store_scalar<OutT>(out, tid, val);
}

template <typename OutT>
__global__ void unpack_q5_k_kernel(const uint8_t* __restrict__ raw, OutT* __restrict__ out, int64_t n_elems) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= n_elems) return;
    int64_t sup_idx = tid >> 8;
    int in_blk = static_cast<int>(tid & 255);
    const uint8_t* blk = raw + sup_idx * 176;

    float d = load_fp16_unaligned(blk);
    float dmin = load_fp16_unaligned(blk + 2);
    int pair = in_blk >> 6;
    int half_idx = (in_blk >> 5) & 1;
    int lane = in_blk & 31;
    int sub = (pair << 1) | half_idx;

    uint8_t sc_u, mn_u;
    decode_k4_scale_min(blk + 4, sub, sc_u, mn_u);
    uint8_t qb = blk[48 + pair * 32 + lane];
    uint8_t q4 = half_idx ? (qb >> 4) : (qb & 0x0F);
    uint8_t hb = ((blk[16 + lane] >> sub) & 1) << 4;
    float val = (d * static_cast<float>(sc_u)) * static_cast<float>(q4 | hb) - (dmin * static_cast<float>(mn_u));
    store_scalar<OutT>(out, tid, val);
}

template <typename OutT>
__global__ void unpack_q6_k_kernel(const uint8_t* __restrict__ raw, OutT* __restrict__ out, int64_t n_elems) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= n_elems) return;
    int64_t sup_idx = tid >> 8;
    int in_blk = static_cast<int>(tid & 255);
    const uint8_t* blk = raw + sup_idx * 210;

    float d = load_fp16_unaligned(blk + 208);
    int n = in_blk >> 7;
    int rem = in_blk & 127;
    int quad = rem >> 5;
    int l = rem & 31;
    int ql_sub = quad & 1;
    int is_hi4 = quad >> 1;

    uint8_t ql_byte = blk[n * 64 + ql_sub * 32 + l];
    uint8_t low4 = is_hi4 ? (ql_byte >> 4) : (ql_byte & 0x0F);
    uint8_t hi2 = ((blk[128 + n * 32 + l] >> (quad * 2)) & 3) << 4;
    int q6 = static_cast<int>(low4 | hi2) - 32;
    int8_t sc = static_cast<int8_t>(blk[192 + n * 8 + quad * 2 + (l >> 4)]);
    store_scalar<OutT>(out, tid, d * static_cast<float>(sc) * static_cast<float>(q6));
}

template <typename OutT>
int launch_typed_unpack(int ggml_type, const uint8_t* raw, OutT* out, int64_t n_elems, cudaStream_t stream) {
    constexpr int kThreads = 256;
    dim3 blocks(static_cast<unsigned int>((n_elems + kThreads - 1) / kThreads));
    switch (ggml_type) {
        case 8:  unpack_q8_0_kernel<OutT><<<blocks, kThreads, 0, stream>>>(raw, out, n_elems); break;
        case 2:  unpack_q4_0_kernel<OutT><<<blocks, kThreads, 0, stream>>>(raw, out, n_elems); break;
        case 3:  unpack_q4_1_kernel<OutT><<<blocks, kThreads, 0, stream>>>(raw, out, n_elems); break;
        case 12: unpack_q4_k_kernel<OutT><<<blocks, kThreads, 0, stream>>>(raw, out, n_elems); break;
        case 13: unpack_q5_k_kernel<OutT><<<blocks, kThreads, 0, stream>>>(raw, out, n_elems); break;
        case 14: unpack_q6_k_kernel<OutT><<<blocks, kThreads, 0, stream>>>(raw, out, n_elems); break;
        default: return -1;
    }
    return static_cast<int>(cudaGetLastError());
}

}  // namespace

extern "C" int phobos_cuda_unpack_ggml(
    int ggml_type,
    int out_dtype_code,
    const void* raw_u8,
    void* out_ptr,
    int64_t n_elems,
    void* stream_ptr
) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const uint8_t* raw = reinterpret_cast<const uint8_t*>(raw_u8);
    if (out_dtype_code == 0) {
        return launch_typed_unpack<__nv_bfloat16>(ggml_type, raw, reinterpret_cast<__nv_bfloat16*>(out_ptr), n_elems, stream);
    }
    if (out_dtype_code == 1) {
        return launch_typed_unpack<__half>(ggml_type, raw, reinterpret_cast<__half*>(out_ptr), n_elems, stream);
    }
    if (out_dtype_code == 2) {
        return launch_typed_unpack<float>(ggml_type, raw, reinterpret_cast<float*>(out_ptr), n_elems, stream);
    }
    return -2;
}

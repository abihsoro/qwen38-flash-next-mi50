// moe_int4_gfx906.h — wave64 port of the fork's MoE int4 decode kernels
// (N-D2). Source template: leapdragon/vllm-rdna2-qwen
// csrc/rocm/skinny_gemms_int4.cu lines 829-1003 (moe_local_expert,
// moe_topk_w, moe_w13_silu_gemv_, moe_w2_gemv_, moe_skinny_int4_decode).
// Wave64 delta per docs/gfx906/N-D2-PLAN.md: wave/lane /64 %64, stride += 64,
// 64-lane shfl_xor ladder (32..1), WAVES=4/block=256. Int4 unpack is scalar
// nibble (no v_dot4 — unavailable on gfx906); expert_map EP logic carries.
// F32_OUT (validation mode) emits the pre-rounding fp32 result.
#ifndef KERNELS_GFX906_MOE_INT4_GFX906_H
#define KERNELS_GFX906_MOE_INT4_GFX906_H

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace gfx906 {

__device__ __forceinline__ int moe_local_expert(const void* ids, bool ids_i64,
                                                const int32_t* expert_map,
                                                int idx) {
  int e = ids_i64 ? (int)reinterpret_cast<const int64_t*>(ids)[idx]
                  : reinterpret_cast<const int32_t*>(ids)[idx];
  if (expert_map != nullptr && e >= 0) e = expert_map[<bus>];
  return e;
}

__device__ __forceinline__ float moe_topk_w(const void* w, bool w_is_half,
                                            int idx) {
  return w_is_half ? __half2float(reinterpret_cast<const __half*>(w)[idx])
                   : reinterpret_cast<const float*>(w)[idx];
}

// act[m][s][n] = silu(gate) * up for expert = map(ids[m][s]); expert < 0
// (non-local under EP) -> act row = 0. Wave-per-output-row over N.
template <int WAVES, bool F32_OUT = false>
__global__ void __launch_bounds__(WAVES * 64)
moe_w13_silu_gemv_gfx906_(const __half* __restrict__ input,
                          const uint32_t* __restrict__ w13,
                          const __half* __restrict__ s13,
                          const void* __restrict__ topk_ids, const bool ids_i64,
                          const int32_t* __restrict__ expert_map,
                          __half* __restrict__ act, const int K, const int N,
                          const int topk, const int group_size) {
  const int m = blockIdx.z, s = blockIdx.y;
  const int wave = threadIdx.x / 64, lane = threadIdx.x % 64;
  const int n = blockIdx.x * WAVES + wave;
  extern __shared__ __half xs[];
  for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = input[m * K + i];
  __syncthreads();
  if (n >= N) return;
  const int expert = moe_local_expert(topk_ids, ids_i64, expert_map,
                                      m * topk + s);
  if (expert < 0) {
    if (lane == 0) {
      const float zero = 0.f;
      if (F32_OUT) reinterpret_cast<float*>(act)[((uint64_t)m * topk + s) * N + n] = zero;
      else act[((uint64_t)m * topk + s) * N + n] = __float2half(zero);
    }
    return;
  }
  const int K8 = K / 8, KG = K / group_size;
  const uint64_t base = (uint64_t)expert * 2 * N;
  const uint32_t* wg = w13 + (base + n) * K8;
  const uint32_t* wu = w13 + (base + N + n) * K8;
  const __half* sg = s13 + (base + n) * KG;
  const __half* su = s13 + (base + N + n) * KG;
  float accg = 0.f, accu = 0.f;
  for (int i = lane; i < K8; i += 64) {
    const uint32_t qg = wg[i], qu = wu[i];
    const int k0 = i * 8;
    float pg = 0.f, pu = 0.f;
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const float xv = __half2float(xs[k0 + j]);
      pg += (float)((int)((qg >> (4 * j)) & 0xF) - 8) * xv;
      pu += (float)((int)((qu >> (4 * j)) & 0xF) - 8) * xv;
    }
    const int g = k0 / group_size;
    accg += pg * __half2float(sg[g]);
    accu += pu * __half2float(su[g]);
  }
#pragma unroll
  for (int off = 32; off >= 1; off >>= 1) {
    accg += __shfl_xor(accg, off);
    accu += __shfl_xor(accu, off);
  }
  if (lane == 0) {
    const float silu = accg / (1.f + __expf(-accg));
    const float v = silu * accu;
    if (F32_OUT) reinterpret_cast<float*>(act)[((uint64_t)m * topk + s) * N + n] = v;
    else act[((uint64_t)m * topk + s) * N + n] = __float2half(v);
  }
}

// out[m][h] = sum_s topk_w[m][s] * sum_n act[m][s][n] * w2_int4 * s2,
// skipping non-local experts. Wave-per-output-row over H.
template <int WAVES, bool F32_OUT = false>
__global__ void __launch_bounds__(WAVES * 64)
moe_w2_gemv_gfx906_(const __half* __restrict__ act,
                    const uint32_t* __restrict__ w2,
                    const __half* __restrict__ s2,
                    const void* __restrict__ topk_ids, const bool ids_i64,
                    const int32_t* __restrict__ expert_map,
                    const void* __restrict__ topk_w, const bool w_is_half,
                    __half* __restrict__ out, const int N, const int H,
                    const int topk, const int group_size) {
  const int m = blockIdx.z;
  const int wave = threadIdx.x / 64, lane = threadIdx.x % 64;
  const int h = blockIdx.x * WAVES + wave;
  extern __shared__ __half as[];
  for (int i = threadIdx.x; i < topk * N; i += blockDim.x)
    as[i] = act[(uint64_t)m * topk * N + i];
  __syncthreads();
  if (h >= H) return;
  const int N8 = N / 8, NG = N / group_size;
  float acc = 0.f;
  for (int s = 0; s < topk; s++) {
    const int expert = moe_local_expert(topk_ids, ids_i64, expert_map,
                                        m * topk + s);
    if (expert < 0) continue;
    const uint32_t* wrow = w2 + ((uint64_t)expert * H + h) * N8;
    const __half* srow = s2 + ((uint64_t)expert * H + h) * NG;
    const __half* xrow = as + s * N;
    float sacc = 0.f;
    for (int i = lane; i < N8; i += 64) {
      const uint32_t q = wrow[i];
      const int k0 = i * 8;
      float p = 0.f;
#pragma unroll
      for (int j = 0; j < 8; j++)
        p += (float)((int)((q >> (4 * j)) & 0xF) - 8) *
             __half2float(xrow[k0 + j]);
      sacc += p * __half2float(srow[k0 / group_size]);
    }
    acc += sacc * moe_topk_w(topk_w, w_is_half, m * topk + s);
  }
#pragma unroll
  for (int off = 32; off >= 1; off >>= 1) acc += __shfl_xor(acc, off);
  if (lane == 0) {
    if (F32_OUT) reinterpret_cast<float*>(out)[(uint64_t)m * H + h] = acc;
    else out[(uint64_t)m * H + h] = __float2half(acc);
  }
}

}  // namespace gfx906

#endif  // KERNELS_GFX906_MOE_INT4_GFX906_H

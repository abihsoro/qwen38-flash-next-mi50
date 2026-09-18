// fused_glue_gfx906.h — wave64 port of the fork's T46 fused decode glue
// (leapdragon/vllm-rdna2-qwen csrc/rocm/rdna_fused_glue.cu). Four kernels:
//   gemv_act_k, hc_up_gate_mix_k, se_gate_up_silu_k, se_down_gated_k
// with RowF16/RowI8 weight policies. Wave64 delta (uniform with N-D1/N-D2):
// wave/lane /64 %64, dot stride 64*4, 64-lane shfl ladder (32..1),
// launch_bounds WAVES*64, block WV*64. HC=4 (config hc_count).
#ifndef KERNELS_GFX906_FUSED_GLUE_GFX906_H
#define KERNELS_GFX906_FUSED_GLUE_GFX906_H

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace gfx906 {

constexpr int kMaxM = 8;

struct RowF16 {
  const __half* w;
  __device__ __forceinline__ RowF16(const void* base, int n, int K, const __half*)
      : w(reinterpret_cast<const __half*>(base) + (size_t)n * K) {}
  template <int MT>
  __device__ __forceinline__ void dot(const __half* __restrict__ x, int K, int lane,
                                      float (&acc)[MT]) const {
    const int K8 = K / 8;
    const uint4* wr = reinterpret_cast<const uint4*>(w);
    const uint4* xr = reinterpret_cast<const uint4*>(x);
    for (int i = lane; i < K8; i += 64 * 4) {
      uint4 wq[<bus>];
#pragma unroll
      for (int u = 0; u < 4; u++) {
        const int idx = i + 64 * u;
        wq[u] = (idx < K8) ? wr[idx] : make_uint4(0, 0, 0, 0);
      }
#pragma unroll
      for (int u = 0; u < 4; u++) {
        const int idx = i + 64 * u;
        if (idx < K8) {
          const __half2* wh = reinterpret_cast<const __half2*>(&wq[u]);
#pragma unroll
          for (int m = 0; m < MT; m++) {
            const uint4 xq = xr[(size_t)m * K8 + idx];
            const __half2* xh = reinterpret_cast<const __half2*>(&xq);
            float a = acc[m];
            a = __builtin_amdgcn_fdot2(wh[<bus>], xh[<bus>], a, false);
            a = __builtin_amdgcn_fdot2(wh[<bus>], xh[<bus>], a, false);
            a = __builtin_amdgcn_fdot2(wh[<bus>], xh[<bus>], a, false);
            a = __builtin_amdgcn_fdot2(wh[<bus>], xh[<bus>], a, false);
            acc[m] = a;
          }
        }
      }
    }
  }
  __device__ __forceinline__ float scale() const { return 1.f; }
};

struct RowI8 {
  const int8_t* w;
  float s;
  __device__ __forceinline__ RowI8(const void* base, int n, int K, const __half* sc)
      : w(reinterpret_cast<const int8_t*>(base) + (size_t)n * K),
        s(__half2float(sc[n])) {}
  template <int MT>
  __device__ __forceinline__ void dot(const __half* __restrict__ x, int K, int lane,
                                      float (&acc)[MT]) const {
    const int K16 = K / 16;
    const uint4* wr = reinterpret_cast<const uint4*>(w);
    const uint4* xr = reinterpret_cast<const uint4*>(x);
    for (int i = lane; i < K16; i += 64 * 4) {
      uint4 wq[<bus>];
#pragma unroll
      for (int u = 0; u < 4; u++) {
        const int idx = i + 64 * u;
        wq[u] = (idx < K16) ? wr[idx] : make_uint4(0, 0, 0, 0);
      }
#pragma unroll
      for (int u = 0; u < 4; u++) {
        const int idx = i + 64 * u;
        if (idx < K16) {
          const int32_t* q = reinterpret_cast<const int32_t*>(&wq[u]);
          __half2 wh[<bus>];
#pragma unroll
          for (int j = 0; j < 4; j++) {
            const int32_t v = q[j];
            wh[2 * j] = __floats2half2_rn((float)(int8_t)(v & 0xff),
                                          (float)(int8_t)((v >> 8) & 0xff));
            wh[2 * j + 1] = __floats2half2_rn((float)(int8_t)((v >> 16) & 0xff),
                                              (float)(int8_t)((v >> 24) & 0xff));
          }
#pragma unroll
          for (int m = 0; m < MT; m++) {
            const uint4 xa = xr[((size_t)m * K16 + idx) * 2];
            const uint4 xb = xr[((size_t)m * K16 + idx) * 2 + 1];
            const __half2* h0 = reinterpret_cast<const __half2*>(&xa);
            const __half2* h1 = reinterpret_cast<const __half2*>(&xb);
            float a = acc[m];
            a = __builtin_amdgcn_fdot2(wh[<bus>], h0[<bus>], a, false);
            a = __builtin_amdgcn_fdot2(wh[<bus>], h0[<bus>], a, false);
            a = __builtin_amdgcn_fdot2(wh[<bus>], h0[<bus>], a, false);
            a = __builtin_amdgcn_fdot2(wh[<bus>], h0[<bus>], a, false);
            a = __builtin_amdgcn_fdot2(wh[<bus>], h1[<bus>], a, false);
            a = __builtin_amdgcn_fdot2(wh[<bus>], h1[<bus>], a, false);
            a = __builtin_amdgcn_fdot2(wh[<bus>], h1[<bus>], a, false);
            a = __builtin_amdgcn_fdot2(wh[<bus>], h1[<bus>], a, false);
            acc[m] = a;
          }
        }
      }
    }
  }
  __device__ __forceinline__ float scale() const { return s; }
};

template <int MT>
__device__ __forceinline__ void wave_reduce(float (&acc)[MT]) {
#pragma unroll
  for (int m = 0; m < MT; m++)
#pragma unroll
    for (int off = 32; off >= 1; off >>= 1) acc[m] += __shfl_xor(acc[m], off);
}
__device__ __forceinline__ float silu_f(float v) { return v / (1.f + __expf(-v)); }
__device__ __forceinline__ float sigmoid_f(float v) { return 1.f / (1.f + __expf(-v)); }

template <typename Row, int WAVES, int MT>
__global__ void __launch_bounds__(WAVES * 64)
gemv_act_k(const __half* __restrict__ x, const void* __restrict__ w,
           const __half* __restrict__ sc, __half* __restrict__ y, const int N,
           const int K, const int act_cols, const float act_scale) {
  const int wave = threadIdx.x / 64, lane = threadIdx.x % 64;
  const int n = blockIdx.x * WAVES + wave;
  if (n >= N) return;
  float acc[MT];
#pragma unroll
  for (int m = 0; m < MT; m++) acc[m] = 0.f;
  Row row(w, n, K, sc);
  row.template dot<MT>(x, K, lane, acc);
  wave_reduce<MT>(acc);
  if (lane == 0) {
    const float s = row.scale();
#pragma unroll
    for (int m = 0; m < MT; m++) {
      float v = acc[m] * s;
      if (n < act_cols) v = silu_f(v * act_scale);
      y[(size_t)m * N + n] = __float2half(v);
    }
  }
}

template <typename Row, int WAVES, int MT, int HC>
__global__ void __launch_bounds__(WAVES * 64)
hc_up_gate_mix_k(const __half* __restrict__ lora, const void* __restrict__ w,
                 const __half* __restrict__ sc, const __half* __restrict__ xn,
                 __half* __restrict__ out, const int H, const int R) {
  const int wave = threadIdx.x / 64, lane = threadIdx.x % 64;
  const int h = blockIdx.x * WAVES + wave;
  if (h >= H) return;
  float mix[MT];
#pragma unroll
  for (int m = 0; m < MT; m++) mix[m] = 0.f;
#pragma unroll
  for (int c = 0; c < HC; c++) {
    float acc[MT];
#pragma unroll
    for (int m = 0; m < MT; m++) acc[m] = 0.f;
    Row row(w, c * H + h, R, sc);
    row.template dot<MT>(lora, R, lane, acc);
    wave_reduce<MT>(acc);
    const float s = row.scale();
#pragma unroll
    for (int m = 0; m < MT; m++) {
      const float g = sigmoid_f(acc[m] * s);
      mix[m] += g * __half2float(xn[(size_t)m * (HC * H) + c * H + h]);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int m = 0; m < MT; m++) out[(size_t)m * H + h] = __float2half(mix[m] / HC);
  }
}

template <typename Row, int WAVES, int MT>
__global__ void __launch_bounds__(WAVES * 64)
se_gate_up_silu_k(const __half* __restrict__ x, const void* __restrict__ w,
                  const __half* __restrict__ sc, __half* __restrict__ act,
                  const int I, const int K) {
  const int wave = threadIdx.x / 64, lane = threadIdx.x % 64;
  const int i = blockIdx.x * WAVES + wave;
  if (i >= I) return;
  float g[MT], u[MT];
#pragma unroll
  for (int m = 0; m < MT; m++) { g[m] = 0.f; u[m] = 0.f; }
  Row rg(w, i, K, sc), ru(w, I + i, K, sc);
  rg.template dot<MT>(x, K, lane, g);
  ru.template dot<MT>(x, K, lane, u);
  wave_reduce<MT>(g);
  wave_reduce<MT>(u);
  if (lane == 0) {
    const float sg = rg.scale(), su = ru.scale();
#pragma unroll
    for (int m = 0; m < MT; m++)
      act[(size_t)m * I + i] = __float2half(silu_f(g[m] * sg) * (u[m] * su));
  }
}

template <typename Row, int WAVES, int MT>
__global__ void __launch_bounds__(WAVES * 64)
se_down_gated_k(const __half* __restrict__ act, const void* __restrict__ w,
                const __half* __restrict__ sc, const __half* __restrict__ x,
                const __half* __restrict__ wg, __half* __restrict__ out,
                const int H, const int I, const int Kx) {
  __shared__ float s_gate[kMaxM];
  const int wave = threadIdx.x / 64, lane = threadIdx.x % 64;
  if (wave == 0) {
    float gacc[MT];
#pragma unroll
    for (int m = 0; m < MT; m++) gacc[m] = 0.f;
    RowF16 rg(wg, 0, Kx, nullptr);
    rg.template dot<MT>(x, Kx, lane, gacc);
    wave_reduce<MT>(gacc);
    if (lane == 0) {
#pragma unroll
      for (int m = 0; m < MT; m++) s_gate[m] = sigmoid_f(gacc[m]);
    }
  }
  __syncthreads();
  const int h = blockIdx.x * WAVES + wave;
  if (h >= H) return;
  float acc[MT];
#pragma unroll
  for (int m = 0; m < MT; m++) acc[m] = 0.f;
  Row row(w, h, I, sc);
  row.template dot<MT>(act, I, lane, acc);
  wave_reduce<MT>(acc);
  if (lane == 0) {
    const float s = row.scale();
#pragma unroll
    for (int m = 0; m < MT; m++)
      out[(size_t)m * H + h] = __float2half(acc[m] * s * s_gate[m]);
  }
}

inline int pick_waves(int N) { return N >= 1152 ? 8 : (N >= 576 ? 4 : (N >= 288 ? 2 : 1)); }

template <typename Row, int WV, int MT>
void launch_gemv_act(int blocks, hipStream_t st, const __half* x, const void* w,
                     const __half* sc, __half* y, int N, int K, int act_cols,
                     float act_scale) {
  gemv_act_k<Row, WV, MT><<<blocks, WV * 64, 0, st>>>(x, w, sc, y, N, K, act_cols, act_scale);
}
template <typename Row, int WV, int MT>
void launch_hc_mix(int blocks, hipStream_t st, const __half* lora, const void* w,
                   const __half* sc, const __half* xn, __half* out, int H, int R) {
  hc_up_gate_mix_k<Row, WV, MT, 4><<<blocks, WV * 64, 0, st>>>(lora, w, sc, xn, out, H, R);
}
template <typename Row, int WV, int MT>
void launch_se_gu(int blocks, hipStream_t st, const __half* x, const void* w,
                  const __half* sc, __half* act, int I, int K) {
  se_gate_up_silu_k<Row, WV, MT><<<blocks, WV * 64, 0, st>>>(x, w, sc, act, I, K);
}
template <typename Row, int WV, int MT>
void launch_se_dn(int blocks, hipStream_t st, const __half* act, const void* w,
                  const __half* sc, const __half* x, const __half* wg,
                  __half* out, int H, int I, int Kx) {
  se_down_gated_k<Row, WV, MT><<<blocks, WV * 64, 0, st>>>(act, w, sc, x, wg, out, H, I, Kx);
}

}  // namespace gfx906

#endif  // KERNELS_GFX906_FUSED_GLUE_GFX906_H

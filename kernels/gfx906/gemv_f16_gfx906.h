// gemv_f16_gfx906.h — wave64 port of the fork's gemv_f16_rdna2_ (N-D1).
//
// Source template: leapdragon/vllm-rdna2-qwen csrc/rocm/skinny_gemms_int4.cu
// lines 1015-1118 (gemv_f16_rdna2_<WAVES,MT> + launcher). Wave64 delta per
// docs/gfx906/N-D1-PLAN.md:
//   wave/lane        : /32 %32  ->  /64 %64
//   load stride      : 32*4     ->  64*4
//   reduce ladder    : shfl_xor 16..1 -> 32,16,8,4,2,1 (64-lane XOR ladder)
//   launch_bounds    : WAVES*32  ->  WAVES*64
//   block dims       : 32*WAVES  ->  64*WAVES (512/256/128/64)
// Carries as-is: __builtin_amdgcn_fdot2 (v_dot2_f32_f16, verified exact on
// gfx906), 16-byte uint4 loads, lane-0 epilogue + bias, N-tiered WAVES choice.
#ifndef KERNELS_GFX906_GEMV_F16_GFX906_H
#define KERNELS_GFX906_GEMV_F16_GFX906_H

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace gfx906 {

// y[m][n] = sum_k x[m][k] * w[n][k] (+bias[n]); fp16 in, fp32 accumulate.
// Wave-per-output-row: each wavefront (64 lanes) computes output row n.
// F32_OUT (validation mode): emit the fp32 accumulator instead of the fp16
// rounded output, so harnesses can validate accumulation independently of the
// fp16 output rounding.
template <int WAVES, int MT, bool F32_OUT = false>
__global__ void __launch_bounds__(WAVES * 64)
gemv_f16_gfx906_(const __half* __restrict__ x, const __half* __restrict__ w,
                 const __half* __restrict__ bias, __half* __restrict__ y,
                 const int N, const int K) {
  const int wave = threadIdx.x / 64, lane = threadIdx.x % 64;
  const int n = blockIdx.x * WAVES + wave;
  if (n >= N) return;
  const int K8 = K / 8;
  const uint4* __restrict__ wrow =
      reinterpret_cast<const uint4*>(w + (size_t)n * K);
  const uint4* __restrict__ xr = reinterpret_cast<const uint4*>(x);
  float acc[MT];
#pragma unroll
  for (int m = 0; m < MT; m++) acc[m] = 0.f;
  for (int i = lane; i < K8; i += 64 * 4) {
    uint4 wq[<bus>];
#pragma unroll
    for (int u = 0; u < 4; u++) {
      const int idx = i + 64 * u;
      wq[u] = (idx < K8) ? wrow[idx] : make_uint4(0, 0, 0, 0);
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
#pragma unroll
  for (int m = 0; m < MT; m++) {
#pragma unroll
    for (int off = 32; off >= 1; off >>= 1) acc[m] += __shfl_xor(acc[m], off);
  }
  if (lane == 0) {
    const float b = bias ? __half2float(bias[n]) : 0.f;
    if (F32_OUT) {
      float* yf = reinterpret_cast<float*>(y);
#pragma unroll
      for (int m = 0; m < MT; m++) yf[(size_t)m * N + n] = acc[m] + b;
    } else {
#pragma unroll
      for (int m = 0; m < MT; m++)
        y[(size_t)m * N + n] = __float2half(acc[m] + b);
    }
  }
}

template <int MT, bool F32_OUT = false>
static void gemv_f16_gfx906_launch(const __half* x, const __half* w,
                                   const __half* bias, __half* y, int N, int K,
                                   hipStream_t s) {
  // N-tiered waves, mirroring the fork; block = 64 * WAVES on wave64.
  if (N >= 1152)
    gemv_f16_gfx906_<8, MT, F32_OUT><<<(N + 7) / 8, 512, 0, s>>>(x, w, bias, y, N, K);
  else if (N >= 576)
    gemv_f16_gfx906_<4, MT, F32_OUT><<<(N + 3) / 4, 256, 0, s>>>(x, w, bias, y, N, K);
  else if (N >= 288)
    gemv_f16_gfx906_<2, MT, F32_OUT><<<(N + 1) / 2, 128, 0, s>>>(x, w, bias, y, N, K);
  else
    gemv_f16_gfx906_<1, MT, F32_OUT><<<N, 64, 0, s>>>(x, w, bias, y, N, K);
}

}  // namespace gfx906

#endif  // KERNELS_GFX906_GEMV_F16_GFX906_H

// nd1_harness.cu — N-D1 validation: gemv_f16_gfx906 vs exact (double) CPU
// reference across the fork's 12 dense shapes at M in {1,4,5,8}.
// Two-level acceptance:
//   (a) fp32 accumulation: kernel F32_OUT output vs double reference, relerr
//       <= 1e-4  (the kernel's real correctness — accumulation agreement).
//   (b) fp16 serving output vs the double reference (fp16 output rounding is
//       inherent to the kernel contract), relerr <= 1e-3 (one fp16 ulp —
//       allows rounding-boundary flips where tiny accumulation differences
//       flip the fp16 output by 1 ulp).
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "gemv_f16_gfx906.h"

struct Shape {
  const char* name;
  int N, K;
};

static const Shape kShapes[] = {
    {"gdn.in_proj_qkv/rank", 2560, 2560}, {"gdn.in_proj_z/rank", 1536, 2560},
    {"gdn.out_proj/rank", 2560, 1536},    {"qsa.q_proj/rank", 3072, 2560},
    {"qsa.o_proj/rank", 2560, 1536},      {"router.gate", 512, 2560},
    {"shared.gate_up", 1280, 2560},       {"shared.down", 2560, 640},
    {"hc.down", 320, 10240},              {"hc.up", 10240, 320},
    {"hc.inject", 4, 10240},              {"lm_head/rank", 62080, 2560},
};
static const int kNumShapes = sizeof(kShapes) / sizeof(kShapes[<bus>]);
static const int kMs[] = {1, 4, 5, 8};

static float frand() { return (float)(rand() % 100000) / 50000.0f - 1.0f; }

template <int MT>
static void launch(const __half* x, const __half* w, const __half* bias,
                   __half* y, int N, int K) {
  gfx906::gemv_f16_gfx906_launch<MT>(x, w, bias, y, N, K, 0);
}

template <int MT>
static void launch_f32(const __half* x, const __half* w, const __half* bias,
                       __half* y, int N, int K) {
  gfx906::gemv_f16_gfx906_launch<MT, true>(x, w, bias, y, N, K, 0);
}

int main() {
  bool all_ok = true;
  srand(42);
  for (int si = 0; si < kNumShapes; si++) {
    const int N = kShapes[si].N, K = kShapes[si].K;
    std::vector<__half> w(N * K), bias(N);
    for (int i = 0; i < N * K; i++) w[i] = __float2half(frand() * 0.02f);
    for (int i = 0; i < N; i++) bias[i] = __float2half(frand() * 0.01f);
    for (int mi = 0; mi < 4; mi++) {
      const int M = kMs[mi];
      std::vector<__half> x(M * K);
      for (int i = 0; i < M * K; i++) x[i] = __float2half(frand());
      std::vector<float> ref(M * N);
      for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
          double s = 0.0;
          for (int k = 0; k < K; k++)
            s += (double)__half2float(x[(size_t)m * K + k]) *
                 (double)__half2float(w[(size_t)n * K + k]);
          ref[(size_t)m * N + n] = (float)(s + (double)__half2float(bias[n]));
        }
      __half *dx, *dw, *db;
      float *dyf = nullptr;
      __half *dyh = nullptr;
      hipMalloc(&dx, M * K * 2);
      hipMalloc(&dw, N * K * 2);
      hipMalloc(&db, N * 2);
      hipMalloc(&dyf, M * N * 4);
      hipMalloc(&dyh, M * N * 2);
      hipMemcpy(dx, x.data(), M * K * 2, hipMemcpyHostToDevice);
      hipMemcpy(dw, w.data(), N * K * 2, hipMemcpyHostToDevice);
      hipMemcpy(db, bias.data(), N * 2, hipMemcpyHostToDevice);
      switch (M) {
        case 1: launch_f32<1>(dx, dw, db, (__half*)dyf, N, K); break;
        case 2: launch_f32<2>(dx, dw, db, (__half*)dyf, N, K); break;
        case 3: launch_f32<3>(dx, dw, db, (__half*)dyf, N, K); break;
        case 4: launch_f32<4>(dx, dw, db, (__half*)dyf, N, K); break;
        case 5: launch_f32<5>(dx, dw, db, (__half*)dyf, N, K); break;
        case 6: launch_f32<6>(dx, dw, db, (__half*)dyf, N, K); break;
        case 7: launch_f32<7>(dx, dw, db, (__half*)dyf, N, K); break;
        default: launch_f32<8>(dx, dw, db, (__half*)dyf, N, K); break;
      }
      hipDeviceSynchronize();
      std::vector<float> yf(M * N);
      hipMemcpy(yf.data(), dyf, M * N * 4, hipMemcpyDeviceToHost);
      hipEvent_t e0, e1;
      hipEventCreate(&e0);
      hipEventCreate(&e1);
      const int reps = 50;
      for (int r = 0; r < 5; r++) {
        switch (M) {
          case 1: launch<1>(dx, dw, db, dyh, N, K); break;
          case 2: launch<2>(dx, dw, db, dyh, N, K); break;
          case 3: launch<3>(dx, dw, db, dyh, N, K); break;
          case 4: launch<4>(dx, dw, db, dyh, N, K); break;
          case 5: launch<5>(dx, dw, db, dyh, N, K); break;
          case 6: launch<6>(dx, dw, db, dyh, N, K); break;
          case 7: launch<7>(dx, dw, db, dyh, N, K); break;
          default: launch<8>(dx, dw, db, dyh, N, K); break;
        }
      }
      hipDeviceSynchronize();
      hipEventRecord(e0);
      for (int r = 0; r < reps; r++) {
        switch (M) {
          case 1: launch<1>(dx, dw, db, dyh, N, K); break;
          case 2: launch<2>(dx, dw, db, dyh, N, K); break;
          case 3: launch<3>(dx, dw, db, dyh, N, K); break;
          case 4: launch<4>(dx, dw, db, dyh, N, K); break;
          case 5: launch<5>(dx, dw, db, dyh, N, K); break;
          case 6: launch<6>(dx, dw, db, dyh, N, K); break;
          case 7: launch<7>(dx, dw, db, dyh, N, K); break;
          default: launch<8>(dx, dw, db, dyh, N, K); break;
        }
      }
      hipEventRecord(e1);
      hipEventSynchronize(e1);
      float ms = 0;
      hipEventElapsedTime(&ms, e0, e1);
      double us = ms * 1000.0 / reps;
      std::vector<__half> yh(M * N);
      hipMemcpy(yh.data(), dyh, M * N * 2, hipMemcpyDeviceToHost);
      double max_ref = 0, err_acc = 0, err_out = 0;
      for (size_t i = 0; i < ref.size(); i++) {
        double r = fabs(ref[i]);
        if (r > max_ref) max_ref = r;
        double ea = fabs((double)yf[i] - ref[i]);
        if (ea > err_acc) err_acc = ea;
        double eo = fabs((double)__half2float(yh[i]) - ref[i]);
        if (eo > err_out) err_out = eo;
      }
      double relerr_acc = max_ref > 0 ? err_acc / max_ref : 0.0;
      double relerr_out = max_ref > 0 ? err_out / max_ref : 0.0;
      bool ok = (relerr_acc <= 1e-4) && (relerr_out <= 1e-3);
      all_ok = all_ok && ok;
      double gb = (double)N * K * 2 / 1e9;
      printf("M=%d %-20s [%6dx%5d] %6.1fMB  gfx906 %7.1fus (%4.0f GB/s)  acc %.2e  out %.2e  %s\n",
             M, kShapes[si].name, N, K, gb * 1e3, us, gb / us * 1e6,
             relerr_acc, relerr_out, ok ? "PASS" : "FAIL");
      hipFree(dx);
      hipFree(dw);
      hipFree(db);
      hipFree(dyf);
      hipFree(dyh);
    }
  }
  printf(all_ok ? "N-D1: ALL SHAPES PASS (acc <= 1e-4, out <= 1e-3)\n"
                : "N-D1: FAILURES PRESENT\n");
  return all_ok ? 0 : 1;
}

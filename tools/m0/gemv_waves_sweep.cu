// gemv_waves_sweep.cu — does the N-based WAVES heuristic actually pick the best variant?
//
// The production launcher (csrc/rocm/skinny_gemms_int4.cu:1074-1081) chooses WAVES purely from N:
//     N >= 1152 -> WAVES=8 (512 threads)     N >= 576 -> WAVES=4 (256)
//     N >=  288 -> WAVES=2 (128)             else     -> WAVES=1 (64)
// The baseline harness (M=1) shows several shapes far off the ~810 GB/s HBM ceiling:
//     hc.down 196 GB/s (N=320 -> WAVES=2, only 160 blocks), router.gate 327 (N=512 -> WAVES=2),
//     shared.down 416 (K=640, tiny work per wave), the o_proj/out_proj family ~520 (K=1536).
// Two plausible causes: (a) too few threads at small N, (b) too little work per wave at small K.
// WAVES is the cheapest knob to test for (a), so this harness tries EVERY WAVES for EVERY shape.
//
// Correctness is checked against a CPU fp32 reference for every (shape, WAVES) combination, so a
// faster-but-wrong variant can never be adopted.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP error %s at %d\n", hipGetErrorString(e), __LINE__); exit(1);} } while (0)

// --- production kernel, verbatim from tools/rdna2/gemv_f16_harness.cu (wave-per-output-row) ---
template <int WAVES, int MT>
__global__ void __launch_bounds__(WAVES * 32)
gemv_f16_rdna2_(const half* __restrict__ x, const half* __restrict__ w,
                const half* __restrict__ bias, half* __restrict__ y,
                const int N, const int K) {
  const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int n = blockIdx.x * WAVES + wave;
  if (n >= N) return;
  const int K8 = K / 8;
  const uint4* __restrict__ wrow = reinterpret_cast<const uint4*>(w + (size_t)n * K);
  const uint4* __restrict__ xr = reinterpret_cast<const uint4*>(x);
  float acc[MT];
#pragma unroll
  for (int m = 0; m < MT; m++) acc[m] = 0.f;
  for (int k = lane; k < K8; k += 32) {
    const uint4 wv = wrow[k];
    const half2* wp = reinterpret_cast<const half2*>(&wv);
#pragma unroll
    for (int m = 0; m < MT; m++) {
      const uint4 xv = xr[(size_t)m * K8 + k];
      const half2* xp = reinterpret_cast<const half2*>(&xv);
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const half2 a = __hmul2(xp[j], wp[j]);
        acc[m] += __low2float(a) + __high2float(a);
      }
    }
  }
#pragma unroll
  for (int m = 0; m < MT; m++) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      acc[m] += __shfl_down(acc[m], off, 32);
  }
  if (lane == 0) {
#pragma unroll
    for (int m = 0; m < MT; m++) {
      float b = bias ? __half2float(bias[n]) : 0.f;
      y[(size_t)m * N + n] = __float2half(acc[m] + b);
    }
  }
}

template <int WAVES, int MT>
static void launch_one(const half* x, const half* w, const half* bias, half* y,
                       int N, int K, hipStream_t s) {
  gemv_f16_rdna2_<WAVES, MT><<<(N + WAVES - 1) / WAVES, WAVES * 32, 0, s>>>(x, w, bias, y, N, K);
}

static void run_variant(int WAVES, int M, const half* x, const half* w, const half* bias,
                        half* y, int N, int K, hipStream_t s) {
  // M is always 1 here (decode); MT=1 keeps this comparable to gemv_f16_rdna2_<*,1>
  switch (WAVES) {
    case 8: launch_one<8, 1>(x, w, bias, y, N, K, s); break;
    case 4: launch_one<4, 1>(x, w, bias, y, N, K, s); break;
    case 2: launch_one<2, 1>(x, w, bias, y, N, K, s); break;
    default: launch_one<1, 1>(x, w, bias, y, N, K, s); break;
  }
}

static inline float h2f(uint16_t h) { __half x; memcpy(&x, &h, 2); return __half2float(x); }
static inline uint16_t f2h(float f) { __half x = __float2half(f); uint16_t h; memcpy(&h, &x, 2); return h; }

static int prod_waves(int N) { return N >= 1152 ? 8 : N >= 576 ? 4 : N >= 288 ? 2 : 1; }

int main(int argc, char** argv) {
  struct S { const char* name; int N, K; };
  S shapes[] = {{"gdn.in_proj_qkv/rank", 2560, 2560}, {"gdn.in_proj_z/rank", 1536, 2560},
                {"gdn.out_proj/rank", 2560, 1536}, {"qsa.q_proj/rank", 3072, 2560},
                {"qsa.o_proj/rank", 2560, 1536}, {"router.gate", 512, 2560},
                {"shared.gate_up", 1280, 2560}, {"shared.down", 2560, 640},
                {"hc.down", 320, 10240}, {"hc.up", 10240, 320}, {"hc.inject", 4, 10240},
                {"lm_head/rank", 62080, 2560}};
  const int WAVES_SET[] = {1, 2, 4, 8};
  const int M = 1;
  const double CEIL = 810.0;
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> uf(-1.f, 1.f);
  int fails = 0;

  printf("M=%d (decode). For each shape: every WAVES variant, vs the production heuristic.\n", M);
  printf("%-22s %6s %6s | %8s %7s %6s | %8s %7s %6s | %s\n",
         "shape", "N", "K", "prod GB/s", "us", "%ceil", "best GB/s", "us", "%ceil", "verdict");

  double prod_total = 0, best_total = 0;
  for (auto& sh : shapes) {
    const int N = sh.N, K = sh.K;
    std::vector<uint16_t> x((size_t)M * K), w((size_t)N * K), b(N), y((size_t)M * N);
    for (auto& v : x) v = f2h(uf(rng));
    for (auto& v : w) v = f2h(uf(rng) * 0.05f);
    for (auto& v : b) v = f2h(uf(rng) * 0.1f);
    std::vector<float> ref((size_t)M * N);
    for (int m = 0; m < M; m++)
      for (int n = 0; n < N; n++) {
        float a = 0;
        const uint16_t* xr = &x[(size_t)m * K]; const uint16_t* wr = &w[(size_t)n * K];
        for (int k = 0; k < K; k++) a += h2f(xr[k]) * h2f(wr[k]);
        ref[(size_t)m * N + n] = a + h2f(b[n]);
      }
    half *dx, *dw, *db, *dy;
    CHECK(hipMalloc(&dx, x.size() * 2)); CHECK(hipMalloc(&dw, w.size() * 2));
    CHECK(hipMalloc(&db, b.size() * 2)); CHECK(hipMalloc(&dy, y.size() * 2));
    CHECK(hipMemcpy(dx, x.data(), x.size() * 2, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dw, w.data(), w.size() * 2, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(db, b.data(), b.size() * 2, hipMemcpyHostToDevice));

    double best_us = 1e18; int best_waves = -1;
    double prod_us = 0;
    for (int W : WAVES_SET) {
      run_variant(W, M, dx, dw, db, dy, N, K, 0);
      CHECK(hipDeviceSynchronize());
      CHECK(hipMemcpy(y.data(), dy, y.size() * 2, hipMemcpyDeviceToHost));
      double maxref = 0, maxerr = 0;
      for (size_t i = 0; i < y.size(); i++) {
        maxref = fmax(maxref, fabs(ref[i]));
        maxerr = fmax(maxerr, fabs(h2f(y[i]) - ref[i]));
      }
      const bool ok = maxerr <= 2e-2 * maxref + 1e-2;
      if (!ok) { fails++; printf("  !! WAVES=%d WRONG on %s (relerr %.2e)\n", W, sh.name, maxerr/(maxref+1e-9)); continue; }
      for (int i = 0; i < 20; i++) run_variant(W, M, dx, dw, db, dy, N, K, 0);
      CHECK(hipDeviceSynchronize());
      const int iters = 200;
      auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < iters; i++) run_variant(W, M, dx, dw, db, dy, N, K, 0);
      CHECK(hipDeviceSynchronize());
      auto t1 = std::chrono::steady_clock::now();
      double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
      if (W == prod_waves(N)) prod_us = us;
      if (us < best_us) { best_us = us; best_waves = W; }
    }
    double gb = (double)N * K * 2 / 1e9;
    double prod_gbs = prod_us > 0 ? gb / prod_us * 1e6 : 0;
    double best_gbs = gb / best_us * 1e6;
    prod_total += prod_us; best_total += best_us;
    const char* verdict = (best_waves != prod_waves(N) && best_us < prod_us * 0.97)
                          ? "CHANGE" : "keep";
    printf("%-22s %6d %6d | %8.0f %7.1f %5.0f%% | %8.0f %7.1f %5.0f%% | W=%d %s\n",
           sh.name, N, K, prod_gbs, prod_us, 100*prod_gbs/CEIL,
           best_gbs, best_us, 100*best_gbs/CEIL, best_waves, verdict);
    hipFree(dx); hipFree(dw); hipFree(db); hipFree(dy);
  }
  printf("\nSUM  production=%.0f us   best-per-shape=%.0f us   (%+.1f%%)\n",
         prod_total, best_total, 100.0 * (best_total - prod_total) / prod_total);
  printf("%s\n", fails ? "SOME VARIANTS FAILED CORRECTNESS" : "ALL VARIANTS CORRECT");
  return fails ? 1 : 0;
}

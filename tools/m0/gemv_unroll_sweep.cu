// gemv_unroll_sweep.cu — test k-loop unrolling as the real lever for the latency-bound shapes.
//
// WHY: the production kernel's inner loop has NO unrolling of k (despite the design comment in the
// harness claiming "4-deep unroll for memory-level parallelism"):
//     for (int k = lane; k < K8; k += 32) { const uint4 wv = wrow[k]; ...math... }
// Each thread therefore keeps ~1-2 loads in flight. That is the textbook cause of the low achieved
// bandwidth on the short-K / small-N shapes from the baseline:
//     hc.down 196-277 GB/s (33%), router.gate ~350 (43%), o_proj/out_proj ~525 (65%),
//     v.s. lm_head 762-779 (94%) which has a huge N and hides latency by block count.
// WAVES was already ruled out (total threads ~= N*32 regardless of WAVES, so it cannot change
// occupancy - measured at only -5.1% total).
//
// WHAT THIS DOES: issue U independent w-loads per iteration so the memory system has U requests in
// flight per thread, then accumulate. U in {1 (=production), 2, 4, 8}. Correctness is verified
// against a CPU fp32 reference for EVERY variant, so a faster-but-wrong version cannot be adopted.
//
// NOTE: the k-tail is handled explicitly (K8 may not divide by U), so this is safe for any K.
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

// U independent k-loads per thread per iteration.
template <int WAVES, int U>
__global__ void __launch_bounds__(WAVES * 32)
gemv_f16_unroll_(const half* __restrict__ x, const half* __restrict__ w,
                 const half* __restrict__ bias, half* __restrict__ y,
                 const int N, const int K) {
  const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int n = blockIdx.x * WAVES + wave;
  if (n >= N) return;
  const int K8 = K / 8;
  const uint4* __restrict__ wrow = reinterpret_cast<const uint4*>(w + (size_t)n * K);
  const uint4* __restrict__ xr = reinterpret_cast<const uint4*>(x);
  float acc = 0.f;

  int k = lane;
  // main unrolled body: U independent loads issued before any use of them
  for (; k + (U - 1) * 32 < K8; k += 32 * U) {
    uint4 wv[U];
#pragma unroll
    for (int u = 0; u < U; u++) wv[u] = wrow[k + u * 32];
#pragma unroll
    for (int u = 0; u < U; u++) {
      const uint4 xv = xr[k + u * 32];
      const half2* wp = reinterpret_cast<const half2*>(&wv[u]);
      const half2* xp = reinterpret_cast<const half2*>(&xv);
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const half2 a = __hmul2(xp[j], wp[j]);
        acc += __low2float(a) + __high2float(a);
      }
    }
  }
  // tail
  for (; k < K8; k += 32) {
    const uint4 wv = wrow[k];
    const uint4 xv = xr[k];
    const half2* wp = reinterpret_cast<const half2*>(&wv);
    const half2* xp = reinterpret_cast<const half2*>(&xv);
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const half2 a = __hmul2(xp[j], wp[j]);
      acc += __low2float(a) + __high2float(a);
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off, 32);
  if (lane == 0) {
    float b = bias ? __half2float(bias[n]) : 0.f;
    y[n] = __float2half(acc + b);
  }
}

template <int WAVES, int U>
static void launch_u(const half* x, const half* w, const half* bias, half* y,
                     int N, int K, hipStream_t s) {
  gemv_f16_unroll_<WAVES, U><<<(N + WAVES - 1) / WAVES, WAVES * 32, 0, s>>>(x, w, bias, y, N, K);
}

static void run_u(int U, int WAVES, const half* x, const half* w, const half* bias,
                  half* y, int N, int K, hipStream_t s) {
  if (WAVES == 8) { if (U == 1) launch_u<8,1>(x,w,bias,y,N,K,s); else if (U == 2) launch_u<8,2>(x,w,bias,y,N,K,s); else if (U == 4) launch_u<8,4>(x,w,bias,y,N,K,s); else launch_u<8,8>(x,w,bias,y,N,K,s); }
  else if (WAVES == 4) { if (U == 1) launch_u<4,1>(x,w,bias,y,N,K,s); else if (U == 2) launch_u<4,2>(x,w,bias,y,N,K,s); else if (U == 4) launch_u<4,4>(x,w,bias,y,N,K,s); else launch_u<4,8>(x,w,bias,y,N,K,s); }
  else if (WAVES == 2) { if (U == 1) launch_u<2,1>(x,w,bias,y,N,K,s); else if (U == 2) launch_u<2,2>(x,w,bias,y,N,K,s); else if (U == 4) launch_u<2,4>(x,w,bias,y,N,K,s); else launch_u<2,8>(x,w,bias,y,N,K,s); }
  else { if (U == 1) launch_u<1,1>(x,w,bias,y,N,K,s); else if (U == 2) launch_u<1,2>(x,w,bias,y,N,K,s); else if (U == 4) launch_u<1,4>(x,w,bias,y,N,K,s); else launch_u<1,8>(x,w,bias,y,N,K,s); }
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
  const int US[] = {1, 2, 4, 8};
  const double CEIL = 810.0;
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> uf(-1.f, 1.f);
  int fails = 0;

  printf("M=1 decode. WAVES fixed at the production choice per shape; sweeping k-unroll depth U.\n");
  printf("%-22s %6s %6s %3s | %8s %7s %5s | %8s %7s %5s %3s | %s\n",
         "shape", "N", "K", "W", "U1 GB/s", "us", "%cl", "best GB/s", "us", "%cl", "U", "verdict");

  double t_u1 = 0, t_best = 0;
  for (auto& sh : shapes) {
    const int N = sh.N, K = sh.K, W = prod_waves(N);
    std::vector<uint16_t> x((size_t)K), w((size_t)N * K), b(N), y((size_t)N);
    for (auto& v : x) v = f2h(uf(rng));
    for (auto& v : w) v = f2h(uf(rng) * 0.05f);
    for (auto& v : b) v = f2h(uf(rng) * 0.1f);
    std::vector<float> ref((size_t)N);
    for (int n = 0; n < N; n++) {
      float a = 0;
      const uint16_t* wr = &w[(size_t)n * K];
      for (int k = 0; k < K; k++) a += h2f(x[k]) * h2f(wr[k]);
      ref[n] = a + h2f(b[n]);
    }
    half *dx, *dw, *db, *dy;
    CHECK(hipMalloc(&dx, x.size() * 2)); CHECK(hipMalloc(&dw, w.size() * 2));
    CHECK(hipMalloc(&db, b.size() * 2)); CHECK(hipMalloc(&dy, y.size() * 2));
    CHECK(hipMemcpy(dx, x.data(), x.size() * 2, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dw, w.data(), w.size() * 2, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(db, b.data(), b.size() * 2, hipMemcpyHostToDevice));

    double u1_us = 0, best_us = 1e18; int best_u = -1;
    for (int U : US) {
      run_u(U, W, dx, dw, db, dy, N, K, 0);
      CHECK(hipDeviceSynchronize());
      CHECK(hipMemcpy(y.data(), dy, y.size() * 2, hipMemcpyDeviceToHost));
      double maxref = 0, maxerr = 0;
      for (size_t i = 0; i < y.size(); i++) {
        maxref = fmax(maxref, fabs(ref[i]));
        maxerr = fmax(maxerr, fabs(h2f(y[i]) - ref[i]));
      }
      if (!(maxerr <= 2e-2 * maxref + 1e-2)) {
        fails++;
        printf("  !! U=%d WRONG on %s (relerr %.2e)\n", U, sh.name, maxerr / (maxref + 1e-9));
        continue;
      }
      for (int i = 0; i < 20; i++) run_u(U, W, dx, dw, db, dy, N, K, 0);
      CHECK(hipDeviceSynchronize());
      const int iters = 200;
      auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < iters; i++) run_u(U, W, dx, dw, db, dy, N, K, 0);
      CHECK(hipDeviceSynchronize());
      auto t1 = std::chrono::steady_clock::now();
      double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
      if (U == 1) u1_us = us;
      if (us < best_us) { best_us = us; best_u = U; }
    }
    double gb = (double)N * K * 2 / 1e9;
    double u1_gbs = u1_us > 0 ? gb / u1_us * 1e6 : 0, best_gbs = gb / best_us * 1e6;
    t_u1 += u1_us; t_best += best_us;
    const char* verdict = (best_u != 1 && best_us < u1_us * 0.97) ? "ADOPT-CANDIDATE" : "keep";
    printf("%-22s %6d %6d %3d | %8.0f %7.1f %4.0f%% | %8.0f %7.1f %4.0f%% %3d | %s\n",
           sh.name, N, K, W, u1_gbs, u1_us, 100 * u1_gbs / CEIL,
           best_gbs, best_us, 100 * best_gbs / CEIL, best_u, verdict);
    hipFree(dx); hipFree(dw); hipFree(db); hipFree(dy);
  }
  printf("\nSUM  U=1 (production loop)=%.0f us   best-U-per-shape=%.0f us   (%+.1f%%)\n",
         t_u1, t_best, 100.0 * (t_best - t_u1) / t_u1);
  printf("%s\n", fails ? "SOME VARIANTS FAILED CORRECTNESS" : "ALL VARIANTS CORRECT");
  return fails ? 1 : 0;
}

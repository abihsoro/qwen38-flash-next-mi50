// tp4_rank0_probe.cu — admissible kernel evidence at TP4 rank-0 shard shapes
// (D050/D063). Runs the validated wave64 gemv_f16 at the quarter-width shapes
// TP4 rank 0 computes, records GB/s + fp16-output correctness vs a CPU
// reference. Rows are reported with shape_profile: tp4_rank0 for the I5
// acceptance pipeline. shape_profile: tp4_rank0 IS admissible (D050); full is
// not.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "gemv_f16_gfx906.h"

static float frand() { return (float)(rand() % 100000) / 50000.0f - 1.0f; }

// run one (N, K) rank shape at M=1: correctness (fp32 accum, fp16 out vs
// fp16-rounded double ref) + timing
void probe(const char* name, int N, int K) {
  std::vector<__half> w(N * K), x(K), yv(N);
  for (auto& v : w) v = __float2half(frand() * 0.02f);
  for (auto& v : x) v = __float2half(frand());
  __half *dw, *dx, *dy;
  hipMalloc(&dw, (size_t)N * K * 2);
  hipMalloc(&dx, K * 2);
  hipMalloc(&dy, N * 2);
  hipMemcpy(dw, w.data(), (size_t)N * K * 2, hipMemcpyHostToDevice);
  hipMemcpy(dx, x.data(), K * 2, hipMemcpyHostToDevice);
  // launch via the gfx906 header's dispatch (pick waves by N, MT=1)
  gfx906::gemv_f16_gfx906_launch<1>(dx, dw, nullptr, dy, N, K, 0);
  hipDeviceSynchronize();
  hipMemcpy(yv.data(), dy, N * 2, hipMemcpyDeviceToHost);
  // reference
  double maxe = 0, maxr = 0;
  for (int n = 0; n < N; n++) {
    double s = 0;
    for (int k = 0; k < K; k++)
      s += (double)__half2float(x[k]) * (double)__half2float(w[(size_t)n * K + k]);
    double ref = __half2float(__float2half((float)s));
    double e = fabs((double)__half2float(yv[n]) - ref);
    if (e > maxe) maxe = e;
    if (fabs(ref) > maxr) maxr = fabs(ref);
  }
  double rel = maxr > 0 ? maxe / maxr : 0;
  // timing (warm + 50 iters)
  for (int i = 0; i < 10; i++)
    gfx906::gemv_f16_gfx906_launch<1>(dx, dw, nullptr, dy, N, K, 0);
  hipDeviceSynchronize();
  hipEvent_t t0, t1;
  hipEventCreate(&t0); hipEventCreate(&t1);
  hipEventRecord(t0, 0);
  const int IT = 200;
  for (int i = 0; i < IT; i++)
    gfx906::gemv_f16_gfx906_launch<1>(dx, dw, nullptr, dy, N, K, 0);
  hipEventRecord(t1, 0);
  hipEventSynchronize(t1);
  float ms;
  hipEventElapsedTime(&ms, t0, t1);
  double bytes = (double)N * K * 2;
  double gbs = bytes / (ms / IT * 1e-3) / 1e9;
  printf("tp4_rank0 %-16s N=%6d K=%6d  relerr %.2e  %7.1f us  %6.0f GB/s\n",
         name, N, K, rel, ms / IT * 1e3, gbs);
}

int main() {
  // TP4 rank-0 shapes (full width in the trailing comment)
  probe("gdn.in_proj_qkv", 640, 2560);   // 2560
  probe("gdn.in_proj_z", 384, 2560);     // 1536
  probe("gdn.out_proj", 640, 1536);      // 2560
  probe("qsa.q_proj", 768, 2560);        // 3072
  probe("qsa.o_proj", 640, 1536);        // 2560
  probe("router.gate", 128, 2560);       // 512
  probe("shared.gate_up", 320, 2560);    // 1280
  probe("shared.down", 640, 640);        // 2560
  probe("hc.down", 80, 10240);           // 320
  probe("hc.up", 2560, 80);              // 10240
  probe("hc.inject", 1, 10240);          // 4
  probe("lm_head.rank", 62080, 2560);    // 248320
  printf("shape_profile: tp4_rank0 (admissible I5 evidence, D050)\n");
  return 0;
}

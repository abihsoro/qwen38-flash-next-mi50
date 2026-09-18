// shape_compare.cu — full-width vs TP4-rank0 (quarter-width) kernel efficiency
// (D080 follow-up: correctness was recorded; the shard-shape QUESTION is
// whether kernels tuned at full width hold up at quarter width - a timing
// question). Runs gemv_f16_gfx906 at M=1 for both widths of each projection,
// reports GB/s and the quarter/full efficiency ratio. A ratio > 1 would flag a
// ranking inversion to investigate before M4.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "gemv_f16_gfx906.h"

static float frand() { return (float)(rand() % 100000) / 50000.0f - 1.0f; }

// name, N_full, N_rank0, K
struct Shape { const char* name; int nf; int nq; int k; };

void measure(const char* tag, const char* name, int N, int K) {
  std::vector<__half> w(N * K), x(K), yv(N);
  for (auto& v : w) v = __float2half(frand() * 0.02f);
  for (auto& v : x) v = __float2half(frand());
  __half *dw, *dx, *dy;
  hipMalloc(&dw, (size_t)N * K * 2);
  hipMalloc(&dx, K * 2);
  hipMalloc(&dy, N * 2);
  hipMemcpy(dw, w.data(), (size_t)N * K * 2, hipMemcpyHostToDevice);
  hipMemcpy(dx, x.data(), K * 2, hipMemcpyHostToDevice);
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
  double gbs = (double)N * K * 2 / (ms / IT * 1e-3) / 1e9;
  printf("%s %-14s N=%6d K=%6d  %7.1f us  %6.0f GB/s\n",
         tag, name, N, K, ms / IT * 1e3, gbs);
  fflush(stdout);
}

int main() {
  Shape shapes[] = {
      {"in_proj_qkv", 2560, 640, 2560}, {"in_proj_z", 1536, 384, 2560},
      {"out_proj", 2560, 640, 1536}, {"qsa.q_proj", 3072, 768, 2560},
      {"qsa.o_proj", 2560, 640, 1536}, {"router.gate", 512, 128, 2560},
      {"shared.gate_up", 1280, 320, 2560}, {"shared.down", 2560, 640, 640},
      {"hc.down", 320, 80, 10240}, {"hc.up", 10240, 2560, 80},
      {"hc.inject", 4, 1, 10240}, {"lm_head", 248320, 62080, 2560},
  };
  printf("=== full width (TP1 shapes) ===\n");
  for (auto& s : shapes) measure("full ", s.name, s.nf, s.k);
  printf("\n=== TP4 rank0 (quarter width) ===\n");
  for (auto& s : shapes) measure("rank0", s.name, s.nq, s.k);
  printf("\n(compare GB/s; a quarter/full ratio > 1 flags a ranking inversion)\n");
  return 0;
}

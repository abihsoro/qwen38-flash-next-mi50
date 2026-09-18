// nd3_reduce.cu — N-D3 inner-loop validation: the fixed-order fp32 reduce
// (wave-agnostic; the wave64-relevant part). Pre-fills a rank's staging slots
// with synthetic peer contributions and runs the reduce for each simulated
// rank, verifying BIT-IDENTICAL output vs the CPU fixed-order fp32 sum and
// across ranks. The concurrent announce/wait protocol is NOT exercised here
// (single-GPU cross-kernel uncached-atomic visibility is unreliable on this
// stack — DECISIONS D031); the protocol validates on the 4-card box (M0/M2)
// where its cross-PCIe semantics match the fork's environment.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const int N = 4096;
static const int WORLD = 4;
static const int NB = 8, NT = 256;
static float frand() { return (float)(rand() % 100000) / 50000.0f - 1.0f; }

__device__ __forceinline__ float ar_to_f(float x) { return x; }

// the fixed-order reduce loop, exactly as in rdna_ar_oneshot's step 3
__global__ void ar_reduce_only(const float* __restrict__ in,
                               float* __restrict__ out,
                               const float* __restrict__ stage, int rank,
                               int world, int n, long long max_elems) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int gstride = gridDim.x * blockDim.x;
  for (int i = gid; i < n; i += gstride) {
    float v = 0.f;
    for (int j = 0; j < world; j++)
      v += (j == rank) ? ar_to_f(in[i])
                       : ar_to_f(stage[(long long)j * max_elems + i]);
    out[i] = v;
  }
}

int main() {
  srand(42);
  std::vector<float> in(WORLD * N);
  for (int r = 0; r < WORLD; r++)
    for (int i = 0; i < N; i++) in[r * N + i] = frand() * 0.5f;
  // CPU reference: the all-reduce result is the fixed-order fp32 sum of ALL
  // ranks' contributions (identical for every rank)
  std::vector<float> ref(N);
  for (int i = 0; i < N; i++) {
    float v = 0.f;
    for (int j = 0; j < WORLD; j++) v += in[j * N + i];
    ref[i] = v;
  }
  float *din[WORLD], *dstage, *dout;
  hipMalloc(&dstage, (size_t)2 * WORLD * N * 4);
  for (int r = 0; r < WORLD; r++) {
    hipMalloc(&din[r], N * 4);
    hipMemcpy(din[r], in.data() + r * N, N * 4, hipMemcpyHostToDevice);
  }
  hipMalloc(&dout, N * 4);
  int mismatches = 0, cross = 0;
  std::vector<std::vector<float>> got(WORLD, std::vector<float>(N));
  for (int r = 0; r < WORLD; r++) {
    for (int s = 0; s < WORLD; s++) {
      if (s == r) continue;
      hipMemcpy(dstage + (long long)s * N, in.data() + s * N, N * 4,
                hipMemcpyHostToDevice);
    }
    ar_reduce_only<<<NB, NT>>>(din[r], dout, dstage, r, WORLD, N, (long long)N);
    hipDeviceSynchronize();
    hipMemcpy(got[r].data(), dout, N * 4, hipMemcpyDeviceToHost);
  }
  for (int r = 0; r < WORLD; r++)
    for (int i = 0; i < N; i++)
      if (got[r][i] != ref[i]) mismatches++;
  for (int i = 0; i < N; i++)
    for (int r = 1; r < WORLD; r++)
      if (got[r][i] != got[<bus>][i]) cross++;
  printf("N-D3 reduce loop W=%d: mismatches_vs_cpu=%d cross_rank_diff=%d %s\n",
         WORLD, mismatches, cross,
         (mismatches == 0 && cross == 0) ? "PASS" : "FAIL");
  return (mismatches == 0 && cross == 0) ? 0 : 1;
}

// n_e3_hw_queues.cu — N-E3: GPU_MAX_HW_QUEUES stream-overlap probe.
// Runs 4 concurrent copy streams and reports the batch time vs the
// sequential sum (overlap ratio) at the queue count set by the caller.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void copy_k(const float* __restrict__ a, float* __restrict__ b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) b[i] = a[i];
}

int main(int argc, char** argv) {
  const int NS = 4;
  const int n = 64 << 20;  // 256 MiB per stream
  float *a[NS], *b[NS];
  hipStream_t s[NS];
  for (int i = 0; i < NS; i++) {
    hipMalloc(&a[i], n * 4);
    hipMalloc(&b[i], n * 4);
    hipMemset(a[i], 1, n * 4);
    hipStreamCreate(&s[i]);
  }
  // sequential baseline
  hipEvent_t e0, e1;
  hipEventCreate(&e0);
  hipEventCreate(&e1);
  hipEventRecord(e0);
  for (int i = 0; i < NS; i++) copy_k<<<(n + 255) / 256, 256, 0, s[i]>>>(a[i], b[i], n);
  hipDeviceSynchronize();
  hipEventRecord(e1);
  hipEventSynchronize(e1);
  float ms = 0;
  hipEventElapsedTime(&ms, e0, e1);
  double seq_us = ms * 1000.0;
  // concurrent batch (warmup + reps)
  double best = 1e18;
  for (int r = 0; r < 10; r++) {
    hipEventRecord(e0);
    for (int i = 0; i < NS; i++) copy_k<<<(n + 255) / 256, 256, 0, s[i]>>>(a[i], b[i], n);
    hipEventRecord(e1);
    hipEventSynchronize(e1);
    hipEventElapsedTime(&ms, e0, e1);
    if (ms * 1000.0 < best) best = ms * 1000.0;
  }
  printf("N-E3 4-stream copy: sequential %.0f us, concurrent %.0f us, overlap x%.2f\n",
         seq_us, best, seq_us / best);
  return 0;
}

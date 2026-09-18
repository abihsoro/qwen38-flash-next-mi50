// n_e2_launch_latency.cu — N-E2: empty-kernel launch latency in-guest (HIP).
// Two measures: back-to-back launches on one stream (issue cost) and full
// round-trip with a synchronize per launch (invocation + completion).
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void empty_k() {}

int main() {
  const int N = 10000;
  hipEvent_t e0, e1;
  hipEventCreate(&e0);
  hipEventCreate(&e1);
  // warmup
  for (int i = 0; i < 100; i++) empty_k<<<1, 1>>>();
  hipDeviceSynchronize();
  // back-to-back (single stream, issue-bound)
  hipEventRecord(e0);
  for (int i = 0; i < N; i++) empty_k<<<1, 1>>>();
  hipEventRecord(e1);
  hipEventSynchronize(e1);
  float ms = 0;
  hipEventElapsedTime(&ms, e0, e1);
  double b2b_us = ms * 1000.0 / N;
  // round-trip (sync per launch)
  hipEventRecord(e0);
  for (int i = 0; i < N; i++) {
    empty_k<<<1, 1>>>();
    hipDeviceSynchronize();
  }
  hipEventRecord(e1);
  hipEventSynchronize(e1);
  hipEventElapsedTime(&ms, e0, e1);
  double rt_us = ms * 1000.0 / N;
  printf("N-E2 HIP empty-kernel: back-to-back %.2f us/launch, round-trip %.2f us\n",
         b2b_us, rt_us);
  return 0;
}

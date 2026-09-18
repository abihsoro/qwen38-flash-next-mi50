// ne5_store_load.cu — N-E5 single-GPU probe: the store-vs-load asymmetry the
// M2 flag protocol depends on, measured LOCALLY on uncached device memory
// (the fork's cross-PCIe peer asymmetry 14.3 vs 5.7 GB/s is a fabric property
// and validates on the 4-card box at M0/M2).
//
// Probes (uncached memory via hipExtMallocWithFlags UNCACHED, like the staging):
//   1. announce-store cost: N system-scope atomic stores per thread
//   2. poll-load cost:      N system-scope atomic loads per thread (the wait loop)
//   3. poll-with-sleep:     the fork's s_sleep(8) backoff loop cadence
// Reports the per-op latency of each so the protocol's latency assumptions
// (announce ~ns, poll ~ns/op) can be checked on gfx906.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

__global__ void k_store(int* __restrict__ dst, long long n, int* sink) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long stride = (long long)gridDim.x * blockDim.x;
  long long cnt = 0;
  for (; i < n; i += stride) {
    __hip_atomic_store(&dst[<bus>], (int)i, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    cnt++;
  }
  if (cnt == 1LL << 62) *sink = 1;
}

__global__ void k_load(const int* __restrict__ src, long long n, int* sink) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long stride = (long long)gridDim.x * blockDim.x;
  int acc = 0;
  for (; i < n; i += stride) {
    acc += __hip_atomic_load(&src[<bus>], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
  }
  *sink = acc;
}

__global__ void k_poll_sleep(const int* __restrict__ src, long long spins, int* sink) {
  // emulate the fork's wait loop: load + s_sleep(8) backoff cadence
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long stride = (long long)gridDim.x * blockDim.x;
  int acc = 0;
  for (; i < spins; i += stride) {
    acc += __hip_atomic_load(&src[<bus>], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
    __builtin_amdgcn_s_sleep(8);
  }
  *sink = acc;
}

template <typename F>
double timeit(const char* name, long long ops, dim3 grid, dim3 block, F&& launch) {
  hipEvent_t e0, e1;
  hipEventCreate(&e0); hipEventCreate(&e1);
  for (int w = 0; w < 3; w++) launch();
  hipDeviceSynchronize();
  double best = 1e18;
  for (int r = 0; r < 5; r++) {
    hipEventRecord(e0, 0);
    launch();
    hipEventRecord(e1, 0);
    hipEventSynchronize(e1);
    float ms;
    hipEventElapsedTime(&ms, e0, e1);
    double per_op_us = ms * 1e3 / (double)ops;
    if (per_op_us < best) best = per_op_us;
  }
  printf("%-22s %.4f us/op  (%.0f Mop/s)\n", name, best, 1.0 / best);
  return best;
}

int main() {
  int* u;  // uncached device memory
  hipExtMallocWithFlags((void**)&u, 4096, hipDeviceMallocUncached);
  int* sink;
  hipMalloc(&sink, 4);
  hipMemset(u, 0, 4096);
  hipDeviceSynchronize();

  // serialize: 1 thread doing N ops in a row (the flag protocols are serial)
  const long long N = 200000;
  timeit("store uncached sys", N, dim3(1), dim3(1),
         [&] { k_store<<<1, 1>>>(u, N, sink); });
  timeit("load  uncached sys", N, dim3(1), dim3(1),
         [&] { k_load<<<1, 1>>>(u, N, sink); });
  timeit("poll+sleep(8)", N, dim3(1), dim3(1),
         [&] { k_poll_sleep<<<1, 1>>>(u, N, sink); });
  // throughput view: many threads hammering
  timeit("store sys 60x256", N * 60 * 256, dim3(60), dim3(256),
         [&] { k_store<<<60, 256>>>(u, N * 60 * 256, sink); });
  timeit("load  sys 60x256", N * 60 * 256, dim3(60), dim3(256),
         [&] { k_load<<<60, 256>>>(u, N * 60 * 256, sink); });
  printf("done\n");
  return 0;
}

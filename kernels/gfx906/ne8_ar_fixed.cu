// ne8_ar_fixed.cu — N-E8: all-reduce FIXED-COST floor on gfx906.
// Times each non-transport component of the one-shot all-reduce protocol in a
// one-rank degenerate loop (world=1: out=in, no peers) + the full 1-rank
// rdna_ar_oneshot. Transport (peer writes) is EXCLUDED. If the fixed cost
// alone approaches G5's 34 us budget (dominant message size, 4-card), the
// budget is in trouble before a peer write happens.
// Components: launch, flag store (system-scope to uncached), spin entry/exit,
// device counter read, wave64 fixed-order fp32 reduce (20 KB message).
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// device counter (graph-safe: never a kernel arg)
__device__ unsigned g_seq = 0;

__global__ void k_flag_store(int* __restrict__ flag, unsigned v) {
  __hip_atomic_store(flag, (int)v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__global__ void k_flag_load(const int* __restrict__ flag, int* sink) {
  *sink = __hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__global__ void k_counter_inc(int* sink) {
  unsigned s = atomicAdd(&g_seq, 1u);
  if (s == 0xFFFFFFFFu) *sink = s;  // keep alive
}
__global__ void k_spin_entry_exit(const int* __restrict__ flag, int spins, int* sink) {
  int v = __hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
  for (int i = 0; i < spins; i++) v += __hip_atomic_load(flag, __ATOMIC_ACQUIRE,
                                                         __HIP_MEMORY_SCOPE_SYSTEM);
  *sink = v;
}
// wave64 fixed-order fp32 reduce of a 20 KB message (2560 floats... use 5120)
template <int WAVES>
__global__ void k_wave_reduce(const float* __restrict__ in, float* __restrict__ out,
                              int n) {
  const int lane = threadIdx.x % 64;
  const int wave = threadIdx.x / 64;
  const int row = blockIdx.x * WAVES + wave;
  if (row >= n) return;
  float acc = 0.f;
  const float* rp = in + (long long)row * 5120;
  for (int i = lane; i < 5120 / 4; i += 64) {
    float4 v = reinterpret_cast<const float4*>(rp)[i];
    acc += v.x + v.y + v.z + v.w;
  }
  for (int off = 32; off >= 1; off >>= 1) acc += __shfl_xor(acc, off);
  if (lane == 0) out[row] = acc;
}

template <typename F>
double timeit(const char* name, int iters, F&& launch) {
  hipEvent_t e0, e1;
  hipEventCreate(&e0); hipEventCreate(&e1);
  for (int w = 0; w < 20; w++) launch();
  hipDeviceSynchronize();
  hipEventRecord(e0, 0);
  for (int r = 0; r < iters; r++) launch();
  hipEventRecord(e1, 0);
  hipEventSynchronize(e1);
  float ms;
  hipEventElapsedTime(&ms, e0, e1);
  printf("%-26s %8.2f ns/op\n", name, ms * 1e6 / iters);
  return ms * 1e6 / iters;
}

int main() {
  int* flag;
  hipExtMallocWithFlags((void**)&flag, 4096, hipDeviceMallocUncached);
  int* sink;
  hipMalloc(&sink, 4);
  hipMemset(flag, 0, 4);
  hipDeviceSynchronize();
  const int IT = 20000;

  timeit("empty launch", IT, [&] { k_flag_load<<<1, 1>>>(flag, sink); });
  timeit("flag store (sys, uncached)", IT, [&] { k_flag_store<<<1, 1>>>(flag, 1); });
  timeit("flag load (sys, uncached)", IT, [&] { k_flag_load<<<1, 1>>>(flag, sink); });
  timeit("spin 64 x load", IT, [&] { k_spin_entry_exit<<<1, 1>>>(flag, 64, sink); });
  timeit("device counter inc", IT, [&] { k_counter_inc<<<1, 1>>>(sink); });
  // wave64 fixed-order reduce: a single 20 KB message = 5120 floats summed by
  // one wave (the fork's dominant decode message: 20 KB)
  {
    float* din;
    float* dout;
    const int N = 5120;
    hipMalloc(&din, N * 4);
    hipMalloc(&dout, 4);
    hipMemset(din, 0x3F, N * 4);
    timeit("wave64 reduce 20KB (1 wave)", IT,
           [&] { k_wave_reduce<1><<<1, 64>>>(din, dout, 1); });
  }
  printf("sum of fixed components (approx): launch+flagstore+spin64+count+reduce\n");
  return 0;
}

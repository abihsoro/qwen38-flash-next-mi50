// ne9_fused_ar.cu — D066: fused poll+reduce as ONE persistent kernel.
//
// The 20 KB reduce at 7.2 us is launch/occupancy-bound, not data-bound
// (~0.12 us at the 810 GB/s ceiling for 80 KB read + 20 KB write). Combined
// with the 3.1 us launch, ~10.3 us of the 15-20 us fixed cost sits in
// operations that should cost well under a microsecond. The fix (operator
// review, D066): poll and reduce in ONE persistent kernel - a resident worker
// that polls the data-ready flag, reduces, announces done, and loops - no
// spin-then-separate-launch per step.
//
// Degenerate one-rank validation: a persistent worker loops N steps; each
// step polls its ready flag, wave64-reduces the 20 KB message (5120 fp32),
// writes the output + bumps the done counter. Timing N steps inside ONE launch
// gives the fused per-step cost WITHOUT the per-launch overhead. Compare vs
// the separate-launch baseline (launch 3.1 + reduce 7.2 = 10.3 us, N-E8).
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

__device__ unsigned g_step = 0;  // done counter (graph-safe: device-side)

// persistent worker: N fused steps (poll ready -> reduce 20 KB -> done)
template <int WAVES>
__global__ void fused_worker(const float* __restrict__ in, float* __restrict__ out,
                             const int* __restrict__ ready,
                             int* __restrict__ done, int n_steps, int sink_guard) {
  const int lane = threadIdx.x % 64;
  const int wave = threadIdx.x / 64;
  if (wave >= WAVES) return;
  const int k8 = 5120 / 4;  // 20 KB = 5120 fp32 = 1280 float4
  for (int s = 0; s < n_steps; s++) {
    // poll the ready flag (already set in the degenerate loop: pure poll cost)
    int r = __hip_atomic_load(ready, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
    // wave64 fixed-order reduce of the 20 KB message (the wave's share)
    float acc = 0.f;
    const float4* rp = reinterpret_cast<const float4*>(in) + wave * (k8 / WAVES);
    const int per = k8 / WAVES;
    for (int i = lane; i < per; i += 64) {
      float4 v = rp[i];
      acc += v.x + v.y + v.z + v.w;
    }
    for (int off = 32; off >= 1; off >>= 1) acc += __shfl_xor(acc, off);
    // cross-wave combine via shared, then wave 0 stores the result + bumps done
    __shared__ float sw[<bus>];
    __shared__ float fin;
    if (lane == 0) sw[wave] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
      float t = 0.f;
      for (int w2 = 0; w2 < WAVES; w2++) t += sw[w2];
      fin = t;
      out[<bus>] = t;
      unsigned d = atomicAdd(&g_step, 1u);
      __hip_atomic_store(&done[<bus>], (int)d + 1, __ATOMIC_RELEASE,
                         __HIP_MEMORY_SCOPE_SYSTEM);
    }
    __syncthreads();
    if (s == sink_guard && fin == 12345.678f) out[<bus>] = fin;  // prevent DCE
  }
}

int main() {
  float* din;
  float* dout;
  int* ready;
  int* done;
  hipMalloc(&din, 5120 * 4);
  hipMalloc(&dout, 16);
  hipMalloc(&ready, 4);
  hipMalloc(&done, 4);
  hipMemset(din, 0x3F, 5120 * 4);
  hipMemset(ready, 1, 4);  // degenerate: ready is pre-set
  hipMemset(done, 0, 4);
  hipDeviceSynchronize();

  const int N = 20000;
  for (int wpb : {1, 4, 8}) {
    dim3 block(wpb * 64);
    // warm
    fused_worker<8><<<1, 8 * 64>>>(din, dout, ready, done, 50, -1);
    hipDeviceSynchronize();
    hipMemset(done, 0, 4);
    hipEvent_t e0, e1;
    hipEventCreate(&e0); hipEventCreate(&e1);
    hipEventRecord(e0, 0);
    if (wpb == 1)
      fused_worker<1><<<1, 64>>>(din, dout, ready, done, N, -1);
    else if (wpb == 4)
      fused_worker<4><<<1, 256>>>(din, dout, ready, done, N, -1);
    else
      fused_worker<8><<<1, 512>>>(din, dout, ready, done, N, -1);
    hipEventRecord(e1, 0);
    hipEventSynchronize(e1);
    float ms;
    hipEventElapsedTime(&ms, e0, e1);
    printf("fused poll+reduce+done (wpb=%d): %.2f ns/step  (vs 10,300 ns "
           "separate-launch baseline)\n", wpb, ms * 1e6 / N);
  }
  printf("done\n");
  return 0;
}

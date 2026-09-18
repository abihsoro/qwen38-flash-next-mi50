// rdna_ar_oneshot_gfx906.cuh — N-D3: wave64 port of the fork's push-based
// one-shot all-reduce (leapdragon/vllm-rdna2-qwen csrc/rocm/rdna_allreduce.cuh,
// T44). The kernel is WAVE-AGNOSTIC: element-parallel grid-stride loops, no
// shuffle ladders or wave reductions; the reduce is a fixed-rank-order fp32
// sum so every rank produces bit-identical output. Changes for gfx906: none
// structural — RDNA_AR_POLL_PAUSE (s_sleep(8)) idles a wave ~64 clocks; on
// wave64 the same call idles 64 lanes for the same wave-time (polling rate
// doubles in lanes; backoff unchanged, PORT-MAP note). M2 design constraints
// honored: device-derived sequence (T4), own-memory flags with system-scope
// stores (T3), no GPU-side waits on peer memory, bounded spins with sticky
// abort. Validated on a single card with a W=4-simulated harness (nd3).
#ifndef KERNELS_GFX906_RDNA_AR_ONESHOT_GFX906_CUH
#define KERNELS_GFX906_RDNA_AR_ONESHOT_GFX906_CUH

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define RDNA_AR_MAX_WORLD 8
#define RDNA_AR_FLAG_PAGE 4096
#define RDNA_AR_SPIN_CAP 2000000ull

__device__ __forceinline__ float rdna_ar_to_f(float x) { return x; }
__device__ __forceinline__ float rdna_ar_to_f(__half x) { return __half2float(x); }
__device__ __forceinline__ void rdna_ar_from_f(float& d, float v) { d = v; }
__device__ __forceinline__ void rdna_ar_from_f(__half& d, float v) { d = __float2half(v); }

struct RdnaArPeers {
  void* stage[RDNA_AR_MAX_WORLD];
  int* flags[RDNA_AR_MAX_WORLD];
};

#define RDNA_AR_POLL_PAUSE() __builtin_amdgcn_s_sleep(8)

template <typename T>
__global__ void rdna_ar_oneshot(const T* __restrict__ in, T* __restrict__ out,
                                RdnaArPeers peers,
                                unsigned int* arrive,
                                int* seqbuf,
                                unsigned* timeout,
                                int rank, int world, int n, long long max_elems,
                                int nblocks, int pace) {
  __shared__ int s_seq;
  __shared__ int s_abort;
  const int t = threadIdx.x, nt = blockDim.x, b = blockIdx.x;
  if (t == 0) {
    s_seq = __hip_atomic_load(seqbuf, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) + 1;
    s_abort = 0;
  }
  __syncthreads();
  const int seq = s_seq;
  const int p = seq & 1;
  const int gid = b * nt + t, gstride = nblocks * nt;

  for (int k = 1; k < world; k++) {
    const int j = (rank + k) % world;
    T* dst = reinterpret_cast<T*>(peers.stage[j]) + ((long long)p * world + rank) * max_elems;
    for (int i = gid; i < n; i += gstride) {
      dst[i] = in[i];
      for (int q = 0; q < pace; q++) __builtin_amdgcn_s_sleep(1);
    }
  }
  __syncthreads();

  if (t == 0) {
    __threadfence_system();
    atomicAdd(&arrive[p], 1u);
    if (b == 0) {
      unsigned long long s = 0;
      while (__hip_atomic_load(&arrive[p], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) <
             (unsigned)nblocks) {
        RDNA_AR_POLL_PAUSE();
        if (++s > RDNA_AR_SPIN_CAP) { *timeout = 1u; s_abort = 1; break; }
      }
      if (!s_abort) {
        arrive[1 - p] = 0u;
        __hip_atomic_store(seqbuf, seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        for (int j = 0; j < world; j++)
          __hip_atomic_store(&peers.flags[j][rank], seq, __ATOMIC_RELEASE,
                             __HIP_MEMORY_SCOPE_SYSTEM);
      }
    }
    if (!s_abort) {
      int* myflags = peers.flags[rank];
      for (int j = 0; j < world && !s_abort; j++) {
        if (j == rank) continue;
        unsigned long long s = 0;
        while (__hip_atomic_load(&myflags[j], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < seq) {
          RDNA_AR_POLL_PAUSE();
          if (++s > RDNA_AR_SPIN_CAP) { *timeout = 1u; s_abort = 1; break; }
        }
      }
    }
  }
  __syncthreads();
  if (s_abort) return;

  const T* mine = reinterpret_cast<const T*>(peers.stage[rank]) + ((long long)p * world) * max_elems;
  for (int i = gid; i < n; i += gstride) {
    float v = 0.f;
    for (int j = 0; j < world; j++)
      v += (j == rank) ? rdna_ar_to_f(in[i]) : rdna_ar_to_f(mine[(long long)j * max_elems + i]);
    rdna_ar_from_f(out[i], v);
  }
}

#endif  // KERNELS_GFX906_RDNA_AR_ONESHOT_GFX906_CUH

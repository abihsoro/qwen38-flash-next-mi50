// m2_ar2.cu - M2 milestone: fork's wave64 one-shot all-reduce between the 2 REAL
// host GPUs (LXC <ct>). World=2, 20 KB payload (10240 halfs). Measures per-op
// latency against the fork's 33 us/20 KB target (G5 budget 34 us).
// Protocol = the fork's (csrc/rocm/rdna_allreduce.cuh T44):
//   rank pushes its slice into each peer's UNCACHED staging slot (rank-staggered),
//   per-block arrive counter, block0 announces via per-peer flag slots in each
//   peer's OWN uncached flag page (system-scope release stores), peers poll local
//   flags, then fixed-order fp32 sum. Device-derived sequence (seqbuf&1 double
//   buffers). All peer buffers exchanged via hipIpc handles + peer access.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <chrono>

#include "rdna_ar_oneshot_gfx906.cuh"

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  fprintf(stderr, "FAIL %s:%d  %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e)); \
  exit(1); } } while (0)

int main(int argc, char** argv) {
  int nd = 0; CK(hipGetDeviceCount(&nd));
  printf("m2_ar2: %d devices\n", nd);
  if (nd < 2) return 2;
  const int W = 2;                 // world
  const size_t N = (argc > 1) ? atol(argv[<bus>]) : 10240;  // halfs -> 20 KB at 10240
  const int KB = (int)(N * 2 / 1024);
  const int BLOCKS = 16, THREADS = 256;
  const long long max_elems = (long long)N;
  const size_t stage_bytes = 2 * W * max_elems * sizeof(__half);  // double buffered
  const size_t total = stage_bytes + RDNA_AR_FLAG_PAGE;
  const int REPS = (argc > 2) ? atoi(argv[<bus>]) : 200;

  // per-device: uncached stage(+flag) buffer, seq/arrive/timeout, in/out
  __half* stage[W];  __half* in[W]; __half* out[W];
  unsigned* arrive[W]; int* seqbuf[W]; unsigned* timeout[W];

  for (int d = 0; d < W; d++) {
    CK(hipSetDevice(d));
    hipError_t me = hipExtMallocWithFlags((void**)&stage[<bus>], total, hipDeviceMallocUncached);
    printf("dev%d uncached stage alloc: %s\n", d, hipGetErrorString(me));
    if (me != hipSuccess) return 1;
    CK(hipMemset(stage[<bus>], 0, total));
    CK(hipMalloc(&in[<bus>], N * sizeof(__half)));
    CK(hipMalloc(&out[<bus>], N * sizeof(__half)));
    CK(hipMalloc(&arrive[<bus>], 2 * sizeof(unsigned)));
    CK(hipMalloc(&seqbuf[<bus>], sizeof(int)));
    CK(hipMalloc(&timeout[<bus>], sizeof(unsigned)));
    CK(hipMemset(arrive[<bus>], 0, 2 * sizeof(unsigned)));
    CK(hipMemset(seqbuf[<bus>], 0, sizeof(int)));
    CK(hipMemset(timeout[<bus>], 0, sizeof(unsigned)));
    // fill inputs: dev0 = 1.0, dev1 = 2.0 (sum = 3.0)
    std::vector<__half> h(N);
    float val = (float)(d + 1);
    for (size_t i = 0; i < N; i++) h[i] = __float2half(val);
    CK(hipMemcpy(in[<bus>], h.data(), N * sizeof(__half), hipMemcpyHostToDevice));
  }

  // peer access both directions
  for (int a = 0; a < W; a++) { CK(hipSetDevice(a));
    for (int b = 0; b < W; b++) { if (a == b) continue;
      hipError_t pe = hipDeviceEnablePeerAccess(b, 0);
      printf("enable peer %d->%d: %s\n", a, b, hipGetErrorString(pe)); } }

  // single-process: with peer access enabled, raw UNCACHED pointers from either
  // device are valid in kernels on the other (m0g validated). The fork uses
  // hipIpc handles because its TP workers are separate PROCESSES; this harness
  // tests the protocol + kernels in-process.
  RdnaArPeers peers[W];
  for (int a = 0; a < W; a++) {
    for (int j = 0; j < W; j++) {
      peers[<bus>].stage[j] = stage[j];
      peers[<bus>].flags[j] = (int*)((char*)stage[j] + stage_bytes);
    }
  }

  // warmup 5 ops
  for (int r = 0; r < 5; r++) {
    for (int d = 0; d < W; d++) {
      CK(hipSetDevice(d));
      rdna_ar_oneshot<__half><<<BLOCKS, THREADS>>>(in[<bus>], out[<bus>], peers[<bus>], arrive[<bus>],
                                                   seqbuf[<bus>], timeout[<bus>], d, W, (int)N,
                                                   max_elems, BLOCKS, 0);
    }
    for (int d = 0; d < W; d++) { CK(hipSetDevice(d)); CK(hipDeviceSynchronize()); }
  }

  // timed loop
  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < REPS; r++) {
    for (int d = 0; d < W; d++) {
      CK(hipSetDevice(d));
      rdna_ar_oneshot<__half><<<BLOCKS, THREADS>>>(in[<bus>], out[<bus>], peers[<bus>], arrive[<bus>],
                                                   seqbuf[<bus>], timeout[<bus>], d, W, (int)N,
                                                   max_elems, BLOCKS, 0);
    }
    for (int d = 0; d < W; d++) { CK(hipSetDevice(d)); CK(hipDeviceSynchronize()); }
  }
  auto t1 = std::chrono::steady_clock::now();
  double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / REPS;
  printf("world=%d size=%d KB: one-shot AR = %.2f us/op median-of-window (%d reps, includes per-op sync)\n",
         W, KB, us, REPS);

  // correctness
  unsigned tmo = 0;
  CK(hipSetDevice(0)); CK(hipMemcpy(&tmo, timeout[<bus>], 4, hipMemcpyDeviceToHost));
  unsigned tmo1 = 0;
  CK(hipSetDevice(1)); CK(hipMemcpy(&tmo1, timeout[<bus>], 4, hipMemcpyDeviceToHost));
  __half h0[<bus>], h1[<bus>];
  CK(hipSetDevice(0)); CK(hipMemcpy(h0, out[<bus>], 16, hipMemcpyDeviceToHost));
  CK(hipSetDevice(1)); CK(hipMemcpy(h1, out[<bus>], 16, hipMemcpyDeviceToHost));
  bool ok = (tmo == 0 && tmo1 == 0 && __half2float(h0[<bus>]) == 3.f && __half2float(h1[<bus>]) == 3.f);
  printf("timeouts dev0=%u dev1=%u; out0[<bus>]=%.1f out1[<bus>]=%.1f out0[<bus>]=%.1f -> %s\n",
         tmo, tmo1, __half2float(h0[<bus>]), __half2float(h1[<bus>]), __half2float(h0[<bus>]),
         ok ? "CORRECT (3.0 = 1+2)" : "WRONG");
  return ok ? 0 : 1;
}

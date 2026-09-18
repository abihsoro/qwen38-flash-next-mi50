// m2_ar4.cu — Phase 2 / M2: fork's wave64 one-shot all-reduce across the FOUR real
// host GPUs (world=4), on the <host> PEX880xx fabric.
//
// Same protocol and kernel as m2_ar2.cu (world=2, validated D117: 30.7 us/op @ 20 KB).
// rdna_ar_oneshot() is already world-generic (RDNA_AR_MAX_WORLD 8, `int world`), so
// this file is the W=4 *harness*: buffer sizing, peer map, per-rank input fill,
// correctness across all ranks, and per-repeat timing with median + IQR (I5).
//
// Protocol (fork T44): rank pushes its slice into each peer's UNCACHED staging slot
// (rank-staggered), per-block arrive counter, block0 announces via per-peer flag slots
// in each peer's OWN uncached flag page (system-scope release stores), peers poll local
// flags, then a fixed-rank-order fp32 sum => bit-identical output on every rank.
//
// NOTE: in-process harness (peer access + raw uncached pointers). The fork uses hipIpc
// handles because its TP workers are separate processes; the multi-process wiring (and
// the hipIpcOpenMemHandle asymmetry, D117) is a separate step.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>

#include "rdna_ar_oneshot_gfx906.cuh"

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  fprintf(stderr, "FAIL %s:%d  %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e)); \
  exit(1); } } while (0)

static double median(std::vector<double> v) {
  if (v.empty()) return -1.0;
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static double iqr(std::vector<double> v) {
  if (v.size() < 4) return -1.0;
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  auto q = [&](double p) {
    double idx = p * (double)(n - 1);
    size_t lo = (size_t)idx;
    double f = idx - (double)lo;
    double hi = (lo + 1 < n) ? v[lo + 1] : v[lo];
    return v[lo] + f * (hi - v[lo]);
  };
  return q(0.75) - q(0.25);
}

int main(int argc, char** argv) {
  int nd = 0;
  CK(hipGetDeviceCount(&nd));
  const int W = 4;
  printf("m2_ar4: devices=%d, world=%d\n", nd, W);
  if (nd < W) { printf("need %d devices, have %d - abort\n", W, nd); return 2; }

  const size_t N = (argc > 1) ? (size_t)atol(argv[<bus>]) : 10240;  // halfs (10240 = 20 KB)
  const int KB = (int)(N * 2 / 1024);
  const int BLOCKS = 16, THREADS = 256;
  const long long max_elems = (long long)N;
  const size_t stage_bytes = 2 * W * max_elems * sizeof(__half);  // double buffered
  const size_t total = stage_bytes + RDNA_AR_FLAG_PAGE;
  const int REPS = (argc > 2) ? atoi(argv[<bus>]) : 200;

  __half* stage[RDNA_AR_MAX_WORLD];
  __half* in[RDNA_AR_MAX_WORLD];
  __half* out[RDNA_AR_MAX_WORLD];
  unsigned* arrive[RDNA_AR_MAX_WORLD];
  int* seqbuf[RDNA_AR_MAX_WORLD];
  unsigned* timeout[RDNA_AR_MAX_WORLD];

  for (int d = 0; d < W; d++) {
    CK(hipSetDevice(d));
    hipError_t me = hipExtMallocWithFlags((void**)&stage[<bus>], total, hipDeviceMallocUncached);
    printf("dev%d uncached stage alloc (%zu bytes): %s\n", d, total, hipGetErrorString(me));
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
    // dev d holds value (d+1); the AR sums to 1+2+3+4 = 10
    std::vector<__half> h(N);
    float val = (float)(d + 1);
    for (size_t i = 0; i < N; i++) h[i] = __float2half(val);
    CK(hipMemcpy(in[<bus>], h.data(), N * sizeof(__half), hipMemcpyHostToDevice));
  }

  for (int a = 0; a < W; a++) {
    CK(hipSetDevice(a));
    for (int b = 0; b < W; b++) {
      if (a == b) continue;
      hipError_t pe = hipDeviceEnablePeerAccess(b, 0);
      if (pe != hipSuccess && pe != hipErrorPeerAccessAlreadyEnabled)
        printf("enable peer %d->%d: %s\n", a, b, hipGetErrorString(pe));
    }
  }

  RdnaArPeers peers[RDNA_AR_MAX_WORLD];
  for (int a = 0; a < W; a++)
    for (int j = 0; j < W; j++) {
      peers[<bus>].stage[j] = stage[j];
      peers[<bus>].flags[j] = (int*)((char*)stage[j] + stage_bytes);
    }

  // warmup
  for (int r = 0; r < 5; r++) {
    for (int d = 0; d < W; d++) {
      CK(hipSetDevice(d));
      rdna_ar_oneshot<__half><<<BLOCKS, THREADS>>>(in[<bus>], out[<bus>], peers[<bus>], arrive[<bus>],
                                                   seqbuf[<bus>], timeout[<bus>], d, W, (int)N,
                                                   max_elems, BLOCKS, 0);
    }
    for (int d = 0; d < W; d++) { CK(hipSetDevice(d)); CK(hipDeviceSynchronize()); }
  }

  // timed: per-repeat so we can report median + IQR (I5)
  std::vector<double> per_rep;
  for (int r = 0; r < REPS; r++) {
    for (int d = 0; d < W; d++) { CK(hipSetDevice(d)); CK(hipDeviceSynchronize()); }
    auto t0 = std::chrono::steady_clock::now();
    for (int d = 0; d < W; d++) {
      CK(hipSetDevice(d));
      rdna_ar_oneshot<__half><<<BLOCKS, THREADS>>>(in[<bus>], out[<bus>], peers[<bus>], arrive[<bus>],
                                                   seqbuf[<bus>], timeout[<bus>], d, W, (int)N,
                                                   max_elems, BLOCKS, 0);
    }
    for (int d = 0; d < W; d++) { CK(hipSetDevice(d)); CK(hipDeviceSynchronize()); }
    auto t1 = std::chrono::steady_clock::now();
    per_rep.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  double med = median(per_rep), iq = iqr(per_rep);
  printf("world=%d size=%d KB: one-shot AR median=%.2f us/op  iqr=%.2f  min=%.2f  (%d reps, incl per-op sync)\n",
         W, KB, med, iq,
         per_rep.empty() ? -1.0 : *std::min_element(per_rep.begin(), per_rep.end()), REPS);

  // -------- correctness: every rank must hold the full sum, bit-identically --------
  const float expect = (float)(W * (W + 1) / 2);  // 10.0 for W=4
  bool all_ok = true;
  std::vector<__half> ref;
  for (int d = 0; d < W; d++) {
    unsigned tmo = 0;
    CK(hipSetDevice(d));
    CK(hipMemcpy(&tmo, timeout[<bus>], sizeof(unsigned), hipMemcpyDeviceToHost));
    std::vector<__half> h(N);
    CK(hipMemcpy(h.data(), out[<bus>], N * sizeof(__half), hipMemcpyDeviceToHost));
    bool val_ok = true;
    for (size_t i = 0; i < N; i++)
      if (__half2float(h[i]) != expect) { val_ok = false; break; }
    bool ident = true;
    if (d == 0) ref = h;
    else for (size_t i = 0; i < N; i++) if (h[i] != ref[i]) { ident = false; break; }
    printf("  dev%d: timeout=%u  all_elems==%.1f: %s  bit-identical_to_dev0: %s\n",
           d, tmo, expect, val_ok ? "YES" : "NO", ident ? "YES" : "NO");
    all_ok = all_ok && val_ok && (tmo == 0) && ident;
  }

  printf("RESULT world=%d %dKB median=%.2f us  %s\n", W, KB, med,
         all_ok ? "CORRECT" : "WRONG");
  return all_ok ? 0 : 1;
}

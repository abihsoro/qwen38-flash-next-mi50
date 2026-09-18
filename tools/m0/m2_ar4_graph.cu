// m2_ar4_graph.cu — Phase 2: custom wave64 one-shot all-reduce at world=4 under HIP GRAPH REPLAY.
//
// WHY THIS EXISTS (TP4_BRINGUP_PLAN Phase 2 + "Graph replay tests"):
//   "Any custom op or kernel used under graphs must pass eager AND graph replay checks before
//    serving." rdna_ar_oneshot was only ever validated EAGER (D117 world=2, D141 world=4).
//   The TP4 serve runs with cudagraph_mode=full, so if this kernel is ever wired in it WILL be
//   captured and replayed - and replay is a different correctness regime from eager launch.
//
// THE SPECIFIC RISK: the kernel derives its sequence number DEVICE-SIDE from `seqbuf`
//   (s_seq = atomic_load(seqbuf) + 1;  p = seq & 1;  ... atomic_store(seqbuf, seq))
// because a HOST-derived counter cannot advance across a graph replay. The double-buffered
// staging then alternates on `seq & 1`. If the device-side derivation were wrong, replay would
// either (a) reuse the same buffer and race against the previous iteration's readers, or
// (b) fail to advance at all - neither of which eager execution would necessarily reveal.
//
// WHAT IS CHECKED
//   1. eager:    every rep's output == the exact sum, and bit-identical across all 4 ranks
//   2. capture:  the kernel is captured into a per-device HIP graph and instantiated
//   3. replay:   EVERY replayed iteration is re-verified (not just the first), because a
//                stale-sequence bug would corrupt only later iterations
//   4. seqbuf:   the device-side sequence actually advanced by (#eager + #replays)
//   5. timeouts: the kernel's own bounded-spin abort flag stays 0 throughout
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

static double median(std::vector<double> v) {
  if (v.empty()) return -1.0;
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

int main(int argc, char** argv) {
  int nd = 0;
  CK(hipGetDeviceCount(&nd));
  const int W = 4;
  printf("m2_ar4_graph: devices=%d world=%d\n", nd, W);
  if (nd < W) { printf("need %d devices\n", W); return 2; }

  const size_t N = (argc > 1) ? (size_t)atol(argv[<bus>]) : 10240;   // halfs (10240 = 20 KB)
  const int EAGER_REPS = (argc > 2) ? atoi(argv[<bus>]) : 100;
  const int GRAPH_REPS = (argc > 3) ? atoi(argv[<bus>]) : 500;
  const int BLOCKS = 16, THREADS = 256;
  const long long max_elems = (long long)N;
  const size_t stage_bytes = 2 * W * max_elems * sizeof(__half);
  const size_t total = stage_bytes + RDNA_AR_FLAG_PAGE;
  const int KB = (int)(N * 2 / 1024);

  __half* stage[RDNA_AR_MAX_WORLD];
  __half* in[RDNA_AR_MAX_WORLD];
  __half* out[RDNA_AR_MAX_WORLD];
  unsigned* arrive[RDNA_AR_MAX_WORLD];
  int* seqbuf[RDNA_AR_MAX_WORLD];
  unsigned* timeout[RDNA_AR_MAX_WORLD];
  hipStream_t stream[RDNA_AR_MAX_WORLD];
  hipGraph_t graph[RDNA_AR_MAX_WORLD];
  hipGraphExec_t gexec[RDNA_AR_MAX_WORLD];

  for (int d = 0; d < W; d++) {
    CK(hipSetDevice(d));
    hipError_t me = hipExtMallocWithFlags((void**)&stage[<bus>], total, hipDeviceMallocUncached);
    if (me != hipSuccess) { printf("dev%d uncached alloc: %s\n", d, hipGetErrorString(me)); return 1; }
    CK(hipMemset(stage[<bus>], 0, total));
    CK(hipMalloc(&in[<bus>], N * sizeof(__half)));
    CK(hipMalloc(&out[<bus>], N * sizeof(__half)));
    CK(hipMalloc(&arrive[<bus>], 2 * sizeof(unsigned)));
    CK(hipMalloc(&seqbuf[<bus>], sizeof(int)));
    CK(hipMalloc(&timeout[<bus>], sizeof(unsigned)));
    CK(hipMemset(arrive[<bus>], 0, 2 * sizeof(unsigned)));
    CK(hipMemset(seqbuf[<bus>], 0, sizeof(int)));
    CK(hipMemset(timeout[<bus>], 0, sizeof(unsigned)));
    CK(hipStreamCreate(&stream[<bus>]));
    std::vector<__half> h(N);
    float val = (float)(d + 1);
    for (size_t i = 0; i < N; i++) h[i] = __float2half(val);
    CK(hipMemcpy(in[<bus>], h.data(), N * sizeof(__half), hipMemcpyHostToDevice));
  }

  for (int a = 0; a < W; a++) { CK(hipSetDevice(a));
    for (int b = 0; b < W; b++) { if (a == b) continue;
      hipError_t pe = hipDeviceEnablePeerAccess(b, 0);
      if (pe != hipSuccess && pe != hipErrorPeerAccessAlreadyEnabled)
        printf("enable peer %d->%d: %s\n", a, b, hipGetErrorString(pe)); } }

  RdnaArPeers peers[RDNA_AR_MAX_WORLD];
  for (int a = 0; a < W; a++)
    for (int j = 0; j < W; j++) {
      peers[<bus>].stage[j] = stage[j];
      peers[<bus>].flags[j] = (int*)((char*)stage[j] + stage_bytes);
    }

  const float expect = (float)(W * (W + 1) / 2);   // 10.0 for world=4

  // verify every rank holds the exact sum, bit-identical across ranks
  auto verify_all = [&](const char* tag, long long iter) -> bool {
    std::vector<__half> ref;
    bool all = true;
    for (int d = 0; d < W; d++) {
      unsigned tmo = 0;
      CK(hipSetDevice(d));
      CK(hipMemcpy(&tmo, timeout[<bus>], sizeof(unsigned), hipMemcpyDeviceToHost));
      std::vector<__half> h(N);
      CK(hipMemcpy(h.data(), out[<bus>], N * sizeof(__half), hipMemcpyDeviceToHost));
      bool valok = true;
      for (size_t i = 0; i < N; i++) if (__half2float(h[i]) != expect) { valok = false; break; }
      bool ident = true;
      if (d == 0) ref = h; else for (size_t i = 0; i < N; i++) if (h[i] != ref[i]) { ident = false; break; }
      if ((tmo != 0) || !valok || !ident) {
        printf("  !! %s iter=%lld dev%d timeout=%u valok=%d identical=%d\n",
               tag, iter, d, tmo, (int)valok, (int)ident);
        all = false;
      }
    }
    return all;
  };

  auto launch_eager = [&]() {
    for (int d = 0; d < W; d++) {
      CK(hipSetDevice(d));
      rdna_ar_oneshot<__half><<<BLOCKS, THREADS, 0, stream[<bus>]>>>(
          in[<bus>], out[<bus>], peers[<bus>], arrive[<bus>], seqbuf[<bus>], timeout[<bus>], d, W, (int)N,
          max_elems, BLOCKS, 0);
    }
    for (int d = 0; d < W; d++) { CK(hipSetDevice(d)); CK(hipStreamSynchronize(stream[<bus>])); }
  };

  // ---------------- eager baseline ----------------
  for (int r = 0; r < 5; r++) launch_eager();
  int eager_ok = 0;
  std::vector<double> eager_us;
  for (int r = 0; r < EAGER_REPS; r++) {
    auto t0 = std::chrono::steady_clock::now();
    launch_eager();
    auto t1 = std::chrono::steady_clock::now();
    eager_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    if (verify_all("EAGER", r)) eager_ok++;
  }
  printf("EAGER : %d/%d reps correct   median=%.2f us\n", eager_ok, EAGER_REPS, median(eager_us));

  // ---------------- capture into per-device graphs ----------------
  for (int d = 0; d < W; d++) {
    CK(hipSetDevice(d));
    CK(hipStreamBeginCapture(stream[<bus>], hipStreamCaptureModeGlobal));
    rdna_ar_oneshot<__half><<<BLOCKS, THREADS, 0, stream[<bus>]>>>(
        in[<bus>], out[<bus>], peers[<bus>], arrive[<bus>], seqbuf[<bus>], timeout[<bus>], d, W, (int)N,
        max_elems, BLOCKS, 0);
    CK(hipStreamEndCapture(stream[<bus>], &graph[<bus>]));
    CK(hipGraphInstantiate(&gexec[<bus>], graph[<bus>], nullptr, nullptr, 0));
  }
  printf("captured %d device graph(s)\n", W);

  auto launch_graph = [&]() {
    for (int d = 0; d < W; d++) { CK(hipSetDevice(d)); CK(hipGraphLaunch(gexec[<bus>], stream[<bus>])); }
    for (int d = 0; d < W; d++) { CK(hipSetDevice(d)); CK(hipStreamSynchronize(stream[<bus>])); }
  };

  for (int r = 0; r < 5; r++) launch_graph();      // graph warmup

  // ---------------- replay, verifying EVERY iteration ----------------
  int graph_ok = 0;
  std::vector<double> graph_us;
  for (int r = 0; r < GRAPH_REPS; r++) {
    auto t0 = std::chrono::steady_clock::now();
    launch_graph();
    auto t1 = std::chrono::steady_clock::now();
    graph_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    if (verify_all("REPLAY", r)) graph_ok++;
  }
  printf("REPLAY: %d/%d replayed iterations correct   median=%.2f us\n",
         graph_ok, GRAPH_REPS, median(graph_us));

  // ---------------- did the DEVICE-SIDE sequence actually advance? ----------------
  int total_seq = 0;
  for (int d = 0; d < W; d++) {
    int s = 0;
    CK(hipSetDevice(d));
    CK(hipMemcpy(&s, seqbuf[<bus>], sizeof(int), hipMemcpyDeviceToHost));
    total_seq += s;
    printf("  dev%d device-side seqbuf = %d\n", d, s);
  }
  // each device ran: 5 eager warmup + EAGER_REPS + 5 graph warmup + GRAPH_REPS
  const int expected_seq = 5 + EAGER_REPS + 5 + GRAPH_REPS;
  bool seq_ok = (total_seq == W * expected_seq);

  printf("\nRESULT world=%d %dKB: eager %d/%d correct, replay %d/%d correct, seqbuf %d (expected %d) %s\n",
         W, KB, eager_ok, EAGER_REPS, graph_ok, GRAPH_REPS, total_seq, W * expected_seq,
         seq_ok ? "OK" : "MISMATCH");
  bool pass = (eager_ok == EAGER_REPS) && (graph_ok == GRAPH_REPS) && seq_ok;
  printf("VERDICT: %s\n", pass
    ? "PASS - custom AR is correct EAGER and under GRAPH REPLAY, and the device-derived sequence advances"
    : "FAIL");
  return pass ? 0 : 1;
}

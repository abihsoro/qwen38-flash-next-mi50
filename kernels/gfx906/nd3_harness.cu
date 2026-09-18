// nd3_harness.cu — N-D3 validation: rdna_ar_oneshot inner reduce loop on a
// single card, simulating W=4 ranks (pre-filled staging slots, self-consistent
// arrive/announce/wait protocol). Acceptance: every rank's output is
// BIT-IDENTICAL to the CPU fp32 fixed-order sum (the all-reduce contract) and
// to every other rank's output; world=1 degenerate (out == in); protocol runs
// 200 iterations with incrementing sequence without timeout.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "rdna_ar_oneshot_gfx906.cuh"

static const int N = 4096;
static const int WORLD = 4;
static const int NB = 8;     // blocks (4 ranks x 8 = 32 <= resident capacity; a barrier over 64 blocks deadlocks below ~60 CUs)
static const int NT = 256;   // threads/block
static float frand() { return (float)(rand() % 100000) / 50000.0f - 1.0f; }

int main() {
  srand(42);
  bool ok = true;
  // one in/out per simulated rank
  std::vector<float> in(WORLD * N), out(WORLD * N);
  for (int r = 0; r < WORLD; r++)
    for (int i = 0; i < N; i++) in[r * N + i] = frand() * 0.5f;
  float* din[WORLD];
  for (int r = 0; r < WORLD; r++) {
    hipMalloc(&din[r], N * 4);
    hipMemcpy(din[r], in.data() + r * N, N * 4, hipMemcpyHostToDevice);
  }
  // per-rank device buffers: staging (2*WORLD*N floats) + flag page
  float *dstage[WORLD], *dout[WORLD];
  int* dflags[WORLD];
  unsigned* darrive[WORLD];
  int* dseq[WORLD];
  unsigned* dtimeout[WORLD];
  hipStream_t streams[WORLD];
  const size_t stage_elems = (size_t)2 * WORLD * N;
  for (int r = 0; r < WORLD; r++) {
    hipMalloc(&dstage[r], stage_elems * 4);
    hipMalloc(&dout[r], N * 4);
    hipExtMallocWithFlags((void**)&dflags[r], RDNA_AR_FLAG_PAGE, hipDeviceMallocUncached);  // T18
    hipMalloc(&darrive[r], 2 * 4);
    hipMalloc(&dseq[r], 4);
    hipMalloc(&dtimeout[r], 4);
    hipMemset(dflags[r], 0, RDNA_AR_FLAG_PAGE);
    hipMemset(darrive[r], 0, 8);
    hipMemset(dseq[r], 0, 4);
    hipMemset(dtimeout[r], 0, 4);
    hipStreamCreate(&streams[r]);
  }
  // protocol stress: 200 iterations (seq increments via the device counter).
  // All four simulated ranks run the REAL kernel against shared staging
  // buffers, so the pushes populate the slots exactly as the live protocol
  // would: rank j writes in[j] into every peer's slot (p*world + j).
  int nblocks = NB, pace = 0;
  for (int it = 0; it < 200; it++) {
    for (int r = 0; r < WORLD; r++) {
      hipMemset(darrive[r], 0, 8);
      hipMemset(dtimeout[r], 0, 4);
    }
    // one stream per simulated rank so they run concurrently (the protocol's
    // cross-rank flag exchange needs live peers); each rank announces into
    // EVERY peer's own flag page and polls its own.
    for (int r = 0; r < WORLD; r++) {
      RdnaArPeers peers;
      for (int j = 0; j < WORLD; j++) { peers.stage[j] = dstage[j]; peers.flags[j] = dflags[j]; }
      rdna_ar_oneshot<float><<<NB, NT, 0, streams[r]>>>(din[r], dout[r], peers, darrive[r],
                                                        dseq[r], dtimeout[r], r, WORLD, N,
                                                        (long long)N, nblocks, pace);
    }
    for (int r = 0; r < WORLD; r++) hipStreamSynchronize(streams[r]);
    unsigned to = 0;
    hipMemcpy(&to, dtimeout[<bus>], 4, hipMemcpyDeviceToHost);
    if (to) {
      unsigned tos[WORLD]; int seqs[WORLD];
      for (int r = 0; r < WORLD; r++) { hipMemcpy(&tos[r], dtimeout[r], 4, hipMemcpyDeviceToHost); hipMemcpy(&seqs[r], dseq[r], 4, hipMemcpyDeviceToHost); }
      printf("iteration %d: TIMEOUT timeouts=[%u %u %u %u] seqs=[%d %d %d %d]\n",
             it, tos[<bus>], tos[<bus>], tos[<bus>], tos[<bus>], seqs[<bus>], seqs[<bus>], seqs[<bus>], seqs[<bus>]);
      ok = false; break;
    }
  }
  // gather final outputs (parity of seq 200 -> even -> parity 0 staging used)
  std::vector<std::vector<float>> got(WORLD, std::vector<float>(N));
  for (int r = 0; r < WORLD; r++)
    hipMemcpy(got[r].data(), dout[r], N * 4, hipMemcpyDeviceToHost);
  // CPU reference: fixed-order fp32 sum of all contributions (identical for
  // every rank — that is the all-reduce contract)
  std::vector<std::vector<float>> ref(WORLD, std::vector<float>(N));
  for (int r = 0; r < WORLD; r++)
    for (int i = 0; i < N; i++) {
      float v = 0.f;
      for (int j = 0; j < WORLD; j++) v += in[j * N + i];
      ref[r][i] = v;
    }
  int mismatches = 0;
  for (int r = 0; r < WORLD; r++)
    for (int i = 0; i < N; i++)
      if (got[r][i] != ref[r][i]) mismatches++;
  // cross-rank bit-identity
  int cross = 0;
  for (int i = 0; i < N; i++)
    for (int r = 1; r < WORLD; r++)
      if (got[r][i] != got[<bus>][i]) cross++;
  printf("N-D3 W=%d: protocol 200 iters ok=%d; mismatches_vs_cpu=%d; cross_rank_diff=%d\n",
         WORLD, ok, mismatches, cross);
  ok = ok && mismatches == 0 && cross == 0;
  // world=1 degenerate
  {
    float *din, *dout1;
    hipMalloc(&din, N * 4);
    hipMalloc(&dout1, N * 4);
    hipMemcpy(din, in.data(), N * 4, hipMemcpyHostToDevice);
    RdnaArPeers peers;
    peers.stage[<bus>] = din;
    peers.flags[<bus>] = dflags[<bus>];
    rdna_ar_oneshot<float><<<NB, NT>>>(din, dout1, peers, darrive[<bus>], dseq[<bus>],
                                        dtimeout[<bus>], 0, 1, N, (long long)N, NB, 0);
    hipDeviceSynchronize();
    std::vector<float> g1(N);
    hipMemcpy(g1.data(), dout1, N * 4, hipMemcpyDeviceToHost);
    int bad = 0;
    for (int i = 0; i < N; i++) if (g1[i] != in[i]) bad++;
    printf("N-D3 world=1 degenerate: mismatches=%d %s\n", bad, bad == 0 ? "PASS" : "FAIL");
    ok = ok && bad == 0;
  }
  printf(ok ? "N-D3: ALL PASS\n" : "N-D3: FAILURES PRESENT\n");
  return ok ? 0 : 1;
}

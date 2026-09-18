// m07_order.cu — M0.7: peer-BAR payload->fence->arrive->flag ORDERING test (G2.2 [ABS]).
//
// Property under test (what every collective here depends on):
//   If a writer GPU pushes a payload into a PEER's uncached staging and then publishes an
//   arrive count / flag with system-scope release semantics, an owner GPU polling its OWN
//   LOCAL flag and observing it MUST see the COMPLETE payload. It must be IMPOSSIBLE to
//   observe the flag with a stale or partially visible payload.
//
// Protocol modelled (faithful to the fork's T44 / rdna_ar_oneshot_gfx906.cuh):
//   writer, NBLK blocks, block b owns its own slice:
//       write slice
//       __threadfence_system()            <- the ordering edge under test
//       thread0: atomicAdd(peer_arrive, 1)
//   block 0 thread 0:
//       spin until arrive >= NBLK
//       __threadfence_system() ; system-release store flag[slot] = seq
//   owner:
//       bounded spin on its LOCAL flag[slot] >= seq
//       verify every word == pattern(seq, i)
//
// WHY THE ARRIVE COUNTER MATTERS (an earlier version of this test lacked it and was proven
// BLIND by its own negative control): the hazard is that a block's arrive-increment becomes
// visible to block 0 BEFORE that block's payload writes are visible. Block 0 then publishes
// the flag and the owner sees a "complete" signal over incomplete data. Without the per-block
// fence that reorder is architecturally legal.
//
// Negative control: `nofence=1` removes the per-block __threadfence_system(). A test with
// teeth MUST then report ORDER_VIOLATIONS. A clean control means the test is blind and any
// PASS is meaningless.
//
// Result codes: 0 clean, 1 ORDER_VIOLATION (stale word seen), 2 MISMATCH (corruption), 3 TIMEOUT.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  fprintf(stderr, "FAIL %s:%d  %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e)); \
  exit(1); } } while (0)

#define NBLK 32
#define NTHR 256
#define FLAG_PAGE 4096
#define SPIN_CAP 400000000ull

#define PAY_WORDS (1u << 18)          // 2^18 words = 1 MB: a drain is not instantaneous
#define SCRAMBLE_STEP 40503u          // odd => i = (j*STEP)&(per_blk-1) permutes the slice

__device__ __forceinline__ unsigned pat(unsigned seq, unsigned i) {
  return (seq * 2654435761u) ^ (i * 2246822519u) ^ 0xA5A5A5A5u;
}

__global__ void k_writer(unsigned* __restrict__ peer_stage,
                         unsigned* __restrict__ peer_flag,
                         unsigned* __restrict__ peer_arrive,
                         unsigned seq, int slot, int nofence, int pace, int early) {
  const unsigned b = blockIdx.x, t = threadIdx.x;
  const unsigned per_blk = PAY_WORDS / NBLK;         // power of two

  // PROTOCOL-BUG CONTROL (early=1): announce arrival BEFORE writing the payload. This is not a
  // hardware reorder - it is the protocol defect the fence/ordering exists to make impossible.
  // It must produce ORDER_VIOLATIONS, which proves the owner-side CHECKER actually works.
  if (early && t == 0) atomicAdd(peer_arrive, 1u);

  // Deliberate per-block stagger: block b spins briefly before writing, so blocks finish at
  // different times and the reorder window the fence must close is wide.
  if (pace > 0)
    for (int q = 0; q < pace * (int)(b + 1); q++) __builtin_amdgcn_s_sleep(1);

  for (unsigned j = t; j < per_blk; j += blockDim.x) {
    unsigned i = b * per_blk + ((j * SCRAMBLE_STEP) & (per_blk - 1u));
    peer_stage[i] = pat(seq, i);
  }
  __syncthreads();
  if (!early && t == 0) {
    if (!nofence) __threadfence_system();     // NEGATIVE CONTROL removes exactly this edge
    atomicAdd(peer_arrive, 1u);
  }
  __syncthreads();

  if (b == 0 && t == 0) {
    unsigned long long s = 0;
    while (__hip_atomic_load(peer_arrive, __ATOMIC_ACQUIRE,
                             __HIP_MEMORY_SCOPE_AGENT) < (unsigned)NBLK) {
      __builtin_amdgcn_s_sleep(8);
      if (++s > SPIN_CAP) return;
    }
    __threadfence_system();
    __hip_atomic_store(&peer_flag[slot], seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__global__ void k_owner(const unsigned* __restrict__ my_stage,
                        const unsigned* __restrict__ my_flag,
                        unsigned seq, int slot, unsigned* out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    unsigned long long s = 0;
    out[<bus>] = 0u; out[<bus>] = 0xFFFFFFFFu;
    while (__hip_atomic_load(&my_flag[slot], __ATOMIC_ACQUIRE,
                             __HIP_MEMORY_SCOPE_SYSTEM) < seq) {
      __builtin_amdgcn_s_sleep(8);
      if (++s > SPIN_CAP) { out[<bus>] = 3u; return; }
    }
  }
  __syncthreads();
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned gstride = gridDim.x * blockDim.x;
  // EARLY EXIT: the checker's job is binary (violation or not). Scanning the whole 1 MB and
  // issuing an atomic per bad word made the deliberately-broken control run for minutes on
  // contended atomics. Record the first violation and stop.
  for (unsigned i = gid; i < PAY_WORDS; i += gstride) {
    unsigned got = my_stage[i], want = pat(seq, i);
    if (got != want) {
      unsigned res = (got == pat(seq - 1u, i)) ? 1u : 2u;
      atomicMin(&out[<bus>], res);
      atomicMin(&out[<bus>], i);
      return;
    }
  }
}

int main(int argc, char** argv) {
  int nd = 0; CK(hipGetDeviceCount(&nd));
  const int REPS    = (argc > 1) ? atoi(argv[<bus>]) : 200;
  const int NOFENCE = (argc > 2) ? atoi(argv[<bus>]) : 0;
  const int PACE    = (argc > 3) ? atoi(argv[<bus>]) : 0;
  const int CACHED  = (argc > 4) ? atoi(argv[<bus>]) : 0;   // 1 = cached staging (hazard amplifier)
  const int EARLY   = (argc > 5) ? atoi(argv[<bus>]) : 0;   // 1 = arrive BEFORE payload (protocol-bug control)
  printf("m07_order: devices=%d reps=%d nofence=%d pace=%d cached=%d early=%d payload=%u KB blocks=%d\n",
         nd, REPS, NOFENCE, PACE, CACHED, EARLY, PAY_WORDS * 4 / 1024, NBLK);
  if (EARLY) printf("  *** PROTOCOL-BUG CONTROL: arrive published BEFORE the payload. Violations\n"
                    "      MUST appear; this proves the owner-side checker works. ***\n");
  if (NOFENCE) printf("  *** NEGATIVE CONTROL: per-block fence omitted - violations EXPECTED;\n"
                      "      a clean control means the test is BLIND and a PASS is meaningless ***\n");
  if (CACHED) printf("  *** cached staging: peer writes drain LAZILY, so the reorder window is\n"
                     "      real. cached+nofence should FAIL; cached+fence should PASS. ***\n");
  if (nd < 2) { printf("need >=2 devices\n"); return 2; }

  const size_t pay_bytes = (size_t)PAY_WORDS * sizeof(unsigned);
  unsigned *stage[<bus>] = {0}, *flags[<bus>] = {0}, *arrive[<bus>] = {0}, *res[<bus>] = {0};
  for (int d = 0; d < nd; d++) {
    CK(hipSetDevice(d));
    hipError_t e1;
    if (CACHED) e1 = hipMalloc((void**)&stage[<bus>], pay_bytes);
    else        e1 = hipExtMallocWithFlags((void**)&stage[<bus>], pay_bytes, hipDeviceMallocUncached);
    hipError_t e2 = hipExtMallocWithFlags((void**)&flags[<bus>], FLAG_PAGE, hipDeviceMallocUncached);
    hipError_t e3 = hipExtMallocWithFlags((void**)&arrive[<bus>], 64, hipDeviceMallocUncached);
    if (e1 || e2 || e3) { printf("dev%d alloc: %s / %s / %s\n", d,
        hipGetErrorString(e1), hipGetErrorString(e2), hipGetErrorString(e3)); return 1; }
    CK(hipMemset(stage[<bus>], 0, pay_bytes));
    CK(hipMemset(flags[<bus>], 0, FLAG_PAGE));
    CK(hipMemset(arrive[<bus>], 0, 64));
    CK(hipMalloc(&res[<bus>], 2 * sizeof(unsigned)));
    CK(hipMemset(res[<bus>], 0, 2 * sizeof(unsigned)));
  }
  for (int a = 0; a < nd; a++) { CK(hipSetDevice(a));
    for (int b = 0; b < nd; b++) { if (a == b) continue;
      hipError_t pe = hipDeviceEnablePeerAccess(b, 0);
      if (pe != hipSuccess && pe != hipErrorPeerAccessAlreadyEnabled)
        printf("  enable peer %d->%d: %s\n", a, b, hipGetErrorString(pe)); } }

  unsigned out[<bus>];
  int pairs = 0, clean = 0;
  long long tv = 0, tm = 0, tt = 0;

  for (int w = 0; w < nd; w++) for (int o = 0; o < nd; o++) {
    if (w == o) continue;
    pairs++;
    unsigned viol = 0, mism = 0, tmo = 0;
    const int slot = w;
    const unsigned seqbase = 1000u + (unsigned)(w * 16 + o) * 100000u;

    for (int r = 0; r < REPS; r++) {
      unsigned seq = seqbase + (unsigned)r + 1u;

      CK(hipSetDevice(o)); CK(hipMemsetAsync(arrive[o], 0, 64));   // reset the arrive counter
      CK(hipSetDevice(w));
      k_writer<<<NBLK, NTHR>>>(stage[o], flags[o], arrive[o], seq, slot, NOFENCE, PACE, EARLY);
      CK(hipSetDevice(o));
      CK(hipMemsetAsync(res[o], 0, 2 * sizeof(unsigned)));
      k_owner<<<16, NTHR>>>(stage[o], flags[o], seq, slot, res[o]);

      CK(hipSetDevice(w)); CK(hipDeviceSynchronize());
      CK(hipSetDevice(o)); CK(hipDeviceSynchronize());
      CK(hipMemcpy(out, res[o], 2 * sizeof(unsigned), hipMemcpyDeviceToHost));

      if (out[<bus>] == 1u) { if (!viol) printf("  !! ORDER_VIOLATION w=%d o=%d rep=%d word=%u\n", w,o,r,out[<bus>]); viol++; }
      if (out[<bus>] == 2u) { if (!mism) printf("  !! MISMATCH        w=%d o=%d rep=%d word=%u\n", w,o,r,out[<bus>]); mism++; }
      if (out[<bus>] == 3u) { if (!tmo)  printf("  !! TIMEOUT         w=%d o=%d rep=%d\n", w,o,r); tmo++; }
    }
    tv += viol; tm += mism; tt += tmo;
    bool ok = (viol == 0 && mism == 0 && tmo == 0);
    if (ok) clean++;
    printf("  pair w=%d -> o=%d : %d reps  order_violations=%u mismatches=%u timeouts=%u  %s\n",
           w, o, REPS, viol, mism, tmo, ok ? "OK" : "FAIL");
  }

  printf("\nRESULT M0.7: %d/%d ordered pairs clean over %d reps (violations=%lld mismatches=%lld timeouts=%lld)\n",
         clean, pairs, REPS, tv, tm, tt);
  if (EARLY)
    printf("VERDICT(CHECKER CONTROL): %s\n", (clean == pairs)
      ? "CHECKER IS BROKEN - it cannot see flag-before-payload even when the protocol is"
        " deliberately wrong; all other results are MEANINGLESS"
      : "checker WORKS - a deliberately premature flag IS detected");
  else if (NOFENCE)
    printf("VERDICT(CONTROL): %s\n", (clean == pairs)
      ? "fence omission NOT detectable on this hardware (no hardware reorder induced)"
      : "test has teeth - removing the fence IS detected");
  else
    printf("VERDICT: %s\n", (clean == pairs && pairs > 0) ? "PASS" : "FAIL");
  return (clean == pairs && pairs > 0) ? 0 : 1;
}

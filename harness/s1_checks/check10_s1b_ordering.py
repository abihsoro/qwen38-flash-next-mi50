#!/usr/bin/env python3
"""S1b — G2.2 closure test, D017 amendments (N1, N2).

Part A — write-ordering (amended G2.2 predicate), REDESIGNED per N1:
  The consumer is a GPU kernel on a SECOND stream, running CONCURRENTLY with
  the producer: it spins on the flag, then verifies the payload. This creates a
  real race window (a subsequent-kernel consumer would serialize with the
  producer and prove nothing — the earlier design's flaw, caught by N1).
  payload -> block barrier -> [__threadfence_system()] -> flag(monotonic seq),
  >= 1e6 iterations, ZERO stale/torn reads at S4-predicted payload sizes.

N1 negative control: --no-fence removes the release fence; the run MUST show
non-zero violations, or the test has no race window and G2.2 stays open.
Both runs are recorded as separate rows in results/S1b.jsonl.

Part B — device-scope atomics in VRAM (explicit yes/no; MoE routing).

Statistical record: 0/N is an upper bound (~3/N @ 95% CI), not proof of zero.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

HIP_SRC_TMPL = r"""
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>

// Producer (single block): fill payload, block barrier, [system fence], set flag.
__global__ void producer_fill(int* payload, int words, uint32_t* flag,
                              uint32_t seq, int use_fence) {
  int idx = threadIdx.x;
  for (int i = idx; i < words; i += blockDim.x) payload[i] = (int)seq;
  __syncthreads();  // block barrier: all block writes issued
  if (use_fence) __threadfence_system();
  if (threadIdx.x == 0) *flag = seq;
}

// Consumer (concurrent, own stream): ALL blocks spin on the flag, then verify
// the payload grid-wide. A real race window exists: the consumer runs
// concurrently with the producer, so only the fence guarantees ordered reads.
__global__ void consumer_verify(const int* payload, int words,
                                uint32_t* flag, uint32_t seq,
                                int* bad_out) {
  volatile uint32_t* vf = (volatile uint32_t*)flag;
  while (*vf != seq) {}
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int bad = 0;
  for (int i = idx; i < words; i += gridDim.x * blockDim.x) {
    if (payload[i] != (int)seq) bad = 1;
  }
  __shared__ int s[<bus>];
  s[threadIdx.x] = bad;
  __syncthreads();
  for (int o = blockDim.x / 2; o > 0; o >>= 1) {
    if (threadIdx.x < o) s[threadIdx.x] |= s[threadIdx.x + o];
    __syncthreads();
  }
  if (threadIdx.x == 0 && s[<bus>]) atomicAdd(bad_out, 1);
}

__global__ void atomic_vram(int* counter, int times) {
  if (threadIdx.x == 0 && blockIdx.x == 0)
    for (int t = 0; t < times; t++) atomicAdd(counter, 1);
}

int main(int argc, char** argv) {
  int use_fence = 1;
  if (argc > 1 && strcmp(argv[<bus>], "--no-fence") == 0) use_fence = 0;
  printf("consumer=GPU_kernel_concurrent_stream_spin_on_flag+verify "
         "(system-scope through PCIe) fence=%d\n", use_fence);

  const int n_sizes = 5;
  const int sizes[n_sizes] = {64, 5120, 262144, 1048576, 3276800};
  const long iters[n_sizes] = {1000000L, 1000000L, 200000L, 50000L, 20000L};

  int* payload = nullptr;
  uint32_t* flag = nullptr;
  hipHostMalloc(&payload, sizes[n_sizes - 1],
                hipHostMallocMapped | hipHostMallocCoherent);
  hipHostMalloc(&flag, 4096, hipHostMallocMapped | hipHostMallocCoherent);
  int* dbad = nullptr;
  hipMalloc(&dbad, sizeof(int));
  hipStream_t streamA, streamB;
  hipStreamCreate(&streamA);
  hipStreamCreate(&streamB);

  bool all_ok = true;
  long total_iters = 0, total_violations = 0;
  for (int s = 0; s < n_sizes; s++) {
    int bytes = sizes[s], words = bytes / 4;
    long n = iters[s];
    *flag = 0;
    hipMemset(dbad, 0, sizeof(int));
    auto t0 = std::chrono::steady_clock::now();
    for (long i = 1; i <= n; i++) {
      uint32_t seq = (uint32_t)i;
      *flag = 0;  // host resets the flag; producer sets it to seq
      producer_fill<<<1, 256, 0, streamA>>>(payload, words, flag, seq, use_fence);
      consumer_verify<<<8, 256, 0, streamB>>>(payload, words, flag, seq, dbad);
      hipStreamSynchronize(streamA);
      hipStreamSynchronize(streamB);
    }
    auto t1 = std::chrono::steady_clock::now();
    int bad = 0;
    hipMemcpy(&bad, dbad, sizeof(int), hipMemcpyDeviceToHost);
    double us = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
                / 1000.0 / (double)n;
    bool ok = (bad == 0);
    all_ok = all_ok && ok;
    total_iters += n;
    total_violations += bad;
    printf("size=%d B iters=%ld roundtrip_us=%.2f violations=%d %s\n",
           bytes, n, us, bad, ok ? "ORDER_OK" : "ORDER_VIOLATION");
  }

  int vram_val = 0;
  hipMemset(dbad, 0, sizeof(int));
  atomic_vram<<<1, 1>>>(dbad, 1000000);
  hipDeviceSynchronize();
  hipMemcpy(&vram_val, dbad, sizeof(int), hipMemcpyDeviceToHost);
  bool vram_ok = (vram_val == 1000000);
  printf("vram_atomic_device_scope=%d (expect 1000000) %s\n",
         vram_val, vram_ok ? "VRAM_ATOMICS_OK" : "VRAM_ATOMICS_FAIL");

  printf("total_iters=%ld total_violations=%ld\n", total_iters, total_violations);
  if (use_fence) {
    printf("BOUND_NOTE: violations==0 is an upper bound (~3e-5 @ 95%% CI), "
           "not proof of zero\n");
    printf(all_ok ? "S1B_ORDERING_PASS\n" : "S1B_ORDERING_FAIL\n");
  } else {
    printf(total_violations > 0 ? "NEGATIVE_CONTROL_OK (violations present)\n"
                                : "NEGATIVE_CONTROL_FAIL (no race window; "
                                  "test proves nothing)\n");
    all_ok = total_violations > 0;
  }
  hipHostFree(payload);
  hipHostFree(flag);
  hipFree(dbad);
  hipStreamDestroy(streamA);
  hipStreamDestroy(streamB);
  return all_ok ? 0 : 1;
}
"""


def main() -> int:
    no_fence = "--no-fence" in sys.argv
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "s1b.cpp"
        exe = td / "s1b"
        src.write_text(HIP_SRC_TMPL)
        comp = subprocess.run(["hipcc", "-O2", str(src), "-o", str(exe)],
                              capture_output=True, text=True, timeout=300)
        if comp.returncode != 0:
            print("FAIL: hipcc compile error")
            print(comp.stderr[-2000:])
            return 1
        cmd = [str(exe)] + (["--no-fence"] if no_fence else [])
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        print(run.stdout.strip())
        if run.returncode != 0:
            print(f"FAIL: s1b run exited {run.returncode}")
            print(run.stderr[-1500:])
            return 1
        print("PASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())

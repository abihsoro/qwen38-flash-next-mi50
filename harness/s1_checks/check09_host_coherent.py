#!/usr/bin/env python3
"""S1 check 09 — host-coherent primitives (non-deferrable; the building blocks
of the mandatory gfx906_ar_oneshot, work order §6.S1 + §9.M2).

Verifies, on gfx906 in the VFIO guest:
  1. fine-grained host allocation (hipHostMalloc with hipHostMallocCoherent)
  2. GPU atomic on host memory (hipAtomicAdd on coherent host memory)
  3. flag-based signal/wait round-trip GPU->host, latency recorded (us)
"""
import subprocess
import sys
import tempfile
from pathlib import Path

HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <chrono>

__global__ void atomic_add_host(int* p, int v) {
  atomicAdd_system(p, v);  // system-scope atomic on fine-grained host memory (gfx906)
}
__global__ void set_flag(uint32_t* f, uint32_t v) {
  __threadfence_system();
  *f = v;
}

int main() {
  int* hmem = nullptr;
  hipError_t e = hipHostMalloc(&hmem, 4096, hipHostMallocMapped | hipHostMallocCoherent);
  if (e != hipSuccess) { printf("FAIL: hipHostMalloc coherent %s\n", hipGetErrorString(e)); return 1; }
  hmem[<bus>] = 0;
  atomic_add_host<<<1, 1>>>(hmem, 7);
  hipDeviceSynchronize();
  bool ok1 = (hmem[<bus>] == 7);
  printf("gpu_atomic_on_host=%d (expect 7)\n", hmem[<bus>]);

  uint32_t* flag = nullptr;
  e = hipHostMalloc(&flag, 4096, hipHostMallocMapped | hipHostMallocCoherent);
  if (e != hipSuccess) { printf("FAIL: hipHostMalloc flag %s\n", hipGetErrorString(e)); return 1; }
  *flag = 0;
  set_flag<<<1, 1>>>(flag, 1); hipDeviceSynchronize();  // warmup
  *flag = 0;

  const int R = 1000;
  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < R; r++) {
    *flag = 0;
    set_flag<<<1, 1>>>(flag, 1);
    while (*flag != 1) {}
  }
  auto t1 = std::chrono::steady_clock::now();
  double us = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
              / 1000.0 / (double)R;
  printf("flag_roundtrip_us=%.2f (R=%d)\n", us, R);
  bool ok2 = (*flag == 1);
  bool ok = ok1 && ok2;
  hipHostFree(hmem);
  hipHostFree(flag);
  return ok ? 0 : 1;
}
"""


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "host_coherent.cpp"
        exe = td / "host_coherent"
        src.write_text(HIP_SRC)
        comp = subprocess.run(["hipcc", "-O2", str(src), "-o", str(exe)],
                              capture_output=True, text=True, timeout=300)
        if comp.returncode != 0:
            print("FAIL: hipcc compile error")
            print(comp.stderr[-2000:])
            return 1
        run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=300)
        print(run.stdout.strip())
        if run.returncode != 0:
            print(f"FAIL: host-coherent run exited {run.returncode}")
            print(run.stderr[-1000:])
            return 1
        print("PASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())

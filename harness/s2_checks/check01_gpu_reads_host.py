#!/usr/bin/env python3
"""S2 task 1 (operator N5, D017) — GPU-reads-host-memory direction.

The n-gram offload (§3.3) has the GPU gather rows from the host-resident
102 GB table. This test validates the direction on gfx906:
  1. host writes a table (model-sized rows: hidden 2560 x 2 B = 5 KiB),
  2. GPU gathers R rows per step from host memory,
  3. correctness verified against the host pattern,
  4. gather latency measured (R in {1,4,16}, 1000 reps each),
  5. two §3.3 paths: hipHostMallocCoherent (fine-grained) and
     malloc+hipHostRegister (zero-copy pinned),
  6. host->GPU visibility ordering: fill then gather with and without an
     explicit stream sync (record whether the driver orders it).
"""
import subprocess
import sys
import tempfile
from pathlib import Path

HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>

#define ROWS 262144u
#define ROW_WORDS 2560u   // hidden_size (uint16 words)

__global__ void gather_rows(const uint16_t* table, const int32_t* idx,
                            uint16_t* out, int rows, int row_words) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * row_words;
  for (int i = t; i < total; i += gridDim.x * blockDim.x) {
    int r = i / row_words, c = i % row_words;
    out[i] = table[(size_t)idx[r] * row_words + c];
  }
}

static inline uint16_t pat(uint32_t row, uint32_t col) {
  return (uint16_t)((row * 31u + col * 7u) & 0xFFFFu);
}

void fill_table(uint16_t* t) {
  for (uint32_t r = 0; r < ROWS; r++)
    for (uint32_t c = 0; c < ROW_WORDS; c++) t[(size_t)r * ROW_WORDS + c] = pat(r, c);
}

bool verify(uint16_t* out, int32_t* idx, int rows) {
  for (int r = 0; r < rows; r++)
    for (int c = 0; c < ROW_WORDS; c++)
      if (out[(size_t)r * ROW_WORDS + c] != pat((uint32_t)idx[r], (uint32_t)c)) return false;
  return true;
}

int run_case(const char* name, uint16_t* table, bool sync_after_fill) {
  const int reps = 1000;
  const int rowsets[<bus>] = {1, 4, 16};
  int32_t* didx = nullptr;
  uint16_t* dout = nullptr;
  hipMalloc(&didx, 16 * sizeof(int32_t));
  hipMalloc(&dout, 16 * ROW_WORDS * sizeof(uint16_t));
  int32_t hidx[<bus>];
  for (int i = 0; i < 16; i++) hidx[i] = (int32_t)((i * 7919u) % ROWS);

  bool all_ok = true;
  for (int s = 0; s < 3; s++) {
    int rows = rowsets[s];
    hipMemcpy(didx, hidx, rows * sizeof(int32_t), hipMemcpyHostToDevice);
    if (sync_after_fill) hipDeviceSynchronize();
    // correctness
    gather_rows<<<64, 256>>>(table, didx, dout, rows, ROW_WORDS);
    hipDeviceSynchronize();
    uint16_t* hout = (uint16_t*)malloc(rows * ROW_WORDS * 2);
    hipMemcpy(hout, dout, rows * ROW_WORDS * 2, hipMemcpyDeviceToHost);
    bool ok = verify(hout, hidx, rows);
    all_ok = all_ok && ok;
    // latency (reps)
    hipEvent_t ev0, ev1;
    hipEventCreate(&ev0); hipEventCreate(&ev1);
    hipEventRecord(ev0);
    for (int r = 0; r < reps; r++)
      gather_rows<<<64, 256>>>(table, didx, dout, rows, ROW_WORDS);
    hipEventRecord(ev1);
    hipEventSynchronize(ev1);
    float ms = 0.0f;
    hipEventElapsedTime(&ms, ev0, ev1);
    double us = ms * 1000.0 / (double)reps;
    double bytes = (double)rows * ROW_WORDS * 2.0;
    printf("%s rows=%d sync=%d ok=%d gather_us=%.2f eff_gbps=%.2f\n",
           name, rows, sync_after_fill, ok ? 1 : 0, us, bytes / us / 1000.0);
    free(hout);
  }
  hipFree(didx); hipFree(dout);
  return all_ok ? 0 : 1;
}

int main() {
  int rc = 0;
  // Case A: fine-grained host allocation (hipHostMallocCoherent)
  uint16_t* fine = nullptr;
  hipHostMalloc(&fine, (size_t)ROWS * ROW_WORDS * 2, hipHostMallocMapped | hipHostMallocCoherent);
  fill_table(fine);
  rc |= run_case("hipHostMallocCoherent", fine, true);
  rc |= run_case("hipHostMallocCoherent", fine, false);  // no explicit sync after fill
  hipHostFree(fine);

  // Case B: malloc + hipHostRegister (zero-copy pin, §3.3 path)
  uint16_t* reg = (uint16_t*)malloc((size_t)ROWS * ROW_WORDS * 2);
  fill_table(reg);
  hipHostRegister(reg, (size_t)ROWS * ROW_WORDS * 2, hipHostRegisterMapped);
  rc |= run_case("malloc+hipHostRegister", reg, true);
  rc |= run_case("malloc+hipHostRegister", reg, false);
  hipHostUnregister(reg);
  free(reg);

  printf(rc == 0 ? "S2_T1_GPU_READS_HOST_PASS\n" : "S2_T1_GPU_READS_HOST_FAIL\n");
  return rc;
}
"""


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "s2t1.cpp"
        exe = td / "s2t1"
        src.write_text(HIP_SRC)
        comp = subprocess.run(["hipcc", "-O2", str(src), "-o", str(exe)],
                              capture_output=True, text=True, timeout=300)
        if comp.returncode != 0:
            print("FAIL: hipcc compile error")
            print(comp.stderr[-2000:])
            return 1
        run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=900)
        print(run.stdout.strip())
        if run.returncode != 0:
            print(f"FAIL: run exited {run.returncode}")
            print(run.stderr[-1500:])
            return 1
        print("PASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())

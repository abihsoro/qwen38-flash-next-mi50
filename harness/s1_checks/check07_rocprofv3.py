#!/usr/bin/env python3
"""S1 check 07 — rocprofv3 produces a non-empty kernel trace with kernel names.

Profiles a raw HIP kernel (saxpy built with hipcc) — NOT torch: the prebuilt
torch rocm7.2 wheel's bundled rocBLAS lacks gfx906 Tensile libraries and aborts,
which would mask rocprofv3 itself. Kernel name 'saxpy' must appear in the trace.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>
__global__ void saxpy(float a, const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}
int main() {
  const int n = 1 << 20;
  const float a = 2.5f;
  float* hx = new float[n];
  float* hy = new float[n];
  for (int i = 0; i < n; i++) { hx[i] = i * 0.001f; hy[i] = 1.0f; }
  float *dx, *dy;
  hipMalloc(&dx, n * sizeof(float));
  hipMalloc(&dy, n * sizeof(float));
  hipMemcpy(dx, hx, n * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(dy, hy, n * sizeof(float), hipMemcpyHostToDevice);
  saxpy<<<(n + 255) / 256, 256>>>(a, dx, dy, n);
  hipDeviceSynchronize();
  hipMemcpy(hy, dy, n * sizeof(float), hipMemcpyDeviceToHost);
  float err = 0.0f;
  for (int i = 0; i < n; i++) {
    float want = a * hx[i] + 1.0f;
    err = fmaxf(err, fabsf(hy[i] - want));
  }
  printf("saxpy_max_abs_err=%g\n", err);
  delete[] hx; delete[] hy; hipFree(dx); hipFree(dy);
  return err < 1e-3f ? 0 : 1;
}
"""


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "saxpy.cpp"
        exe = td / "saxpy"
        src.write_text(HIP_SRC)
        comp = subprocess.run(["hipcc", "-O2", str(src), "-o", str(exe)],
                              capture_output=True, text=True, timeout=300)
        if comp.returncode != 0:
            print("FAIL: hipcc compile error")
            print(comp.stderr[-2000:])
            return 1
        proc = subprocess.run(
            ["rocprofv3", "--stats", "--kernel-trace", "-d", str(td / "trace"),
             "--", str(exe)],
            capture_output=True, text=True, timeout=300,
        )
        print(proc.stdout[-800:])
        if proc.returncode != 0:
            print(f"FAIL: rocprofv3 exited {proc.returncode}")
            print(proc.stderr[-1500:])
            return 1
        files = list((td / "trace").rglob("*"))
        has_saxpy = False
        for p in files:
            if not p.is_file():
                continue
            if p.suffix == ".csv":
                if "saxpy" in p.read_text(errors="replace"):
                    has_saxpy = True
                    break
            elif p.suffix == ".db":
                try:
                    import sqlite3
                    con = sqlite3.connect(str(p))
                    tables = [r[<bus>] for r in con.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'")]
                    for t in tables:
                        cols = [c[<bus>] for c in con.execute(f"PRAGMA table_info({t})")]
                        for c in cols:
                            if "name" in c.lower():
                                rows = con.execute(f"SELECT DISTINCT {c} FROM {t} LIMIT 5000")
                                if any("saxpy" in str(r[<bus>]) for r in rows):
                                    has_saxpy = True
                                    break
                        if has_saxpy:
                            break
                    con.close()
                except Exception as exc:
                    print(f"(sqlite inspect failed: {exc})")
                if has_saxpy:
                    break
        print(f"trace files: {[p.name for p in files if p.is_file()]}")
        print(f"has_saxpy_kernel: {has_saxpy}")
        ok = bool(files) and has_saxpy
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

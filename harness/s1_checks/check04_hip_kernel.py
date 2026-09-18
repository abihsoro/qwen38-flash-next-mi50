#!/usr/bin/env python3
"""S1 check 04 — custom HIP kernel compiles (hipcc), runs, exit 0, output verified.

saxpy kernel, compiled at runtime with hipcc, output checked against CPU.
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
  bool ok = err < 1e-3f;
  delete[] hx; delete[] hy; hipFree(dx); hipFree(dy);
  return ok ? 0 : 1;
}
"""


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "saxpy.cpp"
        exe = Path(td) / "saxpy"
        src.write_text(HIP_SRC)
        comp = subprocess.run(["hipcc", "-O2", str(src), "-o", str(exe)],
                              capture_output=True, text=True, timeout=300)
        if comp.returncode != 0:
            print("FAIL: hipcc compile error")
            print(comp.stderr[-2000:])
            return 1
        run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
        print(run.stdout.strip())
        if run.returncode != 0:
            print(f"FAIL: kernel run exited {run.returncode}")
            print(run.stderr[-1000:])
            return 1
        print("PASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())

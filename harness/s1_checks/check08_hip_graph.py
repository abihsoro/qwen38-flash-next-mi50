#!/usr/bin/env python3
"""S1 check 08 — HIP graph capture/replay (Gate G2).

A multi-kernel synthetic step (scale -> add -> relu) is captured into a HIP
graph, replayed, and its output compared bit-exactly-ish with the uncaptured
execution. G2 passes iff the captured replay output matches the uncaptured run.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>

__global__ void scale_k(float s, float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] *= s;
}
__global__ void add_const_k(float c, float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] += c;
}
__global__ void relu_k(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

void launch_step(float* d, int n, hipStream_t s) {
  scale_k<<<(n + 255) / 256, 256, 0, s>>>(0.5f, d, n);
  add_const_k<<<(n + 255) / 256, 256, 0, s>>>(1.0f, d, n);
  relu_k<<<(n + 255) / 256, 256, 0, s>>>(d, n);
}

int main() {
  const int n = 1 << 20;
  float* hx = new float[n];
  srand(42);
  for (int i = 0; i < n; i++) hx[i] = (float)(rand() % 1000) / 100.0f - 5.0f;
  float *d, *ref, *got;
  hipMalloc(&d, n * sizeof(float));
  hipMalloc(&ref, n * sizeof(float));
  hipMalloc(&got, n * sizeof(float));

  // uncaptured reference
  hipMemcpy(d, hx, n * sizeof(float), hipMemcpyHostToDevice);
  launch_step(d, n, nullptr);
  hipDeviceSynchronize();
  hipMemcpy(ref, d, n * sizeof(float), hipMemcpyDeviceToDevice);

  // captured
  hipMemcpy(d, hx, n * sizeof(float), hipMemcpyHostToDevice);
  hipStream_t stream = nullptr;
  hipStreamCreate(&stream);
  hipError_t err = hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal);
  if (err != hipSuccess) { printf("FAIL: hipStreamBeginCapture %s\n", hipGetErrorString(err)); return 1; }
  launch_step(d, n, stream);
  hipGraph_t graph = nullptr;
  err = hipStreamEndCapture(stream, &graph);
  if (err != hipSuccess) { printf("FAIL: hipStreamEndCapture %s\n", hipGetErrorString(err)); return 1; }
  hipGraphExec_t exec = nullptr;
  err = hipGraphInstantiate(&exec, graph, NULL, NULL, 0);
  if (err != hipSuccess) { printf("FAIL: hipGraphInstantiate %s\n", hipGetErrorString(err)); return 1; }
  hipGraphLaunch(exec, stream);
  hipStreamSynchronize(stream);
  hipMemcpy(got, d, n * sizeof(float), hipMemcpyDeviceToDevice);

  float max_err = 0.0f;
  for (int i = 0; i < n; i++) max_err = fmaxf(max_err, fabsf(ref[i] - got[i]));
  printf("graph_vs_uncaptured_max_abs_err=%g\n", max_err);
  bool ok = max_err < 1e-5f;
  hipGraphExecDestroy(exec);
  hipGraphDestroy(graph);
  hipStreamDestroy(stream);
  hipFree(d); hipFree(ref); hipFree(got);
  delete[] hx;
  return ok ? 0 : 1;
}
"""


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "graph.cpp"
        exe = td / "graph"
        src.write_text(HIP_SRC)
        comp = subprocess.run(["hipcc", "-O2", str(src), "-o", str(exe)],
                              capture_output=True, text=True, timeout=300)
        if comp.returncode != 0:
            print("FAIL: hipcc compile error")
            print(comp.stderr[-2000:])
            return 1
        run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=120)
        print(run.stdout.strip())
        if run.returncode != 0:
            print(f"FAIL: graph run exited {run.returncode}")
            print(run.stderr[-1000:])
            return 1
        print("PASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())

"""Shared helpers: GPU count, peer access, verdict plumbing."""
import json
import subprocess
import sys


def gpu_count() -> int:
    try:
        out = subprocess.run(["rocm-smi", "--showproductname"],
                             capture_output=True, text=True, timeout=20)
        idx = {line.split(":")[<bus>].strip() for line in out.stdout.splitlines()
               if "GPU[" in line}
        return max((int(i[4:-1]) for i in idx if i[4:-1].isdigit()), default=-1) + 1
    except Exception:
        return 0


def hip_peer_matrix() -> list:
    """hipDeviceCanAccessPeer matrix via a tiny hipcc probe (run once, cached)."""
    import os, tempfile
    cache = os.path.join(os.path.dirname(__file__), ".peer_matrix.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    src = r'''
#include <hip/hip_runtime.h>
#include <cstdio>
int main() {
  int n = 0; hipGetDeviceCount(&n);
  printf("%d\n", n);
  for (int a = 0; a < n; a++)
    for (int b = 0; b < n; b++) {
      int ok = 0;
      hipDeviceCanAccessPeer(&ok, a, b);
      printf("%d %d %d\n", a, b, ok);
    }
  return 0;
}'''
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "pm.cpp")
        open(p, "w").write(src)
        subprocess.run(["hipcc", p, "-o", os.path.join(td, "pm")], check=True)
        out = subprocess.run([os.path.join(td, "pm")], capture_output=True,
                             text=True, check=True)
    lines = out.stdout.splitlines()
    n = int(lines[<bus>])
    mat = [[<bus>] * n for _ in range(n)]
    for line in lines[1:]:
        a, b, ok = map(int, line.split())
        mat[<bus>][<bus>] = ok
    json.dump(mat, open(cache, "w"))
    return mat


class Verdict:
    def __init__(self):
        self.passed = []
        self.skipped = []
        self.failed = []

    def ok(self, name, detail=""):
        self.passed.append(name)
        print(f"  PASS  {name} {detail}")

    def skip(self, name, why):
        self.skipped.append(name)
        print(f"  SKIP  {name} ({why})")

    def fail(self, name, detail=""):
        self.failed.append(name)
        print(f"  FAIL  {name} {detail}")

    def summary(self, suite: str) -> int:
        print(f"\n{suite}: {len(self.passed)} passed, {len(self.skipped)} skipped, "
              f"{len(self.failed)} failed")
        return 1 if self.failed else 0

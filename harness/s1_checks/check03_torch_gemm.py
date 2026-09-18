#!/usr/bin/env python3
"""S1 check 03 — torch GEMM: fp32 and fp16 results within tolerance vs CPU.

Reference: float64 CPU matmul. Tolerance: fp32 rel err <= 1e-4, fp16 <= 5e-3.
"""
import sys

import torch


def main() -> int:
    torch.manual_seed(0)
    M, K, N = 512, 512, 512
    a = torch.randn(M, K, device="cuda")
    b = torch.randn(K, N, device="cuda")
    ref = (a.double() @ b.double()).cpu()

    results = {}
    for name, dtype, tol in (("fp32", torch.float32, 1e-4), ("fp16", torch.float16, 5e-3)):
        g = torch.matmul(a.to(dtype), b.to(dtype)).float().cpu()
        rel = (g - ref).abs().max().item() / ref.abs().max().item()
        results[name] = {"rel_err": rel, "ok": rel <= tol}
        print(f"{name} rel_err={rel:.3e} tol={tol} {'PASS' if rel <= tol else 'FAIL'}")

    ok = all(r["ok"] for r in results.values())
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

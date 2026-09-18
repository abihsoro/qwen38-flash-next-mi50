#!/usr/bin/env python3
"""S1 check 05 — Triton vector-add: exit 0, output verified vs torch reference."""
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, x + y, mask=mask)


def main() -> int:
    n = 1 << 20
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    ref = x + y
    add_kernel[(triton.cdiv(n, 1024),)](x, y, n, BLOCK=1024)
    torch.cuda.synchronize()
    err = (y - ref).abs().max().item()
    print(f"triton_add_max_abs_err={err:.3e}")
    ok = err < 1e-4
    print(f"triton version: {triton.__version__}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

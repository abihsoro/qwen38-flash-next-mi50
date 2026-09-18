#!/usr/bin/env python3
"""N-E2 (Triton half): empty-kernel launch latency in-guest via Triton."""
import sys
import time

import torch
import triton
import triton.language as tl


@triton.jit
def empty_k(x_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs)
    tl.store(x_ptr + offs, v)


def main() -> int:
    x = torch.zeros(64, device="cuda")
    empty_k[(1,)](x, BLOCK=64)  # JIT warmup
    torch.cuda.synchronize()
    N = 2000
    # back-to-back
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        empty_k[(1,)](x, BLOCK=64)
    torch.cuda.synchronize()
    b2b_us = (time.perf_counter() - t0) * 1e6 / N
    # round-trip (sync each)
    t0 = time.perf_counter()
    for _ in range(N):
        empty_k[(1,)](x, BLOCK=64)
        torch.cuda.synchronize()
    rt_us = (time.perf_counter() - t0) * 1e6 / N
    print(f"N-E2 Triton empty-kernel: back-to-back {b2b_us:.2f} us/launch, "
          f"round-trip {rt_us:.2f} us")
    return 0


if __name__ == "__main__":
    sys.exit(main())

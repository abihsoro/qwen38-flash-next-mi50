#!/usr/bin/env python3
"""count_copybuffer.py — count the copy/family kernels in a trace directory.

The decisive measurement for the skinny-GEMM causal test: does turning
VLLM_ROCM_USE_SKINNY_GEMM off remove the copyBuffer family?

Tests the GZIP STREAM, not the file size (TRAPS #19: the truncated first-pass traces look valid by
size and fail only on decompression).

Usage: count_copybuffer.py [tracedir]
"""
import glob
import gzip
import json
import os
import sys
from collections import defaultdict

TRACEDIR = sys.argv[<bus>] if len(sys.argv) > 1 else "<workdir>/tprof"
WATCH = {
    "copyBuffer": "copyBuffer",
    "float16tofloat32_copy": "fp16->fp32 copy",
    "float32tofloat16_copy": "fp32->fp16 copy",
    "float16_copy_kernel": "float16 copy",
    "direct_copy_kernel": "direct copy",
    "CUDAFunctorOnSelf_add": "add (self, fp32)",
    "CUDAFunctor_add": "add",
    "FillFunctor": "fill",
    "MulFunctor": "mul",
    "MeanOps": "mean/reduce",
    "rsqrt": "rsqrt",
    "pow_tensor_scalar": "pow",
}


def valid(p):
    try:
        with gzip.open(p, "rt") as fh:
            while fh.read(1 << 20):
                pass
        return True
    except Exception:
        return False


files = [f for f in sorted(glob.glob(os.path.join(TRACEDIR, "**", "*.json.gz"), recursive=True))
         if "rank" in os.path.basename(f)]
good = [f for f in files if valid(f)]
print(f"traces: {len(files)} found, {len(good)} valid (of which rank traces)")
if not good:
    print("NO VALID TRACES - cannot count (treat as measurement failure, not as zero)")
    raise SystemExit(2)

counts = defaultdict(int)
times = defaultdict(float)
kernel_total = 0.0
kernel_count = 0
for f in good:
    with gzip.open(f, "rt") as fh:
        d = json.load(fh)
    for e in d.get("traceEvents", []):
        if e.get("ph") != "X" or e.get("cat") not in ("kernel", "gpu_op", "Kernel"):
            continue
        nm = e.get("name") or ""
        dur = float(e.get("dur", 0.0))
        kernel_total += dur
        kernel_count += 1
        for pat, label in WATCH.items():
            if pat in nm:
                counts[label] += 1
                times[label] += dur

traced_tokens = int(os.environ.get("TRACED_TOKENS", "32"))
print(f"kernel events: {kernel_count}  total {kernel_total/1000:.1f} ms")
print(f"profiled window: ~{traced_tokens} tokens\n")
print(f"{'launches':>10} {'total ms':>10} {'per tok':>9}  kernel family")
for label in WATCH.values():
    if counts[label] or label == "copyBuffer":
        # divide by traced_tokens AND by the number of rank traces: the count spans all ranks
    per = counts[label] / max(traced_tokens * len(good), 1)
        print(f"{counts[label]:10d} {times[label]/1000:10.2f} {per:9.2f}  {label}")

cb = counts["copyBuffer"]
print(f"\ncopyBuffer launches = {cb}  ({cb/max(traced_tokens*len(good),1):.1f} per token per rank, "
      f"over {len(good)} rank traces x {traced_tokens} tokens)")
print("REFERENCE: with skinny GEMM ON the earlier full-request profile showed ~93 copyBuffer/token/rank")

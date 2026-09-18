#!/usr/bin/env python3
"""p5_attrib_v3.py — kernel -> launching op attribution with THREE linkage paths, measured separately.

Decoded from the trace (p5_link_discover.py):
    kernel.args = {"External id": E, "correlation": C, "bytes": ..., "stream": ...}
    * E matches the Ev Idx of the launching cpu_op            -> path A
    * E ALSO appears on the cuda_runtime (hipMemcpyAsync etc.) event that is on the CPU tid
      -> containment on that tid recovers the op              -> path B
    * C matches a flow event ph='s' cat='ac2g' whose (ts,tid) is the launching CPU thread
      -> containment there                                    -> path C

v2 scored 13.1% because it only had A, and its containment fallback compared GPU kernel tids
against cpu_op tids (kernels are on stream tids, ops on CPU tids) so it could never match.
This version measures each path's yield so a weak attribution cannot pass as a strong one.
"""
from __future__ import annotations

import argparse
import bisect
import glob
import gzip
import json
import os
import re
import sys
from collections import defaultdict

FAMILY = re.compile(
    r"copyBuffer|vectorized_elementwise|elementwise_kernel|CUDAFunctor|FillFunctor|"
    r"float16tofloat32|float32tofloat16|float16_copy|direct_copy|CatArrayBatchedCopy|"
    r"index_elementwise|_scatter_gather|reduce_kernel|TensorIterator", re.I)
COPYISH = re.compile(r"copyBuffer|tofloat32|float16_copy|direct_copy|CatArrayBatchedCopy", re.I)


def load(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as fh:
            return json.load(fh)
    with open(path) as fh:
        return json.load(fh)


def valid(path) -> bool:
    try:
        with gzip.open(path, "rt") as fh:
            while fh.read(1 << 20):
                pass
        return True
    except Exception:
        return False


def kernel_op(nm: str) -> str:
    m = re.search(r"(CUDAFunctorOnSelf_add|CUDAFunctor_add|FillFunctor|"
                  r"float16tofloat32_copy_kernel_cuda|float32tofloat16_copy_kernel_cuda|"
                  r"float16_copy_kernel_cuda|direct_copy_kernel_cuda|MeanOps|ReduceOp|"
                  r"pow_tensor_scalar_kernel_impl|rsqrt_kernel_cuda|sigmoid_kernel_cuda|"
                  r"MulFunctor|DivFunctor)", nm)
    if "copyBuffer" in nm:
        return "copyBuffer"
    return m.group(1) if m else nm[:50]


class Intervals:
    def __init__(self):
        self.by_tid = defaultdict(list)

    def add(self, tid, s, e, name):
        self.by_tid[tid].append((s, e, name))

    def build(self):
        self.starts = {}
        for t, v in self.by_tid.items():
            v.sort()
            self.starts[t] = [x[<bus>] for x in v]

    def find(self, tid, ts):
        lst = self.by_tid.get(tid)
        if not lst:
            return None
        i = bisect.bisect_right(self.starts.get(tid, []), ts) - 1
        best = None
        while i >= 0:
            s, e, nm = lst[i]
            if s <= ts <= e:
                if best is None or (e - s) < (best[<bus>] - best[<bus>]):
                    best = (s, e, nm)
                i -= 1
                if ts - s > 2e6:
                    break
            else:
                break
        return best[<bus>] if best else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tracedir", nargs="?", default="<workdir>/tprof")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    files = [f for f in sorted(glob.glob(os.path.join(args.tracedir, "**", "*.json.gz"),
                                         recursive=True))
             if "rank" in os.path.basename(f)]
    good = [f for f in files if valid(f)]
    print(f"rank traces: {len(files)} (valid {len(good)}; invalid = truncated first pass)")

    path_hits = defaultdict(int)
    fam_count = 0
    fam_us = 0.0
    kernel_us = 0.0
    copy_us = 0.0
    copy_count = 0
    attribution = defaultdict(lambda: [0.0, 0])
    copy_by_op = defaultdict(lambda: [0.0, 0])

    for f in good:
        try:
            d = load(f)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {os.path.basename(f)}: {e}")
            continue
        evs = d.get("traceEvents", [])

        evidx_name = {}        # Ev Idx -> name           (path A target)
        runtime_by_ext = {}    # External id -> (tid, ts) (path B target)
        flow_start = {}        # correlation id -> (tid, ts) (path C target)
        ops = Intervals()

        for e in evs:
            if e.get("ph") != "X":
                continue
            a = e.get("args") or {}
            idx = a.get("Ev Idx")
            if idx is not None:
                evidx_name[idx] = e.get("name") or "?"
            cat = e.get("cat")
            if cat == "cuda_runtime":
                ext = a.get("External id")
                if ext is not None and e.get("ts") is not None:
                    runtime_by_ext.setdefault(ext, (e.get("tid"), e.get("ts")))
            if cat in ("cpu_op", "user_annotation", "python_function"):
                ts, dur = e.get("ts"), e.get("dur", 0)
                if ts is not None:
                    ops.add(e.get("tid"), ts, ts + dur, e.get("name") or "?")
            if e.get("ph") == "s" and cat == "ac2g":
                flow_start[e.get("id")] = (e.get("tid"), e.get("ts"))
        ops.build()

        for e in evs:
            if e.get("ph") != "X" or e.get("cat") not in ("kernel", "gpu_op", "Kernel"):
                continue
            nm = e.get("name") or ""
            dur = float(e.get("dur", 0.0))
            kernel_us += dur
            if not FAMILY.search(nm):
                continue
            fam_count += 1
            fam_us += dur
            a = e.get("args") or {}
            ext = a.get("External id")
            corr = a.get("correlation")

            op = None
            if ext is not None:
                op = evidx_name.get(ext)
                if op:
                    path_hits["A: External id -> Ev Idx"] += 1
            if op is None and ext is not None and ext in runtime_by_ext:
                tid, ts = runtime_by_ext[ext]
                op = ops.find(tid, ts)
                if op:
                    path_hits["B: External id -> cuda_runtime -> containment"] += 1
            if op is None and corr is not None and corr in flow_start:
                tid, ts = flow_start[corr]
                op = ops.find(tid, ts)
                if op:
                    path_hits["C: correlation -> flow start -> containment"] += 1
            if op is None:
                path_hits["UNRESOLVED"] += 1
                op = "<unresolved>"

            ko = kernel_op(nm)
            attribution[(op, ko)][<bus>] += dur
            attribution[(op, ko)][<bus>] += 1
            if COPYISH.search(nm):
                copy_us += dur
                copy_count += 1
                copy_by_op[op][<bus>] += dur
                copy_by_op[op][<bus>] += 1

        del evs, d

    tot = sum(path_hits.values())
    print(f"\n=== LINKAGE RESOLUTION (family launches {fam_count}) ===")
    for k, v in sorted(path_hits.items(), key=lambda kv: -kv[<bus>]):
        print(f"  {k:48} {v:8d}  {100*v/max(tot,1):5.1f}%")
    print(f"  {'TOTAL RESOLVED':48} {tot - path_hits['UNRESOLVED']:8d}  "
          f"{100*(tot-path_hits['UNRESOLVED'])/max(tot,1):5.1f}%")
    print(f"\n  family time   : {fam_us/1000:.1f} ms = {100*fam_us/kernel_us:.2f}% of kernel time")
    print(f"  copyish subset: {copy_count} launches, {copy_us/1000:.1f} ms = "
          f"{100*copy_us/kernel_us:.2f}% of kernel = ~{100*copy_us/kernel_us*0.855:.2f}% of wall")

    print(f"\n=== TOP {args.top} (parent op -> kernel op) BY TIME ===")
    for (op, ko), (us, n) in sorted(attribution.items(), key=lambda kv: -kv[<bus>][<bus>])[: args.top]:
        o = op if len(op) <= 52 else op[:49] + "..."
        print(f"{us/1000:8.2f} ms {n:8d} {us/max(n,1):7.2f} us {100*us/kernel_us:5.2f}% | {o} -> {ko}")

    print(f"\n=== COPYISH BY PARENT OP ===")
    for op, (us, n) in sorted(copy_by_op.items(), key=lambda kv: -kv[<bus>][<bus>])[: args.top]:
        o = op if len(op) <= 78 else op[:75] + "..."
        print(f"{us/1000:8.2f} ms {n:8d} {100*us/kernel_us:5.2f}% | {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

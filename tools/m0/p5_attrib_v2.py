#!/usr/bin/env python3
"""p5_attrib_v2.py — close the attribution gap: link family kernels to their launching ops.

D145/D146 used only `args["External id"]`, which resolved ~10% of launches - too weak to choose a
target on. This version:

  1. DISCOVERS the trace's linkage mechanism (flow events? correlation ids? nothing?) rather than
     assuming, and reports what is actually available.
  2. Falls back to TIME-CONTAINMENT on the same tid: the parent op is the innermost cpu_op whose
     [ts, ts+dur] contains the kernel's ts. That works regardless of whether the profiler emitted
     correlation metadata, and is the standard approach for chrome traces.
  3. Reports the resolution rate explicitly, so a weak attribution cannot masquerade as a strong one.
  4. Rolls the family up by (parent op -> kernel functor), which is what identifies WHERE to cut.

Usage: p5_attrib_v2.py [tracedir]
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
from collections import Counter, defaultdict

FAMILY = re.compile(
    r"copyBuffer|vectorized_elementwise|elementwise_kernel|elementwise_kernel_manual_unroll|"
    r"CUDAFunctor|FillFunctor|float16tofloat32|float32tofloat16|float16_copy|direct_copy|"
    r"CatArrayBatchedCopy|index_elementwise|_scatter_gather|reduce_kernel|TensorIterator",
    re.I,
)
COPYISH = re.compile(r"copyBuffer|tofloat32|float16_copy|direct_copy|CatArrayBatchedCopy", re.I)


def load(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as fh:
            return json.load(fh)
    with open(path) as fh:
        return json.load(fh)


def classify_kernel(nm: str) -> str:
    """Reduce a kernel name to the operation it implements, for grouping."""
    m = re.search(r"(CUDAFunctorOnSelf_add|CUDAFunctor_add|FillFunctor|"
                  r"float16tofloat32_copy_kernel_cuda|float32tofloat16_copy_kernel_cuda|"
                  r"float16_copy_kernel_cuda|direct_copy_kernel_cuda|"
                  r"MeanOps|ReduceOp|pow_tensor_scalar_kernel_impl|rsqrt_kernel_cuda|"
                  r"sigmoid_kernel_cuda|MulFunctor|DivFunctor|CopyFunctor)", nm)
    if "copyBuffer" in nm:
        return "copyBuffer (rocclr)"
    return m.group(1) if m else nm[:60]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tracedir", nargs="?", default="<workdir>/tprof")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    files = [f for f in sorted(glob.glob(os.path.join(args.tracedir, "**", "*.json.gz"),
                                         recursive=True))
             if "rank" in os.path.basename(f)]
    if not files:
        print(f"no rank traces under {args.tracedir}")
        return 2
    print(f"rank traces: {len(files)}")

    phcat = Counter()
    flow_ids = set()
    discovered = False
    ext_link = 0
    contain_link = 0
    unresolved = 0
    kernel_total = 0.0
    fam_total = 0.0
    fam_count = 0
    copy_total = 0.0
    copy_count = 0

    # (op -> kernel_op) -> [us, count]
    attribution: dict[tuple[str, str], list] = defaultdict(lambda: [0.0, 0])
    # op -> [us, count] for the copyish subset only (the >2% hypothesis)
    copy_by_op: dict[str, list] = defaultdict(lambda: [0.0, 0])
    file_stats = []

    for f in files:
        try:
            d = load(f)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {os.path.basename(f)}: {e}")
            continue
        events = d.get("traceEvents", [])
        for ev in events:
            phcat[(ev.get("ph"), ev.get("cat"))] += 1
            if ev.get("ph") in ("s", "f"):
                flow_ids.add(ev.get("id"))

        if not discovered:
            print("\n=== DISCOVERY: (ph, cat) counts ===")
            for (ph, cat), n in phcat.most_common(14):
                print(f"  ph={ph!r:6} cat={cat!r:22} {n}")
            print(f"  flow events present: {len(flow_ids) > 0} (distinct ids {len(flow_ids)})")
            discovered = True

        # --- build cpu_op interval index per tid (for containment) ---
        ops_by_tid: dict[int, list] = defaultdict(list)
        evidx_name: dict[int, str] = {}
        for ev in events:
            if ev.get("ph") != "X":
                continue
            a = ev.get("args") or {}
            if a.get("Ev Idx") is not None:
                evidx_name[a["Ev Idx"]] = ev.get("name") or "?"
            if ev.get("cat") in ("cpu_op", "user_annotation", "python_function"):
                ts = ev.get("ts")
                dur = ev.get("dur", 0)
                if ts is not None:
                    ops_by_tid[ev.get("tid")].append((ts, ts + dur, ev.get("name") or "?"))

        for tid in ops_by_tid:
            ops_by_tid[tid].sort()
        starts: dict[int, list] = {t: [o[<bus>] for o in v] for t, v in ops_by_tid.items()}

        def parent_of(tid, ts, name):
            """innermost cpu_op interval containing ts on this tid"""
            lst = ops_by_tid.get(tid)
            if not lst:
                return None
            i = bisect.bisect_right(starts.get(tid, []), ts) - 1
            best = None
            while i >= 0:
                s, e, nm = lst[i]
                if s <= ts <= e:
                    if best is None or (e - s) < (best[<bus>] - best[<bus>]):
                        best = (s, e, nm)
                    i -= 1
                    if ts - s > 5e6:      # stop scanning absurdly far back (5 s)
                        break
                else:
                    break
            return best[<bus>] if best else None

        f_kernel_total = 0.0
        f_fam_total = 0.0
        f_fam_count = 0

        for ev in events:
            if ev.get("ph") != "X":
                continue
            if ev.get("cat") not in ("kernel", "gpu_op", "Kernel"):
                continue
            nm = ev.get("name") or ""
            dur = float(ev.get("dur", 0.0))
            f_kernel_total += dur
            kernel_total += dur
            if not FAMILY.search(nm):
                continue
            f_fam_total += dur
            f_fam_count += 1
            fam_total += dur
            fam_count += 1

            is_copy = bool(COPYISH.search(nm))
            if is_copy:
                copy_total += dur
                copy_count += 1

            # linkage 1: External id -> Ev Idx
            ext = (ev.get("args") or {}).get("External id")
            op = evidx_name.get(ext) if ext is not None else None
            if op:
                ext_link += 1
            else:
                op = parent_of(ev.get("tid"), ev.get("ts"), nm)
                if op:
                    contain_link += 1
                else:
                    unresolved += 1
            if op is None:
                op = "<unresolved>"

            ko = classify_kernel(nm)
            attribution[(op, ko)][<bus>] += dur
            attribution[(op, ko)][<bus>] += 1
            if is_copy:
                copy_by_op[op][<bus>] += dur
                copy_by_op[op][<bus>] += 1

        file_stats.append((os.path.basename(f), f_fam_count, f_fam_total / 1000))

    resolved = ext_link + contain_link
    print(f"\n=== LINKAGE RESOLUTION ===")
    print(f"  family launches           : {fam_count}")
    print(f"  resolved via External id  : {ext_link} ({100*ext_link/max(fam_count,1):.1f}%)")
    print(f"  resolved via containment  : {contain_link} ({100*contain_link/max(fam_count,1):.1f}%)")
    print(f"  UNRESOLVED                : {unresolved} ({100*unresolved/max(fam_count,1):.1f}%)")
    print(f"  TOTAL RESOLVED            : {100*resolved/max(fam_count,1):.1f}%")
    print(f"  family time               : {fam_total/1000:.1f} ms "
          f"= {100*fam_total/kernel_total:.1f}% of kernel time")
    print(f"  copyish subset            : {copy_count} launches, {copy_total/1000:.1f} ms "
          f"= {100*copy_total/kernel_total:.1f}% of kernel time "
          f"= ~{100*copy_total/kernel_total*0.855:.1f}% of wall")

    print(f"\n=== TOP {args.top} (parent op -> kernel op) BY TIME ===")
    print(f"{'total ms':>9} {'count':>8} {'us each':>8} {'%kern':>6}  op -> kernel")
    for (op, ko), (us, n) in sorted(attribution.items(), key=lambda kv: -kv[<bus>][<bus>])[: args.top]:
        o = op if len(op) <= 58 else op[:55] + "..."
        print(f"{us/1000:9.2f} {n:8d} {us/max(n,1):8.2f} {100*us/kernel_total:5.1f}%  {o} -> {ko}")

    print(f"\n=== COPYISH BY PARENT OP (the >2% hypothesis) ===")
    print(f"{'total ms':>9} {'count':>8} {'%kern':>6}  {100*copy_total/kernel_total:.1f}% total")
    for op, (us, n) in sorted(copy_by_op.items(), key=lambda kv: -kv[<bus>][<bus>])[: args.top]:
        o = op if len(op) <= 80 else op[:77] + "..."
        print(f"{us/1000:9.2f} {n:8d} {100*us/kernel_total:5.1f}%  {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

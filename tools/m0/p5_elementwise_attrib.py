#!/usr/bin/env python3
"""p5_elementwise_attrib.py — focused attribution of the elementwise / copyBuffer residue.

WHY: D143 put `elementwise` at 11.8% of TP4 kernel time across 303,380 launches, dominated by
`__amd_rocclr_copyBuffer` (47,526 launches) and `vectorized_elementwise_kernel` add/cast/fill
variants. Launch-count-dominated means a different failure mode from the bandwidth-saturated dense
GEMV work (D145), so it has a plausible path to a >2% measurable wall win.

WHAT: groups those kernels by their FULL name - the template/functor suffix is what identifies the
operation, e.g.
    at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<float>, ...>
                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ the op
and reports count, total time, per-call time and share of kernel time, so ONE target can be chosen
on evidence rather than by picking the biggest vague bucket.

Also attempts parent-op attribution via the profiler's External id linkage, reporting what it finds
so the correlation's reliability is visible rather than assumed.

Usage: p5_elementwise_attrib.py [tracedir] [--top N]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
import sys
from collections import defaultdict

FAMILY = re.compile(
    r"copyBuffer|vectorized_elementwise|elementwise_kernel|CUDAFunctor|FillFunctor|"
    r"float16tofloat32|float32tofloat16|direct_copy|CatArrayBatchedCopy|index_|Index|"
    r"at::native::copy|reduce_kernel|TensorIterator",
    re.I,
)


def load(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as fh:
            return json.load(fh)
    with open(path) as fh:
        return json.load(fh)


def short(name: str, width: int = 150) -> str:
    return name if len(name) <= width else name[: width - 3] + "..."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tracedir", nargs="?", default="<workdir>/tprof")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    files = [f for f in sorted(glob.glob(os.path.join(args.tracedir, "**", "*.json.gz"),
                                         recursive=True))
             if "rank" in os.path.basename(f)]
    if not files:
        print(f"no rank traces under {args.tracedir}")
        return 2
    print(f"rank traces: {len(files)}")

    by_name: dict[str, list] = defaultdict(lambda: [0.0, 0])      # name -> [us, count]
    by_parent: dict[str, list] = defaultdict(lambda: [0.0, 0])    # parent op -> [us, count]
    kernel_total = 0.0
    kernel_count = 0
    evidx_to_name: dict[int, str] = {}
    linked = 0

    for f in files:
        try:
            d = load(f)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {os.path.basename(f)}: {e}")
            continue
        events = d.get("traceEvents", [])

        # pass 1: map Ev Idx -> event name for cpu_op / annotation events (candidate parents)
        for ev in events:
            if ev.get("ph") != "X":
                continue
            a = ev.get("args") or {}
            idx = a.get("Ev Idx")
            if idx is not None:
                evidx_to_name[idx] = ev.get("name") or "?"

        # pass 2: kernels
        for ev in events:
            if ev.get("ph") != "X":
                continue
            nm = ev.get("name") or ""
            cat = ev.get("cat", "")
            dur = float(ev.get("dur", 0.0))
            if cat in ("kernel", "gpu_op", "Kernel"):
                kernel_total += dur
                kernel_count += 1
                if not FAMILY.search(nm):
                    continue
                by_name[nm][<bus>] += dur
                by_name[nm][<bus>] += 1
                # try to attribute to a parent op
                ext = (ev.get("args") or {}).get("External id")
                parent = evidx_to_name.get(ext)
                if parent:
                    by_parent[parent][<bus>] += dur
                    by_parent[parent][<bus>] += 1
                    linked += 1

    fam_us = sum(v[<bus>] for v in by_name.values())
    fam_n = sum(v[<bus>] for v in by_name.values())
    print(f"\nkernel events: {kernel_count}  total {kernel_total/1000:.1f} ms")
    print(f"FAMILY (elementwise/copy residue): {fam_n} launches, {fam_us/1000:.1f} ms "
          f"= {100*fam_us/kernel_total:.1f}% of kernel time")
    print(f"  distinct kernel names: {len(by_name)}")
    print(f"  parent-op linkage resolved for {linked}/{fam_n} launches "
          f"({100*linked/max(fam_n,1):.0f}%)")

    print(f"\n=== TOP {args.top} BY TOTAL TIME ===")
    print(f"{'total ms':>9} {'count':>8} {'us each':>8} {'%kern':>6}  kernel")
    for nm, (us, n) in sorted(by_name.items(), key=lambda kv: -kv[<bus>][<bus>])[: args.top]:
        print(f"{us/1000:9.2f} {n:8d} {us/max(n,1):8.2f} {100*us/kernel_total:5.1f}%  {short(nm)}")

    print(f"\n=== TOP {args.top} BY LAUNCH COUNT ===")
    print(f"{'count':>8} {'total ms':>9} {'us each':>8} {'%kern':>6}  kernel")
    for nm, (us, n) in sorted(by_name.items(), key=lambda kv: -kv[<bus>][<bus>])[: args.top]:
        print(f"{n:8d} {us/1000:9.2f} {us/max(n,1):8.2f} {100*us/kernel_total:5.1f}%  {short(nm)}")

    if by_parent:
        print(f"\n=== TOP PARENT OPS (where the linkage resolved) ===")
        print(f"{'total ms':>9} {'count':>8} {'%kern':>6}  parent op")
        for nm, (us, n) in sorted(by_parent.items(), key=lambda kv: -kv[<bus>][<bus>])[:15]:
            print(f"{us/1000:9.2f} {n:8d} {100*us/kernel_total:5.1f}%  {short(nm, 110)}")

    print("\n=== WALL-EFFECT BUDGET ===")
    print(f"  kernel time is ~85.5% of wall (D143). So 1% of kernel time ~= 0.855% of wall.")
    print(f"  To clear the 2% wall adoption bar, a target must be >= "
          f"{2/0.855:.1f}% of kernel time.")
    print(f"  Largest single name above = {max(v[<bus>] for v in by_name.values())/1000:.1f} ms "
          f"= {100*max(v[<bus>] for v in by_name.values())/kernel_total:.1f}% of kernel time.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

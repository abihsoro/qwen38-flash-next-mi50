#!/usr/bin/env python3
"""p5_gemv_shapes.py — identify the exact shapes feeding gemv_f16_rdna2_<8,1> from the D143 trace.

WHY: D143 showed gemv_f16_rdna2_<8,1> = 1253 ms / 50,691 launches (~24.7 us each), the single
largest kernel family contributor. Before microbenching variants we need the REAL shapes.

HOW: the launcher selects the variant by N and encodes N in the grid:
    N >= 1152 -> <8,MT> grid=(N+7)/8 block=512     <- this is <8,1>, the profile's hot kernel
    N >=  576 -> <4,MT> grid=(N+3)/4 block=256
    N >=  288 -> <2,MT> grid=(N+1)/2 block=128
    else      -> <1,MT> grid=N       block=64
  so grid.x recovers N and block recovers the variant. M is the MT template arg (1 here), and
  K is x.size(1) which we recover by correlating with the trace's cpu_op shapes where available.

Outputs the shapes ranked by TOTAL TIME, which is what actually matters for the ledger.
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import sys
from collections import defaultdict

TRACEDIR = "<workdir>/tprof"


def load(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as fh:
            return json.load(fh)
    with open(path) as fh:
        return json.load(fh)


def main() -> int:
    files = sorted(glob.glob(os.path.join(TRACEDIR, "**", "*.json.gz"), recursive=True))
    files = [f for f in files if "rank" in os.path.basename(f)]
    if not files:
        print(f"no rank traces under {TRACEDIR}")
        return 2
    print(f"rank traces: {len(files)}")

    # (template, grid_x, block, shared) -> [total_us, count, k_seen]
    agg: dict[tuple, list] = defaultdict(lambda: [0.0, 0, set()])
    name_forms: dict[str, int] = defaultdict(int)
    arg_keys_seen: set[str] = set()
    dumped = 0

    for f in files:
        try:
            d = load(f)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {os.path.basename(f)}: {e}")
            continue
        for ev in d.get("traceEvents", []):
            nm = ev.get("name") or ""
            if "gemv_f16_rdna2" not in nm:
                continue
            if ev.get("ph") != "X":
                continue
            dur = float(ev.get("dur", 0.0))
            args = ev.get("args") or {}
            if dumped < 2:
                arg_keys_seen.update(args.keys())
                print(f"\n  SAMPLE EVENT: name={nm}")
                print(f"    dur={dur:.1f} us  args={json.dumps(args)[:400]}")
                dumped += 1

            grid = args.get("grid") or args.get("Grid") or []
            block = args.get("block") or args.get("Block") or []
            shared = args.get("shared memory") or args.get("sharedMemory") or 0
            gx = grid[<bus>] if isinstance(grid, (list, tuple)) and grid else -1
            bx = block[<bus>] if isinstance(block, (list, tuple)) and block else -1

            # template marker, e.g. gemv_f16_rdna2_<8, 1>
            tpl = "?"
            if "<" in nm:
                tpl = nm[nm.index("<"):].split(">")[<bus>].replace(" ", "") + ">"
            name_forms[tpl] += 1
            agg[(tpl, gx, bx, shared)][<bus>] += dur
            agg[(tpl, gx, bx, shared)][<bus>] += 1

    if not agg:
        print("no gemv_f16_rdna2 events found")
        return 3

    print(f"\nname forms seen: {dict(name_forms)}")
    print(f"arg keys available: {sorted(arg_keys_seen)}")

    def implied_N(tpl, gx, bx):
        if bx == 512 or tpl.startswith("<8"):
            return gx * 8 if gx > 0 else None
        if bx == 256 or tpl.startswith("<4"):
            return gx * 4 if gx > 0 else None
        if bx == 128 or tpl.startswith("<2"):
            return gx * 2 if gx > 0 else None
        return gx if gx > 0 else None

    rows = []
    for (tpl, gx, bx, sh), (us, n, _) in agg.items():
        rows.append((us, n, tpl, gx, bx, sh, implied_N(tpl, gx, bx)))
    rows.sort(reverse=True)
    total_us = sum(r[<bus>] for r in rows)
    total_n = sum(r[<bus>] for r in rows)

    print(f"\n=== gemv_f16_rdna2 SHAPES BY TOTAL TIME (M=1 decode) ===")
    print(f"{'total ms':>10} {'count':>8} {'us/each':>8} {'tpl':>10} {'grid.x':>8} {'block':>6} "
          f"{'shared':>7} {'implied N':>10}")
    for us, n, tpl, gx, bx, sh, Nimp in rows[:25]:
        print(f"{us/1000:10.2f} {n:8d} {us/max(n,1):8.2f} {tpl:>10} {gx:8d} {bx:6d} {sh:7d} "
              f"{str(Nimp):>10}")
    print(f"\ntotal: {total_us/1000:.1f} ms across {total_n} launches")

    print("\n=== AGGREGATED BY IMPLIED N (top by time) ===")
    byN: dict = defaultdict(lambda: [0.0, 0])
    for us, n, tpl, gx, bx, sh, Nimp in rows:
        key = (Nimp, bx)
        byN[key][<bus>] += us
        byN[key][<bus>] += n
    for (Nimp, bx), (us, n) in sorted(byN.items(), key=lambda kv: -kv[<bus>][<bus>])[:15]:
        print(f"  N={str(Nimp):>7} block={bx:<5} {us/1000:9.2f} ms  {n:7d} launches  "
              f"{100*us/total_us:5.1f}% of gemv time")
    return 0


if __name__ == "__main__":
    sys.exit(main())

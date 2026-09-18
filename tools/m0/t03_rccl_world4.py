#!/usr/bin/env python3
"""t03_rccl_world4.py — REAL RCCL world=N all-reduce latency sweep (Phase 2 floor).

The repo's tools/m0/t03_rccl_sweep.py is a scaffold (prints PASS, measures nothing);
this is the executable version. One process per GPU via torchrun.

Protocol discipline (TP4_BRINGUP_PLAN Phase 2):
  * sizes: 5 KB, 20 KB, 80 KB + a spread of plausible hidden-state sizes
  * N >= 50 repeats, median + IQR reported (I5 discipline)
  * an out-of-band correctness check on every size (sum of rank+1 values)
  * GPU-side timing via CUDA events, not CPU wall clock

Usage: torchrun --nproc_per_node=4 t03_rccl_world4.py [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import torch
import torch.distributed as dist


def iqr(vals):
    if len(vals) < 4:
        return -1.0
    s = sorted(vals)
    n = len(s)

    def q(p):
        idx = p * (n - 1)
        lo = int(idx)
        f = idx - lo
        hi = s[lo + 1] if lo + 1 < n else s[lo]
        return s[lo] + f * (hi - s[lo])

    return q(0.75) - q(0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    dev = f"cuda:{local_rank}"

    if rank == 0:
        print(f"RCCL world={world} devices={torch.cuda.device_count()} "
              f"torch={torch.__version__} hip={torch.version.hip}")

    # sizes in bytes (fp16 => /2 elements)
    sizes_bytes = [5 * 1024, 20 * 1024, 80 * 1024, 160 * 1024, 320 * 1024,
                   640 * 1024, 1280 * 1024]
    expect = float(sum(range(1, world + 1)))

    results = {}
    for sb in sizes_bytes:
        n = sb // 2
        t = torch.empty(n, dtype=torch.float16, device=dev)
        for _ in range(args.warmup):
            t.fill_(rank + 1)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

        lats = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(args.repeats):
            t.fill_(rank + 1)
            torch.cuda.synchronize()
            start.record()
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            end.record()
            torch.cuda.synchronize()
            lats.append(start.elapsed_time(end) * 1000.0)  # ms -> us

        ok = bool(torch.all(t == expect).item())
        med = statistics.median(lats)
        results[sb] = {
            "median_us": round(med, 2),
            "iqr_us": round(iqr(lats), 2),
            "min_us": round(min(lats), 2),
            "repeats": args.repeats,
            "correct": ok,
        }
        if rank == 0:
            print(f"  {sb/1024:8.0f} KB  median={med:9.2f} us  "
                  f"iqr={iqr(lats):7.2f}  min={min(lats):8.2f}  correct={ok}")

    if rank == 0 and args.json:
        with open(args.json, "w") as fh:
            json.dump({"world": world, "backend": "rccl",
                       "torch": torch.__version__, "hip": torch.version.hip,
                       "repeats": args.repeats, "sizes_bytes": sizes_bytes,
                       "results": {str(k): v for k, v in results.items()}}, fh, indent=2)
        print(f"wrote {args.json}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

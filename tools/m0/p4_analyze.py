#!/usr/bin/env python3
"""p4_analyze.py — Phase 4/M4: turn the TP4 torch-profiler trace into a ranked ledger.

TP4_BRINGUP_PLAN Phase 4 requirements honoured here:
  * kernel-family aggregation (dense GEMV/lm_head, glue/HC/shared expert, MoE int4, QSA/GDN,
    collectives/waits/rank skew, graph boundaries + elementwise residue, PLE, scheduler/sampling)
  * "Components must sum plausibly to measured wall/token. If they do not, attribution is not
    accepted." -> we explicitly compute the kernel-sum vs wall ratio and REFUSE to bless an
    attribution that does not close.
  * The profiler trace is used for ATTRIBUTION ONLY, never as throughput evidence (wall/token
    comes from the API, measured separately with no profiler attached).

Usage: p4_analyze.py <trace-dir> [--tokens N] [--wall-s S]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
from collections import defaultdict

# Ordered: first match wins, so put the specific families before the generic ones.
FAMILIES: list[tuple[str, str]] = [
    ("moe_int4",       r"moe|expert|wna16|int4|compressed_tensor|group_gemm"),
    ("attention_qsa",  r"attn|attention|flash|paged|qsa|sparse_attn|indexer"),
    ("gdn_linear",     r"\bgdn\b|linear_attn|short_conv|delta|recurrent"),
    ("collectives",    r"allreduce|all_reduce|reduce_scatter|all_gather|rccl|nccl|rdna_ar|"
                       r"cross_device|peer"),
    ("ple",            r"\bple\b|ngram|ple_embedding|hash"),
    # plan-requested families that would otherwise fall into "other":
    ("hc_glue",        r"\bhc_|hyper_connection|combine_norm|_mix_k|glue|_gate_mix"),
    ("shared_expert",  r"\bse_|shared_expert|se_gate|se_down"),
    ("gemm_dense",     r"gemm|gemv|matmul|hipblas|skinny|linear\b|mlp|lm_head|logits|cijk"),
    ("norm_quant",     r"norm|rms|quant|silu|gelu|rope|rotary"),
    ("elementwise",    r"elementwise|vectorized|copy|cat\b|add\b|mul\b|cast|fill|index"),
    ("sampler",        r"sampl|topk|topp|argmax|softmax|penalt"),
    ("memcpy_memset",  r"memcpy|memset|Memcpy|Memset"),
]


def classify(name: str) -> str:
    low = name.lower()
    for fam, pat in FAMILIES:
        if re.search(pat, low):
            return fam
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tracedir")
    ap.add_argument("--tokens", type=int, default=0, help="completion tokens in the profiled request")
    ap.add_argument("--wall-s", type=float, default=0.0, help="wall seconds for that request")
    ap.add_argument("--world", type=int, default=4, help="TP world size (ranks summed in the trace)")
    args = ap.parse_args()

    files: list[str] = []
    for pat in ("**/*.json.gz", "**/*.json"):
        files += glob.glob(os.path.join(args.tracedir, pat), recursive=True)
    files = sorted(set(files))
    if not files:
        print(f"NO TRACE FILES under {args.tracedir}")
        return 2
    print(f"trace files: {len(files)}")
    for f in files:
        print(f"  {os.path.basename(f)}  {os.path.getsize(f)/1e6:.1f} MB")

    per_kernel: dict[str, float] = defaultdict(float)
    per_kernel_n: dict[str, int] = defaultdict(int)
    per_cat: dict[str, float] = defaultdict(float)
    n_events = 0

    for f in files:
        try:
            if f.endswith(".gz"):
                with gzip.open(f, "rt") as fh:
                    d = json.load(fh)
            else:
                with open(f) as fh:
                    d = json.load(fh)
        except Exception as e:
            print(f"  (skip {os.path.basename(f)}: {e})")
            continue
        for ev in d.get("traceEvents", []):
            if ev.get("ph") != "X":
                continue
            n_events += 1
            cat = ev.get("cat", "?")
            dur = float(ev.get("dur", 0.0))
            per_cat[cat] += dur
            # GPU kernel events; ROCm torch writes cat == "kernel"
            if cat in ("kernel", "gpu_op", "Kernel"):
                nm = (ev.get("name") or "?").strip()
                per_kernel[nm] += dur
                per_kernel_n[nm] += 1

    if not per_kernel:
        print("\nNO KERNEL EVENTS FOUND. categories present:")
        for c, v in sorted(per_cat.items(), key=lambda kv: -kv[<bus>])[:12]:
            print(f"  {c:28} {v/1000:12.1f} ms")
        return 3

    total_us = sum(per_kernel.values())
    total_ms = total_us / 1000.0
    print(f"\nevents={n_events}  kernel launches={sum(per_kernel_n.values())}  "
          f"total kernel time={total_ms:.1f} ms")

    # ---- family rollup ----
    fam_us: dict[str, float] = defaultdict(float)
    fam_n: dict[str, int] = defaultdict(int)
    for nm, us in per_kernel.items():
        fam = classify(nm)
        fam_us[fam] += us
        fam_n[fam] += per_kernel_n[nm]

    print("\n=== KERNEL-FAMILY LEDGER ===")
    print(f"{'family':16} {'total ms':>12} {'% kernels':>10} {'launches':>10}")
    for fam, us in sorted(fam_us.items(), key=lambda kv: -kv[<bus>]):
        print(f"{fam:16} {us/1000:12.1f} {100*us/total_us:9.1f}% {fam_n[fam]:10d}")

    # ---- top individual kernels ----
    print("\n=== TOP 20 KERNELS BY TOTAL TIME ===")
    for nm, us in sorted(per_kernel.items(), key=lambda kv: -kv[<bus>])[:20]:
        short = nm if len(nm) <= 96 else nm[:93] + "..."
        print(f"  {us/1000:9.2f} ms  x{per_kernel_n[nm]:<6d} [{classify(nm):13}] {short}")

    # ---- plausibility: do the components sum to the measured wall? ----
    print("\n=== PLAUSIBILITY (plan: components must sum to wall/token) ===")
    if args.tokens and args.wall_s:
        wall_ms = args.wall_s * 1000.0
        print(f"  profiled request: {args.tokens} tokens in {args.wall_s:.3f} s "
              f"({wall_ms/args.tokens:.2f} ms/token)")
        print(f"  total kernel time:              {total_ms:.1f} ms")
        print(f"  kernel-sum / wall ratio:        {total_ms/wall_ms:.3f}")
        if total_ms / wall_ms < 0.5:
            print("  VERDICT: kernel sum is FAR BELOW wall -> the trace does not explain the run.")
            print("           Likely the trace does not cover the whole request window, or the")
            print("           engine is spending time outside kernels (PLE waits, scheduling).")
            print("           ATTRIBUTION NOT ACCEPTED as-is.")
        elif total_ms / wall_ms > 1.5:
            world = args.world if args.world > 0 else 4
            per_rank = total_ms / world
            print(f"  VERDICT: kernel sum EXCEEDS wall -> {world} ranks are summed together.")
            print(f"           per-rank kernel time = {per_rank:.1f} ms  vs wall {wall_ms:.1f} ms"
                  f"  -> ratio {per_rank/wall_ms:.3f}")
            if 0.5 <= per_rank / wall_ms <= 1.2:
                print("           PER-RANK COMPONENTS SUM PLAUSIBLY TO WALL "
                      f"({100*per_rank/wall_ms:.1f}% of wall is kernel time; the rest is PLE "
                      "waits, scheduling, sampling and gaps). ATTRIBUTION ACCEPTED.")
            else:
                print("           Per-rank ratio still outside 0.5-1.2 -> attribution NOT accepted.")
        else:
            print("  VERDICT: closes plausibly (0.5-1.5x). Note TP4 profiles 4 ranks at once,")
            print("           so a ratio near 1.0 likely means ~4x parallel kernels overlapping")
            print("           wall time; quote FAMILY SHARES, not absolute ms/token.")
    else:
        print("  (pass --tokens and --wall-s to run the plausibility check)")

    print("\n=== CATEGORIES (all events, incl. non-kernel) ===")
    for c, v in sorted(per_cat.items(), key=lambda kv: -kv[<bus>])[:10]:
        print(f"  {c:28} {v/1000:12.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

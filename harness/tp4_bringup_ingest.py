#!/usr/bin/env python3
"""tp4_bringup_ingest.py — ingest TP4 bring-up artifacts (Phase 1/2) via results.py.

TP4_BRINGUP_PLAN.md runbook step 1 asks for `results/tp4_bringup/`. Invariant I3 says
only harness code writes under `results/`, so this script is the writer: it reads the
raw measurement JSONs produced on the rig and emits them as harness artifacts.

Usage:
    python3 harness/tp4_bringup_ingest.py --dir <dir-with-jsons> [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import results  # noqa: E402


def load(path: Path):
    if not path.exists():
        return None, "absent"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh), None
    except Exception as exc:
        return None, f"unreadable: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory holding the raw JSON artifacts")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    d = Path(args.dir).expanduser().resolve()
    problems: list[str] = []
    written: list[str] = []

    # ---- Phase 1: fabric / peer matrix ----
    fab, e_fab = load(d / "p2p_<host>.json")
    if fab:
        recs = []
        for p in fab.get("pairs", []):
            recs.append({"kind": "peer_pair", "phase": "phase1_fabric", **p})
        for s in fab.get("self_copy", []):
            recs.append({"kind": "self_copy", "phase": "phase1_fabric", **s})
        for h in fab.get("host_transfer", []):
            recs.append({"kind": "host_transfer", "phase": "phase1_fabric", **h})
        recs.append({"kind": "summary", "phase": "phase1_fabric",
                     "device_count": fab.get("device_count"),
                     "devices": fab.get("devices"),
                     "repeats": fab.get("repeats")})
        if not args.dry_run:
            p = results.write_artifact("tp4_bringup/fabric_matrix.jsonl", recs, force=True)
            written.append(str(p.relative_to(results.ROOT)))
        print(f"fabric_matrix: {len(recs)} records ({len(fab.get('pairs', []))} pairs)")
    else:
        problems.append(f"p2p_<host>.json: {e_fab}")

    # ---- Phase 2: RCCL world=4 ----
    rccl, e_rccl = load(d / "rccl_world4.json")
    if rccl:
        recs = []
        for sz, r in rccl.get("results", {}).items():
            recs.append({"kind": "allreduce_latency", "phase": "phase2_collectives",
                         "impl": "rccl", "world": rccl.get("world"),
                         "bytes": int(sz), "median_us": r.get("median_us"),
                         "iqr_us": r.get("iqr_us"), "min_us": r.get("min_us"),
                         "repeats": r.get("repeats"), "correct": r.get("correct")})
        recs.append({"kind": "summary", "phase": "phase2_collectives", "impl": "rccl",
                     "world": rccl.get("world"), "torch": rccl.get("torch"),
                     "hip": rccl.get("hip"), "backend": rccl.get("backend")})
        if not args.dry_run:
            p = results.write_artifact("tp4_bringup/rccl_world4.jsonl", recs, force=True)
            written.append(str(p.relative_to(results.ROOT)))
        print(f"rccl_world4: {len(recs)} records")
    else:
        problems.append(f"rccl_world4.json: {e_rccl}")

    # ---- Phase 2: custom one-shot AR (recorded from the rig run) ----
    ar, e_ar = load(d / "ar4_results.json")
    if ar:
        recs = [{"kind": "allreduce_latency", "phase": "phase2_collectives",
                 "impl": "custom_wave64_oneshot", **r} for r in ar.get("runs", [])]
        if not args.dry_run:
            p = results.write_artifact("tp4_bringup/custom_ar_world4.jsonl", recs, force=True)
            written.append(str(p.relative_to(results.ROOT)))
        print(f"custom_ar_world4: {len(recs)} records")
    else:
        problems.append(f"ar4_results.json: {e_ar}")

    print("problems:", problems if problems else "none")
    for w in written:
        print("  wrote", w)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

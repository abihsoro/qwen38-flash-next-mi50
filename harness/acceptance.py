#!/usr/bin/env python3
"""acceptance.py — the ONLY code path that may emit an accept/reject verdict
(I5/D048). Reads steps_per_second only. tokens_per_second is never consulted.

Refuses (exit 2 / verdict REFUSE):
  - schema_version: 1 rows (readable but inadmissible - D048)
  - thermally_valid is False (invalid rows are excluded, never averaged - I5.2)
  - shape_profile "full" rows submitted for a TP4 acceptance decision (D050)
  - an MTP-on comparison unless both sides carry accepted_tokens_per_step (I5.1)

Accepts iff all three hold (I5.3): median steps_per_second improvement >= 3%;
95% CIs of before and after do not overlap; N >= 15 thermally-valid repeats per
side. tp4_rank0 rows are projected by tp4_projection_factor (default 0.83)
BEFORE the 3% threshold (I5.4).

Usage:
  python3 harness/acceptance.py --before FILE --after FILE [--tp4] [--mtp-on]
    FILE = a JSONL results file (results/<task>.jsonl); the calculator selects
    the thermally-valid rows.
"""
import argparse
import json
import statistics
import sys


def _load_rows(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _refuse(reason: str) -> int:
    print(f"REFUSE: {reason}")
    return 2


def verdict(before_rows: list[dict], after_rows: list[dict],
            tp4: bool, mtp_on: bool) -> int:
    for side, rows in (("before", before_rows), ("after", after_rows)):
        for r in rows:
            if r.get("schema_version", 1) == 1:
                return _refuse(f"{side} row {r.get('run_id')}: schema_version 1 (inadmissible)")
            if r.get("thermally_valid") is False:
                return _refuse(f"{side} row {r.get('run_id')}: thermally_valid false")
            if tp4 and r.get("shape_profile") == "full":
                return _refuse(f"{side} row {r.get('run_id')}: full profile inadmissible for TP4 (D050)")
        if not rows:
            return _refuse(f"{side}: no rows")
        if len(rows) < 15:
            return _refuse(f"{side}: {len(rows)} rows < N=15")
        if any(r.get("steps_per_second") is None for r in rows):
            return _refuse(f"{side}: rows missing steps_per_second")
        if mtp_on and any(r.get("accepted_tokens_per_step") is None for r in rows):
            return _refuse(f"{side}: MTP-on comparison needs accepted_tokens_per_step on both sides")

    def project(r: dict) -> float:
        s = r["steps_per_second"]
        if r.get("shape_profile") == "tp4_rank0":
            s *= r.get("tp4_projection_factor", 0.83)
        return s

    b = sorted(project(r) for r in before_rows)
    a = sorted(project(r) for r in after_rows)
    med_b, med_a = statistics.median(b), statistics.median(a)
    iqr_b = statistics.quantiles(b, n=4)[<bus>] - statistics.quantiles(b, n=4)[<bus>]
    iqr_a = statistics.quantiles(a, n=4)[<bus>] - statistics.quantiles(a, n=4)[<bus>]
    # 95% CI via the median +/- 1.58 * IQR / sqrt(N) (notched-box approx; N>=15)
    import math
    ci_b = 1.58 * iqr_b / math.sqrt(len(b))
    ci_a = 1.58 * iqr_a / math.sqrt(len(a))
    improvement = (med_a - med_b) / med_b
    overlap = (med_b + ci_b) > (med_a - ci_a) and (med_a + ci_a) > (med_b - ci_b)
    print(f"before: median {med_b:.3f} steps/s (N={len(b)}, IQR {iqr_b:.3f})")
    print(f"after:  median {med_a:.3f} steps/s (N={len(a)}, IQR {iqr_a:.3f})")
    print(f"improvement {improvement * 100:+.2f}% (need >= +3%) | CI overlap: {overlap}")
    if improvement < 0.03:
        print("REJECT: improvement < 3%")
        return 1
    if overlap:
        print("REJECT: 95% CIs overlap")
        return 1
    print("ACCEPT")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--tp4", action="store_true",
                    help="this is a TP4 acceptance decision (full-profile rows refused)")
    ap.add_argument("--mtp-on", action="store_true")
    args = ap.parse_args()
    try:
        b = _load_rows(args.before)
        a = _load_rows(args.after)
    except FileNotFoundError as e:
        print(f"REFUSE: {e}")
        return 2
    return verdict(b, a, args.tp4, args.mtp_on)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env bash
# rehearse_day1.sh — dry-run of the M0-M3 day-one sequence at gpu_count=1.
# Tests the MECHANICS (unattended run, clean skips, schema-valid rows, timing),
# not the results. Peer paths take their "requires N GPUs" skips.
set -u
cd "$(dirname "$0")"
T0=$(date +%s)
echo "=== day-one rehearsal at $(date) ==="
N=$(rocm-smi --showproductname 2>/dev/null | grep -oE "GPU\[[<bus-range>]+\]" | sort -u | wc -l)
echo "GPU count: $N (peer paths will skip if < 2)"

echo "--- [1/4] M-track suite (run_m0.sh) ---"
T1=$(date +%s)
PATH=~/gfx906-venv/bin:$PATH bash run_m0.sh 2>&1 | grep -E "^---|PASS|SKIP|FAIL|passed|skipped|failed"
T1d=$(($(date +%s) - T1)); echo "run_m0.sh: ${T1d}s"

echo "--- [2/4] kernel-measurement rows present ---"
for f in ../../results/N-D1.jsonl ../../results/N-D4.jsonl ../../results/TP4-RANK0.jsonl; do
  [ -f "$f" ] && echo "  ok: $f ($(wc -l < "$f") rows)" || echo "  MISSING: $f"
done

echo "--- [3/4] surrogate serve startup (measured ~6-8 min on this box) ---"
echo "  (rehearsal notes the earlier measured startups: eager ~7 min; graphs-on ~9 min)"

echo "--- [4/4] row-writing: schema-valid rows via the harness ---"
PATH=~/gfx906-venv/bin:$PATH python3 write_rehearsal_row.py
Td=$(($(date +%s) - T0))
echo "=== rehearsal total: ${Td}s ==="

#!/usr/bin/env bash
# run_m0.sh — M-track day-one entry (D063). Detects GPUs, runs the suite,
# reports PASS/SKIP/FAIL. Single-GPU: the degenerate modes run, N-GPU tests
# skip. Four cards: everything runs.
set -u
cd "$(dirname "$0")"
N=$(rocm-smi --showproductname 2>/dev/null | grep -oE "GPU\[[<bus-range>]+\]" | sort -u | wc -l)
echo "M0 suite: $N GPU(s) detected"
for t in t01_p2p_matrix t02_hip_ipc t03_rccl_sweep t04_ar_oneshot t05_comparator; do
  echo "--- $t ---"
  python3 "$t.py" || true
done

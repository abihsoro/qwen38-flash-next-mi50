#!/usr/bin/env bash
# run_all.sh — execute all nine S1 toolchain-gate checks on the GPU guest
# (<vm>) via harness/run_task.py, so every check writes a results row with an
# explicit timeout (I2). Run on the guest where the repo is synced:
#   ~/qwen38-flash-next-mi50  (repo copy; run from there)
#   ~/s1-venv                 (venv with torch 2.12.1+rocm7.2 + triton 2.3.1)
#
# Exit code: 0 iff every check passed; prints a summary. Rows land in
# results/S1.jsonl on the guest and are rsync'd back to the main repo.
#
# VENV_PY selects the venv python (default: the gfx906 community-stack venv).
set -u

cd "$(dirname "${BASH_SOURCE[<bus>]}")/../.." || exit 2
PY="${VENV_PY:-$HOME/gfx906-venv/bin/python3}"
[ -x "$PY" ] || { echo "run_all: venv python missing: $PY (set VENV_PY)"; exit 2; }

# check -> timeout seconds
TO01=300; TO02=300; TO03=300; TO04=600; TO05=600; TO06=900; TO07=900; TO08=600; TO09=600

fail=0
run_one() {
  local n="$1" t="$2"
  echo "=== S1 check $n (timeout ${t}s) ==="
  if python3 harness/run_task.py --task S1 --timeout "$t" -- "$PY" "harness/s1_checks/check${n}_"*.py; then
    echo "--- check $n PASS"
  else
    echo "--- check $n FAIL"
    fail=1
  fi
}

run_one 01 "$TO01"
run_one 02 "$TO02"
run_one 03 "$TO03"
run_one 04 "$TO04"
run_one 05 "$TO05"
run_one 06 "$TO06"
run_one 07 "$TO07"
run_one 08 "$TO08"
run_one 09 "$TO09"

echo "=== G1 verdict: $([ $fail -eq 0 ] && echo 'all nine checks exit 0' || echo 'at least one check failed') ==="
exit $fail

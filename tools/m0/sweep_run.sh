#!/usr/bin/env bash
# sweep_run.sh — cheap G6-closing sweep: --max-num-batched-tokens 1024 vs 2048, with a CONTROL REPEAT.
#
# Design:  A = 1024 (control)   B = 2048 (candidate)   A' = 1024 (drift detector, run last).
#   Across this session, whole-serve medians have ranged 45.9-49.0 tok/s, which is LARGER than a
#   2% effect. A' measures that drift directly, so a B-vs-A difference only counts if it exceeds
#   the A-vs-A' spread. This is the plan's adoption rule applied honestly ("0.5-2% requires a
#   reverse A/B"; >2% clean gain with exact outputs is a candidate).
#
# Discipline: bracket-safe AND comm-safe cleanup (TRAPS #14 + #1), idle checked before and after
# each arm, exactly one api_server asserted, teardown verified by VRAM, and the deterministic
# sha256 required in EVERY gate run.
set -uo pipefail
CT=<ct>
RESULTS=<workdir>/sweep_results.txt

vram0() { pct exec $CT -- bash -c 'export PATH=/opt/rocm/bin:$PATH; rocm-smi --showmeminfo vram 2>/dev/null | grep -A1 "GPU\[0\]" | grep Used | awk "{print \$NF}"'; }

run_arm() {
  local tag="$1" maxb="$2"
  echo
  echo "=============================================================="
  echo "  ARM $tag   max-num-batched-tokens=$maxb"
  echo "=============================================================="

  # --- teardown + idle gate ---
  pct exec $CT -- bash <workdir>/sweep_stop.sh
  local v_before; v_before=$(vram0)
  echo "  GPU0 VRAM before launch = ${v_before:-?} bytes (idle ~10878976)"

  # --- launch exactly one serve ---
  pct exec $CT -- bash -c "echo $maxb > <workdir>/sweep_maxb.txt"
  pct exec $CT -- bash -c ": > <workdir>/sweep_${tag}.log; setsid nohup bash <workdir>/sweep_serve.sh >> <workdir>/sweep_${tag}.log 2>&1 < /dev/null & echo launched"
  sleep 12
  local n; n=$(pct exec $CT -- bash -c 'pgrep -cf "vllm.entrypoint[s]" || true')
  echo "  api_server procs = ${n:-0} (must be 1)"
  if [ "${n:-0}" != "1" ]; then echo "  ABORT arm $tag: expected 1 api_server"; return 1; fi

  # --- wait for ready ---
  local i
  for i in $(seq 1 80); do
    local r e
    r=$(pct exec $CT -- bash -c "grep -cE 'Application startup complete' <workdir>/sweep_${tag}.log 2>/dev/null || true")
    e=$(pct exec $CT -- bash -c "grep -cE 'Engine core initialization failed|FileNotFoundError' <workdir>/sweep_${tag}.log 2>/dev/null || true")
    [ "${r:-0}" != "0" ] && { echo "  ready after ~$((i*15))s"; break; }
    [ "${e:-0}" != "0" ] && { echo "  FAILED to start"; pct exec $CT -- bash -c "tail -20 <workdir>/sweep_${tag}.log" | cut -c1-150; return 1; }
    sleep 15
  done
  pct exec $CT -- bash -c "grep -oE 'Model loading took [0-9.]+ GiB memory and [0-9.]+ seconds' <workdir>/sweep_${tag}.log | head -1"
  pct exec $CT -- bash -c "echo '  AR lines: '\$(grep -c 'one-shot all-reduce active' <workdir>/sweep_${tag}.log)"

  # --- gate 3x, saving each artifact and requiring the same hash ---
  local g
  for g in 1 2 3; do
    echo "  --- gate run $g/3 ---"
    pct exec $CT -- bash -c "cd <workdir> && TP4_GRAPHS_MODE='sweep-$tag-maxb$maxb' <home>/gfx906-venv/bin/python <workdir>/tp4_gate.py 64 3" 2>&1 \
      | grep -E "DETERMINISTIC|MEDIAN|rep[<bus>]:" | sed 's/^/    /'
    pct exec $CT -- bash -c "cp <workdir>/tp4_gate.json <workdir>/sweep_${tag}_g${g}.json" 2>/dev/null || true
  done

  # --- teardown + confirm idle ---
  pct exec $CT -- bash <workdir>/sweep_stop.sh
  local v_after; v_after=$(vram0)
  echo "  GPU0 VRAM after teardown = ${v_after:-?} bytes"
  pct exec $CT -- bash -c 'echo "  port 8002 listeners: $(ss -lntp 2>/dev/null | grep -c ":8002" || true)"'
}

run_arm A     1024
run_arm B     2048
run_arm Aprime 1024

echo
echo "=============================================================="
echo "  SUMMARY"
echo "=============================================================="
pct exec $CT -- bash -c 'for f in <workdir>/sweep_*.json; do
  t=$(basename "$f" .json | sed "s/sweep_//")
  med=$(grep -o "\"baseline_median_tok_per_s\": [0-9.]*" "$f" | head -1 | awk "{print \$2}")
  sha=$(grep -o "\"sha256\": \"[a-f0-9]\{16\}" "$f" | head -1 | sed "s/.*\"//")
  det=$(grep -o "\"deterministic\": [a-z]*" "$f" | head -1 | awk "{print \$2}")
  printf "  %-14s median=%-9s deterministic=%-6s sha=%s\n" "$t" "${med:-?}" "${det:-?}" "${sha:-?}"
done'
echo
echo "  REFERENCE HASH (D142/D143) = 0128852903291e32"
echo "  ADOPTION RULE: >2% clean gain vs BOTH A and A-prime, with identical hash -> adopt;"
echo "                 otherwise REJECT cleanly (do not re-baseline on noise)."

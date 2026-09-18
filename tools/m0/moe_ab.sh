#!/usr/bin/env bash
# moe_ab.sh — Phase 5, target 1: MoE autotune then a controlled A/B/A' with the tuned config.
#
# Stage 1: run the autotune (moe_tune.sh), then VERIFY the produced filename equals what the
#          serve will look for. If the names disagree the serve silently ignores the config and
#          the A/B would measure nothing - so this is a hard gate, not a nicety.
# Stage 2: A (control, no folder) / B (tuned folder) / A' (control, drift detector) - the same
#          three-arm design used for the max-num-batched-tokens sweep, because the rig's
#          whole-serve noise floor is ~1% and the adoption rule is >2%.
#
# Acceptance: identical deterministic sha256 (0128852903291e32) in every gate run, and a gain that
# clears >2% against BOTH controls. Otherwise reject cleanly - no re-baselining on noise.
set -uo pipefail
CT=<ct>
TUNED=<workdir>/moe-tuned-gfx906
EXPECT='E=128,N=640,device_name=AMD_Radeon_Graphics,dtype=int4_w4a16.json'

vram0() { pct exec $CT -- bash -c 'export PATH=/opt/rocm/bin:$PATH; rocm-smi --showmeminfo vram 2>/dev/null | grep -A1 "GPU\[0\]" | grep Used | awk "{print \$NF}"'; }

# ---------------------------------------------------------------- stage 1: autotune
echo "=============================================================="
echo "  STAGE 1: MoE autotune"
echo "=============================================================="
pct exec $CT -- bash <workdir>/sweep_stop.sh
pct exec $CT -- bash -c 'rm -rf <workdir>/moe-tuned-gfx906; : > <workdir>/moe_tune.log; setsid nohup bash <workdir>/moe_tune.sh >> <workdir>/moe_tune.log 2>&1 < /dev/null & echo launched'

echo "  waiting for the tuner (up to ~50 min; it prints a per-M table as it goes)"
for i in $(seq 1 200); do
  DONE=$(pct exec $CT -- bash -c 'ls <workdir>/moe-tuned-gfx906/*.json 2>/dev/null | wc -l' || true)
  ALIVE=$(pct exec $CT -- bash -c 'pgrep -cf "benchmark_moe" || true')
  if [ "${DONE:-0}" != "0" ] && [ "${ALIVE:-0}" = "0" ]; then
    echo "  tuner finished after ~$((i*15))s; produced ${DONE} json file(s)"
    break
  fi
  if [ $((i % 8)) -eq 0 ]; then
    echo "  [$((i*15))s] alive=${ALIVE:-0} json=${DONE:-0} | $(pct exec $CT -- bash -c 'tail -1 <workdir>/moe_tune.log 2>/dev/null' | cut -c1-110)"
  fi
  sleep 15
done

echo
echo "  --- tuner output tail ---"
pct exec $CT -- bash -c 'tail -18 <workdir>/moe_tune.log 2>/dev/null' | cut -c1-160
echo
echo "  --- files produced ---"
pct exec $CT -- bash -c 'ls -la <workdir>/moe-tuned-gfx906/ 2>/dev/null'

# HARD GATE: the filename must match what the serve will compute
echo
echo "  --- FILENAME GATE ---"
echo "  serve will look for: $EXPECT"
GOT=$(pct exec $CT -- bash -c "ls $TUNED/ 2>/dev/null | head -1")
echo "  tuner produced:      ${GOT:-<nothing>}"
if [ "$GOT" != "$EXPECT" ]; then
  echo "  *** MISMATCH -> the serve would silently ignore the config. STOPPING before the A/B. ***"
  exit 2
fi
echo "  OK: names match"

# ---------------------------------------------------------------- stage 2: A / B / A'
run_arm() {
  local tag="$1" folder="$2"
  echo
  echo "=============================================================="
  echo "  ARM $tag   VLLM_TUNED_CONFIG_FOLDER=[${folder:-<unset = CONTROL>}]"
  echo "=============================================================="
  pct exec $CT -- bash <workdir>/sweep_stop.sh
  echo "  GPU0 VRAM before launch = $(vram0) bytes"
  pct exec $CT -- bash -c "printf '%s' '$folder' > <workdir>/moe_tuned_folder.txt"
  pct exec $CT -- bash -c ": > <workdir>/moeab_${tag}.log; setsid nohup bash <workdir>/moe_serve.sh >> <workdir>/moeab_${tag}.log 2>&1 < /dev/null & echo launched"
  sleep 12
  local n; n=$(pct exec $CT -- bash -c 'pgrep -cf "vllm.entrypoint[s]" || true')
  echo "  api_server procs = ${n:-0} (must be 1)"
  [ "${n:-0}" = "1" ] || { echo "  ABORT arm $tag"; return 1; }

  local i
  for i in $(seq 1 80); do
    local r e
    r=$(pct exec $CT -- bash -c "grep -cE 'Application startup complete' <workdir>/moeab_${tag}.log 2>/dev/null || true")
    e=$(pct exec $CT -- bash -c "grep -cE 'Engine core initialization failed|FileNotFoundError' <workdir>/moeab_${tag}.log 2>/dev/null || true")
    [ "${r:-0}" != "0" ] && { echo "  ready after ~$((i*15))s"; break; }
    [ "${e:-0}" != "0" ] && { echo "  FAILED"; pct exec $CT -- bash -c "tail -20 <workdir>/moeab_${tag}.log" | cut -c1-150; return 1; }
    sleep 15
  done
  pct exec $CT -- bash -c "grep -oE 'Model loading took [0-9.]+ GiB memory and [0-9.]+ seconds' <workdir>/moeab_${tag}.log | head -1"
  echo "  AR lines: $(pct exec $CT -- bash -c "grep -c 'one-shot all-reduce active' <workdir>/moeab_${tag}.log")"
  echo "  MoE config line (THE POINT):"
  pct exec $CT -- bash -c "grep -iE 'Using configuration from|Using default MoE config' <workdir>/moeab_${tag}.log | head -2" | cut -c1-170 | sed 's/^/    /'

  local g
  for g in 1 2 3; do
    echo "  --- gate $g/3 ---"
    pct exec $CT -- bash -c "cd <workdir> && TP4_GRAPHS_MODE='moe-$tag' <home>/gfx906-venv/bin/python <workdir>/tp4_gate.py 64 3" 2>&1 \
      | grep -E "DETERMINISTIC|MEDIAN|rep[<bus>]:" | sed 's/^/    /'
    pct exec $CT -- bash -c "cp <workdir>/tp4_gate.json <workdir>/moeab_${tag}_g${g}.json" 2>/dev/null || true
  done

  pct exec $CT -- bash <workdir>/sweep_stop.sh
  echo "  GPU0 VRAM after teardown = $(vram0) bytes"
}

run_arm A      ""       # CONTROL: folder unset  -> hits the default MoE config
run_arm B      "$TUNED" # TUNED: folder set      -> should log "Using configuration from ..."
run_arm Aprime ""       # CONTROL REPEAT: measures drift, so B must clear it to count

echo
echo "=============================================================="
echo "  SUMMARY"
echo "=============================================================="
pct exec $CT -- bash -c 'for f in <workdir>/moeab_*.json; do
  t=$(basename "$f" .json | sed "s/moeab_//")
  med=$(grep -o "\"baseline_median_tok_per_s\": [0-9.]*" "$f" | head -1 | awk "{print \$2}")
  sha=$(grep -o "\"sha256\": \"[a-f0-9]\{16\}" "$f" | head -1 | sed "s/.*\"//")
  det=$(grep -o "\"deterministic\": [a-z]*" "$f" | head -1 | awk "{print \$2}")
  printf "  %-12s median=%-9s deterministic=%-6s sha=%s\n" "$t" "${med:-?}" "${det:-?}" "${sha:-?}"
done'
echo
echo "  REFERENCE HASH = 0128852903291e32 | rule: >2% vs BOTH controls with identical hash -> adopt."

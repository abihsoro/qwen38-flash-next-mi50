#!/usr/bin/env bash
# moe_ab2.sh — A/B/A' for the tuned MoE config. The JSON already exists, so this is measurement
# only (no tuning). Same three-arm design as the earlier sweeps because the whole-serve noise
# floor on this rig is ~1% and the adoption rule is >2%.
#
# Critically, each arm greps for the MoE config log line: arm A must show "Using default MoE
# config", arm B must show "Using configuration from <path>". Without that, a flat result would be
# indistinguishable from the config simply never being loaded.
set -uo pipefail
CT=<ct>
TUNED=<workdir>/moe-tuned-gfx906
EXPECT='E=128,N=640,device_name=AMD_Radeon_Graphics,dtype=int4_w4a16.json'
REF_SHA=0128852903291e32

vram0() { pct exec $CT -- bash -c 'export PATH=/opt/rocm/bin:$PATH; rocm-smi --showmeminfo vram 2>/dev/null | grep -A1 "GPU\[0\]" | grep Used | awk "{print \$NF}"'; }

echo "=== PRECONDITION: the tuned JSON exists under the exact expected name ==="
if ! pct exec $CT -- bash -c "test -f $TUNED/$EXPECT"; then
  echo "  MISSING $TUNED/$EXPECT - aborting (an A/B without it would measure nothing)"
  exit 2
fi
pct exec $CT -- bash -c "ls -la $TUNED/$EXPECT; echo '--- contents ---'; cat $TUNED/$EXPECT"

run_arm() {
  local tag="$1" folder="$2"
  echo
  echo "=============================================================="
  echo "  ARM $tag   VLLM_TUNED_CONFIG_FOLDER=[${folder:-<unset = CONTROL>}]"
  echo "=============================================================="
  pct exec $CT -- bash <workdir>/sweep_stop.sh
  echo "  GPU0 VRAM before launch = $(vram0) bytes (idle ~10878976)"
  pct exec $CT -- bash -c "printf '%s' '$folder' > <workdir>/moe_tuned_folder.txt"
  pct exec $CT -- bash -c ": > <workdir>/moeab2_${tag}.log; setsid nohup bash <workdir>/moe_serve.sh >> <workdir>/moeab2_${tag}.log 2>&1 < /dev/null & echo launched"
  sleep 12
  local n; n=$(pct exec $CT -- bash -c 'pgrep -cf "vllm.entrypoint[s]" || true')
  echo "  api_server procs = ${n:-0} (must be 1)"
  [ "${n:-0}" = "1" ] || { echo "  ABORT arm $tag"; return 1; }

  local i
  for i in $(seq 1 80); do
    local r e
    r=$(pct exec $CT -- bash -c "grep -cE 'Application startup complete' <workdir>/moeab2_${tag}.log 2>/dev/null || true")
    e=$(pct exec $CT -- bash -c "grep -cE 'Engine core initialization failed|FileNotFoundError' <workdir>/moeab2_${tag}.log 2>/dev/null || true")
    [ "${r:-0}" != "0" ] && { echo "  ready after ~$((i*15))s"; break; }
    [ "${e:-0}" != "0" ] && { echo "  FAILED to start"; pct exec $CT -- bash -c "tail -20 <workdir>/moeab2_${tag}.log" | cut -c1-150; return 1; }
    sleep 15
  done
  pct exec $CT -- bash -c "grep -oE 'Model loading took [0-9.]+ GiB memory and [0-9.]+ seconds' <workdir>/moeab2_${tag}.log | head -1"

  echo "  --- MoE config line (THE POINT OF THIS A/B) ---"
  pct exec $CT -- bash -c "grep -iE 'Using configuration from|Using default MoE config' <workdir>/moeab2_${tag}.log | head -2" \
    | cut -c1-175 | sed 's/^/    /'

  local g
  for g in 1 2 3; do
    echo "  --- gate $g/3 ---"
    pct exec $CT -- bash -c "cd <workdir> && TP4_GRAPHS_MODE='moe-ab-$tag' <home>/gfx906-venv/bin/python <workdir>/tp4_gate.py 64 3" 2>&1 \
      | grep -E "DETERMINISTIC|MEDIAN|rep[<bus>]:" | sed 's/^/    /'
    pct exec $CT -- bash -c "cp <workdir>/tp4_gate.json <workdir>/moeab2_${tag}_g${g}.json" 2>/dev/null || true
  done

  pct exec $CT -- bash <workdir>/sweep_stop.sh
  echo "  GPU0 VRAM after teardown = $(vram0) bytes"
}

run_arm A      ""
run_arm B      "$TUNED"
run_arm Aprime ""

echo
echo "=============================================================="
echo "  SUMMARY"
echo "=============================================================="
pct exec $CT -- bash -c 'for f in <workdir>/moeab2_*.json; do
  t=$(basename "$f" .json | sed "s/moeab2_//")
  med=$(grep -o "\"baseline_median_tok_per_s\": [0-9.]*" "$f" | head -1 | awk "{print \$2}")
  sha=$(grep -o "\"sha256\": \"[a-f0-9]\{16}" "$f" | head -1 | sed "s/.*\"//")
  det=$(grep -o "\"deterministic\": [a-z]*" "$f" | head -1 | awk "{print \$2}")
  printf "  %-12s median=%-9s deterministic=%-6s sha=%s\n" "$t" "${med:-?}" "${det:-?}" "${sha:-?}"
done'
echo
echo "  REFERENCE HASH = $REF_SHA"
echo "  ADOPTION RULE: >2% clean gain vs BOTH controls with identical hash -> adopt; else reject."
echo "  EXPECTANCY: this tunes the Triton MoE tile config, NOT topkGating."
echo "              A clean 2-5% wall gain is a win; >4.6% closes G6."

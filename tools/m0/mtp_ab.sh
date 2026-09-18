#!/usr/bin/env bash
# mtp_ab.sh — O5: MTP=3 vs MTP=0, A/B/A' with PP512 and TG128 measured.
#
# Arms: A = MTP0 (verified baseline) | B = MTP3 (candidate) | A' = MTP0 (drift check).
# Each arm: serve -> PP512/TG128 benchmark (3 reps) -> greedy determinism check.
#
# CORRECTNESS GATE: MTP is lossless speculative decoding, so a correct implementation must reproduce
# the MTP=0 greedy hash. DO-NOT-ATTEMPT.md records that num_speculative_tokens=4 CHANGED the target
# text on V620 and was rejected, so this is a real gate. Any hash change is an automatic reject
# regardless of speed - the operator's standing rule is that deterministic-output changes from a
# supposedly lossless change are rejected or re-baselined only with sign-off.
#
# ADOPTION: >2% on TG (the metric O5 exists to move) versus BOTH controls, with the hash unchanged.
# PP512 is recorded but is not the adoption metric - it is a check that MTP does not regress prefill.
set -uo pipefail
CT=<ct>

vram0() { pct exec $CT -- bash -c 'export PATH=/opt/rocm/bin:$PATH; rocm-smi --showmeminfo vram 2>/dev/null | grep -A1 "GPU\[0\]" | grep Used | awk "{print \$NF}"'; }

run_arm() {
  local tag="$1" mtp="$2"
  echo
  echo "=============================================================="
  echo "  ARM $tag   MTP=$mtp"
  echo "=============================================================="
  pct exec $CT -- bash <workdir>/sweep_stop.sh
  echo "  GPU0 VRAM before = $(vram0) bytes (idle ~10878976)"
  pct exec $CT -- bash -c "printf '%s' '$mtp' > <workdir>/mtp_val.txt"
  pct exec $CT -- bash -c ": > <workdir>/mtp_${tag}.log; setsid nohup bash <workdir>/mtp_serve.sh >> <workdir>/mtp_${tag}.log 2>&1 < /dev/null & echo launched"
  sleep 12
  local n; n=$(pct exec $CT -- bash -c 'pgrep -cf "vllm.entrypoint[s]" || true')
  echo "  api_server procs = ${n:-0} (must be 1)"
  [ "${n:-0}" = "1" ] || { echo "  ABORT arm $tag"; return 1; }

  local i
  for i in $(seq 1 80); do
    local r e
    r=$(pct exec $CT -- bash -c "grep -cE 'Application startup complete' <workdir>/mtp_${tag}.log 2>/dev/null || true")
    e=$(pct exec $CT -- bash -c "grep -cE 'Engine core initialization failed|FileNotFoundError|Traceback' <workdir>/mtp_${tag}.log 2>/dev/null || true")
    [ "${r:-0}" != "0" ] && { echo "  ready after ~$((i*15))s"; break; }
    [ "${e:-0}" != "0" ] && { echo "  FAILED to start - tail:"; pct exec $CT -- bash -c "tail -25 <workdir>/mtp_${tag}.log" | cut -c1-160; return 1; }
    sleep 15
  done
  pct exec $CT -- bash -c "grep -oE 'Model loading took [0-9.]+ GiB memory and [0-9.]+ seconds' <workdir>/mtp_${tag}.log | head -1"
  pct exec $CT -- bash -c "grep -oiE 'mtp=(0|3)[^)]*|speculative[^,)]*|num_speculative_tokens[^,}]*' <workdir>/mtp_${tag}.log | head -3" | sed 's/^/  log says: /'
  echo "  --- VRAM (MTP=0 should free ~2.5 GiB/card KV per PORT-MAP) ---"
  pct exec $CT -- bash -c 'export PATH=/opt/rocm/bin:$PATH; rocm-smi --showmeminfo vram 2>/dev/null | grep -A1 "GPU\[0\]" | head -2' | sed 's/^/  /'

  echo "  --- PP512 / TG128 benchmark (3 reps) ---"
  pct exec $CT -- bash -c "cd <workdir> && <home>/gfx906-venv/bin/python <workdir>/bench_pp_tg.py --reps 3 --tag '$tag' --json <workdir>/bench_${tag}.json" 2>&1 | sed 's/^/  /'

  echo "  --- determinism cross-check vs the standing gate ---"
  pct exec $CT -- bash -c "cd <workdir> && TP4_GRAPHS_MODE='mtp-$tag' <home>/gfx906-venv/bin/python <workdir>/tp4_gate.py 64 3" 2>&1 | grep -E "DETERMINISTIC|MEDIAN" | sed 's/^/  /'
  pct exec $CT -- bash -c "cp <workdir>/tp4_gate.json <workdir>/mtpgate_${tag}.json" 2>/dev/null || true

  pct exec $CT -- bash <workdir>/sweep_stop.sh
  echo "  GPU0 VRAM after = $(vram0) bytes"
}

run_arm A 0
run_arm B 3
run_arm Aprime 0

echo
echo "=============================================================="
echo "  SUMMARY"
echo "=============================================================="
pct exec $CT -- bash -c 'for f in <workdir>/bench_A.json <workdir>/bench_B.json <workdir>/bench_Aprime.json; do
  [ -f "$f" ] || { echo "  $(basename $f): MISSING"; continue; }
  <home>/gfx906-venv/bin/python -c "
import json,sys
d=json.load(open(\"$f\"))
print(f\"  {d[\\\"tag\\\"]:8} PP{d[\\\"prompt_tokens\\\"]}={d[\\\"pp\\\"]} tok/s   TG{d[\\\"gen\\\"]}={d[\\\"tg\\\"]} tok/s   (incl prompt {d[\\\"tg_incl_prompt\\\"]})   sha={d[\\\"sha256\\\"][:16]}\")
"
done'
echo
echo "  STANDING MTP=0 HASH (tp4_gate) = 0128852903291e32"
echo "  ADOPTION: >2% TG vs BOTH controls AND unchanged output hash. A changed hash is an"
echo "            automatic reject (MTP=4 precedent), regardless of speed."

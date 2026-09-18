#!/usr/bin/env bash
# skinny_ab.sh — DIRECT CAUSAL TEST: is the copyBuffer family caused by skinny-GEMM's
# per-call `x.reshape(-1, k).contiguous()`?
#
# QUESTION: VLLM_ROCM_USE_SKINNY_GEMM=0 removes the skinny path -> do the ~47.5k copyBuffer launches
# go away? The trace cannot answer this (13.1% parent attribution, D145 addendum), so it is settled
# causally here.
#
# ARMS: A = skinny ON (control, current baseline) | B = skinny OFF | A' = skinny ON (drift check).
# Each arm: serve -> gate x3 (throughput + deterministic hash, NO profiler active) -> one short
# profiled request -> count copyBuffer. The profiler is configured on every arm so the counts are
# comparable; the tok/s numbers come from the unprofiled gate, never from the trace.
#
# The whole tprof directory is CLEARED per arm (D143: stale traces were mixed into an analysis).
set -uo pipefail
CT=<ct>

vram0() { pct exec $CT -- bash -c 'export PATH=/opt/rocm/bin:$PATH; rocm-smi --showmeminfo vram 2>/dev/null | grep -A1 "GPU\[0\]" | grep Used | awk "{print \$NF}"'; }

run_arm() {
  local tag="$1" val="$2"
  echo
  echo "=============================================================="
  echo "  ARM $tag   VLLM_ROCM_USE_SKINNY_GEMM=$val"
  echo "=============================================================="
  pct exec $CT -- bash <workdir>/sweep_stop.sh
  echo "  GPU0 VRAM before = $(vram0) bytes (idle ~10878976)"

  # clear the trace dir so each arm's counts are its own (D143 mixed stale traces once)
  pct exec $CT -- bash -c 'rm -rf <workdir>/tprof/* 2>/dev/null; true'
  pct exec $CT -- bash -c "printf '%s' '$val' > <workdir>/skinny_val.txt"
  pct exec $CT -- bash -c ": > <workdir>/skinny_${tag}.log; setsid nohup bash <workdir>/skinny_serve.sh >> <workdir>/skinny_${tag}.log 2>&1 < /dev/null & echo launched"
  sleep 12
  local n; n=$(pct exec $CT -- bash -c 'pgrep -cf "vllm.entrypoint[s]" || true')
  echo "  api_server procs = ${n:-0} (must be 1)"
  [ "${n:-0}" = "1" ] || { echo "  ABORT arm $tag"; return 1; }

  local i
  for i in $(seq 1 80); do
    local r e
    r=$(pct exec $CT -- bash -c "grep -cE 'Application startup complete' <workdir>/skinny_${tag}.log 2>/dev/null || true")
    e=$(pct exec $CT -- bash -c "grep -cE 'Engine core initialization failed|FileNotFoundError' <workdir>/skinny_${tag}.log 2>/dev/null || true")
    [ "${r:-0}" != "0" ] && { echo "  ready after ~$((i*15))s"; break; }
    [ "${e:-0}" != "0" ] && { echo "  FAILED"; pct exec $CT -- bash -c "tail -20 <workdir>/skinny_${tag}.log" | cut -c1-150; return 1; }
    sleep 15
  done
  pct exec $CT -- bash -c "grep -oE 'Model loading took [0-9.]+ GiB memory and [0-9.]+ seconds' <workdir>/skinny_${tag}.log | head -1"
  pct exec $CT -- bash -c "grep -oE 'VLLM_ROCM_USE_SKINNY_GEMM=[<bus>]' <workdir>/skinny_${tag}.log | head -1" | sed 's/^/  confirmed: /'

  echo "  --- gate x3 (throughput + hash, NO profiler active) ---"
  local g
  for g in 1 2 3; do
    pct exec $CT -- bash -c "cd <workdir> && TP4_GRAPHS_MODE='skinny-$tag' <home>/gfx906-venv/bin/python <workdir>/tp4_gate.py 64 3" 2>&1 \
      | grep -E "DETERMINISTIC|MEDIAN" | sed 's/^/    /'
    pct exec $CT -- bash -c "cp <workdir>/tp4_gate.json <workdir>/skinny_${tag}_g${g}.json" 2>/dev/null || true
  done

  echo "  --- profiled 32-token request, then count copyBuffer ---"
  pct exec $CT -- bash -c 'curl -s -m 30 -X POST http://127.0.0.1:8002/start_profile -o /dev/null -w "    start_profile http=%{http_code}\n"'
  pct exec $CT -- bash -c 'cd <workdir> && <home>/gfx906-venv/bin/python - <<PY
import json,time,urllib.request
body={"model":"qwen38-flash-next","prompt":"Explain the difference between a PCIe switch and a PCIe bridge in exactly three sentences.","max_tokens":32,"temperature":0.0,"top_p":1.0,"ignore_eos":True,"seed":0}
req=urllib.request.Request("http://127.0.0.1:8002/v1/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
d=json.loads(urllib.request.urlopen(req,timeout=900).read().decode())
print("    profiled request:",d.get("usage",{}).get("completion_tokens"),"tokens")
PY'
  pct exec $CT -- bash -c 'curl -s -m 900 -X POST http://127.0.0.1:8002/stop_profile -o /dev/null -w "    stop_profile http=%{http_code}\n"'

  # wait for the flush, then verify integrity (D143: a short timeout truncated all traces once)
  local prev=""
  for i in $(seq 1 60); do
    local sz; sz=$(pct exec $CT -- bash -c 'cat <workdir>/tprof/*.json.gz 2>/dev/null | wc -c')
    if [ -n "$prev" ] && [ "$prev" = "$sz" ] && [ "${sz:-0}" != "0" ]; then echo "    traces stable at $sz bytes"; break; fi
    prev="$sz"; sleep 10
  done
  pct exec $CT -- bash -c 'for f in <workdir>/tprof/*.json.gz; do gzip -t "$f" 2>/dev/null || echo "    CORRUPT $(basename $f)"; done'
  pct exec $CT -- bash -c 'TRACED_TOKENS=32 <home>/gfx906-venv/bin/python <workdir>/count_copybuffer.py <workdir>/tprof' | sed 's/^/  /'

  pct exec $CT -- bash <workdir>/sweep_stop.sh
  echo "  GPU0 VRAM after = $(vram0) bytes"
  pct exec $CT -- bash -c 'echo "  port 8002 listeners: $(ss -lntp 2>/dev/null | grep -c ":8002" || true)"'
}

run_arm A 1
run_arm B 0
run_arm Aprime 1

echo
echo "=============================================================="
echo "  SUMMARY"
echo "=============================================================="
pct exec $CT -- bash -c 'for f in <workdir>/skinny_*.json; do
  t=$(basename "$f" .json | sed "s/skinny_//")
  med=$(grep -o "\"baseline_median_tok_per_s\": [0-9.]*" "$f" | head -1 | awk "{print \$2}")
  det=$(grep -o "\"deterministic\": [a-z]*" "$f" | head -1 | awk "{print \$2}")
  printf "  %-14s median=%-9s deterministic=%s\n" "$t" "${med:-?}" "${det:-?}"
done'
echo
echo "  REFERENCE HASH 0128852903291e32 | skinny ON = control (the verified baseline config)"
echo "  DECISION RULES:"
echo "   * copyBuffers gone + tok/s DOWN   -> skinny GEMM still worth it; real target becomes the"
echo "     per-call .contiguous()/materialisation inside the skinny path."
echo "   * copyBuffers gone + tok/s UP >2% -> generic path wins in serve; investigate why it beats"
echo "     the skinny microbench."
echo "   * copyBuffers PERSIST            -> hypothesis DEAD; stop chasing copy attribution."

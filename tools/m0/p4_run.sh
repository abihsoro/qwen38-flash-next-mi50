#!/usr/bin/env bash
# p4_run.sh — Phase 4/M4 orchestrator (v2). Host-side; every container action runs a FILE.
#
# Discipline (the mistakes of this session, not repeated):
#   * cleanup is a FILE using BOTH matchers: `-f vllm.entrypoint[s]` for the parent (cmdline)
#     AND bare `VLLM` for the engine/workers (comm only — TRAPS #1: they prctl-rename, so
#     `-f 'VLLM::'` matches nothing and silently "succeeds")
#   * verify IDLE before and after (VRAM + /dev/kfd users)
#   * exactly ONE api_server; assert it
#   * v2 FIX: /stop_profile must be given time to FLUSH. v1 used -m 60 and the 180 MB of gzip
#     was still being written when the serve was killed -> all four rank traces truncated
#     ("Compressed file ended before the end-of-stream marker"). Now: long timeout, then wait
#     for the byte size to stop changing, then a gzip integrity check per file.
#   * throughput is measured with NO profiler attached; the trace is attribution-only
set -uo pipefail
CT=<ct>

say() { echo; echo "=== $* ==="; }

say "0) rig must be IDLE before we start"
pct exec $CT -- bash <workdir>/p4_stop.sh
V=$(pct exec $CT -- bash -c 'export PATH=/opt/rocm/bin:$PATH; rocm-smi --showmeminfo vram 2>/dev/null | grep -A1 "GPU\[0\]" | grep Used | awk "{print \$NF}"')
echo "  GPU0 VRAM used = ${V} bytes (idle ~10878976)"

say "1) launch ONE serve with the profiler enabled"
pct exec $CT -- bash -c ': > <workdir>/p4_serve.log; setsid nohup bash <workdir>/p4_serve.sh >> <workdir>/p4_serve.log 2>&1 < /dev/null & echo launched'
sleep 10
N=$(pct exec $CT -- bash -c 'pgrep -cf "vllm.entrypoint[s]" || true')
echo "  api_server processes = ${N:-0} (must be 1)"
[ "${N:-0}" = "1" ] || { echo "  ABORT: expected exactly 1 api_server"; exit 1; }

say "2) wait for startup"
for i in $(seq 1 80); do
  R=$(pct exec $CT -- bash -c 'grep -cE "Application startup complete" <workdir>/p4_serve.log 2>/dev/null || true')
  E=$(pct exec $CT -- bash -c 'grep -cE "Engine core initialization failed|FileNotFoundError" <workdir>/p4_serve.log 2>/dev/null || true')
  [ "${R:-0}" != "0" ] && { echo "  ready after ~$((i*15))s"; break; }
  [ "${E:-0}" != "0" ] && { echo "  FAILED to start"; pct exec $CT -- bash -c 'tail -25 <workdir>/p4_serve.log' | cut -c1-160; exit 1; }
  sleep 15
done
pct exec $CT -- bash -c 'grep -oE "Model loading took [0-9.]+ GiB memory and [0-9.]+ seconds" <workdir>/p4_serve.log | head -1'
pct exec $CT -- bash -c 'echo "  AR lines: $(grep -c "one-shot all-reduce active" <workdir>/p4_serve.log)  profiler refs: $(grep -c "torch_profiler_dir" <workdir>/p4_serve.log)"'

say "3) WALL TIMING with NO profiler attached (attribution must not use the trace)"
pct exec $CT -- bash -c "cd <workdir> && TP4_GRAPHS_MODE='phase4 profile run' <home>/gfx906-venv/bin/python <workdir>/tp4_gate.py 64 3" 2>&1 | tail -14

say "4) profiler ON -> one 128-token request -> profiler OFF (with flush wait)"
pct exec $CT -- bash -c 'curl -s -m 30 -X POST http://127.0.0.1:8002/start_profile -o /dev/null -w "  start_profile http=%{http_code}\n"'
pct exec $CT -- bash -c 'cd <workdir> && <home>/gfx906-venv/bin/python - <<PY
import json,time,urllib.request
body={"model":"qwen38-flash-next","prompt":"Explain the difference between a PCIe switch and a PCIe bridge in exactly three sentences.","max_tokens":128,"temperature":0.0,"top_p":1.0,"ignore_eos":True,"seed":0}
req=urllib.request.Request("http://127.0.0.1:8002/v1/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
t0=time.perf_counter()
d=json.loads(urllib.request.urlopen(req,timeout=900).read().decode())
wall=time.perf_counter()-t0
n=d.get("usage",{}).get("completion_tokens")
open("<workdir>/p4_request.json","w").write(json.dumps({"tokens":n,"wall_s":wall}))
print(f"  profiled request: {n} tokens in {wall:.3f}s")
PY'
echo "  issuing stop_profile (long timeout: the rank traces are ~180 MB of gzip)"
pct exec $CT -- bash -c 'curl -s -m 900 -X POST http://127.0.0.1:8002/stop_profile -o /dev/null -w "  stop_profile http=%{http_code}\n"'

say "5) wait for the traces to stop growing, then check integrity"
PREV=""
for i in $(seq 1 90); do
  SZ=$(pct exec $CT -- bash -c 'cat <workdir>/tprof/*.json.gz 2>/dev/null | wc -c')
  echo "  [$((i*10))s] total trace bytes = ${SZ:-0}"
  if [ -n "$PREV" ] && [ "$PREV" = "$SZ" ] && [ "${SZ:-0}" != "0" ]; then echo "  size stable"; break; fi
  PREV="$SZ"
  sleep 10
done
pct exec $CT -- bash -c 'for f in <workdir>/tprof/*.json.gz; do if gzip -t "$f" 2>/dev/null; then echo "  OK      $(basename $f)"; else echo "  CORRUPT $(basename $f)"; fi; done'

say "6) stop the serve and confirm the rig is IDLE again"
pct exec $CT -- bash <workdir>/p4_stop.sh
pct exec $CT -- bash -c 'export PATH=/opt/rocm/bin:$PATH; rocm-smi --showmeminfo vram 2>/dev/null | grep -E "GPU\[|Used" | head -8'
echo "  port 8002 listeners: $(pct exec $CT -- bash -c 'ss -lntp 2>/dev/null | grep -c ":8002" || true')"

say "7) ANALYSE (attribution only)"
REQ=$(pct exec $CT -- bash -c 'cat <workdir>/p4_request.json 2>/dev/null || echo "{}"')
TOK=$(echo "$REQ" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("tokens",0))' 2>/dev/null || echo 0)
WALL=$(echo "$REQ" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("wall_s",0))' 2>/dev/null || echo 0)
echo "  request: tokens=$TOK wall=$WALL"
pct exec $CT -- bash -c "<home>/gfx906-venv/bin/python <workdir>/p4_analyze.py <workdir>/tprof --tokens $TOK --wall-s $WALL > <workdir>/p4_ledger.txt 2>&1"
pct exec $CT -- cat <workdir>/p4_ledger.txt

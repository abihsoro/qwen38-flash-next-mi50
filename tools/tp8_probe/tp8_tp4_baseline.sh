#!/bin/bash
# tp8_tp4_baseline.sh — fresh TP4 reference under teardown-day hardware state, with rocm-smi
# sampling throughout. Runs the operator's exact commands; only the quoting is restructured so the
# monitor can be killed reliably even if the benchmark fails.
probe=<tuning>/tp8_probe
mkdir -p "$probe"

echo "=== bench script present? ==="
ls -la <tuning>/run_bench.py 2>/dev/null || {
  echo "  NOT at the top level; searching:"
  find <tuning> -maxdepth 3 -name "run_bench.py" 2>/dev/null
}
echo
echo "=== serve health before ==="
curl -s -m 8 -o /dev/null -w "  health http=%{http_code}\n" http://127.0.0.1:8002/health

# monitor: 1 Hz sampling for the whole run
(while true; do
  date -u +"%Y-%m-%dT%H:%M:%SZ"
  rocm-smi --showuse --showmemuse --showpower --showclocks --showtemp
  sleep 1
done) > "$probe/rocm_smi_during_vllm.txt" 2>&1 &
monpid=$!
echo "  monitor pid=$monpid"

# make sure it dies no matter how the benchmark exits
trap 'kill "$monpid" 2>/dev/null' EXIT

echo
echo "=== benchmark ==="
<home>/gfx906-venv/bin/python3 -u <tuning>/run_bench.py \
  --tag tp8_probe_tp4_baseline \
  --sizes 1000 10000 30000 \
  --kinds prose \
  --reps 3 \
  --gen 128
rc=$?

kill "$monpid" 2>/dev/null
wait "$monpid" 2>/dev/null
echo
echo "=== benchmark rc=$rc ==="
echo "  monitor samples: $(grep -c '^20' "$probe/rocm_smi_during_vllm.txt" 2>/dev/null)"
echo "  monitor bytes:   $(wc -c < "$probe/rocm_smi_during_vllm.txt" 2>/dev/null)"

echo
echo "=== health after ==="
curl -s -m 8 -o /dev/null -w "  health http=%{http_code}\n" http://127.0.0.1:8002/health

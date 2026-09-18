#!/usr/bin/env bash
# moe_noray_run.sh — run the ray-free serial MoE tuner (Phase 5 target 1).
set -uo pipefail
CT=<ct>

echo "=== 0) clear every leftover from the Ray attempts ==="
pkill -f 'moe_ab\.sh' 2>/dev/null || true
pct exec $CT -- bash <workdir>/sweep_stop.sh
pct exec $CT -- bash -c 'pkill -9 -f "benchmar[k]_moe" 2>/dev/null
pkill -9 -f "rayle[t]" 2>/dev/null
pkill -9 -f "gcs_serve[r]" 2>/dev/null
sleep 3
echo "  leftovers: $(pgrep -cf "rayle[t]|benchmar[k]_moe" 2>/dev/null || echo 0)"'

echo
echo "=== 1) push the ray-free tuner ==="
pct push $CT <workdir>/moe_tune_noray.py <workdir>/moe_tune_noray.py

echo
echo "=== 2) run it serially on ONE GPU (all 4 GPUs would only parallelise the same shapes) ==="
pct exec $CT -- bash -c '
export HOME=<workdir>
export PYTHONPATH=<workdir>/vllm-w:<workdir>
# LD_LIBRARY_PATH needs ROCm sysdeps dir: libamd_smi.so has unmet deps without it
export LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib/rocm_sysdeps/lib
unset ROCR_VISIBLE_DEVICES
export HIP_VISIBLE_DEVICES=0
export VLLM_MOE_TUNE_CACHE_CLEAR_INTERVAL=25
cd <workdir>
timeout 5400 env -u ROCR_VISIBLE_DEVICES <home>/gfx906-venv/bin/python \
  <workdir>/moe_tune_noray.py --save-dir <workdir>/moe-tuned-gfx906 \
  --batch-size 1 2 4 8 16 --device 0 2>&1 | grep -viE "amd_smi|libamd" | tail -45
'

echo
echo "=== 3) files produced ==="
pct exec $CT -- bash -c 'ls -la <workdir>/moe-tuned-gfx906/ 2>/dev/null'

#!/usr/bin/env bash
# run_waves_sweep.sh — microbench every WAVES variant per shape vs the production heuristic.
set -uo pipefail
CT=<ct>
pct push $CT <workdir>/gemv_waves_sweep.cu <workdir>/gemv_waves_sweep.cu
echo "=== build ==="
pct exec $CT -- bash -c '
export PATH=/opt/rocm/bin:$PATH
cd <workdir>
hipcc -O2 -std=c++17 -Wl,-rpath,/opt/rocm/lib gemv_waves_sweep.cu -o gemv_waves_sweep 2>&1 | grep -E "error|Error" | head -8
[ -x ./gemv_waves_sweep ] && echo "build OK" || echo "BUILD FAILED"
'
echo
echo "=== run ==="
pct exec $CT -- bash -c '
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-7.14/lib
cd <workdir>
timeout 1800 ./gemv_waves_sweep 2>&1 | grep -viE "amd_smi|libamd"
'

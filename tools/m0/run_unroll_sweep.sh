#!/usr/bin/env bash
# run_unroll_sweep.sh — microbench k-unroll depth U in {1,2,4,8} per shape (M=1 decode).
set -uo pipefail
CT=<ct>
pct push $CT <workdir>/gemv_unroll_sweep.cu <workdir>/gemv_unroll_sweep.cu
echo "=== build ==="
pct exec $CT -- bash -c '
export PATH=/opt/rocm/bin:$PATH
cd <workdir>
hipcc -O2 -std=c++17 -Wl,-rpath,/opt/rocm/lib gemv_unroll_sweep.cu -o gemv_unroll_sweep 2>&1 | grep -iE "error" | head -10
[ -x ./gemv_unroll_sweep ] && echo "build OK" || echo "BUILD FAILED"
'
echo
echo "=== run (U = independent k-loads per thread per iteration) ==="
pct exec $CT -- bash -c '
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-7.14/lib
cd <workdir>
timeout 1800 ./gemv_unroll_sweep 2>&1 | grep -viE "amd_smi|libamd"
'

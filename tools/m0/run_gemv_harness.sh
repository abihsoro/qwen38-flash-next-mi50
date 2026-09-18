#!/usr/bin/env bash
# run_gemv_harness.sh — build + run the existing fp16 GEMV harness at the TP4 per-rank shapes.
#
# This is the Phase 5/O9 starting point: it reports achieved GB/s per shape at M=1..8, against the
# D044 HBM ceiling (~810 GB/s), and checks each shape against a CPU fp32 reference. Shapes below
# the ceiling are where geometry/tile work can pay; shapes already at it cannot be improved.
set -uo pipefail
CT=<ct>
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-7.14/lib:${LD_LIBRARY_PATH:-}

pct push $CT <workdir>/gemv_f16_harness.cu <workdir>/gemv_f16_harness.cu

echo "=== build ==="
pct exec $CT -- bash -c '
export PATH=/opt/rocm/bin:$PATH
cd <workdir>
hipcc -O2 -std=c++17 -Wl,-rpath,/opt/rocm/lib gemv_f16_harness.cu -o gemv_f16_harness 2>&1 | tail -12
[ -x ./gemv_f16_harness ] && echo "build OK" || echo "BUILD FAILED"
'

echo
echo "=== run (M = 1, 4, 8; reports GB/s vs the ~810 GB/s HBM ceiling) ==="
pct exec $CT -- bash -c '
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-7.14/lib
cd <workdir>
timeout 1800 ./gemv_f16_harness 2>&1 | grep -viE "amd_smi|libamd"
echo "rc=$?"
'

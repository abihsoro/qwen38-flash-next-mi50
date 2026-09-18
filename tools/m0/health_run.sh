#!/usr/bin/env bash
# health_run.sh — container-side: build + run the minimal HIP health probe.
set -uo pipefail
cd <workdir>/m0v
LD=""
for d in /opt/rocm /opt/rocm-* /opt/rocm/core-*; do
  [ -x "$d/bin/hipcc" ] && export PATH="$d/bin:$PATH"
  [ -d "$d/lib" ] && LD="$d/lib${LD:+:$LD}"
done
export LD_LIBRARY_PATH="$LD"

hipcc -O2 -std=c++17 health.cpp -o health 2>&1 | tail -5
echo "--- health probe ---"
./health
echo "health rc=$?"
echo "--- vram after health ---"
rocm-smi --showmeminfo vram 2>&1 | grep -E "GPU\[|Used" | head -12
echo "--- kernel GPU reset count ---"
dmesg 2>/dev/null | grep -ciE "GPU reset" || echo 0

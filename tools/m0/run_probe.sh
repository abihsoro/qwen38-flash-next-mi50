#!/usr/bin/env bash
# run_probe.sh — container-side (LXC <ct>) build + run of the M0 P2P matrix probe.
# stdout MUST stay pure JSON (the host parses it); all diagnostics go to stderr.
#
# ROCm in this container lives under /opt/rocm/core-7.14 (TheRock layout), so
# neither PATH nor LD_LIBRARY_PATH is set in a bare `pct exec` shell. Discover
# both rather than hard-coding, and link with an rpath so the binary is runnable
# from anywhere.
set -uo pipefail

cd <workdir>/m0v || { echo '{"error":"<workdir>/m0v missing"}'; exit 9; }

ROCM_LIBS=""
add_lib() {
  [ -d "$1" ] || return 0
  case ":$ROCM_LIBS:" in *":$1:"*) ;; *) ROCM_LIBS="$1${ROCM_LIBS:+:$ROCM_LIBS}" ;; esac
}

for d in /opt/rocm /opt/rocm-* /opt/rocm/core-* /usr/lib/rocm* /usr/local/rocm*; do
  [ -x "$d/bin/hipcc" ] && export PATH="$d/bin:$PATH"
  add_lib "$d/lib"
  add_lib "$d/lib64"
  add_lib "$d/lib/llvm/lib"
  add_lib "$d/lib/x86_64-linux-gnu"
done
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"

HIPDIR="$(find /opt /usr/lib -name 'libamdhip64.so*' 2>/dev/null | head -1)"
[ -n "$HIPDIR" ] && add_lib "$(dirname "$HIPDIR")"
export LD_LIBRARY_PATH="$ROCM_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

{
  echo "=== container pre-run state ==="
  echo "hipcc: $(command -v hipcc || echo MISSING)"
  echo "hip runtime found: ${HIPDIR:-NONE}"
  echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
  echo "leaked VLLM:: procs: $(pgrep -af 'VLLM::' 2>/dev/null | wc -l)"
  echo "--- vram before ---"
  rocm-smi --showmeminfo vram 2>&1 | grep -iE "GPU\[|used" | head -12
} >&2

if ! command -v hipcc >/dev/null 2>&1; then
  echo '{"error":"hipcc not found in container"}'
  exit 3
fi

echo "=== building probe ===" >&2
if [ -n "${HIPDIR:-}" ]; then
  hipcc -O2 -std=c++17 m0_p2p_matrix.cpp -o m0_p2p_matrix \
    -Wl,-rpath,"$(dirname "$HIPDIR")" 2>&1 | tail -30 >&2
else
  hipcc -O2 -std=c++17 m0_p2p_matrix.cpp -o m0_p2p_matrix 2>&1 | tail -30 >&2
fi
if [ ! -x ./m0_p2p_matrix ]; then
  echo '{"error":"build failed"}'
  exit 4
fi

echo "=== running probe ===" >&2
./m0_p2p_matrix
rc=$?
if [ $rc -ne 0 ]; then
  echo "--- ldd (diagnostic) ---" >&2
  ldd ./m0_p2p_matrix 2>&1 | grep -i "not found" >&2 || true
fi
echo "probe rc=$rc" >&2
exit $rc

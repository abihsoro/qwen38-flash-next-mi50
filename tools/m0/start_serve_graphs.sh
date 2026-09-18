#!/usr/bin/env bash
# start_serve_graphs.sh — container-side launcher for the graphs run (a FILE, no pkill self-match).
set -uo pipefail
L=<workdir>/serve_tp4_graphs.log
pkill -9 -f 'VLLM::' 2>/dev/null || true
pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
sleep 4
: > "$L"
setsid nohup bash <workdir>/serve_tp4_48l_graphs.sh >> "$L" 2>&1 < /dev/null &
echo "launched pid $!"
sleep 6
pgrep -af 'serve_tp4_48l_graphs|vllm.entrypoints' | head -2 || echo "(nothing running!)"

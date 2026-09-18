#!/bin/bash
# p4_stop.sh — container-side SINGLETON cleanup. Two traps fixed here:
#   TRAPS #14: never inline a pkill pattern in `bash -c` (it self-matches) -> this is a FILE.
#   TRAPS #1 : vLLM's engine/workers rename themselves with prctl, so the name shows in the
#              process COMM but NOT in /proc/PID/cmdline. Therefore `pkill -f 'VLLM::'` matches
#              NOTHING and silently reports success. We must match the NAME too (no -f).
pkill -9 -f 'vllm.entrypoint[s]' 2>/dev/null     # parent api_server: cmdline match works
sleep 2
pkill -9 'VLLM' 2>/dev/null                      # engine+workers: COMM match (no -f!)
pkill -9 -f 'm2_ar[<bus>]' 2>/dev/null
pkill -9 -f 'm07_orde[r]' 2>/dev/null
sleep 6
A=$(pgrep -cf 'vllm.entrypoint[s]' 2>/dev/null) || A=0
B=$(pgrep -c 'VLLM' 2>/dev/null) || B=0
echo "p4_stop: api_server left=${A} engine_comm left=${B}"
exit 0

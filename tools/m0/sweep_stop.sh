#!/bin/bash
# sweep_stop.sh — container-side singleton cleanup for the sweep.
#
# TRAPS #14 (self-match): every pattern is bracket-escaped AND this runs as a FILE, so no inline
#   `bash -c` can match its own command line.
# TRAPS #1  (prctl rename): the engine/workers rename themselves, so 'VLLM::' appears in the
#   process COMM but NOT in /proc/PID/cmdline. `pkill -f 'VLLM[M]::'` therefore matches NOTHING
#   and silently "succeeds" while five processes keep holding 31 GB each. The comm match below
#   (no -f) is the one that actually works; the -f form is kept as belt-and-braces.
pkill -9 -f 'vllm.entrypoint[s]' 2>/dev/null    # parent api_server: cmdline match DOES work
sleep 2
pkill -9 -f 'VLLM[M]::'          2>/dev/null    # bracket-safe; usually matches nothing (see above)
pkill -9 'VLLM'                  2>/dev/null    # THE ONE THAT MATTERS: comm match, no -f
sleep 6
A=$(pgrep -cf 'vllm.entrypoint[s]' 2>/dev/null) || A=0
B=$(pgrep -c 'VLLM' 2>/dev/null) || B=0
echo "cleanup: api_server left=$A  engine_comm left=$B"
[ "$A" = "0" ] && [ "$B" = "0" ] || echo "  WARNING: processes survived cleanup"
exit 0

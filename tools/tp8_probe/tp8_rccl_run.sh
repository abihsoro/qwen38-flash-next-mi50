#!/bin/bash
# tp8_rccl_run.sh — run the collective sweep (binaries already built by tp8_rccl.sh).
#
# Fixes the two environment issues hit during the build/run:
#   * hipify-perl is only at /opt/rocm/core-7.14/bin (build-time; already done).
#   * the freshly built binaries carry no rpath, so libhsa-runtime64.so.1 is not found unless
#     LD_LIBRARY_PATH includes /opt/rocm/lib. (This is the legitimate case for that variable -
#     see TRAPS #11/#17: the correction there was about torch hiding GPUs, not about a locally
#     built binary that simply lacks an rpath.)
#
# rccl-tests interprets -b/-e as BYTES (K/M/G suffixes), so -e 256M is a 256 MiB message, not
# 256M elements. That matters here because the live TP4 serve leaves only ~2.7 GB free per card.
export PATH=/opt/rocm/bin:/opt/rocm/core-7.14/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-7.14/lib:${LD_LIBRARY_PATH:-}
PROBE=<tuning>/tp8_probe
cd "$PROBE/rccl-tests" || exit 1

echo "=== sanity: can the binary load now? ==="
./build/all_reduce_perf --help 2>&1 | head -3
echo

run() {
  local out="$1"; shift
  echo "=== $out ==="
  local t0=$SECONDS
  "$@" > "../$out" 2>&1
  local rc=$?
  echo "  rc=$rc elapsed=$((SECONDS-t0))s"
  if [ $rc -ne 0 ]; then tail -10 "../$out" | sed 's/^/    /'; fi
  echo
}

run rccl_all_reduce_small.txt  env HIP_VISIBLE_DEVICES=0,1,2,3 ./build/all_reduce_perf  -b 1K  -e 1M   -f 2 -g 4 -w 100 -n 1000
run rccl_all_reduce_large.txt  env HIP_VISIBLE_DEVICES=0,1,2,3 ./build/all_reduce_perf  -b 1M  -e 256M -f 2 -g 4 -w 20  -n 200
run rccl_all_gather_small.txt  env HIP_VISIBLE_DEVICES=0,1,2,3 ./build/all_gather_perf  -b 1K  -e 1M   -f 2 -g 4 -w 100 -n 1000
run rccl_all_gather_large.txt  env HIP_VISIBLE_DEVICES=0,1,2,3 ./build/all_gather_perf  -b 1M  -e 256M -f 2 -g 4 -w 20  -n 200

echo "=== results ==="
ls -la "$PROBE"/rccl_*.txt | sed 's/^/  /'
echo
echo "=== all_reduce (small) ==="
grep -E "^ +[<bus-range>]+ +[<bus-range>]+ +float +(none|sum)" "$PROBE/rccl_all_reduce_small.txt" | head -20
echo
echo "=== all_reduce (large) ==="
grep -E "^ +[<bus-range>]+ +[<bus-range>]+ +float +(none|sum)" "$PROBE/rccl_all_reduce_large.txt" | head -20

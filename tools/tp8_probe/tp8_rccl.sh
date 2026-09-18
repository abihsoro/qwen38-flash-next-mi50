#!/bin/bash
# tp8_rccl.sh — build rccl-tests and run the TP8-predictive collective sweep.
#
# Two container quirks handled here:
#   * git is NOT installed, so the source comes from the GitHub tarball.
#   * hipify-perl lives at /opt/rocm/core-7.14/bin/hipify-perl and is NOT on PATH; only
#     hipify-clang is. rccl-tests hipifies at build time, so that directory must be on PATH.
#
# The four runs are exactly the operator's requested commands. Note the serve is LIVE but IDLE
# (~17 W/card) so the fabric is free; the binding constraint is VRAM, since the running TP4 serve
# holds ~31.5 GB of 34.3 GB per card.
export PATH=/opt/rocm/bin:/opt/rocm/core-7.14/bin:$PATH
PROBE=<tuning>/tp8_probe
cd "$PROBE" || exit 1

echo "=== 0) source ==="
if [ ! -d rccl-tests ]; then
  curl -sL https://github.com/ROCm/rccl-tests/archive/refs/heads/master.tar.gz -o rccl-tests.tar.gz \
    && tar xzf rccl-tests.tar.gz && mv rccl-tests-master rccl-tests
fi
echo "  hipify-perl: $(command -v hipify-perl)"

echo
echo "=== 1) build ==="
cd "$PROBE/rccl-tests" || exit 1
make clean >/dev/null 2>&1
make -j"$(nproc)" 2>&1 | tail -12
echo "  artifacts:"
ls build/ 2>/dev/null | grep -E "all_reduce_perf|all_gather_perf" | sed 's/^/    /'
for b in build/all_reduce_perf build/all_gather_perf; do
  [ -x "$b" ] && echo "    OK $b" || echo "    MISSING $b"
done

cd "$PROBE/rccl-tests" || exit 1
echo
echo "=== 2) free VRAM before the runs ==="
rocm-smi --showmeminfo vram 2>/dev/null | grep -E "GPU\[[<bus-range>]\]" | head -4 | sed 's/^/  /'

run() {
  local out="$1"; shift
  echo
  echo "=== $out ==="
  local t0=$SECONDS
  "$@" > "../$out" 2>&1
  local rc=$?
  echo "  rc=$rc  elapsed=$((SECONDS-t0))s  size=$(wc -c < "../$out") bytes"
  if [ $rc -ne 0 ]; then
    echo "  --- last 12 lines of the failure ---"
    tail -12 "../$out" | sed 's/^/    /'
  fi
}

# NOTE: -b/-e in rccl-tests are ELEMENT counts, not bytes (the tool scales by dtype size).
run rccl_all_reduce_small.txt  env HIP_VISIBLE_DEVICES=0,1,2,3 ./build/all_reduce_perf  -b 1K  -e 1M   -f 2 -g 4 -w 100 -n 1000
run rccl_all_reduce_large.txt  env HIP_VISIBLE_DEVICES=0,1,2,3 ./build/all_reduce_perf  -b 1M  -e 256M -f 2 -g 4 -w 20  -n 200
run rccl_all_gather_small.txt  env HIP_VISIBLE_DEVICES=0,1,2,3 ./build/all_gather_perf  -b 1K  -e 1M   -f 2 -g 4 -w 100 -n 1000
run rccl_all_gather_large.txt  env HIP_VISIBLE_DEVICES=0,1,2,3 ./build/all_gather_perf  -b 1M  -e 256M -f 2 -g 4 -w 20  -n 200

echo
echo "=== 3) what was produced ==="
ls -la "$PROBE"/rccl_*.txt 2>/dev/null | sed 's/^/  /'

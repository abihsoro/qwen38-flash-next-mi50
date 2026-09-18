#!/usr/bin/env bash
# run_negctl.sh — ACS NEGATIVE CONTROL on the four-card fabric (M0-DECISION-TREE §5).
#
# The positive measurements say all 12 pairs run at ~14.4 GB/s. This control asks
# the falsifiable question: IS THE ACS STATE WHAT GOVERNS PEER BANDWIDTH, or would
# the fabric behave identically with ACS asserted?
#
# Design (controlled, single-variable, reversible):
#   target = the GPU3 (<pci>) path hops that are UNIQUE to GPU3 and ACS-capable:
#            <pci> and <pci>
#            (<pci> is deliberately EXCLUDED: it is shared with GPU2 <pci>)
#   assert  = the switch's own asserted default, read empirically from <pci>
#             (SrcValid+ ReqRedir+ CmpltRedir+ UpstreamFwd+ = 0x001d)
#   affected pairs = every pair involving GPU3 (0<->3, 1<->3, 2<->3)
#   control pairs  = pairs among GPU0/1/2 (0<->1, 0<->2, 1<->2) — same probe run
#
# Expected if ACS governs peer routing: GPU3 pairs collapse toward the host-link
# rate while the control pairs stay at ~14.4 GB/s.
# Expected if ACS does NOT govern it: all 12 pairs are unchanged.
#
# A trap always restores the clear, and the restore is verified.
set -uo pipefail

OUT=<workdir>/m0out
CT=300
UNIQUE_HOPS="<pci> <pci>"
ASSERT=001d

acsline() { lspci -vvv -s "$1" 2>/dev/null | grep -m1 'ACSCtl:' | sed 's/^[[:space:]]*//'; }

restore() {
  echo "--- RESTORE: clearing ACS on $UNIQUE_HOPS ---"
  for p in $UNIQUE_HOPS; do setpci -s "$p" ECAP_ACS+6.w=0000 2>/dev/null; done
  for p in $UNIQUE_HOPS; do echo "  post-restore $p: $(acsline "$p")"; done
}
trap restore EXIT

run_probe() {
  local tag="$1"
  pct exec "$CT" -- bash <workdir>/m0v/run_probe.sh > "$OUT/C_${tag}.json" 2> "$OUT/C_${tag}.err"
  echo "  probe[$tag] exit=$?"
}

echo "=== ACS state before ==="
for p in $UNIQUE_HOPS; do echo "  $p: $(acsline "$p")"; done

echo
echo "=== STEP 0: negotiated link state (host link vs endpoint links) ==="
for b in <pci> <pci> <pci> <pci> <pci> <pci>; do
  printf '%-16s cur=%s x%s   max=%s x%s\n' "$b" \
    "$(cat /sys/bus/pci/devices/$b/current_link_speed 2>/dev/null || echo -)" \
    "$(cat /sys/bus/pci/devices/$b/current_link_width 2>/dev/null || echo -)" \
    "$(cat /sys/bus/pci/devices/$b/max_link_speed 2>/dev/null || echo -)" \
    "$(cat /sys/bus/pci/devices/$b/max_link_width 2>/dev/null || echo -)"
done
for b in <pci> <pci> <pci> <pci>; do
  echo "--- $b ---"
  lspci -vvv -s "$b" 2>/dev/null | grep -E "LnkCap:|LnkSta:|LnkCap2:|LnkSta2:" | sed 's/^[[:space:]]*/  /'
done

echo
echo "=== STEP 1: baseline probe (ACS cleared) ==="
run_probe before

echo
echo "=== STEP 2: ASSERT ACS (0x$ASSERT) on the GPU3-unique hops ==="
for p in $UNIQUE_HOPS; do setpci -s "$p" ECAP_ACS+6.w=$ASSERT; done
for p in $UNIQUE_HOPS; do echo "  asserted $p: $(acsline "$p")"; done

echo
echo "=== STEP 3: probe with ACS ASSERTED ==="
run_probe asserted

echo
echo "=== STEP 4: restore ACS clear ==="
restore
trap - EXIT

echo
echo "=== STEP 5: probe after restore (must match baseline) ==="
run_probe after

echo
echo "=== OUTPUTS ==="
ls -la "$OUT"/C_*.json

echo
echo "=== COMPARISON ==="
python3 - <<'PY'
import json
def load(tag):
    try:
        return json.load(open(f"<workdir>/m0out/C_{tag}.json"))
    except Exception:
        return None
def peer(d):
    if not d or "pairs" not in d:
        return {}
    return {(p["src"], p["dst"]): p["bw_gbps_median"] for p in d["pairs"]}
b, a, af = peer(load("before")), peer(load("asserted")), peer(load("after"))
d0 = load("before")
host = {}
if d0 and "host_transfer" in d0:
    host = {x["dev"]: max(x["h2d_gbps"], x["d2h_gbps"]) for x in d0["host_transfer"]}
print("reference host-link ceiling per device:", host)
print()
print("pair        class   before  asserted    after    delta%")
for src in range(4):
    for dst in range(4):
        if src == dst:
            continue
        k = (src, dst)
        bb, aa, ff = b.get(k), a.get(k), af.get(k)
        if bb is None or aa is None:
            continue
        delta = (aa - bb) / bb * 100.0 if bb else float("nan")
        cls = "GPU3" if 3 in k else "ctrl"
        print(f"  {src}->{dst}   {cls}   {bb:6.2f}  {aa:8.2f}  {ff if ff is not None else float('nan'):6.2f}   {delta:+7.2f}%")
gpu3 = [ (aa - b[k]) / b[k] * 100 for k, aa in a.items() if 3 in k and k in b and b[k] ]
ctrl = [ (aa - b[k]) / b[k] * 100 for k, aa in a.items() if 3 not in k and k in b and b[k] ]
if gpu3 and ctrl:
    print()
    print(f"GPU3 pairs  mean change when ACS asserted: {sum(gpu3)/len(gpu3):+.2f}%")
    print(f"ctrl pairs  mean change when ACS asserted: {sum(ctrl)/len(ctrl):+.2f}%")
    print("INTERPRETATION: a large negative GPU3 change vs ~0 control change => ACS state GOVERNS peer routing.")
    print("                both ~0 => ACS state does NOT govern it; the ~14.4 GB/s limit is elsewhere.")
PY
echo "=== NEGATIVE CONTROL COMPLETE ==="

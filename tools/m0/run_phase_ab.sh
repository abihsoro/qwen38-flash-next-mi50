#!/usr/bin/env bash
# run_phase_ab.sh — M0 4-card transport validation: Phase A (fabric/ACS, read-only
# + idempotency re-run) and Phase B (P2P matrix across all ordered GPU pairs).
#
# The ACS NEGATIVE CONTROL is deliberately NOT here: it mutates live silicon and
# runs separately, after these positive measurements are banked (M0-DECISION-TREE
# §5 discipline: bank the measurement, then perturb with a control).
set -uo pipefail

OUT=<workdir>/m0out
VDIR=<workdir>/m0v
CT=300
mkdir -p "$OUT"

echo "### PHASE A1: fabric/ACS report (pre) ###"
python3 "$VDIR/m0_fabric_report.py" > "$OUT/A1_fabric_pre.json" 2> "$OUT/A1.err"
rc=$?
echo "A1 exit=$rc bytes=$(wc -c < "$OUT/A1_fabric_pre.json" 2>/dev/null || echo 0)"

echo
echo "### PHASE A2: re-run clear-pex-acs.sh (idempotency + no-regression) ###"
/usr/local/sbin/clear-pex-acs.sh > "$OUT/A2_acs_rerun.log" 2>&1
rc=$?
echo "A2 (clear-pex-acs.sh) exit=$rc"
echo "--- A2 log ---"
cat "$OUT/A2_acs_rerun.log"

echo
echo "### PHASE A3: fabric/ACS report (post re-run) ###"
python3 "$VDIR/m0_fabric_report.py" > "$OUT/A3_fabric_post.json" 2> "$OUT/A3.err"
rc=$?
echo "A3 exit=$rc"

echo
echo "### PHASE B: 4-card P2P matrix inside LXC $CT ###"
pct exec "$CT" -- mkdir -p <workdir>/m0v
pct push "$CT" "$VDIR/m0_p2p_matrix.cpp" <workdir>/m0v/m0_p2p_matrix.cpp
rc=$?
if [ $rc -ne 0 ]; then echo "FATAL: push cpp failed rc=$rc"; exit 5; fi
pct push "$CT" "$VDIR/run_probe.sh" <workdir>/m0v/run_probe.sh
rc=$?
if [ $rc -ne 0 ]; then echo "FATAL: push run_probe failed rc=$rc"; exit 6; fi

pct exec "$CT" -- bash <workdir>/m0v/run_probe.sh > "$OUT/B_matrix.json" 2> "$OUT/B_matrix.err"
rc=$?
echo "B (probe) exit=$rc"
echo "--- B stderr (tail 40) ---"
tail -40 "$OUT/B_matrix.err"
echo "--- B stdout size: $(wc -c < "$OUT/B_matrix.json" 2>/dev/null || echo 0) bytes ---"
python3 - "$OUT/B_matrix.json" <<'PY' 2>&1 | tail -30
import json, sys
try:
    d = json.load(open(sys.argv[<bus>]))
except Exception as e:
    print("JSON parse FAILED:", e); sys.exit(0)
if "error" in d and "device_count" not in d:
    print("probe error:", d["error"]); sys.exit(0)
print("device_count:", d.get("device_count"))
print("repeats:", d.get("repeats"), "lat_repeats:", d.get("lat_repeats"))
print("devices:")
for dv in d.get("devices", []):
    print("  ", dv["index"], dv["name"], dv["gcn_arch"], dv["pci_bus_id"], f'{dv["mem_gb"]:.1f}GB')
print("pairs (src->dst  can  en  bw_GB/s  ok  lat_us):")
for p in d.get("pairs", []):
    print(f'   {p["src"]}->{p["dst"]}  can={p["can_access_peer"]} en={p["peer_access_enabled"]} '
          f'bw={p["bw_gbps_median"]:.2f} ok={p["bytes_ok"]} lat={p["lat_us_median"]:.2f} {p["err"]}')
print("self copy:")
for s in d.get("self_copy", []):
    print(f'   dev{s["dev"]}  bw={s["bw_gbps_median"]:.2f} ok={s["bytes_ok"]} {s["err"]}')
PY

echo
echo "### PHASE A4: fabric/ACS report (final) ###"
python3 "$VDIR/m0_fabric_report.py" > "$OUT/A4_fabric_final.json" 2>/dev/null
echo "A4 exit=$?"

echo
echo "### OUTPUTS ###"
ls -la "$OUT"
echo "### PHASE A/B COMPLETE ###"

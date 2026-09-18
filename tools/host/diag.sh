#!/usr/bin/env bash
# diag.sh — post-negative-control health check (read-only).
set -uo pipefail
echo "=== C_asserted.err (tail 45) ==="
tail -45 <workdir>/m0out/C_asserted.err 2>&1
echo
echo "=== C_after.err (tail 45) ==="
tail -45 <workdir>/m0out/C_after.err 2>&1
echo
echo "=== C_asserted.json tail (partial?) ==="
tail -c 600 <workdir>/m0out/C_asserted.json 2>&1
echo
echo "=== ACS state NOW (restore verification) ==="
for p in <pci> <pci> <pci> <pci> <pci>; do
  echo "  $p: $(lspci -vvv -s $p 2>/dev/null | grep -m1 'ACSCtl:' | sed 's/^[[:space:]]*//')"
done
echo
echo "=== link state NOW ==="
for b in <pci> <pci> <pci> <pci> <pci>; do
  printf '%-16s cur=%s x%s  max=%s x%s\n' "$b" \
    "$(cat /sys/bus/pci/devices/$b/current_link_speed 2>/dev/null || echo -)" \
    "$(cat /sys/bus/pci/devices/$b/current_link_width 2>/dev/null || echo -)" \
    "$(cat /sys/bus/pci/devices/$b/max_link_speed 2>/dev/null || echo -)" \
    "$(cat /sys/bus/pci/devices/$b/max_link_width 2>/dev/null || echo -)"
done
echo
echo "=== GPU health NOW (host) ==="
rocm-smi --showmeminfo vram 2>&1 | head -14
echo "--- kfd p2p links ---"
for n in /sys/class/kfd/kfd/topology/nodes/*/properties; do
  grep -H -E "p2p_links_count|io_links_count" "$n" 2>/dev/null
done
echo
echo "=== recent kernel complaints ==="
dmesg 2>/dev/null | grep -iE "amdgpu|kfd|amd_iommu|pcieport|AER|reset" | tail -30
echo
echo "=== container GPU health ==="
pct exec 300 -- bash -c 'rocm-smi --showmeminfo vram 2>&1 | head -12'

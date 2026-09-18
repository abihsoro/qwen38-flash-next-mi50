#!/usr/bin/env bash
set -euo pipefail

# Clear ACS redirect bits on GPU paths through the Broadcom/LSI PEX880xx switch.
#
# This intentionally avoids endpoint device IDs. The old version discovered
# ports by walking from V620 endpoints, so it stopped being reliable when the
# machine moved to MI50/Vega20 cards. This version walks from any AMD display /
# accelerator endpoint that sits behind a PEX880xx bridge and clears every
# ACS-capable PCI bridge on that endpoint's path.

declare -A seen=()

is_bdf() {
  [[ "$1" =~ ^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[<bus-range>]$ ]]
}

has_pex_ancestor() {
  local path=$1 bdf dev
  while [ "$path" != "/" ]; do
    bdf=$(basename "$path")
    if is_bdf "$bdf"; then
      dev="/sys/bus/pci/devices/$bdf"
      if [ "$(cat "$dev/vendor" 2>/dev/null || true)" = "0x1000" ] &&
         [ "$(cat "$dev/device" 2>/dev/null || true)" = "0xc010" ]; then
        return 0
      fi
    fi
    path=$(dirname "$path")
  done
  return 1
}

collect_bridge_ancestors() {
  local path=$1 bdf dev
  while [ "$path" != "/" ]; do
    bdf=$(basename "$path")
    if is_bdf "$bdf"; then
      dev="/sys/bus/pci/devices/$bdf"
      if [ "$(cat "$dev/class" 2>/dev/null || true)" = "0x060400" ] &&
         lspci -vvv -s "$bdf" 2>/dev/null | grep -q "ACSCap:"; then
        seen["$bdf"]=1
      fi
    fi
    path=$(dirname "$path")
  done
}

for gpu in /sys/bus/pci/devices/*; do
  [ "$(cat "$gpu/vendor" 2>/dev/null || true)" = "0x1002" ] || continue
  case "$(cat "$gpu/class" 2>/dev/null || true)" in
    0x03*) ;;
    *) continue ;;
  esac

  path=$(readlink -f "$gpu")
  if has_pex_ancestor "$path"; then
    echo "found AMD GPU endpoint behind PEX: $(basename "$gpu")"
    collect_bridge_ancestors "$path"
  fi
done

mapfile -t ports < <(printf "%s\n" "${!seen[@]}" | sort)
if [ "${#ports[@]}" -eq 0 ]; then
  echo "ACS clear FAILED: no ACS-capable GPU-path bridges found behind the PEX880xx"
  exit 1
fi

echo "candidate GPU-path ACS bridge ports: ${ports[*]}"
for p in "${ports[@]}"; do
  echo "clearing ACS on $p"
  setpci -s "$p" ECAP_ACS+6.w=0000
done

failed=0
for p in "${ports[@]}"; do
  ctl=$(lspci -vvv -s "$p" | grep ACSCtl: | head -1 || true)
  if [ -z "$ctl" ]; then
    echo "post-clear $p: missing ACSCtl"
    failed=1
    continue
  fi

  echo "post-clear $p: $ctl"
  echo "$ctl" | grep -q "ReqRedir-" || failed=1
  echo "$ctl" | grep -q "CmpltRedir-" || failed=1
  echo "$ctl" | grep -q "UpstreamFwd-" || failed=1
done

exit "$failed"

#!/usr/bin/env python3
"""m0_fabric_report.py — read-only fabric/ACS state report for M0 validation.

Runs on <source-host> (host). Emits one JSON object on stdout. Touches nothing:
every check is a read of sysfs / lspci / systemd state, so it is safe to run
before, during, and after any ACS manipulation.

What it establishes:
  * which AMD display-class endpoints exist, and which sit behind a PEX880xx
    switch (the predicate the generic clear-pex-acs.sh uses);
  * for every GPU path, each bridge hop with ACSCap/ACSCtl and link width/speed;
  * whether EVERY ACS-capable bridge on every GPU path is actually cleared
    (the property M0's bandwidth numbers depend on);
  * the state of PEX ports with no GPU behind them (regression control: the new
    script is GPU-path-scoped and must NOT clear these);
  * IOMMU groups, KFD nodes, amdgpu params, kernel cmdline.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

SYS = "/sys/bus/pci/devices"
PEX_VENDOR = "0x1000"
PEX_DEVICE = "0xc010"


def sh(cmd: list[str], timeout: int = 20) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.stdout
    except Exception:
        return ""


def rd(path: str) -> str:
    try:
        with open(path) as fh:
            return fh.read().strip()
    except Exception:
        return ""


def config_space(bdf: str) -> str:
    return sh(["lspci", "-vvv", "-s", bdf])


def parse_acs(cfg: str) -> dict:
    out = {"has_acscap": "ACSCap:" in cfg, "acsctl": "", "lnkcap": "", "lnksta": "", "lnkcap2": "", "lnksta2": ""}
    for line in cfg.splitlines():
        s = line.strip()
        if s.startswith("ACSCtl:"):
            out["acsctl"] = re.sub(r"\s+", " ", s)
        elif s.startswith("LnkCap:"):
            # keep only the speed/width clause, drop port numbers
            m = re.search(r"(Speed \S+, Width x\d+)", s)
            if m:
                out["lnkcap"] = m.group(1)
        elif s.startswith("LnkSta:"):
            m = re.search(r"(Speed \S+, Width x\d+)", s)
            if m:
                out["lnksta"] = m.group(1)
        elif s.startswith("LnkCap2:"):
            m = re.search(r"(Speed \S+, Width x\d+)", s)
            if m:
                out["lnkcap2"] = m.group(1)
        elif s.startswith("LnkSta2:"):
            m = re.search(r"(Speed \S+, Width x\d+)", s)
            if m:
                out["lnksta2"] = m.group(1)
    return out


def is_cleared(acsctl: str) -> bool:
    """True iff every ACS control bit reads '-' (no redirect/blocking asserted)."""
    if not acsctl:
        return False
    bits = re.findall(r"(\w+)([+-])", acsctl)
    if not bits:
        return False
    return all(sign == "-" for _, sign in bits)


def ancestors(bdf: str) -> list[str]:
    """Bridge chain from the endpoint up to the root complex (deepest first)."""
    chain = []
    dev = os.path.realpath(os.path.join(SYS, bdf))
    while dev and dev != "/":
        b = os.path.basename(dev)
        if re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[<bus-range>]", b):
            chain.append(b)
        nxt = os.path.dirname(dev)
        if nxt == dev:
            break
        dev = nxt
    return chain


def under_pex(bdf: str) -> bool:
    for b in ancestors(bdf)[1:]:  # skip self
        d = os.path.join(SYS, b)
        if rd(f"{d}/vendor") == PEX_VENDOR and rd(f"{d}/device") == PEX_DEVICE:
            return True
    return False


def main() -> int:
    # ── service state ──────────────────────────────────────────────────────
    script = "/usr/local/sbin/clear-pex-acs.sh"
    svc = {
        "enabled": sh(["systemctl", "is-enabled", "clear-pex-acs.service"]).strip(),
        "active": sh(["systemctl", "is-active", "clear-pex-acs.service"]).strip(),
        "script_sha256": sh(["sha256sum", script]).split()[<bus>] if os.path.exists(script) else "",
        "unit": sh(["systemctl", "cat", "clear-pex-acs.service"]),
    }

    # ── AMD display-class endpoints ────────────────────────────────────────
    endpoints = []
    for entry in sorted(os.listdir(SYS)):
        d = os.path.join(SYS, entry)
        if rd(f"{d}/vendor") != "0x1002":
            continue
        cls = rd(f"{d}/class")
        if not cls.startswith("0x03"):
            continue
        pex = under_pex(entry)
        path = []
        for b in ancestors(entry):
            cfg = config_space(b)
            info = parse_acs(cfg)
            info["bdf"] = b
            info["class"] = rd(os.path.join(SYS, b, "class"))
            info["vendor"] = rd(os.path.join(SYS, b, "vendor"))
            info["device"] = rd(os.path.join(SYS, b, "device"))
            info["is_pex"] = info["vendor"] == PEX_VENDOR and info["device"] == PEX_DEVICE
            path.append(info)
        acs_capable = [h for h in path if h["has_acscap"]]
        endpoints.append({
            "bdf": entry,
            "vendor": "<vendor-id>",
            "device": rd(f"{d}/device"),
            "class": cls,
            "under_pex": pex,
            "path": path,
            "acs_capable_hops": [h["bdf"] for h in acs_capable],
            "acs_capable_not_cleared": [h["bdf"] for h in acs_capable if not is_cleared(h["acsctl"])],
            "all_acs_capable_hops_cleared": all(is_cleared(h["acsctl"]) for h in acs_capable) if acs_capable else None,
        })

    gpus = [e for e in endpoints if e["under_pex"]]

    # ── regression control: PEX ports with no GPU behind them ──────────────
    # Any PEX bridge that is NOT on a GPU path must keep its ACS state untouched
    # by the generic script (the script is GPU-path scoped by design).
    gpu_path_bdfs = set()
    for e in gpus:
        for h in e["path"]:
            gpu_path_bdfs.add(h["bdf"])
    non_gpu_pex_ports = []
    for entry in sorted(os.listdir(SYS)):
        d = os.path.join(SYS, entry)
        if rd(f"{d}/vendor") != PEX_VENDOR or rd(f"{d}/device") != PEX_DEVICE:
            continue
        if rd(f"{d}/class") != "0x060400":
            continue
        # does any GPU sit under it?
        carries_gpu = any(entry in e["path"] and e["path"].index([h for h in e["path"] if h["bdf"] == entry][<bus>]) >= 0 for e in gpus)
        cfg = config_space(entry)
        info = parse_acs(cfg)
        non_gpu_pex_ports.append({
            "bdf": entry,
            "on_gpu_path": entry in gpu_path_bdfs,
            "has_acscap": info["has_acscap"],
            "acsctl": info["acsctl"],
            "cleared": is_cleared(info["acsctl"]),
        })

    # ── IOMMU groups ───────────────────────────────────────────────────────
    iommu = {}
    for e in endpoints:
        grp = os.path.realpath(os.path.join(SYS, e["bdf"], "iommu_group"))
        iommu[e["bdf"]] = os.path.basename(grp) if grp and grp != "/" else "none"

    # ── KFD topology ───────────────────────────────────────────────────────
    kfd = {"nodes": [], "available": os.path.isdir("/sys/class/kfd/kfd/topology/nodes")}
    knodes = "/sys/class/kfd/kfd/topology/nodes"
    if kfd["available"]:
        for node in sorted(os.listdir(knodes), key=lambda s: int(s) if s.isdigit() else 0):
            props = rd(os.path.join(knodes, node, "properties"))
            d = {"node": node}
            for line in props.splitlines():
                parts = line.split()
                if len(parts) >= 2 and ("p2p" in parts[<bus>].lower() or "link" in parts[<bus>].lower()):
                    d[parts[<bus>]] = parts[<bus>]
            kfd["nodes"].append(d)

    # ── amdgpu params / cmdline ────────────────────────────────────────────
    amdgpu = {}
    for p in ("pcie_p2p", "pcie_gen_cap", "pcie_lane_cap", "vm_fragment_size"):
        f = f"/sys/module/amdgpu/parameters/{p}"
        amdgpu[p] = rd(f) if os.path.exists(f) else "(absent)"

    report = {
        "service": svc,
        "endpoints": endpoints,
        "gpu_endpoints_behind_pex": [e["bdf"] for e in gpus],
        "gpu_count_behind_pex": len(gpus),
        "all_gpu_paths_fully_cleared": all(e["all_acs_capable_hops_cleared"] for e in gpus) if gpus else None,
        "any_gpu_path_acs_capable_hop_uncleared": sorted(
            {b for e in gpus for b in e["acs_capable_not_cleared"]}),
        "pex_ports": non_gpu_pex_ports,
        "iommu_groups": iommu,
        "kfd": kfd,
        "amdgpu_params": amdgpu,
        "cmdline": rd("/proc/cmdline"),
        "uptime_s": rd("/proc/uptime").split()[<bus>] if rd("/proc/uptime") else "",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

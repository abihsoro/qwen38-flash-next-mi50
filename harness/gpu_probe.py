"""gpu_probe.py — measure the GPU guest environment for task S0 acceptance.

Two modes (harness-only writes; I1: everything recorded is measured, never
guessed):

  --emit
      Run on the GPU guest (e.g. <vm>): print one JSON document with every
      measurement S0 needs (gpu_count, BAR sizes, gfx target via rocminfo,
      IOMMU groups, hugepages, kernel, ROCm version). No repo access needed.

  --write FILE
      Run where the repo lives: ingest a probe JSON produced by --emit,
      validate it, then write results rows + results/gpu_probe.jsonl through
      results.py (the sole writer, I3) and print the S0 acceptance verdicts.

S0 acceptance items checked here:
  - gfx_target == "gfx906:xnack-"          (work order §6.S0)
  - guest-visible BAR for the GPU >= 32 GB (work order §3.2 / STOP-B)
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# results (harness writer) is imported lazily inside ingest(): --emit must run
# on the GPU guest with no repo access. I3 still holds: rows are written by
# results.py (harness code), never hand-written.


def _run(args, timeout: float = 30):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None


def measure_pci_gpus() -> dict:
    """AMD/ATI display/3D devices with BAR sizes read from sysfs."""
    proc = _run(["lspci", "-nn"])
    gpus = []
    if proc is not None and proc.returncode == 0:
        for line in proc.stdout.splitlines():
            if not re.search(r"VGA|3D controller|Display controller|Accelerated", line, re.I):
                continue
            if "AMD/ATI" not in line and "1002:" not in line:
                continue
            bdf = line.split()[<bus>]
            gpus.append(measure_one_gpu(bdf))
    return {"gpu_count": len(gpus), "gpus": gpus}


def measure_one_gpu(bdf: str) -> dict:
    if bdf.count(":") == 1:
        bdf = f"0000:{bdf}"  # sysfs always uses the full domain prefix
    base = Path(f"/sys/bus/pci/devices/{bdf}")
    info = {"bdf": bdf}
    # vendor:device + class + driver
    def read(p: str) -> str:
        try:
            return (base / p).read_text().strip()
        except Exception:
            return None
    info["vendor_device"] = f"{read('vendor')}:{read('device')}" if read('vendor') else None
    info["class"] = read("class")
    try:
        info["driver"] = (base / "driver").resolve().name if (base / "driver").exists() else None
    except Exception:
        info["driver"] = None
    # BARs: authoritative sizes from resource file st_size (sysfs reports the BAR
    # size as the file size even when content is root-only); start addresses from
    # the file content when readable, else from lspci -vvv.
    region_addrs = _lspci_region_addrs(bdf)
    bars = []
    max_bar = 0
    for i in range(6):
        res = base / f"resource{i}"
        try:
            size = res.stat().st_size
        except Exception:
            continue
        if size <= 0:
            continue
        start = None
        try:
            fields = res.read_text().strip().split()
            if fields:
                start = int(fields[<bus>], 16)
        except Exception:
            start = region_addrs.get(i)
        bars.append({"index": i, "start_hex": hex(start) if start else None,
                     "size_bytes": size, "flags": None})
        if size > max_bar:
            max_bar = size
    info["bars"] = bars
    info["max_bar_bytes"] = max_bar
    info["bar_above_4g"] = any(b["start_hex"] and int(b["start_hex"], 16) >= <bar-addr>
                               for b in bars if b["size_bytes"] > 0)
    return info


def _lspci_region_addrs(bdf: str) -> dict:
    """Parse 'Region N: Memory at <addr> ... [size=..]' lines from lspci -vvv."""
    out = {}
    proc = _run(["lspci", "-vvv", "-s", bdf])
    if proc is None or proc.returncode != 0:
        return out
    for line in proc.stdout.splitlines():
        m = re.match(r"\s*Region (\d): (?:Memory at )?([0-9a-fA-F]+)\s", line)
        if m:
            out[int(m.group(1))] = int(m.group(2), 16)
    return out


def measure_gfx_target() -> dict:
    """gfx906 identification; xnack only as reported by ROCm tooling."""
    proc = _run(["rocminfo"])
    out = proc.stdout if (proc is not None and proc.returncode == 0) else ""
    m = re.search(r"Name:\s+(gfx\d+)", out)
    name = m.group(1) if m else None
    xnack = None
    if "xnack" in out:
        lines = [ln.strip() for ln in out.splitlines() if "xnack" in ln.lower()]
        xnack = lines[<bus>] if lines else None
    return {"rocminfo_name": name, "xnack_line": xnack, "rocminfo_rc": proc.returncode if proc else None}


def measure_rocm_version() -> str:
    proc = _run(["rocm-smi", "--showproductname"])
    if proc is not None and proc.returncode == 0:
        return f"rocm-smi ok"
    for p in ("/opt/rocm/.info/version", "/opt/rocm/share/doc/rocm-version/version"):
        try:
            v = Path(p).read_text().strip()
            if v:
                return v
        except Exception:
            pass
    return "not measured"


def measure_iommu_groups() -> list:
    groups = []
    try:
        for g in sorted(Path("/sys/kernel/iommu_groups").iterdir()):
            devices = sorted(d.name for d in g.iterdir())
            groups.append({"group": g.name, "devices": devices})
    except Exception:
        pass
    return groups


def measure_mem() -> dict:
    out = {"ram_gb": "not measured", "hugepages_total": None, "hugepagesize_kb": None}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                out["ram_gb"] = round(int(line.split()[<bus>]) / 1024 / 1024, 1)
            elif line.startswith("HugePages_Total:"):
                out["hugepages_total"] = int(line.split()[<bus>])
            elif line.startswith("Hugepagesize:"):
                out["hugepagesize_kb"] = int(line.split()[<bus>])
    except Exception:
        pass
    return out


def build_probe() -> dict:
    pci = measure_pci_gpus()
    mem = measure_mem()
    probe = {
        "probe_host": os.uname().nodename,
        "probe_time": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "guest_kernel": os.uname().release,
        "cpu_count": os.cpu_count(),
        "rocm_version": measure_rocm_version(),
        "gfx_target": measure_gfx_target(),
        "pve_version": "not measured (inside guest)",
        "dev_kfd": Path("/dev/kfd").exists(),
        "dev_dri": sorted(p.name for p in Path("/dev/dri").glob("*")) if Path("/dev/dri").is_dir() else [],
        "iommu_groups": measure_iommu_groups(),
        **pci,
        **mem,
    }
    return probe


def acceptance(probe: dict) -> dict:
    gfx = probe.get("gfx_target", {})
    gfx_ok = gfx.get("rocminfo_name") == "gfx906" and gfx.get("xnack_line") is not None
    gpus = probe.get("gpus", [])
    bar_ok = bool(gpus) and max((g.get("max_bar_bytes", 0) for g in gpus), default=0) >= 32 * 1024**3
    return {
        "gfx_target_ok": gfx_ok,
        "gfx_target_detail": gfx,
        "bar_ok": bar_ok,
        "bar_detail": [{"bdf": g["bdf"], "max_bar_bytes": g.get("max_bar_bytes"),
                        "bar_above_4g": g.get("bar_above_4g")} for g in gpus],
    }


def emit() -> int:
    probe = build_probe()
    print(json.dumps(probe, indent=2, sort_keys=True))
    return 0


def ingest(path: Path) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import results  # noqa: E402  (harness writer; repo-side only)
    probe = json.loads(path.read_text(encoding="utf-8"))
    needed = ["gpu_count", "gpus", "gfx_target", "guest_kernel", "rocm_version"]
    missing = [k for k in needed if k not in probe]
    if missing:
        print(f"gpu_probe: invalid probe json, missing {missing}", file=sys.stderr)
        return 1
    verdict = acceptance(probe)
    records = [{"probe": probe, "acceptance": verdict}]
    results.write_artifact("gpu_probe.jsonl", records)
    # one run row for the S0 GPU checks (the probe process itself exited 0 on the guest)
    row = results.make_row(
        task_id="S0",
        exit_code=0,
        wall_seconds=0.0,
        timeout_s=0.0,
        notes=(f"gpu_probe ingested from {path.name}: gfx={verdict['gfx_target_detail']} "
               f"bar_ok={verdict['bar_ok']} gpu_count={probe.get('gpu_count')}"),
        gpu_count=probe.get("gpu_count"),
        kernel_config=f"host={probe.get('probe_host')} kernel={probe.get('guest_kernel')}",
    )
    results.append_row(row)
    print(f"gpu_probe: wrote results/gpu_probe.jsonl + results/S0.jsonl row {row['run_id']}")
    print(f"gpu_probe: gfx_target_ok = {verdict['gfx_target_ok']}  "
          f"({verdict['gfx_target_detail']})")
    print(f"gpu_probe: bar_ok = {verdict['bar_ok']}  ({verdict['bar_detail']})")
    ok = verdict["gfx_target_ok"] and verdict["bar_ok"]
    print(f"gpu_probe: S0 GPU acceptance = {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[<bus>] == "--emit":
        return emit()
    if len(sys.argv) == 3 and sys.argv[<bus>] == "--write":
        return ingest(Path(sys.argv[<bus>]))
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())

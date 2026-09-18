"""env_fingerprint.py — regenerate config/environment.lock from measured facts.

I8: the environment is frozen; every results row carries the fingerprint. The
lock is written ONLY by this script (or future harness code) — never by hand.
I1: anything that cannot be observed is recorded as "not measured", never guessed.

Run:  python3 harness/env_fingerprint.py            # measure this machine
      python3 harness/env_fingerprint.py --from-probe probe.json
                                                     # regenerate from a GPU-guest
                                                     # probe emitted by gpu_probe.py
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_LOCK = ROOT / "config" / "environment.lock"

NOT_MEASURED = "not measured"


def _run(args, timeout: float = 10) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None


def _file_first(paths) -> str:
    for p in paths:
        try:
            text = Path(p).read_text(encoding="utf-8", errors="replace").strip()
            if text:
                return text
        except Exception:
            continue
    return NOT_MEASURED


def measure_gpu_count() -> int | str:
    """Count AMD/ATI VGA/3D/Display devices via lspci (vendor 1002)."""
    proc = _run(["lspci", "-d", "1002:"])
    if proc is None or proc.returncode != 0:
        return NOT_MEASURED
    lines = [ln for ln in proc.stdout.splitlines()
             if re.search(r"VGA|3D controller|Display controller|Accelerated", ln, re.I)]
    return len(lines)


def measure_rocm_version() -> str:
    return _file_first([
        "/opt/rocm/.info/version",
        "/opt/rocm/share/doc/rocm-version/version",
    ])


def measure_python_module_version(mod: str) -> str:
    try:
        proc = _run([sys.executable, "-c", f"import {mod}; print({mod}.__version__)"])
        if proc is not None and proc.returncode == 0:
            return proc.stdout.strip() or NOT_MEASURED
    except Exception:
        pass
    return NOT_MEASURED


def measure_triton_commit() -> str:
    version = measure_python_module_version("triton")
    if version == NOT_MEASURED:
        return NOT_MEASURED
    # Prefer the git commit of the installed triton package when derivable.
    proc = _run([sys.executable, "-c",
                 "import triton, pathlib, subprocess; "
                 "p=pathlib.Path(triton.__file__).resolve().parent.parent; "
                 "print(subprocess.run(['git','-C',str(p),'rev-parse','HEAD'],"
                 "capture_output=True,text=True).stdout.strip())"])
    if proc is not None and proc.returncode == 0 and proc.stdout.strip():
        return proc.stdout.strip()
    return f"version={version}"


def measure_vllm_commit() -> str:
    git_dir = ROOT / "vllm" / ".git"
    if not git_dir.exists():
        return NOT_MEASURED
    proc = _run(["git", "-C", str(ROOT / "vllm"), "rev-parse", "HEAD"])
    if proc is not None and proc.returncode == 0 and proc.stdout.strip():
        return proc.stdout.strip()
    return NOT_MEASURED


def measure_gfx_target() -> str:
    """gfx906:xnack- is the pinned build target (§3.3); measurable only on a GPU guest."""
    proc = _run(["rocminfo"])
    if proc is None or proc.returncode != 0:
        return NOT_MEASURED
    if "gfx906" in proc.stdout and "xnack" in proc.stdout:
        return "gfx906:xnack-"
    return NOT_MEASURED


def measure_ram_gb() -> str:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                kb = int(line.split()[<bus>])
                return f"{kb / 1024 / 1024:.1f}"
    except Exception:
        pass
    return NOT_MEASURED


def measure_pve_version() -> str:
    return _file_first(["/etc/pve/version", "/etc/pve/datacenter.cfg"])


def build_fingerprint() -> dict:
    git_proc = _run(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    git_commit = git_proc.stdout.strip() if (git_proc and git_proc.returncode == 0) else NOT_MEASURED
    status_proc = _run(["git", "-C", str(ROOT), "status", "--porcelain"])
    dirty = bool(status_proc and status_proc.returncode == 0 and status_proc.stdout.strip())

    fingerprint = {
        "_generated_by": "harness/env_fingerprint.py",
        "_generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "_git_commit": git_commit,
        "_dirty_flag": dirty,
        "hostname": platform.node(),
        "guest_kernel": platform.release(),
        "kernel_version": platform.release(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "ram_gb": measure_ram_gb(),
        "rocm_version": measure_rocm_version(),
        "torch_version": measure_python_module_version("torch"),
        "triton_commit": measure_triton_commit(),
        "vllm_commit": measure_vllm_commit(),
        "gfx_target": measure_gfx_target(),
        "pve_version": measure_pve_version(),
        "gpu_count": measure_gpu_count(),
    }
    return fingerprint


def build_from_probe(probe: dict) -> dict:
    """Fingerprint of the GPU guest, taken from a gpu_probe.py --emit document."""
    gfx = probe.get("gfx_target", {})
    gfx_target = NOT_MEASURED
    if gfx.get("rocminfo_name"):
        gfx_target = f"{gfx['rocminfo_name']}:xnack-" if gfx.get("xnack_line") else f"{gfx['rocminfo_name']}:xnack?"
    fingerprint = {
        "_generated_by": "harness/env_fingerprint.py (--from-probe)",
        "_generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "_probe_host": probe.get("probe_host"),
        "_probe_time": probe.get("probe_time"),
        "_git_commit": NOT_MEASURED,
        "_dirty_flag": None,
        "hostname": probe.get("probe_host"),
        "guest_kernel": probe.get("guest_kernel"),
        "kernel_version": probe.get("guest_kernel"),
        "python_version": probe.get("python_version", NOT_MEASURED),
        "cpu_count": probe.get("cpu_count"),
        "ram_gb": str(probe.get("ram_gb", NOT_MEASURED)),
        "rocm_version": probe.get("rocm_version", NOT_MEASURED),
        "torch_version": NOT_MEASURED,
        "triton_commit": NOT_MEASURED,
        "vllm_commit": NOT_MEASURED,
        "gfx_target": gfx_target,
        "pve_version": probe.get("pve_version", NOT_MEASURED),
        "gpu_count": probe.get("gpu_count", 0),
    }
    return fingerprint


def write_lock(fingerprint: dict) -> int:
    ENV_LOCK.parent.mkdir(parents=True, exist_ok=True)
    tmp = ENV_LOCK.with_suffix(".lock.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(fingerprint, fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, ENV_LOCK)
    print(json.dumps(fingerprint, indent=2, sort_keys=True))
    print(f"env_fingerprint: wrote {ENV_LOCK}")
    return 0


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[<bus>] == "--from-probe":
        probe_path = Path(sys.argv[<bus>])
        try:
            probe = json.loads(probe_path.read_text(encoding="utf-8"))
        except Exception as exc:
            sys.exit(f"env_fingerprint: cannot read probe json {probe_path}: {exc}")
        return write_lock(build_from_probe(probe))
    fingerprint = build_fingerprint()
    return write_lock(fingerprint)


if __name__ == "__main__":
    sys.exit(main())

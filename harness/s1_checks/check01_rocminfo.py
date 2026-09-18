#!/usr/bin/env python3
"""S1 check 01 — rocminfo reports gfx906 with xnack- (work order §6.S1)."""
import re
import subprocess
import sys


def main() -> int:
    proc = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        print(f"FAIL: rocminfo exited {proc.returncode}")
        return 1
    out = proc.stdout
    name_m = re.search(r"Name:\s+(gfx\d+)", out)
    gfx = name_m.group(1) if name_m else None
    xnack = "XNACK enabled:" in out and "XNACK enabled:           NO" in out
    print(f"rocminfo gfx = {gfx}, xnack_disabled = {xnack}")
    ok = gfx == "gfx906" and xnack
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

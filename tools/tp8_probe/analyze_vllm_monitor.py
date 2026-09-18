#!/usr/bin/env python3
"""analyze_vllm_monitor.py (v2) — what did the hardware do during the TP4 baseline re-run?

FIX over v1: rocm-smi prints clocks as `sclk clock level: 8: (1800Mhz)`. v1's regex took the FIRST
number, i.e. the LEVEL INDEX, and reported a median of 6 MHz -- physically impossible, and it made
the throttle verdict exactly backwards. v2 reads the MHz value inside the parentheses.

The question that matters: is the modest decode difference explained by throttling, or was the rig
at full clocks? And crucially, is the limiter THERMAL (dangerous, varies day to day) or the POWER
CAP (expected, and a fixed property of the configured operating point)?
"""
import re
import statistics
import sys
from collections import defaultdict

PATH = sys.argv[<bus>] if len(sys.argv) > 1 else \
    "<tuning>/tp8_probe/rocm_smi_during_vllm.txt"

gpu = defaultdict(lambda: defaultdict(list))
timestamps = []

CUR = re.compile(r"GPU\[(\d+)\]\s*:\s*(.*)")
MHZ = re.compile(r"\((\d+)\s*Mhz\)", re.I)
NUM = re.compile(r"(-?\d+\.?\d*)")

DPM_SCLK = {0: 925, 1: 938, 2: 1076, 3: 1179, 4: 1264, 5: 1339, 6: 1556, 7: 1711, 8: 1800}

with open(PATH, errors="replace") as fh:
    for line in fh:
        line = line.rstrip("\n")
        if re.match(r"^\d{4}-\d{2}-\d{2}T", line):
            timestamps.append(line)
            continue
        m = CUR.match(line.strip())
        if not m:
            continue
        idx, rest = int(m.group(1)), m.group(2)

        def val(pat=None):
            if pat:
                mm = re.search(pat, rest)
                return float(mm.group(1)) if mm else None
            mm = NUM.search(rest.split(":", 1)[-1] if ":" in rest else rest)
            return float(mm.group(1)) if mm else None

        if "Graphics Package Power" in rest:
            v = val()
            if v is not None:
                gpu[idx]["power"].append(v)
        elif "sclk clock level" in rest:
            mm = MHZ.search(rest)
            if mm:
                gpu[idx]["sclk"].append(int(mm.group(1)))
        elif "mclk clock level" in rest:
            mm = MHZ.search(rest)
            if mm:
                gpu[idx]["mclk"].append(int(mm.group(1)))
        elif "GPU use" in rest:
            v = val()
            if v is not None:
                gpu[idx]["use"].append(v)
        elif "Sensor edge" in rest:
            v = val()
            if v is not None:
                gpu[idx]["edge"].append(v)
        elif "Sensor junction" in rest:
            v = val()
            if v is not None:
                gpu[idx]["tj"].append(v)
        elif "Sensor memory" in rest:
            v = val()
            if v is not None:
                gpu[idx]["tmem"].append(v)

print(f"samples: {len(timestamps)}")
if timestamps:
    print(f"window : {timestamps[<bus>]} -> {timestamps[-1]}")
print()

print("=== CLOCKS (MHz, from the parenthesised value) ===")
for i in sorted(gpu):
    s = gpu[i]["sclk"]
    mm = gpu[i]["mclk"]
    if not s:
        continue
    at_max = sum(1 for v in s if v >= 1800)
    print(f"  GPU{i}: sclk med {statistics.median(s):>6.0f}  min {min(s):>5}  max {max(s):>5}  "
          f"| at 1800: {100*at_max/len(s):>3.0f}% of samples  "
          f"| mclk med {statistics.median(mm) if mm else 0:.0f}")

print()
print("=== POWER vs the 150 W CONFIGURED CAP ===")
for i in sorted(gpu):
    p = gpu[i]["power"]
    if not p:
        continue
    at_cap = sum(1 for v in p if v >= 145)
    print(f"  GPU{i}: med {statistics.median(p):>6.1f} W  min {min(p):>6.1f}  max {max(p):>6.1f}  "
          f"| >=145 W in {100*at_cap/len(p):>3.0f}% of samples")

print()
print("=== TEMPERATURES (C) ===")
for i in sorted(gpu):
    tj, te = gpu[i]["tj"], gpu[i]["edge"]
    if not tj:
        continue
    print(f"  GPU{i}: junction med {statistics.median(tj):>5.1f}  max {max(tj):>5.1f}   "
          f"edge med {statistics.median(te):>5.1f}  max {max(te):>5.1f}   "
          f"memory max {max(gpu[i]['tmem']) if gpu[i]['tmem'] else 0:>5.1f}")

print()
print("=== GFX ACTIVITY ===")
for i in sorted(gpu):
    u = gpu[i]["use"]
    if u:
        print(f"  GPU{i}: med {statistics.median(u):.0f}%  min {min(u):.0f}%  max {max(u):.0f}%")

print()
print("=== VERDICT ===")
all_sclk = [v for i in gpu for v in gpu[i]["sclk"]]
all_pw = [v for i in gpu for v in gpu[i]["power"]]
all_tj = [v for i in gpu for v in gpu[i]["tj"]]
if all_sclk:
    print(f"  clock range reached      : {min(all_sclk)}-{max(all_sclk)} MHz "
          f"(dpm levels {sorted(set(all_sclk))})")
    print(f"  reached full 1800 MHz    : {max(all_sclk) >= 1800}")
if all_pw:
    print(f"  sitting at the 150 W cap : {statistics.median(all_pw):.1f} W median "
          f"-> {'YES' if statistics.median(all_pw) >= 140 else 'no'}")
if all_tj:
    print(f"  junction peak            : {max(all_tj):.0f} C "
          f"-> {'THERMAL LIMITING LIKELY' if max(all_tj) >= 100 else 'below the ~100 C throttle region'}")
print()
print("  Reading: if the cards sit at the configured power cap, clock variation across the run is")
print("  POWER-limited behaviour, which is expected and reproducible. If instead they were pinned at")
print("  full clock and merely hot, the limiter would be thermal and would vary with ambient/day.")

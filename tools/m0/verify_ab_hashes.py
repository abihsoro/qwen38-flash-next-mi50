#!/usr/bin/env python3
"""verify_ab_hashes.py — confirm every MoE A/B gate artifact carries the reference hash."""
import glob
import json
import os

REF = "0128852903291e32"
files = sorted(glob.glob("<workdir>/moeab2_*.json"))
if not files:
    print("NO ARTIFACTS FOUND")
    raise SystemExit(2)

print("gate artifacts (sha256 of the full greedy output):")
allok = True
for f in files:
    d = json.load(open(f))
    sha = d["correctness_a"]["sha256"]
    det = bool(d["deterministic"])
    med = d["baseline_median_tok_per_s"]
    ok = sha.startswith(REF) and det
    allok = allok and ok
    tag = "OK" if ok else "MISMATCH"
    print(f"  {os.path.basename(f):24} median={med:<9} det={det} sha={sha[:16]} {tag}")

print()
print(f"reference hash      : {REF}")
print(f"ALL {len(files)} ARTIFACTS MATCH THE REFERENCE HASH: {allok}")

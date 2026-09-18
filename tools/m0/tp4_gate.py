#!/usr/bin/env python3
"""tp4_gate.py — Phase 3/4 gate: deterministic greedy correctness + baseline timing.

Discipline (TP4_BRINGUP_PLAN):
  * deterministic greedy (temperature 0), ignore_eos so a short EOS can't fake a result
  * capture the FULL output text + sha256; two identical runs must hash identically
  * real token counts from the API's usage field, not estimates
  * >= 3 serial repeats for the timing; report median, no profiler, no side jobs
  * the number is labelled BASELINE ONLY
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8002"
MODEL = "qwen38-flash-next"
# Label the graph/compile mode from the environment so the artifact cannot mislabel itself.
GRAPHS_MODE = os.environ.get("TP4_GRAPHS_MODE", "unknown")
PROMPT = ("Explain the difference between a PCIe switch and a PCIe bridge "
          "in exactly three sentences.")


def post(path, payload, timeout=900):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def one_run(maxtok, tag):
    body = {
        "model": MODEL,
        "prompt": PROMPT,
        "max_tokens": maxtok,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": True,          # TRAPS #6: EOS must not truncate a rate run
        "seed": 0,
    }
    t0 = time.perf_counter()
    out = post("/v1/completions", body)
    wall = time.perf_counter() - t0
    text = out["choices"][<bus>]["text"]
    usage = out.get("usage", {})
    ct = usage.get("completion_tokens")
    return {
        "tag": tag,
        "completion_tokens": ct,
        "wall_s": round(wall, 4),
        "tok_per_s": round(ct / wall, 3) if ct and wall else None,
        "first_64_tokens": text[:400],
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text_len_chars": len(text),
    }


def main():
    max_tokens = int(sys.argv[<bus>]) if len(sys.argv) > 1 else 64
    reps = int(sys.argv[<bus>]) if len(sys.argv) > 2 else 3

    print(f"=== correctness: 2 deterministic greedy runs @ max_tokens={max_tokens} ===")
    a = one_run(max_tokens, "corr-A")
    b = one_run(max_tokens, "corr-B")
    print(f"  A: tokens={a['completion_tokens']} sha={a['sha256'][:16]} wall={a['wall_s']}s")
    print(f"  B: tokens={b['completion_tokens']} sha={b['sha256'][:16]} wall={b['wall_s']}s")
    deterministic = a["sha256"] == b["sha256"]
    print(f"  DETERMINISTIC: {deterministic}")
    print(f"  output[0:200]: {a['first_64_tokens'][:200]!r}")

    print(f"\n=== baseline timing: {reps} serial repeats, max_tokens=128 ===")
    runs = []
    for i in range(reps):
        r = one_run(128, f"bench-{i+1}")
        runs.append(r)
        print(f"  rep{i+1}: tokens={r['completion_tokens']} wall={r['wall_s']}s "
              f"tok/s={r['tok_per_s']}")
    tps = [r["tok_per_s"] for r in runs if r["tok_per_s"]]
    med = statistics.median(tps) if tps else None
    print(f"  MEDIAN tok/s = {med}  (spread {(max(tps)-min(tps)):.3f})" if tps else "  no rates")

    result = {
        "phase": "phase3_tp4_fully_resident",
        "config": {"tp": 4, "ep": True, "mtp": 0, "dtype": "float16",
                   "graphs": GRAPHS_MODE, "offload": "none",
                   "ple": "int4_sidecar", "max_model_len": 2048},
        "deterministic": deterministic,
        "correctness_a": a, "correctness_b": b,
        "baseline_runs": runs,
        "baseline_median_tok_per_s": med,
        "label": "BASELINE ONLY - not a tuned number",
    }
    with open("<workdir>/tp4_gate.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print("\nwrote <workdir>/tp4_gate.json")


if __name__ == "__main__":
    main()

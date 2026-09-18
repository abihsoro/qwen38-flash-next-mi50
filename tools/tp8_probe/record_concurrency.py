#!/usr/bin/env python3
"""record_concurrency.py — register the decode concurrency sweep in the tracking documents.

Adds TP8-5 (the sweep, done) and retargets TP8-4's justification from throughput to capacity, since
the sweep refuted the weight-bandwidth case. Enforces the pipe-separator guard.
"""
import json
from pathlib import Path

R = Path("<home>/<share>/70_Code/qwen38-flash-next-mi50")
P = R / "config" / "tasks.json"
d = json.loads(P.read_text())
SEP = " \u00b7 "
changed = []

ids = {t["id"] for t in d["tasks"]}
if "TP8-5" not in ids:
    t5 = {
        "id": "TP8-5",
        "title": "TP8 probe - decode concurrency sweep: aggregate FLAT 1->8, refutes the weight-bandwidth case (D150)",
        "human": False, "preconditions": ["TP8-3"],
        "acceptance": "Aggregate and per-request throughput, round latency median/p95, and GPU power measured at 1/2/4/8 concurrent decode-heavy requests",
        "status": "done",
        "progress": (
            "DONE (D150 addendum). 1K prompt, 512 generated, greedy, STREAMING (so round latency is "
            "measured, not inferred), unique prompt salt per request, GPU sampled at 1 Hz per phase. "
            "RESULTS: aggregate 44.41 / 40.21 / 45.24 / 40.30 tok/s at concurrency 1 / 2 / 4 / 8 = "
            "FLAT (12% band, no trend); scaling 1.00x / 0.91x / 1.02x / 0.91x. Ideal linear scaling "
            "at 4 would be 178 tok/s, measured 45 = NONE. Per-request 52.39 / 25.37 / 13.62 / 11.37 "
            "tok/s (scales ~1/n as expected). Round median 43.2 / 66.0 / 197.2 / 200.6 ms and p95 "
            "143.6 / 174.8 / 308.0 / 348.5 ms with multi-second stalls (3466 ms max at 2), so "
            "interactive feel degrades under load. GPU POWER FALLS as concurrency rises - 150.0 / "
            "146.5 / 111.0 / 109.0 W - while the use counter stays ~100%, i.e. the SMs are STALLED "
            "WAITING, not saturated: the workload is LATENCY/SERIALIZATION BOUND and the 100% use "
            "figure counts issue cycles, not useful work. THIS REFUTES the weight-bandwidth case for "
            "TP8 (see D150 section 6): if weights bound decode, batching 4 would read them once and "
            "emit 4x the tokens; it emitted 1.0x. So the earlier 18.23 GiB / 810 GB/s ~= 44 tok/s "
            "agreement with the measured ~47.7 was a NUMERICAL COINCIDENCE - the MoE is sparse "
            "(top-10 of 128 local experts) so the full per-rank footprint is never read per token. "
            "CONSISTENT WITH D145/D146: the elementwise/copy residue is 303k latency-bound launches. "
            "Measurement trap hit and fixed: the first pass counted streamed CHUNKS as tokens, "
            "understating throughput ~2.8x because MTP2 puts several accepted tokens in one chunk; "
            "usage.completion_tokens is authoritative and the ratios were unaffected."
        ),
    }
    idx = next((i for i, x in enumerate(d["tasks"]) if x["id"] == "TP8-4"), len(d["tasks"]) - 1)
    d["tasks"].insert(idx, t5)
    changed.append("TP8-5 added (concurrency sweep)")

for t in d["tasks"]:
    if t["id"] == "TP8-4":
        t["title"] = ("TP8 go/no-go: 56-pair fabric matrix and world=8 RCCL - now justified on "
                      "CAPACITY, not throughput (D150 addendum)")
        t["progress"] = (
            "NEXT (D150 recommendation, awaiting hardware) - JUSTIFICATION RETARGETED. The decode "
            "concurrency sweep (TP8-5) showed aggregate throughput is FLAT from 1 to 8 concurrent and "
            "that GPU power FALLS with concurrency, so decode is latency/serialization bound and TP8 "
            "would not improve throughput single-stream or multi-stream. The weight-bandwidth case "
            "for TP8 is refuted. What remains is CAPACITY: per-rank footprint 18.23 -> ~9.1 GiB, "
            "which buys far more than the current 32K context or allows unquantized weights. Proceed "
            "on that basis only. STILL WORTH DOING FIRST (physics, before any software work): "
            "1) populate the empty PEX88096 ports ([<bus>]/[<bus>] and/or behind [<bus>]) and re-run D141's "
            "fabric matrix, 12 ordered pairs -> 56; all 56 byte-exact at ~14 GB/s settles the "
            "interconnect and any pair falling to host-bounce or RC-crossing speeds stops the "
            "project. 2) Re-measure RCCL at world=8 - small-message all-reduce latency at eight ranks "
            "is the single number that decides whether ANY decode gain survives. 3) Only then the "
            "software side (RCCL/custom-AR at world=8, EP sharding across 8 ranks). POWER/COOLING: "
            "4 x 150 W = 600 W today and eight cards would be 1200 W if per-card draw held, with GPU1 "
            "already at 97 C junction; check PSU and enclosure cooling before committing."
        )
        changed.append("TP8-4 retargeted to capacity, not throughput")

d.setdefault("status_update", {}).setdefault("bullets", []).append(
    "**Decode concurrency sweep: aggregate throughput is FLAT 1->8, and it REFUTES the weight-"
    "bandwidth case for TP8 (D150 addendum).** With 1K prompts and 512 generated tokens at 1/2/4/8 "
    "concurrent requests, aggregate throughput measured **44.4 / 40.2 / 45.2 / 40.3 tok/s** - flat "
    "within a 12% band, i.e. **1.0x scaling where ideal would be 4x**. Per-request falls ~1/n as "
    "expected (52.4 -> 11.4 tok/s), round p95 degrades 144 -> 348 ms with multi-second stalls. The "
    "decisive evidence is the POWER: **150.0 / 146.5 / 111.0 / 109.0 W** while the GPU use counter "
    "stays ~100%, so the SMs are **stalled waiting, not saturated** - the workload is latency/"
    "serialization bound. **This falsifies my own D150 projection:** if weights bound decode, "
    "batching 4 would read them once and emit 4x the tokens; it emitted 1.0x, so the earlier "
    "18.23 GiB / 810 GB/s ~= 44 tok/s agreement with ~47.7 was a numerical coincidence (the MoE is "
    "sparse, so the full per-rank footprint is never read per token). **Revised verdict: TP8 will "
    "not improve decode throughput, single-stream or multi-stream; its case is CAPACITY** (far more "
    "than 32K context, or unquantized weights). TP8-4 is retargeted accordingly, and the physical "
    "pre-checks (56-pair matrix, world=8 RCCL) still stand. Consistent with D145/D146: the "
    "elementwise/copy residue is 303k latency-bound launches."
)
changed.append("status bullet appended (concurrency sweep)")

bad = []
for t in d["tasks"]:
    for f in ("progress", "title", "acceptance"):
        v = t.get(f)
        if isinstance(v, str) and " | " in v:
            t[<bus>] = v.replace(" | ", SEP)
            bad.append(f"{t['id']}.{f}")
if bad:
    changed.append("pipe guard fired: " + ", ".join(bad))

P.write_text(json.dumps(d, indent=2) + "\n")
print("changes:")
for c in changed:
    print("  -", c)

#!/usr/bin/env python3
"""record_tp8_verdict.py — record the operator's FINAL TP8 verdict (D151) in the tracking docs.

TP8-4 stops being a throughput question and becomes a capacity question. The framing here is the
operator's, recorded verbatim in D151; this script updates the registry so PROGRESS.md carries it.
"""
import json
from pathlib import Path

R = Path("<home>/<share>/70_Code/qwen38-flash-next-mi50")
P = R / "config" / "tasks.json"
d = json.loads(P.read_text())
SEP = " \u00b7 "
changed = []

for t in d["tasks"]:
    if t["id"] == "TP8-4":
        t["title"] = ("TP8 go/no-go - CAPACITY decision only: TP8 is a cheap-capacity strategy, NOT a "
                      "performance strategy (D151)")
        t["progress"] = (
            "FINAL VERDICT (D151, operator conclusion) - read this before acting on the D150 "
            "recommendation. **TP8 is a cheap-capacity strategy, not a performance strategy.** It may "
            "let the rig run models or context sizes that are otherwise uneconomical, but it should "
            "NOT be expected to deliver better decode throughput than the current TP4 setup. "
            "REALISTIC EXPECTATION: TP8 decode will probably land BELOW the current ~40-60 tok/s "
            "aggregate range - unless the 8-GPU machine has a MUCH better interconnect/topology than "
            "this one. USE-CASE OUTLOOK: single-stream decode likely WORSE than TP4; multi-user decode "
            "throughput likely FLAT to worse; long prefill/TTFT maybe acceptable, possibly modestly "
            "better; capacity per dollar is STILL THE MAIN REASON to do TP8; the overall 'cheap "
            "128 GB serving box' is viable only if speed expectations are modest. FOUR SIGNALS: "
            "aggregate decode stays ~40-45 tok/s from 1 to 8 concurrency (more requests do not fill "
            "unused compute); per-request speed collapses as concurrency rises (requests share the "
            "same bottleneck); GPU power DROPS to ~111 W at 4-8 concurrency (the GPUs are waiting "
            "more, not working harder); p95 rises sharply (scheduler/communication bubbles become "
            "user-visible). If this were compute-limited, aggregate would CLIMB with concurrency and "
            "power would stay near 150 W - instead aggregate is flat and power falls, pointing to "
            "synchronization, communication, scheduler overhead or CPU/NUMA stalls rather than a lack "
            "of active requests. THE POWER DROP IS DECISIVE: at 4-way concurrency there should have "
            "been plenty of queued work, yet the rig backed off to ~111 W - a saturated machine cannot "
            "reduce its draw when handed MORE work. AND ADDING RANKS IS UNLIKELY TO FIX IT: four more "
            "TP ranks usually INCREASE collective cost and synchronization pressure, so TP8 pushes "
            "the wrong way on the exact axis that already limits the rig. The one escape hatch is a "
            "materially better interconnect/topology than the PEX88096 one measured here - which is "
            "what the physical pre-checks establish. SO: the checks remain worth doing but justified "
            "by CAPACITY PER DOLLAR, not by any decode-speed expectation. If the goal is decode "
            "throughput, the lever is the per-token serialized work identified in D145/D146 and "
            "confirmed here - not more cards."
        )
        changed.append("TP8-4 retitled and rewritten as the D151 capacity verdict")

# retire the now-moot throughput framing on TP8-5
for t in d["tasks"]:
    if t["id"] == "TP8-5":
        t["progress"] = (t.get("progress", "").strip() + SEP if t.get("progress") else "") + (
            "CONSEQUENCE (D151): this sweep is the evidence base for the final verdict that TP8 is a "
            "cheap-capacity strategy, not a performance strategy - aggregate decode flat from 1 to 8 "
            "concurrency, GPU power FALLING to ~111 W while 'use' stays 100%, and sharply rising p95. "
            "Expect TP8 decode to land BELOW the current ~40-60 tok/s aggregate range unless the "
            "8-GPU machine has a much better interconnect. Do not treat these numbers as a baseline "
            "that TP8 is expected to beat."
        )
        changed.append("TP8-5 annotated with the D151 consequence")

d.setdefault("status_update", {}).setdefault("bullets", []).append(
    "**FINAL TP8 VERDICT: a cheap-capacity strategy, not a performance strategy (D151).** "
    "**TP8 is expected to land BELOW the current ~40-60 tok/s aggregate decode range** unless the "
    "8-GPU machine has a much better interconnect/topology than this one. Single-stream decode likely "
    "worse than TP4; multi-user decode throughput likely flat to worse; long prefill/TTFT maybe "
    "acceptable, possibly modestly better; **capacity per dollar remains the only strong reason to do "
    "TP8**, and the 'cheap 128 GB serving box' is viable only with modest speed expectations. The "
    "decisive evidence is the part that is hardest to argue with: **GPU power DROPPED to ~111 W at "
    "4-way concurrency while the use counter stayed at 100%** - a saturated machine cannot reduce its "
    "draw when handed MORE work. Aggregate throughput was flat from 1 to 8 concurrency (ideal 4x, "
    "measured 1.0x), per-request speed collapsed, and p95 climbed 144 -> 348 ms. That points to "
    "synchronization, communication, scheduler overhead or CPU/NUMA stalls rather than a lack of "
    "active requests - and adding four more TP ranks would INCREASE collective cost and sync "
    "pressure, pushing the wrong way on the exact axis that already limits the rig. **If the goal is "
    "decode throughput, the lever is the per-token serialized work identified in D145/D146 - not more "
    "cards.** The physical pre-checks (56-pair fabric matrix, world=8 RCCL) remain worth doing but "
    "are now justified by capacity per dollar. The operating configuration is unchanged: MTP2 / 32K / "
    "150 W with the D148/D149 presets."
)
changed.append("status bullet appended (D151 final verdict)")

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

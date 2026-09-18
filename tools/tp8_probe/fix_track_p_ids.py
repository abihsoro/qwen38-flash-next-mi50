#!/usr/bin/env python3
"""fix_track_p_ids.py — repair the ID collision introduced by record_tp8_probe.py.

That script keyed "already present?" on task id alone. P1 and P2 were ALREADY TAKEN by unrelated
pre-existing tasks ("Resolve the V620 source", "blocked-on-hardware ACS determination"), so the
topology and RCCL tasks were silently skipped while P3/P4 were added - giving a Track P that was
both incomplete and misleadingly numbered.

This renames the TP8 probe tasks to unique TP8-N ids and adds the two that were dropped, so the
probe reads as one coherent lane.
"""
import json
from pathlib import Path

R = Path("<home>/<share>/70_Code/qwen38-flash-next-mi50")
P = R / "config" / "tasks.json"
d = json.loads(P.read_text())
SEP = " \u00b7 "
changed = []

# 1) re-id the two TP8 tasks that WERE added, identified by their content not their id
RENAME = {
    "P3": ("TP8-3", "TP8 probe - fresh TP4 baseline under teardown-day state (D150)"),
    "P4": ("TP8-4", "TP8 go/no-go: 56-pair fabric matrix and world=8 RCCL on the populated switch"),
}
for t in d["tasks"]:
    if t["id"] in RENAME:
        old = t["id"]
        t["id"], t["title"] = RENAME[old]
        changed.append(f"{old} -> {t['id']}")

# 2) add the two that were dropped
ids = {t["id"] for t in d["tasks"]}
DROPPED = [
    {
        "id": "TP8-1",
        "title": "TP8 probe - topology: one switch, one root complex, one NUMA node (D150)",
        "human": False, "preconditions": ["M0", "M4"],
        "acceptance": "Topology captured from host and container; every GPU pair uniform; spare switch ports identified",
        "status": "done",
        "progress": (
            "DONE (D150). All four MI50s hang off ONE PEX88096 under ONE root complex ([<bus>]) via "
            "one GPP bridge (<pci>). rocm-smi --showtopo: EVERY pair identical - weight 40, 2 hops, "
            "link type PCIE - and all four GPUs on NUMA node 1. No XGMI (all inter-GPU traffic is "
            "PCIe, expected for MI50 on a non-AMD platform); agrees with D141's 12/12 ordered pairs "
            "byte-exact at 14.40-14.44 GB/s. HEADROOM CONFIRMED: two downstream ports are unpopulated "
            "- <pci> -> [<bus-range>] with two empty ports [<bus>]/[<bus>], and <pci> -> [<bus>] carrying a WHOLE "
            "SECOND PEX88096 with nothing behind it - so four more cards can be added WITHOUT crossing "
            "root complexes or NUMA nodes. Useful precedent: GPU2/GPU3 already sit behind a "
            "SECOND-LEVEL switch (ae via <pci>) while GPU0/GPU1 are direct, yet weights, hops and "
            "measured bandwidth stay uniform, so an extra switch stage is not automatically harmful. "
            "CORRECTS D078: all four report pcie 16.0GT/s x16 = Gen4 x16, NOT the Gen3 x16 max D078 "
            "recorded; corroborated by the measured 14.4 GB/s (~90% of a Gen4 x16 budget, roughly "
            "double what Gen3 x16 could sustain)."
        ),
    },
    {
        "id": "TP8-2",
        "title": "TP8 probe - RCCL collectives at world=4 (D150)",
        "human": False, "preconditions": ["TP8-1"],
        "acceptance": "all_reduce and all_gather measured small and large, every case validating with zero wrong elements",
        "status": "done",
        "progress": (
            "DONE (D150). Built ROCm/rccl-tests in <container> - see TRAPS #21 for the three build/run "
            "blockers (no git; hipify-perl off-PATH; no rpath). RESULTS: all_reduce small-message "
            "latency 24.6-30.8 us from 2 KiB to 16 KiB, FLAT and stable; all_gather 27.1-31.1 us over "
            "2-32 KiB. Large messages saturate cleanly - all_reduce 14.19 GB/s algbw / 21.29 busbw at "
            "256 MiB, all_gather 26.7 / 20.0 - with NO degradation to 256 MiB, matching the fabric's "
            "measured per-pair capability. All cases #wrong = 0; out-of-place and in-place agree. "
            "VERDICT against the operator's rules: small messages are neither poor nor erratic AND "
            "large-message bandwidth is strong, so the interconnect does NOT block TP8. WATCH ITEM: a "
            "hard algorithm knee at 32 KiB in all_reduce (25.47 us @ 16 KiB -> 42.02 us @ 32 KiB, +65% "
            "latency for 2x payload); the workload's dominant AR message is ~20 KB, just below that "
            "cliff. UN-RECONCILED: three incompatible figures now exist for RCCL world=4 latency at "
            "~20 KB - D141 records 106.96 us, the expert review ~18.4 us (per-rank kernel medians), and "
            "rccl-tests 25-42 us (whole-op wall). They time different things; do not mix them, and "
            "treat D141's as un-reconciled (3-4x higher than a whole-op measurement of the same op)."
        ),
    },
]
for t in DROPPED:
    if t["id"] not in ids:
        idx = next((i for i, x in enumerate(d["tasks"]) if x["id"] == "TP8-3"), len(d["tasks"]) - 1)
        d["tasks"].insert(idx, t)
        changed.append(f"{t['id']} added")

# 3) keep the lane contiguous and ordered
order = {"TP8-1": 0, "TP8-2": 1, "TP8-3": 2, "TP8-4": 3}
first = min((i for i, t in enumerate(d["tasks"]) if t["id"] in order), default=None)
if first is not None:
    lane = sorted([t for t in d["tasks"] if t["id"] in order], key=lambda t: order[t["id"]])
    rest = [t for t in d["tasks"] if t["id"] not in order]
    d["tasks"] = rest[:first] + lane + rest[first:]
    changed.append("Track P lane reordered contiguously (TP8-1..TP8-4)")

# 4) pipe guard
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

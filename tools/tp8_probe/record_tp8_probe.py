#!/usr/bin/env python3
"""record_tp8_probe.py — register the TP8 feasibility probe (D150) in the tracking documents.

Adds a Track P (TP8 feasibility) lane with three tasks reflecting what was actually done, updates
the G6/gate context to note the fresh reference, and appends a status bullet.

Separator is " · " never " | " — a literal pipe adds a markdown column and breaks the table. This is
the fourth time that trap has appeared in this project; the replace guard at the end of this script
enforces it rather than relying on care.
"""
import json
from pathlib import Path

R = Path("<home>/<share>/70_Code/qwen38-flash-next-mi50")
P = R / "config" / "tasks.json"
d = json.loads(P.read_text())
SEP = " \u00b7 "
changed = []

EXISTING = {t["id"] for t in d["tasks"]}

NEW = [
    {
        "id": "P1",
        "title": "TP8 fabric probe - topology: one switch, one root complex, one NUMA node (D150)",
        "human": False,
        "preconditions": ["M0", "M4"],
        "acceptance": ("Topology captured from both the PVE host and <container>; every GPU pair uniform; "
                       "spare switch ports identified for four more cards"),
        "status": "done",
        "progress": (
            "DONE (D150). All four MI50s hang off ONE PEX88096 under ONE root complex ([<bus>]) via "
            "one GPP bridge (<pci>). rocm-smi --showtopo: every pair identical - weight 40, 2 hops, "
            "link type PCIE - and all four GPUs on NUMA node 1. No XGMI (all inter-GPU traffic is "
            "PCIe, expected for MI50 on a non-AMD platform); agrees with D141's 12/12 ordered pairs "
            "byte-exact at 14.40-14.44 GB/s. HEADROOM CONFIRMED: two downstream ports are unpopulated "
            "- <pci> -> [<bus-range>] with two empty ports [<bus>]/[<bus>], and <pci> -> [<bus>] carrying a "
            "WHOLE SECOND PEX88096 with nothing behind it - so four more cards can be added WITHOUT "
            "crossing root complexes or NUMA nodes. Useful precedent: GPU2/GPU3 already sit behind a "
            "SECOND-LEVEL switch (ae via <pci>) while GPU0/GPU1 are direct, yet weights, hops and "
            "measured bandwidth are uniform, so an extra switch stage is not automatically harmful. "
            "CORRECTS D078: all four report pcie 16.0GT/s x16 = Gen4 x16, not the \"Gen3 x16 max\" "
            "D078 recorded - corroborated by the measured 14.4 GB/s (~90% of a Gen4 x16 budget, "
            "roughly double what Gen3 x16 could do)."
        ),
    },
    {
        "id": "P2",
        "title": "TP8 fabric probe - RCCL collectives at world=4 (D150)",
        "human": False,
        "preconditions": ["P1"],
        "acceptance": ("all_reduce and all_gather measured small (1K-1M) and large (1M-256M), all "
                       "cases validating with zero wrong elements"),
        "status": "done",
        "progress": (
            "DONE (D150). Built ROCm/rccl-tests in <container> (see TRAPS #21 for the three build/run "
            "blockers: no git, hipify-perl off-PATH, no rpath). RESULTS: all_reduce small-message "
            "latency 24.6-30.8 us from 2 KiB to 16 KiB, FLAT and stable; all_gather 27.1-31.1 us over "
            "2-32 KiB. Large messages saturate cleanly - all_reduce 14.19 GB/s algbw / 21.29 busbw at "
            "256 MiB, all_gather 26.7 / 20.0 - with NO degradation to 256 MiB, matching the fabric's "
            "measured per-pair capability. All cases #wrong = 0; out-of-place and in-place agree. "
            "VERDICT against the operator's rules: small messages are neither poor nor erratic AND "
            "large-message bandwidth is strong, so the interconnect does NOT block TP8. WATCH ITEM: a "
            "hard algorithm knee at 32 KiB in all_reduce (25.47 us @ 16 KiB -> 42.02 us @ 32 KiB, +65% "
            "latency for 2x payload); the workload's dominant AR message is ~20 KB, just below that "
            "cliff. UN-RECONCILED: three incompatible figures now exist for RCCL world=4 latency at "
            "~20 KB - D141 records 106.96 us, the expert review ~18.4 us (per-rank kernel medians), "
            "rccl-tests 25-42 us (whole-op wall). They time different things; do not mix them, and "
            "treat D141's as un-reconciled (3-4x higher than a whole-op measurement of the same op)."
        ),
    },
    {
        "id": "P3",
        "title": "TP8 fabric probe - fresh TP4 baseline under teardown-day state (D150)",
        "human": False,
        "preconditions": ["P1"],
        "acceptance": ("Service unchanged, 1 Hz rocm-smi sampling throughout, medians at 1K/10K/30K "
                       "compared against the saved finals"),
        "status": "done",
        "progress": (
            "DONE (D150). MTP2 service re-run unchanged with 186 x 1 Hz rocm-smi samples. PREFILL "
            "REPRODUCES FAITHFULLY: 647.4 / 809.0 / 808.4 vs saved 641.9 / 806.9 / 805.1 (+0.9%, "
            "+0.3%, +0.4%). DECODE IS MODESTLY BUT REPRODUCIBLY LOWER: 59.94 / 58.85 / 55.08 vs saved "
            "60.34 / 61.22 / 57.34 (-0.7%, -3.9%, -3.9%); the 1K and 30K ranges do not overlap the "
            "saved ones, so this is a real shift not sampling noise. CAUSE, from the monitor: the rig "
            "is POWER-CAP LIMITED, not thermally throttled - median 148 W against the 150 W cap "
            "(56-60% of samples >=145 W), sclk median 1556 MHz with full 1800 only 12-21% of the "
            "time, GFX 100%, junction peak 97 C (below the ~100 C region). GPU1 runs 14-18 C hotter "
            "than GPU3 (89/97 vs 75/79 junction); in TP4 the SLOWEST RANK GATES EVERY STEP, so a hot "
            "GPU1 costs throughput directly and this worsens with eight cards. REFERENCE DECISION per "
            "the operator's rule: USE THIS RUN for TP8 inference - PP ~647/809/808, decode "
            "~59.9/58.9/55.1 - because it is more conservative and contemporaneous with the RCCL data. "
            "POWER/COOLING CAUTION: 4 x 150 W = 600 W today; eight cards at the same cap would be "
            "1200 W if per-card draw held (TP8 halves each card's workload so per-card power should "
            "fall, but check PSU and enclosure cooling - GPU1 is already at 97 C junction)."
        ),
    },
    {
        "id": "P4",
        "title": "TP8 go/no-go: 56-pair fabric matrix and world=8 RCCL on the populated switch",
        "human": True,
        "preconditions": ["P2", "P3"],
        "acceptance": ("All 56 ordered pairs byte-exact at ~14 GB/s AND small-message all-reduce "
                       "latency at world=8 still in the tens of microseconds; otherwise stop"),
        "status": "pending",
        "progress": (
            "NEXT (D150 recommendation, awaiting hardware). The physics has to be settled before any "
            "software investment: 1) populate the empty PEX88096 ports ([<bus>]/[<bus>] and/or behind [<bus>]) "
            "and re-run D141's fabric matrix, 12 ordered pairs -> 56; all 56 byte-exact at ~14 GB/s "
            "settles the interconnect, and any pair falling to host-bounce or RC-crossing speeds stops "
            "the project there. 2) Re-measure RCCL at world=8 - small-message all-reduce latency at "
            "eight ranks is the single number that decides whether decode gains survive. Only then "
            "invest in the software side (RCCL/custom-AR at world=8, EP sharding across 8 ranks). "
            "ECONOMICS: the case for TP8 is WEIGHT BANDWIDTH, not collectives - at TP4 each rank "
            "holds ~18.23 GiB and decode is weight-bandwidth-bound (18.23 GiB / ~810 GB/s ~= 22.5 "
            "ms/token ~= 44 tok/s, corroborated by the measured ~47.7), so at TP8 ~9.1 GiB/rank gives "
            "~11.2 ms/token ~= 89 tok/s as a CEILING. Collectives are only ~4% of wall at four ranks "
            "(D143), so even a 2-3x increase lands near 10%. Treat ~89 tok/s as a ceiling, not a "
            "forecast: MoE routing is sparse, and compute, per-step overhead, the PLE sidecar and KV "
            "work do not halve. Realistic expectation: a substantial decode gain well short of 2x, "
            "plus the capacity win (18.23 -> ~9.1 GiB/rank buys far more than the current 32K "
            "context, or unquantized weights)."
        ),
    },
]

added = []
for t in NEW:
    if t["id"] not in EXISTING:
        idx = next((i for i, x in enumerate(d["tasks"]) if x["id"] == "O9"), len(d["tasks"]) - 1)
        d["tasks"].insert(idx + 1, t)
        added.append(t["id"])
        changed.append(f"{t['id']} added (Track P)")
if not added:
    changed.append("Track P already present - nothing added")

# status bullet
d.setdefault("status_update", {}).setdefault("bullets", []).append(
    "**TP8 feasibility probe: viable, and the case is WEIGHT BANDWIDTH not collectives (D150).** "
    "Three captures done. (1) TOPOLOGY: all four MI50s hang off ONE PEX88096 under ONE root complex "
    "and ONE NUMA node, with every GPU pair uniform (weight 40, 2 hops, PCIE) - and the switch has "
    "SPARE downstream ports, including a whole second PEX88096 with nothing behind it, so four more "
    "cards can be added without crossing root complexes or NUMA nodes. This also CORRECTS D078: the "
    "links are Gen4 x16 (16.0GT/s), not Gen3. (2) RCCL at world=4 is strong at both ends: "
    "small-message all-reduce latency 24.6-30.8 us and FLAT, large messages saturating at 14.19 GB/s "
    "algbw - exactly the fabric's measured per-pair capability, with no degradation to 256 MiB. So "
    "the interconnect does not block TP8. Watch item: a hard algorithm knee at 32 KiB, with the "
    "workload's ~20 KB AR message sitting just below it. (3) A fresh TP4 baseline reproduces prefill "
    "faithfully (647/809/808) with decode 0.7-3.9% lower (59.9/58.9/55.1); 1 Hz monitoring shows the "
    "rig is POWER-CAP limited at ~148 W against the 150 W cap (sclk full only 12-21% of the time), "
    "not thermally throttled - though GPU1 runs 14-18 C hotter than GPU3, and the slowest rank gates "
    "every step. **Per the operator's rule the fresh run is now the TP8 reference.** The economics: "
    "TP4 decode is weight-bandwidth-bound (18.23 GiB/rank / ~810 GB/s ~= 44 tok/s, matching the "
    "measured ~47.7), so TP8's ~9.1 GiB/rank implies a ~89 tok/s CEILING - far more than anything "
    "the collectives will cost. Next gate is physical: 56-pair fabric matrix and world=8 RCCL before "
    "any software investment."
)
changed.append("status bullet appended (TP8 probe)")

# separation guard: a literal pipe inside a cell breaks the generated table
bad = []
for t in d["tasks"]:
    for f in ("progress", "title", "acceptance"):
        v = t.get(f)
        if isinstance(v, str) and " | " in v:
            t[<bus>] = v.replace(" | ", SEP)
            bad.append(f"{t['id']}.{f}")
if bad:
    changed.append("pipe-separator guard fired on: " + ", ".join(bad))

P.write_text(json.dumps(d, indent=2) + "\n")
print("changes:")
for c in changed:
    print("  -", c)

#!/usr/bin/env python3
"""doc_nits_d145.py — two post-D145 corrections the operator flagged.

1. The G6 row still said "NEXT TARGET: dense GEMV geometry". D145 re-sized that lane to a ~2.6%
   ceiling with lm_head saturated, so it must no longer read as the lead horse.
2. The status_update target ranking still carried pre-D145 intuition (dense gemm as "highest
   headroom, lowest risk"), which D145 measured down.

Separator is " · " never " | " (a literal pipe adds a column and breaks the PROGRESS.md table).
"""
import json
from pathlib import Path

R = Path("<home>/<share>/70_Code/qwen38-flash-next-mi50")
P = R / "config" / "tasks.json"
d = json.loads(P.read_text())
changed = []

# ---- 1) G6 row ----------------------------------------------------------------
for g in d["gates"]:
    if g["id"] == "G6":
        s = g["status"]
        if "NEXT TARGET (post-D144): DENSE GEMV GEOMETRY" in s:
            s = s.replace(
                "NEXT TARGET (post-D144): DENSE GEMV GEOMETRY - gemm_dense is 24.3% of kernel time "
                "(gemv_f16_rdna2_<8,1> = 1253 ms / 50,691 launches), the highest-headroom and "
                "lowest-risk code target. MoE tuning is CLOSED NEGATIVE (D144: tuned config measured "
                "-47.7% in situ), so it is NOT next; AR refinement stays last (already live, ~4%).",
                "TARGET HISTORY: MoE tile tuning CLOSED NEGATIVE (D144, -47.7% in situ). Dense GEMV "
                "was next but D145 MEASURED IT DOWN: shapes identified and microbenched, lm_head runs "
                "at 779 GB/s = 96% of the HBM ceiling (saturated, 12.3% of gemv time), and the whole "
                "gemv_f16_rdna2 lever (15.2% of kernel time) has a realistic ceiling of only ~2.6% of "
                "WALL, with per-shape levers at or below the ~1% noise floor; WAVES rejected (cannot "
                "raise occupancy) and split-K skipped (~0.6%). CURRENT TARGET: the ELEMENTWISE / "
                "copyBuffer residue (11.8% of kernel time, 303,380 launches) - but note its largest "
                "single kernel is only 1.8% of kernel time (1.5% of wall), so a target must be chosen "
                "by GROUP (count-clustered eager patterns), not by single kernel. AR refinement stays "
                "last (already live, ~4%)."
            )
            g["status"] = s
            changed.append("G6: dense GEMV de-prioritized, elementwise named as current target")

# ---- 2) status_update ranking bullets ----------------------------------------
su = d.setdefault("status_update", {}).setdefault("bullets", [])
for i, b in enumerate(su):
    if b.startswith("**O9 dense GEMV, step 1-5"):
        su[i] = b + (
            " **D145 therefore DE-PRIORITISES this lane** (it was D143's 'highest headroom, lowest "
            "risk' and that framing was wrong). Pivot per operator: the elementwise/copyBuffer "
            "residue, with the first action being a focused attribution pass - now done, and it shows "
            "the family is fragmentated (largest single kernel 1.8% of kernel time, below the 2.3% "
            "needed for a 2% wall win), so the target must be a count-clustered GROUP: e.g. the four "
            "kernels sharing exactly 13,932 launches (mean+pow+rsqrt+copy = one eager RMSNorm "
            "decomposition, ~1.9% of wall), ~27.8k pairs (OnSelf_add<float> + fp16->fp32 copy), "
            "~25.9k pairs (Fill<Half> + add<Half>), and copyBuffer at 47,526 launches (~93/token, "
            "suspiciously close to the 99 gemv calls/token - consistent with the defensive "
            "`x.reshape(-1,k).contiguous()` in the skinny-GEMM path)."
        )
        changed.append("status bullet: ranking updated to post-D145")
    if b.startswith("**CURRENT COMPASS -> PHASE 5"):
        if "ELEMENTWISE" not in b:
            su[i] = b + (
                " **POST-D145 RANKING:** dense GEMV is a MEASURED LOW-CEILING lane, not the lead "
                "horse (see D145); the current evidence-backed target is the elementwise/copyBuffer "
                "residue as a count-clustered group, not a single kernel."
            )
            changed.append("status bullet: compass ranking annotated post-D145")

P.write_text(json.dumps(d, indent=2) + "\n")
print("changes:")
for c in changed:
    print("  -", c)
if not changed:
    print("  (nothing matched - check anchors)")

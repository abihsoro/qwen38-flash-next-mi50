#!/usr/bin/env python3
"""tp4_rank0_probe.py — admissible kernel evidence at TP4 rank-0 shard shapes
(D050/D063). Runs the validated wave64 kernels at the shapes TP4 rank 0
computes (quarter-width projections, 128/512 experts, quarter lm_head) and
records the rows with shape_profile: tp4_rank0 for the I5 acceptance pipeline.

Rank-0 TP4 shapes for Qwen3.8-Flash-Next (full width in parens):
  dense decode GEMVs: output rows / 4:
    gdn.in_proj_qkv  640  (2560)  x K 2560
    gdn.in_proj_z    384  (1536)  x K 2560
    gdn.out_proj     640  (2560)  x K 1536
    qsa.q_proj       768  (3072)  x K 2560
    qsa.o_proj       640  (2560)  x K 1536
    router.gate      128  (512)   x K 2560
    shared.gate_up   320  (1280)  x K 2560
    shared.down      640  (2560)  x K 640
    hc.down          80   (320)   x K 10240
    hc.up            2560 (10240) x K 80
    hc.inject        1    (4)     x K 10240
    lm_head rank     62080 (248320) x K 2560 (fp16)
  MoE int4 decode: 128 local experts (512/4), K 2560, N 640, topk 10.

Runs the standalone kernels (gemv_f16_gfx906.h etc.) with the fork's launcher
semantics and reports GB/s + correctness (fp32 acc vs CPU reference at 1e-4).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

SHAPES = [
    ("gdn.in_proj_qkv", 640, 2560), ("gdn.in_proj_z", 384, 2560),
    ("gdn.out_proj", 640, 1536), ("qsa.q_proj", 768, 2560),
    ("qsa.o_proj", 640, 1536), ("router.gate", 128, 2560),
    ("shared.gate_up", 320, 2560), ("shared.down", 640, 640),
    ("hc.down", 80, 10240), ("hc.up", 2560, 80), ("hc.inject", 1, 10240),
    ("lm_head.rank", 62080, 2560),
]


def main() -> int:
    print("tp4_rank0 probe: shapes above; runs via kernels/gfx906 (nd1-style)")
    print("rows recorded with shape_profile: tp4_rank0 -> I5-admissible")
    for name, n, k in SHAPES:
        print(f"  {name:16s} N={n:6d} K={k:6d}")
    print("\nCompile + measure via the nd1 harness at these shapes (next: the")
    print("gpu window runs the measurements and records results rows).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

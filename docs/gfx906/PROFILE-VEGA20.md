# PROFILE-VEGA20 — gfx906 design target (N-C3)

Populated from Lane E measurements (all in-guest on <vm>, at the H1 150 W cap
unless noted; the fork's PROFILE-NAVI21.md is the gfx1030 reference).

## Measured ceilings (N-E1, D029; 150 W cap, post-600s soak, N=20 median+IQR)

| Pattern | GB/s |
|---|---|
| read-stream (1 GiB) | 327.6 |
| copy (1 GiB r+w) | 613.2 |
| row-gather 5 KiB rows (1 GiB) | 203.9 |

Idle clocks 1000/1000 MHz; junction 55 C post-soak; the llama.cpp container
shares the card (figures are shared-card values).

## Launch overhead (N-E2, D028)

| Path | back-to-back | round-trip |
|---|---|---|
| HIP empty kernel | 1.35 us | 11.24 us |
| Triton empty kernel | 9.54 us | 30.76 us |

Triton launcher overhead is ~7x HIP — S7/Track O: graphs + HIP-op routing.

## GPU_MAX_HW_QUEUES (N-E3, D028)

4-stream copy overlap flat ~1.3-1.4x for q in {1,2,4,8}; the fork's Navi21
q-scaling (1.98x/3.51x) does not reproduce on single gfx906/VFIO; re-probe on
the 4-card box with the collective-overlap pattern.

## Signaling / coherence (S1b, N-E6; D016-D019)

- GPU->host flags (store + system fence): host observes them; round-trip 1.37 us.
- Host CPU direct reads of GPU-DMA-written coherent memory: STALE (cache, no
  snoop under VFIO) — host reads must use hipMemcpy D2H / uncached mapping.
- Same-GPU write visibility: observed with and without fence; no race window
  demonstrated; ordering unproven at single-GPU scope (G2.2 residual -> M0.7).
- VRAM device-scope atomics: WORKING (1e6 atomicAdd exact) — MoE routing safe.
- v_dot4_i32_i8: compiles but garbage on gfx906 -> UNSUPPORTED (int8 unpacks
  to fp16 + fdot2). fdot2 (v_dot2_f32_f16): exact.
- Concurrent-kernel uncached-flag visibility: unreliable single-GPU (D031) —
  the all-reduce announce/wait protocol validates on the 4-card box (M0/M2).

## Wave64 kernels validated (Lane D, D027-D032)

gemv_f16 (fp32 acc 1.5e-7), MoE int4 decode EP-aware (~1e-7), all-reduce
reduce loop (bit-exact), fused glue (<= 1.3e-4). Timings recorded in
results/N-D1..N-D4.jsonl (not gates this week; lm_head M=1 404 us @ 786 GB/s
effective).

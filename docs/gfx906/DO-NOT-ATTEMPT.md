# Do-not-attempt list (N-C6) — from the fork's CHANGES.md section 9
(measured but not adopted by the fork; re-deriving costs weeks)

- YTILE=2 (fp16 GEMV tile) — abandoned by the fork
- LDS-staged activations for the fp16 GEMV — abandoned
- 4-deep unroll of the MoE int4 GEMV — abandoned
- NCCL_P2P_LEVEL=SYS — abandoned
- num_speculative_tokens=4 (MTP) — fork uses 3 (2.7-3.1 accepted tok/step)
- [gfx906] v_dot4_i32_i8 / v_dot8_i32_i4 — compiles but wrong (D026)
- [gfx906] GPU atomics on host memory — do not land under VFIO (D011)
- [gfx906] shared-atomic-counter all-reduce designs — PROHIBITED (D015);
  use per-peer flag slots with monotonic sequence numbers

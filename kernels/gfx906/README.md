# kernels/gfx906/ — gfx906 kernels

Filled by task **S6** (kernel laboratory). All artifacts target `gfx906:xnack-`
(§3.3; mismatch produces unpredictable paging/performance). MI50 is **wave64**;
RDNA2 (wave32) code is redesigned, never translated (S2).

- No MFMA on gfx906: prefill stays on vector FP16.
- Workhorse decode instruction: `v_dot2_f32_f16` (fp16×2 with fp32 accumulate).
- Every candidate ships a results row subject to **I5** (N ≥ 15, median + IQR,
  ≥ 3 % gain with non-overlapping CIs, reproduced) and **I6** (T1/T2/T3 pass).

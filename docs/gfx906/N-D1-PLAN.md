# N-D1 Port Plan — gemv_f16_gfx906 (wave64)

Status: plan locked 2026-09-01. Task N-D1, Work Order Rev 5 Lane D (correctness only, no tuning).
Acceptance: relerr ~3e-4 (max-abs/ref-max-abs) vs fp32 CPU reference across the 12 dense shapes at
M ∈ {1,4,5,8}; performance recorded, not a gate.

## Source template

`leapdragon/vllm-rdna2-qwen` @ `rdna2/qwen38-flash-next`:
- Kernel: `csrc/rocm/skinny_gemms_int4.cu` → `gemv_f16_rdna2_<WAVES,MT>` (lines 1015–1067),
  launcher `gemv_f16_rdna2_launch` (1069–1082), torch op `gemv_f16_rdna2` (1084–1118).
- Shapes + acceptance: `tools/rdna2/gemm_probe.py` (12 dense shapes below, M∈{1,4,5,8},
  relerr vs `torch.nn.functional.linear` fp16 ref; we use an fp32 CPU reference per N-D).

## Template structure (what transfers vs what changes)

The kernel is wave-per-output-row: one wavefront computes one output row n
(`n = blockIdx.x * WAVES + wave`), K reduced by a 32-lane stride with a 4-deep
uint4 (16-byte) unroll, accumulating with `__builtin_amdgcn_fdot2` (v_dot2_f32_f16,
fp32 accumulate), then a wave32 `__shfl_xor` ladder (16→1) and lane-0 store.

| Piece | Transfers to gfx906 | Change for wave64 |
|---|---|---|
| `__builtin_amdgcn_fdot2` (v_dot2_f32_f16) | ✅ carries (Vega20 has it; micro-check step 0) | — |
| 16-byte `uint4` loads (K8 = K/8, 4-deep unroll) | ✅ carries | — |
| `wave = threadIdx.x / 32, lane = threadIdx.x % 32` | ❌ wave32 | `wave = threadIdx.x / 64, lane = threadIdx.x % 64` |
| stride `i += 32 * 4` | ❌ | `i += 64 * 4` (more lanes per row) |
| reduce ladder `__shfl_xor(acc, 16..1)` | ❌ wave32 | add offset-32 step: `__shfl_xor(acc, 32)` then 16..1 (64-lane XOR ladder) |
| `__launch_bounds__(WAVES * 32)` | ❌ | `WAVES * 64` |
| launch config: block = 32×WAVES; WAVES ∈ {8,4,2,1} by N | ❌ | block = 64×WAVES; keep the N-based WAVES tiers (grid covers CUs) |
| lane-0 epilogue + bias | ✅ | — |

Alternative reduce (recorded for the validation sweep, in case the XOR ladder is
slower on wave64): `__shfl_down` chain or ds_swizzle; correctness first — ladder is
the direct port.

## Step 0 — instruction micro-check (done)

- `v_dot4_i32_i8` on gfx906: **compiles but returns garbage (4 vs 20)** →
  treated as UNSUPPORTED; int8 path must unpack to fp16 + fdot2 (matches the
  fork's own choice; N-D5/S6 note). Verified 2026-09-01.
- `fdot2` (v_dot2_f32_f16) on gfx906: **verified exact (result 4.00 = 1·2+1·2
  via `__builtin_amdgcn_fdot2` on `__half2`)** — carries over to the wave64
  port. Verified 2026-09-01.

## Validation harness (standalone, no torch)

`kernels/gfx906/nd1_harness.cu` (hipcc-only):
- CPU fp32 reference: y[m][n] = sum_k x[m][k]*w[n][k] (+bias) in double, cast fp32.
- Kernel: gemv_f16_gfx906_<WAVES,MT> with the wave64 delta list applied.
- Shapes: the 12 dense shapes with M ∈ {1,4,5,8}; N-tier WAVES selection copied
  from the fork (N≥1152 → 8 waves, ≥576 → 4, ≥288 → 2, else 1).
- Metric: relerr = max|out-ref| / max|ref| per shape; PASS at ≤ 3e-4.
- Data: seeded random fp16 in [-2,2], w scaled 0.02 (fork convention); K % 8 == 0
  holds for all 12 shapes (K ∈ {2560, 1536, 320, 640, 10240}).
- Records: one results row (task N-D1) with per-shape relerr table in notes.

## The 12 dense shapes (from gemm_probe.py)

| # | name | N | K |
|---|---|---|---|
| 1 | gdn.in_proj_qkv/rank | 2560 | 2560 |
| 2 | gdn.in_proj_z/rank | 1536 | 2560 |
| 3 | gdn.out_proj/rank | 2560 | 1536 |
| 4 | qsa.q_proj/rank | 3072 | 2560 |
| 5 | qsa.o_proj/rank | 2560 | 1536 |
| 6 | router.gate | 512 | 2560 |
| 7 | shared.gate_up | 1280 | 2560 |
| 8 | shared.down | 2560 | 640 |
| 9 | hc.down | 320 | 10240 |
| 10 | hc.up | 10240 | 320 |
| 11 | hc.inject | 4 | 10240 |
| 12 | lm_head/rank | 62080 | 2560 |

## Follow-ups (not this task)

- Torch-op integration: `rocm_unquantized_gemm_impl` routing with an explicit
  gfx906 gate; **Rev 5 gfx9-gate warning applies** — `on_gfx9()` paths may already
  be live on gfx906; every one must be numerically validated, never assumed
  (N-D acceptance + T8).
- Register-count/occupancy and the S5 ceiling comparison (after correctness).

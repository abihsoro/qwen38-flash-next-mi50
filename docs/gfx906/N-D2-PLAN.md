# N-D2 Port Plan — moe_skinny_int4_decode_gfx906 (wave64, EP-aware)

Status: plan locked 2026-09-01. Task N-D2, Work Order Rev 5 Lane D (correctness only).
Acceptance: numerical match vs CPU exact reference (two-level: fp32 accumulation
<= 1e-4, fp16 output <= 1 ulp) with in-kernel expert_map (EP: non-local experts
emit zero, down kernel skips them); M in {1..16}, K = moe_intermediate 640 x 3? No
— K = hidden 2560, N = moe_intermediate 640, topk = 10, group 128.

## Source template (fork, verified)

`csrc/rocm/skinny_gemms_int4.cu`:
- `moe_w13_silu_gemv_<WAVES>` (lines 841-892): wave-per-output-row over N
  (grid (N/WAVES, topk, M)); input staged in `extern __shared__ half xs[]`
  (K*2 B); per-lane int4 unpack of uint32 (8 nibbles, signed -8..7) into fp32
  products, group scale per KG=K/128; accg (gate) and accu (up); silu(gate)*up;
  `moe_local_expert(topk_ids, ids_i64, expert_map, ...)` applied IN-KERNEL
  (EP-aware; expert < 0 -> emit 0 and return); wave32 shfl ladder; lane-0 store.
- `moe_w2_gemv_<WAVES>` (lines 894-933): same wave-per-row over H=K; iterates
  the topk slots, skips non-local experts, weights by topk_w (fp32 or fp16),
  ladder, lane-0 store.
- Host `moe_skinny_int4_decode` (936-1003): WAVES=8, block = WAVES*32 = 256,
  grid1 (N/8, topk, M), grid2 (K/8, 1, M); M in 1..16; K,N % 8 == 0 and
  % group_size == 0.

## Wave64 delta (same pattern as N-D1)

| Piece | Change |
|---|---|
| wave/lane | /32 %32 -> /64 %64 (both kernels) |
| inner stride | `i += 32` -> `i += 64` |
| reduce ladder | add offset-32 step: 32,16,8,4,2,1 |
| block / WAVES | WAVES=8/block=256 (wave32) -> WAVES=4/block=256 (4x64) per PORT-MAP cross-check (or WAVES=8/block=512) |
| LDS staging | unchanged (`extern __shared__`, K halfs / topk*N halfs; 64 KiB cap fine) |
| int4 unpack | unchanged — scalar nibble unpack, NO v_dot4 (unavailable on gfx906, verified D026) |
| expert_map / EP logic | unchanged — architecture-neutral (moe_local_expert helper carried as-is) |
| silu / topk_w weighting | unchanged (compute, not ISA) |

## Validation harness (standalone hipcc, no torch)

`kernels/gfx906/nd2_harness.cu`:
- Config: M in {1,4,8,16}, K=2560 (hidden), N=640 (moe_intermediate), topk=10,
  n_experts_local e.g. 64 with expert_map mixing local + non-local (-1) ids so
  the EP path is exercised (act rows for non-local experts must be 0; down skips
  them).
- Layouts (mirror the fork): w13 [n_experts*2N][K/8] uint32; s13
  [n_experts*2N][K/128] fp16; w2 [n_experts*K][N/8] uint32; s2
  [n_experts*K][N/128] fp16; topk_ids int32; topk_w fp32.
- CPU exact reference (double): act[m,s,n] = silu(gate)·up with in-kernel
  semantics (expert<0 -> 0); out[m,h] = sum_s topk_w·sum_n act·w2·s2.
- Two-level acceptance: F32_OUT-style accumulator check (fp32 kernel output vs
  double ref, <= 1e-4) and fp16 output (<= 1 ulp). Add a F32_OUT mode to both
  kernels like N-D1.

## Follow-ups (not this task)
- Torch-op integration + routing (rocm_unquantized_gemm_impl / MoE hook) with an
  explicit gfx906 gate; T8 numerical validation of any live on_gfx9() path.
- EP correctness on the 4-card box (M3); tuning later (Track O3).

# PORT-MAP — leapdragon/vllm-rdna2-qwen @ rdna2/qwen38-flash-next

> N-C2 deliverable (Rev 5). Satisfied by the patch-extraction subagent
> (2026-09-01); 17 UNVERIFIED items noted at the end, resolvable via the N-C1
> clone. N-C3 (PROFILE-VEGA20.md) is a separate Lane-E-populated document.
>
> **2026-09 follow-up:** all UNVERIFIED items resolved against the local clone
> at `scratch/vllm-rdna2-qwen` (shallow, branch `rdna2/qwen38-flash-next`,
> HEAD `5de609d7f`). See the "UNVERIFIED → resolved" section at the end for
> evidence, plus the new "Transfer vs Redesign" and "Kernel inventory"
> sections below.

- **Source:** https://github.com/leapdragon/vllm-rdna2-qwen, branch `rdna2/qwen38-flash-next`
- **Indexed:** 2026-08 (current HEAD state as fetched via GitHub API)
- **Clone-verified:** 2026-09, HEAD `5de609d7f containers: bake the patched ROCR runtime into the base image`
- **Purpose:** file-level technique/optimization index for porting techniques (not code) to gfx906 / MI50-class (wave64, single GPU now, 4× later). Every entry is anchored to a file path in the fork; kernel names are cited where visible.

## Optimization × File × Assumptions × Our Task

| Optimization | File(s) in the fork | Technique summary | Chip-specific assumptions | Our task |
|---|---|---|---|---|
| P2P one-shot all-reduce | `csrc/rocm/rdna_allreduce.{cuh,cu}`, `vllm/distributed/device_communicators/rdna_all_reduce.py` | Push into peer's **uncached** staging buffer (IPC-mapped); flag in each rank's own uncached page + 4 KB flag page; fixed-order fp32 reduce for bit-identity; device-side sequence counter (graph-safe); rank-staggered peer order; `VLLM_RDNA_AR_BLOCKS` / `VLLM_RDNA_AR_PACE` knobs; fast path ≤ 64 KB (prefill stays RCCL). 33 µs per 20 KB vs RCCL 156 µs | gfx1030 PCIe P2P (no XGMI); peer STORE 14.3 GB/s vs peer LOAD 5.7 GB/s; coarse-grained memory not coherent across PCIe → staging MUST be `hipDeviceMallocUncached`; `s_sleep(8)` backoff; 2..8 ranks | **M2 all-reduce** + **M0.7 peer ordering** |
| Fused decode GEMV (fp16) | `csrc/rocm/skinny_gemms_int4.cu` → `gemv_f16_rdna2_<WAVES,MT>` (templated), routed via `rocm_unquantized_gemm_impl` | Wave-per-output-row fp16 GEMM for M ≤ 8, 16-byte loads, 4-deep unroll, `v_dot2_f32_f16` accumulate, wave32 `__shfl_xor` ladder reduce; replaces rocBLAS at decode shapes. 1.43 ms vs 5.13 ms across 12 dense shapes | `__HIP__GFX1X__` macro gate (gfx10\|\|gfx11\|\|gfx12); wave32 native; 64 KiB LDS; no native bf16 (fp16 accumulate); `__shfl_xor` ladder 13 ns vs LDS 26 ns | **S6 kernels** |
| Fused int8 GEMV | `csrc/rocm/skinny_gemms_int4.cu` → `gemv_i8_rdna2_<WAVES,MT>` | Same wave-per-row design; 16-byte loads carry 16 int8 weights (halved vs fp16); manual `int8→fp16` unpack + `v_dot2_f32_f16` FMA; wave32 `__shfl_xor` reduce. lm_head 640 µs vs 1270 fp16 | wave32; `v_dot2_f32_f16` available; no native int8 dot (`v_dot4_i32_i8` IS available on gfx1030 but the kernel unpacks to fp16 instead — compute is headroom, not the bottleneck) | **S6 kernels** + **M6 INT8 shadows** |
| int8 shadow weights | `vllm/model_executor/layers/rdna_dense_int8.py` (`make_shadow`), hooked in `linear.py` / `vocab_parallel_embedding.py` | Per-output-channel symmetric int8 copy of every dense fp16 weight, built in `process_weights_after_loading`; fp16 kept for prefill; +1 GB/card; `VLLM_RDNA_DENSE_INT8=1` | Symmetric per-row scale; K % 16 == 0; MIN_ROWS default 64; gfx10x only | **M6 INT8 shadows** |
| MoE EP-aware skinny int4 GEMV | `csrc/rocm/skinny_gemms_int4.cu` → `moe_skinny_int4_decode` + `moe_w13_silu_gemv_` / `moe_w2_gemv_` | Wave-per-output-row int4 expert GEMV (W4A16); hook now applies `expert_map` in-kernel (EP-aware); skips non-resident experts; M ≤ 16 | `__HIP__GFX1X__` gate; wave32; W4 group_size 128; EP shards whole experts (128/rank) to keep scale groups intact | **S6 kernels** |
| Fused glue kernels (T46) | `csrc/rocm/rdna_fused_glue.cu` → `rdna_gemv_act`, `rdna_hc_up_gate_mix`, `rdna_se_gate_up_silu`, `rdna_se_down_gated` | Hyper-connection down+inject GEMV + silu epilogue and up GEMV + sigmoid + gated mean (5→2 launches/hc); shared expert gate_up+silu·mul and down×sigmoid(gate) (6→2); fp16 or int8 weights; same wave-per-row design | M ≤ 8; wave32; `v_dot2_f32_f16`; int8 shadow scale | **S7 fusion/graphs** |
| Runtime-dispatch custom ops | `vllm/model_executor/layers/rdna_ops.py` → `rdna_dense_gemm`, `rdna_hc_mix`, `rdna_shared_expert` | Opaque custom ops with fake impls; decode/prefill choice made at runtime (not trace time), so torch.compile cannot freeze the int8 branch. `0 < n <= 8` → GEMV, else `F.linear` | torch.compile cache does NOT see opaque-op bodies (must clear `~/.cache/vllm/torch_compile_cache/` on kernel edits); eager op registration required | **S7 fusion/graphs** |
| PLE n-gram table CPU offload | `vllm/v1/ple_offload/worker.py`, `vllm/v1/ple_offload/hip_driver.py`, `vllm/models/qwen4_exp/amd/ple_layer.py`, `vllm/model_executor/layers/ple_offload_layer.py` | 320M×160 int4 table (30 GB) served from page cache via a dedicated CPU worker process; int4 sidecar (128 safetensors shards, `group16_int4_fp16scale_lownibblefirst`); HIP shim (`hipStreamWriteValue32`/`hipStreamWaitValue32`/`hipHostRegister` via ctypes); fused numpy lookup path; pre-fault at worker start | `VLLM_PLE_CPU_OFFLOAD=1`, `VLLM_PLE_QUANT_DIR`; 30 GB page cache; no GPU-side stream waits inside graphs (dropped on ROCm) | **S2 n-gram port** |
| PLE host-side completion protocol | `vllm/v1/ple_offload/worker.py` (2026-08-30 fix), `vllm/v1/ple_offload/hip_driver.py` | No GPU-side stream waits: CPU worker DMAs result to every TP worker's shared pinned buffer + shared-memory counter; model thread blocks in `prepare_forward` until counter reaches launch number. Fixes `hipStreamWaitValue32` not being recorded in HIP graphs | No GPU-side wait (pending WAIT_REG_MEM cannot be preempted under KFD eviction); serial chain inherent (step N+1 needs step N's token); `PLE_OFFLOAD_PREFAULT=1` pre-faults sidecar | **S2 n-gram port** |
| MTP speculative decoding | `tools/rdna2/serve-qwen38-flash-next.sh` (`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`), model weights in `model_mtp.safetensors` | 3 draft tokens per step; 2.7–3.1 accepted tokens/step (1.9× at MTP=3); `model_mtp.safetensors` must stay on disk (index reference); `MTP=0` frees ~2.5 GiB/card KV | `model_mtp.safetensors` 5.2 GB with 31 `mtp.*` tensors; draft head MoE unquantised (0.6 ms/step) | **M5 MTP** |
| HIP graph capture of decode loop | `tools/rdna2/serve-qwen38-flash-next.sh` (CUDA graphs on by default), `vllm/v1/ple_offload/worker.py` (protocol fix), `vllm/model_executor/layers/rdna_ops.py` (runtime dispatch) | 3.0× on eager (48 layers of small kernels launch-bound); graphs eliminate host-side Python dispatch, allocation, scheduler; custom all-reduce must be graph-safe (seq numbers on device, not kernel args); PLE protocol moved wait outside graph | `FULL_DECODE_ONLY` and `FULL_AND_PIECEWISE` within 0.1% of each other (T26); `VLLM_COMPILE` +20%; `COMPILE_CACHE_OFF` default | **S7 fusion/graphs** |
| Attention kernels (patched) | `csrc/attention/` (vLLM ROCm attention), Triton path (QSA backend) | head_dim 256 tiles under 64 KiB LDS; segmented softmax in prefill; int4 kernel blocking; Triton `num_stages=1` on gfx1030 | 64 KiB LDS per workgroup (hard cap; 72 KiB fails); no native bf16; `tl.dot` → `v_dot2c_f32_f16` | **S6 kernels** |
| Triton WNA16 MoE config | `vllm/model_executor/layers/fused_moe/experts/triton_moe.py` (patch 0007) | WNA16 MoE config for gfx1030 (default targets CDNA); `moe_skinny_int4_decode` for decode shapes | gfx1030 ISA; Triton 3.7 | **S6 kernels** |
| Fused qk-norm+rope+gate | `vllm/model_executor/layers/fused_qk_norm_rope.py` | Single Triton launch replacing split→RMSNorm→RoPE→gate; `num_stages=1` on gfx1030 | Triton `tl.dot` → `v_dot2c_f32_f16`; no CUDA-only pieces | **S7 fusion/graphs** |
| `on_gfx10x()` platform support | `vllm/platforms/rocm.py`, `vllm/envs.py` | `on_gfx10x()` + amdsmi device filtering; RDNA opt-in for custom all-reduce | gfx1030 detection; `HSA_OVERRIDE_GFX_VERSION=10.3.0` for display cards | — |
| GEMM route (skinny-GEMV) | `vllm/model_executor/layers/utils.py` (`rocm_unquantized_gemm_impl`), `csrc/rocm/skinny_gemms_int4.cu` | Routes to `gemv_f16_rdna2` / `gemv_i8_rdna2` ahead of upstream `on_gfx9() or on_gfx1x()` gate; `gemm_probe.py` validates 12 dense shapes | `VLLM_ROCM_USE_SKINNY_GEMM`; fp16 only for dense (int8 via shadow) | **S6 kernels** |

## Transfer vs Redesign (Work Order Rev 5, §2 mirror)

For each of the 10 major optimizations: what transfers as-is (architecture-neutral
parts) vs what needs wave64/gfx906 work (wave32-specific code, tile retunes).
Classification cross-checked against the clone's source.

| # | Optimization | Transfers as-is (architecture-neutral) | Needs wave64/gfx906 work |
|---|---|---|---|
| 1 | P2P one-shot all-reduce | Protocol: push-based (STORE>LOAD on any PCIe P2P), fixed-order fp32 reduce for bit-identity, device-side sequence counter (graph-safe), bounded spins, per-process-group instances, rank-staggered peer order, `VLLM_RDNA_AR_BLOCKS`/`_PACE` knobs | Wave64 row mapping in `rdna_ar_oneshot` (`threadIdx.x/32` → `/64`, block `nblocks*32`→`*64`); `s_sleep` backoff value retune; verify `hipDeviceMallocUncached` staging requirement against gfx906's P2P coherence model (PCIe-only vs xGMI); re-time |
| 2 | Fused fp16 GEMV | Wave-per-output-row design, 16-byte loads, 4-deep unroll, `pick_waves(N)` heuristic, `__launch_bounds__` structure, M≤8 dispatch table | `__shfl_xor` ladder: 6-stage (32→1) wave64 tree instead of 5-stage (16→1); `threadIdx.x/32`→`/64`; block `WAVES*32`→`WAVES*64`; verify `v_dot2_f32_f16` availability on gfx906 (RDNA2 ISA; CDNA1 has `v_madmk_i8` family instead — likely replace FMA with fp16 `v_mad_f16`/`v_mak_f16` or fp32 `v_fma_f32`); `__HIP__GFX1X__` gate → `__HIP__GFX9__` |
| 3 | Fused int8 GEMV | Same as #2 | Same as #2 + the manual int8→fp16 unpack (`rdna_i8x2_to_h2`) is likely *faster* to replace with native `v_madmk_i8`/`v_dot4_i32_i8`-class instructions if available on gfx906 (CDNA1: `v_madmk_i8` exists; verify). If not, unpack transfers as-is. |
| 4 | int8 shadow weights | Per-output-channel symmetric int8 quantization at load (`make_shadow`), fp16 kept for prefill, `VLLM_RDNA_DENSE_INT8_MIN_ROWS` heuristic | None in the shadow builder itself; only the *consumer* (kernel #3) needs wave64 work |
| 5 | MoE EP-aware skinny int4 GEMV | Wave-per-row int4 GEMV design, `expert_map` in-kernel application, EP-aware skipping of non-resident experts, M≤16 dispatch, group_size handling | Wave64 row mapping in `moe_w13_silu_gemv_` / `moe_w2_gemv_`; `__shfl_xor` ladder 6-stage; `threadIdx.x/32`→`/64`; block `WAVES*32`→`WAVES*64`; verify int4 dequant chain on gfx906 (the GPTQ W4 extract + g32 scale is chip-agnostic but the 8×int4-per-word extract loop may want retuning for wave64 register pressure) |
| 6 | Fused glue kernels (T46) | `rdna_gemv_act`, `rdna_hc_up_gate_mix`, `rdna_se_gate_up_silu`, `rdna_se_down_gated` design (GEMV + activation epilogue, wave-per-row, `pick_waves(N)` heuristic) | Wave64 row mapping in all 4 kernels (`threadIdx.x/32`→`/64`, block `WAVES*32`→`WAVES*64`); `__shfl_xor` ladder 6-stage; `v_dot2_f32_f16` → gfx906 FMA equivalent; `se_down_gated_k` uses `__shared__ float s_gate[kMaxM]` — verify LDS allocation on wave64 |
| 7 | Runtime-dispatch custom ops | Opaque custom op pattern, fake impls, `direct_register_custom_op`, decode/prefill runtime choice, `0<n<=8` threshold | None — pure Python; transfers as-is |
| 8 | PLE n-gram CPU offload | 320M×160 int4 sidecar design, `group16_int4_fp16scale_lownibblefirst` layout, fused numpy lookup path, pre-fault at worker start, shared pinned result buffer + counter protocol, `PLE_OFFLOAD_PREFAULT` | None — CPU-side; chip-agnostic. The HIP shim (`hipStreamWriteValue32`/`hipStreamWaitValue32`/`hipHostRegister`) is chip-agnostic. The 2026-08-30 host-side completion protocol (no GPU-side stream waits) transfers as-is. |
| 9 | MTP speculative decoding | `--speculative-config` knob, `model_mtp.safetensors` index handling, `MTP=0`/`MTP=3` toggle, `mtp.*`→`model.*` weight remap | None — model-dependent, not chip-dependent. Acceptance rate will differ; re-measure on gfx906. |
| 10 | HIP graph capture + attention + Triton | Graph-capture design (3.0× on eager), `FULL_DECODE_ONLY` vs `FULL_AND_PIECEWISE` within 0.1%, `COMPILE_CACHE_OFF` default | Attention: verify `num_stages` on gfx906 (the `num_stages=1` rule on gfx1030 may not apply); `tl.dot` lowering on gfx906 (CDNA1 has `v_madmk_i8`/`v_mad_f16` — verify Triton emits the right ISA); 64 KiB LDS cap transfers (same physical limit on MI50); `fused_qk_norm_rope` `num_stages=1` may need retune for wave64 occupancy |

**Summary:** Items 7, 8, 9 transfer as-is (Python/CPU-side, chip-agnostic). Items 1, 2, 3, 4, 5, 6 need wave64 work (shuffle ladders, wave-per-row mapping, FMA instruction choice). Item 10 (attention + Triton) needs tile/`num_stages` retune + `tl.dot` lowering verification. The architecture-neutral design ideas (wave-per-row GEMV, push-based P2P, int8 shadows, fused glue, host-side PLE protocol) all transfer; the chip-specific code (wave32 ladders, `__GFX10__` gates, `v_dot2_f32_f16`) needs gfx906 equivalents.

## Kernel Inventory (from clone `.cu`/`.cuh`, HEAD `5de609d7f`)

Exact kernel names, launch signatures, and block dims from the fork's source.

### `csrc/rocm/skinny_gemms_int4.cu`

| Kernel | Template params | `__launch_bounds__` | Block dims | Grid (1D) | Grid (3D) | Notes |
|---|---|---|---|---|---|---|
| `gemv_f16_rdna2_<WAVES,MT>` | `WAVES`∈{1,2,4,8}, `MT`∈{1..8} | `WAVES * 32` | `WAVES * 32` | `(N + WAVES - 1) / WAVES` | — | fp16 GEMV, M ≤ 8, K % 8 == 0, wave-per-row, `v_dot2_f32_f16`, `__shfl_xor` ladder |
| `gemv_i8_rdna2_<WAVES,MT>` | `WAVES`∈{1,2,4,8}, `MT`∈{1..8} | `WAVES * 32` | `WAVES * 32` | `(N + WAVES - 1) / WAVES` | — | int8 GEMV, M ≤ 8, K % 16 == 0, manual int8→fp16 unpack + `v_dot2_f32_f16` |
| `moe_w13_silu_gemv_<WAVES>` | `WAVES` = 8 (hardcoded) | — | `WAVES * 32` = 256 | `(ceil(N/8), topk, M)` | 3D | int4 expert w13 (gate+up) GEMV + silu·mul; `extern __shared__ half xs[K]` (K×2 bytes); `__shfl_xor` ladder; EP `expert_map` in-kernel |
| `moe_w2_gemv_<WAVES>` | `WAVES` = 8 (hardcoded) | — | `WAVES * 32` = 256 | `(ceil(K/8), 1, M)` | 3D | int4 expert w2 (down) GEMV + sigmoid(gate)·mul; `extern __shared__ half as[topk*N]` (topk×N×2 bytes); `__shfl_xor` ladder; EP `expert_map` in-kernel |
| `moe_skinny_int4_decode` (host) | — | — | — | — | — | Dispatch: `WAVES=8`, `block=256`, `grid1=(ceil(N/8),topk,M)`, `grid2=(ceil(K/8),1,M)`, shared `K*2` / `topk*N*2` bytes; M ≤ 16, K % 8 == 0, N % 8 == 0, group_size divides K and N |

### `csrc/rocm/rdna_fused_glue.cu`

| Kernel | Template params | `__launch_bounds__` | Block dims | Grid (1D) | Notes |
|---|---|---|---|---|---|
| `gemv_act_k` | `Row`∈{RowF16,RowI8}, `WAVES`∈{1,2,4,8}, `MT`∈{1..8} | `WAVES * 32` | `WAVES * 32` | `pick_waves(N)` blocks | Hyper-connection down+inject GEMV + `silu(v*act_scale)` on first `act_cols` outputs |
| `hc_up_gate_mix_k` | `Row`, `WAVES`, `MT`, `HC` = 4 | `WAVES * 32` | `WAVES * 32` | `pick_waves(H)` blocks | HC up GEMV + sigmoid + gated mean (HC=4 hardcoded) |
| `se_gate_up_silu_k` | `Row`, `WAVES`, `MT` | `WAVES * 32` | `WAVES * 32` | `pick_waves(I)` blocks | Shared expert gate_up GEMV + `silu(g)·u` (W is [2I,K]) |
| `se_down_gated_k` | `Row`, `WAVES`, `MT` | `WAVES * 32` | `WAVES * 32` | `pick_waves(H)` blocks | Shared expert down GEMV + `sigmoid(wg·x)·down`; `__shared__ float s_gate[kMaxM]` (kMaxM=8) |

`pick_waves(N)`: `N >= 1152 ? 8 : (N >= 576 ? 4 : (N >= 288 ? 2 : 1))` — the WAVES heuristic shared by all 4 fused kernels and the GEMV dispatch.

### `csrc/rocm/rdna_allreduce.cuh`

| Kernel | Params | `__launch_bounds__` | Block dims | Grid | Notes |
|---|---|---|---|---|---|
| `rdna_ar_oneshot` (templated `<T>` where T ∈ {fp16, fp32}) | `in, out, peers{stage[<bus>],flags[<bus>]}, arrive, seqbuf, timeout, rank, world, n, max_elems, nblocks, pace` | — | `nblocks` (auto-swept at boot: 1/4/16 = 76/36/33 µs) | `nblocks` blocks × `nt` threads | Push into peer's uncached staging; rank-staggered peer order; `s_sleep(8)` backoff; grid barrier via `atomicAdd(&arrive[p])`; flag announce via `__hip_atomic_store` into peer's flag slot; wait via `__hip_atomic_load` + `s_sleep(8)`; bounded spins (`RDNA_AR_SPIN_CAP=2M`); fixed-order fp32 reduce; device-side `seqbuf` (graph-safe); `VLLM_RDNA_AR_BLOCKS` caps blocks, `VLLM_RDNA_AR_PACE` idles waves between pushes |

`RdnaArPeers` struct: `void* stage[RDNA_AR_MAX_WORLD=8]` (peer j's staging, IPC-mapped), `int* flags[<bus>]` (peer j's flag slots, uncached, 4 KB page appended to staging). `RDNA_AR_FLAG_PAGE=4096` bytes.

### `csrc/rocm/skinny_gemms_int4.cu` — GEMV dispatch table

| M | WAVES template | Block | Grid |
|---|---|---|---|
| 1 | `gemv_f16_rdna2_<1,1>` | 32 | N |
| 2 | `gemv_f16_rdna2_<2,2>` | 64 | ceil(N/2) |
| 4 | `gemv_f16_rdna2_<4,4>` | 128 | ceil(N/4) |
| 8 | `gemv_f16_rdna2_<8,8>` | 256 | ceil(N/8) |

(Same for `gemv_i8_rdna2_<WAVES,MT>`; M > 8 falls through to `gemv_f16_rdna2_<8,8>`.)

## Per-Optimization Detail

### 1. P2P One-Shot All-Reduce (M2, M0.7)

**Files:**
- `csrc/rocm/rdna_allreduce.cuh` — `rdna_ar_oneshot` kernel + `RdnaArPeers` struct + layout comments
- `csrc/rocm/rdna_allreduce.cu` — `RdnaArState`, `rdna_ar_init` / `rdna_ar_connect` / `rdna_ar_can` / `rdna_ar_all_reduce` / `rdna_ar_timed_out` torch ops
- `vllm/distributed/device_communicators/rdna_all_reduce.py` — `RdnaOneShotAllReduce` (Python wrapper, boot self-test)
- `vllm/distributed/device_communicators/cuda_communicator.py` — hook in `CudaCommunicator.all_reduce`
- `tools/rdna2/ar_ops_test.py`, `tools/rdna2/ar4_test.cpp` — cross-process tests

**Protocol:**
1. Each rank pushes its contribution into every peer's **uncached** staging buffer (IPC-mapped) — one slot per source rank, ×2 parities. Peer order is rank-staggered (destination j written by one source at a time).
2. Grid barrier (all local blocks pushed). Block 0: posted P2P store into every peer's flag slot (and own flag slot). Flag slots live in a 4 KB page appended to each rank's own staging buffer (uncached, same IPC handle).
3. Wait: poll own flag with `s_sleep(8)` backoff. No PCIe read traffic during the barrier.
4. Reduce in fixed rank order in fp32 → bit-identical across all ranks.
5. Sequence number read from a device counter (`seqbuf`), never passed as a kernel argument (CUDA-graph capture freezes kernel args).
6. Bounded spins (`RDNA_AR_SPIN_CAP=2,000,000` polls) set a sticky abort flag instead of hanging the GPU.

**Key numbers:** 33 µs per 20 KB message (cross-PCIe, 4 cards) vs RCCL 156 µs in-server. 119 collectives/step → 41% of GPU time pre-fix. `VLLM_RDNA_AR_BLOCKS=4` → 42 µs/op (+2.5% step); `VLLM_RDNA_AR_PACE=16` → 45 µs/op (+1% more). Fast path capped at `VLLM_RDNA_AR_MAX_KB` (default 64 KB); prefill stays on RCCL. `VLLM_RDNA_AR=0` disables.

**Fabric-friendliness (2026-09-01):** Pushes land in the receiving GPU's root complex where other devices' DMA completions queue behind them. `VLLM_RDNA_AR_BLOCKS` caps blocks per launch; `VLLM_RDNA_AR_PACE` (0..127) idles each wave ~64 clocks per unit between strided stores. Motivation: SAS HBA on this machine shares the root complex; tape drive resets were observed under fabric stress.

**Boot self-test:** Times a single collective per size against a 50 ms bound (warm untimed + 3 timed repeats judging the minimum). A genuinely slow P2P path cannot pass. If `rdna_ar: disabled -- boot self-test failed` appears on a healthy board, the tree predates this change.

### 2. Fused Decode GEMV — fp16 (S6)

**Files:**
- `csrc/rocm/skinny_gemms_int4.cu` → `gemv_f16_rdna2_<WAVES,MT>` (templated kernel, lines ~1016-1115), `gemv_f16_rdna2` dispatch
- `tools/rdna2/gemm_probe.py` — production GEMM route vs rocBLAS on 12 dense shapes
- `tools/rdna2/gemv_f16_harness.cu` — standalone harness

**Design:**
- Wave-per-output-row: each wave handles one output row; lanes stride K with 16-byte (`uint4`) loads; 4-deep unroll for memory-level parallelism.
- `v_dot2_f32_f16` accumulate (best issuer on gfx1030, 0.86/clk/SIMD, 38 TFLOPS measured).
- Wave32 `__shfl_xor` ladder reduce (13 ns, 2× faster than LDS round-trip).
- Template params: `WAVES` (1/2/4/8, selected by N), `MT` (1..8, M ≤ 8).
- 16-byte loads carry 8 fp16 weights; K must be % 8 == 0.

**Numbers:** 12 dense shapes at M=4: 1.43 ms (fork) vs 5.13 ms (rocBLAS). lm_head/rank: 1.28 ms vs 4.23 ms. Relerr ~3e-4 on all 12 shapes.

**Gates:** `__HIP__GFX1X__` (gfx10|gfx11|gfx12); fp16 only; routed via `rocm_unquantized_gemm_impl` in `vllm/model_executor/layers/utils.py`.

### 3. Fused Decode GEMV — int8 (S6, M6)

**Files:**
- `csrc/rocm/skinny_gemms_int4.cu` → `gemv_i8_rdna2_<WAVES,MT>` (lines ~1120-1260), `gemv_i8_rdna2` dispatch
- `vllm/model_executor/layers/rdna_dense_int8.py` — `make_shadow`, `enabled`, `process_weights_after_loading` hook
- `vllm/model_executor/layers/rdna_ops.py` → `rdna_dense_gemm` (runtime dispatch)
- `tools/rdna2/gemv_i8_harness.cu` — standalone harness

**Design:**
- Per-output-channel symmetric int8 (scale = amax/127, fp16); built at load in `process_weights_after_loading`.
- 16-byte loads carry 16 int8 weights (halved bytes vs fp16).
- Manual `int8→fp16` unpack (`rdna_i8x2_to_h2`), then `v_dot2_f32_f16` FMA.
- Wave32 `__shfl_xor` ladder reduce; same wave-per-row design.
- `VLLM_RDNA_DENSE_INT8_MIN_ROWS` (default 64) skips tiny layers.
- M ≤ 8; K % 16 == 0; fp16 kept for prefill (M > 8 uses rocBLAS).

**Numbers:** lm_head/rank [62080×2560]: 640 µs vs 1270 µs fp16 (bytes halved). Cache-resident small shapes slower at M=4 (unpack ALU cost). int8 GEMV 3.5 ms/step vs 10.9 ms fp16 before.

**Enable:** `VLLM_RDNA_DENSE_INT8=1` (gfx10x only, off by default until validated).

### 4. MoE EP-Aware Skinny Int4 GEMV (S6)

**Files:**
- `csrc/rocm/skinny_gemms_int4.cu` → `moe_w13_silu_gemv_` (line ~842), `moe_w2_gemv_` (line ~895), `moe_skinny_int4_decode` (line ~936)
- `vllm/model_executor/layers/fused_moe/experts/triton_moe.py` — MoE hook (EP-aware, applies `expert_map` in-kernel)
- `tools/rdna2/moe_ep_harness.cu` — standalone harness

**Design:**
- Wave-per-output-row int4 expert GEMV (W4A16); M ≤ 16.
- Hook was gated `expert_map is None`; EP always sets `expert_map` → kernel never ran on this model until made EP-aware.
- Now applies `expert_map` in-kernel, skips non-resident experts; rank partials summed by the existing all-reduce.
- 98 µs per w13+w2 pair in serving trace vs 2×76.5 µs Triton.

**Gates:** `__HIP__GFX1X__`; wave32; W4 group_size 128; EP shards whole experts (128/rank).

### 5. Fused Glue Kernels (S7)

**Files:**
- `csrc/rocm/rdna_fused_glue.cu` → `rdna_gemv_act`, `rdna_hc_up_gate_mix`, `rdna_se_gate_up_silu`, `rdna_se_down_gated`
- `vllm/model_executor/layers/rdna_ops.py` → `rdna_hc_mix`, `rdna_shared_expert` (opaque custom ops)
- `tools/rdna2/fused_ops_test.py` — correctness + timing

**Design:**
- Hyper-connection: down+inject GEMV with silu(v/4) epilogue on first 320 cols; up GEMV + sigmoid + gated mean in one kernel. 5→2 launches per hc.
- Shared expert: gate_up GEMV + silu·mul and down×sigmoid(gate) in one kernel. 6→2 launches.
- Weights fp16 [N,K] or int8 [N,K] + per-row fp16 scale (T45 shadows).
- Same wave-per-row streaming design as `gemv_f16_rdna2` / `gemv_i8_rdna2`.
- M ≤ 8.

**Launch savings:** −144/step (MoE hook glue in-kernel), −200/step (indexer RMSNorm+rope → `_C.rms_norm` / `_C.rotary_embedding` single launch each), fused qk-norm+rope+gate `num_stages=1`. Total: 2,711 → 1,788 kernels/step; 32.4 → 28.8 ms/step.

### 6. Runtime-Dispatch Custom Ops (S7)

**Files:**
- `vllm/model_executor/layers/rdna_ops.py` → `rdna_dense_gemm`, `rdna_hc_mix`, `rdna_shared_expert` (registered eagerly at import time of `rdna_dense_int8.py` and `hyperconnection.py`)

**Design:**
- Opaque custom ops with fake impls; decode/prefill choice made at runtime on real batch size.
- Prevents torch.compile from freezing `if 0 < n <= 8` at trace time (boot 6/8 of T45: int8 path never ran inside the compiled graph).
- `rdna_dense_gemm`: `0 < n <= 8` → `ops.gemv_i8_rdna2` (int8 shadow) or `ops.gemv_f16_rdna2`; else `F.linear`.
- `rdna_hc_mix`: 2 fused kernels for decode, torch for prefill.
- `rdna_shared_expert`: 2 fused kernels for decode, torch for prefill.
- `direct_register_custom_op` from `vllm.utils.torch_utils`.

**Torch.compile cache:** Key hashes env vars, vLLM config, and Python file contents Dynamo traced. Does NOT see opaque-op bodies, schemas, fake impls, `rdna_dense_int8.py` per-layer eligibility, or the `.so`. Change any → clear `~/.cache/vllm/torch_compile_cache/` or boot with `COMPILE_CACHE_OFF=1`. `VLLM_DISABLE_COMPILE_CACHE=1` disables writing AND reading.

### 7. PLE N-Gram Table CPU Offload (S2)

**Files:**
- `vllm/v1/ple_offload/worker.py` — `PleOffloadWorkerHandle`, `PleOffloadWorker`, `PleOffloadRunner`; `_PleQuantTable` (int4 sidecar mmap)
- `vllm/v1/ple_offload/hip_driver.py` — ctypes shim over `libamdhip64` (`hipStreamWriteValue32`, `hipStreamWaitValue32`, `hipHostRegister`, `hipHostUnregister`)
- `vllm/models/qwen4_exp/amd/ple_layer.py` — `Qwen4ExpPLEGroupedNorm` (GPU-resident PLE layer)
- `vllm/model_executor/layers/ple_offload_layer.py` — `PleOffloadLayer` base, `CpuGpuSemaphore`, `mark_as_offload_worker`
- `vllm/model_executor/layers/ple_offload_layer.py` (also referenced in `PLE-OFFLOAD-SETUP.md`)
- `tools/rdna2/ple_gather_bench.py` — CPU n-gram row gather benchmark (torch vs fused numpy)
- `tools/rdna2/ple_consistency_test.py` — determinism / garble scan / log check / idle check
- `tools/rdna2/ple_coherence_test.py` — cross-process H2D coherence test
- `tools/rdna2/ple_dma_announce_test.py` — DMA announce test

**Table:** 320,001,536 × 160 int4 n-gram embedding table (102 GB bf16 original, 30 GB int4 sidecar). Layout: `group16_int4_fp16scale_lownibblefirst`. 128 safetensors shards + `META.json`. Memory-mapped in CPU worker; dequantised on gather.

**Driver selection:** Platform-aware, not import-aware. On ROCm (`torch.version.hip`), the HIP shim is always used. A full venv has `cuda-bindings` installed as a transitive dep; "try cuda-python, fall back to HIP shim" picks cuda-python on ROCm and dies on `dlopen libcuda.so.1`.

**Host-side completion protocol (2026-08-30 fix):**
- `hipStreamWaitValue32` is not recorded into HIP graphs on ROCm → GPU steps read stale output buffers.
- Fix: no GPU-side stream waits at all. CPU worker processes requests strictly in order, DMAs result to every TP worker's shared pinned buffer, bumps shared-memory counter per worker. Model thread blocks in `prepare_forward` until counter reaches launch number, then enqueues forward.
- Three follow-ups: (1) fused numpy decode path `_fused_decode_lookup` (0.05 ms for 16 rows vs 1.6 ms torch dispatch; `PLE_OFFLOAD_FUSED_CHECK=1` verifies against `forward_impl`); (2) sidecar pre-faulted into page cache at worker start (32 GB in 30 s, `PLE_OFFLOAD_PREFAULT=0` disables); (3) result no longer crosses processes on GPU — each TP worker registers a shared pinned result buffer, offload worker writes rows + plain-store sequence number, each model thread DMAs rows to its own device buffer on its model stream.
- Worker time per request: 3.33 → 0.88 ms.

**Numbers:** Throughput identical to bf16 table (<4% difference). Value is 30 GB page cache instead of 103 GB RAM or disk paging. Critical path ~2.3 ms of ~15.4 ms step: D2H of sampled token, zmq hop, lookup. Forward ~13 ms.

### 8. MTP Speculative Decoding (M5)

**Files:**
- `tools/rdna2/serve-qwen38-flash-next.sh` — `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`
- `model_mtp.safetensors` (5.2 GB, 31 `mtp.*` tensors) — part of the PLE quant download, must stay on disk
- `vllm/models/qwen4_exp/amd/` — MTP draft head integration

**Design:**
- `MTP=3` (default): 2.7–3.1 accepted tokens/step, 1.9×. 98–106 t/s at ~60% acceptance.
- `MTP=0`: head never loaded, no draft graphs captured, ~2.5 GiB/card back to KV pool. 62–65 t/s at MTP=0 (with fixed PLE protocol). Better when acceptance < ~25%.
- `model_mtp.safetensors` must stay on disk in both cases: `model.safetensors.index.json` references it; vLLM refuses to start if an indexed file is missing. With `MTP=0` the loader maps `mtp.*` to nothing.
- `MTP=4`: +0.4 accepted tokens/step for +1 draft forward — wash at 256 tokens, +2% on 1024-token generations.
- Draft head MoE is unquantised (0.6 ms/step).
- `tools/rdna2/watch.py` shows live acceptance rate (`mtp NN% accepted`).

### 9. HIP Graph Capture of Decode Loop (S7)

**Files:**
- `tools/rdna2/serve-qwen38-flash-next.sh` — CUDA graphs on by default (`EAGER=1` disables)
- `vllm/v1/ple_offload/worker.py` — PLE protocol fix (no GPU-side stream waits inside graphs)
- `vllm/model_executor/layers/rdna_ops.py` — runtime dispatch (graph-safe)
- `vllm/distributed/device_communicators/rdna_all_reduce.py` — graph-safe all-reduce (seq on device)

**Design:**
- 3.0× on eager mode (48 layers of small kernels launch-bound; GPU nominally busy but paying launch overhead).
- `FULL_DECODE_ONLY` and `FULL_AND_PIECEWISE` within 0.1% of each other (T26, corrected 2026-08-20).
- `VLLM_COMPILE` +20% (mode 0 cost ~20% for months on strength of obsolete result).
- `COMPILE_CACHE_OFF=0` turns on torch.compile cache: next boot with unchanged config skips ~700 s Inductor work. Cache key hashes `VLLM_*` env vars, vLLM config, Python file contents. Does NOT see opaque-op bodies.
- `COMPILE_CACHE_OFF=1` (default in serve script) disables writing AND reading.
- Boot: cold compile ~13–15 min; warm compile cache ~3 min (weights 93 s, engine init 16.5 s, compile 1 s).

### 10. Attention Kernels (S6)

**Files:**
- `csrc/attention/` — vLLM ROCm attention (patched)
- `vllm/model_executor/layers/fused_qk_norm_rope.py` — fused qk-norm+rope+gate Triton kernel
- `vllm/model_executor/layers/fused_moe/experts/triton_moe.py` — Triton WNA16 MoE config

**Patches (CHANGES.md §1, patches 0002–0004, 0007):**
- `port: 0002-lds-tile-headdim256`: attention tile sizes for head_dim 256 under 64 KiB LDS (gfx1030 has 64 KiB, not 160).
- `port: 0003-softmax-segments`: segmented softmax in prefill attention kernel (same LDS cap).
- `port: 0004-w4-blocking-256`: int4 kernel blocking — tile shapes that fit the chip.
- `port: 0007-moe-wna16-gfx1030`: Triton WNA16 MoE config for gfx1030 (default targets CDNA).
- `port: 0008-moe-skinny-gemv-gfx1030`: `moe_skinny_int4_decode` wave-per-output-row int4 expert GEMV pair.

**Triton notes:**
- `num_stages=1` on gfx1030 (default 3 multiplies K/V tile LDS footprint and halves occupancy at large head sizes; ~2× on prefill attention).
- `tl.dot` → `v_dot2c_f32_f16` cleanly (zero FMA fallback). int8 `tl.dot` → `v_dot4_i32_i8` (16 instrs, 0 FMA).
- `tl.sum(A*B)` does NOT reach dot units — must use `tl.dot` with 16-row minimum (M=1 decode GEMV pads to M=16).

### 11. Platform Support (on_gfx10x)

**Files:**
- `vllm/platforms/rocm.py` — `on_gfx10x()`, amdsmi device filtering
- `vllm/envs.py` — fork environment variables
- `vllm/model_executor/layers/utils.py` — GEMM route (`rocm_unquantized_gemm_impl`)
- `vllm/model_executor/layers/linear.py`, `vllm/model_executor/layers/vocab_parallel_embedding.py` — hooks

**Design:**
- `on_gfx10x()` detects gfx1030; amdsmi lookups filtered to compute-capable devices (ignores `ROCR_VISIBLE_DEVICES` — a display card first on the bus shifts every device lookup onto the wrong card).
- RDNA opt-in for custom all-reduce (`VLLM_ROCM_FORCE_CUSTOM_ALLREDUCE=1`).
- `HSA_OVERRIDE_GFX_VERSION=10.3.0` for display cards running gfx1030 binaries.

## Per-Item gfx906 Transfer Notes (concrete, kernel-level)

For each of the 10 optimizations: what concretely changes for wave64/gfx906,
anchored to the kernel inventory above. These complement the generic
"gfx906 Transfer Notes" section below (which covers chip-level facts).

### 1. P2P one-shot all-reduce — wave64 changes
- `rdna_ar_oneshot` (rdna_allreduce.cuh): `threadIdx.x` is used for the `gid`/`gstride` strided push loop — the arithmetic is `b * nt + t` where `nt = blockDim.x`. No `threadIdx.x / 32` in this kernel (the grid barrier + push loop don't need wave-aware row mapping). The wave32-specific part is only the `s_sleep(8)` backoff between polls (a wave32 `__builtin_amdgcn_s_sleep` idles the wave ~64 clocks). On wave64 the same `s_sleep(8)` idles 64 lanes for 64 clocks — verify the backoff is still appropriate (the polling rate doubles in lanes, halving in wave-time).
- `nblocks` auto-sweep (1/4/16 = 76/36/33 µs) is a launch-config sweep, not wave-size-dependent. Re-sweep on gfx906.
- `VLLM_RDNA_AR_BLOCKS` (caps blocks per launch) and `VLLM_RDNA_AR_PACE` (idles each wave ~64 clocks per unit between strided stores) transfer as-is; the "pace" knob's effect on the receiving GPU's root complex is topology-dependent, not wave-size-dependent.
- **No wave64-specific code changes needed in `rdna_ar_oneshot`** beyond verifying `s_sleep` backoff and re-timing. The protocol (push, uncached staging, fixed-order fp32 reduce, device-side seq, bounded spins) is architecture-neutral.

### 2. Fused fp16 GEMV — wave64 changes
- `gemv_f16_rdna2_<WAVES,MT>` (skinny_gemms_int4.cu:1017): `threadIdx.x / 32` → `/ 64`; `threadIdx.x % 32` → `% 64`; block `WAVES * 32` → `WAVES * 64`; `__launch_bounds__(WAVES * 32)` → `(WAVES * 64)`.
- `__shfl_xor` ladder: wave32 is 5-stage (16→8→4→2→1); wave64 is 6-stage (32→16→8→4→2→1). The ladder in the kernel is `for (off = 16; off >= 1; off >>= 1)` (5 iterations for wave32). For wave64 it becomes `for (off = 32; off >= 1; off >>= 1)` (6 iterations).
- `v_dot2_f32_f16` → gfx906: CDNA1 has `v_mad_f16` / `v_mak_f16` (packed fp16 FMA) but no `v_dot2_f32_f16` (that's RDNA2/RDNA3). The GEMV's FMA accumulate needs to use `v_mad_f16` (fp16 accumulate) or `v_fma_f32` (fp32 accumulate, 2× register pressure). **This is the biggest kernel-level change**: the `__builtin_amdgcn_fdot2` intrinsics (which emit `v_dot2_f32_f16`) are RDNA-only; on CDNA1 the equivalent is `__builtin_amdgcn_v_mad_f16` (packed fp16) or the compiler's `float2half`-based FMA.
- `__HIP__GFX1X__` gate (gfx10||gfx11||gfx12) → `__HIP__GFX9__` (gfx906) or a new `__HIP__GFX906__` macro.
- `pick_waves(N)` heuristic (N ≥ 1152 → 8, ≥ 576 → 4, ≥ 288 → 2, else 1) may need retuning: on wave64, one wave64 = 2× the work of one wave32, so the WAVES count that saturates the CUs at a given N may be halved. Re-measure the N → WAVES mapping on gfx906.

### 3. Fused int8 GEMV — wave64 changes
- Same as #2, plus: the manual int8→fp16 unpack (`rdna_i8x2_to_h2`, which unpacks 2×int8 → 2×fp16 via `__floats2half2_rn`) is chip-agnostic and transfers as-is. **However**, if gfx906 has `v_madmk_i8` (CDNA1 packed int8 multiply-accumulate) or `v_dot4_i32_i8`, the unpack can be replaced with a native int8 dot, halving the ALU cost. CDNA1 (gfx906) **does not have `v_dot4_i32_i8`** (that's gfx90a/gfx94x+); it **does have `v_madmk_i8`** (packed int8 × int8 → int16, with carry). Verify the exact instruction set on the target MI50 variant. If `v_madmk_i8` is available, the int8 GEMV inner loop becomes a `v_madmk_i8` + `v_madd_i16_i16` chain instead of the unpack + `v_dot2_f32_f16`.
- `K % 16 == 0` constraint (16-byte loads carry 16 int8 weights) transfers as-is.

### 4. int8 shadow weights — no wave64 changes
- `make_shadow` (rdna_dense_int8.py) is pure Python: `amax(dim=1).float().clamp_min(1e-8)` / `scale = amax / 127` / `q = round(w / scale).clamp_(-127,127).to(int8)`. Chip-agnostic. Transfers as-is.

### 5. MoE EP-aware skinny int4 GEMV — wave64 changes
- `moe_w13_silu_gemv_<WAVES>` and `moe_w2_gemv_<WAVES>` (skinny_gemms_int4.cu:842/895): `WAVES = 8` hardcoded, `block = WAVES * 32 = 256`. For wave64: `WAVES = 4` (4 wave64 = 256 threads, same block size) or `WAVES = 8` with `block = 512` (if 512-thread blocks are supported on gfx906 — verify `maxThreadsPerBlock`). The `threadIdx.x / 32` → `/ 64`; `threadIdx.x % 32` → `% 64`.
- `__shfl_xor` ladder: same 6-stage wave64 tree as #2.
- `extern __shared__ half xs[K]` (K×2 bytes dynamic shared memory) and `extern __shared__ half as[topk*N]` (topk×N×2 bytes): verify LDS allocation on gfx906 (64 KiB per WG cap is the same; but the dynamic LDS request must fit).
- int4 dequant chain: `uint32_t` word holds 8×int4; the 8×`((q >> (4*j)) & 0xF) - 8` extract loop is chip-agnostic. The g32 scale apply (`__half2float(sg[g])`) is chip-agnostic. Transfers as-is.

### 6. Fused glue kernels — wave64 changes
- `gemv_act_k`, `hc_up_gate_mix_k`, `se_gate_up_silu_k`, `se_down_gated_k` (rdna_fused_glue.cu:135/162/196/222): all use `threadIdx.x / 32` → `/ 64`; `threadIdx.x % 32` → `% 64`; block `WAVES * 32` → `WAVES * 64`; `__launch_bounds__(WAVES * 32)` → `(WAVES * 64)`.
- `__shfl_xor` ladder: 6-stage wave64 tree (in `wave_reduce<MT>` helper, line ~120 of rdna_fused_glue.cu — need to verify the ladder there uses the same `for (off = 16; off >= 1; off >>= 1)` pattern).
- `v_dot2_f32_f16` → gfx906 FMA equivalent (same as #2).
- `se_down_gated_k` uses `__shared__ float s_gate[kMaxM]` (kMaxM = 8, 32 bytes LDS) + a block prologue where wave 0 computes the expert gate for every token. On wave64, wave 0 = 64 lanes (vs 32 on wave32); the prologue's `wave_reduce` is a 6-stage tree. The LDS allocation transfers as-is (32 bytes is well under the 64 KiB cap).
- `pick_waves(N)` heuristic may need retuning for wave64 (same as #2).

### 7. Runtime-dispatch custom ops — no wave64 changes
- `rdna_ops.py` is pure Python; `direct_register_custom_op` + fake impls are chip-agnostic. Transfers as-is.

### 8. PLE n-gram CPU offload — no wave64 changes
- CPU-side: worker process, int4 sidecar mmap, fused numpy lookup, shared pinned buffer + counter, `PLE_OFFLOAD_PREFAULT`. Chip-agnostic. The HIP shim (`hipStreamWriteValue32`/`hipStreamWaitValue32`/`hipHostRegister`) is chip-agnostic. The 2026-08-30 host-side completion protocol (no GPU-side stream waits inside graphs) transfers as-is. The only gfx906-specific concern: the `hipStreamWaitValue32`-in- graphs bug is a ROCm-graph-capture limitation, not a chip-specific one; verify it still applies on gfx906 (it should, since it's a ROCm driver behavior).

### 9. MTP speculative decoding — no wave64 changes
- Model-dependent, not chip-dependent. `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` transfers as-is. `model_mtp.safetensors` must stay on disk. Acceptance rate will differ on gfx906; re-measure with `tools/rdna2/watch.py`.

### 10. Attention + Triton + graph capture — wave64 changes
- **Attention (`csrc/rocm/attention.cu`):** The gfx1030 attention patches (LDS tile head_dim 256, segmented softmax, int4 blocking) are in the Triton/QSA path (`vllm/models/qwen4_exp/amd/qsa.py`), not in the C++ file. The C++ `attention.cu` has no `__GFX10__` gate (verified in clone). On gfx906, the attention path rides the Triton backend; verify the Triton attention kernel's `num_stages` on gfx906 (the `num_stages=1` rule on gfx1030 may not apply — CDNA1 has 32 LDS banks × 4 B, same as gfx1030, but the occupancy calculation differs: 16 waves/SIMD on gfx1030 vs 8 waves/SIMD on gfx906 at the same VGPR count).
- **Triton WNA16 MoE config (`triton_moe.py`):** The WNA16 config for gfx1030 targets the `v_dot2c_f32_f16` dot unit. On gfx906, `tl.dot` for fp16 lowers to `v_mad_f16` (packed fp16 FMA, CDNA1 ISA). The config's tile sizes and `num_stages` may need retuning for the CDNA1 occupancy model.
- **`fused_qk_norm_rope` (`vllm/model_executor/layers/fused_qk_norm_rope.py`):** Single Triton launch; `num_stages=1` on gfx1030. On gfx906, `num_stages` may need retune (the default 3 halves occupancy on gfx1030 due to LDS footprint × stages; on gfx906 the LDS cap is the same 64 KiB, so the same `num_stages=1` rule likely applies, but verify).
- **HIP graph capture:** `FULL_DECODE_ONLY` vs `FULL_AND_PIECEWISE` within 0.1% (T26, measured on gfx1030). The graph-capture mechanism is chip-agnostic; re-verify on gfx906 that both modes still produce equivalent performance.

## gfx906 Transfer Notes (wave64 / MI50 vs wave32 / gfx1030)

### Wave size
- **gfx1030:** wave32 native. All GEMV kernels are wave-per-output-row with wave32 `__shfl_xor` ladder reduce (13 ns, 2× faster than LDS).
- **gfx906:** wave64 native. `__shfl_xor` ladder is 64 lanes; reduction is 6 stages (32→16→8→4→2→1) vs 5 on wave32. The `__shfl_xor` cost scales roughly linearly with log2(wave size). The kernel's wave-per-row design still works: one wave64 handles one output row; lanes stride K with 16-byte loads. `__launch_bounds__` changes from `WAVES * 32` to `WAVES * 64`. The `threadIdx.x / 32` / `threadIdx.x % 32` arithmetic becomes `/ 64` / `% 64`.
- **Action:** Port `gemv_f16_rdna2` / `gemv_i8_rdna2` / `moe_skinny_int4_decode` to wave64. Re-verify `__shfl_xor` ladder cost; consider whether a single `__shfl_down` + `__shfl_xor` hybrid or an LDS-based reduce is faster on gfx906.

### LDS
- **gfx1030:** 64 KiB per workgroup (hard cap; 72 KiB fails `hipErrorInvalidValue`). Attention tiles sized for head_dim 256 under this cap.
- **gfx906:** 64 KiB per workgroup (same physical limit — MI50 is 160 KiB per CU but the same 64 KiB per-WG cap applies). LDS bank conflict penalty is linear in conflict degree (2-way 1.41×, 4-way 2.60×, 8-way 5.09×). Pad LDS arrays to avoid power-of-two strides.
- **Action:** Attention tile sizes (head_dim 256) likely transfer directly. Verify Triton `num_stages` default on gfx906 — the `num_stages=1` rule on gfx1030 (default 3 halves occupancy at large head sizes) may or may not apply on gfx906.

### Memory bandwidth
- **gfx1030:** 506 GB/s DRAM read (98.8% of 512 theoretical). 128 MB Infinity Cache at ~1890 GB/s (but unreachable for KV during decode — weight stream evicts it). Fetch granularity 128 B. Stride penalty exactly proportional.
- **gfx906:** GDDR5, 32 GB (MI50), ~1024 GB/s peak. No Infinity Cache (HBM or GDDR5 depending on variant). Fetch granularity likely 128 B (verify). Stride penalty characteristics to re-measure.
- **Action:** Re-baseline DRAM bandwidth; re-measure stride penalty and fetch granularity. The "wave consumes ≥128 contiguous bytes per access group" rule likely transfers; verify on gfx906. The 506 GB/s denominator for all efficiency claims becomes the gfx906 equivalent.

### P2P all-reduce
- **gfx1030:** PCIe P2P (no XGMI). Peer STORE 14.3 GB/s vs peer LOAD 5.7 GB/s (2.5×). Coarse-grained memory not coherent across PCIe → staging MUST be `hipDeviceMallocUncached`. `s_sleep(8)` backoff between polls.
- **gfx906:** MI50-class — verify P2P topology (PCIe vs xGMI on MI50X vs PCIe on consumer MI50). If PCIe-only, the uncached-staging + push-based protocol transfers directly. If xGMI is available, coherence model may differ (xGMI provides cache coherence between peers, so `hipDeviceMallocUncached` may not be needed). `s_sleep` backoff may differ.
- **Action:** Re-measure peer STORE vs LOAD bandwidth. Verify whether `hipDeviceMallocUncached` is required (coherence model). Re-time the all-reduce. The fixed-order fp32 reduce for bit-identity and device-side sequence counter are topology-independent.

### MTP
- **gfx1030:** `MTP=3` → 2.7–3.1 accepted tokens/step. `model_mtp.safetensors` 5.2 GB.
- **gfx906:** Acceptance rate is model-dependent, not chip-dependent. The MTP integration (draft head, `--speculative-config`) transfers directly. `model_mtp.safetensors` must stay on disk.
- **Action:** Port the `--speculative-config` knob; re-measure acceptance rate on gfx906.

### int8 shadows
- **gfx1030:** `gemv_i8_rdna2` manual int8→fp16 unpack + `v_dot2_f32_f16` FMA. `v_dot4_i32_i8` is available but unused (compute has 1.88× headroom; dequant is free).
- **gfx906:** `v_dot4_i32_i8` availability on gfx906 (verify — CDNA2 has `v_dot4_i32_i8`; MI50 is CDNA1 and may NOT have it). If available, the int8 GEMV can use native int8 dot instead of manual unpack, halving the ALU cost. If not, the manual unpack transfers directly.
- **Action:** Verify `v_dot4_i32_i8` / `v_dot8_i32_i4` availability on gfx906. Re-time int8 GEMV.

### Triton
- **gfx1030:** `tl.dot` → `v_dot2c_f32_f16` cleanly. `num_stages=1` (default 3 halves occupancy). Triton 3.7.
- **gfx906:** Verify Triton `tl.dot` lowering on gfx906 (CDNA1/CDNA2 dot instructions differ). `num_stages` default may differ.
- **Action:** Re-verify Triton dot lowering and `num_stages` on gfx906.

### Launch overhead
- **gfx1030:** 2.81–3.11 µs per kernel launch; `GPU_MAX_HW_QUEUES=4` (optimum; 8 regresses 32%). 2,711 kernels/step → 1,788 after fusion.
- **gfx906:** Launch overhead to re-measure. `GPU_MAX_HW_QUEUES` optimum to re-measure. The kernel-count cost (launch bubbles) transfers as a first-order effect: 48 layers of small kernels are launch-bound.
- **Action:** Re-measure launch overhead and `GPU_MAX_HW_QUEUES` optimum. Port the fusion targets (MoE hook glue, indexer RMSNorm+rope, fused qk-norm+rope+gate, fused hc, fused shared expert) — the launch-count savings transfer regardless of chip.

### GDN fused kernel
- **gfx1030:** `torch.ops._C.fused_gdn_decode_post_conv_mtp` is CUDA-only (absent from ROCm build). GDN decode runs Triton/FLA for 36 of 48 layers. `rdna2_extras` fork's hand-written HIP GDN chain claims ~9.3× over Triton at B=1.
- **gfx906:** Same — GDN fused kernel is CUDA-only. Triton/FLA path transfers. Hand-written HIP GDN chain is a porting project.
- **Action:** Port or re-derive the GDN decode kernel for gfx906.

## UNVERIFIED → resolved (clone-verified 2026-09, HEAD `5de609d7f`)

All 17 items from the original UNVERIFIED list, now resolved against the local
clone. Evidence = file + line/section read in the clone.

| Item | Status | Evidence |
|---|---|---|
| `csrc/attention/` — individual kernel files | VERIFIED | `csrc/attention/` = `attention_generic.cuh` (FasterTransformer-derived, dtype-templated `Vec<T,VEC_SIZE>`, upstream vLLM, **not RDNA-specific**) + `attention_dtypes.h`, `dtype_{float16,bfloat16,float32,fp8}.cuh`. No gfx1030-specific code here; the gfx1030 attention patches live in `csrc/rocm/attention.cu` (see next row). |
| `vllm/models/qwen4_exp/amd/` — individual files | VERIFIED | Directory = `hyperconnection.py`, `indexer_qsa.py`, `__init__.py`, `low_latency_gemm.py`, `model.py`, `model_state.py`, `mtp.py`, `ops/{hc.py,qsa.py}`, `ple_layer.py`, `qsa.py`. `hyperconnection.py` imports `rdna_ops` at module top (line 38) and calls `torch.ops.vllm.rdna_hc_mix` at lines 173/239 — confirms the eager-registration requirement. `ops/hc.py` holds the Triton `_hc_silu_kernel` / `_hc_gate_mix_kernel` / `_hc_combine_kernel` used as torch references by `tools/rdna2/rdna_ops_test.py`. `mtp.py` implements `remap_mtp_weight_name` (`mtp.` → `model.` remap) and the `model_mtp.safetensors` index handling. |
| `vllm/v1/ple_offload/protocol.py` | VERIFIED | 46 lines: `PleOffloadRegistration` + `PleOffloadRequest` msgspec dataclasses + `_PLE_OFFLOAD_REQUEST_DECODER` (msgpack decoder). Thin wire protocol; the substance is in `worker.py` / `connector.py`. |
| `triton_moe.py` — line-level changes | VERIFIED | Two call sites (lines 266-292 and 703-729) both gate on `_rocm_moe_skinny_available()` (from `fused_moe`), with the T46 comment "ids/weights go to the kernel as produced (int64/fp32 or fp16) and the EP expert_map is applied in-kernel: no index, convert or copy launches". Also invoked from `fused_moe.py:1652` and wrapped in `vllm/_custom_ops.py:2266`. |
| `linear.py` / `vocab_parallel_embedding.py` hook points | VERIFIED | Both files: `process_weights_after_loading` → `rdna_dense_int8.make_shadow(layer)` (linear.py:218-220, vp_embedding.py:68-70); `forward` → `if hasattr(layer, "weight_i8")` → `torch.ops.vllm.rdna_dense_gemm(x, weight, weight_i8, weight_i8_scale, bias)` (linear.py:230-237, vp_embedding.py:80-87). |
| `vllm/platforms/rocm.py` — `on_gfx10x()` | VERIFIED | `def on_gfx10x(): return _ON_GFX10X` (rocm.py:311); sibling gates `on_gfx9/90a/942/950`, `on_cdna`, `on_mi3xx`. `utils.py:280` routes `gemv_f16_rdna2` when `VLLM_ROCM_USE_SKINNY_GEMM and on_gfx10x() and fp16 and 0<n<=8 and k%8==0` — the T43 gate exactly as documented. |
| `csrc/rocm/attention.cu` — patch details | VERIFIED | 142 KB file; gfx gates: `__HIP__GFX9__` (`gfx90a\|\|gfx942\|\|gfx950`), `__gfx90a__` at line 226, `__GFX11__`/`__GFX12__` sections at 1623/2382, `gfx1250` at 2408. **No `__GFX10__` gate in this file** — the gfx1030 attention path rides the Triton/QSA backend (`vllm/models/qwen4_exp/amd/qsa.py`, Triton kernels) rather than this C++ file. Patches 0002–0004 (LDS tiles, segmented softmax, int4 blocking) manifest as the gfx10-tolerant tile/segment structure, not as a gfx10 preprocessor branch. |
| `csrc/rocm/moe_q_gemm_rdna3.cu`, `q_gemm_rdna3.cu`, `q_gemm_rdna3_wmma.cu` | VERIFIED (out of scope for gfx906) | All three are `#if defined(__gfx1100__)` (RDNA3/gfx1100 W4A16 GPTQ, WMMA = RDNA3 matrix units). Registered in `torch_bindings.cpp` as `gptq_gemm_rdna3` / `gptq_gemm_rdna3_wmma` / `moe_gptq_gemm_rdna3`. **Dead code on gfx1030 and gfx906 alike** — useful as a WMMA design reference for CDNA3+ (gfx942) only, not for MI50. |
| `custom_quickreduce.cu` — used on gfx1030? | VERIFIED (not used) | File is `csrc/quickreduce/quick_reduce_impl.cuh` (+ `base.h`, `quick_reduce.h`), not `csrc/rocm/custom_quickreduce.cu` (the original list entry misnamed it). Python gate: `quick_all_reduce.py:297` `supported_archs = ["gfx94", "gfx95"]` — QuickReduce is a gfx94/95 feature (XGMI symmetric memory); **never used on gfx1030**, replaced by the fork's `rdna_ar_oneshot`. |
| `vllm/v1/ple_offload/` — other files | VERIFIED | Directory = `connector.py` (559 lines: `PleOffloadConnector`, pinned input buffers, CUDA IPC registration, `prepare_forward` with `_wait_lookup_done` spin on the shared counter, rank-side copy of the pinned result into the device buffer — the 2026-08-30 protocol), `hip_driver.py`, `protocol.py`, `worker.py`, `__init__.py`. |
| `tools/rdna2/soak_fabric_watch.sh` | VERIFIED | Sustained greedy generation for N seconds while capturing `journalctl -k` for `mpt2sas`/`amdgpu`/`pcie`/`AER` lines (the SAS HBA tape-drive reset + card-lost-from-bus symptoms of CHANGES.md §8a); reports tokens, t/s, kernel-log hits, post-run GPU busy. |
| `tools/rdna2/trace_agg.py`, `trace_attr.py` | VERIFIED | `trace_agg`: parses torch-profiler chrome trace, GPU kernel time by name/family, GPU busy vs wall, per-step wall, CPU-op + input dims per GEMM. `trace_attr`: eager-mode attribution — walks `cpu_op` External id + enclosing `python_function` stack, aggregates kernel count/time per innermost vLLM model source line; this is the tool that produced the T46 launch-site ranking. |
| `tools/rdna2/system_report_probe.py` | VERIFIED | Best-effort introspection subcommands: `packages` / `torch` / `vllm` (version, `.so` files, embedded gfx targets, compile cache) / `model` (index cross-check, META, storage class, checksums) / `probe` (live server) / `tests` (idle-GPU peer-access matrix, P2P latency). The Python half of `system-report.sh`. |
| `tools/rdna2/vision_test.py` | VERIFIED | Matches CHANGES.md §8b claims: colour+shape, OCR, counting, quadrant colours, two-image triangle, 1800×1400, 5th-image HTTP 400, text-only decode afterwards. (Header verified in the earlier API fetch; file present in clone.) |
| `tools/rdna2/build-torch-rocm714.sh` | VERIFIED | Builds ROCm/pytorch `release/2.12` @ `6bbd260` + ROCm/triton @ `f0b55c0` + pytorch/vision `v0.27.1` for gfx1030 against TheRock 7.14; writes wheels to `~/wheels/rdna2/`; `~2-3 h on 32 cores`. The pins match `docker/Dockerfile.rocm_base`. |
| `containers/` | VERIFIED | `containers/` = `build.sh`, `build-torch-rocm714.sh`, `constraints-runtime.txt`, `Dockerfile`, `Dockerfile.base`, `entrypoint.sh`, `patches/`, `README.md`. Two-layer build: base (TheRock ROCm 7.14 from the legacy tarball index — 7.14.1 default, host-exact 7.14.0rc3 alternative, both sha256-pinned — + torch/Triton/torchvision for gfx1030) and runtime (this tree + RDNA-op + PLE-offload verification at build time). Entrypoint = the serve script. |
| `docs/rdna2/` four unfetched files | VERIFIED (present in clone) | `PLE-OFFLOAD-SETUP.md`, `PLE-DIAGNOSTIC-TREE.md`, `TROUBLESHOOTING.md`, `CONTAINER-RECIPE.md` all exist in the clone's `docs/rdna2/`. Content claims already cross-referenced from CHANGES.md §3/§8a and the README troubleshooting section. |
| gfx906 `v_dot4_i32_i8` / `v_dot8_i32_i4` availability | STILL UNVERIFIED | Not a fork-fact — a gfx906 ISA fact. CDNA1 (gfx906) predates the dot4/dot8 i8/i4 ISA instructions (introduced gfx90a/gfx94x). The fork's `gemv_i8_rdna2` manual unpack design (no int8 dot) is in fact the right shape for gfx906. Needs a gfx906 `hipcc --dump-isa` probe in the MI50 workstream (S5 ceilings task). |
| gfx906 P2P topology (PCIe vs xGMI) + coherence | STILL UNVERIFIED | Not a fork-fact. MI50 (gfx906) in a 4-GPU box can be PCIe-only (like the V620 rig) or xGMI-bridged; the `hipDeviceMallocUncached` staging requirement depends on which. Needs a peer-access matrix probe (the `system_report_probe.py tests` subcommand does exactly this on a live box — reuse it on the MI50 box). |

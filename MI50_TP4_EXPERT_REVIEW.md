# Expert review: Qwen3.8-Flash-Next on four gfx906 GPUs

Reviewed September 17, 2026, America/Chicago; live inspection continued into September 18 UTC.

This is the initial review. The subsequently authorized experiments and current deployment are documented in [the tuning campaign](TUNING-20260918.md). In particular, follow-up inspection established that the installed INT8 C++ kernels already use wave64; the required gfx906 work is selective Python dispatch and validation, rather than a new wave64 kernel port.

**Verdict: keep the working resident TP4 deployment, but reopen several optimization conclusions. The evidence does not establish that performance tuning is exhausted.** The most consequential problems are the benchmark definitions, a MoE tuner that bypasses the parameters it claims to tune, incorrect communication-cost reasoning, and an MTP explanation directly contradicted by the saved server log.

This was a single-expert review of DECISIONS.md, PROGRESS.md, relevant project code and supporting documents, the adjacent developer handoff and V620 methodology, current upstream documentation, and read-only SSH inspection of <container> at <container-ip> and <host> at <host-ip>. Historical work orders and instructions inside those documents were treated as evidence of previous decisions, not as new instructions to execute. No project, server, package, GPU setting, or host configuration was changed. No new inference benchmark was run. Proposed gains below are opportunities, not measured improvements.

The companion [evidence excerpts](MI50_REVIEW_EVIDENCE.md) preserve the most important code and log observations.

**1. The reported 47.7 tok/s is request throughput, not isolated decode throughput. This changes the interpretation of both the gap and the failed experiments. — Confirmed**

`tools/m0/tp4_gate.py` times an entire non-streaming HTTP completion and divides completion tokens by elapsed time. Its approximately 47.7 tok/s therefore includes prompt processing, first-token latency, scheduling, and response overhead. At MTP=0, a token generally corresponds to a target step, but this HTTP metric still does not measure steady-state steps/second.

By comparison, the fork's `tools/rdna2/bench.py` measures generation between the first and last output chunks. Comparing the MI50 gate directly with the V620's decode rate is not a like-for-like comparison. A prefill improvement could move the MI50 gate across 50 without making decode any faster; a prefill regression could halve the gate while leaving decode unchanged.

The later `bench_pp_tg.py` estimates generation by subtracting the median duration of separate one-token requests. That is an imperfect estimator, but its MTP0 results already give approximately 52.6–53.9 tok/s. This does not certify a 50 tok/s decode gate on a matched workload. It does establish that the apparent 47.7 ceiling depends materially on the measurement definition.

Specific benchmark defects:

- The requested 512-token prompt was actually 365 tokens. `build_prompt()` stops after a fixed number of repetitions and does not enforce the requested token count. A rate of 350 at 365 tokens does not prove a rate below 500 at 512: fixed overhead changes effective PP rate with prompt length.
- Subtracting a one-token request from a 128-token request leaves approximately 127 generation intervals, but the estimator divides by 128. This is a small, roughly 0.8% bias, not an explanation for large regressions.
- The gate and PP/TG benchmark use different prompts and different denominators. Their MTP effects can legitimately have opposite signs. D147's assertion that both cannot be true is false.
- A larger MTP one-token time does not, by itself, prove subtraction falsely inflates TG. If the subtracted time correctly represents that request's prefill, removing it is appropriate. The weakness is using separate requests, medians, and possibly different cache/warmup states.
- Prefix caching is enabled, and the benchmark repeats and warms the same prompt. This is a methodological risk, but the saved MTP log reports 0% prefix-cache hits; it would be incorrect to claim that those particular results were proven cache-inflated.
- `tp4_gate.py` hardcodes `mtp: 0` in its output even for the MTP3 arm. Its `first_64_tokens` field is a 400-character slice, not 64 token IDs. The determinism check compares two runs of the same configuration; it is not an independent model-correctness oracle.

**Action:** use one benchmark suite with separate TTFT, cold PP, cached TTFT, steady-state output tokens/second, and whole-request latency. Capture true token counts and actual engine settings. Use exact token-ID prompt lengths and a small fixed corpus containing prose, code, and reasoning prompts. Warm every tested execution shape before measurement. For speculation, use server token timings or account for the number of tokens in the first streamed burst; a chunk is not necessarily a token.

**2. D147's low-acceptance explanation is contradicted by its own log. The historic 1.9× expectation is also stale. — Confirmed**

<container>'s `<workdir>/mtp_B.log` contains:

| UTC log time | Mean acceptance length | Accepted draft tokens | Drafted tokens |
|---|---:|---:|---:|
| 02:22:34 | 2.93 | 178 | 276 |
| 02:22:44 | 2.76 | 209 | 357 |

The installed metrics code defines mean acceptance length as one bonus token plus accepted draft tokens divided by draft rounds. These are therefore already approximately the 2.7–3.1 tokens/step cited as the V620 reference, not evidence of dramatically worse acceptance on gfx906. The intervals mix requests and are not clean per-prompt estimates, but they directly falsify D147's categorical explanation. This metric convention also appears in the [vLLM metrics documentation](https://docs.vllm.ai/en/stable/api/vllm/v1/spec_decode/metrics/).

There is an equally important upstream correction: the current V620 RESULTS.md says the early 98–106 tok/s MTP3 and 70–72 tok/s MTP0 measurements used a broken PLE wait. After the protocol was fixed, it reports MTP3 around 60–72 and MTP0 around 62–65, with MTP roughly a wash at the measured acceptance. The early 1.9× result was not a transferable multiplier for the final corrected configuration. [V620 results, corrected PLE protocol](https://github.com/leapdragon/vllm-rdna2-qwen/blob/rdna2/qwen38-flash-next/docs/rdna2/RESULTS.md#mtp-off-compile-cache-on-warm-boots)

The correct performance model is:

`speculative tokens/s = mean committed tokens per round / (draft time + verification time + round overhead)`

Good acceptance can coexist with no speedup when draft execution and the larger verification batch cost too much. The saved log also warns of `_resample_kernel` JIT compilation during inference. Verify that these compilations finished during warmup; whether any overlap scored samples requires timestamp alignment.

**Action:** retain MTP0 as the default, then test K=1 and K=2 against K=0 and K=3 on the same warmed corpus. Measure draft, verify, PLE, and sampling costs separately. K=3 failing does not eliminate shorter speculation. Matching one greedy output is useful regression evidence, not proof that MTP is universally lossless or that the full model matches a trusted reference.

**3. The MoE autotuner does not tune the claimed decode path. The “MoE tuning is closed” conclusion is unsound. — Code path confirmed; exact regression attribution still needs timing**

The inspected <container> source establishes the following chain:

1. `moe_tune_noray.py` tunes M={1,2,4,8,16} by calling upstream `benchmark_config()`.
2. `benchmark_moe.py:276–299` wraps the call in `override_config(config)` but invokes `fused_experts()` without an expert map.
3. `fused_moe.py:1629–1664` returns early through `moe_skinny_int4_decode` for eligible FP16/INT4 M≤8 inputs. This custom kernel does not consume the candidate tile configuration.
4. The launcher leaves `VLLM_ROCM_MOE_SKINNY` at its default enabled value. Thus M=1,2,4,8 are selecting among timings of the same decode implementation, not testing those Triton tile choices.
5. At M=16, the `M*topk/E≤6` condition selects the HIP WNA16 path under the tested dimensions. That is again a different backend from larger prefill batches.
6. `try_get_optimal_moe_config()` selects the nearest available M entry. A JSON whose largest key is 16 supplies that entry to much larger prompt batches too.
7. The serving `triton_moe.py` likewise has an early custom-kernel return for eligible decode shapes, including EP expert-map handling.

The tuner also models top-10 routing over 128 local experts without the real 512-global-expert EP map. That is not the same local expert participation pattern as EP4.

The -47.7% whole-request regression is credible, and reverting that JSON was correct. It does not show a 47.7% decode-kernel regression. The code makes a prefill/fallback regression a strong explanation; a separated TTFT/decode measurement would settle it. A log saying the JSON was loaded does not establish that the hot decode kernel used it.

**Action:** tune the actual prefill backend at M values such as 128,256,512,1024 and, where valid,2048. Preserve the decode dispatch. Include the production default in the search, reproduce EP routing and packing, and record the actual kernel name for every candidate. Test `num_stages=1` versus 2 as a candidate, not as a universal gfx906 rule. The upstream prefill campaign independently demonstrates why PP deserves its own MoE configuration and measurement. [V620 prefill changes](https://github.com/leapdragon/vllm-rdna2-qwen/blob/rdna2/qwen38-flash-next/docs/rdna2/CHANGES.md#8d-prefill-on-gfx1030--2026-09-0405)

**4. The claimed approximately 4% ceiling on collective optimization is mathematically wrong. — Confirmed**

D143 treats the AR-on/off throughput difference as the entire cost of collectives and explains the small difference by overlap among four ranks. But AR-off selects RCCL; it does not remove communication. The experiment measures the difference between two collective implementations.

Likewise, rank overlap is already accounted for when dividing four-rank cumulative time by four. It cannot be used again to reduce a per-rank communication share to the AR-versus-RCCL improvement. Collectives can include rank-arrival skew, so their durations are not all avoidable transport time; that requires a timeline, not another division by world size.

Illustrative arithmetic using the recorded ledger: 2313.3 ms / 4 / 128 ≈ 4.52 ms of average collective kernel duration per output token in that capture. This is not a forecast of removable wall time. It is enough to show that a claimed hard ceiling derived from a 4% replacement gain is unjustified. The approximately 47.7→50 whole-request goal corresponds to only about 0.96 ms less time per output token under that benchmark.

The profile-to-wall accounting also needs repair. D143 combines a reported 3220.9 ms request with a separate 47.85 tok/s figure, although 128/47.85 is about 2675 ms. The analyzer accepts broad cumulative-time ratios without aligning request windows or computing the union of kernel intervals. A plausible sum is a useful sanity check, not a closed critical-path attribution. Its 85.5% factor should not support precise optimization ceilings elsewhere.

**Action:** capture a warmed decode-only window with per-rank timestamps and aligned collective instances. Separate useful compute, early-rank waiting, transport, and CPU/PLE gaps. Test the existing `VLLM_RDNA_AR_BLOCKS` cap before redesigning the protocol; compare the default with small values such as 1,2,4,8 in the real multiprocess engine. Keep fences and uncached staging. Small-message latency and rank skew matter more here than large-copy peak bandwidth.

**5. D145's WAVES result is not supported by the surviving harness; its occupancy argument is false. — Confirmed source defect**

The source in both the project and <container> `<workdir>/gemv_waves_sweep.cu` differs materially from production:

- It uses 32-lane groups and `WAVES*32` launch bounds, while production uses 64 lanes and `WAVES*64`.
- It uses a simple multiplication loop rather than production's four-deep `fdot2` implementation.
- It casts a 16-byte `uint4` to `half2*` and indexes eight elements. A 16-byte vector contains four `half2` values; accesses 4–7 are out of bounds.

This source cannot establish that the production kernel was properly swept. I did not execute the invalid harness. If the historic numbers came from a different corrected source, that source and build need to be recovered before retaining the conclusion.

Separately, constant total grid threads does not imply constant occupancy. Workgroup granularity, register allocation, resident-block limits, and distribution over CUs can change with block size. AMD's [HIP performance guidance](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html) describes these resource constraints.

The measured near-saturation of the large FP16 lm_head still supports deprioritizing tweaks to that particular kernel. It does not establish a universal 2.6% ceiling for dense improvements. D145's twelve-shape sum also needs occurrence weighting: one lm_head and dozens of layer projections do not have equal per-step weight.

**Action:** if this lane is reopened, parameterize the actual wave64 production kernel, verify each candidate against the same reference, time graph replays with GPU events, and weight shapes by observed occurrence counts. Bundle small wins when appropriate. An arbitrary 2% per-change rule should not discard several cheap, reproducible improvements whose combined effect matters.

**6. Three practical knobs remain, including one currently advertised but inactive. — Live/source observations**

| Observation | Implication | Useful next experiment |
|---|---|---|
| All four GPU PCI functions report NUMA node 1. <container>'s 64 allowed CPUs span nodes 0 and 1; both memory nodes are allowed. | Worker migration and remote CPU/page-cache placement are plausible costs, especially for eager prefill and PLE. This is not proof of a measured bottleneck. | Bind the service and PLE worker to available node-1 cores, with a deliberate memory policy; compare warmed PP, TTFT, decode and jitter. Inspect page residency because existing shared page-cache pages do not automatically move with a new policy. |
| Every GPU currently reports `power1_cap=178000000` µW, while D005/G6 specify 150 W. | Present state does not match the benchmark contract. This does not prove earlier runs were at 178 W. | Record and verify cap, clocks and temperatures at the start and during every run. Re-establish the 150 W reference before comparing caps. |
| The serve exports `VLLM_RDNA_DENSE_INT8=1`, but `rdna_dense_int8.enabled()` requires `on_gfx10x()`. | The flag does not enable dense INT8 shadows on gfx906. | Consider a validated wave64 weight-only INT8 implementation for selected large projections. It is a port plus quality evaluation, not a flag flip. |

The INT8 idea is not disproved by excellent FP16 bandwidth: reducing bytes can still help a bandwidth-saturated operation. The current INT8 implementation is architecture-specific; do not simply widen its Python gate. Weight-only INT8 can unpack to FP16 and use supported arithmetic, so it need not depend on the project's rejected integer dot instructions. Any benefit must exceed unpack cost, especially for small matrices. Start with measured large projections; lm_head alone is only around 0.4 ms per step, so even halving it cannot supply a huge end-to-end win.

The NUMA experiment follows the general CPU/GPU locality principles in AMD's [affinity guide](https://rocm.blogs.amd.com/software-tools-optimization/affinity/part-2/README.html). It should be checked against actual worker affinity after launch, not just the container's allowed CPU list.

**7. Prefill has not had an adequate optimization campaign. — Strongly supported**

The dense skinny route is restricted to M≤8. Larger prompt batches use different GEMM paths. A decode-focused shape or kernel ranking cannot establish that QSA, dense GEMMs, or collective bandwidth are negligible in prefill.

The serve sets `PYTORCH_TUNABLEOP_ENABLED=0`. Offline rocBLAS tuning of the actual FP16 prefill shapes is therefore a real remaining configuration opportunity. Gather shapes at the intended chunk sizes, tune against the installed gfx906 rocBLAS build, validate outputs, then serve in lookup-only mode with `PYTORCH_TUNABLEOP_ENABLED=1` and `PYTORCH_TUNABLEOP_TUNING=0`. Do not import V620 solution IDs. PyTorch documents this offline workflow and separation of tuning from execution. [PyTorch TunableOp](https://docs.pytorch.org/docs/2.14/cuda.tunable.html)

Recommended PP experiments, separately from decode:

- Exact cold prompts at 128,512,1024 tokens initially; add longer prompts only after increasing and validating the current 2048 model-length limit.
- Chunk budgets 512,1024,2048 where supported. D143's 1024-versus-2048 short-request decode test does not close the prefill question.
- Correct prefill MoE tuning, including pipeline stages and the backend transition discussed above.
- A prefill-only profile to rank dense GEMMs, MoE, QSA/GDN, RCCL, PLE and host dispatch.
- A scoped RCCL/P2P configuration test for PP if its profile warrants it. The local blanket prohibition on `NCCL_P2P_LEVEL=SYS` is stale: the current upstream changes distinguish a failed decode experiment from a later successful prefill experiment. This is a candidate on this topology, not a promised gain. [Upstream disposition](https://github.com/leapdragon/vllm-rdna2-qwen/blob/rdna2/qwen38-flash-next/docs/rdna2/CHANGES.md#9-what-was-measured-but-not-adopted)

For short-prompt latency, graph coverage and launch fusion may matter more than GEMM throughput. The installed log explicitly downgrades FULL to FULL_DECODE_ONLY for the QSA backend. Increasing the capture-size list alone does not prove prefill is captured. Preserve the working compile-disabled baseline; investigate selective graph/fusion support or a narrow miscompile fix as a separate engineering task.

**8. Keep the successful deployment decisions; narrow the claims they support.**

I would retain resident TP4+EP, the working custom AR, skinny FP16 decode, CPU PLE with its completion handshake, and graphs without broad Inductor compilation. The large measured wins from eliminating weight offload and enabling graphs are credible. Reverting the regressing tuned JSON was also correct.

The approximately 14.4 GB/s peer-copy observation does not alone prove an x8 hardware cap when the ports report Gen4 x16. Transfer engine, direction, packet behavior and synchronization can limit throughput. Unchanged performance with x8 versus x16 uplink does not mathematically prove the traffic stayed within the switch either: 14.4 GB/s fits beneath an x8 Gen4 uplink ceiling. The combined topology and peer-access evidence supports P2P, but avoid turning a bandwidth resemblance into a link diagnosis. Do not restart risky ACS experiments merely to pursue a larger copy number; measure the actual collective bottleneck first.

The environment record is stale: `config/environment.lock` still describes the old one-GPU VM and nominal ROCm 7.2.4. Several progress/gate rows contradict later addenda. Preserve the historical log, but maintain a compact current-state record with exact source and binary hashes, active kernel routes, power/NUMA settings, model and sidecar revisions, and benchmark definitions.

Finally, deterministic text and exact matches on one prompt are not full-model correctness validation. The absent independent full-model golden remains real. Establish a broader regression corpus and teacher-forced/logit or likelihood checks before changing quantization, accumulation, fusion, or speculative behavior. This is especially relevant because the project has already encountered coherent-looking but incorrect compiled output.

**Recommended order of work**

| Priority | Work | Reason and completion criterion |
|---|---|---|
| 1 | Correct the benchmark and snapshot the actual system | Obtain matched TTFT, PP and decode baselines at verified power. This prevents another optimization campaign targeting the wrong quantity. |
| 2 | NUMA placement A/B | Low implementation cost; concrete current topology mismatch. Keep only a reproducible improvement on the same prompts. |
| 3 | Dedicated PP tuning: rocBLAS, actual prefill MoE, chunk size | Largest clearly underexplored area. Require exact PP lengths, cold cache accounting, actual kernel-route evidence, and no decode regression. |
| 4 | MTP K=1/2 with cost breakdown | Acceptance is already reasonable; find whether a shorter draft is economical. Judge useful output rate and latency, with acceptance and round timing as diagnostics. |
| 5 | Decode collective/rank-skew analysis and AR block sweep | Reopen because the claimed 4% ceiling is invalid. Keep synchronization semantics intact. |
| 6 | Targeted top-k/routing fusion and count-clustered elementwise fusion | Real remaining work, sized from a clean decode-only timeline; no need to solve all frame attribution before testing a specific causal change. |
| 7 | Valid wave64 GEMV sweep and selective dense INT8 port | More engineering effort. Reuse production kernels, avoid the invalid harness, and run appropriate numerical/quality validation. |

I would not promise 75 tok/s decode or a particular PP number from this review. I would also not accept “every candidate closed by evidence.” Several candidates were closed using the wrong metric, the wrong execution path, or an invalid upper-bound argument. Correcting those issues and giving prefill its own campaign is the strongest next investment.

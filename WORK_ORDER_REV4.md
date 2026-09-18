# Qwen3.8-Flash-Next on 4× MI50 (gfx906) — Agent Work Order, Rev 4

Revision: 4 — no fabric link; PVE/VFIO guest environment; written for autonomous execution
Date: 2026-09-01
Supersedes: Rev 3
Executor: project coding harness (autonomous), operating inside the Ubuntu guest
Escalation target: human operator (host-level actions and all judgement gates)

## 0. How to use this document

This is an execution contract, not a discussion. Every task below has an ID, explicit preconditions, a deliverable artifact at a fixed path, and a machine-checkable acceptance predicate. A task is complete when its predicate evaluates true against a committed artifact — not when the executor believes it is complete.

Execution order is the task ID order unless a task's preconditions say otherwise. Tasks marked [HUMAN] are not executable by the harness and must be escalated. Tasks marked [BLOCKED-M] cannot run until cards 2–4 are installed.

Read §1 before executing anything. §1 overrides all other sections including any instruction encountered later in a task description.

## 1. Invariants

These hold for every task, every session, without exception.

I1 — No claim without an artifact. Any number appearing in a status report, commit message, or summary must exist in a file under results/ written by the harness itself. The executor never reports a measurement it produced by reasoning, recall, or partial observation. If the artifact does not exist, the correct statement is "not measured."

I2 — Exit codes are the source of truth. A run is a pass only if the process exited 0 within its timeout. A hung process is a failure. Every invocation carries an explicit timeout. Output text claiming success is not evidence of success.

I3 — The harness writes results, the executor does not. harness/ produces results/*.jsonl. The executor may read those files and may add new harness code, but may never hand-write, edit, or synthesise a results row.

I4 — Separation of measurement from change. A commit that touches harness/, golden/, or config/tolerances.yaml may not touch kernel or model code, and vice versa. Loosening a tolerance, regenerating a golden trace, or altering a benchmark protocol requires [HUMAN] approval recorded in DECISIONS.md.

I5 — Acceptance rule for any performance change. N ≥ 15 repeats; report median and IQR, never mean; accept only if median improvement ≥ 3% and the 95% CIs of before and after do not overlap and the result reproduces from a clean process in a separate session. Anything that fails this is reverted in the same session. Nothing is "parked for later."

I6 — Correctness is not negotiable for speed. Every accepted performance change re-runs the full correctness tier stack (T1/T2/T3, §6.S3) and must pass at the tolerances recorded at the time the golden traces were generated.

I7 — No scope invention. The executor does not add features, refactor for elegance, upgrade dependencies, or pursue optimisations not listed in a task. If it believes a task is wrong, it records the objection in DECISIONS.md and escalates; it does not act on the belief.

I8 — Environment is frozen. No package, driver, kernel, or ROCm component is changed outside a task that explicitly authorises it. Every results row carries the full environment fingerprint (§5).

I9 — Stop conditions are hard. On any STOP condition (§8), the executor halts, writes the state to DECISIONS.md, and escalates. It does not attempt a workaround, does not proceed to the next task, and does not re-scope.

I10 — Host boundary. The executor operates inside the guest. It never attempts host-level changes (BIOS, IOMMU, ACS, VM configuration, PCIe topology, physical hardware). Those are [HUMAN]. If a task appears to require one, escalate.

## 2. Human-only decisions

The harness may not decide these. Each requires an entry in DECISIONS.md signed by the operator before dependent tasks unblock.

ID	Decision	Blocks
H1	Deployment power cap (all gates and benchmarks are defined at it)	S0, all benchmarking
H2	ACS / IOMMU configuration on the PEX88096 downstream ports	G3, all of Track M
H3	VM topology for four cards — single VM, one IOMMU group	Track M
H4	Guest RAM allocation and ballooning policy for the n-gram table	S2
H5	Text-only vs multimodal serving scope	S2
H6	Any tolerance change or golden-trace regeneration	I4
H7	Project continuation after any STOP condition	§8
H8	Whether to proceed past G3 failure (bare metal migration, or terminate)	Track M

## 3. Environment: what the PVE guest changes

The single MI50 is VFIO-passed into an Ubuntu guest on Proxmox VE. The four-card configuration is assumed to follow the same pattern. This is not a neutral detail — it changes four things materially.

### 3.1 The ACS conflict — resolve this first

There is a direct conflict between clean VFIO passthrough and working PCIe P2P:

ACS enabled on the PEX88096 downstream ports → clean, separate IOMMU groups, easy passthrough → peer-to-peer traffic is redirected upstream to the root complex, destroying the latency advantage of the switch and probably the collective budget with it.
ACS disabled or overridden → the switch's downstream devices land in one IOMMU group → P2P stays local to the switch → all devices in that group must be passed to the same VM.

The resolution is H3: pass all four GPUs to a single VM, in a single IOMMU group, with ACS not redirecting. Devices sharing a VFIO container share an IOMMU domain, which is what makes device-to-device DMA work.

This is determinable today, with one card, from the host. Read ACSCap and ACSCtl on the PEX88096 downstream ports via lspci -vvv. If ACS is enabled in hardware and cannot be overridden, the four-card target as designed is at serious risk and that should be known now rather than in four months. See task P2.

### 3.2 Large BAR through VFIO

The one-shot all-reduce requires each GPU to write directly into peer VRAM, which requires the full 32GB BAR — not the 256MB default. Under VFIO this needs, in order: host BIOS above-4G decoding and Resizable BAR enabled; vfio-pci exposing the resized BAR; a q35 machine type with OVMF; and a 64-bit MMIO aperture in the guest large enough for four 32GB BARs (on the order of a 1TB PCI hole). Verify from inside the guest, not just from the host.

### 3.3 XNACK is forced off — this constrains the n-gram design

ATS/PRI is generally unavailable to a VFIO guest, so xnack+ and demand-paged SVM are not options. Consequences:

Build all artifacts for gfx906:xnack- and pin that target in the toolchain fingerprint. Mismatched xnack targets produce unpredictable paging and performance behaviour.
The n-gram offload cannot rely on page-fault-driven migration from host memory. It must use explicit transfers or hipHostRegister-backed zero-copy over PCIe. That path works fine without ATS — it is ordinary IOMMU-mediated DMA to pinned host memory — but it must be the design, not a fallback.
### 3.4 VFIO pins all guest memory

The guest's entire RAM allocation is pinned on the host. A guest sized to hold the 102GB BF16 n-gram table in page cache means that much host memory is pinned and unavailable elsewhere. Ballooning must be disabled. Back the guest with hugepages — the n-gram gather is a TLB problem, not a bandwidth problem, and 2MB or 1GB pages matter more here than anywhere else in the project.

### 3.5 Measure the virtualization tax

Kernel launch latency is a first-class term in this project's budget (§4). MMIO and doorbell writes traverse the VFIO path. Task S5 must measure launch latency in the guest and record it as the operating value — no host-native figure is applicable.

## 4. Risk posture after removing fabric link

Infinity Fabric Link is unattainable. Rev 3 treated RCCL-over-xGMI as a possible route to the collective budget that would eliminate the custom all-reduce workstream. That route is gone.

Consequences:

gfx906_ar_oneshot is now mandatory, not conditional. Task M2 is on the critical path.
There is no interconnect fallback. PCIe P2P through the PEX88096, mediated by VFIO, is the only path. If it does not work, the 60–65 t/s target is not reachable in this topology.
G3 becomes the dominant project risk and it remains untestable until cards 2–4 arrive — except for the ACS determination in §3.1, which is available now and is the closest thing to an early read on it.

The budget with no fabric link, restated:

Term	Cost	Implied ceiling
119 collectives × 156 µs (RCCL over PCIe)	18.6 ms	54 t/s — below the "credible" line
119 collectives × 34 µs (custom, required)	4.0 ms	256 t/s
~2,700 kernels × 4 µs launch gap	10.8 ms	92 t/s
Weight streaming (~1.0 GB/GPU/token)	~1.4 ms	~700 t/s

At the 65 t/s target (15.4 ms/token), collectives and launch overhead are ~52% of the budget and weight streaming is under 10%. This project is latency-bound. The kernel work matters least of the three, which is why §6 orders it last.

The 2,700-kernel and 119-collective figures are unverified priors (task P1). Task S4 replaces them with measurement.

## 5. Repository, artifacts, and state

repo/
  DECISIONS.md            append-only; every human decision, every objection,
                          every STOP event, with timestamp and task ID
  config/
    environment.lock      full fingerprint, regenerated by harness only
    tolerances.yaml       [I4] change requires H6
    model.json            config.json-derived facts, written by S0
  harness/                benchmark + correctness runners; sole writer of results/
  golden/                 reference traces [I4]
  kernels/gfx906/
  vllm/                   branch: mi50/qwen38-flash-next-gfx906
  tools/gfx906/
  results/*.jsonl         append-only
  reports/                generated summaries; every figure traceable to results/

Every results row carries: run_id, task_id, git_commit, dirty_flag, rocm_version, torch_version, triton_commit, vllm_commit, kernel_version, gfx_target (must read gfx906:xnack-), guest_kernel, pve_version, gpu_count, tp, ep, mtp, power_cap_w, observed_power_w, gpu_clock_mhz, hbm_clock_mhz, temp_c, soak_seconds, context, gen_tokens, repeat_index, tps, prefill_tps, ttft_ms, kernel_config, correctness_hash, exit_code, wall_seconds, timeout_s, notes.

Thermal validity: no row is valid without ≥ 600s soak. Rows where sustained clocks deviated > 5% from the session median are marked invalid and excluded — not averaged in.

## 6. Track S — single-card execution

Available work: roughly 14–18 weeks. Ordering rule: front-load anything that blocks or unblocks Track M; defer anything that stays single-card forever. The kernel laboratory (S6) never needs peers, so it is last and is explicitly pausable.

P1 [HUMAN] — Resolve the V620 source

Determine whether leapdragon/vllm-rdna2-qwen exists. Every figure in §4 traces to it. Outcome recorded in DECISIONS.md. If it does not exist, §4's priors are unfounded; S4 becomes the sole source of truth for phase ordering and the executor must not cite §4 numbers in any report.

P2 [HUMAN] — ACS determination (§3.1)

Read ACSCap/ACSCtl on the PEX88096 downstream ports from the host. Record whether ACS redirection can be disabled or overridden, and the resulting IOMMU grouping.

STOP-A: if ACS cannot be disabled or overridden, escalate under H8 before any further Track S work is funded. This is the earliest available signal on G3.

P3 [HUMAN] — Power, cooling, guest sizing

Decide H1 (power cap), H4 (guest RAM, ballooning off, hugepage backing). Confirm 4× 300W passive cooling and PSU headroom (~1400W with host). Confirm above-4G decoding and ReBAR in host BIOS.

S0 — Freeze, audit, arithmetic (1 week)

Preconditions: H1 decided.

Actions:

Fetch config.json from wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16. Extract and write to config/model.json: hidden_size, vocab_size, num_hidden_layers, num_experts, num_experts_per_tok, moe_intermediate_size, shared-expert dims, n-gram table shape and injection layer, MTP module shape, layer-type map.
Compute the exact per-token active-parameter split by dtype and the resulting bytes/token/GPU at TP4. Write to results/byte_budget.jsonl.
Compute surrogate sizing (§7) from real values and write the chosen slice depth to config/model.json.
Generate config/environment.lock, including gfx_target verification.
Verify large BAR from inside the guest (§3.2).

Acceptance:

config/model.json exists and every field is populated from the checkpoint, not from any estimate.
results/byte_budget.jsonl contains a per-dtype breakdown summing to the stated active parameter count within 1%.
gfx_target == "gfx906:xnack-".
Guest-visible BAR size for the MI50 ≥ 32GB.

STOP-B: guest-visible BAR < 32GB after H-side remediation. Large BAR is a hard prerequisite for the mandatory custom all-reduce; escalate under H8.

Note for the executor: working backwards from the published ~123B expert parameters across 48 layers implies hidden_size ≈ 2560. This is a derivation, not a fact. Use the value from config.json and record any discrepancy in DECISIONS.md.

S1 — Toolchain gate (2 weeks, hard kill gate)

Preconditions: S0 complete.

Actions: Build TheRock ROCm with -DTHEROCK_AMDGPU_TARGETS=gfx906, current PyTorch, current Triton, all targeting gfx906:xnack-. Then execute, each with a timeout, each writing a results row:

Check	Acceptance predicate
rocminfo	reports gfx906, xnack-
torch device visibility	torch.cuda.device_count() == 1, arch string matches
torch GEMM	fp32 and fp16 results within tolerance vs CPU reference
custom HIP kernel	compiles, runs, exit 0, output verified
Triton vector-add	exit 0, output verified
torch.compile	exit 0, output matches eager within tolerance
rocprofv3	produces a non-empty kernel trace with correct kernel names
HIP graph capture/replay	multi-kernel synthetic step captures, replays, output identical to uncaptured
host-coherent primitives	fine-grained host allocation; GPU atomic on host memory; flag-based signal/wait round-trip; all verified, latency recorded

Gate G1: all rows exit 0 within 14 days.

Gate G2: HIP graph capture row passes.

The last two checks are not optional and are not deferrable. G2 determines whether S7 is a multi-week workstream or a formality. The host-coherent primitives are the building blocks of the mandatory gfx906_ar_oneshot; validating them GPU↔host now means M2 starts from known-good semantics rather than debugging two problems at once.

On G1 failure: fall back to ROCm 6.4 + nlzy/triton-gfx906, record +2 months in DECISIONS.md, and escalate under H7 before proceeding. Do not start on the 6.x path without that approval.

S2 — Architecture layer and model port (4–6 weeks)

Preconditions: S1 (G1) passed; H4, H5 decided.

Actions:

Architecture abstraction: gfx103x and gfx906 as siblings behind a runtime capability struct. Forbidden: any predicate of the form if gfx1030 or gfx906. MI50 is wave64; the RDNA2 code assumes wave32. Reduction trees, register budgets and LDS tiling are redesigned, not translated.
Port architecture-independent pieces: model classes, EP plumbing, loader, custom-op interfaces, validation and benchmark tooling.
n-gram host-offload HIP port. Highest-value single-card deliverable. Upstream ships this NVIDIA-only. Design constraints: single injection at layer 2 indexed by bigrams and trigrams — one host round-trip per step, not 48; hipHostRegister zero-copy or explicit transfer, never demand paging (§3.3); hugepage-backed table (§3.4).
AWQ / compressed-tensors pack-quantized dequant path.

Acceptance:

Grep for the forbidden predicate returns zero matches.
The surrogate slice loads real weights from the real checkpoint, exit 0.
n-gram lookup returns values bit-identical to a CPU reference lookup on 10,000 random indices.
Dequant output matches a CPU fp32 dequant reference within tolerances.yaml.

S3 — Correctness harness and golden traces (2 weeks)

Preconditions: S2 complete. Must complete before any kernel work. Without it no kernel change is falsifiable.

Actions: Build three tiers with tolerances declared in advance and frozen under I4:

T1 kernel: max abs and rel error vs PyTorch reference, per shape.
T2 layer: cosine similarity of hidden states per layer vs golden trace, with a per-layer drift budget.
T3 slice: deterministic output match for the surrogate on fixed inputs.

Reference generation: extract the same layer slice from the checkpoint, run it on CPU in fp32 in the guest, and compare. Exact, cheap, needs neither the full model nor a second machine. Full end-to-end greedy token matching and perplexity are [BLOCKED-M].

Acceptance: golden/ contains traces for ≥ 8 fixed inputs; harness/verify.py exits 0 on an unmodified tree and exits non-zero on a deliberately perturbed kernel (negative control required — an oracle that never fails is not an oracle).

S4 — Surrogate build and first real profile (3 weeks) ⭐

Preconditions: S3 complete.

Actions:

Build the layer-slice surrogate per §7: complete repeating unit (3× GDN + 1× QSA, each with its full 512-expert MoE), real dimensions, real routing, real gated residual, n-gram injection at layer 2, lm_head.
Profile with rocprofv3 and emit to results/profile_s4.jsonl: kernel count per layer and per repeating unit, ×12 projection to 48 layers; gap distribution and launch-overhead share; time share by kernel class; complete shape inventory; HIP graph capture coverage and break points under real data-dependent routing; measured n-gram fetch latency.
Derive the collective count and message-size histogram analytically from the architecture at TP4+EP. Write the prediction to results/collective_prediction.jsonl — it is tested on Track M day one.

Acceptance:

Surrogate runs, exit 0, T3 passes.
profile_s4.jsonl contains a kernel-class time-share breakdown summing to 100% ± 2%.
collective_prediction.jsonl exists with per-size counts.

This replaces §4's unverified priors. Every subsequent priority derives from it. Where S4 and §4 disagree, S4 wins and the executor records the divergence.

S5 — Hardware ceilings (1 week)

Preconditions: S4 complete; H1 decided.

Actions, all at the H1 power cap, all after ≥ 600s soak, all inside the guest: HBM streaming ceiling under the surrogate's real access patterns; skinny GEMV at M ∈ {1,2,4,8} at S4's real dimensions; INT4 unpack → scale → dot; wave64 reduction design comparison; empty-kernel launch latency for HIP and Triton (§3.5 — this is the operating value; no host-native figure applies).

Acceptance: results/ceilings.jsonl populated with ≥ 15 repeats per measurement, median and IQR recorded.

Forbidden: citing 1 TB/s or any datasheet figure. Only measured ceilings are used downstream. Expect roughly 700–850 GB/s and use whatever is actually measured.

S6 — Kernel laboratory (6–10 weeks, pausable)

Preconditions: S4, S5 complete. Ordered strictly by S4 measured time share.

Build tools/gfx906/bench_{dense,moe,qsa,norm,fusion,memory}.py, seeded with S4 shapes. Each candidate reports: PyTorch reference, generic ROCm baseline, gfx906 implementation, numerical error, latency, effective GB/s, occupancy, register count, emitted ISA.

gfx906 instruction selection. v_dot2_f32_f16 — fp16×2 with fp32 accumulate in one instruction — is the workhorse for W4A16 decode: dequantize to fp16x2, accumulate in fp32; better numerics than fp16 accumulation, and free. v_fma_mix_f32 covers mixed paths. v_dot4_i32_i8 helps only if activations are also INT8. v_dot8_i32_i4 requires W4A4 and is almost certainly too lossy. There is no MFMA on gfx906, so prefill stays on vector FP16.

Acceptance thresholds: FP16 skinny GEMV > 60% of the S5 measured streaming ceiling (75%+ good); W4 expert GEMV ≥ 65% (≥ 80% stretch), with expert_map folded into the kernel. QSA and GDN: correctness first under wave64, 64 KiB LDS, head_dim 256, no rocWMMA. The MTP module's own attention layers are QSA, so MTP quality later depends on this.

Every change is subject to I5 and I6.

Pause rule: if cards 2–4 arrive before S6 completes, pause S6 and execute Track M through M1 first, so kernel priorities are set by real four-card profile data rather than the ×12 extrapolation. Resume afterward.

S7 — Fusion or graph coverage (0–8 weeks, conditional on G2)

If G2 passed and S4 shows good capture coverage: extend coverage; minimal fusion work.

If G2 failed: fusion is on the critical path. Fuse the high-frequency sequences S4 identified — norm→reshape→gate→silu→multiply→residual, in-kernel MoE glue, fused RMSNorm/RoPE, fused qk-norm+rope+gate, fused gated residual, fused shared expert. Port concepts, not implementations. Add ~2× to the remaining timeline and record it.

S8 — Track M harness, written now (1 week)

Preconditions: S1 complete (host-coherent primitives validated).

Write and stub the full multi-card suite so day one with four cards is measurement, not development: P2P read/write matrix both directions all pairs; HIP IPC mapping and coherency; RCCL latency sweep at the S4-predicted message sizes; gfx906_ar_oneshot scaffold built on the S1-validated signalling primitives; automated RCCL-vs-custom comparison at identical sizes and protocol.

Acceptance: every M-track harness entry point exists and exits 0 in a single-GPU degenerate mode (self-loop) with a clear "requires N GPUs" skip for peer paths.

## 7. Surrogate sizing

From the published figure of ≈123B routed-expert parameters across 48 layers:

expert params per layer   ≈ 123e9 / 48        ≈ 2.56e9
implied hidden_size        = 2.56e9 / (512 × 3 × 640) ≈ 2560   [VERIFY IN S0]
expert bytes per layer     = 2.56e9 × 0.5156 B ≈ 1.32 GB       (INT4 + g128 fp16 scales)
repeating unit (4 layers)  ≈ 5.3 GB + attention/GDN/norms
two repeating units (8L)   ≈ 11 GB

A single 32GB MI50 holds the complete repeating architectural unit with all 512 experts, with room for activations, KV and n-gram staging. Two units fit. The n-gram injection point (layer 2) falls inside the first unit.

Also size lm_head in S0: if vocab_size ≈ 151k and hidden_size ≈ 2560, it is ~770 MB of BF16 read on every decode step — plausibly a top-three decode bandwidth consumer and a stronger INT8-shadow candidate than the dense projections. Include it in the surrogate.

## 8. Gates and STOP conditions

Gate	Task	Predicate	On failure
G1 Toolchain	S1	all checks exit 0 within 14 days	ROCm 6.4 fallback; escalate H7
G2 HIP graphs	S1	capture/replay output identical	S7 becomes critical path; timeline ~2×
G3 P2P	M0 [BLOCKED-M]	peer write works, large BAR, RCCL measurable	STOP — escalate H8
G4 End-to-end	M3	greedy token IDs match reference, first 64 tokens	STOP
G5 Collectives	M2	≤ 34 µs at dominant message size	recompute budget; 60–65 t/s may be unreachable
G6 Projection	M4	≥ 50 t/s at H1 power cap	STOP or re-target

Named STOP conditions:

STOP-A (P2): ACS cannot be disabled or overridden on the switch.
STOP-B (S0): guest-visible BAR < 32GB after host remediation.
STOP-C (S1): G1 fails and the ROCm 6.4 fallback also fails within 14 days.
STOP-D (M0): P2P blocked or redirected through the root complex.
STOP-E (any): two consecutive sessions with no task reaching its acceptance predicate. This exists to catch grinding.

Decode gates, at the H1 power cap: < 40 broken / 40–50 unoptimized / 50–55 credible / 55–60 succeeding / 60–65 target / 65–70 excellent / > 70 stretch. Plus a separate TTFT gate — expect prefill to regress against the V620 reference, since neither part has matrix acceleration and MI50's ~26.5 TFLOPS FP16 sits below V620's ~40. MI50 wins decode on bandwidth and loses prefill on compute.

Exposure to name plainly

G3 is untestable for the entire 14–18 week Track S window, and with fabric link unavailable it has no fallback. P2 is the only early signal. If G3 fails, most of Track S retains independent value — a working modern gfx906 vLLM path with real kernels is worth having regardless — but the four-card target does not survive it. This is the shape of the bet and it should be made knowingly.

## 9. Track M — on arrival of cards 2–4

Preconditions: H3 (single VM, single IOMMU group), P2 resolved favourably.

M0 — Interconnect truth (3 days). P2P matrix, large BAR under all four, HIP IPC, atomics, coherency. RCCL sweep at S4-predicted sizes. G3 resolves here.
M1 — Validate the S4 prediction (2 days). Compare measured collective count and size histogram against collective_prediction.jsonl. A large miss invalidates the ×12 extrapolation and forces a re-check of S6 kernel priorities.
M2 — gfx906_ar_oneshot (2–4 weeks, mandatory). Write to peers → signal host-coherent state → wait for contributors → local reduce, built on the S1-validated primitives. G5.
M3 — First full end-to-end run (2 weeks). Full checkpoint, TP4 + EP mandatory (moe_intermediate_size 640 gives 160 columns at TP4, not divisible by the 128 quantization group; EP distributes whole experts and the group math stays exact), MTP off, short context, greedy, Triton MoE backend first. G4.
M4 — Integration and context scaling (4 weeks). Swap in S6 kernels one at a time under I5. Sweep 4K → 32K → 128K → 262K recording t/s, TTFT, VRAM. Decide KV quantization. Only 12 of 48 layers are QSA and GDN carries a fixed-size recurrent state, so KV growth should be favourable — measure it. G6.
M5 — MTP (3 weeks). MTP=3, measure acceptance rate.
M6 — INT8 shadow weights (2 weeks, conditional). Only after M5. The drift they introduce reduces MTP acceptance; deciding before MTP risks locking in a choice MTP reverses. Reconsider lm_head as the primary candidate.

## 10. Retained reference

Model facts (verified): 176–180B total = 125B main + 51B n-gram + 4B MTP; 6B active/token; 48 layers as 3× GDN → MoE, 1× QSA → MoE, ×12; 512 routed experts
1 shared; moe_intermediate_size 640; native 262,144 context; multimodal; n-gram table indexed by bigrams/trigrams at layer 2, offload upstream is NVIDIA-only. Checkpoint: wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16, sym g128, compressed-tensors pack-quantized, embed_tokens and lm_head in BF16.
Scope corrections: no INT4 n-gram sidecar — public checkpoints keep it at 102GB BF16 and 512GB host RAM makes that a non-issue; text-only serving initially (H5); INT8 shadow weights deferred past MTP.
Terminology: the draft's "PLE" is the n-gram embedding table; "HyperConnection" is Gated Residual (four branches, element-wise data-dependent read gate, per-branch scalar write gate).
Repository strategy: base on the current Flash-Next-capable vLLM tree, branch mi50/qwen38-flash-next-gfx906. Mine nlzy/vllm-gfx906 (archived Feb 2026) and nlzy/triton-gfx906 for techniques, not as a base. Build the patch-extraction index in S2 week 1: which file solves which gfx906 problem — wave64 reductions, AWQ/GPTQ dequant, Triton lowering fixes, W4A16 paths.
Cost–benefit: roughly $2,000 of hardware saved against 4–7 months of engineering, versus 4× RTX 3090 running the same model today on existing Marlin kernels. Worth doing if the gfx906 path is itself the goal; not on tokens-per-dollar.

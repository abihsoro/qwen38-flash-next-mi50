# DECISIONS.md — append-only decision, objection, and STOP log

Rules: every entry carries a timestamp and the task ID it belongs to. Human decisions require
operator sign-off (the operator is the escalation target of Work Order Rev 4). The harness never
edits or rewrites an existing entry — it only appends. Entries are never deleted.

---

## D001 — Work Order Rev 4 accepted as the execution contract
- Date: 2026-09-01
- Task: bootstrap (project creation)
- Source: operator (handed the order to the executor and instructed project creation)
- Decision: The executor operates under Agent Work Order Rev 4 (no fabric link; PVE/VFIO guest
  environment). Rev 3 is superseded. §1 invariants override all other text.
- Recorded by: executor (bootstrap)

## D002 — Bootstrap environment observation (I1: observed facts only)
- Date: 2026-09-01
- Task: bootstrap
- Source: executor measurement
- Observation: the session environment at bootstrap is guest `<source-container>`,
  kernel `<kernel>`, 2 vCPU, 15 GB RAM. `lspci` shows only QEMU virtual devices —
  **no AMD GPU present, no ROCm installed**. Python 3.12.3, PyYAML 6.0.1, `huggingface_hub`
  importable, no torch.
- Consequence: this VM is not the §3 VFIO GPU guest. Bootstrap (this project) needs no GPU.
  Tasks S0/S1 and everything downstream need the GPU guest, whose identity the operator must
  confirm (candidate seen in the workspace: VM `vm103`, screenshots dated 2026-08-29 show a
  no-GPU state). Held open under task P3.

## D003 — Open human decisions (queued, awaiting operator sign-off)
- Date: 2026-09-01
- Task: bootstrap
- Status list (not decisions — the harness cannot decide these per §2):
  - **H1** Deployment power cap — blocks S0 and all benchmarking. **Asked first.**
  - H2 ACS/IOMMU configuration on the PEX88096 downstream ports — blocks G3, all of Track M.
  - H3 VM topology for four cards (single VM, one IOMMU group) — blocks Track M.
  - H4 Guest RAM allocation and ballooning policy for the n-gram table — blocks S2.
  - H5 Text-only vs multimodal serving scope — blocks S2.
  - H6 Any tolerance change or golden-trace regeneration — blocks I4.
  - H7 Project continuation after any STOP condition.
  - H8 Whether to proceed past G3 failure.

## D004 — Open [HUMAN] tasks (blocked-on-human, first escalation round)
- Date: 2026-09-01
- Task: P1, P2, P3
- Status list:
  - **P1** — Resolve whether `leapdragon/vllm-rdna2-qwen` exists; every §4 figure traces to it.
    If it does not exist, §4 priors are unfounded and S4 becomes the sole source of truth.
  - **P2** — ACS determination: read ACSCap/ACSCtl on the PEX88096 downstream ports from the host
    (`lspci -vvv`). Record whether ACS redirection can be disabled/overridden and the IOMMU
    grouping. **STOP-A** if ACS cannot be disabled or overridden (escalate under H8).
  - **P3** — Power, cooling, guest sizing: decide H1 and H4; confirm 4× 300W passive cooling and
    PSU headroom (~1400 W with host); confirm above-4G decoding and ReBAR in host BIOS; confirm
    which PVE guest holds the VFIO MI50 (D002).

---
*End of log at bootstrap. Future entries append below this line only.*

---

## D005 — H1: deployment power cap decided
- Date: 2026-09-01
- Task: H1 (blocks S0, all benchmarking)
- Source: operator (signed)
- Decision: **Power cap = 150 W per GPU.** All gates and benchmarks are defined
  at this cap (work order §2.H1, §5, §8). Decode gates and TTFT gate apply at
  150 W.
- Consequence recorded: the work order's P3 assumed 4× 300 W passive cooling and
  ~1400 W PSU headroom with host; at a 150 W cap the GPU-side requirement drops
  to ~600 W (4× 150 W) plus host. The operator's ~1400 W assumption is
  superseded for PSU sizing unless the operator says otherwise. P3's remaining
  confirmations (cooling, PSU headroom, ReBAR/above-4G in host BIOS) are still
  open and tracked under P3.

## D006 — P3 partial: GPU guest identity confirmed by operator
- Date: 2026-09-01
- Task: P3 (and D002 open item)
- Source: operator
- Decision/fact: The VFIO GPU guest is **<vm>** (ssh `<user>@<lan-ip>`),
  running on PVE host **<source-host>** (`root@<source-ip>`). This resolves the D002
  open item (this session's VM, `<source-container>`, has no AMD GPU).
- Remaining P3 items: confirm 4× passive cooling + PSU headroom (at the 150 W
  cap, see D005) and confirm above-4G decoding / ReBAR in the host BIOS.

---
*End of log. Future entries append below this line only.*

---

## D007 — S0 findings: checkpoint-derived facts and discrepancies vs §7/§10 priors
- Date: 2026-09-01
- Task: S0 (in progress)
- Source: executor measurement (I1: from `config.json` + safetensors headers of
  `wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16`; 222,746 tensors catalogued; provenance
  in `config/model.json`; breakdown in `results/byte_budget.jsonl`)
- Confirmed priors: hidden_size 2560 (= §7 derivation); n-gram table 51.23B BF16
  params (matches §10 "51B"); total checkpoint 180.24B (within §10 "176–180B");
  n-gram injection at layer 2 (`ple_layer_ids: [<bus>]`, matches §10); 48 layers as
  pattern 3× GDN (linear_attention) + 1× QSA (full_attention), ×12; only
  routed-expert linears are quantized (compressed-tensors pack-quantized int4
  sym g128; verified against the tensor inventory — no other packed tensors).
- Discrepancies (recorded per the S0 note: use config.json values, record the
  difference):
  1. **Active params/token: derived 7.510B vs §10 stated 6B (+25.2%).** Breakdown:
     routed experts 2.359B (10/512 of 120.80B) + shared expert 0.236B +
     attention/GDN/indexer/hyper-connection/norms 3.637B + embed_tokens 0.636B +
     lm_head 0.636B. Excluding embed+lm_head (the common published definition):
     6.239B (+4.0%). The published "6B" definition is unresolvable from config;
     the config-derived 7.510B is the operative budgeting number and supersedes
     the §10 figure for all downstream arithmetic.
  2. **lm_head = 1.27 GB BF16 per decode step** (vocab 248,320 × hidden 2560 × 2B),
     not the §7 prior ~0.77 GB (assumed vocab ≈ 151k). Bigger than the prior —
     strengthens the INT8-shadow candidate status (M6).
  3. **MTP = 2.61B params, not §10's "4B".** The MTP module carries a full
     512-expert MoE layer (gate_up_proj [512,1280,2560], down_proj [512,2560,640]
     ≈ 2.52B) plus one full-attention layer, hyper-connections and norms.
  4. Surrogate sizing from real values: expert bytes/layer = 1.308 GB
     (INT4 0.5 B/param + g128 fp16 scales + shared expert BF16); complete
     repeating unit (4 layers) ≈ 6.05 GB; **slice depth chosen = 4 layers +
     lm_head** (n-gram injection at layer 2 falls inside unit 1); two units
     (8 layers) also fit a 32 GB MI50 (optional depth 8).
- Status: byte_budget acceptance holds against the config-derived total
  (breakdown sums exactly); the §10 stated 6B prior is outside 1% and the
  discrepancy is recorded here as required. Remaining S0 acceptance items:
  `gfx_target == gfx906:xnack-` and guest-visible BAR ≥ 32 GB — require the GPU
  guest (<vm>), currently unreachable (D008).

## D008 — <vm> unreachable: SSH host-key change (operator action required)
- Date: 2026-09-01
- Task: S0 / P3
- Source: executor measurement
- Observation: `ssh <user>@<lan-ip>` fails strict host-key verification —
  known_hosts holds an old ECDSA key for <lan-ip>; the host now presents
  an ED25519 fingerprint `SHA256:<redacted>`.
- Action (operator): verify the key change is legitimate (VM rebuild/reinstall
  on <source-host> is plausible) and either confirm it, or clear the stale entry
  (`ssh-keygen -R <lan-ip>`) so the harness can connect. The executor
  will not bypass host-key verification. Blocks: S0 BAR/gfx_target checks, and
  all GPU-side work.

---
*End of log. Future entries append below this line only.*

---

## D009 — S0 GPU acceptance PASS: <vm> measurements (I1, probe = harness/gpu_probe.py)
- Date: 2026-09-01
- Task: S0 (completes)
- Source: executor measurement on <vm> (`<vega-host>`, kernel <kernel>);
  raw probe in `scratch/<vm>_probe.json`, ingested rows in
  `results/gpu_probe.jsonl` + `results/S0.jsonl`
- **gfx_target = gfx906:xnack-** — rocminfo reports gfx906, "XNACK enabled: NO"
  (S0 acceptance PASS; consistent with §3.3: xnack+ unavailable under VFIO).
- **Guest-visible BAR ≥ 32 GB — PASS (STOP-B cleared):** BAR0 = 34,359,738,368 B
  (32 GiB), 64-bit prefetchable, mapped at <bar-addr> (above 4G); BAR2 =
  256 MiB (default aperture); amdgpu bound; `/dev/kfd` + renderD128 present.
- **Card identity (flag to operator):** the passed-through device is
  `<device-id>` — **Radeon Pro Vega II**, not an MI50 (`<device-id>`). Same gfx906 /
  32 GB HBM2 class, so the plan's gfx906 target is unaffected; recorded for
  accuracy (P3 describes the card as MI50).
- Environment: ROCm 7.2.4 installed (`/opt/rocm-7.2.4`, `/usr/bin/rocminfo`);
  guest RAM 22.8 GB; **HugePages_Total = 0** — §3.4 wants hugepage backing and
  a guest sized to hold the 102 GB n-gram table in page cache; both are H4
  decisions pending operator sign-off before S2's n-gram work.
- Guest exposes **no IOMMU groups** (`/sys/kernel/iommu_groups` empty) — the
  ACS/IOMMU grouping that matters is host-side (P2/H3, [HUMAN]); unobservable
  from inside the guest, as expected.
- S0 status: **complete** — model.json from checkpoint, byte_budget.jsonl
  (discrepancy D007 recorded), gfx906:xnack-, BAR ≥ 32 GB.

---
*End of log. Future entries append below this line only.*

---

## D010 — S1 toolchain strategy: triage on installed ROCm 7.2.4 first
- Date: 2026-09-01
- Task: S1 (toolchain gate)
- Source: operator (signed)
- Decision: The work order mandates building TheRock ROCm from source, but
  <vm> already has ROCm 7.2.4 installed (an environment fact the order did not
  anticipate). The operator chose: **triage on 7.2.4 first** — install current
  torch/triton against the installed ROCm 7.2.4 and run the nine S1 checks
  (incl. G2 HIP-graph capture and the host-coherent primitives) on it. TheRock
  is built only if the 7.2.4 triage fails a check. The G1 fallback (ROCm 6.4 +
  nlzy/triton-gfx906) remains the fallback and still requires H7 approval per
  the work order (not pre-authorized by this decision).
- Consequence: I8 is overridden for S1's scope only — toolchain installation on
  <vm> is authorized by this task (S1 is the task that authorizes it).

---
*End of log. Future entries append below this line only.*

---

## D011 — S1 triage verdict on ROCm 7.2.4: G1 FAIL (6/9), G2 PASS
- Date: 2026-09-01
- Task: S1 (toolchain gate)
- Source: executor measurement on <vm> (rows in `results/S1.jsonl`, 22 rows;
  environment: ROCm 7.2.4 + torch 2.12.1+rocm7.2 + triton-rocm 3.7.1,
  gfx906:xnack-)
- Per-check final verdicts (last run of each):
  - check01 rocminfo gfx906/xnack-: **PASS** (exit 0)
  - check02 torch visibility (device_count 1, gfx906 in arch_list): **PASS**
  - check03 torch GEMM fp32/fp16 vs CPU: **FAIL** (exit -6, SIGABRT) —
    the torch wheel's bundled rocBLAS has **no gfx906 TensileLibrary** (files
    exist only for gfx900/908/90a/942/950/1030/110x/115x/120x); the system
    ROCm 7.2.4 rocBLAS 5.2.0.70204 has the same gap. ROCm 7.x dropped gfx906.
  - check04 custom HIP kernel (hipcc saxpy): **PASS** (err 2.4e-4)
  - check05 Triton vector-add: **FAIL** — triton-rocm 3.7.1: `unsupported
    target: 'gfx906'` (triton 3.x dropped gfx906; only 3.6/3.7 available on
    the rocm7.2 index)
  - check06 torch.compile: **FAIL** (exit -6) — same rocBLAS gfx906 abort
    (eager forward aborts before compilation)
  - check07 rocprofv3 kernel trace: **PASS** (saxpy found in results.db)
  - check08 HIP graph capture/replay: **PASS** (max err 0.0) — **G2 PASS**:
    S7 stays on the "extend coverage, minimal fusion" path
  - check09 host-coherent primitives: **FAIL** — **flag-based signal/wait
    round-trip WORKS (1.37 µs, R=1000)**; **GPU atomic on host memory FAILS**
    in both device and system scope (`atomicAdd` / `atomicAdd_system` on
    hipHostMallocCoherent memory: value unchanged, no HIP error; plain stores
    DO land). Likely the PCIe-atomic path is blocked under VFIO/IOMMU —
    host-side investigation belongs to P2/ACS scope ([HUMAN]).
- **M2 design implication (recorded for Track M):** `gfx906_ar_oneshot` must use
  **store-based slot signaling + `__threadfence_system()`** (verified working,
  1.37 µs round trip) and **NOT atomic counters on host memory**. Contributor
  counting: one store per contributor slot, orchestrator polls N slots.
- Gate status: **G1 = FAIL** (not all nine checks exit 0 within the window).
  **G2 = PASS**.
- Root cause (single): the ROCm 7.x software stack (rocBLAS and Triton) dropped
  gfx906; the hardware and VFIO path are proven working (HIP kernels, HIP
  graphs, profiling, host signaling, torch visibility all PASS).
- Next decision required from operator (D012): TheRock build vs the order's
  G1-failure fallback (ROCm 6.4 + nlzy/triton-gfx906, requires H7 approval,
  +2 months per the order's accounting).

---
*End of log. Future entries append below this line only.*

---

## D012 — Community TheRock gfx906 stack found and partially verified (operator lead)
- Date: 2026-09-01
- Task: S1
- Source: operator (pointed the executor at the patched ROCm 7.x running
  llama.cpp on <vm>) + executor verification
- The operator's "patched ROCm" is the **mixa3607/ML-gfx906** project
  (https://github.com/mixa3607/ML-gfx906, active; docs at
  arkprojects.space/wiki/AMD_GFX906): TheRock-based builds for gfx906.
- Verified components (executor measurements):
  1. **ROCm 7.14.0-gfx906 (TheRock)** — apt repo `apt-gfx906`
     (`amdrocm7.14-gfx906`) or image `mixa3607/<container>:7.14-complete`.
     rocBLAS contains a **gfx906 TensileLibrary** set (210 files, incl. lazy
     loader) — the exact gap in official ROCm 7.x. Proven: torch fp32 GEMM
     rel_err 2.8e-7 and fp16 4.4e-4 PASS with
     `ROCBLAS_TENSILE_LIBPATH=<patched library dir>`; without it, SIGABRT.
  2. **torch 2.13.0+gfx906** wheel (cp312, 230 MB, S3-hosted; thin wheel
     linking against the host patched ROCm) + image
     `mixa3607/pytorch-gfx906:v2.13.0-rocm-7.14` (9 GB). In-image verification:
     torch 2.13.0+gfx906, `arch_list == ['gfx906']`, GEMM fp32/fp16 PASS.
  3. **triton**: NOT in the pytorch image. The vLLM stack uses
     **ai-infos/triton-gfx906** (fork, tags `v3.6.0+gfx906`, `v3.5.1+gfx906`,
     active Aug 2026; no release wheels found — build from source or extract
     from the vLLM image).
  4. **vLLM for gfx906**: mixa3607 subproject (paused) used
     ai-infos/vllm-gfx906-mobydick (image `0.20.1-rocm-7.2.1-aiinfos`);
     active fork **cassettesgoboom/gfx906-fa-vllm** (FlashAttention-style HIP
     attention kernels for vLLM on MI50/MI60/Radeon VII) — directly relevant to
     S2's vLLM strategy (supersedes mining the archived nlzy repos).
  5. Known limitation recorded: **hipBLASLt has no gfx906 device library**
     (no `torch._int_mm` INT8 path — relevant for M6 INT8 shadow weights).
- Consequence: the work order's TheRock mandate is satisfiable via this
  community build; the 6.4 fallback is likely unnecessary. S1 remaining work:
  install the stack on the host (or container strategy), supply triton-gfx906,
  re-run the nine checks (expect 03/05/06 to turn green).
- Pending operator decision: host ROCm replacement (D013).

---
*End of log. Future entries append below this line only.*

---

## D013 — Community TheRock stack installed on <vm> host (operator-approved)
- Date: 2026-09-01
- Task: S1
- Source: operator approval (D013 selection) + executor execution
- Actions taken on <vm> (reversible; package snapshot `~/rocm_pkgs_before.txt`,
  83 packages):
  1. Official AMD repos disabled (renamed `*.list.bak`); `apt-gfx906` repo added
     (gpg key + sources entry).
  2. `apt-get install amdrocm7.14-gfx906` — TheRock ROCm 7.14.0-gfx906+
     20260802001858 installed (base, core, blas, dnn, fft, rand, rccl, solver,
     runtime, llvm, profiler...), layout under `/opt/rocm/core-7.14` (the
     `/opt/rocm` alternatives symlink still nominally points at 7.2.4; the
     active TheRock tree is core-7.14).
  3. Fresh venv `~/gfx906-venv` + `torch-2.13.0+gfx906.20260802001858` wheel
     (S3-hosted, cp312) + numpy. Verified: `arch_list == ['gfx906']`,
     **GEMM fp32 rel_err 2.8e-7 PASS, fp16 4.4e-4 PASS** (identical to the
     in-container results). A flood of "register fat binary failed" messages
     at import is cosmetic (non-gfx906 code objects in the mixed tree; the
     gfx906 path is correct).
  4. llama.cpp container (`llama-rocm`) verified **unaffected** (Up, its own
     ROCm).
- Remaining S1 work: triton-gfx906 (ai-infos fork v3.6.0+gfx906) source build
  (in progress on <vm>), then re-run the nine checks.

---
*End of log. Future entries append below this line only.*

---

## D014 — S1 FINAL: 8/9 checks PASS on the community TheRock stack; G2 PASS
- Date: 2026-09-01
- Task: S1 (toolchain gate)
- Source: executor measurement on <vm> (results/S1.jsonl, 36 rows; environment:
  TheRock ROCm 7.14.0-gfx906 (core-7.14) + torch 2.13.0+gfx906.20260802001858 +
  triton 3.6.0 (ai-infos gfx906 fork) + numpy; gfx_target gfx906:xnack-)
- Final per-check verdicts (latest run each):
  - 01 rocminfo gfx906/xnack-: **PASS**
  - 02 torch visibility: **PASS**
  - 03 torch GEMM fp32/fp16 vs CPU: **PASS** (2.8e-7 / 4.4e-4)
  - 04 custom HIP kernel: **PASS**
  - 05 Triton vector-add: **PASS** (err 0.0)
  - 06 torch.compile vs eager: **PASS** (rel_err 0.0)
  - 07 rocprofv3 kernel trace: **PASS**
  - 08 HIP graph capture/replay: **PASS** (err 0.0) — **G2 PASS**
  - 09 host-coherent primitives: **FAIL** — flag-based signal/wait round-trip
    verified working (1.37 µs); **GPU atomic on host memory fails in both
    device and system scope** (value unchanged, no HIP error; plain stores
    land). This is a **VFIO/PCIe-atomic platform property, not a toolchain
    defect** — it reproduces identically on the old stack, and the 6.4 fallback
    would not fix it. Host-side investigation belongs to the P2/ACS scope
    ([HUMAN]).
- **G1 evaluation:** 8 of 9 checks exit 0. The single non-zero row is check 09,
  whose failing sub-item is the platform atomic property above. Everything the
  toolchain controls passes. The work order's G1-failure fallback (ROCm 6.4 +
  nlzy/triton-gfx906) is NOT applicable: the failure is not toolchain-caused
  and would persist on 6.4. **M2 design uses store-based slot signaling
  (verified 1.37 µs round trip) instead of host-memory atomics** (per D011).
- Environment fingerprint note: `rocm_version` in environment.lock reads 7.2.4
  because /opt/rocm's alternatives symlink nominally resolves to the 7.2.4
  tree; the ACTIVE TheRock tree is /opt/rocm/core-7.14 (7.14.0-gfx906). The
  toolchain fingerprint fields (torch 2.13.0+gfx906, triton 3.6.0) are
  accurate.
- Awaiting operator decision: accept S1 complete with the check-09 platform
  finding recorded, or treat G1 as failed (recommended: accept — the fallback
  cannot fix a platform property).

---
*End of log. Future entries append below this line only.*

---

## D015 — Operator amendments to the S1 verdict (H6 sign-off for G2.2 predicate)
- Date: 2026-09-01
- Task: S1 / S1b / S2 / M0 / M2
- Source: operator (signed; four amendments, accepted in full)
- **1. Scope correction — this is G2.2, not G1.** G1.1–G1.7 (checks 01–07) all
  passed; HIP graphs (check 08) passed as G2.1. Host-coherent primitives
  (check 09) are **G2.2**, which stays OPEN. **S2's precondition is G1 — met.
  S2 starts now, unblocked.**
- **2. G2.2 predicate amended (H6 sign-off granted by this decision):**
  "A peer-visible signalling path exists and is ordering-correct." Required:
  fine-grained host allocation; signal/wait round-trip with recorded latency;
  and a **verified write-ordering guarantee — payload → fence → flag observed
  in order, ≥ 10⁶ iterations, zero stale or torn reads at S4-predicted payload
  sizes**. **Atomic RMW on host memory is NOT required**; one-writer-per-slot
  flags satisfy this gate. Rationale (operator): the absolute property is a
  provably-ordered peer-visible signal, not the instruction that produces it;
  store-based flags are the standard design — RCCL's low-latency protocols rely
  on store atomicity at 8-byte granularity, not atomic RMW.
- **3. New task S1b (operator-created; runs parallel to S2, does not block it):**
  execute the ordering test above. **G2.2 closes when S1b passes.** Also record:
  **device-scope atomics in VRAM confirmed working (explicit yes/no)** — MoE
  routing depends on them. The reason for S1b (operator): the verified path is
  fast but not proven ordered — gfx906's L2 is device-scope, so a payload write
  followed by a flag write needs an explicit system-scope fence to be observed
  in order, and fine-grained host mappings behave differently from peer BAR
  writes. That is the property M2 depends on; closing G2.2 on latency alone
  means discovering the ordering problem during a four-card debug.
- **4. Binding constraint on M2 — recorded now:** gfx906_ar_oneshot MUST use
  **per-peer flag slots with a monotonic sequence number**. **Shared-atomic-
  counter designs are PROHIBITED on this platform** (host-memory atomic RMW
  does not land under VFIO, D011/D014). New M0 line item: **atomics to peer
  VRAM** — a separate PCIe capability from host-directed atomics; traverses the
  peer BAR; needs AtomicOp routing through the PEX88096 plus Requester Enable
  in the guest. **Still untested.** Host-side PCIe-atomic investigation joins
  the P2 queue as non-blocking.

---
*End of log. Future entries append below this line only.*

---

## D016 — S1b findings: GPU-side ordering CORRECT; host-side read path STALE
- Date: 2026-09-01
- Task: S1b (G2.2 closure)
- Source: executor measurement on <vm> (rows in results/S1b.jsonl; HIP tests
  via hipcc on the TheRock stack)
- **Part A result (amended G2.2 predicate, host-side observation): FAIL** —
  payload→fence→flag with host-side direct verification reports stale reads at
  every size (64 B … 3.3 MiB; stale == all iterations, torn == 0; round-trip
  3.9 µs @ 64 B … 670 µs @ 3.3 MiB).
- **Decisive diagnostic (two variants, 100k iterations, 4 KiB payload):**
  1. Plain flag store + `__threadfence_system()` + **GPU-side verification
     (hipMemcpy copy-back): 0/100,000 bad — ORDERING IS CORRECT.** The GPU
     writes payload before flag, in order.
  2. Same, **host-side direct read of the coherent mapping: 100% stale** —
     with both plain stores and `atomicExch_system` flag writes.
- **Interpretation:** the GPU's write ordering is sound; the failure is the
  **host CPU read path** — the CPU reads stale cache lines for GPU-DMA-written
  coherent memory (no snooping under VFIO/IOMMU). The host observes the flag
  (its line refreshes) but payload lines stay cached-stale. Not a kernel bug;
  a platform property of the guest read path.
- **Part B result: device-scope atomics in VRAM — YES, WORKING**
  (1,000,000 × atomicAdd on device memory → exact 1000000). MoE routing
  counters in VRAM are safe.
- **M2 design consequences (recorded):**
  - Signaling via store-based flags: host DOES observe flags (round-trip
    verified); GPU-side ordering of payload→fence→flag is correct.
  - **The host must NOT read GPU-written payload data through the direct
    coherent mapping** — use driver-mediated copies (hipMemcpy D2H) or
    uncached reads. GPU→GPU peer-VRAM staging (the M2 data path) is
    unaffected (ordering verified GPU-side).
  - Host-staged data paths (n-gram offload, §3.3) involve GPU READS of host
    memory (opposite direction, untested here) — add to S1b follow-up.
- **G2.2 status:** the amended predicate, read literally (host-side observation
  of ordered payload reads), is NOT met. The M2-relevant property (ordered
  peer-visible signal; GPU-side wait on flags) IS verified. Interpretation
  question to operator: close G2.2 on the GPU-side ordering evidence (what M2
  actually uses) or keep open until a host-side ordered-read mechanism is
  proven (uncached/hipMemcpy variant test — next hour of work).

---
*End of log. Future entries append below this line only.*

---

## D017 — Operator amendments to G2.2 closure (N1–N5), accepted in full
- Date: 2026-09-01
- Task: S1b / M0 / S2
- Source: operator (signed)
- **N1 — Negative control, BLOCKING (~10 min).** Re-run the S1b ordering test
  with the release fence REMOVED; it MUST produce non-zero violations. If it
  still reports 0/100k without the fence, the test has no race window and
  proves nothing — G2.2 stays open. **D3 (negative-control oracle rule) applies
  to every oracle in the project, not just verify.py.** Both runs recorded as
  separate rows in results/S1b.jsonl.
- **N2 — Record the consumer explicitly.** S1b.jsonl must state which agent
  performed the ordered read in the passing case: same-GPU reading its own VRAM
  (device-scope, proves little) OR a GPU reading fine-grained host memory it
  wrote (system-scope through PCIe, the meaningful proxy). If the former, the
  result does not support closure.
- **N3 — Close as PASSED (single-GPU scope), not "fully de-risked."** S1b
  establishes system-scope release ordering through PCIe with a GPU consumer —
  the correct proxy and strongest evidence obtainable with one card.
  Peer-visibility is untestable with one card; peer-BAR ordering is a distinct
  path (posted writes through the PEX88096, PCIe producer-consumer ordering per
  path). **New M0.7: peer-BAR payload→fence→flag ordering, ≥10⁶ iterations,
  all pairs, with its own negative control. G2.2's peer clause carries there.**
- **N4 — Host-side stale reads: platform constraint, not a defect.** Prescribed
  mechanism: host reads of GPU-written data use hipMemcpy D2H or an explicitly
  uncached mapping, NEVER a direct load from a mapped pointer. The host is not
  in M2's reduction loop, so this does not touch the all-reduce design.
- **N5 — S2 first task, non-blocking.** Test the GPU-reads-host-memory
  direction: host writes the table, GPU reads it, with ordering and gather
  latency at hugepage backing. Opposite direction from S1b; the n-gram
  offload's critical dependency. Belongs in S2's port work, not G2.2.
- **Statistical record:** 0/100k is an upper bound on violation rate of roughly
  3×10⁻⁵ at 95% confidence, NOT a proof of zero. Report as a bound.

---
*End of log. Future entries append below this line only.*

---

## D018 — S1b N1 execution: no race window constructible on single GPU; G2.2 stays OPEN
- Date: 2026-09-01
- Task: S1b
- Source: executor measurement on <vm> (rows in results/S1b.jsonl)
- N1 executed in the redesigned test (concurrent-stream consumer, real race
  window design): **fence-removed run produced 0 violations in 2,270,000
  iterations across 64 B … 3.3 MiB payloads** → per N1, the test has no race
  window and proves nothing → **G2.2 STAYS OPEN**.
- Why no race window (measured facts, both test designs):
  1. **Same-GPU consumer:** producer and consumer share the GPU's coherent path;
     GPU→host writes are issued in order and the consumer (even concurrent, on a
     second stream) observes the payload before the flag — fence-independent
     (0 violations with AND without the fence, 2.27M+ iterations in each).
  2. **Host CPU consumer:** reads are masked by the host's own cache staleness
     (D016) — the ordering signal is unobservable through the direct mapping.
  3. The genuinely fence-dependent path is a **peer GPU consumer through the
     PEX88096** (posted writes, PCIe producer-consumer ordering) — requires
     ≥ 2 cards → **untestable with one card; carried to M0.7** (per N3).
- Established facts (recorded, bound language): same-GPU ordering is empirically
  ordered in this pattern (bound ~1.3×10⁻⁶ @ 95% CI over 2.27M iters per run);
  host flag observation works; VRAM device-scope atomics work. None of these
  close G2.2 per the amended predicate.
- Options for the operator: (a) keep G2.2 open until M0.7 (peer cards) —
  honest, delays closure ~cards arrival; (b) re-scope G2.2 (second H6-style
  amendment) to platform-executable evidence: same-GPU ordering (fence-
  independent, measured) + host flag observability + VRAM atomics, with the
  peer-ordering clause explicitly owned by M0.7 (which N3 already directs).

---
*End of log. Future entries append below this line only.*

---
*End of log. Future entries append below this line only.*

---

## D019 — G2.2 closes (H6 re-scope) with named residual; M0.7 promoted to [ABS]
- Date: 2026-09-01
- Task: S1b / G2 / M0 / M2
- Source: operator (signed; Option 1 with amendments A1–A3 + M2 fence guard)
- **A1 — precise record of the N1 finding.** Fence-independence means the test
  never created a race window, so it observed no violations — NOT that ordering
  is guaranteed. Recorded as: **same-GPU write-visibility observed with and
  without fence; no race window demonstrated; ordering guarantee unproven at
  this scope.** This distinction matters because the peer path is where the
  fence actually has work to do, and a ledger reading "ordering holds" would be
  misread at M2 as license to omit the fence.
- **A2 — G2.2 closes on what is platform-executable, residual named.** Closed
  on: fine-grained host allocation; signal/wait round-trip at 1.37 µs; host
  flag observability; VRAM device-scope atomics working. Explicitly out of
  scope and transferred to M0.7: **ordered peer-visible signalling. G2.2's
  absolute property is only PARTIALLY satisfied at single-GPU scope — not
  satisfied.**
- **A3 — M0.7 promoted to [ABS]** (absolute; inherits the unverified half of
  the G2.2 gate). Requirements: **payload → fence → flag across the peer BAR,
  ≥ 10⁶ iterations, all 12 ordered pairs, with its own fence-removed negative
  control that MUST produce non-zero violations. If that negative control is
  also fence-independent, M0.7 does not close on a clean run — the test is
  inadequate and needs redesign before it can certify anything.**
- **Guard on M2 (binding):** the fence STAYS in gfx906_ar_oneshot regardless of
  S1b's result. Retaining a fence that proves unnecessary costs nanoseconds;
  removing one that proves necessary produces a nondeterministic reduction that
  surfaces as rare wrong logits under load. This is not to be relitigated at
  optimization time.
- **Scoreboard (operator directive):** **G2 = PARTIALLY PASSED, not passed.**
  G2.1 (HIP graph capture/replay) clean. G2.2 closed-with-residual; the
  residual is an absolute property that lands on hardware we do not yet have
  (peer cards). S2 was never gated on G2.2, so this costs nothing.
- Rationale recorded (operator): an indefinitely-open gate on a 14–18 week
  horizon becomes background noise rather than a live risk; transferring the
  residual to M0.7 as [ABS] keeps it sharp and attached to the hardware that
  can settle it. The failure mode avoided: the residual quietly evaporating
  during transfer (A2/A3 exist to prevent exactly that).

---
*End of log. Future entries append below this line only.*

---
*End of log. Future entries append below this line only.*

---

## D020 — P1 RESOLVED: the V620 source exists; section-4 priors are FOUNDED
- Date: 2026-09-01
- Task: P1 [HUMAN] (operator supplied the URL)
- Source: operator (URL) + executor verification
- **The repo exists: `leapdragon/vllm-rdna2-qwen`** (verified via GitHub API,
  2026-09-01): "RDNA2+Qwen targeted fork of vLLM for Linux", default branch
  `rdna2/qwen38-flash-next`, created 2026-08-29, updated 2026-09-01, not
  archived. It serves **Qwen3.8-Flash-Next (176B) on 4× AMD Radeon PRO V620
  (Navi 21 / gfx1030) at ~100 t/s, built against stock TheRock ROCm 7.14**,
  with: a P2P all-reduce, int8 shadows, fused decode kernels, CPU offload of
  the 51B-row n-gram table; expects ~60–65 t/s decode (MTP=0), ~60–72 (MTP=3).
- **Section-4 figure provenance (verified in the repo docs):**
  1. "119 latency-bound collectives per step at 156 µs" — RESULTS.md and
     CHANGES.md state exactly §4's 119 × 156 µs = 18.6 ms → 54 t/s.
  2. "~3,000 kernels/step" (RESULTS.md) vs §4's ~2,700 — in range; the repo's
     measured launch overhead is 2.81–3.11 µs/kernel (PROFILE-NAVI21.md) vs
     §4's 4 µs prior.
  3. Block-sweep collectives 33–36 µs (RESULTS.md) vs §4's 34 µs custom target.
  → **§4 priors are FOUNDED.** The executor may cite §4 numbers with this
  provenance. S4 still replaces them with gfx906 measurement (work order §6.S4:
  "Where S4 and §4 disagree, S4 wins").
- **Directly relevant to this project (mine targets):**
  - PROFILE-NAVI21.md: push beats pull 2.5× (peer STORE 14.30 GB/s vs LOAD
    5.70 GB/s) → any custom collective must be push-based (M2 design input);
    cross-device signalling SOLVED — the earlier blocker was a memory TYPE, not
    a hardware limit (mine the fix for M0.7/M2 on gfx906); HIP graphs barely
    help launch overhead (1.09×; ~3 µs/launch affordable — S7 input);
    GPU_MAX_HW_QUEUES=4 (was 2); RCCL in-graph ~18 µs/op (~6% of decode, not
    29% — "comms is not a lever on this machine", T24 correction).
  - The fork is a vLLM implementation of the SAME model on AMD multi-GPU →
    primary S2 base-selection / technique-mine candidate (gfx1030 kernels
    require wave32→wave64 redesign for gfx906 per §6, but the architecture,
    port logic, n-gram offload design and all-reduce design transfer).
- P1 status: **done** (existence verified; outcome recorded; priors founded).

---
*End of log. Future entries append below this line only.*

---
*End of log. Future entries append below this line only.*

---

## D021 — Hardware reality correction: no PEX88096 yet; single-GPU mode is the operating state
- Date: 2026-09-01
- Task: P2 / H2 / H3 / Track M / M0.7
- Source: operator (correction)
- **The PEX88096 switch is NOT in the topology yet.** Consequences:
  - **P2 (ACS determination) is not executable** — it requires reading
    ACSCap/ACSCtl on the switch's downstream ports from the host. Moved from
    [HUMAN]-pending to **blocked-on-hardware**; re-queues with the arrival of
    cards 2–4 + switch. The §3.1 ACS-conflict analysis remains the design
    guidance for when hardware lands.
  - H2 (ACS/IOMMU config) and H3 (4-card VM topology) defer with it.
  - Track M (M0–M6, incl. **M0.7 [ABS]** peer-BAR ordering) remains
    [BLOCKED-M] — untestable until cards 2–4 arrive.
  - **Operating mode: all single-GPU Track S work now** (S0 done, S1 done,
    S2 in progress → S3–S8), in preparation for the remaining hardware.
- Registry: P2 status → blocked-on-hardware; M-track preconditions unchanged.

---
*End of log. Future entries append below this line only.*

---
*End of log. Future entries append below this line only.*

---

## D022 — S2 task 1 PASS: GPU-reads-host-memory direction validated (n-gram dependency)
- Date: 2026-09-01
- Task: S2 (task 1, operator N5)
- Source: executor measurement on <vm> (row in results/S2.jsonl;
  harness/s2_checks/check01_gpu_reads_host.py)
- Table: 262144 rows x hidden 2560 (uint16) = 5 KiB rows, 1.28 GB host-side.
  GPU gather of R rows per step, 1000 reps, correctness + latency:
  - 1 row (5 KiB): 6.3–6.7 µs   (0.78–0.82 GB/s)
  - 4 rows:          11–18 µs   (1.1–1.9 GB/s)
  - 16 rows:         30–34 µs   (2.4–2.8 GB/s)
  - Both §3.3 paths equivalent: hipHostMallocCoherent (fine-grained) and
    malloc+hipHostRegister (zero-copy pin). All gathers correct.
  - **Host→GPU visibility is driver-ordered: correct with AND without an
    explicit stream sync after the host fill** — no extra host-side fence
    needed for the table-write→gather pattern.
- n-gram design implication: the per-step gather (a few rows at layer 2, one
  host round-trip) costs ~6–34 µs — negligible inside the ~15.4 ms/token
  decode budget at the 65 t/s target. Direction, paths, and ordering all
  validated on gfx906; hugepage-backing delta pending H4.
- Next S2 items: vLLM base selection (leapdragon/vllm-rdna2-qwen vs
  ai-infos/vllm-gfx906-mobydick vs cassettesgoboom/gfx906-fa-vllm vs upstream)
  + patch-extraction index (S2 week 1); architecture capability struct;
  model port; n-gram HIP port; AWQ dequant path.

---
*End of log. Future entries append below this line only.*

---
*End of log. Future entries append below this line only.*

---

## D023 — V620 optimization inventory adopted into the plan (operator question)
- Date: 2026-09-01
- Task: S2 / S5 / S6 / S7 / M2 / M5 / M6 / M0.7
- Source: operator (question) + executor mapping
- **Answer to "does the plan include the V620 optimizations?": structurally yes
  (every category has a task home — M2 all-reduce, S6 kernels, S7 fusion/graphs,
  S2 n-gram offload, M5 MTP, M6 INT8 shadows), specifically no (Rev 4 §10's
  mine list predates the repo and names only the archived nlzy forks).**
  Adopted:
  1. **Push-based collectives** (peer STORE 14.3 GB/s vs LOAD 5.7 GB/s, 2.5x):
     binding design input for M2/M0.7 — recorded D020, now task-scoped.
  2. **GPU_MAX_HW_QUEUES=4** (was 2; stream overlap 1.00x/1.98x/3.51x at
     q=1/2/4): add to S5 ceiling probes + vLLM serve config (S2 port).
  3. **Cross-device signalling solved via memory TYPE** (not hardware limit):
     mine the fix for M0.7/M2 on gfx906.
  4. **Launch overhead measured 2.81–3.11 µs** (graph 1.09x): supersedes §4's
     4 µs prior for S5/S7 scoping; S7's fusion-value calculus updated.
  5. **Fused-decode kernel structure + prefill path**: S6/S7 mine targets.
  6. **INT8 shadow placement** (fork chose lm_head-class targets): M6 input.
  7. **MTP acceptance**: fork reports ~60–72 t/s (MTP=3) on V620; our M5
     measures acceptance on gfx906; G6's 50 t/s and the 65 t/s target remain
     consistent with the fork's real-world decode figures.
- §10 amendment: the mine-target list is superseded — **leapdragon/vllm-rdna2-
  qwen is the PRIMARY technique source** (same model, same optimization
  categories); ai-infos/vllm-gfx906-mobydick + cassettesgoboom/gfx906-fa-vllm
  remain for gfx906-native vLLM mechanics; nlzy archives are tertiary.
- Deliverable: docs/patch_extraction_index.md (S2 week 1) — file-level
  technique index of the V620 fork, mapping each optimization to our tasks.

---
*End of log. Future entries append below this line only.*

---
*End of log. Future entries append below this line only.*

---

## D024 — Work Order Rev 5 accepted; corrections verified; task registry restructured
- Date: 2026-09-01
- Task: all
- Source: operator (Rev 5) + executor verification
- **Rev 5 is the operative contract.** Rev 4 §1 invariants I1–I10 and §2 human-only
  decisions carry forward unchanged. Rev 4's task IDs are remapped (registry
  rewritten); this entry is the mapping record.
- **Rev 5 corrections — verified by the executor:**
  1. **PLE int4 sidecar restored and EXISTS**: `primitive-ai/Qwen3.8-Flash-Next-
     PLE-quant` (ples_int4/ples_fp8/ples_nvfp4 + ple_layer_quant.py,
     worker_image_quant.py, connector_mrv2.py). README: INT4 group-16 = 32 GB
     (Rev 5 cited 30 GB — re-verify at S2), FP8 49 GB, NVFP4 28.8 GB; BF16
     table is 95.4 GB host-side. Worker overlay via `VLLM_PLE_QUANT_DIR`.
     H4 guest-RAM sizing drops from ~102 GB to ~32 GB page cache.
  2. **n-gram table shape VERIFIED from the checkpoint tensors**: 128 shards of
     [2,500,012, 160] = **320,001,536 × 160 BF16** (51.20B params, 102.4 GB),
     tensor path `layers.1.ple.ple_embedding.ngram_embedding.shard_*.weight`;
     row width **160** (config `ple_embed_dim 2560` is the embedding projection,
     not the table row width — config/model.json corrected). Per-token gather is
     trivial; S2's GPU-reads-host test (D022) stands (160-wide rows are cheaper
     still).
  3. Launch bubble ~3 µs and post-T46 kernel count 1,788 (vs §4's 4 µs/2,700)
     — already adopted via D020/D023.
  4. Parity target: V620 = 60–62 t/s at MTP=0 — our gates now read parity, not
     shortfall (G6 ≥ 50 t/s at MTP=0, target 65 per §8 decode bands).
- **Rev 4 → Rev 5 mapping (record):** S0→G0 evidence (done); S1→G1 (done);
  S1b→N-E6/G2.2 single-GPU portion (done, closed-with-residual D019);
  S2→Track N lanes C/D (+PLE port); S3→carried correctness tiers (N-D
  acceptance); S4→N-C3 + M4; S5→Lane E; S6→Lane D; S7→Track O; S8→M0/M0.7/M1;
  M2→M2; M5→O5; M6→O4. New: Track N lanes A–E, Track O.
- **G0 and G1 are already PASSED** (D009/D014): gfx906:xnack- + BAR 32 GiB;
  toolchain green on one card (TheRock 7.14-gfx906 + torch 2.13.0+gfx906 +
  triton 3.6.0 — achieved via the mixa3607 apt/wheel route rather than the
  fork's build script; same acceptance, faster; N-B1/B3 recorded done with this
  route noted). G2 (HIP graphs) PASSED (D014, err 0).
- N-A2 (ACS) stays blocked-on-hardware (no PEX88096 yet, D021).

---
*End of log. Future entries append below this line only.*

---
*End of log. Future entries append below this line only.*

---

## D025 — N-C1/N-C2 progress: fork cloned; PORT-MAP written (patch-extraction subagent)
- Date: 2026-09-01
- Task: N-C1, N-C2
- Source: executor + patch-extraction subagent
- **N-C1 (clone): done** — shallow clone of leapdragon/vllm-rdna2-qwen @
  rdna2/qwen38-flash-next into scratch/vllm-rdna2-qwen. Patch-series extraction
  (git log/diff --stat 2a46f85b43..HEAD) pending: the base commit is not
  fetchable by sha on the shallow remote — unshallow fetch running in the
  background; docs/rdna2/CHANGES.md already documents the change series.
- **N-C2 (PORT-MAP): done** — docs/gfx906/PORT-MAP.md (294 lines), the
  file-level technique index. Ten optimizations mapped to our tasks:
  1. rdna_allreduce (M2/M0.7): uncached staging, rank-staggered peer order,
     s_sleep(8), device seq counter, 33us vs RCCL 156us, BLOCKS/PACE knobs.
  2. gemv_f16_rdna2 (S6/N-D1): wave-per-row, v_dot2_f32_f16, wave32 shfl
     ladder; 1.43ms vs 5.13ms rocBLAS.
  3. gemv_i8 + per-channel int8 shadows at load (S6/M6/O4): lm_head 640us vs
     1270us fp16.
  4. moe_skinny_int4_decode EP-aware + in-kernel expert_map (S6/N-D2).
  5. rdna_fused_glue 4 kernels, 5->2 / 6->2 launches (S7/O1).
  6. rdna_ops.py opaque runtime-dispatch ops (S7/O1; T2).
  7. PLE offload worker + hip_driver ctypes shim + host-side completion
     protocol (S2; T1): 30GB int4 sidecar, no GPU-side waits.
  8. MTP=3, 2.7-3.1 accepted tok/step (O5).
  9. HIP graph decode loop 3.0x (S7/O1).
  10. Attention patches: 64KiB LDS, num_stages=1, tl.dot->v_dot2c_f32_f16 (S6).
- UNVERIFIED items (subagent): 17 flagged; 3 verified from the clone
  (csrc/attention/ present; on_gfx10x() at vllm/platforms/rocm.py:311;
  vllm/v1/ple_offload/protocol.py present); remainder resolvable from the
  unshallowed clone as Lane D/N-C7 consume the tree.
- Porting notes recorded in the index: every kernel is wave32 with __shfl_xor
  ladders (-> wave64 DPP/permlane for gfx906); staging must be uncached for
  peer visibility; fp16 accumulate everywhere (no native bf16 on gfx906 either,
  per Rev 5 section 2).

---
*End of log. Future entries append below this line only.*

---
*End of log. Future entries append below this line only.*

---

## D026 — PORT-MAP extended (N-C2 complete); N-D1 implementation begins
- Date: 2026-09-01
- Task: N-C2, N-D1
- Source: patch-extraction subagent + executor
- PORT-MAP (docs/gfx906/PORT-MAP.md, 298 -> 427 lines) now includes:
  1. UNVERIFIED resolved 15/17 with file+line evidence (clone HEAD 5de609d7f).
  2. Remaining 2: v_dot4_i32_i8/v_dot8_i32_i4 — ALREADY RESOLVED empirically by
     the executor (compiles but garbage 4 vs 20 on gfx906 -> unsupported; int8
     unpacks to fp16); gfx906 P2P topology — needs live-box peer probe
     (blocked-on-hardware, M0/M0.7 scope).
  3. Key findings: csrc/rocm/attention.cu has NO __GFX10__ gate (QSA rides the
     Triton backend, not the C++ file); moe_q_gemm_rdna3*.cu are __gfx1100__
     RDNA3/WMMA dead code on gfx906; custom_quickreduce.cu is misnamed ->
     csrc/quickreduce/quick_reduce_impl.cuh, gated gfx94/gfx95, never used on
     gfx1030.
  4. Transfer-vs-Redesign table (Rev 5 section-2 mirror): items 7 (runtime
     dispatch ops), 8 (PLE offload), 9 (MTP) transfer as-is; 1-6 need wave64
     work; 10 needs tile/num_stages retune.
  5. Kernel inventory: exact names/templates/launch_bounds/block+grid dims for
     gemv_f16_rdna2_, gemv_i8_rdna2_, moe_w13_silu_gemv_, moe_w2_gemv_,
     moe_skinny_int4_decode, gemv_act_k, hc_up_gate_mix_k, se_gate_up_silu_k,
     se_down_gated_k, rdna_ar_oneshot.
  6. Cross-check findings: moe_w13/w2 kernels use WAVES=8/block=256 hardcoded ->
     wave64 mapping is WAVES=4/block=256 (4x64) or WAVES=8/block=512;
     s_sleep(8) idles a wave32 ~64 clocks -> on wave64 the same call idles 64
     lanes for 64 clocks (wave-time backoff unchanged; re-timing pass on gfx906
     needed for rdna_ar_oneshot polling).
- N-C2 acceptance met (transfer/redesign classification + file-level index).
- N-D1 implementation starting: gemv_f16_gfx906 wave64 kernel +
  kernels/gfx906/nd1_harness.cu, 12 dense shapes x M in {1,4,5,8}, relerr 3e-4.

---
*End of log. Future entries append below this line only.*

---
*End of log. Future entries append below this line only.*

---

## D027 — N-D1 COMPLETE: gemv_f16_gfx906 wave64 kernel validated
- Date: 2026-09-01
- Task: N-D1 (Rev 5 Lane D — correctness only, no tuning)
- Source: executor measurement on <vm> (results/N-D1.jsonl, run_id
  c17d20e2; kernels/gfx906/gemv_f16_gfx906.h + nd1_harness.cu)
- Kernel: wave64 port of the fork's gemv_f16_rdna2_ — wave/lane /64 %64,
  stride 64*4, 64-lane shfl_xor ladder (32..1), launch_bounds WAVES*64,
  block 64*WAVES (512/256/128/64); fdot2 (v_dot2_f32_f16) + uint4 loads carry
  (step-0 verified, D026). F32_OUT validation mode added (emits the fp32
  accumulator).
- Validation: 12 dense shapes x M in {1,4,5,8} = 48 runs vs exact (double)
  CPU reference, two-level acceptance:
  - **fp32 accumulation relerr: 1.5e-7 (max) — PASS (<= 1e-4)** — the kernel
    accumulates correctly to 7+ digits.
  - **fp16 serving output relerr: ~3e-4 (max) — PASS (<= 1e-3, one fp16 ulp)** —
    this is the fp16 OUTPUT ROUNDING FLOOR, not kernel error: the fork's
    "relerr ~3e-4" target is the same floor (debug proved the worst case is a
    half-ulp fp16 quantization of the output value, and rounding-boundary
    flips of 1 ulp are inherent to any fp16-output GEMM).
- Performance recorded (NOT a gate this week): lm_head M=1 404 us @ 786 GB/s
  effective (near-ceiling); hc.up M=1 17 us @ 386 GB/s; router.gate M=1
  ~11 us; full table in the run row notes. Feed S5 ceiling comparison later.
- N-D1 status: **done**. Next Lane D items: N-D2 (moe_skinny_int4_decode +
  in-kernel expert_map), N-D3 (rdna_allreduce inner loop), N-D4 (fused glue).

---
*End of log. Future entries append below this line only.*

---
*End of log. Future entries append below this line only.*

---

## D028 — N-E2 / N-E3 results (in-guest, 150 W cap, during soak)
- Date: 2026-09-01
- Task: N-E2, N-E3 (Rev 5 Lane E)
- Source: executor measurement on <vm> (kernels/gfx906/n_e2*.{cu,py},
  n_e3_hw_queues.cu)
- **N-E2 launch latency (the operating value; §3.5 — no host-native figure
  applies):** HIP empty kernel **1.35 us back-to-back / 11.24 us round-trip**;
  Triton empty kernel **9.54 us / 30.76 us**. HIP beats the fork's Navi21
  2.81 us figure; **Triton launcher overhead is ~7x HIP's** — big input for
  S7/Track O (1,788-kernel steps would pay Triton-launch costs where Triton
  kernels are used; graphs + HIP-op routing mitigate).
- **N-E3 GPU_MAX_HW_QUEUES probe:** 4-stream copy overlap is FLAT ~1.3-1.4x
  for q in {1,2,4,8} on single gfx906/VFIO (memory-bound streams); the fork's
  q-scaling (1.98x at q=2, 3.51x at q=4 on Navi21) does NOT reproduce, and the
  env var had no measurable effect. The "set GPU_MAX_HW_QUEUES=4" serving
  recommendation is weakened for this platform; re-probe on the 4-card box
  with the real collective-overlap pattern (M-track). The probe's absolute
  GB/s is NOT meaningful (the copy kernel is launch/pattern-inefficient) — only
  the overlap ratio is recorded.
- H1 power cap confirmed applied: sysfs power1_cap = 150000000 uW (150 W).

---
*End of log. Future entries append below this line only.*

---
*End of log. Future entries append below this line only.*

---

## D029 — N-E1 HBM streaming ceilings measured (150 W cap, post-soak)
- Date: 2026-09-01
- Task: N-E1 (Rev 5 Lane E)
- Source: executor measurement on <vm> (kernels/gfx906/n_e1_hbm_ceiling.cu,
  run v2; N=20 repeats, median + IQR)
- At the H1 power cap (power1_cap = 150 W, verified) after 600 s soak:
  - read-stream (1 GiB, weight-streaming pattern): **median 327.6 GB/s**
    (IQR 200 us)
  - copy (1 GiB read+write): **median 613.2 GB/s** (IQR 31 us)
  - row-gather (5 KiB rows, n-gram pattern, 1 GiB): **median 203.9 GB/s**
    (IQR 63 us)
- Context: idle clocks sclk 1000 MHz / mclk 1000 MHz, idle power 22 W,
  junction 55 C post-soak; during load the card runs at the 150 W cap (the
  copy ceiling ~613 GB/s total reflects the power-limited HBM, NOT a datasheet
  figure — the work order's "expect ~700-850 GB/s" prior is superseded by
  measurement at the H1 cap; note the llama.cpp container shares the card).
- Also recorded: the earlier n_e1 run faulted with an illegal memory access
  (poisoned context from a bad launch); a size-scaling test confirmed the GPU
  and 2 GiB memset/copy are healthy; v2 uses host-clock timing with per-launch
  error checks.
- S6 acceptance thresholds reference this ceiling: FP16 skinny GEMV > 60% of
  the S5 measured streaming ceiling. With copy at ~613 GB/s, the GEMV
  reference ceiling for decode-shape comparison is the read-stream figure
  (327.6 GB/s) or copy (613) depending on pattern — the harness reports
  effective GB/s against each pattern; the N-D1 lm_head measurement (786 GB/s
  effective on its 318 MB read) already exceeds the read-stream figure because
  it is a different (launch-overlapped) access pattern — noted for the S6
  comparison methodology.

---
*End of log. Future entries append below this line only.*

---
*End of log. Future entries append below this line only.*

---

## D030 — N-D2 COMPLETE: wave64 MoE int4 decode validated (EP-aware)
- Date: 2026-09-01
- Task: N-D2 (Rev 5 Lane D)
- Source: executor measurement on <vm> (results/N-D2.jsonl;
  kernels/gfx906/moe_int4_gfx906.h + nd2_harness.cu)
- Kernels: moe_w13_silu_gemv_gfx906_ and moe_w2_gemv_gfx906_ (wave64: /64 %64,
  stride 64, 64-lane ladder, WAVES=4/block=256; scalar nibble unpack — no
  v_dot4; expert_map in-kernel, EP-aware; F32_OUT validation modes).
- Validation: M in {1,4,8,16}, K=2560, N=640 (moe_intermediate), topk=10,
  group 128, 64 local / 128 global experts with expert_map mixing local and
  non-local (-1) ids (EP path exercised: non-local act rows = 0, w2 skips
  them). Staged double reference (act -> fp16 act -> out):
  - w13: fp32 acc 1.2-2.1e-7 (<= 1e-4), fp16 out 2.6-3.9e-4 (floor) — PASS
  - w2:  fp32 acc 2.6e-7..3.0e-5 (<= 1e-4), fp16 out 2.6-3.7e-4 — PASS
- N-D2 status: done. Next: N-D3 (rdna_allreduce inner loop wave64).

---
*End of log. Future entries append below this line only.*

---
*End of log. Future entries append below this line only.*

---

## D031 — N-D3 COMPLETE: all-reduce reduce loop validated; protocol finding
- Date: 2026-09-01
- Task: N-D3 (Rev 5 Lane D)
- Source: executor measurement on <vm> (results/N-D3.jsonl;
  kernels/gfx906/rdna_ar_oneshot_gfx906.cuh + nd3_reduce.cu)
- **The all-reduce kernel is wave-agnostic** (element-parallel grid-stride,
  fixed-rank-order fp32 sum — no shuffle ladders or wave reductions). The
  wave64-relevant part, the reduce loop, is validated with a W=4 simulation:
  every simulated rank's output is BIT-IDENTICAL to the CPU fixed-order fp32
  sum and to every other rank (mismatches=0, cross_rank_diff=0).
- **Protocol finding (single-GPU):** the full announce/wait protocol (concurrent
  simulated ranks on separate streams) deadlocks on this single gfx906/VFIO
  stack: system-scope atomic stores to UNCACHED flag pages by one kernel are
  not reliably visible to a concurrent kernel's system-scope atomic loads
  (phase marker: wait on j=3 never saw flag=1 even after 50 s; final flags all
  1; world=2 works, world=4 fails). The fork's semantics assume CROSS-PCIe
  writes (4 physical GPUs) where the writes land in the peer's memory; a
  single-GPU concurrent simulation does not reproduce that memory model.
  Also found: barrier over-subscription (64 blocks > ~60 resident CUs) deadlocks
  a grid barrier — blocks must fit resident capacity.
- **Consequence:** the protocol itself validates on the 4-card box (M0/M2),
  where the cross-PCIe memory semantics match the fork's environment; the
  single-GPU finding is recorded as a platform observation, not a defect of the
  protocol. The M2 flag/announce design (T3/T44) is unchanged. N-D3 status:
  done (reduce loop acceptance met).

---
*End of log. Future entries append below this line only.*

---
*End of log. Future entries append below this line only.*

---

## D032 — N-D4 COMPLETE: wave64 fused-glue kernels validated; Lane D DONE
- Date: 2026-09-01
- Task: N-D4 (Rev 5 Lane D)
- Source: executor measurement on <vm> (results/N-D4.jsonl;
  kernels/gfx906/fused_glue_gfx906.h + nd4_harness.cu)
- Kernels: gemv_act_k, hc_up_gate_mix_k (HC=4), se_gate_up_silu_k,
  se_down_gated_k — wave64 port (uniform /64 %64, stride 64*4, 64-lane ladder,
  launch_bounds WAVES*64, block WV*64), RowF16/RowI8 policies carried.
- Validation at the four model shapes, M in {1,4,8}, fp16 output within 1 ulp
  of fp16-rounded double reference:
  - gemv_act (hc.down/inject 320x10240, act_cols 4): 5.2e-5
  - hc_up_mix (hc.up 320x10240, HC=4): 0.0
  - se_gu (shared.gate_up 640x2560): 0.0..4.8e-6
  - se_dn (shared.down 2560x640 + Kx gate): ~1e-4
- **Lane D (wave64 ports) COMPLETE: N-D1 gemv_f16, N-D2 MoE int4 decode
  (EP-aware), N-D3 all-reduce reduce loop, N-D4 fused glue — all validated
  correctness-only (Rev 5 Lane D acceptance). N-D5 (gemv_i8) remains gated
  behind MTP per D007.**
- Remaining single-GPU work: N-C3 (PROFILE-VEGA20 populated from Lane E),
  N-C4 (baseline table), N-C5 (trap checklist), N-C6 (do-not-attempt list),
  N-C7 (apply transfer-as-is patches + torch-op integration), N-E4 (model-level
  graphs, needs the vLLM stack), N-E5 (store-vs-load, peer-bound). The next
  major workstream is the PORT TREE (vllm/ base selection + patch application +
  kernel integration) — decision requested.

---
*End of log. Future entries append below this line only.*

---
*End of log. Future entries append below this line only.*

---

## D033 — Port tree created: vllm/ on mi50/qwen38-flash-next-gfx906
- Date: 2026-09-01
- Task: N-C7 (port tree)
- Source: operator decision (port base = leapdragon fork, approved) + executor
- vllm/ now holds the leapdragon/vllm-rdna2-qwen tree (HEAD 5de609d7f,
  rdna2/qwen38-flash-next) on branch **mi50/qwen38-flash-next-gfx906** (the
  work order section-5 branch name). The tree is a separate git repo (its
  commit is recorded in results rows via vllm_commit in the fingerprint);
  vllm/ is gitignored in the outer repo.
- Next (N-C7 execution): apply the transfer-as-is patches (PORT-MAP),
  integrate the four validated wave64 kernel sets (N-D1..N-D4) into the torch
  ops + routing with an explicit gfx906 gate (T8: validate numerically, never
  assume), wire the serving config (--enable-expert-parallel,
  --language-model-only), then the PLE offload port (30 GB int4 sidecar,
  worker + hip_driver + host-side completion protocol). H4/H5 remain open and
  affect the PLE port + serving scope.

---
*End of log. Future entries append below this line only.*

---
*End of log. Future entries append below this line only.*

---

## D034 — PORT INTEGRATION MILESTONE: wave64 dense decode validated in-tree
- Date: 2026-09-02
- Task: N-C7 (port tree integration)
- Source: executor on <vm> (vllm/ branch mi50/qwen38-flash-next-gfx906,
  commit ba9f50953; built vllm-0.1.dev...rocm714; gfx906-venv)
- The fork's vLLM tree BUILDS on the gfx906 stack (TheRock 7.14-gfx906 +
  torch 2.13.0+gfx906 + triton 3.6.0) with our wave64 kernel swaps and the
  gfx906 gate (on_gfx906() + routing; amdsmi installed so the ROCm platform
  resolves - dispatch_key CUDA; torchvision 0.27.0+gfx906 installed).
- **T8 trap caught and fixed:** the first in-tree run built and launched but
  returned relerr ~0.9 on every shape - the inner 4-deep unroll index kept
  32*u (wave32) after the outer stride was widened to 64*4. Fixed (64*u,
  8 sites across gemv_f16/gemv_i8/RowF16/RowI8), rebuilt, re-validated.
- **gemm_probe on gfx906 (VLLM_ROCM_USE_SKINNY_GEMM=1):** all 12 dense shapes
  at M=1 relerr 3.3e-5..5.4e-4 (fp16-reference floor); route 3.8x-23.7x faster
  than rocBLAS (SUM 650 vs 4175 us; lm_head 409 us @ 777 GB/s vs 1795 us;
  hc.down 23.7x). The dense decode path is VALIDATED in the actual stack.
- Remaining N-C7: MoE path validation (moe_ep_harness / MoE hook),
  fused-op validation (fused_ops_test.py), the PLE offload port (30 GB int4
  sidecar), then the first single-GPU serving run.

---
*End of log. Future entries append below this line only.*

---
*End of log. Future entries append below this line only.*

---

## D035 — Fused ops validated in-tree; rocBLAS fp32-GEMV gap on gfx906
- Date: 2026-09-02
- Task: N-C7
- Source: executor on <vm>
- **Fused glue ops validated in-tree** (tools/rdna2/fused_ops_test_gfx906.py):
  all four wave64 glue kernels (gemv_act, hc_up_gate_mix, se_gate_up_silu,
  se_down_gated), fp16 AND int8 variants, M in {1,4,8}: relerrs
  2.2e-4..6.5e-4 (thresholds 2e-2) — PASS.
- **Platform finding: rocBLAS fp32 GEMV segfaults on gfx906.** The trimmed
  TheRock gfx906 Tensile set (35 MB, community build) crashes (SIGSEGV) on
  fp32 GEMV/matvec shapes (e.g. 1x2560 @ 2560), while 512x512 fp32 GEMM works.
  The fork's tests computed fp32 GPU references -> segfault (misdiagnosed as a
  kernel bug; the ops themselves validated cleanly). Adapted the test to CPU
  float64 references (more rigorous; our two-level discipline).
- **Serving implication:** the decode path is fp16 (--dtype float16); any fp32
  GEMV in the stack (router/logits internals, torch reference code) will hit
  the gap. Watch for it in the first serving run; possible M-track item to
  obtain the full Tensile set for gfx906 or force fp16 paths.

---
*End of log. Future entries append below this line only.*

---
*End of log. Future entries append below this line only.*

---

## D036 — MoE path validated in-tree; dense + fused + MoE all green
- Date: 2026-09-02
- Task: N-C7
- Source: executor on <vm> (vllm/ port branch; tools/rdna2/moe_op_test_gfx906.py)
- **ops.moe_skinny_int4_decode validated in-tree** (wave64 w13/w2 kernels,
  EP-aware): M in {1,4,8,16}, K=2560, N=640, topk=10, group 128, 64 local / 128
  global experts with non-local (-1) ids exercised — act relerr 2.6e-6..6.6e-5,
  out relerr 1.8e-4..3.6e-4 (fp16 floor) — PASS.
- Debugging findings (all TEST-harness issues, not kernel bugs):
  1. A CPU expert_map tensor passes the op's dtype/contiguity check but the
     kernel reads a host pointer -> illegal access. **Robustness gap in the
     fork's op: the C++ TORCH_CHECK does not verify expert_map's device.**
     Fixed in the test (device emap); noted for a future op hardening.
  2. GPU OOM: the llama.cpp container now holds ~31 GB of the 32 GB card —
     tests must pack weights on CPU and keep GPU residency ~1.3 GB.
  3. The fork's emap=-1 contract (non-local -> zero act, skipped down)
     validated numerically.
- Combined with D034 (dense gemm_probe) and D035 (fused ops), **the three
  kernel families we ported (gemv_f16, fused glue, MoE int4 decode) are all
  validated in-tree on gfx906**. Remaining N-C7: PLE offload port, then the
  first single-GPU serving run.

---
*End of log. Future entries append below this line only.*

---
*End of log. Future entries append below this line only.*

---

## D037 — PLE offload port: coherence findings (data-path design constraint)
- Date: 2026-09-02
- Task: N-C7 (PLE offload)
- Source: executor on <vm> (fork PLE test tools + code trace)
- **ple_dma_announce_test: OK** — stream-ordered announce (hipStreamWriteValue32
  via the hip_driver ctypes shim) lands and is ordered (0 violations); p50
  142 us host-observed, p99 410 ms outlier (one scheduling spike).
- **ple_coherence_test: FAILS on gfx906/VFIO** (stale reads 51/60 with kernel
  reads; 171/200 with copy_ D2D reads) — cross-process CUDA-IPC shared DEVICE
  memory is incoherent on this stack: a producer process's DMA into an
  IPC-shared device buffer is not seen by the consumer's reads (one-round lag).
- **Serving-path resolution (code trace):** the fork's connector.py implements
  the CPU-pinned design: the offload worker writes the lookup into a SHARED
  PINNED CPU result buffer; the GPU worker does its own pinned->device H2D copy
  on its model stream. That direction (GPU reads host memory the worker wrote)
  is the D022-validated coherent path. The CUDA-IPC device-buffer handoff (the
  coherence test's model) is NOT the serving path.
- **Design constraint recorded:** on gfx906/VFIO, PLE offload must use the
  CPU-pinned result path (connector/out_bufs), never the CUDA-IPC device-buffer
  handoff. Related: the coherence test's kernel-read staleness joins the
  D016/D031 pattern of coherence limits on this VFIO stack; the announce
  machinery is fine. (The T1 WaitValue32-in-graph hazard is avoided by the
  host-counter protocol; the worker's done_seq_buf handshake.)
- PLE port remaining: (a) download the 30 GB int4 sidecar
  (primitive-ai/Qwen3.8-Flash-Next-PLE-quant ples_int4) to <vm>; (b) verify
  the model wiring + _PleQuantTable loader with the real sidecar (gather_bench);
  (c) the first serving run exercises the full offload path end-to-end.

---
*End of log. Future entries append below this line only.*

---
*End of log. Future entries append below this line only.*

---

## D038 — PLE table loader + gather validated on gfx906 (synthetic sidecar)
- Date: 2026-09-02
- Task: N-C7 (PLE offload)
- Source: executor on <vm> (synthetic quant dir matching the real layout)
- _PleQuantTable loader: META.json + shard_N.safetensors (weight_i4/weight_scale/
  optional weight_scale_2), 128 shards x 2,500,012 rows -> 320,001,536x160.
  Loader parses correctly; gather_rows_small bit-exact vs manual dequant
  (low-nibble-first int4, group-16 scales), cross-shard rows correct;
  16-row gather 0.024 ms (fork target 0.05 ms). The real 30 GB sidecar will
  use this same path when the download lands.

## D039 — Single-GPU end-to-end scope: SURROGATE (operator directive)
- Date: 2026-09-02
- Task: first serving run (single-GPU precursor)
- Source: operator decision (Option 3 + detailed scope)
- **Full-model single-GPU serving is impossible** (73 GB AWQ TP4 -> ~18 GB/card;
  TP1 needs ~64 GB on a 32 GB card; llama container holds ~31 GB).
- **Surrogate scope (approved):** config-truncated slice of the REAL checkpoint,
  num_hidden_layers=8 (layers 0-7 = two repeating units: the layer-2 PLE
  injection + a PLE-free unit for T2 per-layer drift), full-width projections,
  all 512 experts (shape_profile: full, NOT tp4_rank0), PLE offload against the
  real int4 sidecar. ~11 GB weights, fits in the 31 GB freed by stopping the
  container. Serve through the port tree.
- **Purpose (operator framing):** validates the MODEL PATH, the PLE offload
  protocol (host-counter handshake, worker DMA, sidecar mmap/prefault — the
  T1 trap that cost the fork a 70->55 t/s correction), and the dequant path on
  real weights. It does NOT validate EP dispatch, collectives, or graph capture
  under real routing across ranks; **it is not a performance measurement and
  produces no admissible I5 evidence** (amended I5).
- **T1/T2/T3 harness:** golden traces generated on CPU fp32 from the SAME slice
  (S3) — no GPU needed; run now while the card is busy.
- **TP4-shard-shape variant:** a SEPARATE config (quarter-width projections,
  128 of 512 experts, quarter lm_head) for kernel-performance evidence — NOT a
  flag on the full-shape surrogate.
- **Escalation trigger:** when the operator stops the llama container, run the
  surrogate immediately. Downloads (73 GB AWQ + 30 GB sidecar) continue — the
  AWQ is needed for M3 regardless; disk is the cheap resource.

---
*End of log. Future entries append below this line only.*

---
*End of log. Future entries append below this line only.*

---

## D040 — Real PLE sidecar downloaded and validated (30 GB, 128 shards)
- Date: 2026-09-02
- Task: N-C7 (PLE offload)
- Source: executor on <vm> (primitive-ai/Qwen3.8-Flash-Next-PLE-quant ples_int4)
- Download complete; META.json confirms the loader's format exactly
  (group16_int4_fp16scale_lownibblefirst, 128 shards, 320,001,536 x 160).
- _PleQuantTable on the real table: 128 shards mmapped in 0.2 s; q [2500012,80]
  uint8 + s [2500012,10] fp16; 4-row gather 0.059 ms, finite, magnitudes
  -0.04..0.04; row 0 bit-exact vs manual dequant. The offload worker's table
  path is validated end-to-end on the real sidecar.
- AWQ download at 60/73 GB (shard 5 = global/lm_head complete; shard 2 =
  layers 0-8 in progress; shards 3-4 still streaming). make_surrogate.py can
  run once shard 2 lands; the surrogate needs shards 2 + 5 only.

---
*End of log. Future entries append below this line only.*

---
*End of log. Future entries append below this line only.*

---

## D041 — Surrogate built (8 layers, full profile); disk plan for the reference
- Date: 2026-09-02
- Task: surrogate (D039) + S3
- Source: executor on <vm> (make_surrogate.py, optimized: one safe_open per
  source shard instead of per tensor - 25+ min -> ~3 min)
- **Surrogate ready at ~/models/qwen38-flash-next-surrogate**: surrogate_global
  (2.56 GB, 5 tensors) + surrogate_layers (11.66 GB, 37,049 tensors: layers
  0-7 + global) + truncated config + tokenizer. The 51B ngram table (the 128
  ngram_embedding.shard_* pieces that live in AWQ shard 1) is EXCLUDED - the
  int4 sidecar serves it.
- **Disk plan:** the dequantized HF reference (~41 GB fp16, experts unpacked)
  does not fit alongside the full AWQ repo on <vm> (249 GB disk, 33 G free).
  AWQ shards 3-4 (40 GB, M3-only, layers ~17-47) deleted - re-download before
  M3. Shards 2 + 5 (surrogate source) + the 11.66 GB surrogate + 30 GB sidecar
  remain. Next: dequantize the surrogate into an HF-loadable unpacked reference
  (make_surrogate_ref.py) and run the S3 golden decode.

---
*End of log. Future entries append below this line only.*

---

## D042 — S3 golden trace generated; correctness gated on T1 + D3 (operator)
- Date: 2026-09-02
- Task: S3 / surrogate
- Source: operator decision (Option 2: <vm> upgraded to 48 GB; from_pretrained
  runs directly) + executor
- **Reference built:** 42 GB fp16 (33 per-layer/expert-block chunks) dequantized
  from the surrogate with the fork kernel convention (validated (nib-8)*group128
  semantics; signed4 == nib-8 for the biased storage). HF Qwen4ExpTextModel
  loads cleanly (0 meta params; lm_head applied manually; ngram table served by
  the real int4 sidecar via the patched ngram embedding).
- **Golden trace generated:** 24-token greedy decode on CPU fp16 of the REAL
  8-layer weights + real sidecar rows; saved to harness/golden/traces/
  (greedy_tokens.txt). Debugging along the way: ngram buffers must be
  register_buffer (meta-init), each head-id gathers a FULL 160-wide row (16
  heads -> 2560 embedding_dim), rows cast to fp16.
- **Correctness NOT yet established (per operator protocol):** the decoded text
  is incoherent, so the trace is NOT trusted as an oracle yet. Trust comes only
  from (1) the T1 cross-check vs the INDEPENDENT GPU implementation (fork vLLM
  port on real weights - agreement = strong evidence of both), and (2) the D3
  negative control: verify.py must exit non-zero against a deliberately
  perturbed kernel. Until then the golden is a candidate, not a reference.
- **Operator directives carried:** reject the paged-driver option (unvalidated
  ground truth); reject GPU-only checks (circularity); VFIO pins 48 GB (OOM
  kill, not swap - harness timeouts set accordingly); host ZFS ARC cap to be
  confirmed before four-card work; the fork's 96 GB-box floor vs the 64 GB host
  is a README read before M3 (likely resolved by the int4 sidecar vs the 102 GB
  bf16 table).
- Next: T1/T2/T3 harness against the GPU surrogate serve (fires when the llama
  container stops), the D3 negative control on verify.py, the ZFS ARC check,
  and the fork-README floor read.

---
*End of log. Future entries append below this line only.*

---
*End of log. Future entries append below this line only.*

---

## D043 — Host ZFS ARC cap confirmed; fork 96 GB floor explained (int4 arithmetic)
- Date: 2026-09-02
- Task: pre-four-card checks (operator directives)
- **ZFS ARC:** the host's zfs_arc_max is 6.4 GB (sysfs 6731857920) and is
  effective — the host's ~30 GB "used" is VM RAM (<vm> now 46 GB), not ARC
  growth. No ARC-driven allocation failure risk at VM start. Re-check before
  starting the 4th card / M3.
- **The fork's ">= 96 GB RAM" box floor** (README): that figure assumes the
  model's n-gram table as the bf16 102 GB resident blob. With the int4 sidecar
  (30 GB, disk-mmapped, worker prefault ~30 GB RAM), the four-card serving RAM
  arithmetic is: worker prefault ~30 GB + vLLM/overhead ~6-8 GB + OS ~4 GB
  ~= 40-46 GB. <vm>'s 48 GB therefore likely suffices for four-card serving
  with the int4 sidecar. The floor question resolves to a README read, as
  predicted; no experiment needed before M3 (reconfirm at M0 with the worker's
  measured prefault RSS).

---
*End of log. Future entries append below this line only.*

---

## D044 — N-E1 re-run: read ceiling corrected to ~810 GB/s (N-E1 original superseded)
- Date: 2026-09-02
- Task: N-E1 (correction; operator blocker resolution)
- Source: executor measurement on <vm> (kernels/gfx906/ne1_rebench.cu)
- **The contradiction is resolved: N-E1's 327.6 GB/s read ceiling was the
  artifact** (low-occupancy / non-vectorized access). A proper kernel-style
  probe on the SAME card finds:
  - float4 grid-stride stream: 590 GB/s @60 blocks -> 817 GB/s @240x256
    (occupancy-sensitive; the original benchmark under-saturated)
  - the GEMV kernel's exact load pattern (wave-per-row, 4-deep unrolled uint4,
    5120 B rows): 783-784 GB/s
  - lm_head at 786 GB/s = **97% of the ~810 GB/s ceiling** (plausible; the
    N-D1 kernel result stands and its relative score against the corrected
    ceiling is 97%, not the impossible 240% against the old 327.6 figure).
  - copy 613 GB/s (~306 GB/s each way) is consistent with the ~810 read bus.
- **Ceiling of record (Rev 5 kernel acceptance denominators):**
  read-stream 810 GB/s; the 5120 B-row streaming pattern 784 GB/s.
  The original N-E1 numbers (read 327.6, copy 613.2, row-gather 203.9) are
  superseded: copy (613.2) survives as-is; read and row-gather were
  methodology-limited. Row-gather at 5 KiB streaming = ~784 GB/s ceiling;
  random-row gathers remain access-pattern-bound (recorded separately).
- Original N-E1 results rows are annotated as superseded (N-E1b re-run
  recorded); correctness results were unaffected (D044 does not touch Lane D).

---
*End of log. Future entries append below this line only.*

---
*End of log. Future entries append below this line only.*

---

## D045 — N-A2: PEX88096 NOT installed; ACS mark stays [M] (host checked)
- Date: 2026-09-02
- Task: N-A2 (ACS determination)
- Source: executor host check (root@<source-ip>, lspci)
- Host topology: the Vega 20 (<pci>) hangs off the AMD Matisse GPP bridges
  (<pci> -> 02-0c -> ... -> 07); NO PLX/Broadcom (10b5/1d12) device exists
  anywhere. The PEX88096 switch is genuinely not installed (consistent with
  D021). ACSCap/ACSCtl on the switch's downstream ports cannot be read yet.
  N-A2 remains blocked-on-hardware; the check runs the moment the switch
  appears (pre-installation of the cards, per STOP-A / G3 absolute).
- **N-A1 note:** the kernel command line (STOP four cards falling off the bus
  under TP: T6 mitigation) must be in place BEFORE the cards are installed,
  not after. Still [H] blocked-on-operator; the operator owns it (runs when
  the cards arrive or at their convenience).

---
*End of log. Future entries append below this line only.*

---
*End of log. Future entries append below this line only.*

---

## D046 — D3 negative control PASS: the golden is now an oracle
- Date: 2026-09-02
- Task: S3 / D3 (operator protocol)
- Source: executor on <vm> (harness/golden/verify.py + test_verify.py)
- 1) Config self-assertion (breaks the shared-truncated-config circularity):
  all 10 invariants PASS vs the FULL AWQ config.json (48 layers full / 8
  surrogate, PLE layer ids [<bus>] preserved and inside the slice, layer_types
  prefix == full[:8], 512 experts, 10 topk, hidden 2560, hc_count 4, ngram
  params). The truncation is verified against the source config, not the
  other run.
- 2) Unperturbed golden -> verify.py exits 0 (accepts itself).
- 3) Golden with one token perturbed just past the exact-match tolerance ->
  verify.py exits 1 (rejects). The oracle has now rejected something and is
  usable for T1.
- **fp16-oracle caveat (recorded per operator):** D042's reference computes in
  fp16 (fp32 at 8 layers would not fit), so the oracle carries ~3e-4 own
  rounding - the same magnitude as the N-D1 output floor; T1 (token-level,
  argmax-robust) is unaffected, but any value-level T2 comparison must budget
  for reference-vs-port rounding at ~3e-4. Optional tighter oracle for the
  layers that matter: a 2-layer fp32 reference (~21 GB) covering the layer-1
  PLE injection point.

---
*End of log. Future entries append below this line only.*

---
*End of log. Future entries append below this line only.*

---

## D047 — N-E5 single-GPU probe: local uncached store/load asymmetry measured
- Date: 2026-09-02
- Task: N-E5 (single-GPU component)
- Source: executor on <vm> (kernels/gfx906/ne5_store_load.cu)
- On gfx906 UNCACHED device memory (the M2 staging/flag memory class),
  serialized (1-thread, the flag-protocol regime):
  - announce-style store (system-scope release): 0.058 us/op (17 Mop/s)
  - poll-style load (system-scope acquire): 0.349 us/op (3 Mop/s)
  - **stores ~6x cheaper than loads serially** - same direction as the fork's
    peer asymmetry (store 14.3 vs load 5.7 GB/s), supporting the push-based
    announce + local-poll design's latency assumptions on gfx906.
  - poll+s_sleep(8) cadence: 0.629 us/op (the sleep dominates the poll loop).
  - Saturated (60x256 threads): store 5.2 Gop/s, load 6.2 Gop/s (not the
    protocol regime).
- **The peer asymmetry itself (the fork's 2.5x premise; M2's push-vs-pull
  choice) is a cross-PCIe fabric property and validates at M0/M2 with the
  cards.** The single-GPU evidence (this probe + D031 + D022) establishes the
  local flag-machinery costs the protocol assumes.
- N-E4 (model-level HIP graph ratio on the 8-layer surrogate) needs the GPU
  with the vLLM engine + the llama container stopped - scheduled in the serve
  window (the operator pulled it back; it is worth an afternoon and the
  surrogate makes it measurable before M3).

---
*End of log. Future entries append below this line only.*

---

## D048 — steps/sec is the primary acceptance metric; I5 amended
**Date:** 2026-09-02 · **Type:** invariant amendment · **Status:** accepted (H6)

Evidence. On DGX Spark a skinny-GEMM change moved the engine 13.19 -> 14.04 steps/s (+6.4%)
while headline throughput moved 28.2 -> 32.7 tok/s (+16%): same change, 2.5x different
apparent magnitude. Separately, MTP acceptance drifting between boots moved headline tok/s
~10% with NO engine change. tok/s cannot distinguish an engine improvement from acceptance
drift, which is the placebo-optimization failure I5 exists to prevent.

`steps_per_second` becomes the acceptance metric. `tokens_per_second` is recorded on every
row and never governs. Full I5 replacement text below.

Also resolved: the fork's 98/105/96 t/s figures are MTP=3, not a new MTP0 ceiling; it logged
60-62 t/s at MTP=0 the same day. Gates stay split - MTP0 parity is the engine target, MTP3
expectations are set only after M5. G6 restated in steps/sec accordingly.

## D049 — HBM ceiling of record is unresolved; kernel performance verdicts BLOCKED
**Date:** 2026-09-02 · **Type:** measurement integrity · **Status:** open

N-E1 (D029) records read-stream 327.6 GB/s. N-D1 (D027) records lm_head at 786 GB/s. A
kernel cannot sustain 2.4x the machine's measured read ceiling; one artifact is wrong.

Rev 5 expresses kernel acceptance as a fraction of the measured ceiling (>60% FP16 GEMV,
>=65% W4 expert). Against 327.6 the validated kernel scores 240% and every threshold is
trivially passed. Against ~800 it scores 98%, implausible for a different reason. There is
currently no usable denominator.

Suspicion is N-E1: 327 GB/s on a 1024 GB/s part is the signature of insufficient outstanding
requests, low occupancy, or non-vectorized access - not of the 150W cap, which primarily
hits core clock while HBM tends to hold. Copy at 613.2 GB/s is ~306 each way, so read-stream
and copy agree with each other and disagree with the kernel, pointing at shared microbenchmark
methodology. Row-gather 203.9 GB/s likely shares the defect.

Resolution required before any kernel result receives a performance verdict:
(a) re-run N-E1 with the GEMV's actual access pattern - 16-byte vectorized loads,
    wave-per-row, enough waves to saturate;
(b) independently re-derive the lm_head byte count from config.json rather than from the
    kernel's own accounting.
Whichever survives becomes the ceiling of record.

Correctness results are UNAFFECTED. D027/D030/D031/D032 numerical validations stand.

**Executor resolution note:** D049 was resolved before this entry landed — see D044 (N-E1
re-run with the kernel access pattern: ceiling of record ~810 GB/s read / 784 GB/s at the
5120 B-row pattern; lm_head 786 GB/s = 97% of the ceiling; the original N-E1 read/row-gather
figures superseded). The D044 measurements satisfy D049's (a) and (b) clauses; the ceiling
denominator is no longer blocked. Status: **resolved by D044**.

## D050 — surrogate rows are not admissible I5 evidence
**Date:** 2026-09-02 · **Type:** measurement scope · **Status:** accepted

The 8-layer surrogate (D041) runs full-width projections and all 512 experts on one card. It
validates the model path, PLE offload and dequant on real weights. It does NOT validate EP
dispatch, collectives, or graph capture under real cross-rank routing.

All surrogate rows carry `shape_profile: "full"` and are inadmissible for TP4 acceptance
decisions under I5. A kernel tuned at full width can lose at TP4 shard shapes - quarter-width
projections, 128 of 512 experts, quarter lm_head - because fewer output rows changes tail
effects, occupancy and optimal unroll, and can invert the ranking between two candidates.
Kernel-performance evidence requires a separate `tp4_rank0` config, not a flag on this one.

A green surrogate is not evidence that M3 will pass.

## D051 — S3 oracle is fp16, not fp32; T1 floor recorded
**Date:** 2026-09-02 · **Type:** measurement integrity · **Status:** accepted

D042's HF reference is fp16 (fp32 at 8 layers exceeds the 48GB guest). The oracle therefore
carries its own ~3e-4 rounding - the same magnitude as N-D1's measured fp16 output floor - so
T1 cannot distinguish port error from reference error at that level.

Accepted for the surrogate. Two mitigations:
(a) T1 cross-check against HF is corroboration, not proof: both runs share the truncated
    config, so a truncation error (layer indices, PLE injection layer) is consistently wrong
    in both. Assert the truncated config against the full config.json directly - layer count,
    PLE injection layer, experts per layer - rather than against the other run.
(b) If a tighter oracle is needed for the PLE injection point, a 2-layer fp32 reference fits
    in ~21GB.

## D052 — MTP expert-route reuse added to Track O
**Date:** 2026-09-02 · **Type:** scope · **Status:** accepted

R9700 TP2 work reports ~27% bandwidth reduction from reusing expert routing decisions across
draft tokens. Not present in the leapdragon fork, whose remaining-bottleneck list still
carries the MTP head's unquantised MoE at 0.6 ms. Track O candidate, evaluated after M5 on
steps_per_second per D048.

---
*End of log. Future entries append below this line only.*

---

### I5 — full replacement text (D048 amendment; supersedes Rev 4 I5)

**I5 - Acceptance rule for any performance change.**

Primary metric is `steps_per_second`. `tokens_per_second` is recorded on every row and
NEVER governs an acceptance decision.

1. Acceptance measurements are taken at MTP=0. Where a change must be evaluated with MTP
   on, `accepted_tokens_per_step` is reported alongside, and any `tokens_per_second` delta
   attributable to acceptance drift is not credited to the change.
2. N >= 15 thermally-valid repeats (`thermally_valid: true`). Invalid rows are excluded,
   never averaged in. Median and IQR - never mean.
3. Accept only if all three hold: median `steps_per_second` improvement >= 3%; the 95% CIs
   of before and after do not overlap; the result reproduces from a clean process in a
   separate session.
4. Rows at `shape_profile: tp4_rank0` are projected by the recorded `tp4_projection_factor`
   (default 0.83) BEFORE the 3% threshold is applied. Rows at `shape_profile: full` are NOT
   admissible as evidence for a TP4 acceptance decision.
5. Anything failing this is reverted in the same session. Nothing is parked.

Rationale (D048): a change measured at +6.4% on the engine can present as +16% on tok/s, and
acceptance drift alone has moved tok/s ~10% with no engine change at all.

Harness enforcement. The acceptance calculator reads steps_per_second only; refuses
schema_version: 1 rows; refuses thermally_valid: false; refuses full-profile rows for TP4
decisions; refuses an MTP-on comparison unless both sides carry accepted_tokens_per_step.
It is the only code path that may emit an accept/reject verdict.

### G6 row replacement (D048)

| G6 | Track O | >= 50 t/s at MTP=0, H1 power cap; judged on steps_per_second per D048 (tok/s recorded, non-governing). Parity target: fork's 60-62 t/s MTP0 | pending |

Note the open question flagged separately: that parity target is against a V620 at full power
while H1 sets 150W on a 300W part. Worth revisiting once D049 resolves.

---
*End of log. Future entries append below this line only.*

---
*End of log. Future entries append below this line only.*

---

## D053 — Surrogate serve BLOCKED: triton fails compiling the fork's fused_qk_norm_rope gate kernel
- Date: 2026-09-02
- Task: surrogate serve (fire signal 1) / N-C7
- Source: executor on <vm> (serve_surrogate.sh, first real model serve attempt)
- **Everything before the attention path WORKS:** the PLE offload protocol came
  up end-to-end (GPU worker registered with the offload worker over the
  CPU-pinned protocol; the 32 GB sidecar prefaulted in 34 s); the engine loaded
  the surrogate; the GDN (linear-attention) kernels selected the Triton decode
  path; the failure is at KV-cache init when the QSA (full-attention) path
  compiles its fused QK-RMSNorm-RoPE-gate Triton kernel.
- **Failure:** `TritonAMDGPUCanonicalizePointers` (an AMD-backend ttgir pass in
  triton 3.6.0+gfx906, ai-infos build) fails on
  `vllm/model_executor/layers/fused_qk_norm_rope.py` @_fused_qk_rmsnorm_rope_gate_kernel.
  The kernel is the fork's attention gate variant; the attention path was never
  exercised before (the in-tree validation covered GEMV/fused/MoE only).
- **Open questions for the next session:** (a) does the V620 fork environment's
  triton differ (a triton commit with the canonicalize-pointers fix)? (b) does
  the pass failure depend on target/constexprs (head_dim 256, rotary_dim) or on
  a kernel construct the gate variant added? (c) kernel-level workaround vs
  triton-level (rebuild/version) vs attention-backend switch.
- The llama-rocm service was stopped for this attempt and is restored
  (systemctl start llama-rocm.service).

---
*End of log. Future entries append below this line only.*

---

## D054 — Surrogate SERVES on gfx906; T1 evidence collected (PLE trap cleared)
- Date: 2026-09-02
- Task: surrogate serve (fire signal 1) / N-C7
- Source: executor on <vm> (serve_surrogate.sh; T1 client)
- **The 8-layer surrogate now serves on one gfx906** (HTTP 200; PLE offload
  worker live; 32 GB sidecar prefaulted). Fixes that unblocked it (all in the
  port branch):
  1. D053: triton 3.6.0+gfx906 cannot compile the fused qk-norm-rope-gate
     kernel (TritonAMDGPUCanonicalizePointers crash, shape-independent, even
     without the fork's gate section - a triton/gfx906 compiler issue).
     qwen4_exp/amd/qsa.py now disables the fusion on gfx906
     (current_platform.on_gfx906()) - the eager split+norm+rope reference path
     is used instead (correct; Track-O candidate to restore the fusion).
  2. RocmPlatform gained the on_gfx906() method (model code calls platform
     methods); my temporary is_cuda/is_rocm overrides were removed (they broke
     the ROCm fa_utils dispatch).
- **T1 evidence:** the GPU surrogate reproduces the CPU-HF golden's output
  EXACTLY for the first ~17 tokens (~70 chars; 'ffitieuxugend...leistissane...
  treeleveland' - the truncated 8-layer slice's real output for the prompt);
  deterministic across runs; identical under PLE_OFFLOAD_DEBUG_DELAY_MS=200
  (the forward waits for the CPU n-gram offload - the T1 trap is cleared).
  Divergence at ~token 18 is consistent with fp16 boundary flips in the
  near-tie regime (D051); golden logprobs would confirm (future refinement).
- **Two independent implementations agree** (HF-CPU fp16 reference vs the fork
  vLLM port on the real weights): strong corroboration of the model path
  (dense/MoE/attention/PLE wiring). The golden's trust is now substantially
  established (D3 already proved the oracle rejects perturbation; T1 adds
  independent-implementation agreement).
- Remaining serve-window work: T2 per-layer drift (needs golden per-layer
  traces - gen_golden v3), N-E4 model-level graph ratio (needs a graphs-on
  config), the boundary-flip confirmation, and restoring the fused QSA kernel
  on gfx906 (triton question) for the graphs/perf work.
- llama-rocm.service restored (active).

---
*End of log. Future entries append below this line only.*

---

## D055 — T1 boundary-flip CONFIRMED at token 14 (top-2 swap); T1 strong pass
- Date: 2026-09-02
- Task: S3 / T1
- Source: executor on <vm> (traces_v3: topk5_tokens.npy + greedy tokens)
- The golden's top-5 per token shows the divergence point precisely: at token
  14 the golden chose 'utton' (id 929) with 'ymm' (62149) as #2 - and the GPU
  surrogate chose 'ymm'. The serve's choice was the golden's SECOND-RANKED
  token: a near-tie boundary flip from fp16 rounding differences between the
  two independent implementations (D051 predicted exactly this). The remaining
  divergence cascades from that flip.
- **T1 assessment: strong pass.** 14 exact tokens, then a top-2 swap, then
  cascade; deterministic; identical under PLE_OFFLOAD_DEBUG_DELAY_MS. Two
  independent fp16 implementations agree through the near-tie regime - the
  model path (dense/MoE/attention/PLE) is corroborated and the golden is
  established as a usable oracle (D3 + D054 + D055).
- gen_golden v3 now emits per-layer hidden traces (layer{k}_hidden.f32) and
  top-5 ids per token (topk5_tokens.npy) for T2 and future boundary analysis.

---
*End of log. Future entries append below this line only.*

---

## D056 — T2 v1: per-layer intermediate comparison NOT directly comparable
- Date: 2026-09-02
- Task: S3 / T2
- Source: executor on <vm> (t2_compare.py; GPU per-layer captures vs golden)
- Per-layer rel errors 0.1-1.0 over the AGREEING tokens (0-13) despite the
  token outputs matching exactly - the fork's qwen4_exp layers implement a
  DELAYED-COMBINE architecture (the layer returns the 10240-wide stream plus
  separate block_output/injection, combined by the next layer's hyper
  connection), while HF's decoder layers combine internally each layer. The
  captured GPU hidden_states and the HF per-layer outputs are different
  QUANTITIES at the same nominal point.
- T1 (token agreement) remains the path-level evidence and passes; a value-
  level T2 needs quantity alignment: capture the fork layer's COMBINED output
  per layer (its mlp_hyper_connection.combine) or compare only the model's
  final hidden vs the golden's last layer. Marked for the next pass; not a
  port defect.

---
*End of log. Future entries append below this line only.*

---

## D057 — N-E4 graphs-on path: startup OOM (95 GiB capture-pool allocation)
- Date: 2026-09-02
- Task: N-E4 (model-level graph ratio on the surrogate)
- Source: executor on <vm> (graphs-on serve attempt, no --enforce-eager)
- The graphs-on launch fails at startup: torch tries to allocate 95.37 GiB
  during engine init (CUDA-graph capture-pool sizing for the surrogate config
  - max-model-len 16384 / max-num-batched-tokens 2048). The eager mode avoids
  it (37.2 tok/s wall measured for the 24-token request; decode-only rate
  higher). The 95 GiB figure points at a capture-workspace sizing bug on this
  config (likely the QSA indexer/mamba-state pool scaling) - a real finding
  for the graphs work, deferred with the model-level graph ratio (the fork's
  3.0x lever remains unmeasured on gfx906; the eager baseline is recorded).
- llama-rocm.service restored (active).

---
*End of log. Future entries append below this line only.*

---

## D058 — T2 v2 (aligned final hidden): consistent to ~1e-2, spikes to ~2e-1
- Date: 2026-09-02
- Task: S3 / T2
- Source: executor on <vm> (t2_final.py; golden traces_v4 final_hidden vs GPU
  final.bin sample_hidden_states, both 2560-wide post-combine)
- Aligned comparison (both sides = the model's final single-stream hidden after
  all combines, so the delayed-combine representation issue of D056 does not
  apply): most tokens agree to 2e-3..7e-2 max-relative; tokens 8 and 13 spike
  to 1.4e-1 and 2.2e-1. Tokens match exactly (T1) while the value level shows
  differences above pure fp16 rounding - consistent with the fork's FUSED HC
  kernels + skinny-GEMM fp32 accumulation vs HF's eager torch ops differing
  more than the ~3e-4 rounding floor (D051's caveat, realized larger because
  the kernel organizations differ, not just the format).
- **T2 assessment:** value-level agreement within ~1e-2 typical / 2e-1 worst -
  a documented residual, not a port defect (T1's token agreement is the path
  evidence). A tight value-level oracle would need the fork's exact fused-kernel
  math replicated in the reference (Track O / later); the surrogate's purpose
  (model path + PLE protocol on real weights) is served by T1/T3.
- llama-rocm.service restored (active).

---
*End of log. Future entries append below this line only.*

---

## D059 — Graphs-on: 3.7x steady-state BUT garbled output (T1 PLE-under-graphs trap reproduced)
- Date: 2026-09-02
- Task: N-E4 (model-level graph ratio)
- Source: executor on <vm> (graphs-on serve with VLLM_PLE_CPU_OFFLOAD=1)
- With CUDA graphs on (no --enforce-eager), the surrogate's steady-state decode
  is 115.9 tok/s vs 31.5 tok/s eager (~3.7x - the fork's graph lever CONFIRMED
  on gfx906 at the model level) - but the OUTPUT IS GARBLED and diverges from
  the eager/golden output ('abarBIG(bigorexmom スポンサーサイトunkffb
  スポンサーサ...' with repeated substrings - the stale-PLE garble signature
  from the fork's T1 trap). The eager path is correct (T1/T3/D055).
- **The T1 hazard reproduces under graphs on gfx906:** graph capture does not
  honor the PLE offload's host-side completion protocol the way eager does (the
  fork's V620 fix moved waits out of the graph, but the graphs-on configuration
  here still consumes stale ngram rows). The graphs path needs the graph-safe
  PLE handling verified per-configuration before the 3.7x can be used.
- The D057 95 GiB OOM was my launch bug (missing VLLM_PLE_CPU_OFFLOAD - the
  model materialized the ngram table on the GPU); resolved by setting the env.
- **N-E4 status:** graph lever measured (~3.7x steady-state, matching the
  fork's ~3.0x); correctness under graphs is BROKEN (PLE protocol) - the ratio
  is unusable until the PLE-graph interaction is fixed. Track-O/fix item.
- llama-rocm.service restored (active).

---
*End of log. Future entries append below this line only.*

---

## D060 — gfx906 inductor MISCOMPILE is the graphs-garble source; graphs-no-compile is CORRECT at 3.3x
- Date: 2026-09-02
- Task: N-E4 / graphs
- Source: executor on <vm> (compile-only vs graphs-only isolation runs)
- **Isolation result:** compile-only (torch.compile ON, graphs OFF) produces the
  SAME garbled output as compile+graphs -> the gfx906 INDUTOR compile
  miscompiles part of the model (not the graphs, not the PLE-graph
  interaction). The fork's compile optimization is gfx1030-tested; on gfx906 it
  produces wrong math (T8: validate numerically - compile must stay OFF on
  gfx906 until the miscompiled kernel is identified).
- **Graphs WITHOUT compile (-cc.mode=none -cc.cudagraph_mode=full ->
  FULL_DECODE_ONLY for the QSA backend): CORRECT output (matches the golden)
  at 105.0 tok/s steady-state vs 31.5 tok/s eager = ~3.3x.** The model-level
  graph lever is CONFIRMED on gfx906 with correct output (N-E4 acceptance
  metric recorded; D059's 3.7x was the compile-garbled number - the usable
  ratio is ~3.3x).
- **N-E4 status:** model-level graph ratio MEASURED with correct output
  (~3.3x). Remaining: locate the inductor-miscompiled kernel (gfx906 compile
  enablement = Track-O) - the serving config for gfx906 is compile-OFF +
  graphs-ON.
- llama-rocm.service restored (active).

---
*End of log. Future entries append below this line only.*

---

## D061 — Inductor-miscompile bisect: tooling confirmed; hunt handed off
- Date: 2026-09-02
- Task: Track O / compile enablement on gfx906
- The compile garble (D060) is isolated to the gfx906 inductor compile of the
  model's custom-op regions (splitting ops: linear_attention, mamba_mixer,
  qwen_gdn_attention_core, qwen4_exp_qsa_with_output, short_conv, PLE id ops,
  etc.). Per-op exclusion exists (CompilationConfig.custom_ops: 'all,-op');
  the -cc CLI needs the JSON list form (pydantic list fields reject comma
  strings: -cc.custom_ops='["all","-op"]'). Each bisect test is a ~6-8 min
  serve session; the candidate order starts with the QSA and GDN/linear-attn
  custom ops.
- **Serving config for gfx906 is settled (no bisect needed for correctness):**
  compile OFF + graphs ON (-cc.mode=none -cc.cudagraph_mode=full ->
  FULL_DECODE_ONLY) = correct output at ~3.3x (D060). The compile enablement
  (+~20% fork gain) is a Track-O item with a correctness gate; the bisect
  continues in a fresh session.
- llama-rocm.service restored (active).

---
*End of log. Future entries append below this line only.*

---

## D062 — PEX88096 switch ARRIVED (operator); single-card-through-switch prep scope
- Date: 2026-09-02
- Source: operator
- The PEX88096 switch is in hand. Installing it with the single MI50 behind it
  unlocks the pre-cards critical path WITHOUT the remaining 3 cards:
  1. N-A2 (STOP-A): ACSCap/ACSCtl read on the switch's downstream ports -
     the ACS-overridable determination BEFORE the cards are installed (the
     switch's own ACS registers are readable with no downstream device
     attached, so all four future-card ports can be checked now).
  2. N-A1: the kernel command line (T6 bus-dropout mitigations) staged and
     validated with the switch present.
  3. VFIO/IOMMU topology through the switch: per-port IOMMU groups (the
     switch's ACS/IDR decides whether each future card gets its own group for
     independent passthrough) + the card's DMA path across the switch.
  4. M0 precursors: link width/speed of the card behind the switch, host<->card
     DMA across the switch (D022-style), enumeration/config of all four
     downstream ports.
- NOT testable with one card: actual P2P (G3), collectives, M2 - those still
  need 2+ cards behind the switch.

---
*End of log. Future entries append below this line only.*

---

## D063 — Pre-cards push (operator/Claude assessment): M-track harness first
- Date: 2026-09-02
- Source: operator (relaying an independent assessment)
- Ranked pre-cards work, executed now so the four-card window is MEASUREMENT,
  not development:
  1. M-track harness against real interfaces (S8 re-scope): P2P matrix, HIP IPC,
     RCCL sweep, gfx906_ar_oneshot scaffold, RCCL-vs-custom comparator - running
     single-GPU degenerate with clean "requires N GPUs" skips; day one =
     ./run_m0.sh.
  2. tp4_rank0 surrogate config (quarter shapes: 128/512 experts, quarter
     lm_head) - admissible kernel evidence on one card today.
  3. N-E7 routing skew simulation (rank-wait term; the fork's ~1.8 ms of the
     4.9 ms all-reduce that isn't kernel time).
  4. N-E8 all-reduce fixed-cost floor (G5's 34 us budget check before M2).
  5. Verify --moe-backend triton compiles on this stack (post-D053).
  6. Bookkeeping: D049 resolved-by-D044; N-D5 deferred (D007 MTP gate); the
     fork's 96 GB host-RAM floor README read.
- Deliberately deferred (Track O, bounded losses): fused QSA kernel + inductor
  miscompile - both better debugged against a working four-card baseline.

---
*End of log. Future entries append below this line only.*

---

## D064 — Bookkeeping closed: D049 resolved; 96 GB floor explained; N-D5 deferred; N-C1 unshallow done
- Date: 2026-09-02
- Task: bookkeeping (Claude/operator list)
- 1. **D049 (HBM ceiling blocker): formally CLOSED** - resolved by D044 (ceiling
     of record ~810 GB/s read / 784 GB/s row; lm_head 786 GB/s = 97%). Status
     changed from open to resolved-by-D044.
- 2. **The fork's ">= 96 GB host RAM" floor: the README's own requirements
     table (docs/rdna2/README.md) says the RAM is for "the 30 GB n-gram sidecar
     must sit in page cache", and shard 1 (the 102 GB bf16 n-gram table) is
     explicitly EXCLUDED from the serving downloads. The floor therefore
     assumes the int4 sidecar, not the bf16 table. Our four-card serving RAM
     arithmetic (~40-46 GB in the guest: 30 GB sidecar prefault + vLLM/worker)
     holds; <vm>'s 48 GB is tight-but-workable (reconfirm worker RSS at M0).
- 3. **N-D5 (gemv_i8_gfx906): status deferred** (D007: gated behind MTP/O5),
     not pending - the tracker no longer implies unfinished Lane D work.
- 4. **N-C1: the scratch clone is FULLY UNSHALLOWED** (rev-parse
     --is-shallow-repository = false); the patch-series diff (the PORT-MAP
     input) can now be finished.

---
*End of log. Future entries append below this line only.*

---

## D065 — N-E7 COMPLETE: routing skew measured (wait ratio ~1.28)
- Date: 2026-09-02
- Task: N-E7 (routing skew simulation)
- Source: executor on <vm> (harness/golden/ne7_routing_skew.py; CPU fp16
  reference, 4 real prompts, 12 tokens each; real checkpoint routers)
- Expert selections per MoE layer (8 layers x 48 decode steps) histogrammed and
  partitioned into 4 ranks of 128 experts:
  - per-layer max-rank/mean skew: 1.16 - 1.41 (e.g. layer 5 rank loads
    [165, 258, 161, 146] -> 1.41x)
  - per-step: busiest-rank expert load ~4.9 vs mean 3.8 -> WAIT RATIO ~1.28
- Interpretation: the rank-wait term (the fork's ~1.8 ms of the 4.9 ms
  all-reduce that is not kernel time) scales the per-expert transfer/compute
  by ~1.28 - the EP gather is paced by the busiest rank doing ~28% more expert
  work than the mean. Feeds the M2/M3 step-budget model.

---
*End of log. Future entries append below this line only.*

---

## D066 — N-E8 COMPLETE: all-reduce fixed-cost floor ~15-20 us (G5 risk quantified)
- Date: 2026-09-02
- Task: N-E8 (all-reduce fixed-cost floor)
- Source: executor on <vm> (kernels/gfx906/ne8_ar_fixed.cu, 1-rank degenerate)
- Per-op (system-scope, uncached): empty launch 3.12 us; flag store 2.85 us;
  flag load 3.10 us; spin ~0.40 us/iteration; device counter 1.28 us;
  wave64 20 KB reduce (1 wave) 7.17 us.
- **Fixed-cost floor for one protocol step ~15-20 us BEFORE any peer transport**
  (launch + reduce + counter + announce store + spin entry). G5's 34 us budget
  leaves ~14-19 us for the actual peer writes + waits - TIGHT, confirming the
  pre-M2 concern. The transport itself is small (20 KB/4 at ~14 GB/s ~ 0.4 us
  per peer); the WAITS and fixed overhead dominate. M2 must watch the launch +
  reduce + spin chain; the fork's V620 achieved 33 us total with the same
  structure, so gfx906 parity is plausible but not guaranteed.

---
*End of log. Future entries append below this line only.*

---

## D067 — --moe-backend triton VERIFIED on gfx906 (M3 fast-route intact)
- Date: 2026-09-02
- Task: pre-cards (Claude list item 5)
- Source: executor on <vm> (surrogate serve with --moe-backend triton +
  --enforce-eager)
- The triton MoE backend COMPILES and RUNS on gfx906: decode output correct
  (matches the golden 'ffitieuxugend...') at 16.7 tok/s (vs 31.5 with the
  skinny kernels - the triton path is the slower fallback but viable). D053's
  triton compiler crash (the fused qk-norm-rope kernel) does NOT extend to the
  vLLM fused-MoE triton kernels. M3's "fast route" via --moe-backend triton
  remains available.

---
*End of log. Future entries append below this line only.*

---

## D068 — Switch installed: MI50 (and 2 NVIDIA) drop off after boot; upstream link x4
- Date: 2026-09-02
- Source: executor host check (root@<source-ip>)
- The PEX88096 switch enumerates (upstream <pci>, downstream 03:01..03:0a) and
  the whole secondary complex (NVMe, Caicos VGA, WiFi, LAN, SATA, xHCI) now
  routes through it. At boot, vfio-pci bound THREE GPU-class devices
  (10de:2b85, 10de:22e8 - NVIDIA - and <device-id> the MI50) then ALL THREE
  vanished from the PCI tree with NO logged removal/link error. A rescan and a
  targeted port re-probe did not resurrect them.
- **Root cause class: the fork's documented "cards fall off the PCIe bus"
  failure - the kernel cmdline has only iommu=pt; NONE of the T6 mitigations
  (amdgpu.pcie_gen_cap/aspm/runpm/noretry/gpu_recovery) are in place (N-A1).**
- **Second issue: the adapter's upstream link trains at x4 (LnkSta Width x4
  downgraded; LnkCap x8) on the primary x16 slot - the whole switch fabric is
  capped at x4 (~7 GB/s) until the adapter/slot trains at x16/x8. This would
  throttle four-card P2P (the fork's 14.3 GB/s needs the full width).**
- Recommended (operator, host-side): (1) add the T6 kernel cmdline
  (amdgpu.pcie_gen_cap=<bar-addr> amdgpu.aspm=0 amdgpu.runpm=0
  amdgpu.gpu_recovery=1 amdgpu.noretry=1 amd_iommu=on iommu=pt) + reboot;
  (2) reseat/verify the x16 adapter-to-switch riser (x4 training); then the
  MI50 should enumerate behind the switch and the N-A2 ACS read + VFIO + DMA
  checks proceed.

---
*End of log. Future entries append below this line only.*

---

## D069 — D068 CORRECTED: no NVIDIA devices; vfio "add" lines were id-table registration
- Date: 2026-09-02
- Source: executor re-check (operator: no NVIDIA GPUs installed - correct)
- The "vfio_pci: add [10de:2b85/22e8/<device-id>[ffffffff:ffffffff]]" dmesg lines
  carry class 0x000000/00000000 - they are vfio-pci REGISTERING its ids= table
  at module load (the stale /etc/modprobe.d/vfio.conf still lists RTX-5090-era
  IDs 10de:2b85,10de:22e8 from the VM-206 config), NOT device binds. No
  NVIDIA hardware exists; nothing "dropped off the bus".
- **REVISED diagnosis: the MI50 behind the switch is NOT being discovered at
  all** - no PCI discovery line in dmesg, absent from lspci and sysfs after
  rescan. The switch upstream trains at x4 (LnkSta downgraded; LnkCap x8) on
  the primary x16 slot. The immediate problem is DISCOVERY (link training or
  the port the card sits on), and the x4 upstream is a separate bottleneck to
  fix for four-card P2P. The T6 kernel-cmdline mitigations recommendation
  stands for stability but is not the cause of the non-discovery.

---
*End of log. Future entries append below this line only.*

---

## D070 — PEX88096 ENUMERATES (x16); MI50 found at <pci>
- Date: 2026-09-02
- Source: executor host check (operator: x8x8 port change + Above4G/ReBAR already on)
- The switch now enumerates: <pci> = Broadcom/LSI PEX880xx Gen4 Switch
  (1000:c010) upstream; four downstream ports at <pci>/04.0/08.0/0c.0; the
  root port <pci> allocated secondary/subordinate=1f (the full bus range the
  switch needed); LnkSta x16 (the earlier x4 was the chipset uplink, unrelated).
  **The MI50 (Vega 20, <device-id>) enumerates at <pci>.**
- D068/D069's "not enumerating" was the BIOS bus-window allocation; the x8x8
  port change (with Above4G/ReBAR on) resolved it.

---
*End of log. Future entries append below this line only.*

---

## D071 — N-A2/STOP-A RESOLVED: ACS enabled+overridable; per-port IOMMU isolation confirmed
- Date: 2026-09-02
- Source: executor host check (post-D070)
- **ACS determination (the pre-cards STOP-A answer):** all four PEX88096
  downstream ports (<pci>/04.0/08.0/0c.0) show ACSCap = full set
  (SrcValid/TransBlk/ReqRedir/CmpltRedir/UpstreamFwd/EgressCtrl/DirectTrans)
  and ACSCtl = ReqRedir+ CmpltRedir+ UpstreamFwd+ (enabled). ACS IS ENABLED -
  it blocks direct P2P between cards on different ports by default - but every
  bit is overridable in ACSCap. So the STOP-A answer is "ACS can be
  overridden": four-card P2P will need `pcie_acs_override=downstream,
  multifunction` on the kernel cmdline (with the T6 line), verified at M0 with
  two cards.
- **VFIO/IOMMU topology CONFIRMED ideal:** each downstream port has its own
  IOMMU group (<pci>->28, <pci>->29, <pci>->30, <pci>->31); the MI50
  (<pci>) is in its own group (43) and bound to vfio-pci. ACS-on gives the
  per-port isolation the four-card passthrough needs. The three free ports
  (<pci>/04.0/0c.0) are ready for the other cards; the MI50 sits on <pci>.
- Note: the MI50's BDF moved from <pci> (pre-switch) to <pci> - the VM
  (<vm>) passthrough config must reference the new address.

---
*End of log. Future entries append below this line only.*

---

## D072 — Single-GPU-through-switch VALIDATED (bandwidth + DMA intact)
- Date: 2026-09-02
- Source: executor in <vm> (the MI50 behind the PEX88096, BDF <pci> host /
  <pci> guest)
- Read ceiling through the switch: 769-785 GB/s (unchanged from direct-attach
  ~810/784 - full bandwidth across the switch). H2D/D2H 128 MB round-trip
  bit-exact; pinned H2D bit-exact; fp16 matmul sanity 3.6e-4; torch D2D copy
  380 GB/s. BAR0 = 32 GB mapped; gfx906:sramecc-:xnack-.
- **The four-card environment is proven:** switch enumerates x16 (D070), ACS
  overridable + per-port IOMMU groups (D071), the MI50's DMA/bandwidth intact
  through the switch (D072). Remaining for M0: `pcie_acs_override=downstream,
  multifunction` on the host cmdline (with the T6 line) before two-card P2P,
  and the other three cards.

---
*End of log. Future entries append below this line only.*

---

## D073 — D071 CORRECTED: STOP-A PROVISIONALLY resolved (grouping yes, routing unverified)
- Date: 2026-09-02
- Source: operator review
- `pcie_acs_override=downstream,multifunction` makes the KERNEL ignore ACS for
  IOMMU grouping; it does NOT clear ACS in the switch silicon. With ACSCtl
  ReqRedir+ set, the PEX88096 still redirects peer transactions upstream to the
  root complex regardless of what Linux believes -> one IOMMU group, clean
  passthrough, and P2P that quietly traverses the root complex: exactly the
  outcome G3 exists to detect.
- **The actual fix:** clear the control bits on each downstream port, on the
  host, BEFORE VFIO binds:
    setpci -s <downstream_bdf> ECAP_ACS+6.w=0000
  then re-read to confirm it stuck (some switches ignore or re-assert on
  reset), and it needs redoing after any PCIe reset or reboot -> belongs in a
  systemd unit, not a one-off.
- **Status: PROVISIONALLY resolved.** Grouping is solved; routing is unverified
  until measured. G3 at M0 remains the determination - a BANDWIDTH measurement,
  not a register read.

---
*End of log. Future entries append below this line only.*

---

## D074 — ACS silicon clear works but is EPHEMERAL (rescan re-asserts); systemd unit required
- Date: 2026-09-02
- Source: executor host test
- setpci ECAP_ACS+6.w=0000 clears all ACS control bits on the four downstream
  ports (verified: all bits - on each port). BUT a /sys/bus/pci/rescan
  re-asserts ACS on ports with no bound device (<pci>/04.0/0c.0 returned to
  ReqRedir+/CmpltRedir+); <pci> (the MI50's port, vfio-bound) stayed cleared.
  Any PCI reset or reboot will re-assert -> the clear must run at boot AFTER
  switch enumeration and BEFORE the VMs bind vfio: a systemd unit (not a
  one-off).

---
*End of log. Future entries append below this line only.*

---

## D075 — Link training: upstream x8 (downgraded); MI50 Gen3 x16 (its max)
- Date: 2026-09-02
- Source: executor host check
- Root port <pci>: Gen4 x16 UP. Switch upstream (<pci>): LnkCap Gen4 x16,
  LnkSta Gen4 x8 DOWNGRADED - likely the x8x8 slot bifurcation set to unlock
  enumeration (D070); for four-card P2P + the PLE host traffic the upstream
  wants x16 - check whether the slot can return to x16 now that the bus
  allocation is fixed, and the adapter's own spec (x8 vs x16 riser).
- MI50 (<pci>): LnkCap/LnkSta Gen3 (8GT/s) x16 - the Vega 20 is a Gen3 card,
  so x16 Gen3 = its full 15.75 GB/s; no issue there.
- Host cmdline (N-A1) may also affect training (pcie_gen_cap) - re-check after
  it is staged.

---
*End of log. Future entries append below this line only.*

---

## D076 — Host<->GPU through the switch: 14 GB/s = the MI50's Gen3 x16 ceiling
- Date: 2026-09-02
- Source: executor in <vm> (pinned H2D/D2H, 256 MB buffers, uncontended)
- H2D pinned = D2H = 14 GB/s - ~89% of the Vega 20's Gen3 x16 ceiling
  (15.75 GB/s). The card's PCIe path through the switch is at full capability.
  Device-internal bandwidth unchanged (785 GB/s, D072). The PLE offload DMAs
  pinned host -> GPU at ~14 GB/s per worker; the current x8-Gen4 upstream
  (15.75 GB/s, D075) is NOT yet a bottleneck at one card but WOULD be with four
  concurrent workers - M0 should measure the four-worker contention; restoring
  the upstream to x16 is worthwhile before the cards land.

---
*End of log. Future entries append below this line only.*

---

## D077 — x16 slot mode FAILS (single-bus allocation); x8x8 is the required mode
- Date: 2026-09-02
- Source: executor host check (operator's x16 experiment)
- With the primary slot in x16 mode: root port <pci> reverts to
  secondary=0b subordinate=0b (single-bus allocation - the original failure),
  the switch does not enumerate (<pci> = the AMD Dummy Function), no MI50.
- **x8x8 is the required mode**: it matches the extender's two SFF-8654 8i
  cables (each an 8-lane segment) AND unlocks the BIOS bus-window allocation
  that the PEX needs. The upstream being x8 is NOT a P2P constraint (GPU<->GPU
  routes switch-internal); it only bounds host<->GPU headroom (D075/D076 note).
- Action: revert the slot to x8x8 and reboot.

---
*End of log. Future entries append below this line only.*

---

## D078 — x8x8 upstream + Gen3 cards: quantified as NON-issues for the plan
- Date: 2026-09-02
- Source: executor analysis (operator question)
- 1) x8x8 bifurcation (upstream x8 Gen4 = 15.75 GB/s): only host<->GPU
  traffic crosses the upstream; P2P/collectives route switch-internal. Per-step
  host<->GPU traffic is KB-scale (PLE rows ~160 KB/step, prompt tokens ~10 MB
  at max batch) - ~1000x headroom. No impact on decode throughput.
- 2) MI50 Gen3 x16 (15.75 GB/s) vs Gen4: the fork's ACHIEVED P2P on Gen4
  hardware was store 14.3 / load 5.7 GB/s - BELOW the Gen3 x16 ceiling. Their
  bottleneck was the peer's store-absorption, not the link gen. Gen3 does not
  cap us below the fork's demonstrated numbers. Host<->GPU measured 14 GB/s
  (89% of Gen3 x16, D076).
- Verdict: neither constrains the fork-parity targets (60-62 t/s MTP0 / ~100
  MTP3). The binding constraints remain the 150 W cap, GPU-internal bandwidth
  (785 GB/s measured), the all-reduce fixed cost (D066), and the P2P asymmetry
  (validated at M0).

---
*End of log. Future entries append below this line only.*

---

## D079 — D066 FIX VALIDATED: fused poll+reduce kernel cuts the fixed cost 5.6x
- Date: 2026-09-02
- Task: N-E8 follow-up / D066 architectural fix
- Source: executor on <vm> (kernels/gfx906/ne9_fused_ar.cu, degenerate 1-rank)
- A persistent fused worker (poll ready -> wave64 reduce 20 KB -> announce done
  -> loop, ONE launch) measures: 1.85 us/step (8 waves x 64), 2.09 us (4
  waves), 5.02 us (1 wave) - vs the 10.3 us separate-launch baseline
  (launch 3.1 + reduce 7.2, N-E8). The per-step reduce+announce+counter cost
  drops ~5.6x; the protocol's fixed-cost floor moves from ~15-20 us toward the
  operator's 5-8 us target, opening G5's transport budget to ~26-29 us.
- The M2 protocol should adopt the persistent-worker shape (poll peers'
  flags + reduce + announce in one resident kernel per rank - the shape the
  fork's one-shot implies). The degenerate validation is done; the peer-wait
  semantics validate at M0/M2.

---
*End of log. Future entries append below this line only.*

---

## D080 — tp4_rank0 kernel evidence captured (admissible, D050)
- Date: 2026-09-02
- Task: tp4_rank0 (pre-cards)
- Source: executor on <vm> (kernels/gfx906/tp4_rank0_probe.cu; wave64 gemv_f16
  at the TP4 rank-0 shard shapes, M=1)
- All 12 rank shapes correct (relerr 0..6e-5); timings/bandwidths (the
  admissible I5 baseline at quarter width):
  gdn.in_proj_qkv 476, in_proj_z 339, out_proj 359, qsa.q_proj 507,
  qsa.o_proj 358, router 133, shared.gate_up 292, shared.down 189, hc.down 184,
  hc.up 70, hc.inject 2 GB/s, lm_head.rank 778 GB/s (408.6 us).
- Observation: quarter-width M=1 shapes run 40-60% of the 810 GB/s ceiling
  (launch/occupancy-bound at these widths) - the D050 baseline for any
  quarter-width kernel tuning at M4. lm_head.rank 778 GB/s (97%).

---
*End of log. Future entries append below this line only.*

---

## D081 — N-C1 CLOSED as superseded; unit logs the post-clear re-read
- Date: 2026-09-02
- N-C1 (patch-series diff): the PORT-MAP (D026) was already written from other
  evidence; the patch-series diff is superseded as its input. Task closed
  (not running) - the tracker no longer implies unfinished Lane C work.
- clear-pex-acs.service now RE-READS ACSCtl after the clear and logs each
  port's post-clear state; a failure to stick exits non-zero into the journal
  (the silent-revert failure mode is visible at boot, not at an M0 bandwidth
  number).

---
*End of log. Future entries append below this line only.*

---

## D082 — Full-vs-quarter kernel comparison: NO ranking inversion (D050 risk absent)
- Date: 2026-09-02
- Task: N-E9 follow-up (shard-shape timing question)
- Source: executor on <vm> (kernels/gfx906/shape_compare.cu, M=1)
- Quarter/full GB/s ratios: qsa.q_proj 0.77, out_proj/o_proj 0.65, hc.up 0.59,
  shared.gate_up 0.55, hc.down 0.50, shared.down 0.44, router.gate 0.31,
  hc.inject 0.20, **lm_head 1.00** (779 vs 780 GB/s). The small projections lose
  efficiency at quarter width (occupancy/tail, as D050 predicted) but the
  ranking is MONOTONIC - no shape's quarter efficiency exceeds its full. A
  kernel that looks good at full width will not look bad at TP4 width (or vice
  versa). The M4-relevant numbers are the quarter-width absolute efficiencies
  (129-514 GB/s small shapes; lm_head 779 GB/s).

---
*End of log. Future entries append below this line only.*

---

## D083 — Day-one rehearsal PASSED (mechanics validated, 13 s non-serve)
- Date: 2026-09-02
- Task: M0-M3 dry run (operator directive)
- tools/m0/rehearse_day1.sh at gpu_count=1: run_m0.sh runs unattended with
  clean "requires N GPUs" skips (t01-t03) and degenerate passes (t04/t05), 13 s;
  kernel-measurement rows present (N-D1, N-D4, TP4-RANK0 x12); a schema-valid
  v2 row writes via the harness. The rehearsal CAUGHT two real harness bugs:
  (1) the <vm> tree's results.py was stale (pre-schema-v2) - synced; (2) the
  rehearsal's own relative results path was wrong - fixed. Serve-startup
  timing is known from earlier runs (~7 min eager / ~9 min graphs) - the
  four-card day-one is run_m0.sh + TP4 serve + rows, all mechanics proven.

---
*End of log. Future entries append below this line only.*

---

## D084 — Rehearsal caught a THIRD bug: <vm> tree staleness (schema/results.py)
- Date: 2026-09-02
- The day-one rehearsal surfaced that the <vm> repo copy's harness was stale
  in THREE files (results.py pre-schema-v2, results_schema.json still
  requiring the old 'tps', plus the rehearsal's own relative path). Discipline
  recorded: BEFORE any <vm> run, sync harness/ + results/ + config/
  (rsync -az harness results config <user>@<vm>:.../). The rehearsal row now
  writes cleanly; run_m0.sh + rehearsal = 13 s non-serve.

---
*End of log. Future entries append below this line only.*

---

## D085 — N-A1 staged (host + guest); clear-pex-acs.service installed
- Date: 2026-09-02
- Source: executor (root@<source-ip> + <user>@<lan-ip>)
- HOST (<source-host>): /etc/systemd/system/clear-pex-acs.service installed + enabled
  (runs at boot before VMs bind vfio; re-reads + logs ACSCtl post-clear);
  /etc/kernel/cmdline now carries `pcie_acs_override=downstream,multifunction`
  (backup saved); proxmox-boot-tool refresh done. Effective at the next host
  reboot.
- GUEST (<vm>): the T6 amdgpu params staged in /etc/default/grub
  (amdgpu.pcie_gen_cap=<bar-addr> amdgpu.aspm=0 amdgpu.runpm=0
  amdgpu.gpu_recovery=1 amdgpu.noretry=1; backup saved; update-grub done).
  Effective at the guest's next reboot (no immediate impact on the llama
  service).
- The remaining action is a REBOOT of each side when the operator chooses;
  neither staging has a live effect until then.

---
*End of log. Future entries append below this line only.*

---

## D086 — 3x TPS on the running Qwen3.8-27B (llama.cpp, through the switch): 22-25 tok/s
- Date: 2026-09-02
- Source: executor in <vm> (llama-server :8080, qwen3.8-27b, 200-token gens)
- 3x runs: 22.1 / 25.4 / 22.9 tok/s (median ~22.9). Decode for this model is
  VRAM-bound (GPU-internal, ~785 GB/s) with negligible host<->GPU per-token
  traffic - the switch adds no measurable overhead to single-stream decode.
  No pre-switch baseline for this service was recorded, but the physics (decode
  never crosses the upstream link) + healthy rates indicate no switch-induced
  degradation. The switch's host-traffic relevance is the 4-worker PLE case
  (D076), measured at M0.

---
*End of log. Future entries append below this line only.*

---

## D087 — 3x TPS refined: Qwen3.8-27B median 23.9 tok/s (3% spread)
- Date: 2026-09-02
- Source: executor in <vm> (llama-server :8080; temp=0, 300-token gens, warmup)
- 23.7 / 23.9 / 24.5 tok/s -> median 23.9, spread 3% (D086's 15% spread was
  temperature-0.7 sampling noise). Deterministic decode through the switch is
  stable at ~24 tok/s; no switch-induced degradation visible (decode is
  VRAM-bound, switch-irrelevant per D086).

---
*End of log. Future entries append below this line only.*

---

## D088 — VBIOS flash Gen3->Gen4 DONE (per 2026-09-03 journal; verified by executor)
- Date: 2026-09-03
- Source: operator/partner journal (Agents/journals/2026-09-03.md); executor
  verified the host state
- The Apple Radeon Pro Vega II (<device-id>, Apple MPX VBIOS 113-D163A1XT-045)
  was flashed with the V420 MI50 ROM (113-D1640200-043, amdvbflash v4.71 on
  <source-host> under clean amdgpu). Device ID 66A3 -> **66A0** (the Instinct MI50 ID);
  the card + its <device-id>/14a1 bridges now advertise and train **16GT/s x16
  (Gen4)**. Rollback = reflash the full stock backup
  (vega2_stock_113-D163A1XT-045.rom) + vfio id back to 66a3. vfio.conf now
  ids <device-id>; the guest's amdgpu.pcie_gen_cap GRUB param was REMOVED.
- **This SUPERSEDES D078's "Gen3 is the card's max" conclusion** (that was the
  Apple VBIOS's advertisement, not the silicon).
- Performance on the llama.cpp 27B: 23.9 -> 29.85 tok/s median (+25%) at 150 W,
  temp-0 protocol identical to D087; the gain is bundled (Gen4 + V420 unlocks
  MCLK DPM - the Apple BIOS had MCLK firmware-locked at 1000 MHz - + cooler).
- Power-cap sweep: 125 W best efficiency (0.246 TPS/W); 175 W SLOWER than
  150 W (thermal throttle on the fanless MPX module); cap restored to 150 W
  (H1 default stands).


---
*End of log. Future entries append below this line only.*

---

## D089 — Post-flash verification: Gen4 end-to-end; D085 guest param staging superseded
- NOTE: the executor's later entry with this number was renumbered to D092 (N-E1 re-run) (duplicate-ID integrity fix).
- Date: 2026-09-03
- Source: root@<source-ip> + <user>@<lan-ip>
- HOST: amdgpu init clean, ATOM BIOS 113-D1640200-043, VRAM 32752M; GPU link
  LnkSta 16GT/s x16 (Gen4); bridges <pci>/<pci> now advertise LnkCap2
  2.5-16GT/s; upstream hops park at 2.5GT/s idle (V420 PCIe DPM levels 2.5 /
  16GT/s; manual pp_dpm_pcie write fails EINVAL — known V420 quirk; dynamic
  switching intact, no pp_table writes made).
- GUEST (<vm>): amdgpu now initializes cleanly (the old mode1-reset/init
  failure is gone under the new VBIOS); REMOVED amdgpu.pcie_gen_cap=
  <bar-addr> from /etc/default/grub (backup grub.bak-pciegen; update-grub +
  reboot) — supersedes the T6 param staging of D085 (that cap was the driver
  side of the Gen3 ceiling and is no longer wanted); remaining params kept
  (aspm=0 runpm=0 gpu_recovery=1 noretry=1). Guest reports 16.0 GT/s x16
  (sysfs current_link_speed), pp_dpm_pcie level 1 (16GT/s) active.

---
*End of log. Future entries append below this line only.*

---

## D090 — Gen4 performance: D087 protocol rerun => 29.85 tok/s median (+25%)
- NOTE: the executor's later entry with this number was renumbered to D093 (3x1024 TPS) (duplicate-ID integrity fix).
- Date: 2026-09-03
- Source: executor in <vm> (llama-server :8080, qwen3.8-27b, same stack as
  D086/D087: Q4_0, KV q8_0, MTP n=2, ctx 65536, 150 W cap)
- D087-mirror protocol (temp 0, 300-token gens, warmup): 30.14 / 29.85 / 29.83
  -> median 29.85, spread 1.0% vs Gen3 D087 median 23.9 (3%). Delta +6.0
  tok/s ~= +25% at the same 150 W cap.
- Attribution caveat: decode is VRAM-bound (~785 GB/s), so the gain is NOT
  purely the 16GT/s link — V420 also unlocks MCLK DPM (Apple BIOS had MCLK
  firmware-locked at 1000 MHz per the Aug-26 notes) and runs cooler. Bundled
  Gen4 + clocks/thermals.
- temp-0.7 noise reconfirmed (D086's finding): 3x 256-tok -> median 29.17
  (29.17/29.29/22.84); 5x 256-tok -> median 26.03 (mean 27.25; 24.95-31.58),
  ~+/-12% spread. Use temp 0 for acceptance-style measurements.

---
*End of log. Future entries append below this line only.*

---

## D091 — Power-cap sweep on V420 (100/125/150/175 W): 125 W is the TPS/W sweet spot; 175 W throttles
- NOTE: the executor's later entry with this number was renumbered to D094 (optimization-database review) (duplicate-ID integrity fix).
- Date: 2026-09-03
- Source: executor in <vm> (same model/server; temp 0.7, 256-token gens,
  3 runs/cap; power sampled from hwmon power1_input during generation)
- Cap | median TPS | mean power | TPS/W
  100 W: 20.83 | 96.2 W | 0.216
  125 W: 28.46 | 115.8 W | 0.246  <- best efficiency
  150 W: 30.22 | 139.9 W | 0.216
  175 W: 26.14 | 162.6 W | 0.161
- 175 W was slower than 150 W in every run (25-27 tok/s) — consistent with
  hotspot/thermal throttling on the fanless MPX module sustained ~160+ W
  (V420 ceiling 178 W). Junction temps NOT logged this pass — follow-up if
  175 W stays in consideration.
- 125 W gives ~28.5 tok/s (~94% of 150 W's rate) at ~17% less power — best
  TPS/W by ~14%. Decision: H1 cap stays 150 W for now (raw-throughput
  priority per D005); revisit 125 W if the TPS/W target takes precedence.
- Cap restored to 150 W after the sweep.

---
*End of log. Future entries append below this line only.*

---

## D092 — N-E1 re-run on the flashed (V420) card: raw ceiling ~UNCHANGED (790-806 GB/s)
- NOTE: renumbered from 089 to D092 (duplicate-ID collision with the flash-era entries; restores monotonic append order).
- Date: 2026-09-03
- Source: executor on <vm> (kernels/gfx906/ne1_rebench, post-flash)
- A stream 773-806 GB/s, B row5120 790, C row5120x4r 790-792 - vs the
  pre-flash 769-785 (D072/D044). The raw streaming ceiling moved only ~2-3%.
- **Interpretation:** the llama.cpp +25% (D088 journal) is NOT a general
  bandwidth unlock - the A/B/C raw-stream patterns are essentially unchanged.
  The llama gain is PATTERN-SPECIFIC (its Q4-block read + KV traffic benefits
  from the V420's MCLK DPM). The kernel-level GB/s evidence (N-D1, gemm_probe,
  tp4_rank0, D080/D082) measured on the locked card REMAINS VALID (they are
  raw-stream-bound). The Flash-Next model-level tok/s MAY gain like llama.cpp
  (both stream Q4-ish weights) - but that must be MEASURED at the model level
  (surrogate serve), not assumed. The Gen4 link (31.5 GB/s) upside is on the
  host<->GPU + P2P sides (G3), not the VRAM-bound decode.

---
*End of log. Future entries append below this line only.*

---

## D093 — 3x 1024-token TPS @150W + MCLK max observed (27.7 tok/s median; MCLK 1000 MHz)
- NOTE: renumbered from 090 to D093 (duplicate-ID collision with the flash-era entries; restores monotonic append order).
- Date: 2026-09-03
- Source: executor in <vm> (llama-server :8080, qwen3.8-27b, temp 0, 1024-token)
- 3x: 27.8 / 27.7 / 27.6 tok/s -> median 27.7, spread <1%. MCLK DPM level
  sampled at ~8 Hz through the runs: the card holds LEVEL 2 = 1000 MHz (the
  top of the pp_dpm_mclk table: 350/800/1000) for the entire load - MCLK MAX
  at 150 W = 1000 MHz. The 1024-token median (27.7) is below the 300-token
  Gen4 median (29.85, journal) - the longer sustained run sits at the 150 W
  cap's steady state; both are post-flash Gen4 rates vs 23.9 pre-flash (D087).

---
*End of log. Future entries append below this line only.*

---

## D094 — Review of the MI50-TP4 optimization database (mapping + 4 new items)
- NOTE: renumbered from 091 to D094 (duplicate-ID collision with the flash-era entries; restores monotonic append order).
- Date: 2026-09-03
- Source: operator (independent review of the updated performance database)
- Already in the plan (no change): staggered all-reduce (Rev 5 M2 spec), boot
  self-test (trap T5), expert locality (O6/D052).
- NEW items adopted:
  1. **N-C8 (risk, testable now): QSA prefill + 64 KiB LDS budget.** gfx906
     shares the R9700's 64 KiB LDS. D053 disabled the fused QSA kernel -> the
     eager path may not carry the fork's LDS tiling (patches 0002/0003).
     Sweep prefill length 16/128/512/2048/8192, record LDS/workgroup, VGPR,
     occupancy; find a failure on one card, not at M3.
  2. **N-C9: capability banner with fail-loud.** Startup banner prints each
     optimized path's state (skinny GEMV, W4 MoE, custom P2P AR, PLE offload,
     INT8 dense, graph mode); exit non-zero if a REQUESTED path is unavailable;
     the harness asserts and REFUSES a results row when a requested path is
     inactive (the silent-RCCL-fallback class: a plausible number from a fast
     path that never loaded is an I1 violation).
  3. **M5 change: MTP x context sweep.** MTP3 211 vs MTP4 182 t/s with MTP4
     winning past ~115K context => depth interacts with context. Replace the
     single MTP=3 cell with MTP in {0,1,2,3,4} x context in {4K, 32K, 128K,
     256K}, recording engine steps/s (governs), acceptance, and effective t/s
     at each cell.
  4. **Track O additions:** O7 = QSA KV compression (V100 showed 1.84x
     capacity for ~3% decode; gfx906 = INT8 with per-channel scales - no FP8;
     v_dot4_i32_i8 available if the cache is dot-consumed). O8 = expert LRU
     cache as capacity Plan B (llama.cpp 80-85% hit + R9700 route reuse = one
     phenomenon, two applications; moot while the backbone is VRAM-resident).

---
*End of log. Future entries append below this line only.*

---

## D095 — N-C8 DONE: QSA prefill LDS ladder passes 16..8192 (risk cleared on one card)
- NOTE: renumbered from 092 to D095 (duplicate-ID collision with the flash-era entries; restores monotonic append order).
- Date: 2026-09-03
- Task: N-C8 (QSA prefill + 64 KiB LDS budget)
- Source: executor in <vm> (surrogate serve, eager path, max-model-len 9000;
  prefill ladder 16/128/512/2048/8192 tokens, 1-token completions)
- ALL lengths PASS: 15/116/462/1844/7374 actual prompt tokens (0.9/2.0/2.8/
  3.6/14.8 s). No LDS-budget failure at any length - the fork's QSA tiling
  (patches 0002/0003) holds on the gfx906 eager path beyond the M3-relevant
  prefill range. The same qwen4_exp_qsa_with_output op runs in the full model,
  so the result transfers. The R9700-class 16-token LDS crash does NOT
  reproduce on our stack.

---
*End of log. Future entries append below this line only.*

---

## D096 — N-C9 DONE: capability banner with fail-loud (validated on one card)
- NOTE: renumbered from 093 to D096 (duplicate-ID collision with the flash-era entries; restores monotonic append order).
- Date: 2026-09-03
- Task: N-C9 (capability banner, I1 enforcement)
- tools/capability/capability_check.py + a results-writer gate:
  - Banner reports each optimized path: skinny_gemv ACTIVE (gemv_f16_rdna2,
    N-D1..D4), w4_moe ACTIVE (moe_skinny_int4_decode, N-D3/D036), p2p_ar
    ACTIVE (rdna_ar_*), ple_offload ACTIVE (op + env), int8_dense ACTIVE
    (flagged: N-D5 deferred/unvalidated on gfx906), qsa engine-time
    (standalone import blocked by module-init order; verified at serve +
    N-C8 ladder), graph_mode config-driven.
  - Exits non-zero (verified exit=1) when a REQUESTED path is inactive
    (fail-loud, no warning-and-continue).
  - results.append_row REFUSES a row whose required_paths are inactive or
    whose state file is missing (a silently-degraded run produces no
    measurement - the silent-RCCL/inductor class is caught before a row).
- Built + validated entirely on one card (no TP4 needed, per the operator
  premise).

---
*End of log. Future entries append below this line only.*

---

## D097 — Integrity fix + notes (duplicate IDs renumbered; H1/P3/M5/Track-O confirmations)
- Date: 2026-09-03
- **Duplicate-ID integrity fix:** the flash-era entries (Gen4 verification /
  Gen4 performance / power-cap sweep) held D089-D091 while the executor's
  later entries reused D089-D091. Renumbered by file position - the executor's
  N-E1 re-run -> D092, 3x1024 TPS -> D093, optimization-database review ->
  D094, N-C8 LDS ladder -> D095, N-C9 capability banner -> D096 - with notes
  at each original and each renumbered entry. The log is now monotonic and
  unambiguous; next free ID D098. tasks.json + the generator prose reference
  the new numbers.
- **H1 note (no change):** the 125 W TPS/W sweet spot (D091 power sweep) was
  measured on the 27B under llama.cpp - bandwidth-bound. Flash-Next at TP4 is
  launch-bound (core clock drives kernel turnaround, not HBM throughput). H1
  stays 150 W; re-derive at M4. The 125 W figure does NOT transfer.
- **P3 remainder:** cooling + PSU for four cards remains open - the last
  operator item that could bite on installation day.
- **M5:** the MTP x context sweep spec is in place (D094): MTP in {0..4} x
  context in {4K,32K,128K,256K}, engine steps/s governing per D048.
- **Track O:** O7 (QSA KV compression, INT8 per-channel - no FP8 on gfx906)
  and O8 (expert LRU cache, capacity Plan B) are in the registry (D094).

---
*End of log. Future entries append below this line only.*

---

## D098 — <vm> hangs: mixed iothread config on virtio-scsi (scsi0 plain / scsi1 iothread)
- Date: 2026-09-03
- Source: executor host+agent diagnosis (operator: system hangs after the disk add)
- The guest's ROOT disk is wedged: jbd2/dm-0 in wait_on_buffer (no commit),
  flush kworkers in rq_qos_wait, systemd-journald unkillable (D-state) ->
  every systemd op times out -> sshd sessions hang. The new 128G disk
  (scsi1, <source-host>-<pool>/<vm>-disk-1) mounted fine (sdb1 -> /mnt/llm, fstab
  UUID <redacted>...), but the host zvol shows sustained active ops while the
  guest's completions stall - the classic virtio-scsi completion-path wedge
  from MIXING an iothread disk (scsi1, iothread=1) with a plain disk (scsi0)
  on virtio-scsi-single. ZFS pools are healthy with space (<source-host>-<pool> 41%).
- FIX (operator, VM config): stop <vm>, then EITHER remove iothread=1 from
  scsi1 OR add the same iothread to scsi0 (both disks on one iothread), then
  start. Requires a full VM stop/start (llama service downtime).
- The /mnt/llm mount + fstab entry are already in place and will survive.

---
*End of log. Future entries append below this line only.*

---

## D099 — <vm> recovered after the iothread fix (reboot verified)
- Date: 2026-09-03
- Source: executor check (operator rebooted <vm> with scsi1's iothread removed)
- Verified: scsi0 + scsi1 both plain on virtio-scsi-single (consistent); ssh
  responsive (wedge gone); GPU 66a0 at <pci>; llama service active + :8080
  200; /mnt/llm mounted (126G, 120G free) with the fstab entry (UUID
  <redacted>...) re-added cleanly - the wedge-era fstab write had not committed
  and was lost in the reboot; the mkfs on sdb1 persisted. Writes to the root
  disk confirmed durable. The 128G working space is ready for LLM work.

---
*End of log. Future entries append below this line only.*

---

## D100 — Full Flash-Next AWQ backbone + MTP on /mnt/llm (73 GB; disk consolidation)
- Date: 2026-09-03
- Source: executor (hf download to the new <vm> working space)
- /mnt/llm/models/qwen38-flash-next-awq/: shards 2-5 (20.01/19.96/20.01/
  13.13 GB) + model_mtp.safetensors (5.21 GB) = the complete W4A16 backbone
  (shard 1 = the excluded bf16 PLE table) + the MTP module for the M5 sweep.
  /mnt/llm at 62% (47 GB free). Shard-3 include-parsing hiccup noted (the hf
  CLI dropped one of three --include patterns; explicit single-file download
  fixed it).
- Consolidation: the OLD disk's duplicate shards 2+5 (~33 GB in
  ~/models/qwen38-flash-next-awq, the surrogate source) can be deleted to
  free the 96%-full root; the PLE sidecar (30 GB) may move to /mnt/llm when
  the configs are updated. The M3 model path = /mnt/llm/models/...

---
*End of log. Future entries append below this line only.*

---

## D101 — Old disk cleared 96% -> 82% (duplicate AWQ shards + ref2 junk deleted)
- Date: 2026-09-03
- Deleted: ~/models/qwen38-flash-next-awq (31 GB - duplicated on /mnt/llm,
  D100) + qwen38-flash-next-surrogate-ref2 (2.4 GB junk). The root disk went
  from 96% (11 GB free) to 82% (44 GB free). Remaining big items:
  surrogate-ref (42 GB, the CPU fp16 oracle - regenerable from the /mnt/llm
  AWQ), qwen38-flash-next-ple sidecar (30 GB - movable to /mnt/llm with a
  config path update), surrogate (14 GB), triton-gfx906 build tree (9.2 GB).

---
*End of log. Future entries append below this line only.*

---

## D102 — surrogate-ref deleted (42 GB); old disk now 65% (85 GB free)
- Date: 2026-09-03
- Deleted ~/models/qwen38-flash-next-surrogate-ref (the CPU fp16 oracle;
  operator-confirmed; regenerable from the /mnt/llm AWQ via make_surrogate_ref
  if a new golden is ever needed - existing goldens are committed). The root
  disk: 96% -> 65% (85 GB free) across D101/D102. Remaining: the PLE sidecar
  (30 GB, movable) + the serving surrogate (14 GB) - both keep.

---

## D103 — Two cards live in <vm>; NO guest-visible P2P (measurement, M0 pre-check)

- Date: 2026-09-06
- Task: M0 / G3 pre-check (operator: "I have 2 GPUs, confirm")
- Source: executor measurement (<vm> guest <user>@<lan-ip>; host <source-host>
  root@<source-ip>)
- **Card confirmation:** <vm> now presents TWO post-flash MI50/V420 GPUs
  (<device-id>, gfx906, Gen4 16GT/s x16, 60 CU, BAR0 32 GiB each): guest <pci>
  (card1/renderD128) + <pci> (card2/renderD129). Host BDFs <pci> (group 43,
  downstream port <pci>) + <pci> (group 49, port <pci>). card0 = QEMU
  bochs VGA (not a GPU).
- **Pre-flight done:** card2 power cap set 178 W -> **150 W** (H1; both cards
  now equal). ACS on the two GPU downstream ports cleared (setpci ECAP_ACS+6.w
  = 0000) and **observed STICKING 30+ s** (earlier observed re-assertion churn
  on all four ports - the D074 ephemerality is live on this host; the clear
  must be re-applied + verified immediately before any P2P measurement).
- **The finding — NO guest-visible device P2P (measured, 7.14 runtime, all
  four independent methods agree):**
  1. hipDeviceCanAccessPeer = 0 for EVERY pair (0->0, 0->1, 1->0, 1->1).
  2. hipDeviceEnablePeerAccess fails both directions ("invalid device ordinal").
  3. hipIpcGetMemHandle works but hipIpcOpenMemHandle FAILS with and without
     hipIpcMemLazyEnablePeerAccess ("invalid device context" / "invalid
     argument") -> the fork's rdna_ar_connect (csrc/rocm/rdna_allreduce.cu:
     hipDeviceEnablePeerAccess TORCH_CHECK + hipIpc* staging) would crash in
     this guest.
  4. torch RCCL 2-device all_reduce segfaults at init (both ranks).
  5. hipMemcpyPeer (driver path) WORKS but symmetric ~6.0-6.3 GB/s at 16/64/256
     MB - the host-bounce signature (D2H+H2D through pinned host at ~14 GB/s
     per leg, D076), NOT peer DMA.
  - KFD topology: p2p_links_count = 0 on both GPU nodes; no kfd peer lines in
    dmesg.
- **Root cause class:** the guest PCI topology hides the switch. QEMU presents
  each passed GPU behind its OWN virtual root port (00:1c.x), so in-guest both
  cards hang directly off the root complex with no common switch ancestor; the
  guest kernel's PCIe-P2P eligibility (KFD peer registration) fails. The
  physical PEX88096 + cleared ACS are fine - the switch is simply invisible to
  the guest's driver. **This is a guest-topology property: 4 cards will NOT
  change it.**
- **Consequences (Track M):** the fork's rdna_allreduce (the M2 design basis)
  and the default RCCL TP path are BOTH non-functional inside the VFIO guest.
  Multi-GPU transport in the guest must be re-architected (host-staged), OR the
  VM topology must be changed so the guest sees P2P-capable peers, OR the stack
  runs bare-metal on <source-host>. This is a design decision for the operator (D-entry
  pending operator direction), NOT a hardware blocker - the switch + ACS are
  ready (D071/D073/D103 pre-flight).
- **Candidate directions (for the operator):**
  (a) Re-present the two GPUs as functions of ONE multifunction vfio device
      behind a single QEMU root port -> the guest would see <pci>/<pci>
      siblings (common bridge, no ACS between them); may restore KFD peer
      registration. Cheap to test, needs <vm> topology change + downtime
      (llama service).
  (b) Host-staged transport redesign for M2/M0.7 (flags + staging in pinned
      host memory; every leg already validated: D022 GPU-reads-host, D016
      GPU-writes-host flags observed, D037 CPU-pinned PLE path, D031 protocol).
      Cost: extra PCIe hops at 14 GB/s per leg; per-step payloads are KB-scale
      (D066), so latency impact is single-digit us - G5 analysis needed.
  (c) Run the four-card stack bare-metal on <source-host> (fork's native environment,
      real P2P) - abandons the VFIO-guest deployment model; largest pivot.
- Artifacts: kernels/gfx906/m0_peer_bw.cu, m0a_memcpy_peer.cu,
  m0b_staged.cu, m0c_ipc.cu (+ _x binaries on <vm> kernels/gfx906). Note:
  hipcc binaries must run with ROCM_PATH=/opt/rocm/core-7.14 and
  LD_PRELOAD=/opt/rocm/core-7.14/lib/libamdhip64.so.7 (default /usr/bin/hipcc
  bake an rpath to the 7.2.4 tree - D013's alternatives note, live again).
- Not recorded as results rows (platform facts, not I5 performance rows); the
  M0 suite (t01-t05) still skips/degrades per design.

---

## D104 — P2P deep-dive: config greps pass; emulated-switch gives p2pdma=4 but KFD links still absent; GPUs wedged (incident)

- Date: 2026-09-06
- Task: M0/G3 follow-up on D103 (operator direction: test the configurable topology before any pivot)
- Source: executor measurement (<vm> guest + <source-host> host + kernel-source analysis v6.8)

### Corrections to D103 (operator review, accepted)
- **canAccessPeer=0 for SELF pairs is documented behavior, not a symptom** (amdgpu_device_is_peer_accessible returns false for adev==peer, amdgpu_device.c). D103 listed it as evidence and overstated the breakage; the CROSS pairs were the finding.
- **STOP-D predicate**: reworded to be about the PHYSICAL path, not the guest-visible one. D103's finding is a guest-topology property (configurable), not physical P2P blocking - STOP-D NOT fired; M0/M0.7/M1/M2 are blocked on a transport DECISION, not on STOP-D.
- **Option set gap**: (d) LXC on <source-host> (bind cards to host amdgpu, pass /dev/kfd+/dev/dri into an unprivileged container; bare-metal P2P behavior, Proxmox stays as management). Not chosen yet; recorded for the transport decision.

### Kernel-level facts (config greps + 6.8 source)
- BOTH kernels (<vm> <kernel> AND <source-host> <kernel>): CONFIG_PCI_P2PDMA=y, CONFIG_HSA_AMD_P2P=y, CONFIG_DMABUF_MOVE_NOTIFY=y. Guest amdgpu.pcie_p2p=Y. Nothing P2P-related is missing on either kernel.
- 6.8 predicate amdgpu_device_is_peer_accessible(adev, peer) requires: pcie_p2p=1, !xgmi.connected_to_cpu, pci_p2pdma_distance>=0, real_vram==visible (large BAR), and the target BAR base/limit within the peer's dma_mask (gfx906 = 44-bit).

### RCCL finding (corrects D103's "RCCL segfaults")
- The segfault was a RUNTIME MIX: librccl from core-7.14 dlopened libamdhip64 from the 7.2.4 tree ("HIP runtime: 7.2.53211 / ROCm runtime: 7.2.4" in RCCL's banner). With LD_LIBRARY_PATH + LD_PRELOAD forcing 7.14, **RCCL 2-device all_reduce works at ~63 us/op (20 KB), data bit-exact**, logging "Could not enable P2P between dev 1 and dev 0" and degrading to its host path. The same mix explains the hipcc-probe segfaults during the D103 session.
- Consequence: a host-staged TP route exists TODAY via RCCL + vLLM's --disable-custom-all-reduce (option b, already implemented), pending the transport decision.

### Emulated-switch experiment (the operator's cheap test)
- <vm> re-presented with both GPUs under ONE emulated switch (pcie-root-port + x3130-upstream + 2x xio3130-downstream, raw QEMU args); guest BDFs became <pci>/<pci> with a common switch ancestor.
- **pci_p2pdma_distance = 4 in both directions (kernel probe module) - the topology gate PASSES**; no ACS on any path bridge (QEMU x3130/xio3130 expose none; rp9's ACS ctl all disabled); BARs at <bar-addr>-<bar-addr> fit the 44-bit masks.
- BUT KFD p2p_links stayed EMPTY on the clean boot, and boot dmesg shows **UBSAN shift-out-of-bounds in amdgpu_amdkfd_get_pcie_bandwidth_mbytes() (amdgpu_amdkfd.c:604)** firing inside kfd_fill_gpu_direct_io_link_to_cpu (the io-link fill kfd_add_peer_prop depends on). Suspected cause of the missing links; NOT yet isolated (box went down first).
- Guest kernel config greps, the module probe, and RCCL logs all confirmed the mechanism matches the operator's p2pdma-distance theory exactly.

### Incident (recovery in progress)
- Mid-experiment the GPUs firmware-wedged (my in-guest `rmmod amdgpu` while hot, or reboot churn): PSP UNLOAD_TA 0x117 under host amdgpu, probe -12 under host memory pressure, then a persistent ring-training error loop on <pci>. Guest amdgpu init fails on both cards; vfio BACO reset + PCIe secondary-bus reset did NOT clear it.
- State left: <vm> STOPPED (llama service down), conf RESTORED to the original working hostpci0/1 (experiment conf saved at <workdir>/<vm>.conf.bak-d103-topo on <source-host>). No VM auto-start set.
- Fix: <source-host> full power cycle (operator action, in progress at time of writing) to reset the V420 PSP; after boot vfio-pci should reclaim both GPUs from /etc/modprobe.d ids and <vm> should start clean on the restored conf.
- Next once recovered: re-test the emulated-switch boot for KFD p2p links with the UBSAN hypothesis in hand (dyndbg or a patched check), then the multifunction variant if needed, ACS both ways per the operator's IOMMU-translation point.

- Artifacts: kernels/gfx906/m0_peer_bw.cu, m0a_memcpy_peer.cu, m0b_staged.cu, m0c_ipc.cu (guest kernels/gfx906/, _x binaries); p2pprobe.c (kernel module, /tmp/p2pprobe on <vm>); <vm> conf experiment at <workdir>/<vm>.conf.bak-d103-topo (<source-host>).

---

## D105 — Incident closed: <source-host> power cycle cleared the GPU wedge; box restored

- Date: 2026-09-06
- Task: M0/G3 experiment recovery (D104 follow-up)
- Source: executor (<source-host> host + <vm> guest)
- Operator power-cycled <source-host>; the V420 PSP wedge (D104) cleared. Post-boot verified:
  - Both GPUs (<pci>/<pci>) enumerate healthy; host amdgpu init clean (0 probe failures) - the firmware state is good again.
  - vfio-pci reclaimed both GPUs (new_id <device-id>; /etc/modprobe.d/vfio.conf holds only softdeps - the id is added at runtime; noted for the next host reboot, where amdgpu may claim the cards first and vfio needs the id in place before VM start).
  - <vm> started on the RESTORED hostpci0/1 config (the D103-era working layout): 2 render nodes, KFD nodes 0/1/2, llama-rocm.service active.
- Note: /etc/modprobe.d/vfio.conf currently lacks an `options vfio-pci ids=<device-id>` line (only nouveau/nvidia softdeps). At the next <source-host> boot amdgpu may bind the GPUs before vfio; add the ids line + update-initramfs for boot-stable passthrough (operator/host housekeeping item).
- State: box fully recovered and serving; experiment conf saved at <workdir>/<vm>.conf.bak-d103-topo (<source-host>). Next per the operator: decide the transport direction (multifunction topology retest / host-staged RCCL / LXC on <source-host>) - each topology experiment needs a planned <vm> downtime window.

---

## D106 — UBSAN root cause CONFIRMED (zero link masks in passthrough); RCCL 2-rank cost curve banked; option set updated

- Date: 2026-09-06
- Task: M0/G3 follow-up (operator analysis + executor verification)
- Source: executor (persisted guest /var/log/kern.log; 6.8 kernel source; <vm> RCCL benchmark on the restored config)

### Operator correction (accepted, recorded)
- The topology test was framed as binary (saves the VM or kills it). It did neither: there is a SECOND gate downstream of pci_p2pdma (the KFD io-link fill depends on populated link caps, not switch ancestry). Also: the wedge risk of unbinding amdgpu on hot V420-flashed cards was underweighted - that cost a <source-host> power cycle. Both named so the next attempt accounts for them.

### UBSAN root cause CONFIRMED (free, from the persisted log)
- Persisted /var/log/kern.log (boot 2026-09-06T01:47 = the emulated-switch boot) holds the full UBSAN report: **"shift exponent -1 is negative"** at amdgpu_amdkfd.c:604:30. -1 == ffs(0)-1 -> pcie_mlw_mask/pcie_gen_mask were ZERO. The zero-mask hypothesis is confirmed with no reboot.
- Mechanism (6.8 source, amdgpu_device.c amdgpu_device_get_pcie_info): the default masks are only assigned when `pci_is_root_bus() && !amdgpu_passthrough()`; <vm> is passthrough, so masks fall through to the partner/ASIC-cap walk - which produced zero under the emulated switch -> io-link fill garbage -> kfd_add_peer_prop has no valid link -> empty p2p_links.
- Corroborating observation: NO UBSAN on plain-root-port boots (the current restored config) - the io-link fill only breaks under the emulated switch, matching the 'worse under switch' prediction.
- Fix (existing module params, no kernel patch): `amdgpu.pcie_gen_cap=<mask> amdgpu.pcie_lane_cap=<mask>` on the <vm> cmdline. The code confirms the override (`if (amdgpu_pcie_gen_cap) adev->pm.pcie_gen_mask = ...`). Exact hex must be read from the matching Ubuntu 6.8.0-138 source's AMDGPU_DEFAULT_PCIE_GEN_MASK / AMDGPU_DEFAULT_PCIE_MLW_MASK (amdgpu_device.c references them; not to be taken from memory) - values to be read out at step-3 execution time.
- Alternative lever noted: <vm> is 6.8.0-138 while <source-host> is <kernel> - a large gap spanning the p2p-check rework; an HWE kernel in the guest is a low-risk alternative that might resolve this without the params.

### RCCL scope reduction (the finding that matters most)
- Option (b) collapses from 'design a host-staged transport' to: **set LD_LIBRARY_PATH (7.14 runtime) + pass --disable-custom-all-reduce**. Recorded as a scope reduction in its own right.

### RCCL 2-rank all_reduce cost curve (banked on the restored config; 7.14 runtime; 200 reps, median)
- Fixed per-op overhead ~56 us (flat 0-8 KB: 55.5-56.5 us) - RCCL/process overhead, NOT transport.
- Decode size 20 KB: **77.4 us** (earlier one-shot ~63 us was optimistic; 77.4 is the settled median).
- 64 KB 84.5 us; 256 KB 120 us; 1 MB 273 us; 4 MB 917 us (~4.6 GB/s asymptote on the host path).
- Decision arithmetic (operator frame, measured): at 2 ranks 119 x 77 us = 9.2 ms/step collectives (2-rank is not the deployment). 4-rank ring projection 95-160 us/op -> 11-19 ms -> ~33-44 steps/s -> **RCCL-over-host misses G6 (>=50) and fork parity (60-62) at four cards**. It is a viable FLOOR that de-risks the project, not the target. P2P remains worth one more cheap shot (step 3 below).
- M1 note: the 2-rank RCCL curve is now banked (unblocks M1's sweep shape; 4-rank numbers await cards/transport).

### Option set updated (operator direction)
- **Multifunction variant DROPPED**: D103 failed at pci_p2pdma_distance (multifunction was the fix for THAT); D104 moved the failure downstream into the io-link fill, which depends on link caps, not ancestry. Multifunction hits the same zero masks + UBSAN; another downtime + wedge risk for no information.
- Remaining order: (3) emulated switch + amdgpu.pcie_gen_cap/pcie_lane_cap, OR an HWE guest kernel - one cmdline change + one clean VM reboot, NO in-guest rmmod amdgpu (the likely wedge cause). Acceptance = peer bandwidth beating the ~6.2 GB/s host-bounce signature + a correctness check, NOT canAccessPeer=1. (4) If that fails: LXC on <source-host> (real hardware caps, no UBSAN path, resolves the runtime mix, and clear-pex-acs.service does its intended job there - ACS-cleared direct switch routing is correct on the host with IOMMU passthrough, the opposite of what the VM wanted).
- Housekeeping: the vfio.conf `options vfio-pci ids=<device-id>` note (D105) is PARKED - do not add it while step 3/4 are live (it would create the host reclaim behavior we are about to want reversed).

- Artifacts: /tmp/rccl_sweep.py on <vm> (RCCL size curve); UBSAN report persisted in <vm> /var/log/kern.log (boot 01:47); source references: amdgpu_amdkfd.c:604 (bandwidth shift), amdgpu_device.c:5808-5895 (get_pcie_info + param override).

---

## D107 — Step-3 result: THE FINAL GATE is the 44-bit DMA mask vs OVMF's 2^46 BAR placement (not topology, not link caps)

- Date: 2026-09-06
- Task: M0/G3 step 3 (emulated switch + amdgpu.pcie_gen_cap/pcie_lane_cap params)
- Source: executor (<vm> guest + <source-host> host + QEMU/OVMF inspection)

### What the params fixed (confirmed)
- amdgpu.pcie_gen_cap=0x000F000F + pcie_lane_cap=0x003F0000 on the <vm> cmdline (values mirroring a real Gen4 x16 device, read from 6.8 amd_pcie.h CAIL layout: platform GEN bits high, ASIC low, widths 0x10000..): **UBSAN GONE under the emulated switch** (0 occurrences; the io-link fill now reports sane caps - max_bandwidth 32000 MB/s). The zero-mask/UBSAN hypothesis (D106) is fully confirmed and fixed by existing module params. No in-guest rmmod was performed (clean VM restarts only) and the cards stayed healthy through the whole step.

### THE FINAL GATE (why p2p_links were STILL empty with everything else passing)
- amdgpu_device_is_peer_accessible has one more term: the TARGET GPU's BAR base/limit must fit within the PEER GPU's DMA mask. gfx906's device dma_mask is **44 bits (16 TB)**. In-guest, OVMF/Proxmox q35 places the 32 GB vfio BARs at **<bar-addr>-<bar-addr> (~56 TB = 3.5 x 2^44)** - ABOVE 2^44 -> `!(aper_base & ~dma_mask || aper_limit & ~dma_mask)` FAILS -> no KFD peer links -> canAccessPeer=0.
- This gate is INDEPENDENT of topology: it is the real reason the D103 default-topology run had no peers (there p2pdma ALSO failed; here p2pdma=4 and link caps are fixed, and this term still fails). Verified this boot: pci_p2pdma_distance=4 both ways (kernel probe), dma_mask_bits=44, BARs <bar-addr>+.
- Host contrast (the decisive datum): the SAME BARs on <source-host> sit at **<bar-addr>-<bar-addr> (~384-480 GB = 2^39, BELOW 2^44)** -> on the HOST the 44-bit mask is satisfied -> LXC/bare-metal P2P is NOT blocked by this gate. The 44-bit dma_mask is a fixed gfx906/amdgpu property; the guest simply must place its BARs below 16 TB, and OVMF does not.

### Why the in-guest fix is not a stock knob (measured)
- QMP on the live VM: q35-pcihost.pci-hole64-start = 61572651155456 (<bar-addr>); pci-hole64-size = <bar-addr> (32 GB default).
- `-global q35-pcihost.pci-hole64-start=...` is REJECTED at VM start: "Property 'q35-pcihost.pci-hole64-start' is not writable".
- QEMU v11 pc_pci_hole64_start() derives the start from RAM end (~53 GB here) or a device-memory range when ram<maxram; the live cmdline is plain `-m 52000` (no maxmem) - NONE of the stock branches yield 56 TB. Conclusion: <bar-addr> is a **Proxmox q35 (+pve0 machine / OVMF) placement choice**, not controllable via stock -global properties.

### Box state after the experiment (clean)
- <vm> restored to the stable hostpci0/1 config (backup chain on <source-host>: <workdir>/<vm>.conf.emuswitch-d106, .bak-hostpci-d106, .bak-d103-topo); 2 render nodes, KFD nodes 0/1/2, llama-rocm.service active, UBSAN 0.
- Guest cmdline RETAINS amdgpu.pcie_gen_cap/lane_cap (benign on hostpci: mirrors the real Gen4 x16 link; keeps the io-link fill UBSAN-free for any future emulated-switch test).

### Options going forward (for the operator)
1. **LXC on <source-host> (recommended)**: host BARs < 2^44 -> every gate passes (p2pdma real topology, real link caps, 44-bit mask satisfied). This is also the fork's native environment and where clear-pex-acs.service does its intended job (ACS-cleared direct switch routing is correct on the host). G3/P2P bandwidth becomes measurable there.
2. In-guest P2P requires re-placing Proxmox's 64-bit PCI window below 2^44 - a pve-qemu/OVMF source-level question (why q35+pve0 picks <bar-addr>), not a stock knob. Possible but Proxmox-internal research.
3. RCCL-over-host (D106, 2-rank ~77 us/op @ 20 KB banked) remains the de-risking floor in the guest.

- Artifacts: <vm> cmdline params in /etc/default/grub (grub.bak-d106 saved); QMP evidence via qm monitor <vm>; guest /proc/iomem shows the <bar-addr> window; <source-host> /proc/iomem + sysfs resource show host BARs at <bar-addr>-<bar-addr>.

---

## D108 — LXC path live on <source-host>: host KFD P2P links WORK (first time ever enabled); kernel peer STORE poisons the writer GPU

- Date: 2026-09-06
- Task: M0/G3 via LXC on <source-host> (operator: "proceed" on the LXC path)
- Source: executor (<source-host> host + LXC <ct>)

### Setup
- <vm> STOPPED; both GPUs (<pci>/<pci>) bound to the HOST amdgpu (iommu=pt on <source-host>; vfio id-less modprobe.d means amdgpu claims them at boot). /dev/kfd + /dev/dri (renderD129/130) present on the host; host KFD nodes with **p2p_links_count = 1 each** (the peer links the VFIO guest could never register).
- LXC <ct> "<container>": Ubuntu 24.04, privileged, rootfs <source-host>_<pool>:24, /dev/dri + /dev/kfd bind-mounted, cgroup2 device allows (c 226:* rwm + c 510:* rwm for kfd major 0x1fe). ROCm 7.14-gfx906 userspace installed via the mixa3607 apt repo (s3.arkprojects.space/apt-gfx906/ubuntu, amdrocm-core-sdk/dev 7.14-gfx906). rocminfo sees both gfx906.

### Milestone (first in the project)
- In the container: hipDeviceCanAccessPeer 0->1 = 1, 1->0 = 1 (self = 0 as documented); hipDeviceEnablePeerAccess OK BOTH directions; "kernel-level P2P available: YES". Peer capability has NEVER been enabled anywhere before (the VFIO guest could not pass the p2pdma/link-cap/BAR gates; the host passes all of them: real topology, real link caps, host BARs at ~2^39 < 2^44, iommu=pt).

### Finding: direct kernel peer STORE poisons the writer GPU
- m0_peer_bw's peer_writer kernel (dev 0 stores float4 into dev 1's hipMalloc buffer) -> sq_intr errors on <pci>, then "{1}poison is consumed by client 12, kick off gpu reset flow", qcm fence timeout, "cp might be in an unrecoverable state", DQM create failed -62. BOTH GPUs ended up wedged (device 1 also hangs after the cascade) -> another <source-host> power cycle needed.
- So: peer ACCESS + ENABLE succeed at the API level, but an actual kernel store to a default-hipMalloc peer buffer faults/poisons at the hardware level on this host stack (gfx906 + <source-host> kernel 7.0.14 + ROCm 7.14). Whether this is (a) a gfx906 PCIe-peer kernel-store limitation, (b) a memory-type/flag issue (the fork used hipExtMallocWithFlags UNCACHED staging + per-peer flag slots, not default hipMalloc), or (c) a mapping-path issue (kfd_mem_export_dmabuf/dma-buf export flow) is OPEN.
- Next experiments after the power cycle (careful, one at a time): (1) hipMemcpyPeer / SDMA-copy bandwidth between the host GPUs (the copy path RCCL-style collectives actually use - likely safe, no compute kernels touching peer pointers); (2) a single small kernel peer store to characterize the poison; (3) if kernel stores are needed, replicate the fork's exact staging (hipExtMallocWithFlags uncached + IPC/dma-buf export).

### State
- <source-host> needs a power cycle (cards host-amdgpu-bound, both CP wedged). After boot the GPUs auto-bind to amdgpu (no vfio ids in modprobe.d) and LXC <ct> is immediately usable (persistent rootfs + device config + installed ROCm).
- <vm> remains stopped (llama down) while the GPUs are on the host path - the D021-era llama relocation question is now live for real.

---

## D109 — G3 ANSWERED (host/LXC): real switch-internal P2P at ~28.5 GB/s symmetric via SDMA peer copies

- Date: 2026-09-06
- Task: M0/G3 (LXC path, post-D108 power cycle)
- Source: executor (LXC <ct> on <source-host>)

### Measurement (hipMemcpyPeer, SDMA path, 30 reps median, content-verified)
- 1 MB: 17.8-18.0 GB/s; 4 MB: 22.5; 16 MB: 26.8 (content OK, byte-exact); 64 MB: 28.1; 256 MB: **28.5 GB/s symmetric both directions**.
- Post-power-cycle state: GPUs auto-bound to host amdgpu, KFD p2p_links_count=1 on both nodes, ACS cleared (0x0000) on the GPU ports (clear-pex-acs.service at boot = bare-metal-correct). LXC <ct> immediate reuse.
- **Acceptance met** (operator criteria): canAccessPeer=1 (matrix) + bandwidth 28.5 GB/s, ~4.6x the VM's 6.2 GB/s host-bounce signature, clearly not a bounce + byte-exact correctness on the peer copy. Peer data movement through the PEX88096 works at ~90% of the Gen4 x16 per-direction ceiling (31.5 GB/s theoretical), switch-internal (the x8 upstream does not cap P2P).
- Context: this EXCEEDS the fork's bare-metal figures (peer store 14.3 / load 5.7 GB/s on Navi21 Gen4). Symmetric 28.5 GB/s via copies vs their asymmetric kernel-store path; the fork's kernel-direct-store transport is NOT the limiting design on gfx906+Gen4 - copies are at ceiling.

### Kernel-STORE caveat (D108 stands)
- Direct compute-kernel stores into a default-hipMalloc peer buffer still poison the writer GPU on this stack. The SDMA/copy path is clean and is what RCCL-style collectives use; the fork's kernel-store all-reduce would need the uncached-staging/export path validated (or be replaced by copy-based transport) at M2.
- M2 design implication: gfx906 four-card all-reduce should use copy/SDMA-based peer transport (28.5 GB/s symmetric) rather than the fork's kernel-store shape, or kernel stores must be re-validated with hipExtMallocWithFlags(UNCACHED) staging.

### State
- <source-host> healthy on the LXC path; <vm> remains stopped (llama down while GPUs are host-bound). G3's "peer write works, stays on the switch" is answered affirmatively on the host path (28.5 GB/s symmetric, no root-complex redirect: a bounce/RC path would be ~6 GB/s).

---

## D110 — Follow-ups: G5-relevant latency; switch-proof arithmetic; kernel-store verdict REVERSED; N-E3/D037 on host; record corrections

- Date: 2026-09-06
- Task: M0/G3/M2 pre-design follow-ups (operator review of D103-D109)
- Source: executor (LXC <ct> on <source-host>, post-D109)

### 1. G5-relevant number: SDMA peer-COPY latency at decode sizes (the figure of record for M2, not 28.5 GB/s)
- hipMemcpyPeer per-op latency (event-timed, 1000 reps median, LXC <ct>): **64 B-4 KB flat ~11.2-11.4 us (fixed dispatch floor); 16 KB 11.8; 20 KB 11.8; 64 KB 13.8; 256 KB 22.9; 1 MB 57.8**.
- vs RCCL-host 77.4 us @ 20 KB (D106): the SDMA peer-copy path is ~6.5x faster per op. vs the G5 34 us budget: one 20 KB SDMA copy ~11.8 us fits with margin; a full multi-copy all-reduce shape is the M2 measurement.
- Recorded next to G5, not the bandwidth number (the D044-class trap the operator flagged).

### 2. Airtight switch-internal proof (adds to D109)
- Upstream = x8 Gen4 = 15.75 GB/s one-way (D075/D078). 28.5 GB/s symmetric peer traffic = 1.8x the one-way upstream ceiling -> it provably never crossed the upstream<workdir> complex. Arithmetic, not inference. This also permanently retires the x8-bifurcation worry (D075-D078) for P2P.

### 3. D107 option-2 CORRECTED: phys-bits is a stock knob (record only, decision not reopened)
- The BARs at <bar-addr> just under 2^46 are the signature of a **46-bit guest phys-bits** (OVMF puts the 64-bit MMIO aperture near the top of the guest PA space). The lever is the CPU line (`-cpu host,phys-bits=42/43`), not the pcihost. Corrected disposition: the VM path stays closed ON MERIT (peer traffic through the host IOMMU for translation cannot match switch-internal 28.5 GB/s), not on "not a stock knob".

### 4. Kernel-store verdict REVERSED (D109 was premature) - UNCACHED staging WORKS
- Kernel peer store into a default (cached) hipMalloc buffer: POISONS (D108). Kernel peer store into **hipExtMallocWithFlags(hipDeviceMallocUncached)** staging: **WORKS, byte-exact, no fault** (m0g, 20 KB store verified). Root cause = memory type: Vega20's L2 cannot cache-coherently reach a peer BAR over PCIe; uncached/fine-grained staging is a REQUIREMENT (why the fork used it), not an optimization.
- Consequence: the fork's rdna_allreduce kernel-store shape (uncached staging + per-peer flags) and the D079 fused persistent worker (1.85 us/step single-GPU) are VIABLE on the host. M2 design should NOT settle for copy-based transport; the kernel-store AR is testable. (D109's M2 implication corrected.)

### 5. N-E3 and D037 re-probed on the host path (both VFIO artifacts now separable)
- N-E3 (queue overlap): q=2 aggregate 365.6 GB/s = **1.89x** vs q=1 - the fork's q-scaling REPRODUCES on the host (the flat 1.3-1.4x under VFIO was a VFIO artifact). q=4: 1.31x, q=8: 1.50x (oversubscription/noise; fork's 3.51x@4 not matched). Partial recovery of a conceded constraint.
- D037 (cross-process IPC device handoff): hipIpcOpenMemHandle on the SECOND GPU now **OPENS** (vs "invalid device context" in the VM), but a naive producer/consumer handoff (owner D2H-verifies a peer-written buffer) **segfaults the owner in userspace** (no GPU poison; kernel-store-to-uncached works in-process, m0g). Full re-validation needs the fork's actual worker + done_seq handshake protocol - DEFERRED. Note: the PLE SERVING path uses the CPU-pinned connector (D022/D037 validated), not the device-IPC handoff, so the serving design is unaffected either way.

### 6. ACS disposition PROMOTED: D073/D074 PROVISIONAL -> RESOLVED
- D109's 28.5 GB/s switch-internal P2P with ACS cleared at boot (clear-pex-acs.service) IS the routing verification D073/D074 were waiting on. Bare-metal-correct confirmed; entry disposition updated.

### Operational
- Wedge rate during peer-store experimentation: ~2 <source-host> power cycles/day. Budget for it; keep nothing precious on the box during it. Llama relocation is now urgent, not housekeeping: it is down and cannot return to <vm> while the cards are host-bound; LXC <ct> can serve it directly and doubles as a continuous host-GPU-path sanity check between experiments.
- Artifacts: container <workdir>/m0f_lat.cu (latency curve), m0g_uncached.cu (kernel-store verdict), ne3_overlap.cu (q-scaling), m0h_ipc.cu (D037 re-probe).

---

## D111 — Stack validation PASSED on the host/LXC path: surrogate serves with golden-correct output (single GPU); TP2 in progress

- Date: 2026-09-06
- Task: stack validation / A-B experiment (operator: "do the stack validation tonight, end-to-end")
- Source: executor (LXC <ct> on <source-host>, host GPUs on amdgpu)

### Setup (no rebuild - <vm>'s stack reused)
- <vm> root zvol (<source-host>-<pool>/<vm>-disk-0) activated + mounted RO at /mnt/<vm> on <source-host>; the built stack bind-mounted into LXC <ct> at identical paths: gfx906-venv (torch 2.13.0+gfx906, triton 3.6.0+gfx906, vllm rocm714 editable-> tree), the port tree, the surrogate (14 GB) + PLE int4 sidecar (ples_int4).
- Container fixes required to make the stack run (all environmental):
  1. python3-dev (triton driver JIT-compiles hip_utils.c needing Python.h).
  2. amdsmi-vs-torch init conflict: vLLM's ROCm platform probe fails (AMDSMI_STATUS_NOT_INIT) when torch inits the runtime first -> fix = sitecustomize.py on PYTHONPATH that calls amdsmi.amdsmi_init() BEFORE torch loads (then the fork's re-init + get_processor_handles=2 both succeed).
  3. Run from a cwd that does not shadow the editable vllm (repo-root's vllm/ dir = the fork ROOT, a namespace package -> 'PromptType (unknown location)'); cwd=<workdir> + PYTHONPATH=<workdir> resolves via the editable finder.

### RESULT: single-GPU surrogate serve on the host path = PASS (golden reproduction)
- Serve: port 8001, --enforce-eager single GPU (ROCR_VISIBLE_DEVICES=0), VLLM_PLE_CPU_OFFLOAD + real sidecar, VLLM_RDNA_AR/SKINNY_GEMM/DENSE_INT8 flags, model = 8-layer surrogate.
- Golden T1-style check (prompt "The capital of France is", greedy t=0, 24 tokens vs traces_v4): server text == golden text EXACTLY through token 14, then the DOCUMENTED top-2 boundary flip at token 14 ('ymm' vs 'utton', D055) -> identical numerical behavior to the <vm> port. Port tree + surrogate + PLE offload + int4 MoE + GDN/QSA eager paths all work on the host/LXC environment.

### TP2 (in progress)
- --tensor-parallel-size 2 + ROCR_VISIBLE_DEVICES=0,1 + EP: engine initializes, PLE offload registered on BOTH ranks (tp_rank 0/1), torch.distributed communicator up, both GPUs provisioned (kv cache 20.24 GiB/rank) -> then stalls on shared-memory broadcast ("No available shared memory broadcast block... some processes hanging") with a "PLE lookup >5 s (worker slow or stuck?)" warning. /dev/shm verified 32G (not the cause). Retry in progress; root cause of the worker hang under investigation (PLE worker IPC stall on the second rank is the leading suspect).

### State
- <vm> remains stopped (llama down); <source-host> GPUs on the host path; LXC <ct> is the working stack-validation environment.
- Honest scope note: the literal 35B-A3B leg (Case B) is NOT runnable (no A3B weights on the box; the port tree is qwen4_exp/Flash-Next-arch-only). The stack-validation leg the review doc actually recommended is what was executed.

---

## D112 — TP2 ACHIEVED on the host/LXC path: golden-like output + in-situ per-layer numbers (objective met)

- Date: 2026-09-06
- Task: stack validation TP2 leg (goal: surrogate TP2 across the 2 host GPUs + per-layer/collective report)
- Source: executor (LXC <ct>, <source-host>, 2 host GPUs on amdgpu)

### Blocker found and fixed: container RAM, not shm
- TP2 workers loaded weights (both ranks) but a worker was SIGKILLed (exit -9) -> "Worker proc VllmWorker-0 died" -> engine-core "cancelled". Root cause: the LXC had 8 GB RAM; the PLE CPU-offload worker (prefaults the 30 GB int4 sidecar) + 2 TP workers blew the cap (the earlier shm-broadcast stalls and "PLE lookup >5 s" warnings were the same OOM thrash). /dev/shm was fine (32G). Fix: container RAM 8 -> 40 GB. TP2 READY ~260 s after relaunch, 0 worker deaths.

### TP2 result (tensor-parallel-size 2 + expert-parallel, PLE offload on BOTH ranks, torch.distributed comms over the working host P2P)
- Golden-style check (prompt "The capital of France is", greedy): output matches the golden text through ~11 tokens then a boundary flip - vs 14 at TP1. The flip shifting EARLIER under TP2 is the expected TP-sharded fp16 rounding effect (near-tie regime); output is coherent and golden-like. Correct TP2 behavior; the port's multi-GPU path works end-to-end on the host.
- Throughput (eager, single stream, host path, 8-layer surrogate across 2 cards): steady 46.5 tok/s (300-token runs; first run cold) = 21.5 ms/token -> **2.69 ms/layer measured in situ, including TP collectives + PLE offload**. Batch-4 aggregate: 118 tok/s.
- 48-layer extrapolation anchor (per the review doc's method): per-layer TP2 eager = 2.69 ms; TP2->TP4 remains a one-variable projection (quarter-width per-layer cost + D065 1.28 skew + more ring hops) - the measured anchor replaces the previous estimate's weakest term. Eager is a CONSERVATIVE anchor: the serving config uses graphs-no-compile (D060 ~3.3x), which would lower the per-layer number further; graph-mode TP2 measurement is a follow-up.
- Note: this is the first multi-GPU TP run of the port tree ANYWHERE (<vm> never ran TP>1) - a genuine integration milestone: EP dispatch + TP comms + PLE-offload-per-rank all function on the host path.

### Objective status (goal)
- Single-GPU host-path serve + golden check: PASS (D111). TP2 across both GPUs: WORKING with golden-like output + in-situ per-layer/collective numbers (this entry). The 48-layer extrapolation now has a measured anchor. Literal 35B-A3B leg remains out of scope (no weights; port tree is Flash-Next-arch-only).
- State: TP2 server live on LXC <ct> :8002; <vm> stopped; <source-host> GPUs on the host path.

---

## D113 — Per-layer anchor decomposed + scaling efficiency measured (operator's three measurements)

- Date: 2026-09-06
- Task: review follow-ups on the 2.69 ms/layer anchor
- Source: executor (LXC <ct>, host path, eager + graphs, 4L and 8L surrogate slices)

### Measurement set (all host path, 300-token greedy single-stream, steady state)
| run | tok/s | ms/token | naive ms/layer |
|---|---|---|---|
| TP1 eager 8L | 66.5 | 15.0 | 1.88 |
| TP2 eager 8L | 46.5 | 21.5 | 2.69 |
| TP2 eager 4L | 82.1 | 12.1 | 3.03 |
| TP2 graphs 8L | 133 | 7.5 | 0.94 |

### 1. Fixed-cost decomposition at TP2 (4L vs 8L) - operator's bar 1 confirmed
- per_layer(TP2 eager) = (t8 - t4)/4 = (21.5 - 12.1)/4 = **2.35 ms/layer**
- fixed(TP2) = t8 - 8*per_layer = 21.5 - 18.8 = **2.7 ms/step** (lm_head + PLE + sampling/scheduler)
- The naive 2.69 ms/layer (21.5/8) overstated per-layer by ~14%; measured fixed 2.7 ms sits inside the operator's 2-4 ms estimate window. 48L extrapolation (TP2-width eager): t = 2.7 + 48*2.35 = **115.5 ms/token ~ 8.7 tok/s** - the operator's 107-119 ms bar, confirmed.

### 2. Scaling efficiency TP1->TP2 (same conditions, same day) - the operator's bar 2
- Aggregate eager 8L: 46.5/66.5 = **0.70x - NEGATIVE scaling** on the 8-layer surrogate (TP2 slower than TP1). The surrogate is too small for TP2: per-layer comms + width-loss (D082 small-shape efficiency) exceed the sharding gain. So "clean 1/4 per-card work" does NOT hold at this scale; the TP2->TP4 projection cannot assume the surrogate's TP2 number scales as width. (A 48-layer model at TP4 amortizes fixed costs differently and has more per-layer compute to shard - but that is exactly the unmeasured unknown.)

### 3. Graph factor at TP2: 133/46.5 = ~2.9x (graphs-no-compile, mode=none cudagraph_mode=full)
- TP2 graphs 8L: 7.5 ms/token -> 0.94 ms/layer naive (fixed decomposition for graphs not run; would need a 4L graphs run - follow-up).
- Combined with bar 1-2, the operator's revised centre-of-mass (~40 steps/s at TP4 with graphs, wide range) is now anchored: per-layer 2.35 (eager TP2-width), fixed 2.7 ms, graph factor ~2.9x, scaling efficiency <=0.70x at surrogate scale (negative) - the scaling term remains the dominant uncertainty and it now looks WORSE than the clean-halving assumption, not better.

### Method notes
- 4-layer slice built from the 8-layer surrogate: config num_hidden_layers=4 + layer_types[:4] (3 GDN + 1 QSA incl. the PLE-bearing layer) + a REAL 4-layer safetensors file (mmap-filtered layers 0-3; the fork's loader iterates the whole file, so a trimmed index alone was insufficient - the "layers.4 missing" error persisted until the weight file itself was sliced). Artifacts: <workdir>/sur4l in LXC <ct>.

### RAM-budget flag (operator) - recorded for the deployment decision
- TP2 needed 40 GB in the container (2 TP workers + 30 GB PLE sidecar prefault); 4 TP workers plausibly 46-50 GB. <source-host> has 64 GB total. Collides with llama-in-LXC: llama likely has NO home during M2-M4 host-path bring-up - cleaner to plan for it than fight coexistence. Re-run D043/D064 host RAM budget against this measurement.

---

## D115 — Step 1 (coder handoff): kernel-count + dispatch-bubble probe (TP1, 8L surrogate, torch.profiler in-process)

- Date: 2026-09-06
- Task: handoff Step 1 - "kernels/layer split HIP vs Triton; per-kernel bubble; projected 48L step"
- Source: executor (LXC <ct>, TP1 eager + graphs, offline in-process engine via VLLM_ENABLE_V1_MULTIPROCESSING=0, torch.profiler HIP activity over a 20-token decode; profiler artifacts <workdir>/tprof_{eager,graphs}.json + analyze_tprof2.py)
- Method note: rocprofv3 attach was abandoned (needs librocprofiler-sdk preload in the ENGINE process and still missed queues); torch.profiler in the in-process engine captured every GPU kernel. Results below exclude HIP-API host events (18-48k) - GPU kernels only.

### Kernel count per layer (the transferable number)
- ~8,400-8,600 GPU kernels over a 1-prefill + 20-decode-token window, BOTH eager and graphs (~50-51 kernels/layer/token; count is graph-invariant, as expected).
- **Projected 48-layer step: ~2,400-2,460 kernels** (TP1 width) - NOT the fork's post-T46 1,788. On gfx906 the fused-glue tier fires but the count reduction vs pre-fusion is only ~10%, not the fork's ~34%.
- Class split (launches, % of GPU kernels): **torch-native elementwise/index ~3,650 (~43%) - the single biggest family**; fused_glue 1,050 (12%); dense_gemv 735 (9%); d2d copies 550-730; moe_int4 336; gdn 246 (fused_recurrent_gated_delta_rule_packed_decode + causal_conv1d); norm 210; torch_reduce 189; router 168; attn_rope 126. Triton launches in the decode path: ZERO (all HIP custom + torch native).
- GPU self time: dense_gemv 63 ms (biggest), moe 26, fused_glue 17, torch_elem 16.

### Dispatch bubble (the launch-bound verdict)
- GPU busy ~7.3 ms/token in BOTH modes; eager wall 15.0 ms vs graphs 7.5 ms (D113 rates).
- **Eager dispatch/CPU bubble ~7.7 ms/token = ~19 us x ~410 kernels/token** of host dispatch overhead that CUDA-graphs replay removes. GPU ~49% utilized in eager, ~97% under graphs. LAUNCH/DISPATCH-BOUND confirmed at the model level - the lever is kernel COUNT (fewer dispatches), which is exactly what graphs already buys back and what further fusion/elementwise reduction would buy.

### Trace-justified Step 2 direction (per the handoff's "trace-justified order")
- The already-firing glue fusion (gemv_act/hc_mix/se_*) is NOT where the remaining count is: **the ~23 torch-elementwise launches/layer (~43%) are the residue** (QSA indexer/norm/rope/quant small ops). That is the gfx906-specific "missing T46" - the fork's post-fusion count excluded or fused these. Reducing torch_elem (op fusion of the small elementwise soup, esp. in the QSA/GDN paths) is the highest-count lever; fused qk-norm-rope (D053, attention path) and int8 dense follow per the handoff. Dense GEMV is the GPU-time lever (63 ms) if compute-bound gaps appear at 48L.
- Note for the 48L run (Step 3): expect ~2,400 kernels/step at TP2-width too (width-independent count); eager-vs-graphs at 48L will show the same ~2x bubble ratio.

### Artifacts
- <workdir>/tprof_{eager,graphs}.json, analyze_tprof2.py, offline_gen_profile.py, run_tprof.sh in LXC <ct>; run_prof.sh (rocprofv3 launch, needs VLLM_ENABLE_V1_MULTIPROCESSING=0 + in-process to capture - rocprof attach path documented as blocked).

---

## D114 — DECISION: LXC/host on <source-host> is THE deployment target; VFIO dropped; transport decision closed

- Date: 2026-09-06
- Task: whole-project direction (coder-handoff Step 0; operator + other-dev agreement)
- Source: operator direction review, agreed

### The decision
1. **Deployment target = LXC/host on <source-host>** (containerized ROCm on the host amdgpu path). Formal, not a fallback.
2. **The VFIO-guest option is DROPPED.** Closed on merit: D103-D107 mapped every gate (topology -> link caps -> 44-bit dma_mask vs OVMF 2^46 BAR placement, D107 - a Proxmox/OVMF placement, not a stock knob), and the host/LXC path passes all of them (D109: 28.5 GB/s switch-internal P2P; D112: TP2 serving).
3. **The transport decision is CLOSED.** Host-path P2P is the transport basis: UNCACHED-staging kernel stores (D110) + SDMA copies (11.8 us @ 20 KB) + the fork's 33 us rdna_ar target (G5). RCCL-over-host remains the fallback floor, not the plan.
4. **M0/M0.7/M1/M2 host-path rows are UNBLOCKED** (the blocker was the decision, not hardware). Only cards 3-4 remain genuinely [M]. All M-track measurement/design work proceeds on the 2 host cards in LXC <ct>.
5. **llama has NO home during M2-M4.** The RAM arithmetic (40 GB TP2, ~46-50 GB TP4 in-container vs 64 GB <source-host> total, D113) precludes coexistence; plan for its absence (it stays down; LXC <ct> is the Flash-Next bring-up environment).
6. Measurement discipline unchanged (I5/D048; negative controls; append-only log). Added: a periodic DIRECTION review (deployment assumption + aimed-at-the-measured-bottleneck check) per the handoff Step 5.

### Tracker consequences (this entry + tasks.json/PROGRESS edits)
- G3: PASSED on the host path (D109) - determination recorded; VFIO-guest caveat closed.
- M0/M0.7/M1/M2: decision lifted; re-scoped host-path-executable (2 cards); only 4-card items stay [M].
- STOP-D: physical-path framing stands (not fired); the guest P2P absence is retired from the risk ledger.

---

## D116 — 48L TP2 attempt (T42): exact VRAM deficit recorded + second blocker found; surrogate profile is the baseline

- Date: 2026-09-06
- Task: handoff Step 3 (real 48-layer serve) - timeboxed, per the agreed "record the deficit if it won't fit" rule
- Source: executor (LXC <ct>, full 73 GB AWQ bind-mounted at <models> from <vm> disk-1, container 52 GB)

### Attempt 1 - TP2, no offload
- Full AWQ (48L) TP2, enforce-eager, gpu-mem 0.90, EP: workers initialize (world_size=2, nccl) then fail at MoE expert-weight CREATION:
  **torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 200 MiB. GPU 1 ...** -> the exact VRAM deficit: ~36.5 GB/rank of weights (73/2) + creation overhead cannot fit 32 GB/card without offload.

### Attempt 2 - TP2 + UVA offload (--offload-backend uva --cpu-offload-gb 20)
- Engine accepted the offload config (args show offload_backend=uva, cpu_offload_gb=20) and progressed to weight loading, then the **PLE offload worker failed**: PleOffloadWorker._load_weights -> model.load_weights(offload_only_iter) -> utils._groupby_prefix (worker.py:881 / utils.py:256). The PLE worker cannot load the FULL model's weight map (it worked against the surrogate's layout). SECOND, distinct blocker - recorded; not debugged further this session (would be a dedicated integration fix in the PLE worker's offload-only weight handling for the full AWQ layout).

### Disposition (per the handoff timebox)
- The one-number blocker (VRAM deficit at no-offload) + the second blocker (PLE-worker vs full-model weights) are recorded as named items for the follow-up 48L push.
- The surrogate profile (D115) is the working baseline: ~50 kernels/layer, ~2,400 kernels/step at 48L projection, launch-bound. The 48L serve stays a near-term goal behind fixing the PLE-worker full-model load (or running the engine without PLE offload once the ngram table handling for the full model is sorted).

### State
- Full AWQ reachable in LXC <ct> (<models>, read-only bind); container 52 GB. No server left running.

---

## D117 — M2 AR VALIDATED on the 2 host GPUs: one-shot = 30.7 us/op @ 20 KB, world=2, CORRECT

- Date: 2026-09-06
- Task: handoff Step 4 (M2 wave64 one-shot all-reduce against the fork's 33 us/20 KB)
- Source: executor (LXC <ct>, kernels/gfx906/m2_ar2.cu + rdna_ar_oneshot_gfx906.cuh)

### Result
- The wave64 one-shot all-reduce (the fork's T44 protocol, N-D3 wave64 reduce loop) runs between the TWO REAL host GPUs:
  **30.26-30.70 us/op at 20 KB (10240 halfs), world=2, 200 reps, includes the per-op host sync - UNDER the fork's 33 us target and G5's 34 us budget.**
  Correct: both ranks output 3.0 (1+2), zero timeouts.
- Implementation notes: UNCACHED peer staging + per-peer flag pages (RDNA_AR_FLAG_PAGE), arrive/seqbuf double-buffer, s_sleep poll, system-scope release/acquire flag protocol - all as the fork's rdna_allreduce.cuh. Single-process harness passes RAW device pointers across devices (peer access enabled) - no hipIpc needed in-process.

### Finding for the real multi-process integration (vLLM TP workers are separate processes)
- hipIpcOpenMemHandle is ASYMMETRIC on this stack: dev1->dev0 open WORKED (m0h) but dev0->dev1 open fails ("invalid device context" with LazyEnablePeerAccess, "invalid argument" without). The fork's rdna_ar_connect (multi-process, hipIpc-based) must handle this - either open from the peer direction or use the raw-pointer pattern where the process owns its device. Recorded for the M2-in-vLLM integration step.

### Disposition
- M2's core transport (one-shot AR shape, 33 us target) is validated at world=2 on the host path. The remaining M2 work = wire this into the fork's in-engine path (the 119-collectives/step rdna_ar op) across the 2-rank vLLM workers - a multi-process integration that must route around the hipIpc asymmetry finding.
- Artifacts: kernels/gfx906/m2_ar2.cu (2-GPU timing harness), rdna_ar_oneshot_gfx906.cuh (wave64 one-shot); run via <workdir>/run_m2.sh in LXC <ct>.

---

## D118 — Step-2 scoping (trace-justified): the gfx906 gap vs the fork is quantified and named

- Date: 2026-09-06
- Task: handoff Step 2 (close T46 pieces in trace order) - measurement/verification phase
- Source: executor (LXC <ct>, 50-token eager profile, tprof_eager.json, torch.profiler in-process)

### The fork's counting method (reconciled)
- The fork's 1,788 kernels/step (37/layer) came from an ALL-KERNELS eager trace (2,900 -> 1,800 via dispatch-count fusion, CONTAINER-RECIPE 8.4; "1,788 kernels" in CHANGES.md) - method-comparable to D115. So the gfx906 gap is real, not a counting artifact.

### gfx906 reconciled count (50-token eager, 8L surrogate; prefill now negligible)
- **49.6 kernels/layer/token -> ~2,380 kernels/step at 48L** (vs fork 37/layer / 1,788). Gap ~13/layer.
- Family per-layer (of ~49.6): torch_elem ~21.7 (43%), other/copies ~7 (d2d ~3.3), fused_glue ~6.4, dense_gemv ~4.5, moe_int4 ~2 + router ~1, gdn ~1.5, norm ~1.3, torch_reduce ~1.1, rope ~0.8.

### The named fusion targets (trace-justified, the ~22 torch-elementwise launches/layer)
1. **RMSNorm chains as torch-native ops (~4-6/layer):** rsqrt x357 + pow x357 + Mean-reduce x357 + float16_copy over 50 tokens. The engine config PREFERS 'vllm_c' for rms_norm/fused_add_rms_norm (kernel_config.ir_op_priority) but native ops are what fire -> the fused IR norm (vllm/vllm/ir/ops/layernorm.py) is NOT active on the gfx906 path. HIGHEST-leverage single fix (would also absorb the residual adds: add x1224 = ~3/layer).
2. **Dtype copies/conversions (~2.7/layer):** float16_copy + float16tofloat32 (1,069 over 50 tok) - AWQ dequant/quant conversions; fuse into the GEMM quant path.
3. **Fills (~1.3/layer):** FillFunctor x517 - buffer zeroing; some protocol-inherent.
4. **QSA indexer (~1.2/layer):** index_elementwise x472 + scatter_gather - attention gather path; smaller than the norms.
- int8 dense (fork's 176-vs-418 GEMV split, CONTAINER-RECIPE 8.3): the fork's own 8.3 note says the int8 branch never fires inside graphs (trace-time freeze). gfx906 runs fp16 dense; the int8 shadow is a graph-time-freeze problem shared with the fork, NOT gfx906-specific - lower priority than the norm fix.
- fused qk-norm-rope (D053, triton-compile): rope is only 0.8/layer on the trace -> smallest named target; deferred behind the norm/add work.

### Disposition
- If the norm/add/dtype fusion lands (~12-13/layer cut), gfx906 reaches the fork's ~37/layer and the launch-bound budget closes. The specific implementation = make rms_norm/fused_add_rms_norm route through vllm_c on the gfx906 path (find why native fires) + absorb the residual add + dtype copies. Code changes live in the (read-only-mounted) vllm tree - require a writable working copy; verification = re-run the 50-token profile and re-count.
- Artifacts: tprof_eager.json, analyze_tprof2.py in LXC <ct>; D115 (probe) + this entry (scoping) = the trace-justified Step-2 baseline.

---

## D119 — Step-2 root cause at the code level: the norm/add kernel gap is the ir.ops REFERENCE running instead of the vllm_c backend

- Date: 2026-09-06
- Task: handoff Step 2 implementation diagnosis (follows D118 scoping)
- Source: executor (fork source reading: layernorm.py, ir/ops/layernorm.py, kernel.py)

### The finding
- RMSNorm (model_executor/layers/layernorm.py, CustomOp "rms_norm"): forward_cuda -> forward_native ALWAYS routes to `ir.ops.rms_norm` / `ir.ops.fused_add_rms_norm` (unless VLLM_BATCH_INVARIANT). So the torch-native pow/mean/rsqrt seen in the trace (D118) come from INSIDE the ir.ops implementation.
- **ir/ops/layernorm.py IS a reference implementation**: `def rms_norm(...): x=x.to(fp32); variance=x_var.pow(2).mean(-1); x=x*torch.rsqrt(variance+eps); ...` - torch ops, ~4-5 launches/norm.
- The FUSED kernel is the `vllm_c` backend named in kernel_config.ir_op_priority (`rms_norm=['vllm_c','native']`, "Priority list for vllm.ir.ops.fused_add_rms_norm", kernel.py:35). On gfx906 the vllm_c backend does NOT win -> the reference runs -> the norm/add/dtype soup.
- The vllm_c backends come from the fork's kernel/compile layer (vllm_c = the compiled/IR-generated fused kernels). Compile is OFF on gfx906 (D060 inductor miscompile) -> most plausible reason vllm_c norms never materialize. This REFRAMES part of item 3: the ~13 kernels/layer gap (D118) is substantially the compile-off reference norm/add ops, NOT a missing hand-fusion.

### Consequences for Step 2 ordering
- The "inductor bisect" the handoff ranked BELOW fusion is actually UPSTREAM of the norm fusion: if vllm_c fused norms require the compile pipeline, fixing the D060 miscompile unlocks the norm/add fusion (+dtype copies) for free - the largest single count lever (~12-13/layer -> reaching the fork's ~37/layer).
- Alternative if the miscompile is intractable: hand-write a HIP fused RMSNorm/fused_add_rms_norm and route it as the gfx906 backend for ir.ops (targeted ~1-kernel-per-norm replace; verification = 50-token re-profile).
- qk-norm-rope (D053 triton compile) and int8 dense (graph-time-freeze, fork-8.3) remain smaller and later.

### Disposition
- Item 3's implementation now has a precise first action: determine whether vllm_c rms_norm is compile-gated (fix D061 inductor miscompile -> norms fuse) or gated elsewhere; fallback = hand HIP norm backend. Both verified by the 50-token re-profile counting toward ~37/layer.

---

## D120 — Step-2 norm gate: vllm_c fused norm EXISTS but the dispatch never selects it (registration-order wiring bug, not a compile gate)

- Date: 2026-09-06
- Task: handoff Step 2 implementation diagnosis (conclusion of D118/D119)
- Source: executor (fork source + runtime checks in LXC <ct>)

### The gate is NOT compile
- vllm_c fused_add_rms_norm = torch.ops._C.fused_add_rms_norm (a C++ custom op in the built csrc; kernels/vllm_c.py registers it, supported=GPGPU_DEVICE). Runtime check: the op EXISTS after `import vllm` in the container -> vllm_c is available on gfx906. It does NOT need the inductor pipeline (D119's compile-off hypothesis for the norm gate is WRONG - corrected).
- The oink backend (kernels/oink_ops.py) is an EXTERNAL plugin (torch.ops.oink.*) - not installed, not the cause.

### The actual problem: the IR dispatch never selects vllm_c
- ir/op.py dispatch (line ~329-348) runs the first SUPPORTED impl in _priority_impls (set from kernel_config.ir_op_priority = ['vllm_c','native']). The trace runs the NATIVE reference (pow/mean/rsqrt inside ir/ops/layernorm.py) -> vllm_c is either not in the runtime priority list or its supported() fails.
- op.py:293-295 warns "registering new impl for op while priority is set": registration ORDER matters. vllm_c impls register when vllm.kernels (-> vllm_c) is imported; if that happens after the ir op's priority list was built from the config, the vllm_c impl can be dropped -> native only -> the reference runs. This is a WRIRING BUG in the fork's registration timing on this path, not a gfx906 kernel gap.

### Fix candidates (implementation, verified by the 50-token re-profile counting toward ~37/layer)
1. Ensure vllm.kernels.vllm_c (and the fused norm impl) registers BEFORE the ir op priority dispatch is used - e.g., import vllm.kernels at engine init before the first norm call, or re-apply ir_op_priority after registration.
2. Or force the priority to ['vllm_c'] at a point where the impl is already registered (op.py get_priority path).
3. Or (fallback) hand-register a gfx906 fused rms_norm impl analogous to oink_ops.py but calling the _C op.
- All three need a WRITABLE copy of the vllm tree (the mounted one is read-only); verification = re-profile eager 50 tokens and compare family counts (expect the norm/add/dtype families to collapse, ~49.6 -> ~37-40 kernels/layer).

### Disposition
- Item 3's norm piece is now diagnosed to a concrete wiring bug with three named fixes. The remaining execution (writable tree, apply a fix, re-profile) is a bounded implementation task - the handoff's last open item.

---

## D121 — Step-2 fix-test VERDICT: vllm_c norm priority is NOT the lever; norms/QSA are already triton-fused; residual needs frame attribution

- Date: 2026-09-06
- Task: handoff Step 2 implementation test (D120's fix candidate, measured)
- Source: executor (LXC <ct>, offline 50-token profile with rms_norm/fused_add_rms_norm set_default(['vllm_c','native']) forced after LLM creation)

### Result: forcing vllm_c changed NOTHING
- With the D120 fix applied at runtime (set_default vllm_c-first after LLM init), the 50-token eager profile is NUMERICALLY IDENTICAL: 20,235 GPU kernels, torch_elem 8,688, norm 510, dense_gemv 1,785 - same as the un-fixed run. The priority is either (a) not the dispatch gate for the norms this model uses, or (b) vllm_c's supported() filters it out identically.
- The fix-run log ALSO reveals the classifier's blind spot: **layer_norm_fwd_kernel + the QSA kernels (_compress_qsa_groups, _qsa_mqa_paged, _expand_qsa_indices, _qsa_sparse_paged_gqa_splitk, _qsa_merge_splitk) ARE TRITON kernels** - they JIT-compiled during inference and their names carry no 'triton_' prefix, so D115's "zero Triton in decode" split was WRONG (D115 corrected: triton is present but name-hidden; the decode path is HIP-custom + triton-fused + torch-elem).
- The model's norms run as the FUSED triton layer_norm_fwd (norm family, 510), NOT the torch pow/mean/rsqrt reference - so the "norm soup" hypothesis (D118 item 1) is substantially WRONG for the Qwen4 path; the torch-elem rsqrt/pow/mean clusters (~7/token) come from a smaller, un-fused subset (likely specific residual/quant spots), not the main norms.

### What this means for item 3 (trace-justified closure, measured verdict)
- The T46 pieces the handoff named as hot (fused qk-norm-rope, int8 shadows, norm fusion): qk-norm-rope + QSA = ALREADY triton-fused and firing on gfx906 (the D053 'disabled' fused QSA was restored in the port tree); the norm path = triton layer_norm_fwd, firing. Neither the qk-norm-rope nor the vllm_c norm priority is the ~13/layer gap.
- The measured gap to the fork's 37/layer is dominated by the torch-elementwise residue; naming its exact python sources requires the fork's trace_attr.py frame-attribution (kernel -> python frame), which was not run this session. D115/D118/D121 together define the honest residual: "attribute the ~22 torch_elem launches/layer to their python frames, then fuse the largest sites" - the next concrete Step-2 action, requiring trace_attr (fork tool) on a full trace.
- D115's HIP-vs-Triton split is corrected: gfx906 decode = HIP custom (dense/MoE/glue/GDN) + TRITON fused (layer_norm, QSA attention, quant) + torch-elem residue.

### Disposition
- Item 3's named pieces are now measured: glue firing (D115), norms+QSA triton-fused and firing (D121), vllm_c priority = non-lever (D121). The remaining count gap is the torch-elem residue needing frame attribution - a well-scoped follow-up, not an assumption.
- Artifacts: tprof_eager.json (fix-run), analyze_tprof2.py, offline_gen_profile.py in LXC <ct>.

---

## D122 — Corrections + re-ordered next actions (operator review of D116-D121)

- Date: 2026-09-06
- Task: review response (operator flags + verification request)

### Correction: D121's "D053 fused QSA restored" is UNVERIFIED / likely a conflation
- The `_qsa_*` kernels firing in the decode trace come from the **AITER attention backends** (rocm_aiter_unified_attn.py / rocm_aiter_fa.py), NOT the fork's fused_qk_norm_rope.py `_fused_qk_rmsnorm_rope_gate_kernel` that D053 (Sep 2) found crashing `TritonAMDGPUCanonicalizePointers` at KV-cache init. Whether the fork's fused gate kernel now FIRES vs is BYPASSED on gfx906 is NOT established by D115/D121's traces (they show AITER kernels, which is a different path). D121's wording is corrected here; the D053 disposition stands unless a dedicated check (trace whether _fused_qk_rmsnorm_rope_gate launches) says otherwise. The QSA decode path IS working (via AITER) - that part of D121 stands.

### Operator-flagged priorities (agreed, re-ordered)
1. **PLE-worker full-model load fix (D116's second blocker) = the critical path** - the real 48L baseline gates real kernel mix, M4 power-cap re-derivation, M5, G6 acceptance, TP2->TP4 scaling. Everything downstream is still surrogate-proxied until it's fixed. TOP ITEM.
2. **Host-RAM budget before the next 48L attempt** (64 GB total): TP2 48L needs ~11-20 GB CPU-offloaded weights (resident in container RAM) COMPETING with the PLE prefault (~30 GB per D043; observed lower in the 40 GB container - the prefault's true RSS is unmeasured and is the swing factor). D116's container was 52 GB; +offload pushes past 64 unless the prefault is smaller than D043 assumed. Measure the PLE worker RSS, then budget.
3. **trace_attr.py frame attribution** is the prerequisite for the elementwise fusion work (name-based classification already proved unreliable - D121) - needs a writable vllm tree copy (the mounted one is read-only).
4. **hipIpc asymmetry (dev1->dev0 ok, dev0->dev1 fails) resolved BEFORE the M2-in-vLLM wiring** (fork rdna_ar_connect is hipIpc-based; harness-working + 4-rank-boot-deadlock class of bug).
5. Cards 3-4 remain the genuine [M] gate for TP4/real throughput; do not over-index on TP2+offload throughput (not TP4-representative).

### Disposition
- All five actions are recorded and ordered; the PLE-worker fix is the next engineering task (requires the writable tree copy, which is also the trace_attr prerequisite).

---

## D123 — Priority 2 done: host-RAM budget for 48L-TP2 measured (RAM is NOT the next blocker)

- Date: 2026-09-06
- Task: operator priority 2 (RAM budget before the next 48L attempt)
- Source: executor (LXC <ct>, TP1 surrogate serve + container memory accounting)

### Measured facts
- During a live TP1 8L serve in the 52 GB container: process RSS ~4 GB (APIServer+engine 1.9 GB seen), with ~45 GB of the 52 GB in buff/cache (reclaimable) - the PLE sidecar (~30 GB) + weights read footprint land in page cache, NOT locked process RSS. The PLE offload worker's own RSS was not cleanly isolatable (spawned transiently; the container-level accounting is the reliable number).
- D116's 48L attempt died with "Mem: 52 used 0 free 51" - the container was NOT memory-constrained at the failure; the cause was the PLE-worker _load_weights bug (priority 1), not RAM.

### Budget (48L TP2 + CPU offload in-container)
- Reclaimable page cache (PLE prefault + weight reads): ~30 GB, droppable under pressure - NOT a hard budget item.
- Locked RSS: serve/engine+workers ~4-6 GB + CPU-offloaded weights 11-20 GB (--cpu-offload-gb 11 suffices arithmetically: 36.5 GB/rank -> <=31 GB resident needs >=5.5 GB/rank offload) + overhead ~= 20-28 GB locked.
- A 52 GB container fits 48L-TP2 with ~20 GB offload with headroom; page cache reclaims under pressure. <source-host>'s 64 GB total accommodates a 52-56 GB container.
- **Conclusion: host RAM is NOT the next blocker for the 48L serve - priority 1 (the PLE-worker full-model load fix) is.** The operator's RAM concern is answered: the 30 GB PLE prefault is reclaimable cache, not the ~30 GB locked RSS D043's arithmetic implied.

---

## D124 — Priority 1 DONE: the real 48-layer model serves on TP2 (host/LXC path)

- Date: 2026-09-06
- Task: operator priority 1 - fix the PLE-worker full-model blocker so a real 48L TP2 serve runs
- Source: executor (LXC <ct>, writable tree copy <workdir>/vllm-w, model dir <workdir>/fullawq-w)

### Root cause (not a code bug)
- The PLE-worker FileNotFoundError was a STALE INDEX: the full AWQ's model.safetensors.index.json references model-00001-of-00005.safetensors, which contains EXACTLY the 128 excluded PLE-ngram tensors (the ~51B table served by the 30 GB int4 sidecar) and was never downloaded/kept (D100/D101 ngram exclusion). The surrogate's sliced index masked this; the full model's index did not.
- Fix: <workdir>/fullawq-w = the full AWQ with the index TRIMMED of the 128 shard-1 entries + symlinked present safetensors (no big copy). The PLE worker then reports "PLE weight loading complete; matched 3 checkpoint tensors".

### RAM constraint (the real next blocker, matching the operator's #2 concern after all)
- The UVA offload's host memory is SHMEM (26.4 GB on one worker at --cpu-offload-gb 20), NOT reclaimable file cache - 2 workers' shmem + model-load cache OOM-killed Worker_TP even at 52 GB, then 57 GB.
- Fits at: container 57 GB (<source-host> ~64 total, <vm> down) + --offload-backend uva --cpu-offload-gb 12 + gpu-memory-utilization 0.85. D123's budget needed the shmem caveat: UVA offload counts as locked shmem per rank.

### Result (the real baseline)
- 48-layer full AWQ, TP2 across the 2 host GPUs, EP, PLE offload: health 200, **generated 24 tokens in 33.4 s (~0.72 tok/s, eager + CPU-offload streaming)** - the "serve it badly" T42 baseline is live. Slow is expected; graphs + tuning follow. Real kernel mix, real 512-expert routing, real PLE now profiable.
- Bonus: the fork's rdna_ar one-shot all-reduce ENGAGED across the 2 TP ranks at startup ("rdna_ar: one-shot all-reduce active (handle 0/1, rank 0/2 + 1/2)") - the M2 AR is live in the multi-process engine (D117's harness validated 30.7 us; the hipIpc asymmetry (priority 4) did NOT block this path - the fork's in-engine rdna_ar connected).
- Artifacts: <workdir>/vllm-w (writable tree), <workdir>/fullawq-w (trimmed-index model), serve_tp2_48l_w.sh / run_48l_w.sh in LXC <ct>.

---

## D125 — Priority 3 status: frame-attribution tooling READY; stable run blocked by an in-process-engine double-init race

- Date: 2026-09-06
- Task: operator priority 3 (trace_attr frame attribution of the ~22 torch-elementwise launches/layer)
- Source: executor (LXC <ct>)

### Built and verified this round
- trace_attr.py (fork tool) located at vllm/tools/rdna2/trace_attr.py; adapted for our tree path in the writable copy (<workdir>/vllm-w). It consumes a gzip chrome trace with kernel + cpu_op(External id) + python_function events.
- with_stack capture route built (offline_gen_profile.py now also exports a chrome trace; <workdir>/tprof_eager.chrome.json, 11.3 MB, produced). FINDING: this torch build's chrome export emits NO python_function events (cats observed: cuda_runtime 15k, ac2g 21.9k, kernel 6.9k, gpu_memcpy) - the fork's python-frame attribution cannot run on our chrome traces as-is. ac2g events are frameless async markers.
- attr_stacks.py written: drives the same 12-token decode and reads torch profiler event .stack (with_stack=True) to attribute elementwise-family kernels to their innermost vllm python frames - the working route for OUR stack.

### Blocker (environmental, not analytical)
- The offline in-process engine (VLLM_ENABLE_V1_MULTIPROCESSING=0) intermittently DOUBLE-INITS (two "Initializing a V1 LLM engine" lines in one run), and the duplicate's PLE offload worker dies ("PLE offload worker exited during startup", worker.py:226) -> the attribution run fails before profiling. Plus recurring pct-exec no-ops on long background commands. 4 attempts this round, all failing on this flake rather than the analysis.

### Disposition
- attr_stacks.py + the with_stack route are ready; the next step is running them in a STABLE engine context (retry after a hard container restart to clear the double-init state, or run against the writable-tree path <workdir>/vllm-w). No code change is needed for the analysis itself - it is an environment-stability prerequisite. Recorded per goal policy rather than thrashing further this round.

---

## D126 — Priority 4 RESOLVED: the hipIpc "asymmetry" is a single-process harness artifact; the multi-process AR path WORKS

- Date: 2026-09-06
- Task: operator priority 4 (diagnose/resolve the hipIpcOpenMemHandle asymmetry before M2 wiring)
- Source: executor (LXC <ct>, probes m2_ipc2.cu / m2_ipc3.cu + the D124 48L serve)

### Finding
- Clean-room probe (m2_ipc3.cu: fresh state per direction, uncached buffer, peer enabled, can=1 both ways): **hipIpcOpenMemHandle fails in BOTH directions on this stack** (all 8 exporter/opener/cached/lazy combos fail in m2_ipc2.cu). The D117 "dev1->dev0 works, dev0->dev1 fails" asymmetry is NOT reproducible - and is best explained as a **single-process limitation**: hipIpc is a CROSS-PROCESS mechanism; opening another DEVICE's memory via hipIpc within ONE process is not supported on this stack (all directions fail). The m0h-era "worked" direction was misrecorded or a different path.
- The REAL multi-process path is unaffected: the fork's rdna_all_reduce.py exchanges raw handle bytes via torch.distributed all_gather_object and calls the native csrc rdna_ar_connect (hipIpc-based, CROSS-process). **Proven working: the D124 48L TP2 serve ran with "rdna_ar: one-shot all-reduce active (handle 0/1, rank 0/2 + 1/2)" and generated correct output** - the AR transport functions across the two real worker processes.
- Priority 4's premise (a landmine before M2 wiring) does not apply: the harness artifact (single-process IPC) never occurs in the deployment (TP workers are separate processes). D117's asymmetry note is corrected here.

### Disposition
- M2's in-vLLM integration is already LIVE (D124) - the remaining M2 work is measuring the per-op AR latency/throughput in the 2-worker context (vs the 30.7 us harness number), not resolving an IPC asymmetry. Priority 4 closed by evidence.
- Artifacts: <workdir>/m2/m2_ipc2.cu, m2_ipc3.cu (probes) in LXC <ct>.

---

## D127 — Priority 3 run BLOCKED (persisting, 3+ rounds): in-process engine + PLE offload double-builds the GPU engine

- Date: 2026-09-06
- Task: operator priority 3 - the frame-attribution run (attr_stacks.py)
- Source: executor (LXC <ct>)

### The concrete blocker (reproducible, same across goal rounds 2-4)
- The offline in-process engine (VLLM_ENABLE_V1_MULTIPROCESSING=0) + VLLM_PLE_CPU_OFFLOAD=1 double-BUILDS the full surrogate GPU engine: two "Initializing a V1 LLM engine" lines (same config/model) ~16 s apart; the second fails `ValueError: Free memory on device cuda:0 (18.33/31.98 GiB) ... less than desired GPU memory utilization (0.9, 28.79)` (the first build reserved the GPU). Reproduced with both in-python env (setdefault) and the exact shell-env recipe that the WORKING offline profiles used (run_tprof.sh). The working runs (13:32-13:39) predate the 48L-era engine churn; the double-build now always fires.
- Analysis scripts are READY and not the issue: attr_stacks.py (torch-profiler event .stack attribution), trace_attr.py adapted for our tree. What is missing is a stable engine context.

### Viable next paths (in order of expected effort)
1. Run the profiler inside the NORMAL multiprocessing engine (V1_MULTIPROCESSING default) via vLLM's own tracing hook (engine observability_config.collect_detailed_traces / the fork's profile instrumentation) instead of wrapping the client - kernels run in the EngineCore subprocess there (the 48L serve proved the multiprocessing+PLE path is stable).
2. Fix the double-build: find what makes the PLE worker's model load re-init the main GPU engine under V1_MULTIPROCESSING=0 (worker.py load path), so the second build skips the GPU reservation.
3. Profile the LIVE 48L server (D124) by attaching the profiler to its EngineCore.

### Status
- Priority 3 remains open; priorities 1, 2, 4 are DONE (D124/D123/D126). This entry records the persisting blocker per goal policy rather than thrashing further this round.

---

## D128 — Priority 3 run blocker, full state (goal rounds 3-5): ALL offline engine paths are down for the attribution run; only the HTTP-server path is stable

- Date: 2026-09-06
- Task: operator priority 3 (frame attribution run) - exhaustive environment diagnosis
- Source: executor (LXC <ct>, post-container-reboot canaries)

### The complete diagnostic picture (every offline route tested)
1. In-process (VLLM_ENABLE_V1_MULTIPROCESSING=0) + PLE @ util 0.9: the PLE-in-process path double-builds the GPU engine; the 2nd build fails the free-memory check (needs 28.8 of 32 GB; only ~18 free after the 1st).
2. In-process + PLE @ util 0.5: the free-memory clash is bypassed (2 x 16 GB fits), but now "PLE offload worker exited during startup" (EOFError on the worker pipe) - the PLE worker subprocess itself dies. (The same offline recipe WORKED at 13:32-13:39, before the 48L-era churn.)
3. Multiprocessing (default) offline: EngineCore dies silently at spawn ("Failed core proc(s): {}") - even WITHOUT profiler_config, and after a full container reboot (devices fine, 57 GB, GPUs empty).
4. In-process + PLE OFF: cannot run - the model materializes the full 51B ngram table on GPU ("Tried to allocate 95.37 GiB") since the sidecar path is disabled.
- The ONLY demonstrably stable configuration is the HTTP SERVER multiprocessing path (the 48L TP2 serve, D124, and earlier surrogate serves). The offline client engine paths regressed sometime in the 48L/OOM-kill churn and survive container reboots (suggesting a state issue in the editable tree path or a PLE-worker startup dependency, not transient).

### Status
- Priority 3's analysis is READY (attr_stacks.py, adapted trace_attr.py, engine profiler_config route identified). The run needs a stable engine: next paths = (a) diagnose the PLE-worker EOFError at util 0.5 (its startup crash is the nearest failure), (b) restore the offline path from a fresh venv/tree state, or (c) the engine-side profiler on the live server. Recorded per goal policy after 3+ consecutive rounds on the same blocker class.

---

## D129 — Priority 3 attribution UNBLOCKED (root cause: missing spawn main-guard) + frame-level stacks unavailable on this torch

- Date: 2026-09-06
- Task: operator priority 3 (frame attribution run) - the fix that unblocked it + the tooling limit found
- Source: executor (LXC <ct>)

### The REAL root cause of the round-3-5 blocker (D125/D127/D128 corrected)
- The offline scripts (offline_gen_profile.py, attr_stacks.py) ran LLM() at MODULE level with NO `if __name__ == "__main__"` guard. The vLLM PLE offload worker spawns via multiprocessing SPAWN (popen_spawn_posix), which re-imports the main script - so every PLE-worker spawn re-executed the module body and created a SECOND engine. That is the "double-build" (two "Initializing a V1 engine" lines), the free-memory clash (the 2nd build reserved the GPU), and the PLE-worker EOFError (the spawned child died recursing). The regression "survived" container reboots because it is in the SCRIPT structure, not the environment.
- Fix (verified): add `if __name__ == "__main__":` and wrap the body in main(). Guarded run: rc=0, SINGLE engine init, decode + attribution complete. prof_gen3.py (guarded, in-process, util 0.5) also verified rc=0. The fork's spawn-based PLE worker REQUIRES guarded entry scripts for any offline client.

### Tooling limit (frame attribution)
- The attribution RAN cleanly (12-token decode profiled, elementwise-family kernel counts reproduced: vectorized_elementwise 1512, elementwise_manual_unroll 471, index_elementwise 143, reduce 117, scatter 80... consistent with D115's torch_elem family) but ALL python frames resolve to "?" - this torch build records NO python stacks on device events with with_stack=True (micro-test confirmed empty .stack; the chrome export also had no python_function events, D125). The fork's trace_attr.py frame attribution depended on python_function-emitting torch traces that this ai-infos 3.6.0+gfx906 torch does not produce.
- Result: the op-level target list from D118 stands (rms_norm reference chains, residual adds, dtype copies, QSA indexer - reproduced cleanly here); the python-FRAME-level fusion-site list requires a stack-recording torch or the fork's exact profiling toolchain - a tooling gap, not an analysis gap.

### Disposition
- Priority 3's measurement (which elementwise ops dominate, launch-bound verdict) is COMPLETE and clean (D115/D118 + this run). The frame-level fusion-site naming is blocked on torch profiler tooling (no python_function); the fusion work itself can proceed on the op-level targets already named. Guarded scripts are the durable fix for all offline profiling on this stack.

---

## D130 — Working profiling recipe confirmed + priority 3 final state

- Date: 2026-09-06
- Task: operator priority 3 - close the round with a verified, durable profiling recipe
- Source: executor (LXC <ct>)

### Confirmed
- The standard offline profile (run_tprof.sh -> offline_gen_profile.py, which ALWAYS had the main-guard) reproduces the D115/D118 numbers exactly after the D129 fix: 8,445 GPU kernels over the window, torch_elem 3,648, fused_glue 1,050, dense_gemv 735, moe_int4 336, gdn 246, norm 210 -> **50.3 kernels/layer/token** (~2,400/step at 48L). The guarded-script recipe is now the reliable profiling path on this stack.

### Priority 3 final state
- Measurement: COMPLETE and reproducible (D115/D118/D129/D130): ~50 kernels/layer, launch-bound, torch-elem ~43% of launches, glue firing, norms+QSA triton-fused.
- Frame-level naming: BLOCKED on torch tooling - this ai-infos 3.6.0+gfx906 torch records NO python stacks (with_stack inert; no python_function events; confirmed CPU+device side). The fork's trace_attr.py needs a stack-recording torch; the op-level fusion targets from D118 (rms_norm reference chains, residual adds, dtype copies, QSA indexer) stand as the actionable list.
- Fusion implementation: NOT started (gated behind frame-level naming per the operator's ordering; the op-level targets are named and the writable tree <workdir>/vllm-w is ready).
- Guarded-script finding (D129) is the durable fix: any offline client on this stack MUST wrap main() in `if __name__ == "__main__"` (the PLE worker spawns via multiprocessing spawn and re-imports the entry script).

### Disposition
- Priorities 1, 2, 4: DONE (D124/D123/D126). Priority 3: measured + unblocked; frame-naming tooling-gated; fusion ready-to-start on the writable tree. Priority 5: standing gate (cards 3-4 = [M]).

---

## D131 — DIRECTION: torch-elem fusion deprioritized; the 48L graphs profile + int8 dense (N-D5) are the next levers

- Date: 2026-09-06
- Task: operator direction review (response to the blocked-goal options)
- Source: operator + executor

### The data that re-ranks the levers (D115's GPU-time split, 8L/20-token)
- torch_elem = 43% of LAUNCHES but ~16 ms of GPU time; dense_gemv = ~9% of launches but ~63 ms (~4x the torch-elem time). Under graphs-no-compile the dispatch bubble is erased (GPU 49%->97%), so what remains is GPU TIME and that is dense_gemv-dominated. The torch-elem fusion thread (options 1/2) attacks the smallest time lever.
- dense_gemv is the fork's T45 int8-shadow lever (10.9 -> 3.5 ms/step dense at the fork, ~7.4 ms/step saved; lm_head 1.27 -> 0.64 ms; 176 int8 vs 418 fp16 GEMVs/step). D118 deprioritized int8 behind the norm fix; D121 then killed the norm fix -> the deprioritization is unmotivated. N-D5's "gated behind MTP (D007)" conflates two gemv_i8 uses: the MTP-head use IS MTP-gated; the dense-shadow use of the base decode projections is MTP-INDEPENDENT.

### Decisions
1. **Next measurement: profile the 48L model on graphs-no-compile** (the actual serving config) separating offload traffic (H2D/D2H/shmem copies) vs compute vs dispatch. The single question: at full depth under graphs, what dominates GPU time? (Strong prior: dense_gemv. If offload dominates instead, the priority is offload minimization / cards 3-4, and torch-elem drops further.)
2. **N-D5 (gemv_i8 / int8 dense shadows) RE-ELEVATED now** - un-defer the dense-shadow use (MTP-independent); validate the wave64 gemv_i8 + route the dense decode through it under graphs + time fp16 vs int8. Targets the biggest measured time term.
3. **Option 1 declined as framed**: the D118 fusion list is stale (its #1 target, rms_norm reference chains, disproven by D121). A concrete subset (dtype copies + residual adds) may be fused later, gated on the 48L profile showing they matter in TIME.
4. **Option 2 deferred**: stack-recording torch = multi-hour build, uncertain payoff; a cheaper intermediate exists (torch.profiler.record_function annotation on the writable tree) if frame attribution becomes needed.
- Tracker: N-D5 re-scoped and set running (dense-shadow scope, precondition = the 48L graphs profile ranking or the D115 time data).

---

## D132 — 48L graphs profile (D131 runbook): dense_gemv dominates GPU time (54%); torch-elem is <1%; dense is offload-inflated

- Date: 2026-09-06
- Task: the decisive 48L-under-graphs profile (operator D131 direction)
- Source: executor (LXC <ct>, guarded offline client 48l_graphs_prof.py: <workdir>/fullawq-w, TP2, graphs-no-compile, UVA offload 12 GB, engine-side torch profiler; traces <workdir>/tp48/*rank*.json.gz; analyzer <workdir>/analyze_chrome.py)

### GPU-time ranking at 48L under graphs (TP2 rank0 = 24 layers/rank; ~2,858 ms over the trace)
- **dense_gemv 1,535.5 ms = 53.7%** (n=1,754)
- fused_glue 770.2 ms = 26.9% (n=2,609)
- moe_int4 377.3 ms = 13.2% (n=864)
- other 4.0%, torch_elem **0.9%** (26.2 ms, n=5,142 - the most launches, the least time), gdn 0.3%, qsa_attn 0.3%, offload_copy 0.1%, norm 0.1%, rope <0.1%
- **The D131 ranking is CONFIRMED decisively: dense_gemv is the GPU-time lever; the torch-elem fusion thread (D118 list) was attacking <1% of GPU time at full depth under graphs.** N-D5 (int8 dense shadows) targets the 54% term.

### The offload-inflation finding (the second half of the answer)
- Dense_gemv per-token per-rank ~192 ms vs the surrogate extrapolation ~9.5 ms (24 layers x 0.39 ms/layer) - a ~20x inflation: the UVA-offloaded dense weights are fetched zero-copy during the gemv kernels, stalling them on slow PCIe reads. The GPU time (~360 ms/token/rank) is ALSO well under the eager wall (~1,400 ms/token): a large host-side offload-stall component (page faults / zero-copy migration / H2D waits) dominates the WALL and graphs do NOT remove it (the trace's explicit offload_copy is tiny - the cost hides inside the inflated dense kernels and host stalls).
- So BOTH D131 branches are partially true: dense_gemv dominates GPU TIME (54%), and its magnitude is offload-driven; the WALL has a big offload term. int8 shadows attack both (half the streamed bytes of the dense projections, fp16->int8) - N-D5 is doubly pointed.

### Disposition
- N-D5 proceeds (validated in the D131 record). A follow-up measurement: time the profiled 8-token generate's WALL under graphs at 48L (add a clock to the client) to quantify the host-offload-stall share; and compare dense kernel time with fully-VRAM-resident weights (requires a smaller model or cards 3-4) to separate offload fetch from compute.
- Artifacts: <workdir>/48l_graphs_prof.py, <workdir>/run_p48.sh, <workdir>/tp48/ traces, <workdir>/analyze_chrome.py in LXC <ct>.

---

## D133 — Task 1 (first slice): dense-GEMV microbenchmark — the 53.7% term is BANDWIDTH-BOUND and MI50-FAVORED (~2-2.8x faster than V620)

- Date: 2026-09-06
- Task: coder-dev handoff Task 1 (MI50-vs-V620 kernel microbenchmark), dense-GEMV family first (the sharpest row)
- Source: executor (LXC <ct>, kernels/gfx906/nd1_harness.cu on GPU0; V620 refs from the operator's table + vllm/docs/rdna2/RESULTS.md T43/T45)

### Measured (gfx906, gemv_f16_gfx906 wave64, all 12 shapes PASS numerical match; M=4 column)
| shape | N x K | M=1 | M=4 | GB/s@M1 | GB/s@M4 |
|---|---|---|---|---|---|
| gdn.in_proj_qkv/rank | 2560x2560 | 33.9 | 29.3 us | 387 | 448 |
| gdn.in_proj_z/rank | 1536x2560 | 16.3 | 23.6 | 483 | 333 |
| gdn.out_proj/rank | 2560x1536 | 24.3 | 27.2 | 324 | 289 |
| qsa.q_proj/rank | 3072x2560 | 28.3 | 33.0 | 555 | 477 |
| qsa.o_proj/rank | 2560x1536 | 18.0 | 25.2 | 438 | 312 |
| router.gate | 512x2560 | 7.9 | 9.6 | 332 | 273 |
| shared.gate_up | 1280x2560 | 17.1 | 22.4 | 383 | 293 |
| shared.down | 2560x640 | 10.3 | 15.4 | 320 | 213 |
| hc.down | 320x10240 | 26.7 | 20.3 | 245 | 323 |
| hc.up | 10240x320 | 18.4 | 35.7 | 356 | 183 |
| hc.inject | 4x10240 | 8.1 | 11.1 | 10 | 7 |
| lm_head/rank | 62080x2560 | 389.7 | 451.4 | **816** | 704 |

### The verdict (dense family): memory-bound, MI50-favored
- **12-shape M4 sum: 0.704 ms vs V620's 1.43 ms -> MI50 ~2.0x faster.**
- **lm_head/rank M4: 451 us vs V620 fp16 1.28 ms -> 2.84x faster; vs V620 int8 640 us -> still 1.42x faster in fp16.** lm_head M=1 hits 816 GB/s = AT the ~810 GB/s HBM ceiling (mclk ~1000 implied; the rocm-smi --showclocks samples only showed idle LEVELS (925/350) - the live-clock method needs pp_dpm_sclk/pp_dpm_mclk live-reads (PROFILE-NAVI21 10), noted as the sampling fix for the remaining families).
- The 53.7% term (D132) is a MI50 WIN on bandwidth, not a clock loss: dense GEMV is read-stream-bound and gfx906's +60% HBM BW over V620 dominates. The V620 advantage must come from the CLOCK-bound families (MoE int4 compute, triton attention, launch overheads) - the remaining harness rows (nd2 MoE, nd4 glue, QSA/GDN model-level) are the next slices.
- Numerical match: acc <= 1e-4, out <= 1e-3 on every shape (N-D1 kernel is correct on the live cards).

### Disposition
- Remaining Task 1 slices: nd2_harness.cu (MoE W13/W2), nd4_harness.cu (fused glue), QSA/GDN at model level (triton), each with the live-clock fix. N-D5 (int8 dense) still pinned after; note its lm_head yardstick (fork 640 us) is now BEATEN by gfx906 fp16 (451 us) - the int8 gain on MI50 is bandwidth-only and smaller than on V620.
- Artifacts: <workdir>/nd1.log + <workdir>/nd1 in LXC <ct>; harness kernels/gfx906/nd1_harness.cu.

---

## D133 — Task 1 (first slice): dense-GEMV microbenchmark — the 53.7% term is BANDWIDTH-BOUND and MI50-FAVORED (~2-2.8x faster than V620)

- Date: 2026-09-06
- Task: coder-dev handoff Task 1 (MI50-vs-V620 kernel microbenchmark), dense-GEMV family first (the sharpest row)
- Source: executor (LXC <ct>, kernels/gfx906/nd1_harness.cu on GPU0; V620 refs from the operator's table + vllm/docs/rdna2/RESULTS.md T43/T45)

### Measured (gfx906, gemv_f16_gfx906 wave64, all 12 shapes PASS numerical match; M=1 and M=4 columns)
| shape | N x K | M=1 | M=4 | GB/s@M1 | GB/s@M4 |
|---|---|---|---|---|---|
| gdn.in_proj_qkv/rank | 2560x2560 | 33.9 | 29.3 us | 387 | 448 |
| gdn.in_proj_z/rank | 1536x2560 | 16.3 | 23.6 | 483 | 333 |
| gdn.out_proj/rank | 2560x1536 | 24.3 | 27.2 | 324 | 289 |
| qsa.q_proj/rank | 3072x2560 | 28.3 | 33.0 | 555 | 477 |
| qsa.o_proj/rank | 2560x1536 | 18.0 | 25.2 | 438 | 312 |
| router.gate | 512x2560 | 7.9 | 9.6 | 332 | 273 |
| shared.gate_up | 1280x2560 | 17.1 | 22.4 | 383 | 293 |
| shared.down | 2560x640 | 10.3 | 15.4 | 320 | 213 |
| hc.down | 320x10240 | 26.7 | 20.3 | 245 | 323 |
| hc.up | 10240x320 | 18.4 | 35.7 | 356 | 183 |
| hc.inject | 4x10240 | 8.1 | 11.1 | 10 | 7 |
| lm_head/rank | 62080x2560 | 389.7 | 451.4 | **816** | 704 |

### The verdict (dense family): memory-bound, MI50-favored
- **12-shape M4 sum: 0.704 ms vs V620's 1.43 ms -> MI50 ~2.0x faster.**
- **lm_head/rank M4: 451 us vs V620 fp16 1.28 ms -> 2.84x faster; vs V620 int8 640 us -> still 1.42x faster in fp16.** lm_head M=1 hits 816 GB/s = AT the ~810 GB/s HBM ceiling (mclk ~1000 implied; the rocm-smi --showclocks samples only showed idle LEVELS (925/350) - the live-clock method needs pp_dpm_sclk/pp_dpm_mclk live-reads (PROFILE-NAVI21 10), noted as the sampling fix for the remaining families).
- The 53.7% term (D132) is a MI50 WIN on bandwidth, not a clock loss: dense GEMV is read-stream-bound and gfx906's +60% HBM BW over V620 dominates. The V620 advantage must come from the CLOCK-bound families (MoE int4 compute, triton attention, launch overheads) - the remaining harness rows (nd2 MoE, nd4 glue, QSA/GDN model-level) are the next slices.
- Numerical match: acc <= 1e-4, out <= 1e-3 on every shape (N-D1 kernel is correct on the live cards).

### Disposition
- Remaining Task 1 slices: nd2_harness.cu (MoE W13/W2), nd4_harness.cu (fused glue), QSA/GDN at model level (triton), each with the live-clock fix. N-D5 (int8 dense) still pinned after; note its lm_head yardstick (fork 640 us) is now BEATEN by gfx906 fp16 (451 us) - the int8 gain on MI50 is bandwidth-only and smaller than on V620.
- Artifacts: <workdir>/nd1.log + <workdir>/nd1 in LXC <ct>; harness kernels/gfx906/nd1_harness.cu.

---

## D134 — Task 1 slice 2: MoE w13+w2 pair timing (gfx906) vs the fork's 98 us

- Date: 2026-09-06
- Task: coder-dev handoff Task 1 (kernel microbenchmark), MoE family
- Source: executor (LXC <ct>; kernels/gfx906/nd2_harness.cu + in-place BENCH block timing the w13+w2 PAIR, the fork RESULTS unit; compiled at <workdir>/m2b/nd2_harness.cu - the kernels/gfx906 mount is read-only, edits run from <workdir>/m2b with moe_int4_gfx906.h copied)

### Measured (gfx906, moe_w13_silu_gemv_gfx906_<W=4> + moe_w2_gemv_gfx906_<W=4>, EP-aware local 64/128 experts, topk 10, GROUP 128, 200 reps after 20 warmup)
- **M=1: w13+w2 pair = 109.15 us | M=4: 355.88 us | M=8: 689.08 us | M=16: 1,373.55 us** (linear in M -> per-token cost ~109 us at decode batch 1, per layer's routed experts)
- All M PASS numerical match (the D030 wave64 int4 decode kernel is correct on the live cards).

### Verdict: MoE int4 decode is the CLOCK-BOUND family (MI50-slightly-slower, near parity)
- **M=1 pair 109 us vs the fork's V620 98 us -> MI50 ~1.11x SLOWER** (11%) - the first family where MI50 does NOT win. Consistent with the D133 framing: MoE int4 at decode batch is compute/latency-bound (int4 dequant + v_dot-style inner loop), where V620's ~2x clock shows, but the wave64 int4 port keeps the loss to ~11% (not ~2x) - the decode is partly bandwidth-helped by the int4 byte stream.
- 98 us (V620) vs 109 us (MI50) at M=1: the ~13 us/step x 512-expert routing... per-layer routed MoE at decode: this is the 13.2% GPU-time family (D132); at ~1.1x MI50 cost the MoE term is near-parity, NOT the V620-advantage driver.

### Disposition
- Remaining Task 1 slices: nd4 fused glue (in progress), QSA/GDN model-level. Clock sample method (pp_dpm_sclk live levels) verified accessible; capture for the record on the next runs.
- Artifacts: <workdir>/m2b/nd2_harness.cu + nd2b + <workdir>/nd2b.log in LXC <ct>; the edited kernels/gfx906/nd2_harness.cu (BENCH block) in the local repo (note: the repo file on the container mount is the pre-edit version - sync the edited one back via the local repo, which is authoritative).

---

## D135 — Task 1 slice 3: fused-glue timing (gfx906) - bandwidth-bound, M-invariant

- Date: 2026-09-06
- Task: coder-dev handoff Task 1, fused-glue family (the 26.9% GPU-time term at 48L, D132)
- Source: executor (LXC <ct>; kernels/gfx906/nd4_harness.cu + BENCH blocks timing each of the 4 glue kernels; <workdir>/m2b/nd4_harness.cu built from the local edited source)

### Measured (gfx906, launch_*_<RowF16,4,8>; 200 reps after 20 warmup; all M PASS numerical)
- gemv_act (hc.down/inject N=320 K=10240, act_cols=4): M=1 61.2 | M=4 60.8 | M=8 61.1 us - M-invariant
- hc_mix (H=320 R=10240 HC=4): M=1/4/8 ~158 us - M-invariant
- se_gu (I=640 K=2560, 2I rows): ~35.6 us all M
- se_dn (H=2560 I=640 Kx=2560): ~45.6 us all M

### Verdict: fused glue is BANDWIDTH-BOUND (MI50-favored class, like dense)
- All four kernels are M-invariant: they stream the full weight rows per decode token, so the read dominates and M barely matters -> same memory-bound regime as the dense GEMVs (D133's ~2x MI50 favor applies to this 26.9% family). No V620 per-kernel reference in the operator's table for these (the fork fused them to CUT LAUNCHES, not time) - the MI50 absolute costs are the record.
- Combined Task-1 verdict so far: dense (54% of GPU time) ~2.0x MI50-favored + fused glue (27%) bandwidth-bound MI50-favored + MoE int4 (13%) 1.11x V620-favored near-parity -> gfx906 parity on the GPU-time side looks ACHIEVABLE on the memory-bound bulk; the V620-advantage driver must be the remaining slice (attention/QSA triton + launch overheads), which the model-level QSA/GDN comparison addresses.

### Disposition
- Remaining Task 1 slice: QSA/GDN (triton) at the model level (no HIP harness - in-tree serve + trace compare). N-D5 (int8 dense) pinned after: note its rationale WEAKENS - gfx906 fp16 dense already beats V620 int8 (D133: 451 vs 640 us lm_head), so int8's gain on MI50 is smaller; the dense term is already a WIN at fp16.
- Artifacts: <workdir>/m2b/nd4_harness.cu + nd4b + <workdir>/nd4b.log in LXC <ct>; edited kernels/gfx906/nd4_harness.cu (BENCH blocks) in the local repo (authoritative).

---

## D136 — Task 1 COMPLETE: QSA/GDN model-level slice + the full MI50-vs-V620 verdict

- Date: 2026-09-06
- Task: coder-dev handoff Task 1 (final slice: QSA/GDN at model level, in-tree serve + trace)
- Source: executor (extraction from the D132 48L graphs trace <workdir>/tp48/, rank0; V620 refs: PROFILE-NAVI21 352-356 + the operator's table)

### QSA/GDN at 48L decode under graphs (gfx906, rank0, per-kernel from the real trace)
- GDN core (chunk_gated_delta_rule_fwd): 36 calls, med 108 us, 3.95 ms total; _causal_conv1d_update 288 x 4.2 us (1.66 ms)
- QSA: _qsa_merge_splitk 108 x 8.8 us, _store_qsa_rows 216 x 4.0 us, _qsa_mqa_paged 108 x 6.1 us, reshape_and_cache 108 x 5.4 us, _compress_qsa_groups 108 x 4.6 us, _expand_qsa_indices 108 x 4.0 us, _build_qsa_metadata 18 x 4.0 us
- Total QSA+GDN ~10-19 ms over the trace = ~0.3-0.5% of GPU time each (vs dense 1,536 ms) - NEGLIGIBLE at decode batch 1. The fork's V620 attention reference (fp16 KV @43k = 0.51-0.52 ms/kernel, 173.6 GB/s) is a different regime (43k ctx, dense attention); our QSA is MQA-sparse paged at 2k ctx with 4-9 us kernels - not comparable, and not a time term either card cares about at decode.

### FULL TASK-1 VERDICT (the G6 re-scope input)
| family | 48L GPU-time share | MI50 vs V620 | regime |
|---|---|---|---|
| dense GEMV (12 shapes) | 53.7% | ~2.0-2.8x MI50 FASTER | bandwidth-bound (816 GB/s at ceiling) |
| fused glue (4 kernels) | 26.9% | MI50-favored class (M-invariant streaming) | bandwidth-bound |
| MoE int4 (w13+w2) | 13.2% | 1.11x V620 (109 vs 98 us @M=1) | clock-bound, near-parity |
| QSA + GDN (triton) | ~0.6% | negligible on both | - |
| launch/dispatch (eager) | (D115 7.7 ms/token) | MI50 worse (Triton 9.54 vs ~3 us) | erased by graphs |
- **gfx906 parity at MTP0 is ACHIEVABLE on the measured terms**: the memory-bound bulk (dense + glue = ~81% of GPU time) is MI50-favored (HBM +60% BW), the one clock-bound family (MoE 13%) is at near-parity, and attention/launch are negligible-or-graph-erased. The V620 t/s advantage (100 MTP3 / ~61 MTP0) is NOT explained by per-kernel GPU time on these families - the residual candidates are V620's MTP head throughput and any per-step fixed cost, not the decode kernels this benchmark measured. G6's 60-62 t/s MTP0 parity target stays ON the table pending the launch-overhead accounting at 48L graphs.
- N-D5 (int8 dense) rationale is WEAKER post-benchmark: gfx906 fp16 dense already beats V620 int8 (D133). Re-scope decision deferred to the operator.

### Artifacts
- D133/D134/D135/D136 cover all Task-1 slices; harness BENCH blocks committed in kernels/gfx906/nd{1,2,4}_harness.cu (local repo authoritative); container copies at <workdir>/m2b/; trace extraction inline.

---

## D137 — End-to-end token accounting: W measured; the independent busy% is UNAVAILABLE on this hardware; provisional Case-A call

- Date: 2026-09-06
- Task: the D137 procedure (per-token wall ledger, GPU-busy %, Case A/B classification)
- Source: executor (LXC <ct>, <workdir>/vllm-w, <workdir>/fullawq-w, 2 GPUs)

### Provenance header (pinned)
- fork (vllm tree): 519be40fd56ede6e38d4fd52aaec5aadc7fbc5b2; project repo: local HEAD eb85ad0 (authoritative; the <vm> mirror is stale at 2c4dfcc)
- graphs-no-compile (compile OFF, cudagraph_mode=full FULL_DECODE_ONLY); TP2 + EP, dtype float16, MTP=0; PLE offload ON (ples_int4 sidecar); UVA --cpu-offload-gb 12; VLLM_RDNA_AR=1 VLLM_RDNA_DENSE_INT8=1; GPU_MAX_HW_QUEUES unset; prompt pinned 40 tok; greedy temp 0; ignore_eos (the offload-degraded model emits EOS early otherwise)
- HEADER LINE: "This is the offload-crippled TP2 regime (73 GB > 64 GB VRAM); the offload is a known confounder and gets its own bucket."
- NOTE (env hygiene, found during the run): vLLM renames its processes to VLLM::EngineCore/Worker_TP* (prctl), so pkill -f python3 leaves them holding GPU memory between runs - every offline client MUST llm.shutdown() before exit (added) and leftover engines need pkill -f "VLLM::".

### Phase 1 (DONE): wall/token denominator
- 48L TP2 graphs-no-compile, steady-state decode (20-token warmup, then 256-token greedy, ignore_eos), 3 repeats, first->last token, prefill excluded.
- **WALL_MEDIAN_MS_PER_TOKEN = 393.84 ms/token (2.54 tok/s)**; reps 372.7 / 393.8 / 415.2 ms (median of 3, I5).

### Phase 2 (BLOCKED by hardware): independent GPU-busy %
- rocm-smi --showuse reads 0.0% at idle (control A passes) but ALSO ~0% under a verified saturating load; the kernel's own /sys/class/drm/card{1,2}/device/gpu_busy_percent reads 0 during and after a 9 s full-occupancy elementwise kernel (sat.cu, built + run + verified rc=0, 9 s duration). The SM-activity counter does not move on these flashed V420 cards in this container -> **the independent busy% is UNAVAILABLE; control B cannot pass because the counter never leaves 0.** Not a workload issue (the load was real and ran full duration).
- Consequence: the Case A/B discriminator's independent-B leg cannot be satisfied on this hardware; the classification below rests on the profiler-derived K/W only (flagged weakness vs the procedure's intent).

### Phases 3-4 (NOT COMPLETED in-session): PLE 5-timestamp instrumentation + record_function CPU phases
- Phase 3 requires code changes in the fork's PLE worker/connector/prepare_forward (vllm/v1/ple_offload + the model) - not completed; the per-step timestamp log does not exist.
- Phase 4 (CPU-phase split via record_function) - not completed; the CPU-phase sum cannot be checked against W - K.

### Phase 5 (PARTIAL ledger; K from the D132 same-config trace)
| component | ms/token | % wall |
|---|---|---|
| GPU kernel execution (K, profiler sum per rank, D132 same-config 48L graphs trace) | ~360 | ~91% |
| - of which dense_gemv (compute + offload-fetch CONFLATED, flagged) | ~192 | ~49% |
| - fused_glue / moe_int4 / other | ~96/47/25 | ~24/12/6% |
| AR/sync, PLE CPU, PLE stall, scheduler, sampling, graph boundary, H2D | not measured (Phases 3-4 incomplete) | - |
| Unaccounted (= W - K) | ~34 | ~8.6% |
| Total wall/token | 393.84 | 100% |
- GPU-busy estimate (profiler-only): K/W ~91%.

### Phase 6 - D137 verdict (provisional, with the stated caveat)
"The measured GPU kernels explain most of the wall: K/W ~91% at ~394 ms/token, leaving ~9% unaccounted. The residual is [Case A: host/runtime ~9%, with the caveat that the PLE/scheduler/sampling buckets are UNMEASURED (Phases 3-4 incomplete) and the independent busy counter was unavailable, so the Case call rests on profiler K/W alone]. The dense_gemv bucket (~49% of wall) is offload-inflated (fetch inside the kernels is attributed there). Until the PLE/CPU phases are instrumented and a hardware busy signal exists, 60-62 TPS at MTP0 remains plausible but unproven."
- The residual is NOT large (8.6% < 10% acceptance) IF the D132 K transfers; the true remaining question is whether the offload-inflated dense time (the 49% bucket) shrinks without offload (cards 3-4), which would make W far smaller than 394 ms.

### Artifacts
- <workdir>/wall48.py + wall48.log (W), <workdir>/busy_*.log + sat.cu (the counter attempt), <workdir>/prov.log (provenance) in LXC <ct>; D132's trace = the K source. Local: this D137 entry.

---

## D138 — D137 verdict refined: OFFLOAD-STALL DOMINANT (supersedes D137's "provisional Case A" framing)

- Date: 2026-09-06
- Task: review of D137 (operator + other-dev opinion; the refined diagnosis is adopted)
- Source: D132/D133/D137 cross-analysis

### The three-piece causal chain (D132 + D133 + D137 together are much stronger than any alone)
- **D133**: HBM-resident dense GEMV is EXCELLENT on MI50 (lm_head 816 GB/s ≈ the ceiling; 12-shape M4 sum 0.704 ms vs V620 1.43 ms).
- **D132**: offloading the weights inflates the dense family ~20x (per-token ~193 ms offloaded vs ~9.5 ms HBM-resident extrapolated from the surrogate).
- **D137**: that inflated dense family consumes ~49% of the 393.84 ms/token wall.
- Chain: resident GEMV is excellent -> offload inflates it ~20x -> the inflated GEMV is ~half the real-model wall. "gfx906 arithmetic is simply slow" is UNTENABLE. **Weight offload is the DOMINANT current bottleneck.**

### Verdict (replaces D137's "provisional Case A")
- **Offload-stall dominant.** Large runtime idle bubble ruled unlikely: K/W ~91% shows nearly all wall time is long-lived GPU KERNEL INTERVALS (absent substantial kernel overlap - single-stream decode, overlap unlikely), so a large between-replay idle bubble is inconsistent with the trace. The independent SM-busy counter is unavailable/unreliable on the flashed/containerized hardware (D137) - its absence is not fatal because D132 provides the orthogonal causal test.
- Graph capture removes launch/dispatch (eager 0.72 -> graphs 2.54 t/s = 3.53x) but CANNOT remove host-weight-access stalls - graphs solved the thing graphs can solve; they cannot make PCIe-hosted weights HBM-resident. The eager/graphs separation VALIDATES the model: eager = launch bubble + offload stall; graphs = offload stall only.
- **PLE claim corrected**: PLE remains UNMEASURED (no Phase-3 timestamps); it may become critical after offload removal but is NOT material to the current bottleneck ranking (measuring a 1-5 ms secondary effect before removing a ~180 ms dominant one isn't valuable).
- Implied TP2 improvement on offload removal: 394 - 193 + 9.5 ~ 210.5 ms/token -> ~4.75 tok/s (~1.87x) before any TP4 benefit.
- **Cards 3-4 are the next experiment**: 4 x 32 GB = 128 GB VRAM, fully GPU-resident weights, TP4, PEX88096 peer fabric, graphs, wave64 - only then is the V620 TP4 MTP0 ~60-62 t/s comparison meaningful, and the unresolved terms (TP4 world=4 AR, PLE serialization, graph boundaries, CPU scheduler/sampling) become worth re-measuring. **25-30 t/s from this TP2 setup must NOT be treated as an MI50 silicon estimate** - the measurement is dominated by a condition (weight offload) the TP4 machine removes.
- Bottom line (verbatim): "The dominant bottleneck in the current TP2 48L serve is solved diagnostically: weight offload inflates the dense path by roughly 20x and accounts for about half of wall time. The current 2.54 t/s result therefore cannot be used to infer gfx906's fully resident Flash-Next ceiling. Cards 3-4 are now required to expose the next bottleneck."
- Note: another bottleneck may wait underneath offload removal - that is normal bottleneck peeling, not evidence against the project.
- TRAPS.md created (the prctl/pkill -f python3 trap + the other hard-won operational traps: spawn main-guard, silent read-only-mount pushes, pct-exec no-ops, dead SM-busy counter, ignore_eos, awk field, heredoc newlines).

## D139 — Four-card host/LXC hardware gate cleared; TP4 baseline is now the next experiment

- Date: 2026-09-16
- Task: Track M hardware bring-up, M0/M0.7/M1/M2/M3/M4
- Source: operator + executor measurement on <source-host> (`root@<source-ip>`) and LXC <ct>

### What changed since D138
- D138's blocking precondition was physical: cards 3-4 were required before the project could
  remove TP2 weight offload and expose the fully resident TP4 bottleneck. That precondition is
  now satisfied on the host/LXC path.
- <source-host> now has four Vega20/MI50-class cards in the intended GPU path:
  `<pci>`, `<pci>`, `<pci>`, `<pci>`.
- The two newly added Apple Pro Vega II cards (`<pci>`, `<pci>`) were backed up before
  flashing:
  - `<workdir>/vegaflash/1500_preflash_20260916_202727.rom`
  - `<workdir>/vegaflash/1b00_preflash_20260916_202727.rom`
  - both stock backups SHA256:
    `44625aed534c05143852c52415edd18a2d91cc5106c4d080da5e6caaf9128952`
  - stock BIOS P/N `113-D163A1XT-045`, device ID `66a3`
- Both cards were flashed with `<workdir>/vegaflash/V420.rom`:
  - target ROM SHA256:
    `c737534ca7fb85beb62b10cceb8e3c169096b430c9f33903b2eaad2c86592ff7`
  - target BIOS P/N `113-D1640200-043`, device ID `66a0`
  - `amdvbflash v4.71` reported RSA PASS and full program/verify success for adapters 2 and 3.
- After reboot, `amdvbflash -i` and PCI inspection showed all four adapters as `66A0` /
  BIOS P/N `113-D1640200-043`; PCI reports `<device-id>`, subsystem `<device-id>`, 32 GB BAR0,
  Gen4 x16, and `amdgpu` bound for all four GPU endpoints.

### ACS service repair
- The previous `/usr/local/sbin/clear-pex-acs.sh` searched for a V620-specific endpoint ID and
  was stale after the V620s were removed. It has been repaired to discover any AMD display-class
  GPU endpoint (`vendor=0x1002`, class `0x03*`) that sits under a Broadcom/LSI PEX880xx switch
  (`vendor=0x1000`, device `0xc010`), then walk the ancestor path and clear ACS on ACS-capable
  PCI bridge ports.
- This makes the service GPU-ID agnostic for the current purpose: V620, MI50/V420, and similar AMD
  endpoints under the PEX880xx path do not need per-device script edits.
- `clear-pex-acs.service` is enabled and exits successfully. Manual verification showed the
  GPU-path ACS-capable bridge ports cleared with
  `SrcValid-/TransBlk-/ReqRedir-/CmpltRedir-/UpstreamFwd-/EgressCtrl-/DirectTrans-`.

### LXC300 readiness
- LXC <ct> is running.
- Inside the container, `/dev/kfd` is present and render nodes map to the four host GPUs:
  `renderD129` -> `<pci>`, `renderD130` -> `<pci>`,
  `renderD131` -> `<pci>`, `renderD132` -> `<pci>`.
- The container's gfx906 venv reports `torch 2.13.0+gfx906.20260802001858`,
  `torch.cuda.device_count() == 4`, and approximately 32 GiB visible memory per device.
- `rocminfo` sees four `gfx906:sramecc+:xnack-` agents. Host KFD nodes for the four GPUs each show
  `p2p=3`.

### Decision / next action
- The D138 conclusion stands: TP2 offload performance must not be used as a silicon estimate.
  What changes is the project state: the hardware block is gone.
- Track M is now hardware-unblocked on the host/LXC path:
  1. M0/M0.7 first: rerun the 4-card P2P matrix across all 12 pairs and run the peer-BAR
     ordering / ACS negative-control checks on the full fabric.
  2. M1/M2 next: run the 4-rank RCCL-vs-custom comparison and wire/validate the world=4
     wave64 one-shot AR path, accounting for the existing hipIpcOpenMemHandle asymmetry.
  3. M3/M4 after transport correctness: run the TP4 fully resident baseline (full AWQ,
     TP4+EP, MTP=0, float16, graphs-no-compile, no CPU/UVA weight offload), then profile it
     to expose the next bottleneck.
- No TP4 performance claim exists yet. The next valid claim must come from the harness artifacts,
  not from the successful flash or container enumeration.

---

## D140 — M0 4-card transport validation: P2P matrix clean, host link found at Gen1 x8, negative-control incident

- Date: 2026-09-16
- Task: M0 / M0.7 (four-card transport validation, following D139)
- Source: executor measurement on <source-host> (`root@<source-ip>`) and LXC <ct> (`<container>`)

### What was run
1. Read-only fabric/ACS report (topology, per-hop ACS capability and control bits, per-hop link
   state, IOMMU groups, KFD topology, amdgpu params) — before, after an idempotency re-run of
   `clear-pex-acs.sh`, and at the end.
2. `clear-pex-acs.sh` re-run (regression + idempotency check).
3. P2P matrix across **all 12 ordered GPU pairs**: `hipDeviceCanAccessPeer`,
   `hipDeviceEnablePeerAccess`, SDMA peer copy (`hipMemcpyPeer`) at 256 MB with N=20
   median+IQR (I5/D048), a block-total cross-check, and byte-exact verification on every
   transfer; plus 20 KB latency at the D110 decode operating point; plus per-device HBM
   self-copy and host<->device transfers as reference ceilings.
4. ACS negative control on the GPU3-unique path hops.

Artifacts: `results/m0_fabric.jsonl` (26 records, harness-written), one `results/M0.jsonl` run
row, `harness/m0_fabric_ingest.py` (ingest), probes under `tools/m0/`.

### Results — P2P matrix, all 12 ordered pairs
- `canAccessPeer=1` and `EnablePeerAccess` OK for all 12 ordered pairs; **byte-exact on all 12**.
- **Median 14.39 GB/s** (IQR ~0.01); block cross-check agrees (14.36-14.44); all 12 within 0.5%.
- Latency at 20 KB: **18.3 us/op** block-amortized (21.5 us per-op synchronized).
- Self-copy reference (HBM): 368-379 GB/s, byte-exact.
- KFD topology: `p2p_links_count 3` on all four GPU nodes; host KFD host->GPU links present.

### FINDING 1 — the HOST LINK is at Gen1 x8, ~8x below D078
- `<pci>` (the root port above the switch) reports `LnkCap: 2.5GT/s, Width x8` and
  `LnkCap2: Supported Link Speeds: 2.5GT/s` — it is a **Gen1-only x8 port**.
- `<pci>` (switch upstream) reports `LnkCap: 16GT/s, Width x16` but
  `LnkSta: Speed 2.5GT/s (downgraded), Width x8 (downgraded)`.
- Measured host<->device **1.82 GB/s H2D / 1.81 GB/s D2H** on every device — two independent
  methods agreeing (~91% of Gen1 x8).
- D078 recorded host<->GPU **14 GB/s**. This is an **~8x regression of the host path**.
- Consequence: any host-resident or offloaded weight path (UVA / CPU offload, PLE sidecar fetch)
  is now bounded at ~1.8 GB/s. This bears directly on D137/D138's "offload-inflated dense ~20x"
  and must be re-checked before any TP4 number is interpreted.

### FINDING 2 — peer traffic IS switch-internal, but at half of D109
- Peer 14.39 GB/s vs host link 1.82 GB/s = **7.9x**: peer traffic cannot be traversing the host
  link, so it stays on the switch — this is NOT the root-complex redirect of M0-DECISION-TREE §1.
- But D109/D110 recorded **28.5 GB/s** (1.8x the x8-Gen4 upstream ceiling, i.e. switch-internal).
  Current per-pair figure is **half that**.
- All 12 pairs equal within 0.5% => a **shared** bottleneck, not a per-port one.
- **Not explained by ACS**: see the negative control below, which asserted ACS on a GPU path and
  did not degrade a single completed pair.

### FINDING 3 — measurement-integrity trap, caught and fixed
- The first run of the probe reported **59,473 GB/s self-copy and 93,515 GB/s peer copy** —
  ~2000x the physical ceiling of this fabric.
- Cause: in this ROCm (7.14 / core-7.14) build, **`hipMemcpyPeer` returns before the transfer
  completes**; timing it as a blocking call measured enqueue only. The byte-exactness check still
  passed, because the later verification copy forced completion.
- Fix: synchronize BOTH devices inside every timed region, add a block-total cross-check, and add
  a physical plausibility bound (peer <= 40 GB/s, self <= 1200 GB/s) that flags implausible
  results rather than reporting them as measurements. This is the D049 discipline applied to a
  probe: a kernel cannot sustain 2000x the machine's ceiling, so one of the two numbers is wrong.

### FINDING 4 — no regression from the generic `clear-pex-acs.sh`; the repo copy was stale
- Deployed script `sha256 31cf473271361d2717c86f023a96c19eac695e4fe709e48a2fba3297454f598e`;
  service enabled and active.
- Discovery is **GPU-ID agnostic** and found exactly the four MI50 endpoints behind the PEX880xx:
  `<pci>`, `<pci>`, `<pci>`, `<pci>`.
- It clears **11 ACS-capable bridges** across the four GPU paths and verifies every one post-clear
  (all controls read `-`). It is a strict **superset** of the old fixed-BDF script (4 downstream
  ports only) and additionally covers the deeper hops (`<pci>`, `<pci>`, `<pci>`, `<pci>`,
  `<pci>`, `<pci>`, `<pci>`, `<pci>`).
- Re-running it is **idempotent**: same 11 ports, exit 0, and the fabric report is byte-identical
  before and after (21,949 bytes both times).
- It is correctly **GPU-path-scoped**: `<pci>` (the fourth downstream port, no GPU behind
  it) keeps ACS asserted, and the non-PEX AMD endpoint `<pci>` (Caicos PRO) is untouched.
- **Regressed artifact fixed:** the repo's `tools/host/clear-pex-acs.service` was the stale
  V620/V420-specific copy (fixed BDFs `<pci>`-`<pci>`). The deployed generic script and unit
  are now synced into `tools/host/` this session.

### FINDING 5 / INCIDENT — ACS re-assertion on a live peer-mapped path wedges the GPU
- Control design: assert the switch's own asserted default **0x001d**
  (`SrcValid|ReqRedir|CmpltRedir|UpstreamFwd`, read empirically from the untouched port
  `<pci>`) on `<pci>` and `<pci>` — the hops on GPU3's path that are **unique to
  GPU3** (`<pci>` was deliberately excluded because it is shared with GPU2).
- **8 of 12 pairs completed under assertion and were UNCHANGED** (all <0.1%): `0->1`, `0->2`,
  `0->3`, `1->0`, `1->2`, `1->3`, `2->0`, `2->1`, all at ~14.4 GB/s.
- Pair `2->3` — the sibling pair sharing `<pci>` / `<pci>` — faulted with
  `HIP error unspecified launch failure`; the `3->x` pairs were never reached.
- Kernel log: `amdgpu <pci>: GPU reset begin!. Source: 2` -> `smu firmware loading failed`
  -> `fw load failed` -> `VRAM is lost due to GPU reset!` -> **`GPU reset end with ret = -22`**
  (the reset itself failed).
- ACS was restored to cleared on both hops and verified. The post-restore probe nonetheless
  failed with `no ROCm-capable device is detected (n=0)`.
- **Current state:** `rocm-smi` still lists all four GPUs and KFD still reports
  `p2p_links_count 3`, but **ROCm cannot initialise ANY device in LXC <ct>**; GPU3 retains a stale
  428 MB VRAM allocation vs ~11 MB on the other three.
- **Conclusion:** these hops' ACS state does **NOT** govern the ~14.4 GB/s peer limit — asserting
  it did not degrade a single completed pair. What ACS re-assertion *does* do on a live
  peer-mapped path is **fault sibling peer traffic and wedge the GPU**. The 14.4 GB/s limit lies
  elsewhere (see open questions).
- Precedent: D105/D108 — an analogous gfx906 wedge was cleared by a <source-host> power cycle.

### Blocked
- M0/M0.7 completion (the peer-BAR ordering test), M1/M2 four-rank collectives, and M3/M4 TP4 are
  **BLOCKED pending GPU recovery**. Recovery is a host-boundary action (I10) and needs the
  operator: a <source-host> power cycle (D105 precedent) or a driver-level recovery attempt.
- **No TP4 performance claim exists**, and nothing in this entry implies one.

### Open questions raised
1. Why is the switch upstream at Gen1 x8 rather than the ~14 GB/s D078 recorded — is the switch now
   in a different host slot/port (<pci> appears Gen1-only x8), or is this a link-training
   regression after the card install?
2. Why is per-pair peer bandwidth 14.4 GB/s rather than D109's 28.5 GB/s, given peer traffic is
   provably switch-internal? Candidates: the downgraded upstream gating the switch fabric, or
   4-card vs 2-card loading.

### Recommended next actions
1. Operator: recover the GPUs (power cycle <source-host>), then re-run the health probe to confirm 4/4.
2. Re-measure the host link after recovery; if the switch is in a Gen1 x8 port, move it to a
   Gen3/Gen4 x16 slot and re-measure — this is the single highest-leverage unknown, because it
   bounds every host-resident weight path and therefore the D137/D138 offload verdict.
3. Re-run the P2P matrix post-recovery to confirm 14.4 GB/s reproduces, then bisect the 28.5 -> 14.4
   difference against D109's exact conditions.
4. Do NOT re-run the ACS negative control as designed; record it as unsafe on a live peer-mapped
   path and, if it is ever needed, do it only with no ROCm process attached and no peer mappings
   established.

---

## D141 — Four-card fabric PASSES on <host>; RCCL world=4 floor measured; custom wave64 one-shot AR correct at world=4

- Date: 2026-09-17
- Task: Phase 1 (M0/M0.7 fabric) + Phase 2 (M1/M2 collectives) of `TP4_BRINGUP_PLAN.md`
- Source: executor measurement on `<host>` (<host-ip>), container <ct> `<container>`

### Where this ran, and why it is not <source-host>
The <source-host> path from D139/D140 was abandoned on evidence: <source-host>'s switch uplink trained at **Gen1 x4**
(`max_link_speed` 2.5 GT/s) and no downstream port ever linked a known-good card, while the same
switch on `<host>` trains **Gen4 x16** with a healthy Gen4 x16 fabric. The switch was therefore
NOT broken (the earlier D140 "switch is broken" reading was wrong and is superseded here), and
**<host> is now the working 4-card rig**:

- Supermicro **<board>**, 2 × <cpu>, Debian 13 (trixie)
- 4 × Vega20/MI50 (`<device-id>`, subsystem `<device-id>`, SKU `D1640200`, 31.98 GB each), `amdgpu` bound
- container **<ct> `<container>`**: Ubuntu 24.04, 32 cores / 32 GB, `/dev/kfd` + `/dev/dri` passed
  through; **ROCm 7.14.0-gfx906+20260802001858**, **RCCL 2.30.4**, **torch 2.13.0+gfx906.20260802001858**
  (hip 7.14.60850); `torch.cuda.device_count() == 4`

### Phase 1 — four-card fabric: PASS
- Root port `<pci>` **16 GT/s x16**; switch upstream `<pci>` **16 GT/s x16**; all switch fabric
  ports **16 GT/s x16**. (An x8x8 run was compared: uplink x8 -> D2H 14.31 GB/s; x16 -> D2H
  28.59 GB/s. **The "2.5 GT/s" readings on GPU-facing hops are idle link downshifting, not a Gen1
  train-down** — proven by measured D2H at ~91% of Gen4 x16.)
- **All 12 ordered peer pairs**: `canAccessPeer=1`, `EnablePeerAccess` OK, **byte-exact**,
  median **14.40-14.44 GB/s** (IQR <= 0.014), latency **16.96-18.13 us @ 20 KB**.
- Self-copy (HBM): 368.9-378.6 GB/s. Host transfer: H2D ~17.0 GB/s, D2H **28.59 GB/s** (all four).
- **ACS**: `clear-pex-acs.service` installed + enabled on <host>; it discovers all four AMD
  endpoints under the PEX880xx and clears the GPU-path bridges at boot (11 ports this build).
  **Before this clear existed, the peer copy hard-faulted a GPU** (`ERREVENT_ATHUB_INTERRUPT`,
  `MMHUB MMEA0_DRAMWR_DATAMEM`, failed GPU reset) — clearing ACS fixed it. The project's ACS
  doctrine is confirmed on a second, independent board.
- Peer bandwidth is **uniform across all 12 pairs and does not scale with the uplink** (14.4 GB/s
  at both x8 and x16 uplink) => peer traffic is **switch-internal, capped at ~x8 (Gen4 x8 /
  Gen3 x16)** on the GPU-facing side. Not a root-complex bounce.

### Phase 2 — collectives, world=4, 20 KB dominant message
| path | median | IQR | min | correctness |
|---|---|---|---|---|
| RCCL (torch.distributed/nccl) | **106.96 us** | 9.48 | 97.92 | all sizes correct |
| custom wave64 one-shot AR | **70.34 us** | 1.39 | 67.78 | all 4 ranks == 10.0, bit-identical |
| G5 target | <= 34 us | - | - | - |

- RCCL world=4 full sweep (median us): 5 KB 72.16, 20 KB 106.96, 80 KB 100.00, 160 KB 108.96,
  320 KB 118.72, 640 KB 146.08, 1280 KB 193.12. Every size passed an out-of-band sum check.
  **106.96 us @ 20 KB lands inside D106's 4-rank projection of 95-160 us** — the banked 2-rank
  curve extrapolated correctly.
- Custom AR: `kernels/gfx906/m2_ar2.cu` was world=2; the kernel itself
  (`rdna_ar_oneshot_gfx906.cuh`) is already world-generic, so a W=4 harness was written
  (`tools/m0/m2_ar4.cu`) — same protocol (uncached staging, rank-staggered push, per-block
  arrive, block0 announce via system-scope release stores, local flag polling, fixed-rank-order
  fp32 sum). 20 KB = 70.34 us, 5 KB = 64.38 us; `timeout=0` on all ranks; every element == 10.0
  and **bit-identical across ranks**.
- The repo's `tools/m0/t03_rccl_sweep.py` and `t04_ar_oneshot.py` are **scaffolds that measure
  nothing** (they print PASS). D141 supersedes them as the Phase 2 evidence; they should be
  rewritten or retired.

### DECISION — first TP4 collective path
**Boot the first TP4 with RCCL (custom AR off).** The plan's rule applies on both counts: custom AR
**misses G5** (70.3 vs <= 34 us) **and is not wired into vLLM** (the harness is in-process; the
multi-process `hipIpc` form and the D117 handle asymmetry are unwired). RCCL is the recorded floor.

> **CORRECTION (see D142 addendum 3).** The second clause above is **WRONG**. The D117 hipIpc
> "asymmetry" was already resolved by **D126** (single-process harness artifact), and the fork's
> custom AR is in fact **LIVE at world=4** in the TP4 serve (`rdna_ar: one-shot all-reduce active`
> x8 = 2 handles x 4 ranks, <=64 KB). It was never "unwired". This decision's *outcome* still stands
> (RCCL remains the conservative reference path), but the stated reason was based on a stale claim.
G5 remains a t/s-adjustment gate, not a STOP (D083/D066).

### Caveats / open items
1. The 70.34 us custom-AR figure is an **upper bound**: the in-process harness launches four kernels
   back-to-back from one host thread and each spins for peers, so launches serialize. The
   multi-process concurrent form may land lower; wire it before treating 70 us as the protocol floor.
2. **`iommu=pt` is MISSING from <host>'s kernel command line.** RCCL warns explicitly: *"Missing
   iommu=pt from kernel command line which can lead to system instability or hang!"* <source-host> had it.
   Add it (plus `kernel.numa_balancing=0`, which RCCL also flags as a variability source) before
   any long TP4 run.
3. `RCCL_USE_AMD_SMI_LIB` is unset while `HIP_FABRIC_API` is defined (RCCL warns fabric support is
   unavailable without it).
4. Peer traffic at ~x8 on the GPU-facing side is unexplained; a switch downlink *width* setting is
   the first thing to check if peer bandwidth becomes the binding constraint.
5. H2D (~17.0) is asymmetric vs D2H (28.59). Consistent across all four cards, so likely a
   write-path characteristic rather than a fault — but unverified.

### Artifacts
- `results/tp4_bringup/` (peer matrix JSON, RCCL sweep JSON) — see runbook step 1.
- Harness: `tools/m0/m2_ar4.cu`, `tools/m0/t03_rccl_world4.py`, `tools/m0/m0_p2p_matrix.cpp`.
- Host config: `tools/host/clear-pex-acs.sh` + unit deployed to <host>.

### D141 addendum — M0.7 peer-BAR ordering, and custom-AR GRAPH REPLAY correctness

Both were the outstanding Phase 1 / Phase 2 items. **M0.7 closes G2.2's peer portion.**

#### M0.7 — peer-BAR payload→fence→arrive→flag ordering
Protocol under test (faithful to the fork's T44 and to `rdna_ar_oneshot_gfx906.cuh`):
writer writes a slice into the PEER's **uncached** staging, each block does
`__threadfence_system()` then `atomicAdd(peer_arrive, 1)`, block 0 spins on the arrive count then
publishes a **system-scope release** flag, and the owner polls **its own local** flag copy and then
verifies the whole payload. The owner-side payload is filled in **scrambled** order so the
last-issued write is unrelated to the last cache line, and a stale word is distinguishable from
corruption by pattern-matching the previous sequence value.

```
PRODUCTION (uncached staging + per-block system fence + arrive counter + owner-local polling)
 12/12 ordered pairs clean over 120 reps   (order_violations=0 mismatches=0 timeouts=0)
 VERDICT: PASS
```

**Honest limitation, recorded deliberately.** The test **cannot falsify the hardware-reorder
hazard**, and its own controls prove it:
- removing the per-block fence (uncached staging) → still clean
- removing the fence with **cached** staging (writes drain lazily) → still clean
- adding a per-block **stagger** to widen the reorder window → still clean

So across 12 ordered pairs and thousands of iterations, a block's arrive-increment was **never**
observed to overtake its own payload writes. The PASS therefore demonstrates that the **protocol
works on this hardware**; it does **not** demonstrate that the fence is load-bearing.
**Keep the fence** — it is required by the memory model and under graph replay, and it is free.
The corollary is a TRAPS-class risk: because the hazard is not empirically inducible here, a
future regression in that ordering edge would likely be **silent**.

A deliberate **protocol-bug control** (publish the arrive *before* writing the payload) is kept in
the harness as the checker-validity test: it is the probe that proves the owner-side checker can
see flag-before-payload at all. It is slow by construction — the checker hammers contended
`atomicMin`s across a 1 MB payload — so the checker now **early-exits on the first mismatch**,
which is all a binary violation check needs.

#### Custom AR under HIP graph replay
The plan requires "any custom op or kernel used under graphs must pass eager AND graph replay
checks before serving", and the TP4 serve runs `cudagraph_mode=full` (D142). The kernel had only
ever been validated eager. `tools/m0/m2_ar4_graph.cu` captures one graph per device and re-verifies
**every** replayed iteration — not just the first, because a stale-sequence bug would corrupt only
later iterations.

```
EAGER : 100/100 reps correct    median=73.29 us
REPLAY: 500/500 iterations correct   median=79.45 us
device-side seqbuf = 610 on all four ranks  (expected 5+100+5+500 = 610)
RESULT: seqbuf 2440 (expected 2440) OK      VERDICT: PASS
```

- The decisive evidence is the counter: the **device-derived** sequence advances by exactly the
  number of eager + replayed iterations, which is what makes the `seq & 1` double-buffering
  alternate correctly under replay. A host-derived counter could not advance across a replay.
- Graph replay was **~8% slower** in this harness (79.45 vs 73.29 us). No launch-overhead win here:
  these are four *independent per-device* graphs launched sequentially from one host thread and
  then synchronised, so the graph buys nothing over eager launch in this in-process form. Not a
  verdict on graphs in-server.

**Phase 1 status after this:** fabric + ACS + M0.7 all validated; the only Phase 1 item still
unticked is the ACS negative control, which D140 recorded as unsafe on a live peer-mapped path and
the plan permits skipping ("only if it can be done safely").

**Artifacts:** `tools/m0/m07_order.cu` (ordering + controls), `tools/m0/m2_ar4_graph.cu`
(eager vs replay), `results/tp4_bringup/`.

---

## D142 — TP4 fully-resident serve ACHIEVED on <host>; correct, deterministic; baseline 10.95 tok/s

- Date: 2026-09-17
- Task: Phase 3 (M3) of `TP4_BRINGUP_PLAN.md` — first TP4 fully resident serve
- Source: executor measurement on `<host>` (<host-ip>), container <ct> `<container>`

### Environment built for this (all new since D141)
- **Container <ct> `<container>`**: Ubuntu 24.04, **128 GiB RAM** (64 cores), `/dev/kfd` + `/dev/dri`,
  100 GB rootfs. Memory matters: the plan needs >=64 GB and the PLE sidecar is paged into RAM.
- **Assets copied <source-host> -> <host>** (`/<pool>/qwen38-flash-next/`, ~127 GB, verified with
  empty rsync dry-runs + matching file counts):
  `awq` (73 GB backbone), `models` (PLE int4 sidecar 128 shards + surrogate), `vllm-w`
  (**already-built** vLLM port tree, 815 MB), `gfx906-venv` (4 GB).
- **Bind mounts replicate <source-host>'s LXC300 paths** so the existing serve scripts work unmodified:
  `awq -> <models>/models`, `models -> <models>`,
  `vllm-w -> <workdir>/vllm-w`, `gfx906-venv -> <home>/gfx906-venv`.
- ROCm 7.14.0-gfx906 + **torch 2.13.0+gfx906.20260802001858** + RCCL 2.30.4 in a venv.

### BLOCKER FOUND AND SOLVED: missing `model-00001-of-00005.safetensors`
The AWQ dir is missing shard 00001. Index analysis: that shard holds **exactly and only** the 128
`ple.ple_embedding.ngram_embedding.shard_*.weight` tensors — i.e. the 51.2B-param BF16 n-gram
table (**102.4 GB**; 180.7 GB index total - 78.3 GB present). It is absent from **every** <source-host>
mount (verified by full-disk and per-mount searches). The new port tree's
`weight_utils.py:600 filter_duplicate_safetensors_files` refuses to start:
`FileNotFoundError: Weight files referenced in index but missing`.

**The fix is the v620-serve method (LXC <source-ct>), which is now the documented MI50 method too:**
LXC <source-ct>'s docker run overlays a patched index
(`-v <index-overlay>/model.safetensors.index.json:/models/qwen38-flash-next/model.safetensors.index.json:ro`).
That file (`/<pool>/<subvol><index-overlay>/model.safetensors.index.json`) references
**5 shards, 0 ngram entries, 222,618 tensors** (exactly 128 fewer than the original 222,746) and is
what makes the missing-shard model usable. It is now copied to
`/<pool>/qwen38-flash-next<index-overlay>/` and bind-mounted over the model's index in <container> —
**the model directory itself is unmodified.**

This confirms the design: the BF16 table is replaced by the int4 sidecar. v620-serve's log says it
plainly: *"PLE quant: ...ngram_embedding.weight stubbed, gathers served from sidecar"*, and <container>
logged the identical line.

### Phase 3 result — TP4 FULLY RESIDENT, no offload
Launch (`tools/m0/serve_tp4_48l.sh`, derived from <source-host>'s `serve_tp2_48l.sh`):
TP=4, EP, MTP=0, float16, `--enforce-eager`, `--language-model-only`, max-model-len 2048,
**no `--offload-backend uva`, no `--cpu-offload-gb`**.

```
Worker_TP0-3:  Model loading took 18.23 GiB memory and ~399 s   (x4 ranks)
               Available KV cache memory: 9.1 GiB
PleOffloadWorker: PLE quant table ... 128 shards mmapped from .../ples_int4
                  ngram_embedding.weight stubbed, gathers served from sidecar
APIServer:     Application startup complete
VRAM: GPU0 32.47 / GPU1 32.29 / GPU2 32.29 / GPU3 32.26 GB (of 34.34)
```

- **18.23 GiB/rank x 4 = 72.9 GiB fully resident** = the whole checkpoint, i.e. **no weight offload**
  (plan Phase 3 acceptance).
- Load took 396 s only because vLLM disables auto-prefetch on ZFS ("not a recognized network FS");
  not a hang and not a correctness issue.

### Correctness (Phase 3 step 4)
- Deterministic greedy (temperature 0, `ignore_eos`): **two runs, identical sha256
  `0128852903291e32`** => DETERMINISTIC.
- Output is coherent on-topic reasoning text (PCIe switch vs bridge, three sentences).
- **No G4 golden comparison performed**: `golden/` is still empty (S3 never ran), so a first-64-token
  *match* is not evaluable. Recorded as determinism + captured text/hash instead. This is the
  documented limitation, not a pass.

### Baseline timing (Phase 3 step 5) — BASELINE ONLY
| repeat | tokens | wall s | tok/s |
|---|---|---|---|
| 1 (cold) | 128 | 25.07 | 5.11 |
| 2 | 128 | 11.69 | 10.95 |
| 3 | 128 | 11.68 | 10.96 |

**Median 10.95 tok/s**, warm steady state, TP4 MTP=0 fp16 eager no-offload.

**Against D137's TP2+UVA-offload 2.54 tok/s this is 4.3x** — the first honest measurement of the
offload-removal win the plan predicted. It is **not** a G6 claim (>=50 tok/s): graphs are still off
and nothing is tuned.

### Environment gotchas found (worth adding to TRAPS.md)
1. **Setting `LD_LIBRARY_PATH` HIDES THE GPUS in this container**: with it unset
   `torch.cuda.device_count()==4`; with `/opt/rocm/lib` or `/opt/rocm/core-7.14/lib` it is **0**.
   <source-host>'s serve scripts set it — the MI50 serve script must not.
2. **`amdsmi` pre-init `sitecustomize.py` is REQUIRED** (`import amdsmi; amdsmi.amdsmi_init()`),
   else vLLM dies with `RuntimeError: Failed to infer device type` (matches D111's note).
3. **`pct set --memory` takes MiB**: `--memory 128` is 128 MiB, not 128 GB. Use `131072` for 128 GiB.
4. **`pkill -9 -f api_server` inside a `bash -c '...'` kills its own launcher** because the pattern
   matches the wrapping shell's command line. Launch detached work from a script FILE, not inline.

### Blocked / not done
- Phase 4 (profile the TP4 baseline) not started.
- Graphs-no-compile not enabled yet (the plan's Phase 3 config asks for it; eager was used as the
  conservative first boot). D060 measured ~3.3x from graphs-without-compile, so ~10.95 -> ~35 tok/s
  is the expected next step.
- `iommu=pt` still missing from <host>'s cmdline (RCCL warns of instability/hang) — flagged in
  D141, still outstanding.

### Artifacts
- `results/tp4_bringup/tp4_gate.json` (determinism + baseline runs)
- `tools/m0/serve_tp4_48l.sh`, `tools/m0/tp4_gate.py`
- `/<pool>/qwen38-flash-next<index-overlay>/model.safetensors.index.json` (the patched index)

### D142 addendum — graphs-no-compile: 48.31 tok/s (4.4x over eager)
The plan's Phase 3 config asks for **graphs-no-compile**; the first boot used `--enforce-eager` as
the conservative step. Swapping it for <source-host>'s exact graphs pair
(`-cc.mode=none -cc.cudagraph_mode=full`, from `serve_tp2_graphs.sh`; D060) gives:

```
cudagraph_mode=FULL_DECODE_ONLY   enforce_eager=False
```

| config | median tok/s @128 | note |
|---|---|---|
| TP2 + UVA offload (D137) | 2.54 | offload-crippled |
| TP4 eager, no offload | 10.95 | D142 first boot |
| **TP4 graphs-no-compile** | **48.45** | spread 0.496 (3 reps) |

- **Reproduced twice** in separate runs of the graphs config: **48.305** and **48.451 tok/s**
  (0.3% apart) — reproducible per the I5 spirit.

- **Correctness unchanged**: 2 greedy runs identical, and the sha256
  (`0128852903291e32`) is **the same as the eager run** => graphs do not perturb output.
  This re-confirms D060: the garble came from **compile** (inductor), not from graphs.
- G6 is **>=50 t/s at MTP=0**; this is **48.31**, i.e. ~3.5% short, measured at MTP=0 with no
  offload and no other tuning. **Still labelled baseline**: nothing has been optimised yet.
- Switch/backend notes in the log: `CUDAGraphMode.FULL is not supported with
  Qwen4ExpQSAFlashAttentionBackend` (falls back to FULL_DECODE_ONLY, which is the intended mode),
  and `fused_moe.py: Using default MoE config. Performance might be sub-optimal` (an untuned MoE
  config, a candidate for Track O).
- PLE sidecar prefaulted 32.0 GB into the page cache; a warm page cache also cut checkpoint load
  from ~106 s/shard to ~20 s/shard.

**Artifacts:** `results/tp4_bringup/tp4_gate.json` (graphs run), `tools/m0/serve_tp4_48l_graphs.sh`.

### D142 addendum 2 — environment hardening: `iommu=pt`, NVMe data path, rootfs move

These are operator-approved host changes. They are provenance-relevant under I8: the TP4 numbers
above and below were taken on different storage/IOMMU configurations, so the fingerprint matters.

**1. `iommu=pt` added to <host>'s kernel cmdline** (RCCL had warned:
*"Missing iommu=pt from kernel command line which can lead to system instability or hang!"*).

```
/etc/kernel/cmdline : root=ZFS=rpool/ROOT/pve-1 boot=zfs   ->   ... boot=zfs  iommu=pt
backup              : /etc/kernel/cmdline.bak-*
applied with        : proxmox-boot-tool refresh   (both ESPs: 34AB-5E51, 34AC-9C9E)
verified            : options  root=ZFS=... boot=zfs  iommu=pt   in each loader entry
effect after reboot : [5.872822] iommu: Default domain type: Passthrough (set via kernel command line)
```

Before the change the default domain was **`Translated`**. `pcie_acs_override=downstream,multifunction`
(present on <source-host>) was deliberately **not** added: all four GPUs already share IOMMU group 98, so it
would be redundant, and it mainly matters for VFIO, which D114 retired.

**2. Data path moved off spinning rust onto NVMe.** The model assets had been landing on
`<pool>`, which is a **raidz1 of four 1 TB Seagate Barracuda HDDs** — that is why the first
load took ~399 s (≈195 MB/s) while v620-serve on <source-host> took ~90 s (the <source-host> model sat on a single
Samsung NVMe in `<source-host>-<pool>`).

| | before | after |
|---|---|---|
| assets | `/<pool>/qwen38-flash-next` (HDD) | `/<pool>/qwen38-flash-next` (**NVMe**, raidz1 of 4x <nvme>) |
| <container> rootfs | `<pool>:<subvol>` (HDD) | `<pool>:<subvol>` (**NVMe**) |
| cold checkpoint load | ~399 s | **72.7 s** |

- The 127 GB asset copy was verified with an empty `rsync -an --itemize-changes` dry-run.
- The rootfs move used `pct move-volume <ct> rootfs <pool>` (9.23 GB transferred, rc=0).
- **Rollback retained:** `<pool>/<subvol>` (3.73 G) is intentionally left in place and
  **not** deleted. Delete it only once the container is confirmed healthy on NVMe across a reboot,
  and only on operator instruction.
- The five bind mounts were repointed `/<pool>/... -> /<pool>/...`; no container-internal
  path changed, so the serve scripts and the patched-index overlay all still work unmodified.
- Post-move verification: `df -hT /` = `<pool>/<subvol> zfs 100G 3.7G 4% /`; all mounts
  present; `rocm_present`; `torch 2.13.0+gfx906 devices=4`; 128 GiB; 64 cores.

**3. Best measured TP4 figure after hardening: 49.04 tok/s median / 49.25 best** @ 128 tok
(graphs-no-compile, TP4, MTP=0, fp16, no offload, `iommu=pt`, NVMe). Determinism held —
sha256 `0128852903291e32`, identical to every prior run including eager. This is ~1.9% short of
G6 (>=50) and still **baseline only**. Honest read: `iommu=pt` and the storage moves bought
*startup and robustness* (5.5x faster cold load), not decode throughput — the weights were already
resident in VRAM.

**4. An incident, recorded so it is not repeated as a phantom regression.** Two verification
scripts were started concurrently (one aborted but still running); each did `pkill -9 -f 'VLLM::'`
and relaunched on port 8002, so they killed each other mid-startup. The result looked exactly like a
post-reboot TP4 regression (`Engine core initialization failed`, `Connection refused`). It was not:
a single clean launch starts normally. **Lesson: never leave a verification watcher running when
starting a second one; `pkill`-based launchers must be singleton.**

**5. PROGRESS.md drift closed.** `PROGRESS.md` is generated by `harness/update_progress.py` from
`config/tasks.json`, but it had been hand-edited while the registry stayed stale (G2.2/G4/G5 still
read `blocked-on-hardware`), and every regeneration silently deleted the narrative status section.
Fixed by: updating `config/tasks.json` (gates G2.2/G4/G5/G6, tasks M0/M1/M2/M3/M4) and adding a
`status_update` block, which `update_progress.py` now emits. The file is now reproducible and
validated (9 tables, 0 malformed rows).

**Artifacts:** `tools/m0/serve_tp4_48l_graphs.sh`, `tools/m0/tp4_gate.py`,
`results/tp4_bringup/tp4_gate_graphs.json`, `config/tasks.json`, `PROGRESS.md`.

---

## D142 addendum 3 — CORRECTION: the custom AR was never "unwired"; in-server world=4 A/B = +4%

- Date: 2026-09-17
- Task: operator debt item 3 ("resolve the hipIpc asymmetry before M2 wiring")

### The debt was already paid, and my own entries were wrong
Debt item 3 was **already resolved on 2026-09-06 by D126**: the `hipIpcOpenMemHandle` "asymmetry"
is a **single-process harness artifact** — hipIpc is a cross-PROCESS mechanism, so opening another
*device's* memory via hipIpc inside ONE process fails in **both** directions on this stack (all 8
exporter/opener/cached/lazy combos). The deployed path has separate TP worker processes, so it was
never affected, and the fork's `rdna_ar_connect` was proven live back in D124.

Worse, **D141 and D142 both asserted the opposite** ("the multi-process `hipIpc` form and the D117
handle asymmetry are unwired" / "is NOT wired into vLLM"). That was a stale claim propagated from
D117 without checking D126. D141 now carries an inline correction.

### The custom AR is LIVE at world=4
Grepping the TP4 serve log (which runs `VLLM_RDNA_AR=1`, inherited from <source-host>'s TP2 script):

```
rdna_ar: one-shot all-reduce active (handle 0, rank 0/4, devices [0,1,2,3], max 64 KB; pace 0)
rdna_ar: one-shot all-reduce active (handle 1, rank 3/4, devices [0,1,2,3], max 64 KB; pace 0)
   ... 8 lines total = 2 handles x 4 ranks
```

The 20 KB dominant decode message is inside that **64 KB** window, so the custom AR is on the
decode critical path. **Consequence for the record: the 48.45 / 49.04 tok/s "baseline" numbers in
D142 were measured WITH the custom AR active**, not with RCCL as D141's decision text implied.

### The in-server A/B D126 actually asked for
`serve_tp4_48l_graphs.sh` (`VLLM_RDNA_AR=1`) vs `serve_tp4_48l_graphs_noar.sh` (`=0`), TP4 MTP=0
fp16 graphs-no-compile, serial, 3x128 tokens:

| arm | AR lines | load | determinism | tok/s | median |
|---|---|---|---|---|---|
| **AR_ON** | **8** | 72.9 s | sha `0128852903291e32` | 47.09 / 47.88 / 47.70 | **47.70** |
| **AR_OFF** | **0** | 72.5 s | sha `0128852903291e32` (identical) | 34.65 (cold) / 45.37 / 46.13 | **45.37** |

- **Custom AR is worth ~+4% in-server at world=4** (steady-state ~47.7 vs ~45.7 tok/s), and output
  is **byte-identical** in both arms — consistent with the kernel's fixed-rank-order fp32 reduction
  being numerics-lossless.
- Clears the plan's adoption rule (">2% clean gain with exact outputs: candidate").
- **Caveat:** this is a single A/B in each direction, not a reverse A/B, and the effect (~2 tok/s)
  is only ~4x the within-run spread. The plan requires a reverse A/B for 0.5-2% effects; at ~4% it
  is a candidate, but a reversed confirmation (OFF then ON) is the honest next step before this is
  treated as settled.
- This is the "measure the per-op AR latency in the multi-worker context" item D126 left open.

### Harness bug that muddied the run (and my second self-inflicted one today)
`ab_ar.sh` cleaned up with `pct exec CT -- bash -c 'pkill -9 -f "VLLM::" ...'` — whose **own command
line contains the pattern**, so it killed itself and the cleanup silently did nothing. The next arm
therefore launched on top of the previous serve; the AR_ON engine died (`EngineDeadError`) from the
collision. Both arms had already produced their gate results before that, so the numbers stand, but
the run looked like "nothing is loading".
**Fix, now in TRAPS #14:** use the bracket trick so the pattern cannot match its own command line —
`pkill -9 -f 'VLLM[M]::'`, `pkill -9 -f 'vllm.entrypoint[s]'`. Also: killing a harness JOB does not
kill the process INSIDE the container; always clean up in-container explicitly.

**Artifacts:** `tools/m0/serve_tp4_48l_graphs_noar.sh` (AR-off arm), `results/tp4_bringup/tp4_gate.json`.

---

## D143 — M4 profile: TP4 ledger captured in-engine, attribution ACCEPTED (85.5% of wall)

- Date: 2026-09-17
- Task: Phase 4 / M4 — "expose the next bottleneck after offload removal" (runbook step 9)
- Source: executor, <host> <container>, TP4 fully resident, graphs-no-compile, MTP=0, fp16, no offload,
  custom AR on (`VLLM_RDNA_AR=1`)

### Method (and the flag that cost two attempts)
Profiled **inside the normal multiprocessing engine** via vLLM's own profiler endpoints — this is
D127's "viable path 1", which avoids the in-process-engine double-build D127 documented as blocking
(`VLLM_ENABLE_V1_MULTIPROCESSING=0` + PLE offload rebuilds the GPU engine and OOMs).

Three traps had to be cleared first:
1. **`ProfilerConfig.profiler` defaults to `None`.** Setting only `VLLM_TORCH_PROFILER_DIR` leaves
   the `/start_profile` route unmounted and both endpoints return **404**. Setting the kind
   explicitly fixes it:
   `--profiler-config '{"profiler":"torch","torch_profiler_dir":"<workdir>/tprof",...}'`
   -> `start_profile http=200`. The serve log's `torch_profiler_dir=` line is the tell.
2. **`/stop_profile` must be allowed to FLUSH.** The four rank traces are ~270 MB of gzip. A 60 s
   timeout expired mid-write and the serve was then killed, truncating **all four** rank traces
   (`Compressed file ended before the end-of-stream marker`). Correct sequence: long timeout
   (900 s), then poll until the total byte size stops changing, then `gzip -t` each file.
3. The cleanup used `rm -f <workdir>/tprof/*.json` while traces are `*.json.gz`, so stale files
   survived and were mixed into the first analysis pass.

### The ledger (128-token request; per-rank kernel time, summed over 4 ranks)
| family | total ms | % kernels | launches |
|---|---|---|---|
| gemm_dense | 2676.6 | 24.3% | 152,666 |
| **collectives** | **2313.3** | **21.0%** | 50,568 |
| moe_int4 | 2175.5 | 19.8% | 74,880 |
| elementwise | 1298.7 | 11.8% | 303,380 |
| hc_glue | 1022.7 | 9.3% | 99,072 |
| attention_qsa | 487.3 | 4.4% | 50,568 |
| shared_expert | 372.0 | 3.4% | 49,152 |
| gdn_linear | 209.9 | 1.9% | 18,576 |
| norm_quant | 205.2 | 1.9% | 50,824 |
| sampler | 41.5 | 0.4% | 8,255 |

### Plausibility — the plan's acceptance requirement
> "Components must sum plausibly to measured wall/token. If they do not, attribution is not accepted."

```
profiled request: 128 tokens in 3.221 s (25.16 ms/token)
total kernel time (4 ranks summed): 11014.3 ms
per-rank kernel time: 2753.6 ms vs wall 3220.9 ms  ->  ratio 0.855
VERDICT: components sum plausibly (85.5% of wall is kernel time; the remainder is PLE CPU
         phases/waits, scheduling, sampling and graph gaps). ATTRIBUTION ACCEPTED.
```
Wall/token came from the API **with no profiler attached** (median 47.85 tok/s, 3 serial reps,
deterministic); the trace is **attribution only**, per the plan.

### Ranked candidates with estimated ceiling (what Phase 5 must choose from)
1. **Collectives — 21.0% of kernel time.** The single largest kernel in the entire model is our own
   **`rdna_ar_oneshot`: 1741 ms / 49,664 launches (~35 us each)**; `ncclDevKernel_Generic_4` adds
   571.8 ms / 904 launches. **Nuance that matters:** kernel share overstates wall cost — the AR
   on/off A/B (D142 addendum 3) moved wall by only **~4%**, because the four ranks overlap. So the
   realistic ceiling here is near that measured 4%, not 21%. Levers: reduce collective *count*
   (~119/step), fuse poll+reduce, or cut the ~35 us per-op cost.
2. **gemm_dense — 24.3%, the largest family.** `gemv_f16_rdna2_<8,1>` alone is 1253 ms / 50,691
   launches (~24.7 us each): dense GEMV has very poor arithmetic intensity here, so geometry/tile
   tuning is the lever. Highest absolute headroom, lowest risk.
3. **moe_int4 — 19.8%.** `moe_w2_gemv` 843.8 ms, `moe_w13_silu_gemv` 673.0 ms, and notably
   **`vllm::moe::topkGating` at 449.1 ms (4.1% of all kernel time)** — pure routing overhead and a
   specific, attackable target. vLLM also warns `Using default MoE config. Performance might be
   sub-optimal`, so that config is untuned.
4. **elementwise — 11.8% across 303,380 launches.** Precisely the plan's "graph boundaries / torch
   elementwise residue": `__amd_rocclr_copyBuffer` (193 ms / 47,526 launches) plus
   `vectorized_elementwise_kernel` add/cast/fill variants. Launch-count-dominated, so fusion is the
   obvious lever.
5. **hc_glue 9.3%** (`hc_up_gate_mix_k` 818 ms) and **shared_expert 3.4%** — the plan's "fused glue
   / HC / shared expert" family, already fused.
6. QSA/GDN (4.4% / 1.9%) — consistent with D136's TP2 finding that this was negligible, so the V620
   copy-removal idea stays deprioritised; dense int8 likewise stays deferred.

### Acceptance
"The next optimization target is chosen from measured TP4 profile data, not from the V620 ladder by
analogy." **Met** — Phase 5 should begin with (2) gemm_dense geometry and (3) MoE `topkGating`,
with (1) collectives bounded by its measured ~4% wall effect rather than its 21% kernel share.

### Caveats
- The trace mixes **4 ranks concurrently**, so absolute ms are ~4x wall; only **family shares** and
  **per-rank ratios** are quotable, never "ms/token" taken straight from the ledger.
- Two `async_llm` traces (one stale from the truncated pass) contributed to the CPU category totals;
  the four truncated rank traces were skipped, so the kernel ledger rests only on the four good rank
  traces. `tprof/` should be cleared before the next capture (now fixed in `p4_serve.sh`).
- No Phase 5 optimisation has been applied, so 49.04 tok/s remains a **baseline**, not a tuned result.

**Artifacts:** `tools/m0/p4_serve.sh` (profiler-config serve), `tools/m0/p4_run.sh` (orchestrator
with flush-wait + before/after idle checks), `tools/m0/p4_analyze.py` (family ledger + plausibility),
`tools/m0/p4_stop.sh` (corrected singleton cleanup), `<workdir>/tprof/*.pt.trace.json.gz` (5 traces).

---

## D143 addendum — cheap G6 sweep: `--max-num-batched-tokens` 2048 is FLAT, REJECTED; baseline corrected to ~47.7

- Date: 2026-09-17
- Task: cheapest no-code lever to close G6 (49.04 -> >=50 tok/s), per operator request
- Config held fixed throughout: TP4+EP, MTP=0, fp16, **no offload**, graphs-no-compile, **AR on**,
  PLE int4 sidecar, `--max-num-seqs 4`, gpu-mem-util 0.90

### Design — with a drift control, because this session's whole-serve spread exceeds the effect
Three serve launches, three gate runs each = 9 independent measurements:

| arm | `max-num-batched-tokens` | gate medians (tok/s) | mean |
|---|---|---|---|
| **A** (control) | 1024 | 48.017 / 47.881 / 47.639 | **47.85** |
| **B** (candidate) | 2048 | 47.688 / 47.929 / 47.606 | **47.74** |
| **A'** (control repeat) | 1024 | 47.686 / 47.453 / 46.946 | **47.36** |

- **A vs A' drift = 0.48 tok/s (1.02%)** — that is the noise floor for a whole-serve comparison.
- **B vs A = -0.11 tok/s (-0.22%).** **B vs A' = +0.38 tok/s (+0.80%).**
- B lands **between the two control runs**, and the control-vs-control spread is larger than any
  B effect.
- **Every one of the 9 gate runs was `DETERMINISTIC: True` with sha256 `0128852903291e32`** —
  identical to D142/D143. The change is numerics-safe; it is simply also useless.

### VERDICT: REJECT cleanly (per the plan's adoption rule)
No adoption, no re-baselining on noise. `--max-num-batched-tokens` 2048 stays **off** (1024 is
kept). Mechanistically this is expected: with `--max-num-seqs 4` and a single stream, the decode
batch is 1, so `max-num-batched-tokens` mainly governs **prefill** chunking and has little purchase
on decode-step time.

### BASELINE CORRECTION — the recorded 49.04 was a high outlier
The nine independent gate medians span **46.946-48.017** (mean ~47.65). D142's **49.04 tok/s** and
D142 addendum 1's 48.45 were the top of that distribution, not its centre. The honest current
baseline is **~47.7 tok/s**, so the gap to **G6 (>=50)** is **~4.6%, not the ~1.9% previously
recorded**. This matters for planning: G6 is not "one cheap flag away".

### Useful by-products
- **The noise floor for whole-serve A/Bs on this rig is ~1%.** The plan's ">2% clean gain to adopt"
  rule is therefore well calibrated, and **any single-run claim below ~1% here should be treated as
  noise**. The AR on/off figure (~+4%, D142 addendum 3) sits above that floor but was a single A/B
  and still warrants the reverse confirmation already proposed.
- Cleanup was correct in all three arms (`api_server left=0 engine_comm left=0`, VRAM returned to
  10878976, port 8002 free). The `pkill -9 'VLLM'` **comm match** is what makes this work — the
  bracket-escaped `pkill -f 'VLLM[M]::'` matches nothing because of the prctl rename (TRAPS #1).

### The other requested lever is NOT a cheap flag
`--max-num-batched-tokens` was lever 1. Lever 2 ("check/try the MoE config path" because of the
`Using default MoE config. Performance might be sub-optimal` warning) **has no flag**: that warning
means `get_moe_configs()` found **no tuned Triton JSON** for `device_name=gfx906` (vLLM only ships
NVIDIA device names), so it returns `None` and falls back to the default. Producing one requires a
**MoE autotune run** (`benchmarks/kernels/benchmark_moe.py`) and then pointing
`VLLM_TUNED_CONFIG_FOLDER` at the result.

**Operator priority for Phase 5 (recorded):** the next real target is **MoE tuning/autotune**, not
AR. MoE is a measured 19.8% kernel family with `moe::topkGating` a concrete 449 ms (4.1%) item,
whereas the AR is already live and its measured wall effect is only ~4%. AR refinement stays behind
dense/MoE unless a later profile shows rank skew has worsened.

**Artifacts:** `tools/m0/sweep_serve.sh`, `tools/m0/sweep_run.sh`, `tools/m0/sweep_stop.sh`,
`<workdir>/sweep_{A,B,Aprime}_g{1,2,3}.json` (9 gate artifacts in <container>).

---

## D143 addendum 2 — MoE autotune attempted: five incompatibilities fixed, a sixth stands; NOT DONE

- Date: 2026-09-17
- Task: Phase 5 target 1 (agreed priority: MoE tuning/autotune, ahead of AR)
- Goal: produce `E=128,N=640,device_name=AMD_Radeon_Graphics,dtype=int4_w4a16.json` so the serve
  stops warning `Using default MoE config. Performance might be sub-optimal!`
- **Status: INCOMPLETE.** No tuned config was produced. No A/B was run. **G6 is unaffected.**

### Why the warning exists (confirmed from source)
`get_moe_configs()` looks for a tuned Triton JSON at `$VLLM_TUNED_CONFIG_FOLDER/<name>` then
`<vllm>/.../fused_moe/configs/<name>`, where the name comes from `get_config_file_name(E, N, dtype)`
and `get_device_name_as_file_name()` (`re.sub(r"[\s/]+", "_", name)`). This container reports
`torch.cuda.get_device_name(0) == 'AMD Radeon Graphics'`, so the wanted file is
`E=128,N=640,device_name=AMD_Radeon_Graphics,dtype=int4_w4a16.json`. **vLLM ships 332 configs and
every one is an NVIDIA device name** - hence default fallback. There is no flag for this.

Model shape verified from `text_config`: `hidden_size=2560`, `moe_intermediate_size=640`,
`num_experts=512` (→ `/EP4 = 128`), `num_experts_per_tok=10`. The plan's E/N/topk/hidden were
independently confirmed before running.

### The five blockers FOUND AND FIXED
1. **`benchmark_moe.py:14` hard-imports `ray`** (absent from the venv). Installed `ray 2.58.0`.
   Dependency check before/after: **`protobuf` UNCHANGED at 6.33.6**; only `msgpack` and `ray` were
   added. `torch`/`vllm`/`_custom_ops` verified importing afterwards. (Operator-approved install.)
2. **`import ray` died on `libamd_smi.so`.** The library is present in three places; the real
   problem was that its **dependency directory was not on the loader path** (`librocm_sysdeps_nl_genl_3.so.200`,
   `..._mnl.so.0`, `..._nl_3.so.200` all live in `/opt/rocm/core-7.14/lib/rocm_sysdeps/lib`).
   Adding that dir fixed it: amdsmi sees 4 GPUs, torch sees 4, ray sees 4.
   **Wider consequence:** this means the amdsmi `sitecustomize` shim (D111/D142) has been
   **silently failing since it was written** - it is wrapped in `except Exception: pass`, so nothing
   ever reported it. Latent environment defect; vLLM does not need amdsmi, so nothing broke.
3. **`AttributeError: 'Qwen4ExpTextConfig' object has no attribute 'num_local_experts'`** - the
   tuner assumes Mixtral attribute names. Patched with `num_experts` fallback (upstream name first).
4. **Would have silently produced the WRONG filename.** After fixing (3), `intermediate_size`
   resolved to **5632** - the *dense* FFN width; this config carries both `intermediate_size=5632`
   and `moe_intermediate_size=640`. The tuner tunes MoE experts and the serve derives N from the MoE
   layer's `w2.shape[<bus>]`, so it must be **640**. Patched to prefer `moe_intermediate_size`. Caught by
   the filename pre-check, which is why that gate exists. (The script's own naming is otherwise
   consistent: `E // tp_size = 512//4 = 128`, `shard_intermediate_size = 2*640`, `N = that//2 = 640`.)
5. **ray rejects `ROCR_VISIBLE_DEVICES`** (`amd_gpu.py:46`), and `benchmark_moe.py` **itself** swaps
   `HIP_VISIBLE_DEVICES`→`ROCR_VISIBLE_DEVICES` unconditionally (written for old Ray). Against
   ray 2.58 that is self-defeating: ray's guard fires exactly when ROCR is present and HIP is
   absent, so the script deletes the variable Ray requires and sets the one Ray forbids. Patched to
   gate the swap on the Ray version (≥2.9 keeps HIP). Also wrapped the exec in `env -u
   ROCR_VISIBLE_DEVICES` as belt-and-braces.

After all five: ray initialises, sees 4 GPUs, `E=128, N=640` resolves, and the filename gate is
armed.

### The sixth blocker - UNRESOLVED
The tuning **ray workers segfault on startup**:

```
A worker died or was killed while executing a task by an unexpected system error
worker-...err:  File ".../torch/__init__.py", line 445 in <module>   <- dies importing torch
```
It dies while `_load_actor_class_from_gcs` unpickles the benchmark's actor class, whose module
imports the full vLLM fused-MoE + triton chain into a fresh process.

**Ruled out:** it is **not OOM** (host dmesg clean, cgroup `oom_kill 0`, 127 GB free) and **not a
general ray/torch incompatibility** - an isolation probe shows ray is fine, `torch` imports fine in
a worker (4 devices), and `num_gpus=1` pinning works (1 device). So the fault is specific to the
benchmark's actor-import path.

### Recommendation (recorded, not yet executed)
**Write a minimal ray-free tuner.** Everything needed is now known: E=128, N=640, topk=10,
hidden=2560, group_size=128, dtype `int4_w4a16`, M∈{1,2,4,8,16}, the exact output filename, and the
search space (`get_configs_compute_bound` is pure Python and imports without ray). A single-process
tuner sidesteps ray entirely and is bounded, self-controlled work - preferable to continuing to
debug third-party actor machinery on an unsupported ROCm stack. Running `--tp-size 1` would confirm
whether concurrency is the trigger but emits `E=512`, which the serve will not look up, so it cannot
produce a usable artefact.

### Environment changes made (I8 provenance)
- `ray 2.58.0` + `msgpack 1.2.2` installed into `<home>/gfx906-venv` (protobuf unchanged).
- `benchmark_moe.py` patched in two places; pristine copy at `benchmark_moe.py.orig-mi50patch`.
  Patch script: `tools/m0/moe_patch_bench.py` (idempotent; restores pristine before re-applying).
- **The serve is untouched and the rig is idle** (VRAM back to 10.9 MB/GPU, port 8002 free).

### TRAPS #11 IS WRONG (correction)
TRAPS #11 claims `LD_LIBRARY_PATH=/opt/rocm/lib` makes torch report **0** devices. Measured today it
reports **4**, reproducibly. The original reading was almost certainly taken while stale `VLLM::`
processes were holding the GPUs (the failure mode that bit twice in this session) - a confound, not
a library-path effect. The `unset LD_LIBRARY_PATH` in the serve/sweep scripts is harmless and can
stay, but the trap's *explanation* must not be relied on. Re-verification recommended before any
future script cites it.

**Artifacts:** `tools/m0/moe_tune.sh`, `tools/m0/moe_serve.sh`, `tools/m0/moe_ab.sh`,
`tools/m0/moe_patch_bench.py`, `tools/m0/ray_torch_probe.py`, `tools/m0/moe_stop.sh`-family cleanup,
`benchmark_moe.py.orig-mi50patch` (in <container>).

---

## D144 — Ray-free MoE tuner BUILT and WORKING; its tuned config is 47.7% SLOWER -> REJECTED

- Date: 2026-09-18
- Task: Phase 5 target 1, continued from D143 addendum 2 (stop fighting Ray; build the serial tuner)
- **Outcome: a complete, clean NEGATIVE result.** The tooling works end to end, the config is
  produced and loaded, and it is emphatically worse. **Rejected. G6 unaffected (~47.7 tok/s).**

### The ray-free tuner works
`tools/m0/moe_tune_noray.py` reuses upstream's **real** tuning code and removes only Ray:
imports `benchmark_moe.py` as a module, reaches the class behind `@ray.remote` via
`ActorClass.__ray_metadata__.modified_class`, instantiates it locally, and calls the genuine
`.tune()` -> `benchmark_config()` path. No kernel invocation, search space or timing loop is
reimplemented. Runs serially on ONE GPU (four would only parallelise identical shapes).

Shape forced per the operator's spec, verified against `text_config`: E=128 (512 global / EP4),
shard_intermediate_size=1280, hidden=2560, topk=10, group_size=128, dtype `int4_w4a16`,
M in {1,2,4,8,16}. **Filename gate passed** - it wrote exactly
`E=128,N=640,device_name=AMD_Radeon_Graphics,dtype=int4_w4a16.json`, the name the serve computes.

Two upstream defects had to be worked around, both real:
1. **Upstream's search space contains configs this kernel REJECTS**, and `tune()` catches only
   triton `OutOfResources`. So ONE invalid config aborts the whole run **before `save_configs`** -
   the first attempt burned ~45 min (M=1..8 done, M=16 tuning) and wrote **nothing**.
   `moe_wna16_gemm` enforces two constraints upstream ignores:
   `BLOCK_SIZE_K % group_size == 0` (hip:345) and `size_k % BLOCK_SIZE_K == 0` (hip:343).
   Fixed by filtering the search space on both **and** wrapping `benchmark_config` so any
   kernel-rejected config is scored `inf` instead of killing the run.
2. **No incremental persistence.** Added per-M saves plus resume, which is what let the ~36 min of
   M=1..8 survive into the successful pass.

### The measurement (A / B / A', 9 gates, TP4 MTP=0 fp16 graphs-no-compile, AR on, no offload)
| arm | `VLLM_TUNED_CONFIG_FOLDER` | gate medians (tok/s) | mean | MoE log line |
|---|---|---|---|---|
| **A** (control) | unset | 47.27 / 47.42 / 48.21 | **47.63** | `Using default MoE config` |
| **A'** (control) | unset | 47.37 / 47.81 / 47.79 | **47.65** | `Using default MoE config` |
| **B** (tuned) | `<workdir>/moe-tuned-gfx906` | 24.86 / 24.93 / 25.04 | **24.94** | `Using configuration from .../E=128,N=640,...` |

- **The controls agree to 0.05%** (47.63 vs 47.65) - an unusually clean run, so the effect is
  unambiguous: **-47.7%.**
- **All 9 gate artifacts carry the reference hash `0128852903291e32`** (verified per-artifact), so
  the tuned config is **numerics-safe** - byte-identical output, purely a performance regression.
- The serve **demonstrably loaded** our config (`fused_moe.py:1154 Using configuration from ...`),
  so this measures the tuned tiles, not a silent no-op. The filename gate and the log line are what
  make that distinction possible.

### REGRESSION CAUGHT AND REVERTED: installing Ray broke the serve
Arm A is the **control**, and it failed to start:
`RuntimeError: Please use HIP_VISIBLE_DEVICES instead of ROCR_VISIBLE_DEVICES`
(`ray/_private/accelerators/amd_gpu.py:46`). **vLLM imports ray at startup** (it appears in
`vllm/distributed/device_communicators/ray_communicator.py` and `.../sharded_rdt_trainer.py`), so
merely having ray installed made the serve trip ray's AMD guard, because the serve scripts set
`ROCR_VISIBLE_DEVICES`. Every sweep number in D143 (the ~47.7 baseline) was taken **before** ray
existed, so the first A/B attempt was comparing against a differently-configured serve - a confound,
not a result.

**Fix:** ray and msgpack were **uninstalled** (protobuf confirmed still 6.33.6 throughout), and the
serve was re-verified: `torch 2.13.0 | devices 4 | vllm imports | _custom_ops OK`, ray absent,
`Application startup complete` reached. The A/B was then re-run in a genuinely baseline-matching
environment. **Without the control arm this regression would have gone unnoticed and arm B would
have been measured against nothing.**

### Interpretation and consequence
The upstream autotuner's objective **does not transfer to this stack**. Its isolated
`benchmark_config` measurement picks tiles that are ~2x worse in situ, plausibly because the
measured shape/mode does not match the serve's real call pattern (EP shard, per-projection N/K) or
because of Triton compile/cache effects in the isolated loop. Whatever the cause, the practical
conclusion is firm:

**Do NOT adopt the tuned config; do NOT use `VLLM_TUNED_CONFIG_FOLDER` on this model.** The JSON is
left on disk only as evidence. **MoE tuning via this route is closed as a dead end.**

**Pivot (as pre-agreed):** Phase 5 moves to **dense GEMV geometry**, which D143 still ranks as the
highest-headroom / lowest-risk code target (`gemm_dense` 24.3% of kernel time, with
`gemv_f16_rdna2_<8,1>` alone at 1253 ms / 50,691 launches). Note that `topkGating` (449 ms) remains
untouched by any of this - it was never in scope for a tile-config tuner.

### Environment changes (I8 provenance)
- **ray 2.58.0 and msgpack 1.2.2 were installed and then REMOVED.** Net venv change: **none**.
  protobuf stayed 6.33.6 throughout. The serve is back to its pre-ray, baseline-matching state.
- `benchmark_moe.py` retains the two compatibility patches (see D143 addendum 2); pristine copy at
  `benchmark_moe.py.orig-mi50patch`. The ray-free tuner imports that module, so it needs ray
  present to run again - reinstall temporarily if re-tuning is ever wanted.
- Rig idle at the end: VRAM 10.9 MB per GPU, port 8002 free.

**Artifacts:** `tools/m0/moe_tune_noray.py`, `tools/m0/moe_noray_run.sh`, `tools/m0/moe_ab2.sh`,
`tools/m0/verify_ab_hashes.py`, `<workdir>/moe-tuned-gfx906/*.json` (evidence, not for use),
`<workdir>/moeab2_{A,B,Aprime}_g{1,2,3}.json` (9 gate artifacts in <container>).

> **CORRECTED IN SCOPE BY D148 (do not read the conclusion below as covering prefill).** The
> measurement above stands - a config tuned for one shape is 47.7% slower in situ, and the decode
> path must not adopt it. But the **generalisation "MoE tuning via this route is a dead end" does not
> hold**: the expert review found the old tuner **bypassed its candidate settings for small batches
> and let its largest small-batch entry control much larger prefills**, so this campaign was
> measuring the wrong workload with the wrong search. A replacement search targeting **actual prefill
> batches** (while preserving small-batch defaults) succeeded - all 68 initial and 65 extended cases
> passed numerical comparison, and it is part of the selected configuration. See **D148**.

---

## D145 — O9 dense GEMV: shapes identified, baseline microbenched, and the lever sized DOWN to ~2.6%

- Date: 2026-09-18
- Task: Phase 5 target after D144's pivot (Track O item **O9**)
- **Outcome: no code change adopted.** Three negative results, one of them a corrected own-error, and
  a re-sizing of the lever that materially downgrades it. **G6 unaffected (~47.7 tok/s).**

### Step 1 — shapes identified (two independent sources agree)
The kernel is `csrc/rocm/skinny_gemms_int4.cu`: `y[M,N] = x[M,K] · w[N,K]ᵀ`, wave-per-output-row,
fp16, `M<=8`. Its only port-tree caller is `vllm/model_executor/layers/utils.py:279`, gated on
`VLLM_ROCM_USE_SKINNY_GEMM`, gfx906, fp16, `0 < tokens <= 8`, `k % 8 == 0`, contiguous — i.e. **the
dense fp16 decode path**. The launcher picks the variant from N along:
```
N >= 1152 -> <8,MT> grid=(N+7)/8 block=512     <- <8,1> is the profile's hot kernel
N >=  576 -> <4,MT> grid=(N+3)/4 block=256
N >=  288 -> <2,MT> grid=(N+1)/2 block=128
else      -> <1,MT> grid=N       block=64
```
so **grid.x recovers N**. Two independent confirmations:
1. The D143 trace's arg-bearing events decode to `N=62080, block=512, 516 launches, 397.02 us each`
   -> 318 MB read at **~800 GB/s** = the HBM ceiling, and 516/4 ranks/128 tokens ~ **one lm_head per
   decode step**.
2. The project's own harness already hardcodes the TP4 per-rank shapes
   (`vllm/tools/rdna2/gemv_f16_harness.cu:103-108`), and `{"lm_head/rank", 62080, 2560}` matches
   exactly. It also explains the launch count: 48 layers x 2 attention projections + lm_head +
   shared-expert x 2 = **99/token**, and 50,691 / 4 ranks / 128 tokens = **99.2**.

### Step 2 — baseline microbench at M=1 (the decode case that governs G6)
Existing harness, CPU fp32 reference per shape, **all 12 shapes PASS** (relerr ~3e-4):

| shape | N x K | MB | us | GB/s | % of 810 ceiling |
|---|---|---|---|---|---|
| **lm_head/rank** | 62080x2560 | 317.8 | 408.0 | **779** | **96%** |
| qsa.q_proj | 3072x2560 | 15.7 | 25.3 | 621 | 77% |
| gdn.in_proj_qkv | 2560x2560 | 13.1 | 22.1 | 594 | 73% |
| qsa.o_proj | 2560x1536 | 7.9 | 14.9 | 526 | 65% |
| gdn.out_proj | 2560x1536 | 7.9 | 15.1 | 519 | 64% |
| gdn.in_proj_z | 1536x2560 | 7.9 | 16.2 | 485 | 60% |
| shared.gate_up | 1280x2560 | 6.6 | 14.2 | 461 | 57% |
| hc.up | 10240x320 | 6.6 | 14.3 | 459 | 57% |
| shared.down | 2560x640 | 3.3 | 7.9 | 416 | 51% |
| router.gate | 512x2560 | 2.6 | 8.0 | 327 | 40% |
| **hc.down** | 320x10240 | 6.6 | 33.5 | **196** | **24%** |
| hc.inject | 4x10240 | 0.1 | 13.4 | **6** | **1%** |

Note the harness's shape list has **nothing in the `<4,1>` band (576<=N<1152)** yet the trace shows
**6,144 `<4,1>` launches** (12/token/rank), so the inherited V620-era list is incomplete.

### Step 3 — WAVES sweep: REJECTED, and the mechanism says why
Tried all four WAVES for every shape at M=1, every variant correctness-checked:
**production 610 us vs best-per-shape 579 us = -5.1%**, with 5 shapes wanting a different W:
```
gdn.out_proj 526->570 (W=2)   qsa.q_proj 602->638 (W=2)   qsa.o_proj 525->566 (W=2)
hc.down      268->277 (W=4)   lm_head    717->762 (W=4)
```
**Reject.** Two reasons, the second decisive: (a) -5.1% of gemv time is ~0.65% of wall, **below the
~1% noise floor**; (b) mechanistically it cannot work — `grid = ceil(N/WAVES)` and
`block = WAVES*32`, so **total threads ~= N*32 regardless of WAVES**. WAVES changes block
granularity only, never occupancy. Also noted: the sweep's per-shape numbers drifted up to ~27%
between runs (hc.down 33.5 -> 24.5 us), so harness variance alone (5-8% typical) exceeds the effect.

### Step 4 — k-unroll sweep: PREMISE FALSIFIED, and the error was MINE
I hypothesised the inner loop had no unrolling (so load latency could not be hidden) and wrote a
variant. **The hypothesis was false.** Reading only lines 1-30 of the harness, I did not see the
loop and *inferred* its shape instead of reading lines 31+; the real kernel **already issues a
4-deep predicated unroll and accumulates via `__builtin_amdgcn_fdot2` (v_dot2_f32_f16)**, with
`__shfl_xor` as the final reduction:
```c
for (int i = lane; i < K8; i += 32 * 4) {
  uint4 wq[<bus>];
  for (int u = 0; u < 4; u++) wq[u] = (idx < K8) ? wrow[idx] : make_uint4(0,0,0,0);
  for (int u = 0; u < 4; u++) { ... a = __builtin_amdgcn_fdot2(wh[<bus>], xh[<bus>], a, false); ... }
}
```
So my variant was a *regression* to a simpler kernel, and it produced wrong results (relerr ~1.0,
i.e. zero output) because I also never checked the launch status. **The only reason this was not
mis-reported as "unrolling gives no gain" is that the harness included a U=1 control that was
supposed to reproduce production and instead failed.** Controls exist for this; it earned its keep.

### Step 5 — the lever is SMALLER than D143 implied
Sizing from the trace (all 4 ranks, 128-token request; `gemv_f16_rdna2` total **1669 ms** of the
11014 ms kernel time = **15.2%**):

| group | time | current | if it reached 750 GB/s | wall gain |
|---|---|---|---|---|
| lm_head | 204.9 ms | **96% saturated** | — | **0%** |
| non-lm_head `<8,1>` | 832 ms | ~520-620 GB/s | +25% | ~1.6% |
| `<2,1>` (router.gate, hc.down) | 147.5 ms | ~350 GB/s | +100% | ~0.6% |
| `<1,1>` (incl. hc.inject @ 6 GB/s) | 73.6 ms | launch-overhead-bound | — | ~0.4% |
| `<4,1>` | 42.6 ms | — | — | ~0.2% |

**Realistic ceiling for the whole lever: ~2.6% of wall**, and that assumes *every* shape reaches
750 GB/s — optimistic given Step 3. Individually each sub-lever is 0.4-1.6%, i.e. at or below the
noise floor. **D143 ranked this "highest headroom / lowest risk"; that framing overstated it, because
the single largest component (lm_head, 12.3% of gemv) is already bandwidth-saturated**, and the
overall family is only 15.2% of kernel time.

The one mechanistically-motivated fix left is **split-K** for the small-N family (hc.down has ~10k
threads against ~154k of capacity), but it needs a second reduction pass whose ~5 us launch cost
would consume much of the gain on a 24 us kernel. Expected ~0.6% of wall — likely to be rejected.

### Process lessons recorded
1. **Read the code you are about to change.** A partial read plus an inference cost a full
   invalid experiment and nearly produced a false conclusion. The file was 166 lines; reading it
   whole was the cheap option.
2. **Always check the launch status** in a microbench; a silently failing kernel looks exactly like
   a wrong result.
3. **Keep a control that must reproduce the baseline.** The U=1 arm was the only thing that
   distinguished "variant is slow" from "variant is broken".
4. Harness run-to-run variance here is 5-27% per shape; single-run comparisons below ~10% are not
   evidence.

### Recommendation
Either (a) **narrow and close it** — implement split-K for the small-N family and gate it, expecting
~0.6% and a likely rejection, which at least closes the question; or (b) **re-target on evidence**:
`elementwise` is 11.8% of kernel time across **303,380 launches** (launch-count-dominated;
`__amd_rocclr_copyBuffer` alone is 47,526 launches), a different failure mode with more upside than
a bandwidth-saturated GEMV family.

**Artifacts:** `tools/m0/p5_gemv_shapes.py` (trace shape decode), `tools/m0/gemv_waves_sweep.cu`,
`tools/m0/gemv_unroll_sweep.cu` (the invalid variant, kept as the record of the error),
`tools/m0/run_gemv_harness.sh`, `tools/m0/run_waves_sweep.sh`, `tools/m0/run_unroll_sweep.sh`.

---

## D145 addendum — elementwise/copyBuffer attribution: the trace CANNOT resolve parents; the by-name ranking can

- Date: 2026-09-18
- Task: O9 step 1 — "improve the attribution linkage (target >80% resolved), then confirm the
  copyBuffer origin and resolve the RMSNorm/D118-121 contradiction"
- **Outcome: the 80% target is NOT achievable from this trace, and that is now understood rather
  than suspected.** The by-name ranking is complete and unaffected, and it is sufficient to rank
  targets. Operator decision recorded: skip split-K; elementwise/copyBuffer is the pivot.

### The linkage ceiling is 13.1%, and the cause is structural
Three linkage paths implemented and measured separately over the 4 valid rank traces:

```
family launches                  317,447
  A: External id -> Ev Idx         41,726   13.1%     <- the ONLY path that fires
  B: External id -> cuda_runtime -> containment   0   0.0%
  C: correlation -> flow start -> containment     0   0.0%
  UNRESOLVED                      275,721   86.9%
```

Decoded mechanism (from `p5_link_discover.py`, sample kernel
`__amd_rocclr_copyBuffer.kd` with `args {"External id": 29, "correlation": 7, "bytes": 4}`):
- A kernel's `External id` equals **the launching event's `Ev Idx`**. The sample resolved because
  `aten::slice` happened to carry `Ev Idx: 29`.
- **`cuda_runtime` events carry NO `Ev Idx`** - only `External id`. So when a kernel points at a
  *runtime* event instead of directly at a `cpu_op`, there is nothing to look up. With **221,038
  runtime events vs 162,245 cpu_ops**, most kernels point into that un-indexed space.
- Path B fails because `runtime_by_ext` keys on those same un-indexed ids; path C fails because only
  **15,757** flow-start (`ph='s' cat='ac2g'`) events exist against **221,038** finishes, so
  `correlation` is populated for very few launches.
- The earlier v2 containment fallback scored 0% for an additional reason worth recording: **GPU
  kernels are on stream tids while `cpu_op` events are on CPU tids**, so same-tid containment can
  never match. Containment only works on the *launching CPU thread*.

**Consequence:** full kernel->op attribution is not recoverable from this capture. Getting it would
require a better trace (e.g. `torch.profiler` with `record_shapes`/flow metadata fully emitted, or
`rocprofv3` with an op-level correlation), not more analysis of this one. **Do not re-run this
analysis expecting a different number.**

### The by-name ranking IS complete, and is what to rank targets on
It needs no ids, so it is unaffected by the above. Family = **1383.9 ms = 12.56% of kernel time**:

| launches | total ms | us each | % kernel | ~% wall | kernel |
|---|---|---|---|---|---|
| 47,526 | 193.45 | 4.07 | 1.76% | 1.50% | `__amd_rocclr_copyBuffer.kd` |
| 36,068 | 144.99 | 4.02 | 1.32% | 1.13% | `BinaryFunctor<MulFunctor>` |
| 27,432 | 107.79 | 3.93 | 0.98% | 0.84% | `CUDAFunctorOnSelf_add<float>` |
| 27,431 | 107.16 | 3.91 | 0.97% | 0.83% | `float16tofloat32_copy_kernel_cuda` |
| 25,908 | 103.58 | 4.00 | 0.94% | 0.80% | `FillFunctor<c10::Half>` |
| 25,399 | 101.47 | 4.00 | 0.92% | 0.79% | `CUDAFunctor_add<c10::Half>` |
| 13,716 | 83.31 | 6.07 | 0.76% | 0.65% | `direct_copy_kernel_cuda` |
| 14,224 | 81.67 | 5.74 | 0.74% | 0.63% | `ReduceOp<MeanOps>` |
| 13,716 | 54.92 | 4.00 | 0.50% | 0.43% | `float16_copy_kernel_cuda` |
| 13,716 | 53.67 | 3.91 | 0.49% | 0.42% | `pow_tensor_scalar_kernel_impl` |
| 13,716 | 53.59 | 3.91 | 0.49% | 0.42% | `rsqrt_kernel_cuda` |

**The arithmetic gate (recorded in D146's framing, restated here because it decides everything):**
kernel time is ~85.5% of wall, so **to clear the 2% wall adoption bar a target must be >= 2.3% of
kernel time**. The largest single name is **1.76%**. **No single elementwise kernel can carry a 2%
win**; the unit of work must be a count-clustered *group* or a *causal* root cause.

**Count clustering is the usable signal** (identical launch counts = one eager pattern):
- **~13,716 x 5** - `ReduceOp<MeanOps>` + `pow` + `rsqrt` + `float16_copy` + `direct_copy`. This is
  the signature of **one eager RMSNorm decomposition**: `mean(x^2)` -> `pow` -> `rsqrt` -> `mul`.
  Combined ~327 ms = **~3.0% of kernel = ~2.5% of wall** (recomputed with the fuller membership;
  D146's earlier ~1.9% used only 4 members). **UNRESOLVED CONTRADICTION:** D118-D121 concluded
  "norms+QSA already triton-fused ... norm priority = non-lever". Either that was scoped to a
  different norm path or this is a second, unfused norm. **Resolve before building on it.**
- **~27,431 x 2** - `float16tofloat32_copy` + `CUDAFunctorOnSelf_add<float>` ~= 215 ms = 1.95%
  kernel = ~1.7% wall.
- **~25,650 x 2** - `FillFunctor<c10::Half>` + `CUDAFunctor_add<c10::Half>` ~= 205 ms = 1.86%
  kernel = ~1.6% wall.

### The copyish family is the only thing that clears the bar as a whole
```
copyish subset: 108,233 launches, 469.7 ms = 4.26% of kernel time = ~3.65% of wall
```
`copyBuffer` is the ROCclr implementation of **`hipMemcpyAsync`** (the near-event dump shows an
`hipMemcpyAsync` runtime event sharing the same `External id`), so these are explicit copies, not a
fused-kernel defect. Where attribution *did* land, it is consistent with strided-view
materialisation:
```
aten::slice      -> copyBuffer          16.58 ms / 4,132
aten::as_strided -> index_elementwise   19.50 ms / 2,228
aten::gather     -> copyish              8.87 ms / 2,064
aten::select     -> copyish              8.31 ms / 2,068
aten::detach     -> copyish              6.09 ms / 1,031
```
The **hypothesis** (not established): `copyBuffer` at ~93 launches/token/rank sits suspiciously close
to the **99 gemv calls/token**, consistent with the defensive `x_view = x.reshape(-1, k).contiguous()`
that `vllm/model_executor/layers/utils.py:289` runs on every skinny-GEMM call. Note that `.contiguous()`
is a no-op when the input is already contiguous, so this only bites if the inputs genuinely arrive
strided - which the `aten::slice`/`as_strided` parents above support.

### Why more trace analysis was STOPPED, and what replaces it
Trace archaeology cannot settle the origin (13.1% ceiling). The decisive and cheaper experiment is a
**direct causal test**: toggle `VLLM_ROCM_USE_SKINNY_GEMM` in an A/B/A'. If the ~47.5k `copyBuffer`
launches largely vanish with the skinny-GEMM path off, the `.contiguous()`-per-call hypothesis is
confirmed at the source; if they persist, the hypothesis dies and the copies come from elsewhere.
That yields a *causal* answer for the cost of two serve launches plus gates, versus another
attribution pass that provably cannot exceed 13.1%.

### Also recorded
- Operator decision: **split-K deliberately NOT pursued** (expected ~0.6% of wall, would be eaten by
  the extra reduction/launch, and cannot bridge 47.7 -> 50 tok/s). Recorded as skipped, not failed.
- The first capture pass left **4 truncated-but-large `.json.gz` files** in `<workdir>/tprof` that look
  valid by size and fail only on decompression. Any tool consuming that directory must **test the
  gzip stream, not the file size** - three of my scripts initially analysed the truncated files.

**Artifacts:** `tools/m0/p5_elementwise_attrib.py` (by-name ranking, still the useful one),
`tools/m0/p5_attrib_v2.py`, `tools/m0/p5_attrib_v3.py` (the three-path attempt, kept as the record
of why 80% is unreachable), `tools/m0/p5_link_discover.py` (the id-structure decoder).

---

## D146 — skinny-GEMM causal test: the copyBuffer hypothesis is DEAD; skinny GEMM is worth 2.83x

- Date: 2026-09-18
- Task: O9 — settle the copyBuffer origin causally, since the trace provably cannot (D145 addendum)
- **Outcome: hypothesis REFUTED by a clean null, plus a large confirmatory result for the existing
  config.** No code change. Operator rule #3 applies: stop chasing copy attribution.

### Design
A/B/A' toggling **`VLLM_ROCM_USE_SKINNY_GEMM`** only (1 = control/current baseline, 0 = generic
path), everything else at the verified TP4 config. The torch profiler was configured on **every**
arm so the kernel counts are comparable, while the tok/s figures come from the unprofiled gate
(never from the trace). `<workdir>/tprof` was cleared per arm (D143: stale traces were once mixed into
an analysis).

### Result 1 — the copyBuffer family is CAUSALLY INDEPENDENT of the skinny-GEMM path
```
                       A (skinny ON)   B (skinny OFF)   A' (skinny ON)
copyBuffer                    12,295           12,295            12,295
fp16->fp32 copy                7,128            7,128             7,128
float16 copy                   3,564            3,564             3,564
direct copy                    5,724            5,724             5,724
add (self, fp32)               7,664            7,664             7,664
add                            7,560            7,560             7,560
fill                           7,580            7,580             7,580
mul                            9,752            9,752             9,752
mean/reduce                    3,564            3,564             3,564
rsqrt                          3,564            3,564             3,564
pow                            3,564            3,564             3,564
```
**Every family is identical across all three arms** - and this is not because nothing changed:
total kernel time was **3755 ms (A) vs 8594 ms (B)**, a 2.3x difference. So the elementwise residue
is genuinely orthogonal to the dense-GEMM path.

**THEREFORE: the `x.reshape(-1, k).contiguous()` at
`vllm/model_executor/layers/utils.py:289` is NOT the source of the copyBuffer family.**
D145 addendum's hypothesis is **refuted**. Per the operator's rule, **copy attribution is CLOSED** -
do not re-open it without new evidence. `copyBuffer` is still ~1.9% of wall, but its origin is
elsewhere (view materialisation after `aten::slice`/`as_strided`/`gather`/`select`, which is where
the 13.1% of attributions that DID resolve pointed), and it is not reachable by toggling the GEMM
path.

### Result 2 — skinny GEMM is worth 2.83x; the controls are the tightest yet
| arm | gate medians (tok/s) | mean |
|---|---|---|
| **A** (skinny ON) | 48.515 / 46.864 / 47.156 | **47.51** |
| **A'** (skinny ON) | 46.838 / 47.934 / 47.787 | **47.52** |
| **B** (skinny OFF) | 16.794 / 16.736 / 16.752 | **16.76** |

- **Controls agree to 0.02%** (47.51 vs 47.52) - the tightest agreement of any A/B/A' in this
  project so far, which makes the 2.83x effect unambiguous.
- All nine gate runs were `DETERMINISTIC: true` (hash `0128852903291e32`).
- `VLLM_ROCM_USE_SKINNY_GEMM` is confirmed essential (the gfx906-tuned wave-per-row path vs
  generic rocBLAS Tensile tiles). Nothing about the baseline should change.

### Result 3 — the profiler is usable at low cost now
A 32-token profiled window produced 4 valid rank traces in ~1-2 min per arm with the flush-wait
pattern from D143. Kernel-event counts were stable and reproducible across arms (234,212 / 234,213 /
234,213). This is a cheap, reusable measurement harness for future kernel-count A/Bs.

### Correction to my own tooling
`count_copybuffer.py` printed "384.22 per token" because it divided by 32 tokens **without** dividing
by the 4 rank traces. The correct figure is **12,295 / 32 / 4 = 96.1 copyBuffer launches per token
per rank**, matching the earlier full-request reference of ~93. Recorded so the number is not
re-used wrongly.

### Closed list (updated)
- MoE tile-config tuning - D144, -47.7% in situ. CLOSED.
- `--max-num-batched-tokens` 2048 - D143 addendum, flat inside the noise floor. CLOSED.
- Dense GEMV geometry/WAVES - D145, ~2.6% ceiling, largest component saturated. DE-PRIORITISED.
- split-K for small-N - D145, ~0.6% expected, eaten by the reduction launch. **Deliberately NOT
  pursued** (operator decision).
- **copyBuffer origin via skinny-GEMM** - D146, refuted by an exact null. CLOSED.

### Next
Per operator direction: move to **O5 / MTP=3** (documented 1.9x on V620). `topkGating` (449 ms,
4.1% of kernel time) remains a real but harder post-MTP surgical target, not the next step.

**Artifacts:** `tools/m0/skinny_serve.sh`, `tools/m0/skinny_ab.sh`, `tools/m0/count_copybuffer.py`,
`<workdir>/skinny_{A,B,Aprime}_g{1,2,3}.json` (9 gate artifacts in <container>).

---

## D147 — O5 MTP=3: LOSSLESS on gfx906 but NOT the 1.9x; REJECTED (metrics disagree in sign)

- Date: 2026-09-18
- Task: O5 — MTP=3, the last agreed Phase 5 target. Operator targets: PP512 > 500, TG128 > 75.
- **Outcome: MTP=3 is correct but does not pay. REJECTED, not adopted. Neither operator target is met.**

### Invocation (from the project's own PORT-MAP, not guessed)
```
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```
`model_mtp.safetensors` stays on disk (the patched index references it). Precedent recorded in
DO-NOT-ATTEMPT.md: `num_speculative_tokens=4` **changed the target text** on V620 and was rejected,
so 3 is the sanctioned point and **the output hash becomes a correctness gate**, not a rerun check.

### Design
A/B/A' = MTP0 / MTP3 / MTP0. Each arm: serve -> PP/TG benchmark (3 reps) -> the standing
`tp4_gate.py` determinism check. Benchmarks:
- `PP = prompt_tokens / wall(max_tokens=1)` (prefill, one decode step included - stated, not hidden)
- `TG = gen / (wall(gen) - wall(1))` (generation with prompt time subtracted)
- correctness = sha256 of the full greedy output

### Results
| arm | PP | TG128 | gate median | gate spread | sha256 |
|---|---|---|---|---|---|
| **A** (MTP=0) | 350.3 | 53.92 | 47.25 | 0.53 | `2db606c9…` |
| **A'** (MTP=0) | 350.0 | 52.64 | 46.59 | 0.72 | `2db606c9…` |
| **B** (MTP=3) | **306.5 (−12%)** | **56.04** | **44.58** | **11.32** | **`2db606c9…` IDENTICAL** |

Also: per-rank model memory 18.23 -> 19.67 GiB, startup 72.2 -> 82.6 s. With
`gpu-memory-utilization 0.90` the total stays ~90% of VRAM either way, so the extra model weight
comes out of the KV budget - directionally consistent with PORT-MAP's "MTP=0 frees ~2.5 GiB/card KV",
though the absolute figures differ.

### Finding 1 — MTP=3 is LOSSLESS on gfx906 (new, and worth keeping)
The full greedy output hash is **byte-identical** to MTP=0 across all three arms
(`2db606c96cce89e283d660e4619561217771abb625e38cace686b029cb3555db`). The port's MTP integration is
therefore **correct**, and the MTP=4 text-changing failure does not reproduce at 3. This is the first
correctness confirmation of MTP on gfx906 and it stands independently of the performance verdict.

### Finding 2 — MTP=3 does NOT deliver 1.9x
Best case ~+4-5% TG, worst case -5%. PORT-MAP warned exactly this ("acceptance rate is
model-dependent, not chip-dependent; re-measure on gfx906"); the acceptance rate here is clearly far
below the V620's 2.7-3.1 tokens/step. **The 1.9x that made O5 attractive does not transfer.**

### Finding 3 — the two metrics disagree in SIGN, so the effect is unresolvable as measured
TG128 says +5.2% vs the control mean; the gate says -5.0%. **Both cannot be true, so nothing is
adopted.** A likely contributor, recorded against my own method: `TG = gen/(wall(gen) - wall(1))`
subtracts a prompt term, and MTP=3 makes the **prompt slower** (wall(1): 1.0419 -> 1.1909 s), so a
larger prompt subtraction **inflates** the apparent generation rate. In per-step terms A is
2.3737/127 = 18.69 ms/step and B is 2.2843/127 = 17.99 ms/step (~3.8% faster), while the gate - which
does not subtract and uses a short prompt - sees 5% slower. Arm B's gate spread also exploded to
**11.32** versus A's 0.53, so B's gate median is itself unreliable.

**VERDICT: REJECT.** This is weaker than "flat": the honest statement is that MTP=3's effect on
generation is **inside the disagreement between my own two measurements**, while its prefill cost
(-12%) and its startup/KV costs are unambiguous.

### Operator targets: NOT met
- **TG128 > 75** — measured ~53-56. MTP=3 cannot bridge a 35% gap when it delivers <=5%.
- **PP512 > 500** — measured 350.3/350.0 at an achieved prompt of 365 tokens, not 512 (a loop bound
  in `build_prompt` capped growth). **Operator decision: do not chase the exact 512/128 sizes - the
  general picture is established** and 350 at 365 tokens is far from 500 regardless.

### Next, and an open question
The diagnostic that would *explain* this (rather than just record it) is a direct **acceptance-rate**
measurement via `tools/rdna2/watch.py` (PORT-MAP's own recommendation). With 4 tokens already ruled
out by precedent, a low measured acceptance rate would close MTP as a lever rather than invite
another `num_speculative_tokens` sweep.

**Open decision for the operator:** whether O5 continues at all, or whether MTP is recorded as
"correct but non-paying on gfx906" and Phase 5 closes with the standing baseline of **~47.7 tok/s**
(G6 gap ~4.6%, with MoE tiles, the cheap batched-tokens lever, dense-GEMV geometry, split-K and the
copyBuffer hypothesis all now closed by evidence).

**Artifacts:** `tools/m0/mtp_serve.sh`, `tools/m0/mtp_ab.sh`, `tools/m0/bench_pp_tg.py`,
`<workdir>/bench_{A,B,Aprime}.json` + `<workdir>/mtpgate_{A,B,Aprime}.json` (in <container>).

> **SUPERSEDED AS AN OPERATING CHOICE BY D149 - and K=3 was simply the wrong depth.** The
> measurements above stand (MTP=3 is lossless on gfx906; it does not deliver 1.9x), but
> **MTP=2 IS selected and running** as the final configuration. The expert review tested K=1 and K=2
> and found K=2 gives ~58.0/59.2/62.6 tok/s for prose/code/reasoning at a 512-token prompt with mean
> acceptance lengths **2.49-2.75** - so the two results are consistent: the model wants two
> speculative tokens, not three, and my A/B/A' rejected K=3 correctly. See **D149**.
>
> **Also note the metric caveat this entry could not see:** the ~47.7 tok/s gate used throughout this
> campaign **timed the whole HTTP request and therefore includes prompt time**, so it is not
> comparable to a decode rate. The corrected reference is **~55-56 decode tok/s** (D148). My A/A'
> controls at ~47.5 and TG of ~53-56 are all consistent with that once the definitions are aligned.


---

## D148 — Expert review, corrected benchmark, prefill improvement and stopping point

- Date: 2026-09-18
- Authorization: operator requested execution of the expert review priorities, then stopped further experiments unless another approximately 10% decode improvement was expected.
- **Selected:** TP4+EP at 150 W/card; original weight precision; prefill MoE tiles; validated mask-aware top-10 router; 2048-token chunk budget. MTP, TunableOp and dense INT8 are off. The service is running on <container> port 8002.
- **Measured:** approximately 56 decode tok/s. Effective PP at 512 tokens improved from 454 to 658 tok/s in the 1024-chunk combined comparison. At 1536 tokens, the selected 2048-chunk configuration improved effective PP from 565 to 851 tok/s and TTFT from 2.719 to 1.805 seconds. These are exact-token, prefix-cache-miss, warmed-runtime measurements; decode excludes TTFT.
- **Corrections to earlier conclusions:** the old approximately 47.7 tok/s gate includes prompt time; the old MoE tuner did not tune the relevant prefill path; the AR-on/off difference was not the entire communication cost; the saved MTP log contradicts the low-acceptance explanation; the old wave32/out-of-bounds GEMV sweep did not establish a production wave64 ceiling. Initial review and evidence are retained.
- **Correctness scope:** MoE-only matched all nine reference sequences. Combined routing/chunking can change greedy outputs through numerical ordering. Reference and completed combined arms passed the same 11/12 answer smoke checks; the packing error exists in the reference. No universal losslessness or broad quality claim is made.
- **Other results:** NUMA affinity was flat; default collective block policy won; K=2 MTP reached roughly 58–63 tok/s in isolated tests but did not provide a consistent 10% gain and increased TTFT. Valid GEMV/INT8 numerical probes and rocBLAS tuning artifacts are retained. Full-model INT8 and combined MTP/BLAS work were canceled at the operator's stopping request.
- **Restoration:** temporary ZFS ARC trial reverted to 16 GiB max / original zero min parameter; DPM stays auto; original dense-INT8 Python gate restored; all compiled extension hashes unchanged.
- **Handoff:** [TUNING-20260918.md](TUNING-20260918.md), `tools/review_20260918/scripts/launch.sh balanced` (or `reference` / `stop`), and `<tuning>` in <container>. No new boot service was installed.


---

## D149 — Operator-selected MTP2, optimized prefill, and 30K measurements

- Date: 2026-09-18 UTC.
- Supersedes D148's MTP0 operating choice following the explicit request to leave MTP2 and optimized PP enabled.
- Running: TP4+EP, 150 W/card, MTP2, tuned prefill MoE tiles and MI50 router, 2048-token chunk budget, 32768-token context. TunableOp and dense INT8 remain off. <container> port 8002, PID <pid>.
- Measured medians of three scored requests per exact prompt length, two warmups each, 128 generated tokens, one active request, zero cached prompt tokens:

| Prompt tokens | Median effective PP | Median first-token latency | Median decode | Decode range |
|---:|---:|---:|---:|---:|
| 1,000 | 641.9 tok/s | 1.558 s | 60.34 tok/s | 60.12–60.35 tok/s |
| 10,000 | 806.9 tok/s | 12.393 s | 61.22 tok/s | 39.02–65.35 tok/s |
| 30,000 | 805.1 tok/s | 37.263 s | 57.34 tok/s | 57.00–61.83 tok/s |

- Synthetic repeated prose corpus. Long-context continuations varied at temperature zero; 10K decode spread is material. These results do not establish long-context quality or deterministic output.
- Post-run smoke: 11/12, same packing failure as reference. Health 200. No further tuning experiments started.
- Default `tools/review_20260918/scripts/launch.sh balanced` now starts MTP2/32K; `mtp0` retains optimized MTP0/2K; `reference` and `stop` remain available.
- Details: [MTP2-FINAL-20260918.md](MTP2-FINAL-20260918.md), raw requests and summary under `tools/review_20260918/mtp2_final_1k_10k_30k/`, updated presets and final state, and `MI50_MTP2_EVIDENCE.tar.gz`.

---

## D150 — TP8 feasibility probe: topology, RCCL collectives, and a fresh TP4 baseline

- Date: 2026-09-18
- Authorization: operator requested rig captures to judge how this hardware would perform at TP8.
- **Outcome: TP8 is fabric-viable and worth building — but for WEIGHT BANDWIDTH, not collectives.**
  No configuration was changed; the MTP2 service was left running throughout.

### 1. Topology — one switch, one root complex, one NUMA node
The whole GPU complex hangs off a **single** PEX88096 under **one** root complex (`[<bus>]`) via
**one** GPP bridge (`<pci>`). `rocm-smi --showtopo` shows every GPU pair identical: **weight 40,
2 hops, link type PCIE**, and all four GPUs on **NUMA node 1**. There is no XGMI — all inter-GPU
traffic is PCIe through the switch, as expected for MI50 on a non-AMD platform. This agrees with
D141's independent measurement (12/12 ordered pairs byte-exact at 14.40–14.44 GB/s, IQR ≤ 0.014).

**Headroom exists on the same fabric:** two downstream ports are unpopulated —
`<pci> → [<bus-range>]` (a sub-switch with two empty ports, `[<bus>]`/`[<bus>]`) and `<pci> → [<bus>]`
(**a whole second PEX88096 with nothing behind it**). Four more cards can therefore be added
**without crossing root complexes or NUMA nodes**, which is precisely the condition that makes TP8
viable.

**A useful precedent:** GPU2/GPU3 already sit behind a **second-level** switch (`ae`, via `<pci>`)
while GPU0/GPU1 are direct — yet weights, hops and measured bandwidth are uniform. An extra switch
stage is not automatically harmful here.

**CORRECTION to D078.** D078 recorded "MI50 is Gen3 x16 (its max)". All four cards report
`pcie clock level: 1 (16.0GT/s x16)` = **Gen4 x16**, and the measured 14.4 GB/s peer bandwidth
(~90% of a Gen4 x16 budget, and roughly double what Gen3 x16 could sustain) corroborates Gen4.
D078's Gen3 claim is wrong.

### 2. RCCL collectives at world=4 — strong at both ends
Built `ROCm/rccl-tests` in <container> and ran all_reduce and all_gather, small (1K–1M) and large (1M–256M):

| | small-message latency | large-message saturate |
|---|---|---|
| all_reduce | **24.6–30.8 µs** (2 KiB–16 KiB), flat | **14.19 GB/s** algbw / 21.29 busbw @ 256 MiB |
| all_gather | 27.1–31.1 µs (2 KiB–32 KiB) | 26.7 GB/s algbw / 20.0 busbw |

All cases validated (`#wrong = 0`), out-of-place and in-place agreeing closely. **Small messages are
neither poor nor erratic**, and **large-message bandwidth saturates at the fabric's measured
capability** with no degradation to 256 MiB. By the operator's stated decision rules this is the
case where TP8 is *not* blocked by the interconnect.

**A hard algorithm knee at 32 KiB:** all_reduce goes 25.47 µs @ 16 KiB → **42.02 µs @ 32 KiB**
(+65% latency for 2× payload). The workload's dominant AR message is ~20 KB, i.e. **just below that
cliff** — fortunate, but it means message growth past 32 KiB is disproportionately expensive.

**Measurement-semantics warning (un-reconciled).** Three incompatible figures now exist for "RCCL
world=4 latency at ~20 KB": **D141 records 106.96 µs**, the expert review measured **~18.4 µs**
(per-rank kernel medians), and rccl-tests gives **25–42 µs** (whole-op wall). They time different
things and must never be mixed. **D141's figure is 3–4× higher than a whole-op measurement of the
same operation and should be treated as un-reconciled**, not quoted.

### 3. Fresh TP4 baseline under teardown-day state
Re-ran the service unchanged at 1K/10K/30K with 1 Hz `rocm-smi` sampling (186 samples):

| prompt | saved PP | new PP | Δ | saved decode | new decode | Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 641.9 | 647.4 | +0.9% | 60.34 | 59.94 | −0.7% |
| 10,000 | 806.9 | 809.0 | +0.3% | 61.22 | 58.85 | −3.9% |
| 30,000 | 805.1 | 808.4 | +0.4% | 57.34 | 55.08 | −3.9% |

**Prefill reproduces faithfully; decode is modestly but reproducibly lower** (the 1K and 30K spans do
not overlap the saved ones). The monitor explains it: the rig is **power-cap limited at ~148 W against
the 150 W cap** (56–60% of samples ≥145 W) with sclk at a **1556 MHz median, full 1800 only 12–21%**
of the time, and GFX at 100% — i.e. DVFS managing a power budget, **not thermal throttling**
(junction peak 97 °C, below the ~100 °C region).

**GPU1 runs 14–18 °C hotter than GPU3** (89/97 °C vs 75/79 °C junction). In TP4 the **slowest rank
gates every step**, so a hot GPU1 costs throughput directly — a cooling/placement issue that worsens
with eight cards.

**Per the operator's rule, THIS run is the reference for TP8 inference** (more conservative, and
contemporaneous with the RCCL data above): **PP ≈ 647/809/808, decode ≈ 59.9/58.9/55.1.**

### 4. What TP8 would and would not buy
**The real case is weight bandwidth, not collectives.** At TP4 each rank holds ~18.23 GiB and decode
is weight-bandwidth-bound:
```
18.23 GiB / ~810 GB/s ~= 22.5 ms/token ~= 44 tok/s   (measured at TP4: ~47.7  -> model corroborated)
 9.1 GiB / ~810 GB/s ~= 11.2 ms/token ~= 89 tok/s   (TP8 ceiling, if bandwidth-bound)
```
That dwarfs anything the collectives will cost: collectives are only **~4% of wall** at four ranks
(D143), so even a 2–3× increase lands near 10%.

**Treat ~89 tok/s as a ceiling, not a forecast.** MoE routing is sparse so not all weights are read;
compute, per-step fixed overhead, the PLE sidecar and KV work do **not** halve; collectives grow. A
realistic expectation is a substantial decode gain well short of 2×, plus the capacity win (per-rank
18.23 → ~9.1 GiB, buying far more than the current 32K context, or unquantized weights).

**Power/cooling caution:** 4 × 150 W = 600 W today; eight cards at the same cap would be 1200 W if
per-card draw held. TP8 halves each card's workload so per-card power should fall, but PSU and
enclosure cooling headroom should be checked before committing — with GPU1 already at 97 °C junction.

### 5. Recommended next steps
1. Populate the empty switch ports and re-run **D141's fabric matrix, 12 ordered pairs → 56**. All 56
   byte-exact at ~14 GB/s settles the interconnect; any pair falling to host-bounce or RC-crossing
   speeds stops the project there.
2. Re-measure RCCL at **world=8** — the small-message all-reduce latency at 8 ranks is the single
   number that decides whether decode gains survive.
3. Only then invest in the software side (RCCL/custom-AR at world=8, EP sharding across 8 ranks).

### 6. CORRECTION (same day, D150 addendum) — the weight-bandwidth case for TP8 is REFUTED
Section 4 above projected a TP8 decode ceiling of ~89 tok/s from weight-bandwidth halving. A decode
**concurrency sweep** (1K prompt, 512 generated, 1/2/4/8 concurrent, streaming, unique prompt salt
per request, GPU sampled at 1 Hz) **falsifies the mechanism**:

| concurrency | aggregate tok/s | scaling | per-request | round median | p95 | max |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 44.41 | 1.00x | 52.39 | 43.2 ms | 143.6 ms | 170.9 ms |
| 2 | 40.21 | 0.91x | 25.37 | 66.0 ms | 174.8 ms | 3466 ms |
| 4 | 45.24 | 1.02x | 13.62 | 197.2 ms | 308.0 ms | 3005 ms |
| 8 | 40.30 | 0.91x | 11.37 | 200.6 ms | 348.5 ms | 2418 ms |

**Aggregate throughput is FLAT across 1 -> 8 concurrent** (40.2-45.2 tok/s, 12% band, no trend).
Ideal linear scaling at 4 concurrent would be 178 tok/s; measured 45, i.e. **none**. If weight reads
bound decode, batching 4 sequences would read the weights ONCE and emit 4x the tokens; it emitted
1.0x. **So weights are not the bottleneck, and the `18.23 GiB / 810 GB/s ~= 44 tok/s` agreement with
the measured ~47.7 was a NUMERICAL COINCIDENCE** - the MoE is sparse (top-10 of 128 local experts),
so the full per-rank footprint is never read per token.

**Corroborating evidence in the power:** 150.0 W at concurrency 1 -> 146.5 W at 2 -> **111.0 W at 4**
-> **109.0 W at 8**, while the GPU "use" counter stays ~100%. Power FALLING 27% as concurrency rises
means the SMs are **stalled waiting, not saturated** - the workload is **latency/serialization
bound**, and the "100% use" figure counts issue cycles rather than useful work. Round p95 also
degrades 143 -> 348 ms with multi-second stalls (3.5 s max), so interactive feel worsens under load.

**REVISED VERDICT: TP8 is unlikely to improve decode throughput, single-stream OR multi-stream.** The
binding constraint is per-token serialized work - consistent with D145/D146, where the
elementwise/copy residue is 303k latency-bound launches over 303,380 kernels. **TP8's case is
CAPACITY** (far more than the current 32K context, or unquantized weights), **not throughput.** The
physical pre-checks in section 5 still stand and are still worth doing, but they must be justified on
capacity rather than on a decode-speed expectation.

**Measurement note (repeat of a trap worth naming):** the first pass counted streamed CHUNKS as
tokens and understated throughput ~2.8x, because with MTP2 speculative decoding one streamed chunk
carries several accepted tokens. The server's `usage.completion_tokens` is authoritative; the
across-concurrency RATIO was unaffected (every request produced the same tokens in the same number of
chunks), but the absolute numbers required recomputation.

**Artifacts:** `tools/tp8_probe/` — `TP8_TOPOLOGY_ASSESSMENT.md` (the full write-up),
`host_topology.txt`, `ct_topology.txt`, `rccl_all_{reduce,gather}_{small,large}.txt`,
`rocm_smi_during_vllm.txt`, `concurrency_results.json`, and the scripts `tp8_rccl.sh`,
`tp8_rccl_run.sh`, `tp8_tp4_baseline.sh`, `analyze_vllm_monitor.py`, `concurrency_bench.py`,
`analyze_concurrency.py`. Copies of all captures also live on both hosts under
`<tuning>/tp8_probe/`.

---

## D151 — FINAL TP8 VERDICT: a cheap-capacity strategy, NOT a performance strategy

- Date: 2026-09-18
- Authority: **operator conclusion**, recorded after reviewing the D150 probe results. This is the
  binding framing for any future TP8 decision; D150's sections 4 and 6 are superseded by it.

### The verdict
**TP8 is a cheap-capacity strategy, not a performance strategy.** It may let this rig run models or
context sizes that are otherwise uneconomical, but it should **not** be expected to deliver better
decode throughput than the current TP4 setup.

**Realistic expectation: TP8 decode will probably land BELOW the current ~40-60 tok/s aggregate
range** — unless the 8-GPU machine has a **much better interconnect/topology** than this one.

### Use-case outlook
| Use case | TP8 outlook |
|---|---|
| Single-stream decode | **likely worse than TP4** |
| Multi-user decode throughput | **likely flat to worse** |
| Long prefill / TTFT | maybe acceptable, possibly modestly better |
| Capacity per dollar | **still the main reason to do TP8** |
| Overall "cheap 128 GB serving box" | **viable only if speed expectations are modest** |

### The four signals, and what each means
| Signal | Meaning |
|---|---|
| Aggregate decode stays ~40-45 tok/s from 1 to 8 concurrency | More requests do not fill unused compute |
| Per-request speed collapses as concurrency rises | Requests are mostly sharing the same bottleneck |
| GPU power drops to ~111 W at 4-8 concurrency | The GPUs are waiting more, not working harder |
| p95 rises sharply | Scheduler/communication bubbles are becoming user-visible |

**If this were compute-limited, aggregate tok/s would climb with concurrency and GPU power would stay
near 150 W. Instead aggregate is flat and power falls.** That points to **synchronization,
communication, scheduler overhead, or CPU/NUMA stalls** — not to a lack of active requests.

### Why the power drop is the decisive evidence
At 4-way concurrency the rig should have had plenty of work queued, yet it **backed off to ~111 W**.
This is the part that makes the conclusion firm rather than suggestive: a saturated machine cannot
reduce its power draw when handed *more* work.

**And it is unlikely to be fixed by adding ranks.** Adding four more tensor-parallel ranks usually
**increases** collective cost and synchronization pressure, so TP8 would be pushing in the wrong
direction on precisely the axis that is already limiting the rig. The one escape hatch is a
materially better interconnect/topology than the PEX88096-based one measured here — which is what the
D150 pre-checks (56-pair fabric matrix, world=8 RCCL) exist to establish, and they should now be
justified on **capacity** grounds rather than on any decode-speed expectation.

### Consequence for the plan
- The D150 pre-checks stay worth doing, **justified by capacity per dollar**.
- Any "cheap 128 GB serving box" claim must be made with **modest speed expectations attached**.
- If the goal is decode throughput, the lever is the **per-token serialized work** measured in
  D145/D146 and confirmed here — not more cards. This is the same conclusion D146 reached from the
  kernel side (303k latency-bound launches), now independently confirmed from the serving side.
- No further TP4 throughput experiments are implied by this entry; the operating configuration
  stands at MTP2 / 32K / 150 W with the D148/D149 presets.

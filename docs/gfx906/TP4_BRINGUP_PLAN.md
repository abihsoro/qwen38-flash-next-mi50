# MI50/gfx906 TP4 bring-up plan from the V620 campaign

Date: 2026-09-16

Source reviewed:
- `qwen38-flash-next-v620/v620-tuning/{STATE,RESULTS,MEASUREMENTS,ACCEPTANCE,DEAD_ENDS,RETAINED_CONFIG}.md`
- `qwen38-flash-next-v620/docs/{04-multi-gpu,05-software-stack,08-vllm,09-gotchas,10-obscurities,11-methodology}.md`
- `qwen38-flash-next-v620/scratch/{v620_RESULTS,v620_PROFILE-NAVI21,v620-BRINGUP-<source-host>}.md`
- MI50 current state through D139 in this project.

## Current MI50 state

D139 cleared the hardware gate. <source-host> has four V420-flashed Vega20/MI50-class cards on the
host/LXC path:

- `<pci>`, `<pci>`, `<pci>`, `<pci>`
- all `<device-id>`, subsystem `<device-id>`, BIOS P/N `113-D1640200-043`
- 32 GB BAR0, Gen4 x16, `amdgpu` bound
- LXC300 sees four `gfx906:sramecc+:xnack-` agents / four torch devices
- `clear-pex-acs` is now GPU-ID agnostic for AMD endpoints under the PEX880xx path

D138 remains the diagnosis: TP2 was offload-stall dominant. The next valid evidence must come
from a fully resident TP4 run, not from the previous TP2+offload results.

## Transfer from V620

### Transfer directly

1. **Campaign discipline.** Freeze a baseline, change one thing, run serial measurements, require
   exact or predeclared correctness gates, record negative results, and retain rollback.
2. **No performance claim from profiler-only or microbench-only evidence.** Isolated timings choose
   candidates; server harness results decide.
3. **Exact-output culture.** For numerics-lossless changes, compare complete greedy outputs and
   hashes. For MI50 TP4, G4 is first-64-token match / golden-like behavior with documented TP fp16
   caveats.
4. **Graph replay tests.** Any custom op or kernel used under graphs must pass eager and graph
   replay checks before serving.
5. **Transport-first ordering.** V620's 100 TPS state depended on correct PEX ACS and P2P routing;
   stale ACS dropped TG128 from ~100 to ~62. MI50 must validate the 4-card fabric before any TP4
   performance run is trusted.
6. **Collective correctness rules.** Use uncached peer staging/flags, fixed-rank-order fp32
   reductions, device-side sequence counters, bounded spin/abort paths, and no graph-dropped stream
   waits.
7. **Small gains need reverse A/B.** Under 1% is noise until a control/reverse run proves it.

### Transfer as process, not code

1. **V620 RDNA2 kernels are not gfx906 kernels.** Wave32/Navi21 tiling, `v_dot` assumptions, and
   V620-specific shared objects should not be copied directly. Re-derive the wave64/gfx906 shape.
2. **V620 final target was TP4+EP4, MTP3, 100 TPS TG128.** MI50's first target is TP4 fully
   resident MTP0. Only after MTP0 is correct and profiled should MTP3 be measured.
3. **V620 E19 draft tile is a later-stage idea.** It is relevant only after the MI50 MTP path is
   enabled and measured. Do not spend first-pass time on draft-head tiling.
4. **V620 GDN copy removal is a candidate only if MI50 profile shows it.** D136 says QSA/GDN was
   negligible in the MI50 TP2 profile. Revisit only if TP4 contradicts that.
5. **V620 dense int8 shadows are lower priority for MI50.** D133-D136 showed gfx906 fp16 dense
   already beats the V620 int8 yardstick on measured resident terms. Dense int8 can remain a later
   experiment, not the bring-up gate.

### Reject / watch

1. Do not remove PLE waits or cross-GPU ordering for speed.
2. Do not trust `gpu_busy_percent`; it is dead/unavailable on the flashed V420 path.
3. Do not compare profiled and unprofiled wall times.
4. Do not let compile cache or stale graphs hide code changes.
5. Do not tune from legacy short-output prompts or early-EOS runs.

## Phase 0 - freeze the MI50 baseline state

Goal: make the environment reproducible before the first TP4 run.

Actions:
- Snapshot <source-host> hardware: PCI tree, link speed/width, BARs, IOMMU groups, KFD node mapping,
  `amdvbflash -i`, ACS post-clear state, power caps, clocks, temperatures.
- Snapshot LXC300: config, mounts, `/dev/kfd`, `/dev/dri`, torch/triton versions, ROCm env,
  visible GPUs, model paths, vLLM branch/commit, project commit.
- Kill stale `VLLM::` processes before every serve run; verify no `/dev/kfd`/render-node users.
- Record exact launch env and arguments for every attempt.
- Define fixed prompts now:
  - MI50 correctness prompt: existing golden/G4 first-64-token prompt.
  - MI50 throughput prompt: adapt the V620 TG128 technical prompt only after the TP4 model is
    correct. Require actual 128 generated tokens if using TG128.

Artifacts:
- `results/tp4_baseline/env_snapshot_*.txt`
- `results/tp4_baseline/launch_*.json`
- appended `DECISIONS.md` entry when the frozen baseline is created.

Exit gate:
- Environment can be reconstructed from saved state.
- No serving or benchmark process is already holding GPU memory.

## Phase 1 - M0/M0.7 four-card fabric validation

Goal: prove the physical fabric before any TP4 number.

Actions:
- Run 4-card peer matrix inside LXC300:
  - `hipDeviceCanAccessPeer` for all ordered pairs
  - `hipDeviceEnablePeerAccess`, clearing sticky `hipErrorPeerAccessAlreadyEnabled`
  - 256 MB `hipMemcpyPeer` bandwidth, all 12 ordered pairs
  - 20 KB peer-copy latency, all 12 ordered pairs
  - byte-exact payload checks
- Verify KFD topology:
  - four GPU nodes
  - each GPU `p2p=3`
  - render node to PCI mapping recorded
- Run peer-BAR ordering test:
  - uncached staging / peer writes
  - owner-local polling
  - repeated stress loop, byte-identity checks
- Run an ACS negative-control only if it can be done safely:
  - capture current cleared state
  - demonstrate the checker detects bad redirect bits or a disabled clear service
  - restore and verify service success

Acceptance:
- All 12 peer pairs pass correctness.
- Bandwidth/latency table is recorded, not assumed from the old 2-card pair.
- ACS clear is verified on the current GPU-path bridges.
- No GPU wedge, reset, or bus-loss events in dmesg.

If fail:
- Stop before TP4. Fix ACS/link/KFD first.

## Phase 2 - M1/M2 collective validation

Goal: establish the world=4 collective floor and choose RCCL vs custom AR for the first TP4 run.

Actions:
- Measure RCCL world=4 all-reduce curve in LXC300:
  - 5 KB, 20 KB, 80 KB, and any exact dominant hidden-state sizes from the model
  - enough reps for median/range
  - no concurrent serving
- Validate custom wave64 one-shot AR at world=4:
  - uncached staging and flags
  - fixed rank-order fp32 reduction
  - device-side sequence counter so graph replay is legal
  - bounded spin with failure path
  - graph replay correctness
  - two identical greedy/determinism tests if wired into vLLM
- Resolve or route around the known `hipIpcOpenMemHandle` asymmetry from D117.
- Compare in-server if possible before sizing the optimization value. V620's out-of-server RCCL
  measurements overstated the server cost.

Acceptance:
- For G5, 20 KB dominant message <= 34 us target if custom AR is selected.
- If custom AR misses or is not wired, record RCCL as the floor and boot TP4 with custom AR off.
- Correctness/determinism outranks speed.

If fail:
- Use RCCL / disable custom AR for first TP4 correctness, then return to AR after M3.

## Phase 3 - M3 first TP4 fully resident serve

Goal: get the full model serving on four MI50s with no weight offload.

Initial conservative config:
- LXC300 host path
- full AWQ backbone + PLE sidecar
- TP4 + EP
- MTP=0
- dtype float16
- graphs-no-compile; compile remains off on gfx906 because D060 found an inductor miscompile
- no CPU/UVA weight offload
- start with the known-safe collective choice from Phase 2

Boot ladder:
1. 4-card environment smoke: torch sees 4 devices; trivial all-reduce/copy tests pass.
2. Small model or layer-limited surrogate if available: prove launch path and graph capture.
3. Full 48-layer model load with no offload; record VRAM per rank after warmup.
4. Correctness run:
   - deterministic greedy
   - G4 first-64-token match or documented golden-like TP fp16 caveat
   - full output text and hash saved
5. Initial timing run:
   - serial, no profiler, no side GPU jobs
   - at least 3 repeats
   - actual token counts from API/harness

Acceptance:
- No offload in logs/config.
- No early EOS used as a throughput win.
- No PLE duplicate/stale warnings.
- G4 passes or produces a documented, bounded TP fp16 caveat consistent with prior TP2 behavior.
- First performance number is labeled baseline only.

If fail:
- Do not optimize. Classify failure first: load/OOM, graph, collective, PLE, numerical, or runtime.

## Phase 4 - M4 profile the TP4 baseline

Goal: expose the next bottleneck after offload removal.

Measurements:
- Wall/token, no profiler, serial repeats.
- Torch/vLLM profiler trace for attribution, explicitly not used as throughput evidence.
- Kernel family aggregation:
  - dense GEMV / lm_head
  - fused glue / HC / shared expert
  - MoE int4
  - QSA/GDN
  - collectives / waits / rank skew
  - graph boundaries / torch elementwise residue
  - PLE CPU phases and waits
  - scheduler/sampling
- PLE timestamps if feasible:
  - request issued
  - CPU gather start/end
  - result visible
  - model thread resumes
- Components must sum plausibly to measured wall/token. If they do not, attribution is not
  accepted.

Output:
- A TP4 ledger analogous to D137/D138 but without the TP2 offload confounder.
- A ranked candidate list with estimated ceiling.

Acceptance:
- The next optimization target is chosen from measured TP4 profile data, not from the V620
  ladder by analogy.

## Phase 5 - optimization ladder after TP4 baseline

Only start this after M3/M4.

Priority order:

1. **Transport / collectives / skew**
   - If AR or rank skew is large, finish world=4 custom AR wiring, reduce collective count, or
     rebalance EP.
   - V620 showed a 33 us custom AR can matter at TP4, but only after in-server sizing.

2. **Graph / dispatch residue**
   - If torch elementwise or graph gaps are material, use frame attribution and fuse the largest
     sites.
   - Keep compile off unless the gfx906 inductor miscompile is bisected and fixed.

3. **PLE**
   - If PLE is now visible after offload removal, instrument/fix waits, prefaulting, and pinned
     result buffers.
   - Never remove correctness waits. V620's false fast path came from a missing wait.

4. **MoE / glue kernels**
   - Re-test V620-like scheduling ideas only if the MI50 profile shows a ceiling:
     - early skip of non-local experts before staging
     - block/wave geometry sweeps in wave64 form
     - local-only staging
   - Require bit-exact eager + graph replay before serving.

5. **Dense int8**
   - Defer unless TP4 profile makes dense the dominant resident term and fp16 no longer looks
     good enough.
   - D133-D136 weakened this as a first lever.

6. **QSA/GDN copy removal**
   - Defer unless TP4 profile says QSA/GDN or copy/cat paths are non-negligible.
   - V620 E08 is a useful pattern, not a proof that MI50 needs it.

7. **MTP**
   - After MTP0 TP4 is correct and profiled, test MTP3 with the same exact-output/actual-token
     discipline as V620.
   - MTP4 is not presumed safe: V620 changed target text and was rejected.

Per-experiment format:

```text
ID:
Question:
Baseline:
Single change:
Correctness gate:
Measurement protocol:
Result:
Verdict:
Artifacts:
Rollback:
```

Adoption rules:
- >2% clean gain with exact outputs: candidate.
- 0.5-2%: reverse A/B required.
- <0.5%: reject unless it removes risk or enables a larger later change.
- Any deterministic-output change from a supposedly lossless change: reject or re-baseline only
  with operator sign-off.

## Immediate next runbook

1. Create `results/tp4_bringup/`.
2. Snapshot host/LXC state.
3. Run 4-card M0/M0.7 peer matrix and ACS verification.
4. Append D140 with the fabric matrix result.
5. Run world=4 RCCL/custom AR validation.
6. Append D141 with the chosen first-TP4 collective path.
7. Boot TP4 MTP0 fully resident.
8. Append D142 with correctness, no-offload proof, and baseline timing.
9. Profile TP4 and append D143 with the ranked bottleneck ledger.

The first MI50 TP4 number should be boring and honest. The V620 campaign's biggest transferable
lesson is that boring honest baselines are what make the later cleverness real.

# Qwen3.8-Flash-Next on 4× MI50 (gfx906)

**Qwen3.8-Flash-Next** — a 176 B hybrid model (~6 B active, plus a 51 B-row int4 n-gram sidecar
served from CPU) — running **fully resident** on four 32 GB **MI50** cards (Vega 20 / gfx906) behind
a Broadcom/LSI PEX88096 Gen4 switch, tensor-parallel × 4 with expert parallelism.

The port took decode from **2.54 tok/s** (TP2 + weight offload) to **~60 tok/s with a 32K context**,
and prefill to **~640–810 tok/s**, without touching the model's weights. Every step was gated the
same way: byte-identical greedy output and an A/B/A′ protocol with a deterministic hash.

---

## How to think about this — the approach, before the numbers

For a smart reader who hasn't done this before: the whole campaign is one idea repeated — **find the
bottleneck by measuring, attack that specific bound, and reject anything the measurement doesn't
support.** Everything below is an instance of that loop. The knobs we turned divide cleanly into two
piles: *general* levers that apply to any LLM inference stack, and *hardware-specific* levers that
exist because this is four MI50 cards, not eight H100s.

### 1. An LLM serve is two different jobs, not one

The first thing to internalize is that "inference" is two phases with opposite shapes:

- **Prefill** (prompt → first token) is a **big parallel batch**. All prompt tokens are known up
  front, so it is a large matrix-multiply: throughput-bound, bandwidth-bound, and it *loves* batching.
- **Decode** (token → next token) is a **serial loop**. Each step produces one token, and the next
  step can't start until the previous one finishes. Each step is small and *latency-sensitive*, and it
  re-touches the model's weights every step.

The two have different bottlenecks, so they need different treatments. Most of the confusion in this
space comes from conflating them — a change that helps prefill often does nothing for decode, and
vice versa. We kept them separate from the start, and it paid off repeatedly.

### 2. The general toolbox (applies anywhere)

Five axes cover nearly every lever in LLM serving:

1. **Fit the weights.** A 176 B model doesn't fit one 32 GB card. You split it across cards (tensor
   parallelism), shrink it (quantization), or spill it to CPU/RAM (offload). All three have a cost.
2. **Reduce the work per token.** Fuse kernels, remove intermediate copies, pick better GEMM tiles,
   and use **graphs** to eliminate per-kernel launch overhead.
3. **Break the serial dependency.** Speculative decoding: have a cheap draft model *guess* several
   tokens, then verify them in one parallel pass. Correct, but only pays if the guesser is right
   often enough.
4. **Reduce the communication.** In tensor parallelism, every layer ends with an all-reduce, and that
   sits on the critical path of every decode step.
5. **Match the geometry to the shape.** A GEMM tile that's optimal for a batch of 512 is often wrong
   for a batch of 1 — this is the whole reason "autotuning" exists.

### 3. The hardware lens — why *this* box is a specific box

The four cards here are **MI50 / Vega 20 (gfx906)**: they have **no NVLink, no Infinity Fabric, and
no FP8**. Inter-GPU traffic goes over **PCIe Gen4 through a switch**. Three consequences shaped every
decision:

- **The collective is the tax on parallelism.** On NVLink hardware the all-reduce is nearly free; here
  it's a real cost, so a *custom, hardware-matched* all-reduce is worth writing.
- **Quantization is int4/int8, not FP8.** The giant n-gram sidecar is int4; the active weights stay
  fp16.
- **The stock kernels don't know this chip.** vLLM ships NVIDIA-tuned paths; a wave64-optimized fp16
  GEMV (`skinny GEMM`) is 2.83× faster than the generic fallback on gfx906.

The meta-point: **optimization is relative to the silicon.** A lever that's huge on one chip is a
no-op on another, and the only way to know which is to measure *on your hardware*.

### 4. The discipline (why every number is trustworthy)

Every candidate change followed the same gate, and the gate was calibrated to the rig's own noise:

1. **Isolated correctness first** — a microbench with a CPU reference and a tolerance, so a
   faster-but-wrong kernel can never sneak in.
2. **In-serve A/B/A′** — control / candidate / control, three gates each, because whole-serve drift
   is ~1% and a single control isn't a baseline.
3. **Byte-identical output** — greedy output must hash identically across arms; a "lossless" change
   that changes output is rejected out of hand.
4. **Adopt only if >2%** vs both controls — anything less is noise by construction.

This is the unglamorous part that makes the rest *mean* something. It's also why the campaign is full
of "rejected" results: rejecting cleanly is a feature, not a failure.

### 5. The knobs we actually turned, and what each one taught

| knob | general or hardware-specific? | what it was testing |
|---|---|---|
| TP4 + expert parallelism | general (fit) | the only way to fit 176 B on 4 × 32 GB |
| weight offload → fully resident | general, with a hardware lesson | offload "works" but is ~20× slower at decode; residency is the goal |
| custom wave64 one-shot all-reduce | **hardware-specific** | no NVLink → PCIe AR; a matched kernel beat RCCL by ~4% in-serve |
| skinny fp16 GEMV (gfx906) | **hardware-specific** | a ported kernel for the chip; 2.83× over the generic path |
| MoE tile autotuning | general, with a shape caveat | tiles transfer *per shape*; decode-tuned configs were −47.7%, prefill-tuned ones won |
| MTP (speculative decoding) | general | breaking the serial loop; acceptance is model-dependent (K=2 beat K=3) |
| graphs, compile-off | general + hardware | removes launch overhead; compile left off as a port-risk call |
| power cap (150 W/card) | general + hardware | throughput per watt; decode turned out CPU/comm-bound, not power-bound |
| concurrency / batching | general | batching amortizes weights *if* compute-bound; here it didn't — which was the diagnostic |

The through-line: we **measured which bound was live** (offload first, then weight bandwidth,
then — via the concurrency sweep — *latency/serialization*), and only spent effort where the
measurement said there was headroom. That's the whole methodology in one sentence.

---

## Headline numbers (this rig, measured)

| metric | result |
|---|---|
| **Decode, MTP2**, 1K / 10K / 30K prompt | **60.34 / 61.22 / 57.34 tok/s** (medians of 3) |
| Decode, MTP2 acceptance | 2.49–2.75 accepted tokens/step |
| **Prefill PP**, 1K / 10K / 30K | **641.9 / 806.9 / 805.1 tok/s** |
| Decode, MTP0 (gate, whole request) | ~47.7 tok/s — *includes prompt time*; corrected decode reference ~55–56 tok/s |
| Prefill, 128 / 512 / 1536 tokens | 342.7 / 657.7 / 851.1 tok/s (vs reference 195.8 / 453.6 / 565.0) |
| Peer-to-peer (fabric matrix) | **14.40–14.44 GB/s**, 12/12 ordered pairs byte-exact |
| RCCL all-reduce 20 KB / 256 MiB | 25–42 µs / **14.19 GB/s** algbw |
| Custom one-shot all-reduce, in-server | ~+4% vs RCCL (byte-identical) |
| Power under decode | ~148 W/card at **150 W caps** |
| Aggregate decode, 1→8 concurrent | **flat ~40–45 tok/s** — the workload is latency-bound, not compute-bound |

Numbers follow the measurement contract in [`docs/gfx906/README.md`](docs/gfx906/README.md): exact
token lengths, unique cache salts (zero cached prompt tokens), two warmups then three scored
requests, greedy decoding.

## The campaign in one ladder

| step | change | result | verdict |
|---|---|---|---|
| D137 | TP2 + UVA weight offload (baseline) | 2.54 tok/s | offload-bound |
| D142 | TP4 **fully resident** (no offload), eager | 10.95 tok/s | 4.3× |
| D142 | TP4 graphs-no-compile (MTP=0) | ~47.7 tok/s gate | the resident win |
| D142a | custom wave64 one-shot all-reduce live | +4% in-server | kept |
| D148 | **prefill MoE tiles** + mask-aware top-10 router + 2048-token chunks | PP **+45–75%** at 128–1536 tokens | kept |
| D149 | **MTP2** (two speculative tokens) | decode **60.3 tok/s** @ 1K | **retained** |

Rejected with data: MoE decode-side autotuning (−47.7%), `--max-num-batched-tokens` 2048 (flat),
dense-GEMV geometry (~2.6% wall ceiling), split-K (~0.6%), the copyBuffer/contiguity hypothesis
(exact null), and MTP3 (lossless but ≤5%, prefill −12%). Details in
[`docs/gfx906/campaign/DEAD_ENDS.md`](docs/gfx906/campaign/DEAD_ENDS.md) and
[`MEASUREMENTS.md`](docs/gfx906/campaign/MEASUREMENTS.md).

## Notable findings

- **Decode on this stack is latency/serialization-bound, not compute-bound and not
  weight-bandwidth-bound.** The decisive evidence is the concurrency sweep: aggregate throughput is
  flat from 1 to 8 concurrent requests while GPU power *falls* (150 → 109 W) — a saturated machine
  cannot reduce its draw when handed more work. That points at synchronization, communication,
  scheduler overhead, or CPU/NUMA stalls. It also means TP8 is a **capacity** play (more context /
  unquantized weights), not a throughput play — see
  [`docs/gfx906/campaign/RESULTS.md`](docs/gfx906/campaign/RESULTS.md) and the TP8 probe in
  [`data/tp8_probe/`](data/tp8_probe/).
- **The headline "47.7 tok/s" timed the whole HTTP request, so it included prompt time.** The
  corrected decode reference is ~55–56 tok/s (D148). Any future claim must state which of the two it
  is measuring.
- **The links are Gen4 ×16, not Gen3** — a reading of `16.0 GT/s ×16` plus 14.4 GB/s peer bandwidth
  corrects the earlier "MI50 is Gen3 ×16" note (D150).
- **All four cards sit on one switch, one root complex, one NUMA node**, with uniform weight/hops for
  every pair and spare downstream ports for four more cards. The extra switch stage that half the
  cards already traverse costs nothing measurable.
- **GPU1 runs 14–18 °C hotter than GPU3** (97 vs 79 °C junction), and in TP the slowest rank gates
  every step — a cooling/placement issue worth remembering if decode speed is ever revisited.

## Repo map

```
README.md ................... this page
DECISIONS.md ................ append-only working log (D001–D151) — the authoritative record
PROGRESS.md ................. generated tracker (config/tasks.json is the machine-readable source)
TRAPS.md .................... 21 hard-won gotchas (hardware/stack/methodology)
docs/gfx906/
  README.md ................. measurement contract (facts-that-changed, config, yardsticks, rules)
  campaign/ ................. the archived campaign: STATE / MEASUREMENTS / RETAINED_CONFIG /
                             ACCEPTANCE / DEAD_ENDS / RESULTS + accepted-sha256.json
  worklog/BRINGUP.md ........ narrative bring-up
  BASELINE.md · PORT-MAP.md · TP4_BRINGUP_PLAN.md · …   (planning docs)
tools/ ...................... m0/ (bring-up + optimisation scripts) · tp8_probe/ · review_20260918/
data/ ....................... raw evidence (tp8_probe captures)
results/tp4_bringup/ ........ ingested measurement rows (fabric, RCCL, custom-AR, TP4 gate)
config/ · harness/ .......... task registry + the PROGRESS generator
```

## Notes & caveats

- MTP comparisons need the acceptance rate — MTP0 is the clean yardstick; MTP2's speed varies with
  per-prompt draft acceptance (2.49–2.75 tok/step here).
- Long-context continuations at temperature zero are **not** deterministic on this rig (the 10K/30K
  runs varied across reps); determinism claims are limited to the fixed prompt and short context.
- This is a **redacted public mirror**. Host identifiers (IPs, hostnames, and internal filesystem
  paths) have been replaced with `<…>` placeholders. See `docs/gfx906/campaign/README.md` for provenance.
- The port source tree (`vllm/`) is a separate repo and is not committed here.

## Housekeeping

- Measurement contract & yardsticks: [`docs/gfx906/README.md`](docs/gfx906/README.md)
- Campaign archive: [`docs/gfx906/campaign/`](docs/gfx906/campaign/README.md) ·
  bring-up: [`docs/gfx906/worklog/BRINGUP.md`](docs/gfx906/worklog/BRINGUP.md)
- Working log: [`DECISIONS.md`](DECISIONS.md) · generated tracker: [`PROGRESS.md`](PROGRESS.md)
- Raw evidence: [`data/`](data/) · tooling: [`tools/`](tools/)

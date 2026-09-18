# TP8 topology probe — 2026-09-18

**Question asked:** would TP8 span sockets, root complexes or weak bridges? If so, single-stream
decode is very likely to flatten or regress.

**Answer: no — the current four cards already share one switch, one root complex and one NUMA node,
and that switch has physical headroom for four more. TP8 is fabric-viable provided the new cards
attach to the *same* PEX88096.**

Raw captures: [`host_topology.txt`](host_topology.txt) (<host>, <host-ip>) and
[`ct_topology.txt`](ct_topology.txt) (<container>, <container-ip>). Also left on both systems at
`<tuning>/tp8_probe/`.

## 1. One switch, one root complex, one NUMA node

The PCIe tree on the host (`lspci -tv`) shows the whole GPU complex under a single root complex:

```
+-[<bus>]-+-00.0  Starship/Matisse Root Complex            <- ONE root complex
 |           +-03.1-[<bus-range>]----00.0-[<bus-range>]--+               <- ONE GPP bridge up to the switch
 |           |                               +-00.0-[<bus-range>]----...----00.0  Vega 20    <- GPU0 @ a7
 |           |                               +-04.0-[<bus-range>]----...----00.0  Vega 20    <- GPU1 @ ac
 |           |                               +-08.0-[<bus-range>]----00.0-[<bus-range>]--+-00.0-[<bus-range>]----...--00.0 Vega 20  <- GPU2 @ b2
 |           |                               |                               \-10.0-[<bus-range>]----...--00.0 Vega 20  <- GPU3 @ b5
 |           |                               +-0c.0-[<bus-range>]----00.0-[<bus-range>]--+-14.0-[<bus>]--   <- UNUSED
 |           |                               |                               \-15.0-[<bus>]--   <- UNUSED
 |           |                               \-1c.0-[<bus>]----00.0  Broadcom/LSI PEX880xx PCIe Gen 4 Switch  <- UNUSED
```

`rocm-smi --showtopo` on <container> confirms it is uniform **and** single-socket:

```
Weight between two GPUs      Hops between two GPUs       Link Type
     GPU0 GPU1 GPU2 GPU3          GPU0 GPU1 GPU2 GPU3         GPU0 GPU1 GPU2 GPU3
GPU0    0   40   40   40     GPU0    0    2    2    2    GPU0    0 PCIE PCIE PCIE
GPU1   40    0   40   40     GPU1    2    0    2    2    GPU1 PCIE    0 PCIE PCIE
GPU2   40   40    0   40     GPU2    2    2    0    2    GPU2 PCIE PCIE    0 PCIE
GPU3   40   40   40    0     GPU3    2    2    2    0    GPU3 PCIE PCIE PCIE    0

Numa Nodes: GPU[0..3] Numa Node: 1, Numa Affinity: 1   (all four)
pcie clock level: 1 (16.0GT/s x16)                     (all four)
```

Every pair is identical: **weight 40, 2 hops, PCIE**. There is no XGMI / Infinity Fabric between
these cards — all inter-GPU traffic is PCIe through the switch — which is expected for MI50 on a
non-AMD-fabric platform.

This agrees with the independent measurement in D141: all **12 ordered peer pairs byte-exact at
14.40–14.44 GB/s (IQR ≤ 0.014)**, latency 16.96–18.13 µs @ 20 KB.

## 2. The cards are NOT all at the same switch depth — and it doesn't matter

GPU0 and GPU1 hang **directly** off the PEX88096 (ports 00.0 and 04.0). GPU2 and GPU3 hang off a
**second-level** switch (`ae`) reached via port `<pci>`. So half the traffic nominally crosses one
extra switch stage — yet rocm-smi reports uniform weights/hops for all pairs and the measured
bandwidth is uniform to within 0.3%. **The extra level costs nothing measurable**, which is a
useful precedent: adding another level for new cards is not automatically harmful.

## 3. There IS headroom for four more cards on the same fabric

Two downstream ports of the main switch are unpopulated:

- `<pci> → [<bus-range>]` — a sub-switch at `b7` with **two empty downstream ports** (`[<bus>]`, `[<bus>]`)
- `<pci> → [<bus>]` — **a whole second PEX880xx Gen4 switch with nothing behind it**

So additional cards can be attached **without crossing root complexes or NUMA nodes**, which is the
condition that makes TP8 fabric-viable.

## 4. Links are Gen4 x16 — correcting a stale note

All four report `16.0GT/s x16` (Gen4), not Gen3. D078's claim that "MI50 is Gen3 x16 (its max)" is
contradicted both by the link status and by the measured peer bandwidth: 14.4 GB/s is ~90% of a
Gen4 x16 lane budget and roughly double what Gen3 x16 could sustain.

## 5. NUMA and host-side placement

Two nodes, ~516 GB each, 48 cores exposed per node (`0-47,96-143` and `48-95,144-191`). All four
GPUs are on **node 1**, and the selected preset correctly pins the server with
`numactl --cpunodebind=1 --preferred=1`. The host itself has no ROCm stack installed
(`rocm-smi`/`numactl` are absent there — expected, since the GPUs are passed through to <container>), so
the host capture is `lspci` only.

## What this means for TP8

**Upside is real but it is not single-stream decode.** TP8 would halve the per-rank weight footprint
(18.23 GiB → ~9.1 GiB), buying context and quantization headroom. It does **not** obviously buy
tokens/s: the measured collectives are ~21% of kernel time but only **~4% of wall** at four ranks
because the ranks overlap (D143), so the sync cost has little room to hurt — but equally little room
to help. TP8's wins are capacity (32K → far more) or running unquantized, not TG.

**The shared resource to watch.** Every added card and every added switch level shares the
PEX88096's internal crossbar, and the switch's upstream to the root complex is **one** GPP bridge.
Inter-GPU allreduce stays switch-local and never traverses that upstream link — but with eight cards
each capable of Gen4 x16 the aggregate concurrent demand on the switch fabric roughly doubles.

**Cheap decisive test before committing:** populate the empty ports (`[<bus>]`/`[<bus>]` and/or behind
`[<bus>]`) and re-run the D141 fabric matrix — 12 ordered pairs becomes **56**. If all 56 pairs are
still byte-exact at ~14 GB/s, TP8's interconnect is settled and the remaining question is purely
software (RCCL/custom-AR at world=8, EP sharding across 8 ranks). If any pair falls to host-bounce
or RC-crossing speeds, stop there.

---

# RCCL collective sweep (2026-09-18) — the direct TP8 predictor

Built `ROCm/rccl-tests` in <container> and ran the four requested sweeps at world=4.
Raw output: [`rccl_all_reduce_small.txt`](rccl_all_reduce_small.txt),
[`rccl_all_reduce_large.txt`](rccl_all_reduce_large.txt),
[`rccl_all_gather_small.txt`](rccl_all_gather_small.txt),
[`rccl_all_gather_large.txt`](rccl_all_gather_large.txt).

**Two container quirks, for the record:** `git` is not installed (source came from the GitHub
tarball), and `hipify-perl` is only at `/opt/rocm/core-7.14/bin/hipify-perl` — rccl-tests hipifies
at build time, so that directory must be on PATH. The freshly linked binaries also carry **no
rpath**, so `LD_LIBRARY_PATH=/opt/rocm/lib` is required at run time. (Note: that last point is the
*legitimate* use of that variable — TRAPS #11/#17 corrected a different claim, that it hides GPUs
from torch.)

`-b`/`-e` in rccl-tests are **bytes**, so `-e 256M` is a 256 MiB message, not 256M elements. That
matters because the live TP4 serve leaves only ~2.7 GB free per card.

## all_reduce — the decode-critical collective

| size | time | algbw | busbw | | size | time | algbw | busbw |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 1 KiB | 30.79 µs | 0.03 | 0.05 | | 1 MiB | 128.7 µs | 8.14 | 12.22 |
| 2 KiB | 24.62 µs | 0.08 | 0.12 | | 2 MiB | 191.9 µs | 10.93 | 16.39 |
| 4 KiB | 24.85 µs | 0.16 | 0.25 | | 4 MiB | 333.7 µs | 12.57 | 18.85 |
| 8 KiB | 25.58 µs | 0.32 | 0.48 | | 8 MiB | 621.7 µs | 13.49 | 20.24 |
| 16 KiB | 25.47 µs | 0.64 | 0.97 | | 16 MiB | 1195.8 µs | 14.03 | 21.04 |
| 32 KiB | **42.02 µs** | 0.78 | 1.17 | | 64 MiB | 4746.4 µs | 14.14 | 21.21 |
| 64 KiB | 42.69 µs | 1.54 | 2.30 | | 128 MiB | 9469.6 µs | 14.17 | 21.26 |
| 128 KiB | 46.50 µs | 2.82 | 4.23 | | **256 MiB** | 18916 µs | **14.19** | 21.29 |
| 256 KiB | 57.58 µs | 4.55 | 6.83 | | | | | |
| 512 KiB | 81.17 µs | 6.46 | 9.69 | | | | | |

## all_gather

Small sizes track all_reduce almost exactly (27.1–31.1 µs from 2 KiB to 32 KiB). Large sizes
saturate at **26.7 GB/s algbw / 20.0 GB/s busbw** by 64 MiB, flat to 256 MiB.

## Interpretation

1. **Small-message latency is strong AND stable — it is neither poor nor erratic.** all_reduce is
   **24.6–30.8 µs** from 2 KiB to 16 KiB; all_gather is 27.1–31.1 µs across the same band.
   Out-of-place and in-place agree closely and every case validated with `#wrong = 0`. By the
   operator's stated rule, this is the case where **TP8 decode is not blocked by the interconnect**.

2. **Large-message bandwidth saturates at the fabric's measured capability.** all_reduce reaches
   **14.19 GB/s algbw**, which matches D141's independent per-pair measurement of 14.40–14.44 GB/s
   and is ~90% of a Gen4 x16 lane budget. It plateaus smoothly from 16 MiB with **no degradation at
   256 MiB**. By the operator's rule, this is the case where **TP8 has a real chance at better
   long-prefill throughput** — and small messages are not weak, so decode is not the price.

3. **There is a clear algorithm knee at 32 KiB in all_reduce: 25.47 µs → 42.02 µs for a 2× size
   increase.** The dominant AR message in this workload is ~20 KB, which sits *just below* that
   step. That is fortunate, but it also means the message size is close to a cliff: any change that
   pushes the all-reduce message past 32 KiB would cost ~65% latency for a 2× payload.

4. **Measurement-semantics warning.** Three different numbers now circulate for "RCCL world=4
   latency at ~20 KB": D141 records **106.96 µs**, the expert review records **~18.4 µs** (per-rank
   kernel medians), and rccl-tests here measures **25–42 µs** (whole-op wall). These are not
   comparable — they time different things. **Do not mix them**, and treat D141's 106.96 µs as
   un-reconciled with today's figure (it is 3–4× higher than a whole-op measurement of the same
   operation).

## What TP8 would and would not buy

**Collectives are not the obstacle.** The fabric is uniform, low-latency at the sizes decode
actually uses, and saturates at full per-pair bandwidth for prefill-scale messages. Going 4 → 8
ranks roughly doubles the all-reduce bus bandwidth per step and adds latency, but collectives are
only **~4% of wall** at four ranks (D143), so even a 2–3× increase lands around 10%.

**The real TP8 case is weight bandwidth, not collectives.** At TP4 each rank holds ~18.23 GiB, and
decode is weight-bandwidth-bound:

```
18.23 GiB / ~810 GB/s  ~= 22.5 ms/token  ~= 44 tok/s      (measured at TP4: ~47.7 tok/s)
 9.1 GiB / ~810 GB/s  ~= 11.2 ms/token  ~= 89 tok/s      (the TP8 ceiling, if bandwidth-bound)
```

That is a far larger effect than anything the collectives will cost, and it is consistent with the
measured TP4 number.

**Caveats — treat ~89 tok/s as a ceiling, not a forecast.** MoE routing is sparse so not all weights
are read per token; compute, the fixed per-step overhead, the PLE sidecar and KV work do **not**
halve; and the collectives grow. A realistic expectation is a substantial decode gain well short of
2×, plus the capacity win (per-rank footprint 18.23 → ~9.1 GiB, which buys far more than the current
32K context, or unquantized weights).

**Recommendation:** the interconnect evidence supports proceeding with TP8, and the strongest reason
is weight-bandwidth halving rather than collective improvement. Confirm with the 56-pair fabric
matrix once the cards are physically placed, then re-measure RCCL at world=8 — the small-message
latency at 8 ranks is the single number that decides whether decode gains survive.

---

# TP4 baseline re-run under teardown-day hardware state (2026-09-18)

Same MTP2 service, no config change, 1 Hz `rocm-smi` sampling for the whole run.
Monitor log: [`rocm_smi_during_vllm.txt`](rocm_smi_during_vllm.txt) (186 samples, 13:03:18–13:08:08Z).

## Result

| prompt | saved PP | **new PP** | Δ | saved decode | **new decode** | Δ | saved decode range | new decode range |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1,000 | 641.9 | **647.4** | +0.9% | 60.34 | **59.94** | **−0.7%** | 60.12–60.35 | 59.907–59.957 |
| 10,000 | 806.9 | **809.0** | +0.3% | 61.22 | **58.85** | **−3.9%** | 39.02–65.35 | 53.96–62.83 |
| 30,000 | 805.1 | **808.4** | +0.4% | 57.34 | **55.08** | **−3.9%** | 57.00–61.83 | 43.95–56.85 |

**Prefill reproduces faithfully** (within +0.4%); **decode is modestly but reproducibly lower**.
The 1K spans do not overlap (60.12–60.35 vs 59.907–59.957) and neither do the 30K spans
(57.00–61.83 vs 43.95–56.85), so these are real shifts rather than sampling noise.

## Why — the hardware state, from the monitor

```
power     : median 148 W  (56-60% of samples >= 145 W)   -> sitting AT the 150 W cap
sclk      : median 1556 MHz, range 925-1800, only 12-21% of samples at full 1800
mclk      : median 1000 MHz (max)
GFX use   : 100% median  -> fully busy
junction  : GPU0 77/83   GPU1 89/97   GPU2 85/92   GPU3 75/79    (median/max, C)
edge      : GPU0 60/63   GPU1 69/76   GPU2 66/69   GPU3 56/58
memory    : GPU0 67      GPU1 80      GPU2 74      GPU3 64        (max, C)
```

**The rig is POWER-CAP limited, not thermally throttled.** The cards sit at ~148 W against the
configured 150 W cap and DVFS moves sclk between DPM levels to hold that budget — full 1800 MHz only
12–21% of the time. Junction peaks at 97 °C (GPU1), below the ~100 °C throttle region. This is
expected, reproducible behaviour of a 150 W operating point, not a fault.

**But note the thermal spread: GPU1 runs 14–18 °C hotter than GPU3** (97 vs 79 °C junction). In TP4
the **slowest rank gates every step**, so a hot GPU1 costs throughput directly. That is a
cooling/placement issue, and it gets worse with eight cards.

**A monitor-parsing trap worth recording:** `rocm-smi` prints clocks as
`sclk clock level: 8: (1800Mhz)`. A regex taking the first number reads the **level index** and
reports ~6 MHz — physically impossible, and it inverts the throttle verdict to "clocks never reached
full speed". Read the value inside the parentheses.

## Which reference to use for TP8 inference

Per the operator's rule (use the newer run if it is materially worse, since it reflects teardown-day
state): **use this run.** Decode is 0.7–3.9% lower and prefill matches, so it is the more
conservative and more contemporaneous reference — and critically, it is the state the RCCL data
above was measured in, which makes the two sets directly comparable.

Carry these forward as the TP8 baseline: **PP ≈ 647 / 809 / 808, decode ≈ 59.9 / 58.9 / 55.1** at
1K / 10K / 30K, on a rig that is power-cap limited at 150 W/card with one card running noticeably
hotter than the rest.

**Power note for the TP8 plan:** 4 × 150 W = 600 W today; 8 cards at the same cap would be 1200 W if
per-card draw held. TP8 halves each card's weight footprint, so per-card power should fall — but PSU
and cooling headroom for the enclosure should be checked before committing, especially with GPU1
already at 97 °C junction.

---

# Decode concurrency sweep (2026-09-18) — flat aggregate, and it REFUTES the weight-bandwidth case

Everything above measures a **single active request**. The obvious missing question was whether TP8
helps *serving several*. It does not, and the reason also invalidates the projection in the previous
section.

Method: 1K prompt, 512 generated, greedy, **streaming** so per-round latency is measured rather than
inferred; each concurrent request gets a **unique prompt salt** (identical prompts would share KV and
flatter the result); GPU sampled at ~1 Hz and attributed per phase.

Raw: [`concurrency_results.json`](concurrency_results.json), [`concurrency_bench.py`](concurrency_bench.py),
[`analyze_concurrency.py`](analyze_concurrency.py).

| concurrency | aggregate tok/s | scaling | per-request | round median | p95 | max |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | **44.41** | 1.00× | 52.39 | 43.2 ms | 143.6 ms | 170.9 ms |
| 2 | **40.21** | 0.91× | 25.37 | 66.0 ms | 174.8 ms | 3466 ms |
| 4 | **45.24** | 1.02× | 13.62 | 197.2 ms | 308.0 ms | 3005 ms |
| 8 | **40.30** | 0.91× | 11.37 | 200.6 ms | 348.5 ms | 2418 ms |

*(Token counts recomputed from the server's authoritative usage: the first pass counted streamed
chunks, and with MTP2 each chunk carries ~2.84 accepted tokens.)*

**Aggregate throughput is flat** — 40.2–45.2 tok/s across 1→8, a 12% band with no trend. Ideal linear
scaling at 4 concurrent would be **178 tok/s; measured 45 = 25% of ideal, i.e. none.** Round p95 also
degrades (143 → 348 ms) with multi-second stalls, so interactive feel worsens under load even if
capacity is fine.

## Why — and it is measurable in the power

```
concurrency 1: 150.0 W      concurrency 4: 111.0 W
concurrency 2: 146.5 W      concurrency 8: 109.0 W      (GPU "use" ~100% throughout)
```

**Power drops 27% as concurrency rises while reported utilisation stays at 100%.** That is the
signature of **latency/serialization-bound** execution — SMs stalled waiting, not saturated. The
"100% use" counter counts issue cycles, not useful work.

## Correction: the weight-bandwidth case for TP8 does not hold

The section above projected TP8 at ~89 tok/s because TP4 decode is weight-bandwidth-bound
(`18.23 GiB/rank ÷ ~810 GB/s ≈ 22.5 ms/token ≈ 44 tok/s`, "corroborated by the measured ~47.7").

**This sweep falsifies the mechanism.** If weight reads bound decode, batching 4 sequences would read
the weights **once** and emit **4×** the tokens, so aggregate would approach 4×. Measured: **1.0×**.
Weights are not the bottleneck, and the 44-vs-47.7 agreement was a **numerical coincidence** — the
MoE is sparse (top-10 of 128 local experts), so the full per-rank footprint is never read per token.

**Revised verdict: TP8 is unlikely to improve decode throughput, single-stream or multi-stream.** The
binding constraint is per-token serialized work — consistent with D145/D146, where the
elementwise/copy residue is 303k latency-bound launches. **TP8's value is capacity** (far more than
32K context, or unquantized weights), **not throughput.** The physical pre-checks below still stand,
since capacity is worth having — but they should be justified on capacity, not on a decode-speed
expectation.



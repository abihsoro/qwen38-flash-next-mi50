# Measured results

All numbers are medians unless stated. Sources are the decision entries (D140–D151) and the raw
artifacts in `data/` / `results/`. "Saved" = the operator-accepted final run (D149); "fresh" = the
teardown-day re-run (D150).

## Decode and prefill (single active request, MTP2, 150 W)

| prompt | PP (saved) | PP (fresh) | decode (saved) | decode (fresh) |
|---:|---:|---:|---:|---:|
| 1,000 | 641.9 tok/s | 647.4 | 60.34 | 59.94 |
| 10,000 | 806.9 | 809.0 | 61.22 (range 39.02–65.35) | 58.85 (range 53.96–62.83) |
| 30,000 | 805.1 | 808.4 | 57.34 (range 57.00–61.83) | 55.08 (range 43.95–56.85) |

Effective PP = prompt tokens / server time-to-first-token (includes first-token work). Decode
excludes that interval. The MTP0 gate median is ~47.7 tok/s but **timed the whole HTTP request**, so
it is not comparable to the decode rows above.

## Decode concurrency (1K prompt, 512 generated, streaming)

| concurrency | aggregate tok/s | per-request tok/s | round p95 | GPU power |
|---:|---:|---:|---:|---:|
| 1 | 44.41 | 52.39 | 143.6 ms | 150.0 W |
| 2 | 40.21 | 25.37 | 174.8 ms | 146.5 W |
| 4 | 45.24 | 13.62 | 308.0 ms | 111.0 W |
| 8 | 40.30 | 11.37 | 348.5 ms | 109.0 W |

Aggregate is flat (12% band, no trend) while power **falls** — the signature of a latency/
serialization-bound workload. Ideal linear scaling at 4 concurrent would be ~178 tok/s; measured 45.

## Interconnect

| measurement | result |
|---|---|
| peer-to-peer (D141 fabric matrix) | 14.40–14.44 GB/s, 12/12 ordered pairs byte-exact (IQR ≤ 0.014) |
| peer latency @ 20 KB | 16.96–18.13 µs |
| RCCL all-reduce small (2–16 KiB) | 24.6–30.8 µs, flat |
| RCCL all-reduce large (256 MiB) | 14.19 GB/s algbw / 21.29 busbw |
| RCCL all-gather small (2–32 KiB) | 27.1–31.1 µs |
| RCCL all-gather large (256 MiB) | 26.7 GB/s algbw / 20.0 busbw |
| custom wave64 one-shot AR, in-server | ~+4% vs RCCL (47.70 vs 45.37 tok/s, byte-identical) |

Note the RCCL all-reduce has a hard knee at 32 KiB: 25.47 µs @ 16 KiB → 42.02 µs @ 32 KiB. The
workload's dominant AR message is ~20 KB, just below the cliff.

## Optimisation attempts (all gated by ACCEPTANCE.md)

| candidate | effect | verdict |
|---|---|---|
| MTP3 | +3.9% TG, −5% gate (metrics disagreed in sign), prefill −12% | rejected (D147) |
| MTP2 | 60.34/61.22/57.34 decode at 1K/10K/30K | **selected** (D149) |
| skinny GEMM off | 16.76 tok/s vs 47.51 (2.83×) | skinny GEMM is essential (D146) |
| MoE decode autotune | −47.7% in situ | rejected (D144) |
| prefill MoE tiles (corrected search) | PP +45–75% at 128–1536 tokens | **selected** (D148) |
| batched-tokens 2048 | flat (0.22% vs 1.02% drift) | rejected (D143 addendum) |
| dense GEMV WAVES | −5.1% of gemv time = ~0.65% wall | rejected (D145) |

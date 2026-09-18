# Rejected or deferred approaches

Each is closed by evidence, not abandoned for lack of time. See the referenced decision entries in
DECISIONS.md for the full measurements.

| approach | result | why it stays closed |
|---|---|---|
| MoE decode-side tile autotuning (ray-free tuner) | −47.7% in situ | a config tuned for one shape does not transfer; the tuner's isolated objective did not match the serving workload (D144) |
| `--max-num-batched-tokens` 2048 vs 1024 | flat | B 47.74 sat between controls A 47.85 / A′ 47.36; control-vs-control drift 1.02% > effect 0.22% (D143 addendum) |
| Dense GEMV geometry / WAVES sweep | ~2.6% wall ceiling | lm_head already at 96% of the HBM ceiling; WAVES cannot raise occupancy (total threads ≈ N×32 regardless) (D145) |
| split-K for small-N GEMV | ~0.6% wall | eaten by the reduction pass's launch cost; cannot bridge a 4.6% gap (D145) |
| copyBuffer origin via skinny-GEMM | exact null | toggling `VLLM_ROCM_USE_SKINNY_GEMM` left every elementwise family byte-identical while total kernel time differed 2.3× (D146) |
| MTP=3 | lossless but ≤5%, prefill −12% | K=3 is the wrong depth; the two metrics disagreed in sign; MTP2 (K=2) was selected instead (D147/D149) |
| Ray-based MoE tuning | worker segfault importing torch | unresolved on this stack; the ray-free replacement worked but its output was rejected (D144) |
| TP8 as a throughput play | flat aggregate 1→8 concurrent, power *falls* | decode is latency/serialization bound; TP8 is capacity-only (D151) |

## Explicitly NOT pursued (operator decision)

- **split-K** was deliberately skipped: expected ~0.6% wall, would be consumed by the extra
  reduction/launch, and cannot close the gap to ≥50 tok/s.
- **MTP=4** is ruled out by precedent (changed target text on V620).

## Deferred, still real

- **`topkGating` (449 ms, 4.1% of kernel time)** — a pure routing overhead, not reachable by a tile
  config; flagged as a harder post-MTP surgical target, not the next step.
- **elementwise/copy residue** — 303k latency-bound launches; the concurrency sweep (D150/D151)
  confirms per-token serialized work is the decode bottleneck, which re-opens this lane if decode
  throughput is ever the goal again.

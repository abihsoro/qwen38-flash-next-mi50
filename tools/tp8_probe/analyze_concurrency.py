#!/usr/bin/env python3
"""analyze_concurrency.py — recompute the concurrency sweep with REAL token counts.

concurrency_bench.py counted streamed CHUNKS as tokens and understated throughput. Because every
request ran with ignore_eos=True and max_tokens=512, each generated EXACTLY 512 tokens, and the
recorded chunk counts (~180 per request, identical across phases) give ~2.84 tokens/chunk -- the
expected MTP2 accepted-token rate (the review measured mean acceptance 2.49-2.75). So the true
aggregate is (concurrency x 512) / wall, and per-request is the chunk-based rate x tokens-per-chunk.

The across-concurrency RATIO was already valid either way, since every request produced the same
token count in the same number of chunks. This just restores the absolute scale.
"""
import json
import statistics
import sys

PATH = sys.argv[<bus>] if len(sys.argv) > 1 else \
    "<tuning>/tp8_probe/concurrency_results.json"
d = json.load(open(PATH))
GEN = d["gen_target"]                      # max_tokens per request, ignore_eos=True
results = d["results"]

print(f"per-request target: {GEN} tokens (ignore_eos=True, so exactly {GEN} were produced)\n")
hdr = (f"{'phase':>9} {'conc':>4} {'wall s':>8} {'AGG tok/s':>10} {'per-req tok/s':>13} "
       f"{'ITL tok ms':>10} {'round ms':>9} {'p95 round':>10} {'tok/chunk':>10}")
print(hdr)
print("-" * len(hdr))

rows = []
for r in results:
    n = r["concurrency"]
    wall = r["wall_s"]
    chunks = r["tokens_total"]
    tok_per_chunk = (n * GEN) / chunks if chunks else 0
    agg = (n * GEN) / wall if wall else 0
    per_req = (r["per_request_tps_median"] or 0) * tok_per_chunk
    # the recorded ITL is per streamed chunk == per speculative ROUND, not per token
    round_med = r["itl_ms_median"]
    itl_tok = (round_med / tok_per_chunk) if tok_per_chunk else None
    rows.append({**r, "agg_real": agg, "per_req_real": per_req, "tok_per_chunk": tok_per_chunk,
                 "round_med": round_med, "itl_tok": itl_tok})
    print(f"{r['phase']:>9} {n:>4} {wall:>8.2f} {agg:>10.2f} {per_req:>13.2f} "
          f"{itl_tok:>10.2f} {round_med:>9.1f} {str(r['itl_ms_p95']):>10} {tok_per_chunk:>10.2f}")

base = rows[<bus>]["agg_real"]
print(f"\n=== SCALING vs concurrency 1 ===")
for r in rows:
    print(f"  concurrency {r['concurrency']:>2}: aggregate {r['agg_real']:>6.2f} tok/s "
          f"= {r['agg_real']/base:>5.2f}x   per-request {r['per_req_real']:>6.2f} tok/s   "
          f"wall {r['wall_s']:>7.2f}s")

aggs = [r["agg_real"] for r in rows]
print(f"\n  aggregate spread across 1..8 concurrent: {min(aggs):.2f}-{max(aggs):.2f} tok/s "
      f"({100*(max(aggs)-min(aggs))/statistics.mean(aggs):.0f}% band)")
print(f"  ideal linear scaling at 4 concurrent would be {4*base:.0f} tok/s; "
      f"measured {rows[<bus>]['agg_real']:.0f} ({100*rows[<bus>]['agg_real']/(4*base):.0f}% of ideal)")

print(f"\n=== LATENCY (per streamed round, i.e. one speculative pass) ===")
for r in rows:
    print(f"  concurrency {r['concurrency']:>2}: round median {r['round_med']:>7.1f} ms  "
          f"p95 {r['itl_ms_p95']:>7.1f} ms  max {r['itl_ms_max']:>7.1f} ms  "
          f"-> per-token ~{r['itl_tok']:.1f} ms")

print(f"\n=== GPU BY PHASE (from the sampler) ===")
for phase, g in sorted(d.get("gpu_by_phase", {}).items()):
    if not phase.startswith("decode"):
        continue
    print(f"  {phase:>9}: use {g['use_median']}%  power {g['power_median']}W (max {g['power_max']})  "
          f"tj max {g['tj_max']}C  n={g['n_samples']}")

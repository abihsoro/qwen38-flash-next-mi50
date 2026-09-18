# Retained configuration — 2026-09-18

## Measured result and limits

TP4 + EP, 150 W/card, MTP2, tuned prefill MoE tiles, mask-aware MI50 top-10 router, 2048-token
prefill chunks, 32768-token context. Median of three scored requests per exact prompt length
(synthetic prose corpus, zero cached prompt tokens):

| prompt | PP | decode |
|---:|---:|---:|
| 1,000 | 641.9 tok/s | 60.34 tok/s |
| 10,000 | 806.9 tok/s | 61.22 tok/s (range 39.02–65.35) |
| 30,000 | 805.1 tok/s | 57.34 tok/s (range 57.00–61.83) |

Limits recorded: 10K/30K continuations are **not deterministic** at temperature zero; the 10K decode
spread is material; and the answer smoke test is 11/12 (one packing question is wrong in the
reference too). Long-context quality is not established by these throughput runs.

## Selected changes (over the reference)

| change | effect |
|---|---|
| prefill MoE tiles (`VLLM_TUNED_CONFIG_FOLDER=…/moe_configs`) | M/N/K 8/64/32 at batch 128–512, 32/64/32 above; replaces the old tuner that bypassed small-batch settings |
| mask-aware top-10 router (`VLLM_MI50_TOPK=1`) | FP16 logits, 512 experts, top-10 softmax, renormalisation, batches 1–8, padding preserved |
| 2048-token prefill chunks (`--max-num-batched-tokens 2048`) | 1536-token TTFT 2.719 → 1.805 s |
| MTP2 (`--speculative-config {"method":"mtp","num_speculative_tokens":2}`) | ~58–63 tok/s isolated; mean acceptance 2.49–2.75 |
| NUMA node-1 binding | retained for reproducibility; no gain is attributed to it (flat) |
| power cap 150 W/card | earlier runs were 140 W |

## Presets (`tools/review_20260918/scripts/launch.sh`)

| preset | what it does |
|---|---|
| `balanced` | **default** — MTP2, 32K, tuned MoE tiles, MI50 router, 2048 chunks, 150 W |
| `mtp0` | optimized MTP0 / 2K — same MoE tiles and router, no speculative decoding |
| `reference` | original routing and MoE defaults, 1024-token chunks, 2K context |
| `stop` | stops only the campaign-owned service |

## Operate, reproduce, or roll back

```bash
tools/review_20260918/scripts/launch.sh balanced   # start the retained config
tools/review_20260918/scripts/launch.sh mtp0       # MTP0 variant
tools/review_20260918/scripts/launch.sh reference  # repeat-control
tools/review_20260918/scripts/launch.sh stop       # stop

# re-measure (service already running)
<home>/gfx906-venv/bin/python3 run_bench.py --tag UNIQUE --sizes 1000 10000 30000 --kinds prose --reps 3 --gen 128
```

Rollback is `launch.sh reference` (original defaults) or `stop` (nothing running). No boot service
was installed; there is no automatic restart. Source/binary hashes are pinned in
`accepted-sha256.json`; the compiled extensions were not rebuilt.

## Deferred follow-up

- **56-pair fabric matrix + world=8 RCCL** — only if TP8 is pursued for capacity (D151).
- **`topkGating` (449 ms, 4.1% of kernel time)** — a real routing overhead, not reachable by tile
  config; a harder post-MTP surgical target.
- **elementwise/copy residue** — the decode bottleneck per the concurrency sweep; re-open only if
  decode throughput becomes the goal again.

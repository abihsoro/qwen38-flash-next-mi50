# gfx906 (MI50 / Vega 20) — measurement contract

This directory mirrors the reporting structure of the V620 campaign repo
(`qwen38-flash-next-v620`, `docs/rdna2/`): a measurement contract (this file), an archived
`campaign/`, a narrative `worklog/`, raw `data/`, and the `tools/` that produce the numbers.
DECISIONS.md / PROGRESS.md / TRAPS.md at the repo root are the append-only working log; the
`campaign/` files are the curated, reproducible view.

## 0. Facts that have changed (superseded in place, not overwritten)

| item | recorded earlier | current |
|---|---|---|
| "Baseline 49.04 tok/s" | D142 | **high outlier**; honest gate median ~47.7 (D143 addendum) |
| "~47.7 tok/s" as decode rate | headline through D147 | **the gate timed the whole HTTP request** (includes prompt); corrected decode reference **~55–56 tok/s** (D148) |
| MI50 link generation | "Gen3 x16 (its max)" (D078) | **Gen4 x16** (16.0 GT/s), corroborated by 14.4 GB/s peer bandwidth (D150) |
| MoE tile tuning | "closed as a dead end" (D144) | **works for PREFILL** with a search targeting real prefill batches (D148); only the decode-side autotuner was wrong |
| speculative depth | MTP=3 tested/rejected (D147) | **MTP=2 selected** (D149); K=3 was the wrong depth, K=2 gives mean acceptance 2.49–2.75 |
| RCCL world=4 @ 20 KB | 106.96 µs (D141) | **25–42 µs** (rccl-tests, whole-op); D141 figure un-reconciled (D150) |
| TP8 decode outlook | ~89 tok/s projected from weight bandwidth (D150 §4) | **refuted by the concurrency sweep**; TP8 is a capacity strategy, not a throughput one (D151) |

## 1. What is being measured

Qwen3.8-Flash-Next (176 B, ~6 B active, 51 B-row int4 n-gram sidecar served from CPU) on
**4× MI50 (Vega 20 / gfx906, 32 GB)** behind a Broadcom/LSI PEX88096 Gen4 switch, TP=4 + expert
parallelism, no weight offload, served by the MI50 port of the leapdragon vLLM fork.

Final operating point (D149): **MTP2**, 150 W/card, tuned prefill MoE tiles, mask-aware MI50
top-10 router, 2048-token prefill chunks, 32768-token context. TunableOp and dense INT8 off.

## 2. Configuration (environment)

| knob | value | notes |
|---|---|---|
| speculative | `--speculative-config {"method":"mtp","num_speculative_tokens":2}` | MTP0 and MTP3 also measured (see MEASUREMENTS.md) |
| `--max-model-len` | 32768 | `balanced` preset; `reference` uses 2048 |
| `--max-num-batched-tokens` | 2048 | prefill chunk budget |
| `--max-num-seqs` | 4 | hardcoded by the serve script; concurrency 8 queues rather than batches |
| `--gpu-memory-utilization` | 0.90 | KV pool shrinks when MTP is on |
| power cap | 150 W/card (`power1_cap` 150,000,000 µW) | earlier runs were 140 W (V620) |
| router / MoE | `VLLM_MI50_TOPK=1`, `VLLM_TUNED_CONFIG_FOLDER=<tuning>/moe_configs` | mask-aware top-10 router + prefill MoE tiles |
| CPU placement | `numactl --cpunodebind=1 --preferred=1` | all four GPUs are on NUMA node 1 |

## 3. Reproducing the measurements

```bash
# the operator-selected service (MTP2/32K)
tools/review_20260918/scripts/launch.sh balanced      # also: mtp0, reference, stop

# single-request PP/decode at exact prompt lengths (2 warmups + 3 scored per length)
<home>/gfx906-venv/bin/python3 run_bench.py --tag UNIQUE --sizes 1000 10000 30000 --kinds prose --reps 3 --gen 128

# decode concurrency sweep (streaming, unique prompt per request, 1 Hz GPU sampling)
python3 tools/tp8_probe/concurrency_bench.py --sizes 1 2 4 8 --prompt-tokens 1000 --gen 512

# RCCL collectives (world=4)
cd <tuning>/tp8_probe/rccl-tests && \
  HIP_VISIBLE_DEVICES=0,1,2,3 ./build/all_reduce_perf -b 1K -e 256M -f 2 -g 4 -w 20 -n 200
```

## 4. Concurrency / batching measurement rules (learned, not assumed)

- **verify the batch actually happened** — `--max-num-seqs 4` means a "concurrency 8" run queues; the
  sweep in MEASUREMENTS.md reports this rather than hiding it.
- **unique prompt per request and per rep** — identical prompts hit the prefix cache and flatter
  later runs.
- **report aggregate, decode-aggregate, and per-stream separately** — per-stream × C ≠ aggregate.
- **state input/output lengths** — everything here is ~1K prompt / 512 generated (short context).
- **count tokens, not streamed chunks** — with MTP2 one chunk carries ~2.8 accepted tokens; the
  server's `usage.completion_tokens` is authoritative.

## 5. Yardsticks to expect (this system, 2026-09-18, 150 W)

| case | result |
|---|---|
| decode MTP0, single request (gate) | ~47.7 tok/s (whole request, includes prompt) |
| decode MTP2, 1K prompt | 60.34 tok/s (saved) / 59.94 (fresh re-run) |
| decode MTP2, 10K / 30K prompt | 61.22 / 57.34 tok/s (saved), wide variance at 10K |
| prefill PP, 1K / 10K / 30K | 641.9 / 806.9 / 805.1 tok/s |
| aggregate decode, 1→8 concurrent | flat ~40–45 tok/s (no batching gain) |
| all-reduce 20 KB (rccl-tests) | 25–42 µs |
| all-reduce large (256 MiB) | 14.19 GB/s algbw |
| peer-to-peer (D141 fabric matrix) | 14.40–14.44 GB/s, 12/12 ordered pairs byte-exact |

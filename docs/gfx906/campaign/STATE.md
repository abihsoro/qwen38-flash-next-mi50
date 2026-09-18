# MI50 performance campaign — current state (2026-09-18)

## What is running

<container> (`<container>`, Ubuntu 24.04, 128 GiB / 64 cores, ROCm 7.14.0-gfx906, torch 2.13.0+gfx906)
on <host>. Port 8002, PID <pid>, launch `<launch-tag>`.

- TP4 + expert parallelism, fp16, **no weight offload**; PLE served from the int4 sidecar.
- **MTP2** (`num_speculative_tokens=2`), tuned prefill MoE tiles, mask-aware MI50 top-10 router,
  2048-token prefill chunks, 32768-token context.
- 150 W/card power caps; TunableOp and dense INT8 off.

## Wrap-up

The operator stopped further tuning experiments at the ~10%-additional-decode-gain threshold
(D148), then selected MTP2 with the 32K context (D149). The final numbers are:

| prompt | effective PP | decode (median of 3) |
|---:|---:|---:|
| 1,000 | 641.9 tok/s | 60.34 tok/s |
| 10,000 | 806.9 tok/s | 61.22 tok/s (range 39.02–65.35) |
| 30,000 | 805.1 tok/s | 57.34 tok/s |

A fresh teardown-day re-run (D150) reproduced prefill within +0.4% and decode 0.7–3.9% lower
(59.94 / 58.85 / 55.08), with the rig power-cap limited at ~148 W (median) and GPU1 running 14–18 °C
hotter than GPU3.

## Measurement contract

Every number follows the contract in [`../README.md`](../README.md): exact token lengths, unique
cache salts (zero cached prompt tokens), two warmups then three scored requests, greedy
(temperature 0) decoding, and a deterministic-hash cross-check.

## TP8 verdict (recorded, not pending)

The TP8 feasibility probe (D150/D151) concluded **TP8 is a cheap-capacity strategy, not a
performance strategy**: the decode concurrency sweep showed aggregate throughput is flat from 1 to 8
concurrent requests while GPU power *falls* (150 → 109 W), i.e. the workload is latency/serialization
bound. The physical pre-checks (56-pair fabric matrix, world=8 RCCL) remain worth doing but are
justified by **capacity per dollar**, not decode speed.

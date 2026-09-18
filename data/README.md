# data/ — raw evidence

Mirror of the V620 repo's `data/` convention: raw measurement artifacts live here, one directory per
probe, with the scripts that produced them under `tools/`.

| path | contents |
|---|---|
| `tp8_probe/` | topology captures, RCCL sweeps, the teardown-day baseline, and the concurrency sweep |
| `../results/tp4_bringup/` | fabric matrix, RCCL world=4, custom-AR and TP4-gate JSONL (ingested) |
| `../tools/review_20260918/` | the expert-review campaign evidence (requests, summary, final state) |

## tp8_probe artifacts

- `host_topology.txt`, `ct_topology.txt` — `lspci`/`rocm-smi` topology from both systems (D150)
- `rccl_all_{reduce,gather}_{small,large}.txt` — RCCL world=4 sweeps
- `rocm_smi_during_vllm.txt` — 1 Hz GPU sampling during the fresh baseline (186 samples)
- `concurrency_results.json` — the decode concurrency sweep

Note: these captures have been redacted - host identifiers are replaced with `<…>` placeholders.

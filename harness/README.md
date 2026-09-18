# harness/ — benchmark + correctness runners; the SOLE writer of results/

## Who may write results (I3)

Only code under `harness/` writes `results/*.jsonl`. The executor reads those
files freely and may add new harness code, but never hand-writes, edits, or
synthesises a results row. `harness/results.py` is the single gateway:

- `results.make_row(...)` builds a complete 34-field row (schema:
  `results_schema.json`, fields per Work Order Rev 4 §5).
- `results.append_row(...)` appends to `results/{task_id}.jsonl` and refuses
  duplicate run_ids (append-only).
- `results.write_artifact(...)` writes task artifacts under `results/`
  (e.g. `profile_s4.jsonl`); it refuses to overwrite without `force=True`.
- `results.read_rows(...)` reads rows back for reporting.

## Rules encoded here

- **I1** No claim without an artifact: every value in a row is either measured
  by the harness or `"not measured"`/`null`. Nothing is guessed.
- **I2** Exit codes are truth: `run_task.py` enforces the explicit timeout,
  kills the process group on expiry, and records `exit_code=124` for a hung
  run. A pass requires exit 0 within the timeout.
- **I8** Every row snapshots `config/environment.lock` at write time. The lock
  is regenerated only by `harness/env_fingerprint.py` — never by hand.
- **Thermal validity (§5)** applies from S5 onward: no row is valid without
  ≥ 600 s soak; rows whose sustained clocks deviate > 5 % from the session
  median are marked invalid (`notes`) and excluded, never averaged in.

## Dependencies

Stdlib only, plus PyYAML (present at bootstrap; required for parsing
`config/tolerances.yaml`, which the harness reads from S3 onward).
`env_fingerprint.py` and `run_task.py` need `git`, `lspci`, `rocminfo` present
in the guest when those measurements are actually taken.

## Components

| File | Purpose |
|---|---|
| `results_schema.json` | JSON Schema for the 34-field run row (additionalProperties: false) |
| `results.py` | row build / validate / append; artifact writer; readers |
| `run_task.py` | `--task ID --timeout S -- cmd...` → one results row |
| `env_fingerprint.py` | regenerates `config/environment.lock` |
| `verify_bootstrap.py` | bootstrap acceptance checker (exit 0 on clean tree) |
| `verify.py` | S3 correctness-oracle stub — exits non-zero until S3 lands |
| `s1_checks/` | nine toolchain-gate checks (work order §6.S1); run via `run_task.py --task S1`, `run_all.sh` orchestrates on the GPU guest |
| `gpu_probe.py` | GPU-guest probe: `--emit` on the guest (no repo access), `--write` ingests measured JSON via results.py |
| `s0_model_audit.py` | S0 checkpoint audit: config.json + safetensors headers -> config/model.json + results/byte_budget.jsonl |

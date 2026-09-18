# results/ — append-only measurement rows (I1, I3)

- Written ONLY by `harness/` (sole writer, I3). The executor never hand-writes
  a row. Empty at bootstrap by design — nothing is fabricated.
- Run rows: `results/{task_id}.jsonl`, conforming to `harness/results_schema.json`
  (all 34 fields of Work Order Rev 4 §5).
- Task artifacts: e.g. `byte_budget.jsonl` (S0), `profile_s4.jsonl` and
  `collective_prediction.jsonl` (S4), `ceilings.jsonl` (S5) — harness-written.
- **Thermal validity (§5):** no row is valid without ≥ 600 s soak; rows whose
  sustained clocks deviate > 5 % from the session median are invalid and
  excluded, never averaged in.

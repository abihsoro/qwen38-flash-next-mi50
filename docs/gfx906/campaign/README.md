# MI50 campaign archive — 2026-09-17/18

The optimisation campaign that produced the retained MTP2 configuration measured in this repo:
target ≥50 tok/s, exact-output matching, and full acceptance criteria. These files are the curated,
reproducible view; DECISIONS.md (D140–D151) is the append-only working log they are drawn from.

| file | what it is |
|---|---|
| `STATE.md` | current state: what is running, where, and the wrap-up |
| `MEASUREMENTS.md` | per-experiment medians in one table (gate, PP, concurrency, RCCL, fabric) |
| `RETAINED_CONFIG.md` | the retained `balanced`/`mtp0`/`reference` presets, limits, operate/rollback |
| `ACCEPTANCE.md` | the A/B/A′ protocol and the deterministic-hash gate |
| `DEAD_ENDS.md` | approaches tried and rejected, and why |
| `RESULTS.md` | condensed per-decision narrative (D140 → D151) |
| `accepted-sha256.json` | SHA-256 pins of the accepted configuration and the greedy-output hash |

## Provenance

- These files are curated from DECISIONS.md, PROGRESS.md and TRAPS.md, which remain the authoritative
  record at the repo root.
- **This is a redacted public mirror.** Host identifiers (IPs, hostnames, and internal filesystem
  paths) have been replaced with `<…>` placeholders, following the V620 repo's convention. The
  authoritative un-redacted record remains private.
- The port source tree (`vllm/`) is a separate repo and is deliberately **not** committed here; its
  commit is recorded via the environment fingerprint in the measurement rows.

# Acceptance protocol

Every candidate change is gated the same way, and the rules were calibrated against the measured
noise floor (~1% whole-serve, D143 addendum).

## The gate

1. **Isolated correctness first** — a microbench/harness with a CPU reference and a tolerance check,
   so a faster-but-wrong variant can never be adopted (D145's k-unroll incident is the cautionary
   tale: an unchecked launch looked exactly like a wrong answer).
2. **In-serve A/B/A′** — control / candidate / control, three gates each. A′ exists because
   whole-serve drift is ~1%, so a single control is not a baseline.
3. **Deterministic hash** — greedy output must be byte-identical across all arms. The reference for
   the fixed prompt is `0128852903291e32` (see `accepted-sha256.json`). A changed hash from a
   supposedly lossless change is an automatic reject (the MTP=4 precedent).
4. **Adopt only if >2%** vs BOTH controls, with the hash unchanged. Sub-2% is noise by construction.

## Correctness scope

- MTP is lossless speculative decoding; MTP2 reproduced the MTP0 greedy hash byte-identically
  (D147/D149), and MTP4 changed the target text on V620 and was rejected (DO-NOT-ATTEMPT).
- Long-context continuations at temperature zero are **not** deterministic here (the 10K/30K runs
  varied across reps); determinism claims are limited to the fixed prompt and short context.
- The answer smoke test passes 11/12 in the retained config, with a packing question that is wrong in
  the reference too — a regression check, not a quality evaluation.

## Recorded exceptions

- The D143 addendum batched-tokens experiment and the D145–D147 work were rejected *by* this
  protocol (flat / sub-2% / metrics disagreeing in sign), which is the protocol working, not a
  failure of it.

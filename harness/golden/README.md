# S3 golden-trace harness (T1/T2/T3) — design

**Purpose (D039):** generate CPU fp32 golden traces from the SAME 8-layer slice
the GPU surrogate serves, then verify the GPU model path (T2 per-layer drift),
the PLE offload protocol (T1: no stale/garbled output — the trap that cost the
fork a 70->55 t/s correction), and end-to-end determinism (T3). Golden
generation needs NO GPU — it runs while the card is busy.

**Golden source:** `transformers==5.16.1` ships `qwen4_exp` (verified on <vm>:
Qwen4ExpTextModel etc., AutoConfig loads the AWQ config). The reference is the
HF Qwen4Exp text backbone in fp32 on CPU with `num_hidden_layers=8`, fed the
surrogate weights. The fork's vLLM port mirrors this HF math, so HF is the
reference the port must match.

**Reference-input coupling (T3):** the layer-2 PLE injection needs the n-gram
rows. The reference reads the SAME int4 sidecar the GPU offload worker serves
(via the vllm `_PleQuantTable` loader on CPU) so both sides see identical rows.
For the harness the ngram ids are chosen to hit known rows (a fixed sequence of
rows, incl. cross-shard boundaries).

**Checkpoints (trace files under harness/golden/traces/):**
1. `tok{0..N}.layer{k}.in.f32` / `.out.f32` — per-layer input/output for N
   decode tokens (T2 compares these against the GPU surrogate's per-layer
   buffers, needs the fork's model to expose them — see below).
2. `tok{0..N}.embed.f32` — post-embedding activations (first-layer drift).
3. `tok{0..N}.final_logits.f32` — lm_head output slice (top-k matches).
4. `ngram_row{0..R}.f32` — the sidecar rows served at the PLE point (T3
   checks the GPU's injected vector equals the reference's).

**T1 (PLE protocol):** greedy decode of a long structured prompt through the
surrogate; two runs must be byte-identical AND match the CPU reference's greedy
trace; garble scan (doubled tags/words, unbalanced brackets); GPU busy must
return to ~0 after the run; `PLE_OFFLOAD_DEBUG_DELAY_MS` run must stay
identical (proves the forward waits for the offload — the T1 fix).

**T2 (per-layer drift):** |gpu - ref| per layer, fp32 accumulation tolerance
1e-4-class; drift must not grow monotonically (the PLE-free unit at layers 3-7
bounds the comparison).

**T3 (dequant + offload correctness):** the injected n-gram rows must equal the
reference's rows bit-for-bit after the offload round-trip (worker gather ->
DMA -> pinned -> H2D -> model add), with PLE_OFFLOAD_DEBUG_DELAY_MS to force
the race window the fork hit.

**GPU model introspection:** the fork's model must expose per-layer hooks for
T2. Check vllm/models/qwen4_exp for a trace hook; if absent, add one under a
`VLLM_PLE_SURROGATE_TRACE` env gate in the PORT TREE ONLY (never upstream).

**Execution order:** (1) make_surrogate.py once shard 2+5 land; (2) generate
golden traces on CPU (needs ~24 GB RAM for fp32 of the 12 GB slice);
(3) serve_surrogate.sh after the operator stops the llama container;
(4) run the T1/T2/T3 comparisons against the live server.

## Constraints discovered (2026-09-02)

- **<vm> RAM = 22 GB total.** The 8-layer slice is ~12 GB fp16 (weights); fp32
  of everything does NOT fit. The reference must run **layer-by-layer** (weights
  resident bf16 ~12 GB; activations fp32; MoE experts dominate the footprint),
  or the golden generation must move to a host with more RAM. buff/cache
  (~14 GB) is reclaimable when the model loads.
- **The HF Qwen4ExpTextNGramEmbedding instantiates the FULL ngram table**
  (~51 B params across per-head prime-sized slices — the 102 GB bf16 table).
  The golden reference must NOT instantiate it: monkeypatch the PLE lookup to
  gather rows from the same int4 sidecar the vLLM worker serves (via the
  `_PleQuantTable` loader), using the row-ids the model's own ngram hash
  produces (mirror the fork's ple_layer row mapping; verify against
  transformers' `Qwen4ExpTextNGramEmbedding` prime-split math).
- Transformers 5.16.1 on <vm> ships the reference (Qwen4ExpTextModel etc.);
  AutoConfig loads the AWQ config without remote code.

## Second constraint (measured, 2026-09-02)

- **HF holds experts UNPACKED: ~5 GB bf16 per layer (512 experts x 1920x2560).**
  8 layers = ~40 GB - far over the 22 GB RAM. Even the bf16 whole-model
  reference cannot be resident. The generator MUST drive the decoder layers
  one at a time: build each Qwen4ExpTextDecoderLayer (importable from
  transformers), load its per-layer state dict slice, forward on the running
  hidden state, capture the trace, free. State (linear-attn conv states + A
  decay, full-attn KV, PLE conv states, ngram context) must be carried
  manually between the per-layer forwards for decode traces.
- The plumbing smoke test validated: trimmed text_config builds, the sidecar
  loader + ngram-patch install cleanly, and the patch skips the 102 GB table
  allocation (the earlier orig_init attempt OOM'd at 102 GB).

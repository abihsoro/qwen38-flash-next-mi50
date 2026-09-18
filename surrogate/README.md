# Surrogate (D039) — 8-layer full-shape single-GPU slice

**Purpose (operator framing):** validate the model path, the PLE offload
protocol (the layer-2 n-gram injection: host-counter handshake, worker DMA,
sidecar mmap/prefault — trap T1), and the dequant path on REAL weights.
It does NOT validate EP dispatch, collectives, or cross-rank graph capture;
it is not a performance measurement and produces **no admissible I5 evidence**
(amended I5). shape_profile: **full** (full-width projections, all 512 experts,
full lm_head) — NOT tp4_rank0.

**Slice:** `num_hidden_layers=8` (layers 0-7 = [la,la,la,fa,la,la,la,fa] —
two repeating units; layer 2 carries the PLE injection, layers 3-7 give a
PLE-free unit for T2 per-layer drift). All layer-0..8 tensors are in
`model-00002-of-00005.safetensors`; lm_head + embed_tokens + the global
hyper-connection-mixer tensors are in `model-00005-of-00005.safetensors`.

**Artifacts:**
- `make_surrogate.py` — builds the slice: truncated config.json + verbatim
  tensor extraction into `surrogate_weights/` (pack-quantized tensors copied
  as-is; the loader quantizes as for the full model).
- `serve_surrogate.sh` — the serve command (TP1, EP on, `--language-model-only`,
  PLE offload against the real int4 sidecar, dtype float16).
- The T1/T2/T3 harness (golden traces on CPU fp32 from the same slice) lives
  in harness/golden/ (S3).

**Run trigger:** execute the moment the operator stops the llama.cpp container
(31 GB freed). `make_surrogate.py` needs the downloaded AWQ shards 2 and 5.

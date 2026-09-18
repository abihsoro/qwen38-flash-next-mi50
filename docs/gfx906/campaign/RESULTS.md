# Campaign narrative (condensed from DECISIONS.md)

The append-only log is DECISIONS.md at the repo root; this is the condensed arc.

## Bring-up (D140–D143)

Four MI50-class Vega 20 cards behind a PEX88096 Gen4 switch, on a Supermicro <board> / dual EPYC
<cpu>. Fabric validated (12/12 ordered peer pairs byte-exact at 14.4 GB/s), custom wave64 one-shot
all-reduce brought live and confirmed ~+4% in-server, and TP4 made **fully resident** (18.23 GiB/
rank × 4, no offload, PLE from the int4 sidecar). The M4 profile attributed 85.5% of wall to kernel
time; `gemm_dense` 24.3%, collectives 21.0%, `moe_int4` 19.8%, elementwise 11.8%.

## Optimisation (D144–D147) — six lanes, six closures

Each candidate was run through the A/B/A′ + deterministic-hash gate. MoE decode-side autotuning
measured −47.7% in situ (D144); dense-GEMV geometry was sized down to a ~2.6% wall ceiling with
lm_head already at 96% of the HBM ceiling (D145); the copyBuffer hypothesis was refuted by an exact
null while confirming skinny GEMM is worth 2.83× (D146); MTP3 was lossless but ≤5% with −12%
prefill (D147).

## Expert review and final selection (D148–D149)

The review corrected three things: the ~47.7 gate **included prompt time** (corrected decode
reference ~55–56), the MoE tuner **had measured the wrong workload** (a corrected prefill search
succeeded and is retained), and **MTP2 — not MTP3 — was the right depth** (mean acceptance 2.49–2.75).
The operator selected MTP2 with optimized prefill and a 32K context; the service runs on <container>:8002.

## TP8 probe and verdict (D150–D151)

Topology (one switch, one root complex, one NUMA node, spare ports for four more cards), RCCL
world=4 (strong at both ends: flat 25 µs small-message, 14.2 GB/s large), and a fresh baseline all
looked favourable — until the **decode concurrency sweep**: aggregate throughput is flat from 1 to 8
concurrent requests while GPU power *falls* to ~109 W, i.e. the workload is latency/serialization
bound. **Final verdict (D151): TP8 is a cheap-capacity strategy, not a performance strategy**; its
value is context/quantization headroom, not decode speed.

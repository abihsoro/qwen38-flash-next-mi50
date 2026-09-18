# Baseline comparison (N-C4) — V620 (fork, gfx1030) vs gfx906 measured

Source: leapdragon/vllm-rdna2-qwen docs/rdna2/RESULTS.md (gfx1030) vs our
results/ (gfx906, 150 W cap, in-guest). These are reference numbers; Track M4
produces the real four-card profile.

| Metric | V620 (gfx1030, 4x) | gfx906 (1x, measured) |
|---|---|---|
| Decode (MTP=0) | 60-62 t/s | not yet served (port in progress) |
| Decode (MTP=3) | ~100 t/s (2.7-3.1 accepted tok/step) | O5 |
| Launch overhead | 2.81-3.11 us | 1.35 us (HIP b2b); Triton 9.54 us |
| Collectives | 119 x 156 us RCCL; 33 us custom | reduce loop bit-exact (single-card) |
| lm_head fp16 | 1270 us; int8 640 us | 404 us M=1 (786 GB/s effective) |
| Kernel count/step | ~1,788 (post-T46) | M4 (four-card) |
| HBM copy | (Navi21 profile) | 613.2 GB/s @ 150 W |
| Push vs pull P2P | 14.3 vs 5.7 GB/s | M0 (peer) |

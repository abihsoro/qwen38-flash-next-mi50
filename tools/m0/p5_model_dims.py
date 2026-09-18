#!/usr/bin/env python3
"""p5_model_dims.py — enumerate the dense fp16 projection shapes this model actually has.

The caller (vllm/model_executor/layers/utils.py:279) routes a projection to
ops.gemv_f16_rdna2 only when: VLLM_ROCM_USE_SKINNY_GEMM, gfx906, fp16 x and w,
0 < tokens <= 8, k % 8 == 0, contiguous. So at decode (M=1) EVERY dense fp16
projection goes through this kernel, and the variant is picked by N:
    N >= 1152 -> <8,1>   (the profile's hot kernel)
    N >=  576 -> <4,1>
    N >=  288 -> <2,1>
    else      -> <1,1>

We pair each candidate N with its weight footprint (N*K*2 bytes) and the time that
footprint SHOULD take at the known HBM ceiling, so measured per-launch times can be
matched to shapes without relying on trace args.
"""
import json

CFG = "<models>/models/qwen38-flash-next-awq/config.json"
t = json.load(open(CFG))["text_config"]

H = t["hidden_size"]
NM = t.get("num_attention_heads")
NKV = t.get("num_key_value_heads")
HD = t.get("head_dim") or (H // NM if NM else None)
MI = t.get("moe_intermediate_size")
NE = t.get("num_experts")
NEXP = t.get("num_experts_per_tok")
NL = t.get("num_hidden_layers")

print("=== model shape facts ===")
for k in ("hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim",
          "moe_intermediate_size", "num_experts", "num_experts_per_tok",
          "num_hidden_layers", "vocab_size", "linear_num_key_heads",
          "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim"):
    if k in t:
        print(f"  {k:28} = {t[k]}")
print(f"  derived head_dim              = {HD}")

q_dim = (NM * HD) if (NM and HD) else None
kv_dim = (NKV * HD) if (NKV and HD) else None

CEIL_GBPS = 810.0     # D044 corrected ceiling; D144 lm_head measured ~800 GB/s

print("\n=== candidate dense projections at M=1 (N x K), with ceiling-time ===")
cands = []
def add(label, N, K, per_layer=1, rank="tp4"):
    if not N or not K:
        return
    bytes_ = N * K * 2
    us = bytes_ / (CEIL_GBPS * 1e3) * 1e6      # bytes / (GB/s) -> us
    cands.append((label, N, K, bytes_, us, per_layer))

add("q_proj", q_dim, H)
add("k_proj", kv_dim, H)
add("v_proj", kv_dim, H)
add("o_proj", H, q_dim)
add("lm_head", t.get("vocab_size"), H)
add("shared_expert.gate_up", 2 * MI, H)
add("shared_expert.down", H, MI)
add("router(mlp.gate)", NE, H)

print(f"{'projection':24} {'N':>8} {'K':>6} {'MB read':>9} {'us @810GB/s':>12} {'variant':>9}")
for label, N, K, b, us, pl in sorted(cands, key=lambda c: -c[<bus>]):
    var = "<8,1>" if N >= 1152 else "<4,1>" if N >= 576 else "<2,1>" if N >= 288 else "<1,1>"
    print(f"{label:24} {N:8d} {K:6d} {b/1e6:9.1f} {us:12.1f} {var:>9}")

print(f"\nNOTE ceiling-time = weight bytes / {CEIL_GBPS} GB/s (the D044 HBM ceiling).")
print("Measured per-launch times can now be matched to shapes:")
print("  a launch at ~16.4 us  <-> 13.1 MB  <-> N=2560,K=2560 at ~100% of ceiling")
print("  a launch at ~397 us   <-> 318 MB   <-> N=62080,K=2560 (observed: lm_head)")
print(f"\nlayers = {NL}; if the hot <8,1> count is ~99/token and there are {NL} layers,")
print(f"that implies ~{99/NL if NL else 0:.2f} dense fp16 projections per layer per token.")

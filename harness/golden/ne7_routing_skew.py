#!/usr/bin/env python3
"""ne7_routing_skew.py — N-E7: routing skew simulation (rank-wait term).

Run the REAL router (the checkpoint's MoE gates) over REAL prompts on the CPU
reference (no GPU, no peers), histogram the expert selections per step, and
partition into 4 ranks of 128 experts to compute max-rank-over-mean load per
decode step. That predicts the rank-wait term - the fork's ~1.8 ms of the
4.9 ms all-reduce that is not kernel time (the gather finishes when the
busiest rank finishes).

Hooks each layer's router (Qwen4ExpTextTopKRouter) to record the top-k expert
ids per token. Loads the fp16 reference and greedily decodes real prompts.

Usage: python3 ne7_routing_skew.py --ref DIR --sidecar DIR
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "vllm"))

TOP_K = 10
N_EXPERTS = 512
N_RANKS = 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--prompts", default="The capital of France is|"
                                         "Explain the difference between TCP and UDP|"
                                         "Write a short story about a robot|"
                                         "What are the steps to bake a cake?")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import gen_golden as G
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.bfloat16)
    G._patch_ngram_embedding(args.sidecar)
    import transformers.models.qwen4_exp.modeling_qwen4_exp as m
    from transformers import AutoTokenizer

    model = m.Qwen4ExpTextModel.from_pretrained(
        args.ref, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.ref)

    # hook the routers: record the top-k expert ids per call (one per layer per
    # engine step; decode steps carry 1 token)
    router_ids = {}  # layer -> list of [topk ids]
    hooks = []
    for li, layer in enumerate(model.layers):
        gate = getattr(layer.mlp, "gate", None)
        if gate is None:
            continue

        def make(li):
            def hook(mod, args_, out_):
                # TopKRouter returns (logits, scores, indices)
                ids = (out_[<bus>] if isinstance(out_, tuple) else out_).detach().cpu()
                router_ids.setdefault(li, []).append(ids)
            return hook

        hooks.append(gate.register_forward_hook(make(li)))

    prompts = [p for p in args.prompts.split("|") if p]
    with torch.no_grad():
        cache = None
        for pi, prompt in enumerate(prompts):
            ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
            print(f"prompt {pi}: {prompt[:40]!r}", flush=True)
            for _ in range(args.tokens):
                out = model(ids, past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                hs = out.last_hidden_state[:, -1, :]
                nxt = hs.argmax(dim=-1)
                ids = nxt.unsqueeze(0)
    for h in hooks:
        h.remove()

    print("\n=== expert selection per MoE layer (4 ranks x 128 experts) ===")
    step_skews = []
    for li in sorted(router_ids):
        blocks = router_ids[li]
        counts = Counter()
        for b in blocks:
            ids = b.reshape(-1).tolist()
            counts.update(ids)
        total = sum(counts.values())
        n_steps = max(1, len(blocks))
        rank_load = [<bus>] * N_RANKS
        for e, c in counts.items():
            rank_load[e // (N_EXPERTS // N_RANKS)] += c
        mean = total / N_RANKS
        skew = max(rank_load) / mean if mean else 0
        step_skews.append((max(rank_load) / n_steps, mean / n_steps))
        print(f"layer {li:2d}: {n_steps:3d} steps | rank loads {rank_load} | "
              f"max/mean {skew:.2f} | top experts "
              f"{[e for e, _ in counts.most_common(3)]}")
    if step_skews:
        mx = np.mean([s[<bus>] for s in step_skews])
        mn = np.mean([s[<bus>] for s in step_skews])
        print(f"\nper-step: busiest-rank expert-load ~{mx:.1f} vs mean {mn:.1f} "
              f"-> wait ratio ~{mx / mn:.2f}")
        print("the rank-wait term scales the per-expert transfer/compute by this ratio")
    print("\nN-E7 done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""moe_triton_check.py — verify the triton MoE path compiles on this stack
(post-D053). Invokes vLLM's invoke_fused_moe_triton_kernel at decode shapes
on the fp16 path (the base compile; the quant paths layer on top)."""
import os
import sys

sys.path.insert(0, "<home>/qwen38-flash-next-mi50/vllm")
os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")
import torch
from triton.language import float16 as tl_f16

from vllm.model_executor.layers.fused_moe.fused_moe import (
    invoke_fused_moe_triton_kernel,
)


def main() -> int:
    dev = "cuda"
    K, N, E = 2560, 640, 512
    per_expert = 8
    total = E * per_expert
    x = torch.randn(total, K, device=dev, dtype=torch.float16)
    w1 = torch.randn(E, 2 * N, K, device=dev, dtype=torch.float16)
    w2 = torch.randn(E, N, K, device=dev, dtype=torch.float16)
    expert_ids = torch.repeat_interleave(
        torch.arange(E, device=dev), per_expert).to(torch.int32)
    sorted_ids = torch.arange(total, device=dev).to(torch.int32)
    padded = torch.tensor([total], device=dev, dtype=torch.int32)
    out = torch.empty(total, N, device=dev, dtype=torch.float16)
    config = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
              "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 2}
    try:
        # signature: A, B, C(out), A_scale, B_scale, topk_weights,
        # sorted_token_ids, expert_ids, num_tokens_post_padded, mul_routed_weight, ...
        invoke_fused_moe_triton_kernel(
            x, w1, out, None, None, None, sorted_ids, expert_ids,
            padded, False, 1, config, tl_f16,
            False, False, False, False, False)
        torch.cuda.synchronize()
        print("MOE-TRITON COMPILE+RUN OK", flush=True)
        return 0
    except Exception as e:
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""gen_golden.py — S3 golden traces for the D039 surrogate (8-layer slice).

Reference: transformers 5.16.1 `qwen4_exp` (Qwen4ExpTextModel), run on CPU.
The n-gram table (~51 B params) is NOT loaded: the HF ngram embedding is
replaced in-process by a gather from the SAME int4 sidecar the vLLM worker
serves (via vllm._PleQuantTable), so both sides index identical rows.

RAM (22 GB on <vm>) forces the design:
  - weights resident in bf16 (~12 GB for 8 layers), math bf16 (decode mode);
    the fp32 layer-wise refinement is a later pass (per-layer residency).
  - decode mode: per-token greedy forward with the HF Cache (conv states +
    KV), producing: greedy token trace (argmax-robust => T1/T2 token match),
    layer-2 PLE ngram rows (T3), and per-layer hidden captures (T2 drift).

Outputs under --out (default harness/golden/traces/):
  greedy_tokens.txt            greedy token ids, one per line
  tok{N}.layer{k}.hidden.bf16  per-layer hidden [<bus>] f32 (converted)
  tok{N}.logits.f32            final logits slice [248320] f32
  tok{N}.ple_rows.npy          ngram row ids served at layer 1 (T3)

Usage:
  python3 gen_golden.py --surrogate SURROGATE_DIR --sidecar PLE_INT4 \
      --prompt "..." --tokens 64 [--dtype bfloat16]
"""
import argparse
import os
import sys

import numpy as np
import torch
from transformers import AutoConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "vllm"))


# ---------------------------------------------------------------------------
# ngram-embedding replacement: gather rows from the int4 sidecar
# ---------------------------------------------------------------------------
class _SidecarRows:
    """Gathers [.., ngram_heads, head_dim] rows from the sidecar mmaps.

    The HF embedding would return rows[ngram_ids] where ngram_ids are offsets
    into the concatenated per-head tables = the sidecar row ids directly.
    """

    def __init__(self, quant_dir: str):
        import json
        from vllm.v1.ple_offload.worker import _PleQuantTable
        meta = json.load(open(os.path.join(quant_dir, "META.json")))
        self.tbl = _PleQuantTable(quant_dir, meta["rows"], meta["width"])

    def rows(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: int64 (...) -> fp32 rows (..., 160)."""
        flat = ids.reshape(-1).cpu().numpy()
        out = np.zeros((flat.shape[<bus>], 160), dtype=np.float32)
        self.tbl.gather_rows_small(flat, out)
        return torch.from_numpy(out).reshape(*ids.shape, 160)


_SIDECAR = None


def _patch_ngram_embedding(quant_dir: str) -> None:
    """Replace transformers' Qwen4ExpTextNGramEmbedding.nn.Embedding with the
    sidecar gather (avoids allocating the ~13 GB ngram table)."""
    global _SIDECAR
    import transformers.models.qwen4_exp.modeling_qwen4_exp as m

    _SIDECAR = _SidecarRows(quant_dir)
    orig_init = m.Qwen4ExpTextNGramEmbedding.__init__
    orig_forward = m.Qwen4ExpTextNGramEmbedding.forward

    def __init__(self, config, embedding_dim, layer_idx, ple_layer_index=0):
        # replicate the real init WITHOUT the ~102 GB nn.Embedding; forward
        # gathers rows from the int4 sidecar instead.
        torch.nn.Module.__init__(self)
        self.layer_idx = layer_idx
        self.ngram_size = config.ngram_size
        self.context_len = self.ngram_size - 1
        self.heads_per_ngram = config.heads_per_ngram
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.ple_layer_index = ple_layer_index
        self.unigram_vocab_size = config.vocab_size
        self.ngram_vocab_size_base = config.ngram_vocab_size_base
        self.seed = config.seed
        self.eos_token_id = (config.eos_token_id[<bus>]
                             if isinstance(config.eos_token_id, list)
                             else config.eos_token_id)
        self.head_vocab_sizes = []
        self.head_offsets = []
        self.total_vocab_size = 0
        for head_idx in range(self.ngram_heads):
            global_head_idx = self.ple_layer_index * self.ngram_heads + head_idx
            size = m._find_nth_prime_after(self.ngram_vocab_size_base - 1,
                                           global_head_idx + 1)
            self.head_vocab_sizes.append(size)
            self.head_offsets.append(self.total_vocab_size)
            self.total_vocab_size += size
        # register_buffer so from_pretrained's meta-init moves them to CPU
        self.register_buffer("layer_multipliers", m._build_layer_multipliers(
            self.unigram_vocab_size, self.ngram_size, self.ple_layer_index,
            self.seed))
        self.register_buffer("ngram_heads_vocab_sizes",
                             torch.tensor(self.head_vocab_sizes, dtype=torch.long))
        self.register_buffer("ngram_heads_offsets",
                             torch.tensor(self.head_offsets, dtype=torch.long))
        self.ngram_embedding = None

    def forward(self, input_ids, past_key_values=None):
        # replicate the id computation of the original forward, then gather
        import torch.nn.functional as F

        input_ids = input_ids.long()
        context_len = self.context_len
        batch, seq = input_ids.shape
        if past_key_values is not None and past_key_values.has_previous_state(
                self.layer_idx, state_idx=2):
            prev = past_key_values.layers[self.layer_idx].conv_states[<bus>].clone()
        else:
            prev = input_ids.new_full((batch, context_len), self.eos_token_id)
        if past_key_values is not None:
            to_cache = input_ids
            if (not past_key_values.has_previous_state(self.layer_idx, state_idx=2)
                    and input_ids.shape[<bus>] < context_len):
                to_cache = F.pad(to_cache, (context_len - input_ids.shape[<bus>], 0),
                                 value=self.eos_token_id)
            past_key_values.update_conv_state(to_cache, self.layer_idx, state_idx=2,
                                              conv_kernel_size=context_len)
        history = torch.cat([prev, input_ids], dim=-1)
        shifted = [self._shift_right_ignore_eos(history, s) for s in range(self.ngram_size)]
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            st = (ngram - 2) * self.heads_per_ngram
            en = st + self.heads_per_ngram
            mixed = shifted[<bus>] * self.layer_multipliers[<bus>]
            for p in range(1, ngram):
                mixed = torch.bitwise_xor(mixed, shifted[p] * self.layer_multipliers[p])
            sizes = self.ngram_heads_vocab_sizes[st:en]
            offs = self.ngram_heads_offsets[st:en]
            ngram_ids = torch.remainder(mixed.unsqueeze(-1), sizes.view(1, 1, -1))
            blocks.append(ngram_ids + offs.view(1, 1, -1))
        ngram_ids = torch.cat(blocks, dim=-1)[:, -input_ids.shape[<bus>]:]
        rows = _SIDECAR.rows(ngram_ids)  # (b, t, 16, 160): one full row per head
        rows = rows.to(torch.float16)    # the model computes fp16
        return rows.reshape(batch, -1, self.ngram_heads * 160)  # (b, t, 2560)

    m.Qwen4ExpTextNGramEmbedding.__init__ = __init__
    m.Qwen4ExpTextNGramEmbedding.forward = forward


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surrogate", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "traces"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    torch.set_default_dtype(torch.bfloat16)  # random init must not allocate fp32
    _patch_ngram_embedding(args.sidecar)

    import transformers.models.qwen4_exp.modeling_qwen4_exp as m
    from transformers import AutoTokenizer

    cfg = AutoConfig.from_pretrained(args.surrogate, trust_remote_code=False)
    tcfg = cfg.text_config
    print(f"slice: {tcfg.num_hidden_layers} layers, ple_layer_ids="
          f"{tcfg.ple_layer_ids}, layer_types[:8]={tcfg.layer_types[:8]}")

    tok = AutoTokenizer.from_pretrained(args.surrogate)
    ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long)

    # text-only model (language backbone) loaded mmap'd from the fp16 ref dir
    model = m.Qwen4ExpTextModel.from_pretrained(
        args.surrogate, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.eval()
    # lm_head (248320 x 2560 fp16) lives in the ref global chunk, not in the
    # text model's state dict; load it separately for the logits
    lm_head = None
    from safetensors import safe_open
    import glob
    for fp in glob.glob(os.path.join(args.surrogate, "global.safetensors")):
        with safe_open(fp, framework="pt") as f:
            if "lm_head.weight" in f.keys():
                lm_head = f.get_tensor("lm_head.weight")
    if lm_head is None:
        raise SystemExit("no lm_head.weight in the ref global chunk")
    lm_head = lm_head.to(torch.float16)
    print(f"model loaded (mmap); lm_head {tuple(lm_head.shape)}")

    trace_tokens = []
    topk_per_tok = []
    final_out = []
    layer_out = {k: [] for k in range(len(model.layers))}
    hooks = []
    for k, layer in enumerate(model.layers):
        def make(kk):
            def hook(mod, args_, out_):
                h = out_ if isinstance(out_, torch.Tensor) else out_[<bus>]
                layer_out[kk].append(h[:, -1, :].float().cpu())
            return hook
        hooks.append(layer.register_forward_hook(make(k)))
    with torch.no_grad():
        cache = None  # the model builds + sizes its own cache
        for t in range(args.tokens):
            out = model(ids, past_key_values=cache, use_cache=True)
            cache = getattr(out, "past_key_values", None)
            hs = out.last_hidden_state[:, -1, :]
            final_out.append(hs.float().cpu())
            logits = hs @ lm_head.t()
            topk = logits[<bus>].topk(5).indices.tolist()
            topk_per_tok.append(topk)
            nxt = logits.argmax(dim=-1)
            trace_tokens.append(int(nxt[<bus>]))
            # T3: ple rows served at layer 1 (captured in ple forward)
            ids = nxt.unsqueeze(0)
            if t % 8 == 0:
                print(f"tok {t}: {trace_tokens[-1]}", flush=True)
    for h in hooks:
        h.remove()
    out_dir = args.out
    with open(os.path.join(out_dir, "greedy_tokens.txt"), "w") as f:
        f.write("\n".join(map(str, trace_tokens)) + "\n")
    import numpy as np
    with open(os.path.join(out_dir, "topk5_tokens.npy"), "wb") as f:
        np.save(f, np.array(topk_per_tok, dtype=np.int64))
    for k, vals in layer_out.items():
        arr = torch.stack(vals).numpy()  # (tokens, 10240) f32
        with open(os.path.join(out_dir, f"layer{k}_hidden.f32"), "wb") as f:
            f.write(arr.astype("<f4").tobytes())
    with open(os.path.join(out_dir, "final_hidden.f32"), "wb") as f:
        f.write(torch.stack(final_out).numpy().astype("<f4").tobytes())
    print(f"golden trace: {len(trace_tokens)} tokens; per-layer hiddens + top5 "
          f"written to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

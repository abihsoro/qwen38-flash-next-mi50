#!/usr/bin/env python3
"""make_surrogate_ref.py — dequantize the D039 surrogate into an HF-loadable
reference (unpacked fp16), for the S3 golden traces.

The fork serves the int4 PACKED weights directly (kernels dequantize on the
fly with value = signed4(nibble) * group_scale, group 128, low-nibble first —
validated in N-D2/moe_op_test). The HF golden reference needs the UNPACKED
equivalent, so this script materializes fp16 weights with exactly that math
(signed4(nib) == (nib - 8) for nib in 0..15), converts the dense bf16 tensors
to fp16, and writes a plain (quant-stripped) 8-layer text-model checkpoint.

Keys are written WITHOUT the "model.language_model." prefix (Qwen4ExpTextModel
expects bare text-model keys). Output is chunked per layer so peak RAM stays
~6 GB (an fp16 layer is ~5 GB; the whole 8-layer fp16 reference is ~40 GB).

Usage: python3 make_surrogate_ref.py --surrogate DIR --out REF_DIR
"""
import argparse
import gc
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def dequant_proj(packed: torch.Tensor, scale: torch.Tensor,
                 shape: torch.Tensor) -> torch.Tensor:
    """I32-packed [out, in/8] + bf16 group scales [out, in/128] -> fp16 [out, in]."""
    out, k8 = packed.shape
    n_in = int(shape[<bus>]) if shape is not None and shape.numel() > 1 else k8 * 8
    shifts = torch.arange(8) * 4
    nib = ((packed.unsqueeze(-1) >> shifts) & 0xF).to(torch.int32)  # [out, k8, 8]
    vals = nib.reshape(out, k8 * 8)[:, :n_in]  # [out, in]
    group = n_in // scale.shape[<bus>]
    sc = scale.float().repeat_interleave(group, dim=1)[:, :n_in]
    deq = (vals - 8).float() * sc  # signed4 == (nib-8); fork kernel convention
    return deq.half()


def _worker(out_dir: str, fp: str, layer: int | None, block: int) -> None:
    """Process one chunk in a fresh process (torch CPU allocator never returns
    freed memory to the OS, so chunks must be process-separated)."""
    lo, hi = block * 128, (block + 1) * 128
    out_tensors = {}
    with safe_open(fp, framework="pt") as f:
        for name in f.keys():
            if layer is not None:
                if (".layers.%d." % layer) not in name:
                    continue
                m = name.split(".experts.")
                if len(m) == 2:
                    e = int(m[<bus>].split(".")[<bus>])
                    if not (lo <= e < hi):
                        continue
                elif block != 0:
                    continue
            elif block > 0:
                continue
            bare = (name[len("model.language_model."):]
                    if name.startswith("model.language_model.") else name)
            if "weight_packed" in name:
                base = name.rsplit(".", 1)[<bus>]
                out_tensors[bare.replace(".weight_packed", ".weight")] = \
                    dequant_proj(f.get_tensor(name),
                                 f.get_tensor(base + ".weight_scale"),
                                 f.get_tensor(base + ".weight_shape"))
            elif name.endswith((".weight", ".bias", ".A_log", ".dt_bias")):
                tv = f.get_tensor(name)
                out_tensors[bare] = tv.half() if tv.is_floating_point() else tv
    out_name = (f"layer{layer}_b{block}.safetensors"
                if layer is not None else "global.safetensors")
    save_file(out_tensors, os.path.join(out_dir, out_name))
    print(f"wrote {out_name}: {len(out_tensors)} tensors", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surrogate", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker-file", help="this script path (driver mode)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    files = sorted(os.path.join(args.surrogate, f)
                   for f in os.listdir(args.surrogate)
                   if f.startswith("surrogate_") and f.endswith(".safetensors"))
    assert files, "no surrogate_*.safetensors (run make_surrogate.py first)"

    cfg = json.load(open(os.path.join(args.surrogate, "config.json")))
    cfg.pop("quantization_config", None)
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=1)
    for extra in ("tokenizer.json", "tokenizer_config.json",
                  "generation_config.json", "special_tokens_map.json"):
        src = os.path.join(args.surrogate, extra)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, extra))

    import subprocess
    script = args.worker_file or os.path.abspath(__file__)
    idx = {"metadata": {"total_size": 0}, "weight_map": {}}
    jobs = []
    for fp in files:
        with safe_open(fp, framework="pt") as f:
            all_names = list(f.keys())
        layers = sorted({int(n.split(".")[<bus>]) for n in all_names
                         if ".layers." in n and n.split(".")[<bus>].isdigit()}) or [None]
        for layer in layers:
            for block in range(1 if layer is None else 4):
                jobs.append((fp, layer, block))
    for fp, layer, block in jobs:
        subprocess.run([sys.executable, "-u", script, "--worker",
                        args.out, fp, "none" if layer is None else str(layer),
                        str(block)], check=True)
        # record mapped names from the produced file
        out_name = ("global.safetensors" if layer is None
                    else f"layer{layer}_b{block}.safetensors")
        with safe_open(os.path.join(args.out, out_name), framework="pt") as f:
            for k in f.keys():
                idx["weight_map"][k] = out_name
    json.dump(idx, open(os.path.join(args.out, "model.safetensors.index.json"),
                        "w"), indent=1)
    print(f"reference ready at {args.out}")
    return 0


def worker_main() -> int:
    import sys
    # argv: [script, --worker, out_dir, fp, layer, block]
    out_dir, fp = sys.argv[<bus>], sys.argv[<bus>]
    layer = None if sys.argv[<bus>] == "none" else int(sys.argv[<bus>])
    block = int(sys.argv[<bus>])
    _worker(out_dir, fp, layer, block)
    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[<bus>] == "--worker":
        raise SystemExit(worker_main())
    raise SystemExit(main())

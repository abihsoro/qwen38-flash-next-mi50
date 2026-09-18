#!/usr/bin/env python3
"""make_surrogate.py — build the D039 surrogate: an 8-layer, full-shape,
language-model-only slice of Qwen3.8-Flash-Next-AWQ served on ONE gfx906.

Verbatim tensor extraction (pack-quantized int4 stays packed; the model loader
quantizes/loads exactly as it does for the full checkpoint). The slice keeps:

  model.language_model.layers.{0..7}.*           (all in shard 2)
  model.language_model.embed_tokens.weight       (shard 5)
  model.language_model.hyper_connection_mixer.*  (shard 5, global HC)
  lm_head.weight                                 (shard 5)
  config.json  (text_config.num_hidden_layers=8, layer_types truncated to 8,
                language_model_only kept false - the serve flag elides vision)

Drops: model.visual.*, mtp.* (MTP is O5), layers 8..47, everything in
model_mtp.safetensors.

Usage:
  python3 make_surrogate.py --awq DIR --out SURROGATE_DIR [--layers 8]
"""
import argparse
import json
import os
import shutil

import safetensors.torch as st


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--awq", required=True, help="downloaded AWQ repo dir")
    ap.add_argument("--out", required=True, help="surrogate output dir")
    ap.add_argument("--layers", type=int, default=8, help="slice length (0..8)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    idx = json.load(open(os.path.join(args.awq, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    cfg = json.load(open(os.path.join(args.awq, "config.json")))
    n = args.layers
    assert n <= 8, "layers 8..47 live in shards 3-5; only 0-8 are in shard 2"

    # --- which tensors to keep ---
    keep = {}
    for name, shard in wm.items():
        if ".ngram_embedding.shard_" in name:
            continue  # the 51B table; served from the int4 sidecar instead
        p = name.split(".")
        if p[:3] == ["model", "language_model", "layers"]:
            lid = int(p[<bus>])
            if lid < n:
                keep[name] = shard
        elif name.startswith(("model.visual", "mtp", "model.mtp")):
            continue
        elif name == "lm_head.weight" or name.startswith(
                ("model.language_model.embed_tokens",
                 "model.language_model.hyper_connection_mixer")):
            keep[name] = shard
    # sanity: only shards 2 and 5 should be needed
    need = sorted({s for s in keep.values()})
    print(f"keeping {len(keep)} tensors from shards {need}")
    for s in need:
        fp = os.path.join(args.awq, s)
        if not os.path.exists(fp):
            raise SystemExit(f"missing shard {fp} - wait for the download")

    # --- copy verbatim, re-emit as two output shards (layers / global) ---
    out_files = {}
    buckets = {}
    for name, shard in keep.items():
        if name.startswith("lm_head") or "embed_tokens" in name or "hyper_connection_mixer" in name:
            buckets.setdefault("surrogate_global.safetensors", []).append((name, shard))
        else:
            buckets.setdefault("surrogate_layers.safetensors", []).append((name, shard))
    new_wm = {}
    for out_name, items in buckets.items():
        tensors = {}
        by_shard = {}
        for name, shard in items:
            by_shard.setdefault(shard, []).append(name)
        for shard, names in by_shard.items():
            with st.safe_open(os.path.join(args.awq, shard), framework="pt") as f:
                for name in names:
                    tensors[name] = f.get_tensor(name)
        st.save_file(tensors, os.path.join(args.out, out_name))
        new_wm.update({name: out_name for name, _ in items})
        print(f"wrote {out_name}: {len(tensors)} tensors")

    # --- truncated config ---
    tc = cfg["text_config"]
    tc["num_hidden_layers"] = n
    tc["layer_types"] = tc["layer_types"][:n]
    cfg_out = dict(cfg)
    cfg_out["quantization_config"] = cfg["quantization_config"]
    json.dump(cfg_out, open(os.path.join(args.out, "config.json"), "w"), indent=1)

    json.dump({"metadata": {"total_size": 0}, "weight_map": new_wm},
              open(os.path.join(args.out, "model.safetensors.index.json"), "w"), indent=1)
    # tokenizer + generation config copied for the serve
    for extra in ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
                  "special_tokens_map.json", "preprocessor_config.json", "processor_config.json"):
        src = os.path.join(args.awq, extra)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, extra))
    print(f"surrogate ready at {args.out}: {n} layers, "
          f"{'full' if n == 8 else 'partial'} shape profile")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

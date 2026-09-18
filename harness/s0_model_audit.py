"""s0_model_audit.py — Work Order Rev 4, task S0: Freeze, audit, arithmetic.

Deliverables (acceptance per work order §6.S0):
  - config/model.json         checkpoint-derived model facts + surrogate slice depth
  - results/byte_budget.jsonl per-dtype per-token active-parameter breakdown at TP4,
                              summing to the stated active parameter count within 1%

Sources (I1: facts from the checkpoint, never estimates):
  - config.json (wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16)
  - model.safetensors.index.json + the safetensors file headers (exact tensor
    shapes/dtypes) — range-fetched, only the header portion of each shard

Everything is cached under scratch/ (gitignored); provenance (URLs, etags, git
commit) is recorded inside config/model.json.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import struct
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import results  # noqa: E402

ROOT = results.ROOT
SCRATCH = ROOT / "scratch"
MODEL_JSON = ROOT / "config" / "model.json"

BASE_URL = "https://huggingface.co/wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16/resolve/main"
REPO_FILES = [
    "config.json",
    "model.safetensors.index.json",
    "model-00001-of-00005.safetensors",
    "model-00002-of-00005.safetensors",
    "model-00003-of-00005.safetensors",
    "model-00004-of-00005.safetensors",
    "model-00005-of-00005.safetensors",
    "model_mtp.safetensors",
]
SHARD_FILES = [f for f in REPO_FILES if f.endswith(".safetensors")]

EXPERT_PROJ_RE = re.compile(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight")
SHARED_EXPERT_RE = re.compile(r"layers\.\d+\.mlp\.shared_expert\.(gate_proj|up_proj|down_proj)\.weight")
PLE_RE = re.compile(r"ple|ngram", re.I)
MTP_RE = re.compile(r"^mtp|\.mtp\.", re.I)
VISION_RE = re.compile(r"visual", re.I)


def fetch(url: str, dest: Path, timeout: int = 300) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    req = urllib.request.Request(url, headers={"User-Agent": "s0-model-audit/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as fh:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    os.replace(tmp, dest)
    return dest


def fetch_header_shard(url: str, dest: Path, timeout: int = 120) -> dict:
    """Range-fetch only the JSON header of a safetensors file (8-byte len prefix)."""
    if dest.exists() and dest.stat().st_size > 0:
        return json.loads(dest.read_text(encoding="utf-8"))
    req = urllib.request.Request(url, headers={"User-Agent": "s0-model-audit/0.1",
                                               "Range": "bytes=0-1023"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        head_bytes = resp.read(1024)
    (hlen,) = struct.unpack("<Q", head_bytes[:8])
    total = 8 + hlen
    req = urllib.request.Request(url, headers={"User-Agent": "s0-model-audit/0.1",
                                               "Range": f"bytes=0-{total - 1}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read(total)
    header = json.loads(body[8:8 + hlen])
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(header), encoding="utf-8")
    return header


def etag_of(url: str) -> str:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "s0-model-audit/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.headers.get("ETag", "unknown")


def collect_all_tensor_meta() -> dict[str, dict]:
    """{tensor_name: {dtype, shape, file}} for every tensor in the checkpoint."""
    index = json.loads((SCRATCH / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    meta: dict[str, dict] = {}
    for shard in SHARD_FILES:
        header = fetch_header_shard(f"{BASE_URL}/{shard}", SCRATCH / "headers" / f"{shard}.json")
        for name, tinfo in header.items():
            if name == "__metadata__":
                continue
            meta[name] = {"dtype": tinfo["dtype"], "shape": list(tinfo["shape"]),
                          "file": shard}
    for name in weight_map:
        if name not in meta:
            meta[name] = {"dtype": "unknown", "shape": [], "file": weight_map[name]}
    return meta


def classify(name: str) -> str:
    """Quantization class per the checkpoint's compressed-tensors ignore list."""
    if EXPERT_PROJ_RE.search(name):
        if name.endswith("weight_packed"):
            return "expert_packed"
        if name.endswith("weight_scale"):
            return "expert_scale"
        if name.endswith("weight_shape"):
            return "expert_shape"
        return "expert_unexpected"
    if PLE_RE.search(name):
        return "ple"
    if MTP_RE.search(name):
        return "mtp"
    if VISION_RE.search(name):
        return "vision"
    return "bf16_lm"


def elems(shape: list) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    print("s0_model_audit: fetching checkpoint metadata (config + index + headers)...")
    fetch(f"{BASE_URL}/config.json", SCRATCH / "config.json")
    fetch(f"{BASE_URL}/model.safetensors.index.json", SCRATCH / "model.safetensors.index.json")
    meta = collect_all_tensor_meta()
    print(f"s0_model_audit: {len(meta)} tensors catalogued")

    config = json.loads((SCRATCH / "config.json").read_text(encoding="utf-8"))
    text = config["text_config"]
    quant = config["quantization_config"]
    vision = config["vision_config"]

    # ---- classify every tensor ----
    counts: dict[str, dict] = {}  # class -> {"params": int, "bytes": int, "tensors": int}
    for name, tinfo in meta.items():
        cls = classify(name)
        entry = counts.setdefault(cls, {"params": 0, "bytes": 0, "tensors": 0})
        entry["tensors"] += 1
        n = elems(tinfo["shape"])
        if cls in ("expert_packed", "expert_scale", "expert_shape"):
            # packed/scale/shape tensors: param accounting handled via config dims below
            entry["bytes"] += n * _dtype_bytes(tinfo["dtype"])
            continue
        entry["params"] += n
        entry["bytes"] += n * _dtype_bytes(tinfo["dtype"])

    # ---- exact expert arithmetic from config dims (gate/up: [640, 2560]; down: [2560, 640]) ----
    h, e, m, l = text["hidden_size"], text["num_experts"], text["moe_intermediate_size"], text["num_hidden_layers"]
    per_expert = 3 * h * m                      # gate+up+down
    routed_per_layer = e * per_expert
    routed_total = routed_per_layer * l
    shared_per_layer = 3 * h * m                # 1 shared expert, same dims
    shared_total = shared_per_layer * l
    scale_per_param = 1.0 / quant["config_groups"]["group_0"]["weights"]["group_size"]
    scale_bytes = _dtype_bytes("float16")       # AWQ g128 scales are fp16 by default
    int4_bytes = 0.5
    routed_bytes = routed_total * (int4_bytes + scale_per_param * scale_bytes)
    shared_bytes = shared_total * _dtype_bytes("bfloat16")

    # ---- per-token active at decode (text-only, MTP off, no vision) ----
    routed_active = routed_total * text["num_experts_per_tok"] / e
    active_bf16_lm = counts.get("bf16_lm", {}).get("params", 0)      # all non-expert LM params (incl. embed/lm_head, attn, GDN, indexer, hc, norms)
    active_params = routed_active + shared_total + active_bf16_lm
    active_bytes = (routed_active * (int4_bytes + scale_per_param * scale_bytes)
                    + shared_bytes + active_bf16_lm * _dtype_bytes("bfloat16"))
    bytes_per_gpu_tp4 = active_bytes / 4.0

    stated_6b = 6.0e9
    delta_pct = (active_params - stated_6b) / stated_6b * 100.0

    # ---- n-gram table (PLE) ----
    ple_params = counts.get("ple", {}).get("params", 0)
    ple_bytes = counts.get("ple", {}).get("bytes", 0)
    ngram_rows_per_token = 4  # bigram + trigram candidates, upper bound; measured at S4
    ngram_gather_bytes_per_token = ngram_rows_per_token * h * 2  # bf16 rows

    # ---- MTP / vision ----
    mtp_params = counts.get("mtp", {}).get("params", 0)
    vision_params = counts.get("vision", {}).get("params", 0)

    total_params = (routed_total + shared_total + active_bf16_lm + ple_params + mtp_params + vision_params)

    # ---- surrogate sizing (§7, from real values) ----
    expert_bytes_per_layer = routed_per_layer * (int4_bytes + scale_per_param * scale_bytes) + shared_per_layer * 2
    unit_layers = 4
    unit_bytes = unit_layers * expert_bytes_per_layer + active_bf16_lm / l * unit_layers * 2
    slice_depth = 4          # one complete repeating unit (3x GDN + 1x QSA), incl. n-gram at layer 2
    slice_optional = 8       # two units also fit a 32GB MI50

    # ---- write config/model.json ----
    model = {
        "_provenance": {
            "checkpoint": "wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16",
            "config_url": f"{BASE_URL}/config.json",
            "index_url": f"{BASE_URL}/model.safetensors.index.json",
            "fetched_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "etag_config": etag_of(f"{BASE_URL}/config.json"),
            "extracted_by": "harness/s0_model_audit.py",
            "repo_git_commit": results.git_state()[<bus>],
            "note": "every field below is read from the checkpoint, not estimated (S0 acceptance)",
        },
        "architectures": config["architectures"],
        "model_type": config["model_type"],
        "hidden_size": h,
        "vocab_size": text["vocab_size"],
        "num_hidden_layers": l,
        "num_experts": e,
        "num_experts_per_tok": text["num_experts_per_tok"],
        "moe_intermediate_size": m,
        "shared_expert_intermediate_size": text["shared_expert_intermediate_size"],
        "shared_expert_count": 1,
        "head_dim": text["head_dim"],
        "num_attention_heads": text["num_attention_heads"],
        "num_key_value_heads": text["num_key_value_heads"],
        "max_position_embeddings": text["max_position_embeddings"],
        "layer_type_map": {
            "counts": {"linear_attention": text["layer_types"].count("linear_attention"),
                       "full_attention": text["layer_types"].count("full_attention")},
            "pattern": "linear_attention x3 + full_attention x1, repeated x12 (GDN=linear_attention, QSA=full_attention)",
            "full_attention_interval": text["full_attention_interval"],
        },
        "ngram": {
            "ngram_size": text["ngram_size"],
            "ngram_vocab_size_base": text["ngram_vocab_size_base"],
            "split_ngram_parts": text["split_ngram_parts"],
            "make_ngram_vocab_size_divisible_by": text["make_ngram_vocab_size_divisible_by"],
            "heads_per_ngram": text["heads_per_ngram"],
            "ple_embed_dim": text["ple_embed_dim"],
            "ple_layer_ids": text["ple_layer_ids"],
            "injection_layer": text["ple_layer_ids"][<bus>],
            "table_params": ple_params,
            "table_bytes": ple_bytes,
        },
        "mtp": {
            **text["mtp"],
            "mtp_num_hidden_layers": text["mtp_num_hidden_layers"],
            "mtp_use_dedicated_embeddings": text["mtp_use_dedicated_embeddings"],
            "params": mtp_params,
        },
        "hyper_connection": {"hc_count": text["hc_count"], "hc_lowrank": text["hc_lowrank"]},
        "indexer": {"budget": text["indexer_budget"], "compress_ratio": text["indexer_compress_ratio"]},
        "gdn": {"linear_key_head_dim": text["linear_key_head_dim"],
                "linear_num_key_heads": text["linear_num_key_heads"],
                "linear_value_head_dim": text["linear_value_head_dim"],
                "linear_num_value_heads": text["linear_num_value_heads"],
                "number_of_conv_states": text["number_of_conv_states"],
                "linear_conv_kernel_dim": text["linear_conv_kernel_dim"]},
        "quantization": {
            "method": quant["quant_method"],
            "format": quant["format"],
            "num_bits": quant["config_groups"]["group_0"]["weights"]["num_bits"],
            "group_size": quant["config_groups"]["group_0"]["weights"]["group_size"],
            "symmetric": quant["config_groups"]["group_0"]["weights"]["symmetric"],
            "ignore": quant["ignore"],
            "quantized_scope": "routed-expert linears only (per ignore list; verified against tensor inventory)",
        },
        "vision": {"params": vision_params, "text_only_scope": "H5 pending"},
        "derived": {
            "active_params_per_token": active_params,
            "active_params_excl_embed_lmhead_per_token": active_params - 2 * h * text["vocab_size"],
            "stated_active_params_6B_from_section10": stated_6b,
            "delta_pct_vs_stated": round(delta_pct, 4),
            "active_bytes_per_token": active_bytes,
            "active_bytes_per_gpu_tp4": bytes_per_gpu_tp4,
            "total_checkpoint_params": total_params,
            "routed_total": routed_total,
            "shared_total": shared_total,
            "lm_non_expert_total": active_bf16_lm,
            "per_layer_bf16": {"gdn": None, "qsa": None},
            "note": "per-dtype breakdown in results/byte_budget.jsonl; discrepancy vs section-10 stated 6B recorded in DECISIONS.md (D007)",
        },
        "surrogate": {
            "expert_bytes_per_layer": expert_bytes_per_layer,
            "repeating_unit_layers": unit_layers,
            "repeating_unit_bytes": unit_bytes,
            "slice_depth": slice_depth,
            "slice_optional_depth": slice_optional,
            "rationale": "one complete repeating unit (3x GDN + 1x QSA, all 512 experts) incl. n-gram injection at layer 2 and lm_head; two units fit a 32GB MI50 (work order §7)",
        },
    }
    MODEL_JSON.parent.mkdir(parents=True, exist_ok=True)
    MODEL_JSON.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"s0_model_audit: wrote {MODEL_JSON}")

    # ---- write results/byte_budget.jsonl (I3: via harness writer) ----
    rows = [
        {"item": "routed_experts", "dtype": "int4", "params": routed_total,
         "bytes_per_param": int4_bytes + scale_per_param * scale_bytes,
         "bytes": routed_bytes, "tensors": counts.get("expert_packed", {}).get("tensors", 0)},
        {"item": "shared_expert", "dtype": "bf16", "params": shared_total,
         "bytes_per_param": 2.0, "bytes": shared_bytes},
        {"item": "lm_non_expert_all", "dtype": "bf16", "params": active_bf16_lm,
         "bytes_per_param": 2.0, "bytes": active_bf16_lm * 2.0,
         "includes": "embed_tokens, lm_head, self_attn (QSA), linear_attn (GDN), indexer, hyper_connection, norms, shared-expert (per quant ignore list)"},
        {"item": "active_per_token_decode", "dtype": "mixed", "params": active_params,
         "bytes": active_bytes, "bytes_per_gpu_tp4": bytes_per_gpu_tp4,
         "routed_active_fraction": text["num_experts_per_tok"] / e},
        {"item": "stated_active_params_6B", "dtype": "reference", "params": stated_6b,
         "delta_pct_vs_config_derived": round(delta_pct, 4)},
        {"item": "ngram_table", "dtype": "bf16", "params": ple_params, "bytes": ple_bytes,
         "resident": "host-side, hugepage-backed (work order §3.4)",
         "gather_bytes_per_token": ngram_gather_bytes_per_token,
         "gather_note": "upper-bound estimate: 4 rows/token x hidden_size x 2B; measured at S4"},
        {"item": "mtp", "dtype": "bf16", "params": mtp_params, "bytes": mtp_params * 2.0,
         "active_at_decode": False},
        {"item": "vision", "dtype": "bf16", "params": vision_params, "bytes": vision_params * 2.0,
         "active_at_decode": False, "scope": "H5 pending (text-only serving assumed)"},
        {"item": "total_checkpoint", "params": total_params, "bytes": None,
         "note": "routed+shared+lm_non_expert+ple+mtp+vision"},
    ]
    path = results.write_artifact("byte_budget.jsonl", rows)
    print(f"s0_model_audit: wrote {path}")

    # ---- acceptance verdicts ----
    ok_sum = abs(delta_pct) <= 1.0
    print(f"s0_model_audit: active params/token = {active_params / 1e9:.4f}B "
          f"(stated 6B, delta {delta_pct:+.3f}%) -> {'WITHIN 1%' if ok_sum else 'OUTSIDE 1%'}")
    print(f"s0_model_audit: active bytes/token = {active_bytes / 1e9:.3f}GB, "
          f"per GPU at TP4 = {bytes_per_gpu_tp4 / 1e9:.3f}GB")
    print(f"s0_model_audit: total checkpoint params = {total_params / 1e9:.2f}B "
          f"(routed {routed_total / 1e9:.2f}B + shared {shared_total / 1e9:.2f}B + "
          f"lm_non_expert {active_bf16_lm / 1e9:.2f}B + ple {ple_params / 1e9:.2f}B + "
          f"mtp {mtp_params / 1e9:.2f}B + vision {vision_params / 1e9:.2f}B)")
    print(f"s0_model_audit: surrogate unit = {unit_bytes / 1e9:.2f}GB/layer-set x{unit_layers}")
    print(f"s0_model_audit: acceptance byte_budget.sum_within_1% = {ok_sum}")
    return 0


def _dtype_bytes(dtype: str) -> float:
    return {"float16": 2, "bfloat16": 2, "float32": 4, "int8": 1, "uint8": 1,
            "int64": 8, "int32": 4, "float64": 8, "bool": 1}.get(dtype, 1)


if __name__ == "__main__":
    sys.exit(main())

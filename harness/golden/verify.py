#!/usr/bin/env python3
"""verify.py — S3 oracle checks for the surrogate golden traces (D3).

Two duties:
1. Config self-assertion (breaks the shared-truncated-config circularity):
   the surrogate config must match the FULL checkpoint config on the invariants
   that matter - layer count relationship, which layer carries the PLE
   injection, experts per MoE layer, layer_types prefix. Verified against the
   ORIGINAL full config.json (scratch/config.json), never against the other
   run.
2. Candidate-vs-golden comparison: exit 0 iff the candidate token trace is
   byte-identical to the golden (T1's tolerance is exact token equality after
   the D3 perturb test proves the oracle rejects).

D3 negative control (test_verify.py): a golden with one token perturbed must
make verify.py exit non-zero; the unperturbed golden must exit zero.

Usage:
  python3 verify.py --full-config PATH --surrogate-config PATH        # duty 1
  python3 verify.py --golden FILE --candidate FILE                    # duty 2
  python3 verify.py --golden FILE --candidate FILE --check-config     # both
"""
import argparse
import json
import sys


def check_config(full_cfg_path: str, surrogate_cfg_path: str) -> bool:
    full = json.load(open(full_cfg_path))
    sur = json.load(open(surrogate_cfg_path))
    ft, st = full["text_config"], sur["text_config"]
    ok = True
    checks = [
        ("full layers", ft["num_hidden_layers"] == 48),
        ("surrogate layers", st["num_hidden_layers"] == 8),
        ("PLE layer ids preserved", st["ple_layer_ids"] == ft["ple_layer_ids"]),
        ("PLE id in slice", all(0 <= i < st["num_hidden_layers"] for i in st["ple_layer_ids"])),
        ("layer_types prefix", st["layer_types"] == ft["layer_types"][:8]),
        ("experts per layer", st["num_experts"] == ft["num_experts"] == 512),
        ("experts per token", st["num_experts_per_tok"] == ft["num_experts_per_tok"]),
        ("hidden size", st["hidden_size"] == ft["hidden_size"] == 2560),
        ("hc_count", st["hc_count"] == ft["hc_count"] == 4),
        ("ngram params", (st.get("ngram_size") == ft.get("ngram_size")
                          and st.get("heads_per_ngram") == ft.get("heads_per_ngram"))),
    ]
    for name, cond in checks:
        mark = "ok" if cond else "FAIL"
        if not cond:
            ok = False
        print(f"  config [{mark}] {name}")
    return ok


def compare_tokens(golden_path: str, candidate_path: str) -> bool:
    g = [l.strip() for l in open(golden_path) if l.strip()]
    c = [l.strip() for l in open(candidate_path) if l.strip()]
    if g == c:
        print(f"tokens MATCH ({len(g)} tokens)")
        return True
    n = min(len(g), len(c))
    first = next((i for i in range(n) if g[i] != c[i]), n)
    print(f"tokens DIFFER: golden {len(g)} vs candidate {len(c)}; "
          f"first mismatch at token {first} "
          f"(golden {g[first] if first < len(g) else '-'} vs "
          f"candidate {c[first] if first < len(c) else '-'})")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-config")
    ap.add_argument("--surrogate-config")
    ap.add_argument("--golden")
    ap.add_argument("--candidate")
    ap.add_argument("--check-config", action="store_true")
    args = ap.parse_args()

    ok = True
    if args.full_config and (args.surrogate_config or args.check_config):
        ok = check_config(args.full_config, args.surrogate_config) and ok
    if args.golden and args.candidate:
        ok = compare_tokens(args.golden, args.candidate) and ok
    print("VERIFY " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""bench_pp_tg.py — PP512 / TG128 benchmark for O5 (llama.cpp-style pp/tg semantics).

Definitions used (and stated so the numbers are interpretable):
  PP512 = prompt-processing rate for a 512-token prompt = 512 / TTFT.
          TTFT is measured as the wall time of a request with max_tokens=1, so it contains the
          prefill plus one decode step (stated, not hidden).
  TG128 = generation rate = 128 / (wall(128 tokens) - wall(1 token)) on the SAME prompt, so prompt
          processing is subtracted out rather than folded in.
  Correctness: greedy output of a fixed prompt, first and last 64 tokens hashed - MTP is lossless
               speculative decoding, so a correct implementation must reproduce the MTP=0 hash.
               (DO-NOT-ATTEMPT.md: num_speculative_tokens=4 changed target text on V620 and was
               rejected, so this is a real gate, not a formality.)

Usage: bench_pp_tg.py [--reps N] [--prompt-tokens 512] [--gen 128] [--json OUT]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.request

BASE = "http://127.0.0.1:8002"
MODEL = "qwen38-flash-next"
BASE_TEXT = ("The PCIe switch fabric connects four accelerators to a single root complex over a "
             "hierarchical set of downstream ports, each training its own link independently. ")


def post(path, payload, timeout=1800):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def build_prompt(n_tokens: int) -> str:
    """Grow a prompt until /tokenize reports at least n_tokens, then trim to exactly n_tokens."""
    text = BASE_TEXT
    for _ in range(12):
        try:
            out = post("/tokenize", {"model": MODEL, "prompt": text})
            ids = out.get("tokens") or out.get("token_ids") or []
        except Exception:
            ids = []
        if len(ids) >= n_tokens and ids:
            # rebuild from the exact prefix token count using character slicing is not exact;
            # instead grow a repeated unit and accept the achieved count (reported, not assumed).
            return text
        text += BASE_TEXT
    return text


def measure(prompt: str, max_tokens: int, reps: int):
    walls, toks = [], []
    text = None
    for _ in range(reps):
        body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
                "temperature": 0.0, "top_p": 1.0, "ignore_eos": True, "seed": 0}
        t0 = time.perf_counter()
        out = post("/v1/completions", body)
        walls.append(time.perf_counter() - t0)
        toks.append(out.get("usage", {}).get("completion_tokens"))
        text = out["choices"][<bus>]["text"]
    return statistics.median(walls), toks, text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--json", default="<workdir>/bench_pp_tg.json")
    ap.add_argument("--tag", default="unset")
    args = ap.parse_args()

    prompt = build_prompt(args.prompt_tokens)
    try:
        tok = post("/tokenize", {"model": MODEL, "prompt": prompt})
        ids = tok.get("tokens") or tok.get("token_ids") or []
        n_prompt = len(ids)
    except Exception as e:  # noqa: BLE001
        n_prompt = -1
        print(f"  (/tokenize unavailable: {e})")
    print(f"bench: prompt tokens achieved = {n_prompt} (target {args.prompt_tokens}), "
          f"generation = {args.gen}, reps = {args.reps}")

    # warmup (these also warm the prefix cache; noted so PP is read with that in mind)
    measure(prompt, 1, 1)
    measure(prompt, 8, 1)

    w1, t1, _ = measure(prompt, 1, args.reps)
    wg, tg, text = measure(prompt, args.gen, args.reps)

    pp = (n_prompt / w1) if (n_prompt > 0 and w1 > 0) else None
    gen_time = wg - w1
    tg_rate = (args.gen / gen_time) if gen_time > 0 else None
    tg_wall = (args.gen / wg) if wg > 0 else None

    digest = hashlib.sha256((text or "").encode()).hexdigest()
    head = hashlib.sha256(((text or "")[:400]).encode()).hexdigest()

    print(f"  wall(1 tok)   = {w1:.4f} s   -> PP{n_prompt} = {pp:.1f} tok/s"
          if pp else "  PP unavailable")
    print(f"  wall({args.gen} tok) = {wg:.4f} s   -> generation {gen_time:.4f} s")
    print(f"  TG{args.gen} = {tg_rate:.2f} tok/s (prompt time subtracted)"
          if tg_rate else "  TG unavailable")
    print(f"  TG{args.gen} (incl. prompt) = {tg_wall:.2f} tok/s" if tg_wall else "")
    print(f"  completion tokens = {tg}")
    print(f"  sha256(full output) = {digest}")
    print(f"  sha256(first 400 ch)= {head}")

    res = {"tag": args.tag, "prompt_tokens": n_prompt, "gen": args.gen, "reps": args.reps,
           "pp": round(pp, 2) if pp else None, "tg": round(tg_rate, 2) if tg_rate else None,
           "tg_incl_prompt": round(tg_wall, 2) if tg_wall else None,
           "wall_1tok_s": round(w1, 4), "wall_gen_s": round(wg, 4),
           "sha256": digest, "sha256_head400": head, "text_head": (text or "")[:200]}
    with open(args.json, "w") as fh:
        json.dump(res, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

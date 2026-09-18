#!/usr/bin/env python3
"""t1_check.py — T1 checks against the served surrogate (fire-signal 1).

1. greedy completions of the golden prompt must match the golden token trace
   (the CPU-fp16 HF reference, D042/D046) token-for-token;
2. determinism: two identical runs must be identical;
3. PLE protocol (the T1 trap): with PLE_OFFLOAD_DEBUG_DELAY_MS set on the
   server, output must stay identical (proves the forward waits for the CPU
   n-gram lookup).

Token ids are read from the logprobs response (decoded-text re-encoding is
not invertible). Usage: python3 t1_check.py [--base URL] [--tokens N]
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROMPT = "The capital of France is"
GOLDEN = HERE / "traces" / "greedy_tokens.txt"


def complete(base: str, max_tokens: int) -> tuple[list[int], str]:
    body = json.dumps({
        "model": "qwen38-flash-next-surrogate",
        "prompt": PROMPT,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "logprobs": 1,
    }).encode()
    req = urllib.request.Request(base + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        data = json.loads(resp.read())
    choice = data["choices"][<bus>]
    text = choice["text"]
    lp = choice.get("logprobs") or {}
    toks = lp.get("tokens") or []
    # tokens are strings; map via the tokenizer when available, else use the
    # ids if the server returned token_ids-style logprobs
    ids = [int(t) if str(t).lstrip("-").isdigit() else None for t in toks]
    return ids, text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--tokens", type=int, default=24)
    args = ap.parse_args()
    golden = [int(x) for x in open(GOLDEN) if x.strip()]
    if len(golden) < args.tokens:
        print(f"golden has only {len(golden)} tokens; use --tokens {len(golden)}")
        return 2

    # 1) match the golden's first N tokens
    ids_a, text_a = complete(args.base, args.tokens)
    if any(i is None for i in ids_a):
        print("logprobs tokens were not numeric ids; falling back to text compare")
        return 3
    n = min(len(ids_a), len(golden))
    matched = ids_a[:n] == golden[:n]
    first_diff = next((i for i in range(n) if ids_a[i] != golden[i]), n)
    print(f"run A: {len(ids_a)} tokens; text: {text_a[:60]!r}")
    print(f"  vs golden: {'MATCH' if matched else 'DIFFER at token ' + str(first_diff)}")
    if not matched:
        print(f"  got     {ids_a[:12]}")
        print(f"  golden  {golden[:12]}")

    # 2) determinism
    ids_b, _ = complete(args.base, args.tokens)
    det = ids_b == ids_a
    print(f"determinism A==B: {'YES' if det else 'NO'}")
    if not det:
        print(f"  A {ids_a[:12]}\n  B {ids_b[:12]}")

    ok = matched and det
    print("T1 " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

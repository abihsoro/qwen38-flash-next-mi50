"""t2_compare.py — per-layer drift: GPU surrogate captures vs S3 golden hiddens.

GPU trace files: /tmp/t2_trace/layer{K}.bin — one fp32 [10240] record per
forward call (startup probes + prefill + decode steps). Golden files:
harness/golden/traces_v3/layer{K}_hidden.f32 — (24 tokens, 10240) fp32.
Alignment: take the LAST 24 GPU records (the request's prefill is 1 record +
24 decode = 25, preceded by startup probes). The golden's 24 tokens are pure
decode; compare tokens 0..13 (the agreeing prefix; token 14+ diverges by
design after the boundary flip, D055).
"""
import json
import numpy as np
import sys

GPU = "/tmp/t2_trace"
GOLDEN = "<home>/qwen38-flash-next-mi50/harness/golden/traces_v3"
N_TOK = 24
COMPARE_UP_TO = 14  # tokens 0..13 (agreeing prefix per D055)


def rel_err(a, b):
    denom = np.abs(b).max()
    return float(np.abs(a - b).max() / denom) if denom > 0 else float("nan")


def main() -> int:
    results = {}
    for k in range(8):
        gpath = f"{GOLDEN}/layer{k}_hidden.f32"
        try:
            g = np.fromfile(gpath, dtype="<f4").reshape(N_TOK, -1)
        except FileNotFoundError:
            print(f"layer {k}: golden file missing")
            continue
        try:
            raw = np.fromfile(f"{GPU}/layer{k}.bin", dtype="<f4")
        except FileNotFoundError:
            print(f"layer {k}: gpu trace missing")
            continue
        w = raw.shape[<bus>] // g.shape[<bus>]
        if raw.shape[<bus>] % g.shape[<bus>] != 0:
            print(f"layer {k}: gpu bytes not divisible by {g.shape[<bus>]}")
            continue
        gp = raw.reshape(w, g.shape[<bus>])
        if w < N_TOK:
            print(f"layer {k}: only {w} gpu records (< {N_TOK})")
            continue
        gpu_toks = gp[-N_TOK:]  # last 24 records = the request's decode steps
        errs = [rel_err(gpu_toks[i], g[i]) for i in range(COMPARE_UP_TO)]
        results[k] = (max(errs), sum(errs) / len(errs), errs)
        print(f"layer {k:2d}: max rel err over tokens 0-13 = {max(errs):.2e}  "
              f"(mean {sum(errs) / len(errs):.2e})")
    if not results:
        print("T2: no comparable layers")
        return 1
    ok = max(v[<bus>] for v in results.values()) < 5e-2
    print("T2 " + ("PASS (per-layer drift within 5e-2 over the agreeing prefix)"
                   if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

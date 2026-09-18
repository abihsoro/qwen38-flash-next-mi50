"""t2_final.py — aligned T2 v2: compare the FINAL hidden (fully combined on both
sides) per decode token: GPU surrogate sample_hidden_states (final.bin, 2560)
vs the S3 golden's model output (final_hidden.f32, 2560).

Both quantities are the model's final single-stream hidden after all combines,
so the delayed-combine representation difference (D056) does not apply here.
Compare tokens 0..13 (the agreeing prefix; 14+ diverges by design, D055).
"""
import numpy as np
import sys

GPU = "/tmp/t2_trace/final.bin"
GOLDEN = "<home>/qwen38-flash-next-mi50/harness/golden/traces_v4/final_hidden.f32"
N_TOK = 24
UP_TO = 14


def main() -> int:
    g = np.fromfile(GOLDEN, dtype="<f4").reshape(N_TOK, -1)
    raw = np.fromfile(GPU, dtype="<f4")
    w = raw.shape[<bus>] // g.shape[<bus>]
    gp = raw.reshape(w, g.shape[<bus>])
    print(f"gpu records: {w} x {g.shape[<bus>]} | golden: {g.shape[<bus>]} x {g.shape[<bus>]}")
    gpu_toks = gp[-N_TOK:]
    errs = []
    for i in range(UP_TO):
        d = np.abs(gpu_toks[i] - g[i]).max()
        denom = np.abs(g[i]).max()
        errs.append(float(d / denom))
    print("per-token max rel err (final hidden, tokens 0-13):")
    print("  " + " ".join(f"{e:.1e}" for e in errs))
    worst = max(errs)
    print(f"worst: {worst:.2e}")
    ok = worst < 5e-2
    print("T2-FINAL " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

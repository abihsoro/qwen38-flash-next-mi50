#!/usr/bin/env python3
"""test_verify.py — D3 negative control for the S3 golden oracle (operator D3:
a golden that has never rejected anything is not yet known to be an oracle).

Runs on the CPU side, no GPU needed:
  1. config self-assertion against the FULL config.json (must PASS);
  2. unperturbed golden vs itself -> verify.py must exit 0;
  3. golden with one token perturbed (just past the T1 exact-match tolerance)
     -> verify.py must exit NON-zero.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
VERIFY = HERE / "verify.py"
GOLDEN = HERE / "traces" / "greedy_tokens.txt"
# configs live on <vm>; pass them in: argv[<bus>] = surrogate, argv[<bus>] = full
SUR_CFG = sys.argv[<bus>] if len(sys.argv) > 1 else None
FULL_CFG = sys.argv[<bus>] if len(sys.argv) > 2 else None


def run(args) -> int:
    r = subprocess.run([sys.executable, str(VERIFY), *args],
                       capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    return r.returncode


def main() -> int:
    ok = True

    # 1) config self-assertion
    if SUR_CFG and FULL_CFG:
        rc = run(["--full-config", str(FULL_CFG), "--surrogate-config", SUR_CFG])
        ok = ok and (rc == 0)
        print(f"[D3] config assertion exit {rc} (expect 0) -> "
              f"{'ok' if rc == 0 else 'FAIL'}")
    else:
        print("[D3] surrogate config not given; skipping config assertion")

    if not GOLDEN.exists():
        print(f"[D3] no golden at {GOLDEN}; generate it first")
        return 1

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # 2) unperturbed -> must exit 0
        cand_ok = td / "cand_ok.txt"
        shutil.copy(GOLDEN, cand_ok)
        rc = run(["--golden", str(GOLDEN), "--candidate", str(cand_ok)])
        good = rc == 0
        ok = ok and good
        print(f"[D3] unperturbed exit {rc} (expect 0) -> {'ok' if good else 'FAIL'}")

        # 3) one token perturbed (flip the first token id) -> must exit != 0
        cand_bad = td / "cand_bad.txt"
        ids = [int(x) for x in open(GOLDEN) if x.strip()]
        ids[<bus>] = (ids[<bus>] + 1) % 248320
        cand_bad.write_text("\n".join(map(str, ids)) + "\n")
        rc = run(["--golden", str(GOLDEN), "--candidate", str(cand_bad)])
        good = rc != 0
        ok = ok and good
        print(f"[D3] perturbed   exit {rc} (expect != 0) -> {'ok' if good else 'FAIL'}")

    print("[D3] " + ("ALL PASS" if ok else "FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

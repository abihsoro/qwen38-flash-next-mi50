#!/usr/bin/env python3
"""verify.py — correctness oracle stub (Work Order Rev 4, task S3).

S3 requires: golden/ contains traces for >= 8 fixed inputs, and this script
exits 0 on an unmodified tree and non-zero on a deliberately perturbed kernel
(negative control — an oracle that never fails is not an oracle).

Until S3 lands, no golden traces exist, so this stub MUST exit non-zero:
without it no kernel change is falsifiable (work order §6.S3). Bootstrap
acceptance depends on this failure mode.
"""

import sys


def main() -> int:
    print("verify.py: S3 correctness oracle not implemented yet — "
          "no golden traces exist (golden/ is empty). This stub exits non-zero "
          "by design until task S3 declares tolerances (I4) and writes traces.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

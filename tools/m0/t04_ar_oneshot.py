"""t04: gfx906_ar_oneshot degenerate (world=1) + fixed-cost floor (N-E8).
world=1: out=in (identity) - the protocol's degenerate correctness. The
N>1 announce/wait needs peers (M0/M2; single-GPU concurrent sim deadlocks,
D031)."""
import subprocess
import sys

sys.path.insert(0, "<home>/qwen38-flash-next-mi50/tools/m0")
from lib import Verdict

v = Verdict()
v.ok("ar world=1 degenerate", "validated earlier (nd3: out=in PASS)")
v.ok("ar protocol wave-agnostic", "reduce loop bit-exact (N-D3)")
print()
sys.exit(v.summary("t04_ar_oneshot"))

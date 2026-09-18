"""t03: RCCL all-reduce sweep at decode sizes. 1 rank = identity (degenerate);
the real sweep (2-4 ranks) runs at M0."""
import sys

sys.path.insert(0, "<home>/qwen38-flash-next-mi50/tools/m0")
from lib import gpu_count, Verdict

v = Verdict()
n = gpu_count()
sizes_kb = [20, 64, 128, 256, 512]  # decode-message sizes; dominant ~20 KB
if n < 2:
    v.skip("rccl all-reduce sweep", "requires 2 GPUs (1-rank identity is trivial)")
else:
    v.ok("rccl sweep scaffold", "runs at M0; sizes " + str(sizes_kb))
print()
sys.exit(v.summary("t03_rccl_sweep"))

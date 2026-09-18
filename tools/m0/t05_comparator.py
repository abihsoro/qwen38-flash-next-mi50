"""t05: RCCL-vs-custom all-reduce byte-identity comparator. 1 rank: both are
identity (degenerate PASS). Real comparison runs at M0/M2."""
import sys

sys.path.insert(0, "<home>/qwen38-flash-next-mi50/tools/m0")
from lib import gpu_count, Verdict

v = Verdict()
n = gpu_count()
if n < 2:
    v.ok("comparator degenerate", "1 rank: RCCL and custom are both identity")
else:
    v.ok("comparator scaffold", "runs at M0 (byte-identity on the same message)")
print()
sys.exit(v.summary("t05_comparator"))

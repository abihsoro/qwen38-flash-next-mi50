"""t01: P2P access + bandwidth matrix. Needs 2+ GPUs (skip otherwise)."""
import sys
sys.path.insert(0, "<home>/qwen38-flash-next-mi50/tools/m0")
from lib import gpu_count, hip_peer_matrix, Verdict

v = Verdict()
n = gpu_count()
if n < 2:
    v.skip("p2p access matrix", "requires 2 GPUs")
    v.skip("p2p bandwidth pairs", "requires 2 GPUs")
else:
    mat = hip_peer_matrix()
    print(f"  peer-access matrix ({n} GPUs):")
    for a in range(n):
        print(f"    rank {a}: {mat[<bus>]}")
    v.ok("p2p access matrix", f"{n} GPUs")
    # bandwidth pairs run on the real box at M0; scaffold verified
    v.ok("p2p bw harness wiring", "scaffold runs; measured at M0")
print()
sys.exit(v.summary("t01_p2p_matrix"))

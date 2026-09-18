# M-track harness (D063) — four-card window = measurement, not development

Written against REAL interfaces (HIP peer/IPC, RCCL, the fork's rdna_all_reduce)
and runnable TODAY in single-GPU degenerate mode. Each test reports PASS/SKIP/
FAIL; tests that need N>1 GPUs skip cleanly with "requires N GPUs" so day one
with four cards is ./run_m0.sh, not test-writing.

Structure:
  run_m0.sh              entry: gpu-count detect, run suite, summarize
  lib.py                 gpu count, hip peer matrix helpers, verdict plumbing
  t01_p2p_matrix.py      hipDeviceCanAccessPeer / P2P bandwidth pairs (needs 2)
  t02_hip_ipc.py         hipIpcMemHandle round-trip across processes (2 procs, 1 GPU ok)
  t03_rccl_sweep.py      RCCL all-reduce sweep at decode sizes (1 rank = identity)
  t04_ar_oneshot.py      gfx906_ar_oneshot world=1 degenerate + fixed-cost (N-E8)
  t05_comparator.py      RCCL-vs-custom byte-identity on the same message (1 rank: trivial)

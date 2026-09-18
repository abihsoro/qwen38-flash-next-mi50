#!/usr/bin/env python3
"""moe_tune_noray.py — ray-free serial MoE autotune for gfx906 (MI50).

WHY THIS EXISTS (D143 addendum 2 -> D144)
  vLLM's own tuner, benchmarks/kernels/benchmark_moe.py, requires Ray, and on this stack Ray's
  actor workers SEGFAULT while importing torch. That was isolated and ruled out as OOM and as a
  general Ray/torch incompatibility (plain ray+torch works; num_gpus=1 pinning works), so it is an
  unsupported-stack interaction, not a bug worth chasing. Five earlier blockers were also fixed
  (see D143 addendum 2) before hitting it.

  This script keeps the REAL tuning code and removes only Ray:
    * it imports benchmark_moe.py as a module,
    * reaches the underlying worker class via ActorClass.__ray_metadata__.modified_class,
    * instantiates it locally instead of as a Ray actor,
    * calls the genuine .tune() -> benchmark_config() path, unchanged.
  Nothing about the kernel invocation, the search space or the timing loop is reimplemented, so the
  resulting config is produced by exactly the same code as upstream would use.

SHAPE (verified from the model's text_config, and cross-checked against the runtime MoE log)
    E  = 128    (global 512 experts / EP4 -> per-rank)
    shard_intermediate_size = 1280  (== 2 * moe_intermediate_size 640; NOT divided, EP shards
                                     experts not the intermediate width)
    so the saved filename uses N = shard_intermediate_size // 2 = 640
    hidden = 2560, topk = 10, group_size = 128, dtype = int4_w4a16
  -> E=128,N=640,device_name=AMD_Radeon_Graphics,dtype=int4_w4a16.json
  which is precisely what get_config_file_name() yields at serve time.

EXPECTATION (operator): this tunes the Triton fused-MoE EXPERT kernel tile config. It does NOT
  touch topkGating (449 ms, 4.1% of kernel time). So it attacks the configurable part of the
  19.8% MoE kernel family, not all of it. A clean 2-5% wall gain is a win; >4.6% closes G6.

Usage (inside <container>):
  moe_tune_noray.py [--save-dir DIR] [--batch-size 1 2 4 8 16] [--device 0]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

BM_PATH = Path("<workdir>/vllm-w/benchmarks/kernels/benchmark_moe.py")

# Shape constants, per the operator's spec and verified against text_config.
E_LOCAL = 128                    # 512 global / EP4
SHARD_INTERMEDIATE = 1280        # 2 * moe_intermediate_size (640)
HIDDEN = 2560
TOPK = 10
GROUP_SIZE = 128
EXPECTED_NAME = "E=128,N=640,device_name=AMD_Radeon_Graphics,dtype=int4_w4a16.json"


def load_upstream():
    """Import benchmark_moe.py as a module WITHOUT running its __main__."""
    if not BM_PATH.exists():
        sys.exit(f"FATAL: {BM_PATH} not found")
    spec = importlib.util.spec_from_file_location("benchmark_moe_upstream", str(BM_PATH))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_moe_upstream"] = mod
    spec.loader.exec_module(mod)          # module-level `import ray` is fine; we never ray.init()
    return mod


def make_local_worker(BM):
    """Reach the class behind @ray.remote and drop the one Ray-specific line in __init__."""
    actor = getattr(BM, "BenchmarkWorker")
    # @ray.remote turns the class into an ActorClass; the original is on __ray_metadata__.
    inner = getattr(getattr(actor, "__ray_metadata__", None), "modified_class", None)
    if inner is None:
        sys.exit("FATAL: could not reach BenchmarkWorker's underlying class "
                 "(ray internals changed) - refusing to guess")

    class LocalWorker(inner):                      # type: ignore[misc,valid-type]
        def __init__(self, seed: int = 0, device_id: int = 0) -> None:
            import torch
            from vllm.utils.torch_utils import set_random_seed

            torch.set_default_device("cuda")
            set_random_seed(seed)
            self.seed = seed
            self.device_id = int(device_id)        # the ONLY change: no ray.get_gpu_ids()

    return LocalWorker


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-dir", default="<workdir>/moe-tuned-gfx906")
    ap.add_argument("--batch-size", type=int, nargs="+", default=[1, 2, 4, 8, 16],
                    help="decode-relevant M only (MTP0 single-stream is M=1)")
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    os.environ.setdefault("HOME", "<workdir>")
    import torch
    from vllm.platforms import current_platform

    print("moe_tune_noray: starting")
    print(f"  ROCR_VISIBLE_DEVICES = {os.environ.get('ROCR_VISIBLE_DEVICES', '<unset>')}")
    print(f"  HIP_VISIBLE_DEVICES  = {os.environ.get('HIP_VISIBLE_DEVICES', '<unset>')}")
    print(f"  cuda device_count    = {torch.cuda.device_count()}")
    print(f"  device               = {args.device}")
    torch.cuda.set_device(args.device)

    BM = load_upstream()
    print(f"  upstream module      = {BM_PATH}")

    use_int4 = True
    use_fp8 = use_int8 = False
    use_deep_gemm = False
    dtype = torch.float16                       # resolve_dtype() -> fp16 on ROCm
    block_quant_shape = [0, GROUP_SIZE]         # int4_w4a16 => [0, group_size]

    # is_fp16 mirrors main(): not(any quantised path)
    is_fp16 = not (use_fp8 or use_int8 or use_int4)
    tune_block_quant_shape = None if use_int4 else block_quant_shape
    search_space = BM.get_configs_compute_bound(is_fp16, tune_block_quant_shape)
    if use_int4:
        # SPLIT_K is a required constexpr for the gptq_awq kernel; only SPLIT_K=1 is used at
        # runtime, so upstream fixes it during tuning too.
        for cfg in search_space:
            cfg["SPLIT_K"] = 1
    before = len(search_space)
    # FILTER OUT CONFIGS THE KERNEL REJECTS. Upstream keeps them (its comment claims the gptq_awq
    # kernel tolerates arbitrary BLOCK_SIZE_K) but this build's moe_wna16_gemm enforces TWO limits,
    # and upstream tune() catches only triton OutOfResources - so ONE bad config aborts the whole
    # run BEFORE save_configs and nothing is written:
    #   moe_wna16.hip:345  BLOCK_SIZE_K must divisible by group_size   (group_size = 128)
    #   moe_wna16.hip:343  size_k must divisible by BLOCK_SIZE_K       (size_k is hidden and the
    #                                                                  expert width, so BLOCK_SIZE_K
    #                                                                  must divide their gcd)
    # Such configs can never be selected at runtime: dropping them is a correctness fix.
    import math
    k_gcd = math.gcd(HIDDEN, SHARD_INTERMEDIATE // 2)      # gcd(2560, 640) = 640
    search_space = [
        c for c in search_space
        if c.get("BLOCK_SIZE_K") is None
        or (c["BLOCK_SIZE_K"] % GROUP_SIZE == 0 and k_gcd % c["BLOCK_SIZE_K"] == 0)
    ]
    print(f"  search space         = {len(search_space)} configurations "
          f"(filtered {before - len(search_space)}: BLOCK_SIZE_K must be a multiple of {GROUP_SIZE} "
          f"AND divide gcd(hidden,expert_width)={k_gcd})")
    print(f"  tuning M values      = {args.batch_size}")

    # SAFETY NET: any other config the kernel rejects is treated as invalid (inf time) rather than
    # killing the run. benchmark_config is resolved from the module globals inside tune(), so
    # patching the module attribute takes effect without touching upstream's loop.
    _orig_benchmark_config = BM.benchmark_config

    def _safe_benchmark_config(*a, **kw):
        try:
            return _orig_benchmark_config(*a, **kw)
        except Exception as exc:  # noqa: BLE001 - deliberately broad: an invalid config must not
            # kill a run that has already spent tens of minutes tuning. Mirrors upstream's existing
            # OutOfResources skip, with a wider net over kernel-side validation errors.
            msg = str(exc)
            if ("moe_wna16_gemm" in msg
                    or "must divisible" in msg
                    or "out of resource" in msg.lower()):
                return float("inf")
            raise

    BM.benchmark_config = _safe_benchmark_config

    LocalWorker = make_local_worker(BM)
    worker = LocalWorker(seed=0, device_id=args.device)

    print(f"  platform is_rocm     = {current_platform.is_rocm()}")
    start = time.time()
    best_configs = {}

    # RESUME: the first successful pass tuned M=1..8 (~36 min) and then hit an invalid config at
    # M=16. Those results were persisted incrementally, so reuse them instead of re-tuning.
    partial_path = Path(args.save_dir) / "partial_progress.json"
    if partial_path.exists():
        try:
            prior = json.loads(partial_path.read_text())
            for k, v in prior.items():
                if int(k) in args.batch_size:
                    best_configs[int(k)] = v
            if best_configs:
                print(f"  RESUMED from {partial_path}: already tuned M={sorted(best_configs)}")
        except Exception as exc:  # noqa: BLE001
            print(f"  (could not read {partial_path}: {exc} - tuning everything fresh)")

    for M in args.batch_size:
        if M in best_configs:
            print(f"  M={M:<4} reused {best_configs[M]}", flush=True)
            continue
        t0 = time.time()
        cfg = worker.tune(
            M, E_LOCAL, SHARD_INTERMEDIATE, HIDDEN, TOPK, dtype,
            use_fp8, use_int8, use_int4,
            search_space, block_quant_shape, use_deep_gemm,
        )
        best_configs[M] = BM.sort_config(cfg)
        print(f"  M={M:<4} best={best_configs[M]}  ({time.time()-t0:.1f}s)", flush=True)
        # INCREMENTAL SAVE: the first attempt tuned M=1..8 (~36 min) and then died at M=16 with
        # nothing written, because save_configs only runs at the very end. Persist after every M.
        _partial = Path(args.save_dir) / "partial_progress.json"
        _partial.write_text(json.dumps({str(k): v for k, v in best_configs.items()}, indent=2))
        print(f"          (saved partial progress for {sorted(best_configs)} -> {_partial})",
              flush=True)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    BM.save_configs(
        best_configs, E_LOCAL, SHARD_INTERMEDIATE, HIDDEN, TOPK, dtype,
        use_fp8, use_int8, use_int4, block_quant_shape, args.save_dir,
    )
    print(f"  tuning took {time.time()-start:.1f}s")

    # ---- GATE: the file MUST be the name the serve computes, or the serve silently ignores it ----
    produced = sorted(p.name for p in Path(args.save_dir).glob("*.json"))
    print(f"\n  files produced: {produced}")
    if EXPECTED_NAME not in produced:
        print(f"  *** FILENAME MISMATCH ***\n  expected: {EXPECTED_NAME}")
        print("  The serve derives this name via get_config_file_name(E, N, dtype, block_shape);")
        print("  a different name means the serve will NOT load it and the A/B would measure nothing.")
        return 2
    print(f"  FILENAME GATE OK: {EXPECTED_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/bin/bash
# moe_tune.sh — MoE autotune for gfx906: produce the tuned Triton fused-MoE tile configs.
#
# Why: the serve warns "Using default MoE config. Performance might be sub-optimal!" because
# get_moe_configs() finds no JSON for device_name=AMD_Radeon_Graphics (vLLM ships 332 configs,
# all NVIDIA). It needs E=128,N=640,device_name=AMD_Radeon_Graphics,dtype=int4_w4a16.json.
#
# Scope caveat (operator): the tuned JSON drives the Triton fused-MoE kernel TILE config. It does
# NOT fix topkGating (449 ms, 4.1% of kernel time). So this attacks the MoE GEMM/config portion of
# the 19.8% MoE family, not all of it. Expectations are kept grounded accordingly.
#
# Decode-first tuning: MTP0 single-stream decode is effectively M=1, and --max-num-seqs 4 gives a
# small safety envelope, so tune M in {1,2,4,8,16} rather than the full ladder to 4096 (expensive
# and mostly prefill/noise for this goal).
# LD_LIBRARY_PATH must carry ROCm's rocm_sysdeps dir. libamd_smi.so has three unmet deps
# (librocm_sysdeps_nl_genl_3 / mnl / nl_3) without it -> amdsmi fails to load -> which silently
# broke the amdsmi sitecustomize shim (it swallows exceptions) -> which makes ray's AMD GPU
# detection raise at import ("ModuleNotFoundError: ray" is three layers from the real cause).
# Verified: with this dir, amdsmi sees 4 GPUs, torch sees 4, ray sees 4.
export LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib/rocm_sysdeps/lib
export HOME=<workdir>
export ROCM_PATH=/opt/rocm
export PYTHONPATH=<workdir>/vllm-w:<workdir>
# ray REFUSES ROCR_VISIBLE_DEVICES at import:
#   RuntimeError: Please use HIP_VISIBLE_DEVICES instead of ROCR_VISIBLE_DEVICES
#   (ray/_private/accelerators/amd_gpu.py:46)
# The serve scripts use ROCR_VISIBLE_DEVICES (fine for torch/vLLM) - the tuner must not.
unset ROCR_VISIBLE_DEVICES
export HIP_VISIBLE_DEVICES=0,1,2,3

export VLLM_MOE_TUNE_CACHE_CLEAR_INTERVAL=25
export PYTHONDONTWRITEBYTECODE=1

mkdir -p <workdir>/moe-tuned-gfx906
cd <workdir>

echo "moe_tune: starting $(date -Is)"
echo "moe_tune: LD_LIBRARY_PATH=[${LD_LIBRARY_PATH:-<unset>}]  save-dir=<workdir>/moe-tuned-gfx906"
echo "moe_tune: ROCR_VISIBLE_DEVICES=[${ROCR_VISIBLE_DEVICES:-<unset>}]  HIP_VISIBLE_DEVICES=[${HIP_VISIBLE_DEVICES:-<unset>}]"

# BELT AND BRACES: amd_gpu.py:46 raises if ROCR_VISIBLE_DEVICES is merely PRESENT, and something
# is re-injecting it (the shell `unset` above is not sufficient). `env -u` guarantees the child
# python never sees it, whatever the source turns out to be.
exec env -u ROCR_VISIBLE_DEVICES \
  <home>/gfx906-venv/bin/python3 \
  <workdir>/vllm-w/benchmarks/kernels/benchmark_moe.py \
  --model <models>/models/qwen38-flash-next-awq \
  --tp-size 4 \
  --enable-expert-parallel \
  --dtype int4_w4a16 \
  --batch-size 1 2 4 8 16 \
  --tune \
  --save-dir <workdir>/moe-tuned-gfx906

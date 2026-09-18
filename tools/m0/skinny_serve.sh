#!/bin/bash
# skinny_serve.sh — serve for the VLLM_ROCM_USE_SKINNY_GEMM causal A/B/A'.
#
# Reads the arm value from <workdir>/skinny_val.txt (1 = skinny path ON, 0 = generic path).
# Everything else is the verified TP4 baseline config. The profiler is enabled on EVERY arm so the
# copyBuffer counts are comparable; throughput is taken from the unprofiled gate in the driver.
export HOME=<workdir>
export ROCM_PATH=/opt/rocm
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=<workdir>/vllm-w:<workdir>
unset LD_LIBRARY_PATH                    # TRAPS #11 (defensive; see #17 for the correction)

export ROCR_VISIBLE_DEVICES=0,1,2,3
export HSA_NO_SCRATCH_RECLAIM=1
export VLLM_ROCM_USE_AITER=0
export TORCH_BLAS_PREFER_HIPBLASLT=0
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export PYTORCH_TUNABLEOP_ENABLED=0
export VLLM_PLE_CPU_OFFLOAD=1
export VLLM_PLE_QUANT_DIR=<models>/qwen38-flash-next-ple/ples_int4
export VLLM_PLE_OFFLOAD_READY_TIMEOUT=3600
export VLLM_RDNA_DENSE_INT8=1
export VLLM_RDNA_AR=1
export VLLM_DISABLE_COMPILE_CACHE=1

# ---- THE EXPERIMENT: the one variable under test -------------------------------
SKINNY=$(cat <workdir>/skinny_val.txt 2>/dev/null | tr -d '[:space:]')
export VLLM_ROCM_USE_SKINNY_GEMM="${SKINNY:-1}"
echo "skinny_serve: VLLM_ROCM_USE_SKINNY_GEMM=$VLLM_ROCM_USE_SKINNY_GEMM (1=skinny ON, 0=generic)"

# ---- profiler, enabled for every arm so copyBuffer counts are comparable ------
export VLLM_TORCH_PROFILER_DIR=<workdir>/tprof
mkdir -p <workdir>/tprof
cd <workdir>

exec <home>/gfx906-venv/bin/python3 -m vllm.entrypoints.openai.api_server \
  --model <models>/models/qwen38-flash-next-awq --served-model-name qwen38-flash-next \
  --dtype float16 --tensor-parallel-size 4 --enable-expert-parallel \
  --max-model-len 2048 --gpu-memory-utilization 0.90 \
  --max-num-seqs 4 --max-num-batched-tokens 1024 \
  -cc.mode=none -cc.cudagraph_mode=full \
  --language-model-only --enable-prefix-caching \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"<workdir>/tprof","torch_profiler_with_stack":false}' \
  --host 0.0.0.0 --port 8002

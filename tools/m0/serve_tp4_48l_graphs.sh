#!/bin/bash
# serve_tp4_48l_graphs.sh — TP4 fully resident, GRAPHS-NO-COMPILE (D060 config).
#
# Identical to serve_tp4_48l.sh (TP4+EP, MTP=0, fp16, no offload, PLE int4 sidecar)
# except: --enforce-eager is replaced by the graphs-no-compile pair
#   -cc.mode=none              compilation DISABLED (D060: inductor miscompiles on gfx906)
#   -cc.cudagraph_mode=full    cudagraph FULL -> FULL_DECODE_ONLY
# which is exactly what <source-host>'s serve_tp2_graphs.sh used.
export HOME=<workdir>
export ROCM_PATH=/opt/rocm
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=<workdir>/vllm-w:<workdir>
# LD_LIBRARY_PATH deliberately UNSET (see TRAPS #11: it hides the GPUs here)
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
export VLLM_ROCM_USE_SKINNY_GEMM=1
cd <workdir>

pkill -9 -f 'VLLM::' 2>/dev/null
pkill -9 -f 'vllm.entrypoints' 2>/dev/null
sleep 4

exec <home>/gfx906-venv/bin/python3 -m vllm.entrypoints.openai.api_server \
  --model <models>/models/qwen38-flash-next-awq --served-model-name qwen38-flash-next \
  --dtype float16 --tensor-parallel-size 4 --enable-expert-parallel \
  --max-model-len 2048 --gpu-memory-utilization 0.90 \
  --max-num-seqs 4 --max-num-batched-tokens 1024 \
  -cc.mode=none -cc.cudagraph_mode=full \
  --language-model-only --enable-prefix-caching \
  --host 0.0.0.0 --port 8002

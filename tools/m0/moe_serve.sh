#!/bin/bash
# moe_serve.sh — TP4 serve for the MoE-tuned A/B. Identical to sweep_serve.sh except that
# VLLM_TUNED_CONFIG_FOLDER is taken from <workdir>/moe_tuned_folder.txt:
#   non-empty -> export it (tuned arm)     empty/missing -> unset (control arm)
# Everything else held fixed: TP4+EP, MTP=0, fp16, no offload, graphs-no-compile, AR on,
# PLE int4 sidecar, max-num-batched-tokens 1024 (the sweep REJECTED 2048).
export HOME=<workdir>
export ROCM_PATH=/opt/rocm
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=<workdir>/vllm-w:<workdir>
unset LD_LIBRARY_PATH                    # TRAPS #11 - explicitly unset, never merely commented

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

FOLDER=$(cat <workdir>/moe_tuned_folder.txt 2>/dev/null | tr -d '[:space:]')
if [ -n "$FOLDER" ]; then
  export VLLM_TUNED_CONFIG_FOLDER="$FOLDER"
else
  unset VLLM_TUNED_CONFIG_FOLDER
fi
echo "moe_serve: VLLM_TUNED_CONFIG_FOLDER=[${VLLM_TUNED_CONFIG_FOLDER:-<unset = CONTROL>}]"
cd <workdir>

exec <home>/gfx906-venv/bin/python3 -m vllm.entrypoints.openai.api_server \
  --model <models>/models/qwen38-flash-next-awq --served-model-name qwen38-flash-next \
  --dtype float16 --tensor-parallel-size 4 --enable-expert-parallel \
  --max-model-len 2048 --gpu-memory-utilization 0.90 \
  --max-num-seqs 4 --max-num-batched-tokens 1024 \
  -cc.mode=none -cc.cudagraph_mode=full \
  --language-model-only --enable-prefix-caching \
  --host 0.0.0.0 --port 8002

#!/bin/bash
# sweep_serve.sh — TP4 serve for the max-num-batched-tokens sweep.
# Reads the value from <workdir>/sweep_maxb.txt. Everything else is held fixed vs the D142/D143
# configuration: TP4+EP, MTP=0, fp16, NO offload, graphs-no-compile, AR on, PLE int4 sidecar.
export HOME=<workdir>
export ROCM_PATH=/opt/rocm
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=<workdir>/vllm-w:<workdir>

# TRAPS #11: setting LD_LIBRARY_PATH makes torch.cuda.device_count() report 0 on this container.
# A comment is not enough - the launcher shell may have inherited it, so UNSET it explicitly.
unset LD_LIBRARY_PATH

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

MAXB=$(cat <workdir>/sweep_maxb.txt)
echo "sweep_serve: max-num-batched-tokens=$MAXB  LD_LIBRARY_PATH=[${LD_LIBRARY_PATH:-<unset>}]"

exec <home>/gfx906-venv/bin/python3 -m vllm.entrypoints.openai.api_server \
  --model <models>/models/qwen38-flash-next-awq --served-model-name qwen38-flash-next \
  --dtype float16 --tensor-parallel-size 4 --enable-expert-parallel \
  --max-model-len 2048 --gpu-memory-utilization 0.90 \
  --max-num-seqs 4 --max-num-batched-tokens "$MAXB" \
  -cc.mode=none -cc.cudagraph_mode=full \
  --language-model-only --enable-prefix-caching \
  --host 0.0.0.0 --port 8002

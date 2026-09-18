#!/bin/bash
# mtp_serve.sh — serve for the O5 MTP A/B/A'. Reads <workdir>/mtp_val.txt:
#   0 -> MTP off (current verified baseline)
#   3 -> --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
#
# Per docs/gfx906/PORT-MAP.md the invocation is exactly
#   --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
# and model_mtp.safetensors must stay on disk (the patched index references it - it does).
# DO-NOT-ATTEMPT.md: num_speculative_tokens=4 was REJECTED on V620 for changing target text, so 3
# is the sanctioned point and the deterministic hash is now a CORRECTNESS gate, not just reruns.
export HOME=<workdir>
export ROCM_PATH=/opt/rocm
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=<workdir>/vllm-w:<workdir>
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

MTP=$(cat <workdir>/mtp_val.txt 2>/dev/null | tr -d '[:space:]')
MTP="${MTP:-0}"
if [ "$MTP" = "0" ]; then
  SPEC_ARGS=""
  echo "mtp_serve: MTP=0 (speculative decode OFF - baseline)"
else
  SPEC_ARGS="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":$MTP}"
  echo "mtp_serve: MTP=$MTP (speculative decode ON)"
fi
cd <workdir>

# shellcheck disable=SC2086
exec <home>/gfx906-venv/bin/python3 -m vllm.entrypoints.openai.api_server \
  --model <models>/models/qwen38-flash-next-awq --served-model-name qwen38-flash-next \
  --dtype float16 --tensor-parallel-size 4 --enable-expert-parallel \
  --max-model-len 2048 --gpu-memory-utilization 0.90 \
  --max-num-seqs 4 --max-num-batched-tokens 1024 \
  -cc.mode=none -cc.cudagraph_mode=full \
  --language-model-only --enable-prefix-caching \
  $SPEC_ARGS \
  --host 0.0.0.0 --port 8002

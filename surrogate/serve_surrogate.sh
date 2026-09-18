#!/usr/bin/env bash
# serve_surrogate.sh — D039 surrogate serve: 8-layer full-shape slice, TP1,
# PLE CPU offload on the real int4 sidecar, language-model-only.
# Run ONLY after the llama.cpp container is stopped (frees ~31 GB).
#   SURROGATE=... PLE_INT4=... DRYRUN=1 ./serve_surrogate.sh   # show env+cmd
#   SURROGATE=... PLE_INT4=... ./serve_surrogate.sh
set -euo pipefail
: "${SURROGATE:?set SURROGATE to the surrogate dir (config.json + surrogate_*.safetensors)}"
: "${PLE_INT4:?set PLE_INT4 to the ples_int4 sidecar (128 shards + META.json)}"
PORT="${PORT:-8000}"
GPUUTIL="${GPUUTIL:-0.90}"
MAXLEN="${MAXLEN:-16384}"
MTP="${MTP:-0}"                      # MTP is O5; the surrogate runs MTP off
export ROCR_VISIBLE_DEVICES=0
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export HSA_NO_SCRATCH_RECLAIM=1
export VLLM_ROCM_USE_AITER=0
export TORCH_BLAS_PREFER_HIPBLASLT=0
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export PYTORCH_TUNABLEOP_ENABLED=0
export VLLM_PLE_CPU_OFFLOAD=1
export VLLM_PLE_QUANT_DIR="$PLE_INT4"
export VLLM_PLE_OFFLOAD_READY_TIMEOUT=3600
export VLLM_RDNA_DENSE_INT8=1
export VLLM_RDNA_AR=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_ROCM_USE_SKINNY_GEMM=1   # our wave64 dense decode route (gfx906 gate)

SPEC=()
VISIONARGS=(--language-model-only --skip-mm-profiling)
CMD=(python3 -m vllm.entrypoints.openai.api_server
  --model "$SURROGATE" --served-model-name qwen38-flash-next-surrogate
  --dtype float16
  --tensor-parallel-size 1 --enable-expert-parallel
  --max-model-len "$MAXLEN" --gpu-memory-utilization "$GPUUTIL"
  --max-num-seqs 4 --max-num-batched-tokens 2048
  --enforce-eager
  "${VISIONARGS[@]}"
  --enable-prefix-caching
  "${SPEC[@]}"
  --host 0.0.0.0 --port "$PORT")

if [ "${DRYRUN:-0}" = "1" ]; then
  env | grep -E '^(ROCR_|ROCM_|HSA_|NCCL_|VLLM_|TORCH_BLAS|FLASH_ATTENTION|PYTORCH_)' | sort | sed 's/^/#   /'
  printf '# command:\n#   '; printf '%q ' "${CMD[@]}"; echo
  exit 0
fi
exec "${CMD[@]}"

#!/bin/bash
# vllm-rocm-tq entrypoint — runs vLLM OpenAI server on a specific AMD GPU.
#
# Key design decisions:
#   - ROCR_VISIBLE_DEVICES for GPU filtering (HSA-level, no HIP remapping)
#   - No HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES (causes Triton error 101)
#   - --device rocm as fallback for amdsmi platform detection on kernels
#     lacking /sys/class/amdgpu/
set -e

# ─── GPU Selection ──────────────────────────────────────────────────────────
export ROCR_VISIBLE_DEVICES="${GPU_ID:-0}"
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
export HSA_SIGNAL_TIMEOUT=-1
export AMDGPU_GPU_RECOVERY=1

echo "[vllm-rocm-tq] ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES}"
echo "[vllm-rocm-tq] Model: ${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct-AWQ}"
echo "[vllm-rocm-tq] Quantization: ${QUANTIZATION:-compressed-tensors}"

# ─── Model Arguments ─────────────────────────────────────────────────────────
MODEL="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct-AWQ}"
SERVED_NAME="${SERVED_MODEL_NAME:-$(echo $MODEL | sed 's|.*/||;s|-.*||')}"
# Use 'awq' by default — ROCR_VISIBLE_DEVICES avoids the Triton error 101
# that previously required the compressed-tensors workaround.
# vLLM 0.24.0 validates that --quantization matches the model config.
QUANT="${QUANTIZATION:-awq}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEMORY_UTILIZATION:-0.85}"
TENSOR_PARALLEL="${TENSOR_PARALLEL_SIZE:-1}"
EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

# ─── vLLM Server ─────────────────────────────────────────────────────────────
exec python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port "${VLLM_PORT:-8001}" \
  --host 0.0.0.0 \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size "$TENSOR_PARALLEL" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --quantization "$QUANT" \
  --trust-remote-code \
  --enforce-eager \
  --disable-log-stats \
  $EXTRA_ARGS
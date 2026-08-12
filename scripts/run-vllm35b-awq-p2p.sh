#!/bin/bash
# Custom entrypoint for vllm-35b-awq-p2p — Qwen3.6-35B-A3B-uncensored-heretic-AWQ vLLM TP=2
# Uses the lemonade-tq image with host ROCm for working P2P on gfx1100.
#
# Model: sahilchachra/Qwen3.6-35B-A3B-uncensored-heretic-AWQ (21.8 GB, W4A16 compressed-tensors)
# Image: ghcr.io/[username]/lemonade-tq:latest (vLLM v0.1.dev1, host ROCm 7.14)
#   256 experts, 8 active/token (A3B = ~3B active)
#
# P2P: ENABLED — lemonade-tq image uses host ROCm 7.14 HIP runtime which
# handles hipIpcGetMemHandle correctly on gfx1100. Stock container images
# (nightly, rocm7.14) have a regression that breaks this.
#
# Based on run-35b-tp2-bundled.sh (the working P2P TP=2 script).
set -e

echo "[vllm-35b-awq-p2p] Starting Qwen3.6-35B-A3B-uncensored-heretic-AWQ vLLM TP=2 (P2P)..."

# ─── Ensure vLLM Python exists ──────────────────────────────────────────────
VLLM_PY=/root/.cache/lemonade/bin/vllm/rocm/bin/python3

if [ ! -x "$VLLM_PY" ]; then
    echo "[vllm-35b-awq-p2p] vLLM not cached yet — starting lemond to download..."
    /entrypoint.sh &
    LEMOND_PID=$!
    for i in $(seq 1 120); do
        if [ -x "$VLLM_PY" ]; then
            echo "[vllm-35b-awq-p2p] vLLM downloaded"
            break
        fi
        sleep 1
    done
    kill $LEMOND_PID 2>/dev/null || true
    wait $LEMOND_PID 2>/dev/null || true
fi

if [ ! -x "$VLLM_PY" ]; then
    echo "[vllm-35b-awq-p2p] ERROR: vLLM Python not found"
    exit 1
fi

# ─── Install C compiler + dev headers for Triton JIT ─────────────────────────
if ! command -v gcc &>/dev/null; then
    echo "[vllm-35b-awq-p2p] Installing gcc + libc6-dev for Triton JIT..."
    apt-get update -qq && apt-get install -y -qq gcc libc6-dev 2>&1 | tail -2
fi
export CC=/usr/bin/gcc
export TRITON_CACHE_DIR=/root/.triton

# Symlink ld.lld for Triton JIT
BUNDLED_LLVM_BIN=/root/.cache/lemonade/bin/vllm/rocm/lib/python3.12/site-packages/_rocm_sdk_core/lib/llvm/bin
if [ -x "/opt/rocm/lib/llvm/bin/ld.lld" ] && [ ! -f "$BUNDLED_LLVM_BIN/ld.lld" ]; then
    ln -sf /opt/rocm/lib/llvm/bin/ld.lld "$BUNDLED_LLVM_BIN/ld.lld"
    ln -sf /opt/rocm/lib/llvm/bin/lld "$BUNDLED_LLVM_BIN/lld"
fi

# ─── Set up LD_LIBRARY_PATH ──────────────────────────────────────────────────
ROCM_SDK=/root/.cache/lemonade/bin/vllm/rocm/lib/python3.12/site-packages/_rocm_sdk_core/lib
RCCL_DIR=/root/.cache/lemonade/bin/vllm/rocm/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx110X_all/lib
export LD_LIBRARY_PATH="/opt/lemonade/extra-libs:/usr/lib/x86_64-linux-gnu:/opt/rocm/lib:/opt/lemonade/llama/rocm:/opt/lemonade/llama/vulkan:${ROCM_SDK}:${ROCM_SDK}/rocm_sysdeps/lib:${ROCM_SDK}/host-math/lib:${LD_LIBRARY_PATH:-}"

# ─── Library version-mismatch symlinks ──────────────────────────────────────
mkdir -p /opt/lemonade/extra-libs
if [ ! -f /opt/lemonade/extra-libs/librocm_smi64.so.7 ]; then
    ln -sf "${ROCM_SDK}/librocm_smi64.so.1" /opt/lemonade/extra-libs/librocm_smi64.so.7 2>/dev/null || true
fi
if [ ! -f /opt/lemonade/extra-libs/librocsolver.so.1 ]; then
    ln -sf /opt/rocm/lib/librocsolver.so.0 /opt/lemonade/extra-libs/librocsolver.so.1 2>/dev/null || true
fi

# ─── RCCL fix for gfx1100 TP>1 ──────────────────────────────────────────────
# LD_PRELOAD order: jemalloc → ROCm SDK libs → full RCCL → nccl stub
# Full librccl.so.1 provides ncclCommWindowDeregister and other symbols
# that libtorch_hip.so needs; the stub alone is insufficient.
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export LD_PRELOAD="${RCCL_DIR}/libjemalloc.so.2:${ROCM_SDK}/libamdhip64.so.7:${ROCM_SDK}/librocm_smi64.so.1:${ROCM_SDK}/librocprofiler-register.so.0:${ROCM_SDK}/librccl.so.1:${RCCL_DIR}/libnccl_stub.so"
export C_INCLUDE_PATH=/usr/include
export CPLUS_INCLUDE_PATH=/usr/include

# ─── GPU visibility — BOTH GPUs ──────────────────────────────────────────────
export ROCR_VISIBLE_DEVICES=0,1
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
export HSA_SIGNAL_TIMEOUT=-1
export AMDGPU_GPU_RECOVERY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/root/.cache/huggingface
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE

# ─── NCCL / P2P configuration for TRX50 WS (IOMMU off, no ACS) ───────────────
# P2P enabled: both GPUs on separate PCIe roots but no ACS blocks DMA.
# NCCL_PROTO=Simple: reliable for non-xGMI PCIe P2P (avoids LL/LL128 issues).
# GPU_MAX_HW_QUEUES=1: prevents amdgpu scheduler contention on gfx1100.
export NCCL_PROTO=Simple
export GPU_MAX_HW_QUEUES=1
# NCCL_P2P_DISABLE and HSA_ENABLE_SDMA are intentionally NOT set.
# With IOMMU off + no ACS, direct P2P works and SDMA copy engines are available.

echo "[vllm-35b-awq-p2p] ROCR_VISIBLE_DEVICES=0,1 (host ROCm, P2P enabled)"

# ─── vLLM server ────────────────────────────────────────────────────────────
"$VLLM_PY" -m vllm.entrypoints.openai.api_server \
  --model "sahilchachra/Qwen3.6-35B-A3B-uncensored-heretic-AWQ" \
  --port 13313 \
  --host 0.0.0.0 \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 1 \
  --gpu-memory-utilization 0.92 \
  --quantization compressed-tensors \
  --max-model-len 65536 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --dtype bfloat16 \
  --language-model-only \
  --enforce-eager \
  --reasoning-parser qwen3 \
  --load-format safetensors \
  --safetensors-load-strategy lazy &

SERVER_PID=$!

# Forward signals so `docker stop` terminates vLLM cleanly
trap 'kill -TERM "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID"' TERM INT

echo "[vllm-35b-awq-p2p] Waiting for /health... (first startup may take 20-30 min for Triton compilation)"
for i in $(seq 1 1200); do
  curl -sf http://localhost:13313/health >/dev/null 2>&1 && break
  sleep 2
done

# ─── Warmup ─────────────────────────────────────────────────────────────────
echo "[vllm-35b-awq-p2p] Warming up..."
for i in 1 2 3; do
  curl -sf http://localhost:13313/v1/completions -H "Content-Type: application/json" \
    -d '{"model":"sahilchachra/Qwen3.6-35B-A3B-uncensored-heretic-AWQ","prompt":"system online","max_tokens":8,"temperature":0.7,"top_p":0.9,"top_k":50}' \
    >/dev/null 2>&1 || true
done

curl -sf http://localhost:13313/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"sahilchachra/Qwen3.6-35B-A3B-uncensored-heretic-AWQ","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"temperature":0.7,"top_p":0.9,"top_k":50}' \
  >/dev/null 2>&1 || true

echo "[vllm-35b-awq-p2p] Warmup complete"
wait "$SERVER_PID"
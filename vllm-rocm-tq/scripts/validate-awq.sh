#!/bin/bash
# validate-awq.sh — Small-model-first AWQ validation on vllm-rocm-tq
#
# Tests the full inference path: model download → AWQ dequantize → generation
# Uses Qwen/Qwen2.5-0.5B-Instruct-AWQ (downloads in seconds, loads in <1s).
#
# Usage:
#   ./scripts/validate-awq.sh [GPU_ID] [MODEL_NAME]
#
# Defaults: GPU_ID=0, MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct-AWQ
set -e

GPU_ID="${1:-0}"
MODEL_NAME="${2:-Qwen/Qwen2.5-0.5B-Instruct-AWQ}"
PORT=13309

echo "=== AWQ Validation ==="
echo "GPU: $GPU_ID"
echo "Model: $MODEL_NAME"
echo "Port: $PORT"
echo ""

# Start container with validation model
echo "[1/3] Starting vllm-rocm-tq..."
GPU_ID=$GPU_ID MODEL_NAME="$MODEL_NAME" SERVED_MODEL_NAME="validation" \
  MAX_MODEL_LEN=4096 GPU_MEMORY_UTILIZATION=0.25 \
  docker compose up -d

# Wait for health
echo "[2/3] Waiting for server to be healthy..."
for i in $(seq 1 120); do
  if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo "  Server healthy after ${i}s"
    break
  fi
  if [ $i -eq 120 ]; then
    echo "  ERROR: Server not healthy after 120s"
    docker compose logs --tail 30
    exit 1
  fi
  sleep 1
done

# Test inference
echo "[3/3] Testing inference..."
RESPONSE=$(curl -s "http://localhost:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "validation",
    "messages": [{"role": "user", "content": "Say hello in one word."}],
    "max_tokens": 10,
    "temperature": 0
  }')

echo "  Response: $RESPONSE"
echo ""

# Verify we got actual output
if echo "$RESPONSE" | python3 -c "import json,sys; d=json.load(sys.stdin); assert d['choices'][0]['message']['content']" 2>/dev/null; then
  echo "✅ AWQ inference validated on GPU$GPU_ID"
  CONTENT=$(echo "$RESPONSE" | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])")
  echo "   Output: $CONTENT"
else
  echo "❌ Inference failed"
  echo "   Raw: $RESPONSE"
  exit 1
fi

# Cleanup
docker compose down
echo ""
echo "Done."
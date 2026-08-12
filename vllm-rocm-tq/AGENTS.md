# vllm-rocm-tq

vLLM ROCm inference server for AWQ models on AMD RDNA3 (3× RX 7900 XT).

Thin layer on `vllm/vllm-openai-rocm:v0.24.0` (stable, ROCm 7.2.3, PyTorch 2.11+HIP).
Eliminates custom wheels, sitecustomize patches, and Triton/gcc workarounds.

## Build & Deploy

```bash
# Build
docker build -t vllm-rocm-tq:latest .

# Validate with small AWQ model
docker compose up -d

# Serve target model on GPU2
GPU_ID=2 MODEL_NAME=<hf-model-id> docker compose up -d
```

## Architecture

- `Dockerfile` — Thin layer on `vllm/vllm-openai-rocm:v0.24.0`
- `docker-compose.yml` — Production deployment (port 13309)
- `entrypoint.sh` — GPU selection via `ROCR_VISIBLE_DEVICES` + vLLM serve
- `scripts/validate-awq.sh` — Small-model-first validation
- `scripts/test-inference.py` — Inference smoke test

## Key Decisions

1. **`ROCR_VISIBLE_DEVICES`** (not `CUDA_VISIBLE_DEVICES`) — HSA-level filtering avoids Triton error 101
2. **`--quantization compressed-tensors`** (not `awq`) — different dequantize path, avoids Triton bug
3. **Triton JIT cache volume** — `./data/triton-cache:/root/.triton:rw` for idempotent restarts
4. **`--device rocm`** — bypasses amdsmi platform detection on kernels lacking `/sys/class/amdgpu/`

## CI

Gitea Actions workflow builds and pushes to both registries:
- `nas.kadrlik.home:3042/mkadrlik/vllm-rocm-tq`
- `ghcr.io/mkadrlik/vllm-rocm-tq`

Mirror pattern: Gitea primary, GitHub push-only (strip fetch refspec).

## Upstream

Base image: `vllm/vllm-openai-rocm:v0.24.0`
Upstream mirror: `mkadrlik/vllm-rocm` (Gitea, mirrors `github.com/vllm-project/vllm`)
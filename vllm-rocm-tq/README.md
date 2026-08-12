# vllm-rocm-tq

vLLM ROCm inference server for **AWQ models** on AMD RDNA3 (RX 7900 XT). Thin layer on `vllm/vllm-openai-rocm:v0.24.0`.

## What This Eliminates

| Old | vllm-rocm-tq |
|-----|-------------|
| `sitecustomize_vllm.py` patch (both copies) | Not needed — v0.24.0 has correct amdsmi |
| Custom vLLM wheel (`vllm-0.1.dev1+gf2069b005.rocm724...whl`) | Official v0.24.0 (stable release) |
| `run-ornith-vllm.sh` Triton/gcc workarounds | Not needed — proper env vars in entrypoint |
| `rocm/vllm-dev:base` source build | Pre-built official image |

## Architecture

```
vllm-rocm-tq/
├── Dockerfile           # Thin layer on vllm/vllm-openai-rocm:v0.24.0
├── docker-compose.yml   # Production deployment (port 13309)
├── entrypoint.sh        # GPU selection (ROCR_VISIBLE_DEVICES) + vLLM serve
├── scripts/
│   ├── validate-awq.sh  # Small-model-first validation pattern
│   └── test-inference.py # Quick inference smoke test
└── data/
    ├── hf-cache/         # HuggingFace model cache
    └── triton-cache/     # Triton JIT kernel cache (critical for idempotent restarts)
```

## Key Design Decisions

### GPU Selection: `ROCR_VISIBLE_DEVICES`

Uses `ROCR_VISIBLE_DEVICES` instead of `CUDA_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES`. This filters at the HSA kernel driver level — HIP device 0 IS the target GPU directly, no remapping. This avoids Triton error 101 (invalid device ordinal) caused by HIP remapping.

### AWQ Quantization: `compressed-tensors`

Uses `--quantization compressed-tensors` instead of `--quantization awq`. Same AWQ model weights, different dequantize kernel path. The `awq` flag routes through `awq_triton.py` which has a Triton device ordinal bug on single-GPU visibility.

### Triton JIT Cache

Mounts `./data/triton-cache:/root/.triton:rw` — critical for idempotent restarts. Without it, every container restart recompiles all Triton kernels from scratch.

### `--device rocm` Flag

The `--device rocm` CLI flag bypasses vLLM's import-time amdsmi platform detection. On kernels lacking `/sys/class/amdgpu/` (Fedora 44, kernel 7.0.x), amdsmi returns 0 handles and vLLM falls to `UnspecifiedPlatform`.

## Usage

### Validate AWQ Inference (Small Model First)

```bash
# Default: Qwen/Qwen2.5-0.5B-Instruct-AWQ on GPU0
docker compose up -d
```

### Serve DiffusionGemma AWQ on GPU2

```bash
GPU_ID=2 \
MODEL_NAME=pixelkaiser/diffusiongemma-26B-A4B-it-AWQ-MLP-W4A16-G64-S32-L1024 \
SERVED_MODEL_NAME=DiffusionGemma-26B-A4B \
MAX_MODEL_LEN=4096 \
GPU_MEMORY_UTILIZATION=0.70 \
docker compose up -d
```

### Serve Ornith AWQ on GPU2

```bash
GPU_ID=2 \
MODEL_NAME=cyankiwi/Ornith-1.0-9B-AWQ-INT4 \
SERVED_MODEL_NAME=Ornith-1.0-9B \
MAX_MODEL_LEN=32768 \
VLLM_EXTRA_ARGS="--enable-auto-tool-choice --tool-call-parser hermes --language-model-only" \
docker compose up -d
```

## Verified Models

| Model | Size (VRAM) | GPU | Notes |
|-------|-------------|-----|-------|
| `Qwen/Qwen2.5-0.5B-Instruct-AWQ` | ~0.5 GiB | Any | Validation model |
| `Qwen/Qwen2.5-7B-Instruct-AWQ` | ~5.2 GiB | Any | Mid-range test |
| `cyankiwi/Ornith-1.0-9B-AWQ-INT4` | ~16.8 GiB | GPU2 | GDN attention, Triton autotuner first boot 20-30 min |
| `pixelkaiser/diffusiongemma-26B-A4B-it-AWQ-MLP-W4A16-G64-S32-L1024` | ~15.14 GiB | Any | Diffusion LLM |

## Base Image

`vllm/vllm-openai-rocm:v0.24.0` — stable release on ROCm 7.2.3, PyTorch 2.11+HIP.

| Property | Value |
|----------|-------|
| vLLM version | 0.24.0 |
| ROCm | 7.2.3 |
| PyTorch | 2.11.0+gitd0c8b1f, HIP 7.2.53211 |
| amdsmi | 26.2.2+c2d9476115 |
| DiffusionGemma | ✅ `DiffusionGemmaForBlockDiffusion` |
| Gemma-4 | ✅ `Gemma4ForConditionalGeneration` |
| Image size | ~43.5 GB |
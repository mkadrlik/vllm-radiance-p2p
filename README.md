# vLLM Inference on AMD RDNA3 (gfx1100)

> This project is a port for AMD Radeon RX 7900 XTX (gfx1100). The base container
> image `stilldeadcode/vllm-radiance:0.5.7` is built from the Radiance fork by
> [StillDeadcode](https://codeberg.org/StillDeadcode/vllm-radiance/), which provides the TP2 P2P
> patches, CUDA-graph tuning, and AITER integration that make this work. Huge thanks
> to them for the excellent upstream work — this repo is just a deployment wrapper
> and tuning guide built on top of their container.

## What this is

Prebuilt Docker image + `docker-compose.yml` for serving quantized Qwen models on
**2× AMD RX 7900 XTX (gfx1100)** with tensor-parallel 2 and PCIe P2P, or any AWQ
model on a single GPU. No CUDA, no NVIDIA.

| Profile | Model | GPUs | Port | Quant |
|---------|-------|------|------|-------|
| `radiance-27b` | Qwen3.6-27B-Quark-W8A8 | 2× RX 7900 XTX (TP2) | 13313 | Quark W8A8 |
| `radiance-35b` | Qwen3.6-35B-A3B-Quark | 2× RX 7900 XTX (TP2) | 13313 | Quark W8A8 |
| `awq` | Any AWQ model (configurable) | 1 GPU | 13309 | AWQ |

## Requirements

- 1 or 2× AMD RX 7900 XTX (gfx1100; RDNA3). A single 24 GB card runs the `awq`
  profile; the radiance profiles need **two** cards.
- Linux host with the `amdgpu` kernel driver and **IOMMU disabled** (or ACS
  overridden) for TP2 P2P — see [AGENTS.md](./AGENTS.md) → *P2P on gfx1100*.
- Docker Engine with the `docker compose` plugin, and `--device` access to
  `/dev/kfd` and `/dev/dri` (the compose file handles this).
- ~40 GB free disk for the image, plus model weights (~27–35 GB each).
- A HuggingFace account that has accepted the model's license terms.

## Quick start

### 1. Configure

```bash
cp .env.example .env
```

Open `.env` and set `HF_TOKEN` (get one at
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).
Everything else has working defaults. Optional: change ports, model names, or
memory limits — see [Environment variables](#environment-variables).

### 2. Pick a profile and start

```bash
docker compose --profile radiance-27b up -d   # Qwen3.6-27B Quark, TP2
# or
docker compose --profile radiance-35b up -d   # Qwen3.6-35B-A3B Quark, TP2
# or
docker compose --profile awq up -d            # Generic AWQ, single GPU
```

The radiance profiles pull the prebuilt image
`ghcr.io/mkadrlik/vllm-radiance-p2p:gfx1100` (see
[Image provenance](#image-provenance-read-this)); the `awq` profile builds
locally from `Dockerfile.awq` on first run.

**First boot takes 20–30 minutes** — vLLM compiles kernels for your exact GPU
set (the healthcheck `start_period` of 1800 s accounts for this). Subsequent
boots reuse the caches under `./data/` and start in ~2 minutes.

### 3. Verify

```bash
docker compose --profile radiance-27b logs -f qwen-27b   # wait for "Uvicorn running"
curl http://localhost:13313/v1/models
```

Then point any OpenAI-compatible client at `http://localhost:13313/v1` with the
model name `qwen-27b` (or `qwen-35b` / your `AWQ_SERVED_MODEL_NAME`):

```bash
curl http://localhost:13313/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen-27b", "messages": [{"role": "user", "content": "hello"}]}'
```

## ⚠️ Image provenance — read this

**Pin the `:gfx1100` tag — do not rely on `:latest`, and do not run
`docker compose build` or `docker build .` for the radiance profiles.**

- `ghcr.io/mkadrlik/vllm-radiance-p2p:gfx1100` — the verified gfx1100 build
  (immutable digest `sha256:6253c8e6cf9c…`). This is what the compose file
  references by default.
- `:latest` currently points at the same gfx1100 image (retagged back on
  2026-08-12 after being clobbered), but it is **mutable** and has been broken
  before. The `main-<sha>` tags on this registry are gfx1201-broken builds. On
  a 7900 XTX a broken image dies at startup:

  ```
  torch.AcceleratorError: CUDA error: device kernel image is invalid   # hipErrorInvalidImage
  arch check : FAIL (0/2 gfx1201)
  ```

- The repo `Dockerfile` builds from `stilldeadcode/vllm-radiance:0.5.7`, whose
  source has drifted to gfx1201/RDNA4, so **it cannot reproduce a gfx1100
  image yet**. The compose file therefore has no `build:` for the radiance
  profiles — they always use the prebuilt image.

The complete gfx1100 adaptation (patches, HIP kernels, AITER configs,
entrypoint) is recovered from the working image and the reproducible source
build is documented in [AGENTS.md](./AGENTS.md) → *gfx1100 build*. Until that
build is automated, run the prebuilt `:gfx1100` image.

## Environment variables

All are set in `.env` (copy from `.env.example`, which documents each one).
The compose file provides defaults for every variable, so `.env` only needs
`HF_TOKEN` to get started.

| Variable | Default | Used by | Purpose |
|----------|---------|---------|---------|
| `HF_TOKEN` | — | all | HuggingFace download auth (first boot only; weights are cached) |
| `RADIANCE_IMAGE` | `ghcr.io/mkadrlik/vllm-radiance-p2p:gfx1100` | radiance | prebuilt image to run |
| `VLLM_HOST_PORT` | `13313` | radiance | host port for the API |
| `HIP_VISIBLE_DEVICES` | `0,1` | radiance | the two TP2 GPUs |
| `TENSOR_PARALLEL_SIZE` | `2` | radiance | number of GPUs |
| `VLLM_SHM_SIZE` | `16G` | radiance | shared memory for NCCL |
| `VLLM_27B_GPU_MEM_UTIL` / `VLLM_35B_GPU_MEM_UTIL` | `0.85` | radiance | KV-cache budget per GPU |
| `VLLM_27B_MAX_MODEL_LEN` | `65537` | 27b | context length |
| `VLLM_35B_MAX_MODEL_LEN` | `32768` | 35b | context length — **do not raise, OOMs** |
| `VLLM_START_PERIOD` | `1800` | radiance | healthcheck grace for first-boot compile |
| `AWQ_IMAGE` | `vllm-radiance-p2p-awq:latest` | awq | local build tag |
| `AWQ_MODEL_NAME` | `Qwen/Qwen2.5-0.5B-Instruct-AWQ` | awq | any AWQ model on HF |
| `AWQ_HOST_PORT` | `13309` | awq | host port for the API |
| `AWQ_GPU_ID` | `0` | awq | which single GPU |
| `VLLM_EXTRA_ARGS` | — | awq | extra vLLM flags, appended verbatim |

## 27B vs 35B — argument differences

These two profiles share **identical** image, environment variables, and all
arguments except four:

| Argument | 27B Quark | 35B-A3B Quark | Why |
|----------|-----------|---------------|-----|
| Model | `nameistoken/Qwen3.6-27B-Quark-W8A8-INT8` | `nameistoken/Qwen3.6-35B-A3B-Quark-W8A8-INT8` | Different model weights |
| `--served-model-name` | `qwen-27b` | `qwen-35b` | API endpoint label |
| `--max-model-len` | **65537** | **32768** | 35B-A3B is larger — 65k OOMs at TP2. 32k is the stable ceiling. |
| Cache volume | `./data/radiance-cache-27b-quark:/cache` | `./data/radiance-cache-35b-a3b-quark:/cache` | Separate compile caches (Triton/Inductor are model-specific) |

**Gotcha:** Do not set `--max-model-len=65537` on the 35B profile — it will OOM.
The 27B profile needs it because it fits; the 35B profile caps at 32k.

## Tuning notes

- Radiance profiles: `NCCL_PROTO=Simple`, `RADIANCE_FAST_REDUCE=0`,
  `NCCL_P2P_DISABLE` NOT needed (IOMMU off, no ACS).
- AWQ: uses `ROCR_VISIBLE_DEVICES` (not `HIP_VISIBLE_DEVICES`) to avoid a Triton
  error 101 on ROCm 7.
- Caches under `./data/` are per-profile and gitignored. If boot dies mid-compile
  after a crash: `find ./data -name '*.json' -size 0 -delete`.
- `RADIANCE_RUN_BWTEST` must stay `0` — see [AGENTS.md](./AGENTS.md).

## Structure

```
.
├── docker-compose.yml      # All profiles in one file
├── .env.example            # Copy to .env and set HF_TOKEN
├── Dockerfile.awq          # AWQ build (awq profile only)
├── Dockerfile              # ⚠️ broken for gfx1100 (drifted base) — see above
├── data/                   # All caches (gitignored, created on first run)
├── scripts/                # Helper scripts (optional)
└── AGENTS.md               # Engineering notes (not needed to run)
```

## Performance

| Config | Decode | Notes |
|--------|--------|-------|
| 27B Quark (CUDA-graph) | ~22 tok/s | tg128, MTP off |
| 35B-A3B Quark (CUDA-graph) | ~19 tok/s | tg128, MTP off |
| AWQ (eager) | varies | Depends on model size |

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `device kernel image is invalid` at startup | Running a gfx1201 image (`:latest` or a local build) on gfx1100 | Use `:gfx1100`; do not rebuild |
| Hangs after `Using network interface lo` | Custom all-reduce path | `RADIANCE_FAST_REDUCE=0` (compose default) |
| Host reboots on boot | P2P bandwidth sweep | `RADIANCE_RUN_BWTEST=0` (compose default) |
| `JSONDecodeError` on boot | 0-byte Triton cache from a prior crash | `find ./data -name '*.json' -size 0 -delete` |
| 35B OOM at load | Context too long | keep `VLLM_35B_MAX_MODEL_LEN=32768` |
| 401 / model not found on first boot | HF license not accepted | accept terms on the model page, set `HF_TOKEN` |

More detail in [AGENTS.md](./AGENTS.md).

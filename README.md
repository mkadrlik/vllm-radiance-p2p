# vLLM Inference on AMD RDNA3 (gfx1100)

> This project is a port for AMD Radeon RX 7900 XTX (gfx1100). The base container
> image `stilldeadcode/vllm-radiance:0.5.7` is built from the Radiance fork by
> [StillDeadcode](https://codeberg.org/StillDeadcode/vllm-radiance/), which provides the TP2 P2P
> patches, CUDA-graph tuning, and AITER integration that make this work. Huge thanks
> to them for the excellent upstream work — this repo is just a deployment wrapper
> and tuning guide built on top of their container.

## ⚠️ gfx1100 image status (2026-08-12)

**The published `:latest` IS the gfx1100 build.** It was retagged on 2026-08-12 from
the known-good `vllm-radiance:gfx1100` image (which was a full ROCm 7.14 source build
for `GFX_ARCH=gfx1100`).

**The repo `Dockerfile` does NOT reproduce a gfx1100 image — yet.** It is
`FROM stilldeadcode/vllm-radiance:0.5.7` (stock), and that base has drifted to a newer
Radiance source that targets **gfx1201/RDNA4**: `_aiter_ops.py` now uses `on_gfx12x`
and `aiter/ops/triton/gemm_a8w8.py` moved. Building it and running on RX 7900 XTX
(gfx1100) fails at startup with:

```
torch.AcceleratorError: CUDA error: device kernel image is invalid   # hipErrorInvalidImage
arch check : FAIL (0/2 gfx1201)
```

because `patch_gfx1100.py`'s anchor (`is_aiter_found_and_supported: anchor matched 0x,
expected 1`) can't apply on the drifted base.

**How to run on gfx1100 (recommended):** use the pre-built `:latest` — it IS gfx1100.

```bash
docker pull ghcr.io/mkadrlik/vllm-radiance-p2p:latest
```

**How to build for gfx1100 (source build, not yet automated):** the full gfx1100
adaptation lives in [`build/`](./build/) — the complete patch set, HIP kernel sources
(`router_gemm.hip`, `radiance_ar_ext.hip`), radiance modules, AITER/GEMM/MoE/FP8 configs,
`radiance_preamble.py`, and `radiance_entrypoint.sh`, recovered from the working image.
It was originally built from a ROCm 7.14 base with source-built wheels for
`GFX_ARCH=gfx1100`; those wheels are not recoverable, so reproducing it requires
rebuilding that source pipeline. See `AGENTS.md` → *gfx1100 build* for the exact steps.
Until automated: do not `docker build` the repo and expect gfx1100 — run the pre-built image.

One command to start. Three profiles. Pick one.

```bash
docker compose --profile radiance-27b up -d   # Qwen3.6-27B Quark, TP2
docker compose --profile radiance-35b up -d   # Qwen3.6-35B-A3B Quark, TP2
docker compose --profile awq up -d            # Generic AWQ, single GPU
```

## Profiles

| Profile | Model | GPUs | Port | Quant |
|---------|-------|------|------|-------|
| `radiance-27b` | Qwen3.6-27B-Quark-W8A8 | 2× RX 7900 XTX (TP2) | 13313 | Quark W8A8 |
| `radiance-35b` | Qwen3.6-35B-A3B-Quark | 2× RX 7900 XTX (TP2) | 13313 | Quark W8A8 |
| `awq` | Any AWQ model (configurable) | 1 GPU | 13309 | AWQ |

## Quick Start

### Run from pre-built image (no build)

```bash
docker pull ghcr.io/mkadrlik/vllm-radiance-p2p:latest
```

Then run directly — no compose needed. Arguments vary by profile:

```bash
# 35B-A3B (TP2)
docker run -d --name vllm \
  --gpus all --shm-size 16G -e ROCM_PATH=/opt/rocm -e HIP_PATH=/opt/rocm \
  --privileged --security-opt seccomp=unconfined \
  --device /dev/kfd --device /dev/dri \
  -p 13313:13313 \
  --entrypoint vllm \
  ghcr.io/mkadrlik/vllm-radiance-p2p:latest \
  serve --host 0.0.0.0 --port 13313 \
  nameistoken/Qwen3.6-35B-A3B-Quark-W8A8-INT8 \
  --served-model-name vllm-35b --quantization quark \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.85 \
  --max-model-len 32768 --max-num-batched-tokens 2048 \
  --max-num-seqs 64 --dtype bfloat16 \
  --attention-backend ROCM_ATTN --enable-prefix-caching \
  --compilation-config='{"cudagraph_capture_sizes":[1,2,4,8,16,32,64,128],"max_cudagraph_capture_size":128}' \
  --no-async-scheduling --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --language-model-only --trust-remote-code \
  -e HIP_VISIBLE_DEVICES=0,1 -e NCCL_PROTO=Simple -e GPU_MAX_HW_QUEUES=1
```

### Build from source

1. Clone
2. Set env vars (see `.env.example`)
3. `docker compose --profile <profile> up -d`

First boot takes 7-15 minutes (compilation). Subsequent boots use cached `./data/`.

## Structure

```
.
├── docker-compose.yml      # All profiles in one file
├── Dockerfile              # Radiance build (radiance-27b/35b)
├── Dockerfile.awq          # AWQ build (awq profile)
├── .env.example            # Copy to .env and fill in HF_TOKEN
├── data/                   # All caches (gitignored)
│   ├── cache/              # Radiance HF cache
│   ├── radiance-cache-*/   # Radiance compile caches
│   ├── awq-hf/             # AWQ HF cache
│   └── awq-triton/         # AWQ Triton cache
├── scripts/                # Helper scripts (optional)
│   └── radiance_build_state.sh
├── AGENTS.md               # Engineering notes (not needed to run)
└── .ci/                    # CI pipeline
```

## Configuration

Environment variables in `.env` override compose defaults:

- `HF_TOKEN` — required for Radiance profiles (HuggingFace download)
- `VLLM_HOST_PORT` — AWQ host port (default 13309)
- `MODEL_NAME` — AWQ model path (default Qwen/Qwen2.5-0.5B-Instruct-AWQ)
- `GPU_ID` — AWQ GPU index (default 0)

## Radiance 27B vs 35B — Argument Differences

These two profiles share **identical** build context, Dockerfile, environment variables, and all arguments except four:

| Argument | 27B Quark | 35B-A3B Quark | Why |
|----------|-----------|---------------|-----|
| Model | `nameistoken/Qwen3.6-27B-Quark-W8A8-INT8` | `nameistoken/Qwen3.6-35B-A3B-Quark-W8A8-INT8` | Different model weights |
| `--served-model-name` | `vllm-27b` | `vllm-35b` | API endpoint label |
| `--max-model-len` | **65537** | **32768** | 35B-A3B is larger — 65k OOMs at TP2. 32k is the stable ceiling. |
| Cache volume | `./data/radiance-cache-27b-quark:/cache` | `./data/radiance-cache-35b-a3b-quark:/cache` | Separate compile caches (Triton/Inductor are model-specific) |

**Shared arguments** (identical between both): TP2, CUDA-graph sizes 1–128, `--no-async-scheduling`, `--compilation-config`, all Radiance env vars (`RADIANCE_*`, `VLLM_ROCM_USE_AITER_*`), AITER/GEMM settings, `GPU_MAX_HW_QUEUES=1`.

**Gotcha:** Do not set `--max-model-len=65537` on the 35B profile — it will OOM. The 27B profile needs it because it fits; the 35B profile caps at 32k.

## Tuning Notes

- Radiance profiles: `NCCL_PROTO=Simple`, `RADIANCE_FAST_REDUCE=0`, `NCCL_P2P_DISABLE` NOT needed (IOMMU off, no ACS)
- AWQ: Uses `ROCR_VISIBLE_DEVICES` (not HIP_VISIBLE_DEVICES) to avoid Triton error 101
- Shared cache dirs are gitignored — wipe `find <cache> -name '*.json' -size 0 -delete` if boot fails mid-compile

## Performance

| Config | Decode | Notes |
|--------|--------|-------|
| 27B Quark (CUDA-graph) | ~22 tok/s | tg128, MTP off |
| 35B-A3B Quark (CUDA-graph) | ~19 tok/s | tg128, MTP off |
| AWQ (eager) | varies | Depends on model size |

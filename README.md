# vLLM Inference on AMD RDNA3 (gfx1100)

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
└── .gitea/                 # CI pipeline (not needed to run)
```

## Configuration

Environment variables in `.env` override compose defaults:

- `HF_TOKEN` — required for Radiance profiles (HuggingFace download)
- `VLLM_HOST_PORT` — AWQ host port (default 13309)
- `MODEL_NAME` — AWQ model path (default Qwen/Qwen2.5-0.5B-Instruct-AWQ)
- `GPU_ID` — AWQ GPU index (default 0)

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

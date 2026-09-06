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
docker pull nas.kadrlik.home:3042/mkadrlik/vllm-radiance-p2p:latest
```

**How to build for gfx1100 (source build, not yet automated):** the full gfx1100
adaptation lives in [`build/`](./build/) — the complete patch set, HIP kernel sources
(`router_gemm.hip`, `radiance_ar_ext.hip`), radiance modules, AITER/GEMM/MoE/FP8 configs,
`radiance_preamble.py`, and `radiance_entrypoint.sh`, recovered from the working image.
It was originally built from a ROCm 7.14 base with source-built wheels for
`GFX_ARCH=gfx1100`; those wheels are not recoverable, so reproducing it requires
rebuilding that source pipeline. See `AGENTS.md` → *gfx1100 build* for the exact steps.
Until automated: do not `docker build` the repo and expect gfx1100 — run the pre-built image.

One command to start. Five profiles. Pick one.

```bash
docker compose --profile qwen38-27b up -d     # Qwen3.8-27B AWQ-INT4, TP2 + MTP (:13305) ← current prod
docker compose --profile ornith-9b up -d      # Ornith-1.5-9B AWQ-INT4, TP1 + vision (:13318) ← current prod
docker compose --profile radiance-27b up -d   # Qwen3.6-27B Quark, TP2
docker compose --profile radiance-35b up -d   # Qwen3.6-35B-A3B Quark, TP2
docker compose --profile awq up -d            # Generic AWQ, single GPU
```

## Profiles

| Profile | Model | GPUs | Port | Quant | Status |
|---------|-------|------|------|-------|--------|
| `qwen38-27b` | Qwen3.8-27B-AWQ-INT4 (MTP spec-decode) | 2× RX 7900 XTX (TP2) | 13305 | compressed-tensors W4 g32 | **production worker** |
| `ornith-9b` | Ornith-1.5-9B-AWQ-INT4 (vision-capable) | 1× RX 7900 XTX (TP1) | 13318 | compressed-tensors W4 g32 | **production facade+vision** |
| `radiance-27b` | Qwen3.6-27B-Quark-W8A8 | 2× RX 7900 XTX (TP2) | 13313 | Quark W8A8 | legacy |
| `radiance-35b` | Qwen3.6-35B-A3B-Quark | 2× RX 7900 XTX (TP2) | 13313 | Quark W8A8 | legacy |
| `awq` | Any AWQ model (configurable) | 1 GPU | 13309 | AWQ | generic template |

The two production profiles mirror the live big-chungus deployment
(`docker/vllm-rocm-main/docker-compose-{27b,9b}.yml`) — same flags, same env.
Key findings that shaped them:

- **W8A8 → AWQ on the 27B:** single-stream +18-25% (not 2× — TP2 all-reduce,
  not bandwidth, is the floor), batch throughput ~2×, KV pool +80%.
  Full data: [`docs/qwen38-27b-quant-comparison.md`](docs/qwen38-27b-quant-comparison.md)
- **TP1 concurrency cliff:** ≥5 simultaneous decode streams collapse vLLM 0.26
  on gfx1100 → `--max-num-seqs 4` cap. [`docs/vllm-9b-concurrency-cliff.md`](docs/vllm-9b-concurrency-cliff.md)
- **Vision on gfx1100:** requires `--limit-mm-per-prompt '{"image":4,"video":0}'`
  **and** `--mm-processor-kwargs '{"max_pixels":1003520}'` or the ViT dummy
  profile OOMs at 256 GB. [`docs/vllm-vision-gfx1100.md`](docs/vllm-vision-gfx1100.md)
- **`--default-chat-template-kwargs '{"enable_thinking": false}'`** on both:
  gateways that strip per-request `chat_template_kwargs` otherwise make every
  call pay hidden thinking tokens.

## Quick Start

### Run from pre-built image (no build)

```bash
docker pull nas.kadrlik.home:3042/mkadrlik/vllm-radiance-p2p:latest
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
  nas.kadrlik.home:3042/mkadrlik/vllm-radiance-p2p:latest \
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

Measured on 3× RX 7900 XTX, radiance 0.5.7 / vLLM 0.26.0, thinking disabled
server-side, MTP depth 2 where noted. Repro: `scripts/batch_curve.py`.

| Config | Single-stream | Batch curve | KV pool | Notes |
|--------|--------------|-------------|---------|-------|
| Qwen3.8-27B W8A8 TP2 (radiance A/B) | 20.4 t/s | — | ~90K tok | superseded 2026-09-05 |
| **Qwen3.8-27B AWQ TP2 + MTP** | **23.8–28.9 t/s** | 12.2@8 · 10.3@32 → **~380 agg** | **163K tok** | prod worker; step time ~131 ms is TP2 AR-bound |
| **Ornith-1.5-9B AWQ TP1** | **80–86 t/s** | 49.5@4 (capped; ≥5 cliffs — see docs) | **274K tok + vision** | prod facade; single card, zero AR |
| 27B Quark (CUDA-graph) | ~22 tok/s | tg128, MTP off | | legacy profile |
| 35B-A3B Quark (CUDA-graph) | ~19 tok/s | tg128, MTP off | | legacy profile |
| AWQ (eager) | varies | Depends on model size | | generic template |

The 27B single-stream number is dominated by the TP2 all-reduce floor (one-shot
custom AR is NOT VIABLE on RDNA3 consumer PCIe — see
`fix/fast-reduce-mtp-capture` PR + the flush/LL protocol spec). Quant choice
buys the bandwidth residual only. For single-stream latency on this silicon:
run small models TP1 (9B = 4× the 27B's decode), reserve TP2 for capacity.

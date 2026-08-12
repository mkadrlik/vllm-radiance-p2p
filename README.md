# vLLM + Radiance TP2 P2P on AMD RDNA3 (gfx1100)

**Status:** Production · **Date:** 2026-08-12 · **Host:** gpu-node (TRX50 WS, 2× RX 7900 XTX gfx1100)

**Radiance:** [StillDeadcode/radiance](https://codeberg.org/StillDeadcode/radiance) v0.5.7 — custom vLLM fork with GDN hybrid linear-attention, MTP self-spec, dynamic draft, and ROCm-native TP2 P2P.

## Quick Start

```bash
# Qwen3.6-27B-Quark-W8A8 (Radiance, TP2)
docker compose --profile radiance-27b up -d

# Qwen3.6-35B-A3B-Quark-W8A8 (Radiance, TP2)
docker compose --profile radiance-35b up -d

# AWQ models via vllm-rocm-tq (single GPU)
docker compose --profile awq up -d
```

All profiles share the repo root. Only one profile runs at a time (all map to port 13313, except awq which uses 13309).

## Models & Performance

| Model | Decode | Profile | Notes |
|-------|--------|---------|-------|
| Qwen3.6-27B-Quark-W8A8 | **21.9 tok/s** | radiance-27b | W8A8 INT8, GDN hybrid FLA |
| Qwen3.6-35B-A3B-Quark-W8A8 | **19.5 tok/s** | radiance-35b | MoE 3B active, W8A8 INT8 |
| Qwen3.6-35B-A3B-AWQ | **27.4 tok/s** | awq | W4A16, single GPU |

## Architecture

```
vllm-radiance-p2p/
├── README.md                    # This file
├── AGENTS.md                    # Engineering notes for agents
├── docker-compose.yml           # Single compose — use --profile to select
├── Dockerfile                   # Radiance image (stilldeadcode/vllm-radiance:0.5.7)
├── .env.example                 # All env vars documented
├── .gitignore
├── data/                        # Runtime caches (gitignored)
│   ├── cache/                   # HuggingFace weights
│   └── radiance-cache-{model}/  # Compile caches (Triton/Inductor/AITER)
└── vllm-rocm-tq/                # AWQ inference path
    ├── Dockerfile               # Builds on vllm/vllm-openai-rocm:v0.24.0
    ├── entrypoint.sh            # GPU selection + vLLM launch
    └── data/                    # AWQ caches (gitignored)
        ├── hf-cache/
        └── triton-cache/
```

## The 8-Fix Chain

These are the fixes that took to boot Radiance TP2 on gfx1100:

1. **`s_wait_storecnt` → `s_waitcnt expcnt(0)`** in `router_gemm.hip` — gfx1201-only instruction broke hipcc on gfx1100
2. **`gfx1100-GEMM-A8W8.json`** baked into image — AITER Triton W8A8 GEMM tuning config; missing = engine init death
3. **`RADIANCE_FAST_REDUCE=0`** — Radiance custom all-reduce broke TP allgather (600s timeout)
4. **`NCCL_PROTO=Simple`** — reliable for non-xGMI PCIe P2P (avoids LL/LL128 issues)
5. **`RADIANCE_RUN_BWTEST=0`** — startup P2P bandwidth sweep, run concurrently across crash-looping boots, caused SMU hangs → kernel panics → host reboots
6. **Removed `--disable-torch-compile`** — not a valid vLLM 0.26 flag → instant CLI death, 14 restarts
7. **`--language-model-only`** — ViT 128 GiB profile OOM (vision tower in checkpoint)
8. **Wiped corrupted Triton autotuner cache** — 0-byte JSONs from crash-loop era → `JSONDecodeError`; also TP0/TP1 race on shared cache

## P2P Findings

### What Worked

- **P2P is ENABLED** in production — the 600s allgather hang was `RADIANCE_FAST_REDUCE`, NOT P2P. `FAST_REDUCE=0` + `NCCL_PROTO=Simple` → P2P works.
- `NCCL_P2P_DISABLE=1` is **NOT** needed on this hardware (IOMMU off, no ACS).
- `GPU_MAX_HW_QUEUES=1` — prevents amdgpu scheduler contention on gfx1100.

### What Didn't

- `RADIANCE_FAST_REDUCE=1` — custom all-reduce hangs on TP allgather (600s timeout → watchdog kill)
- `RADIANCE_RUN_BWTEST=1` — concurrent P2P bandwidth sweep across crash-looping boots → SMU hang → kernel panic → host reboot
- MTP with 512-token CUDA graph capture sizes — graph pools 5.9 GiB, doesn't fit 24 GiB per GPU
- `NCCL_PROTO=LL` or `LL128` — unreliable on non-xGMI PCIe P2P

## Environment Variables

All are in `.env.example`. Key ones:

```bash
HF_TOKEN=                          # HuggingFace token for model download
HIP_VISIBLE_DEVICES="0,1"          # TP2: both GPUs
GPU_ID=0                           # AWQ path: which GPU
VLLM_HOST_PORT=13309               # AWQ path: host port mapping
```

## Troubleshooting

- **Boot dies mid-autotune:** `find <cache> -name '*.json' -size 0 -delete` then restart
- **TP0/TP1 race on shared cache:** Use separate cache dirs per container (already configured)
- **GDN prefill warmup warnings:** Normal IF first inference still works; if warmup fails, wipe Triton cache

## Rollback

- 27B: legacy `compose/tp2-27b-quark-radiance.yml` (lemonade image, P2P enabled, 16.5 tok/s)
- 35B: legacy AWQ path via `vllm-rocm-tq/` (single GPU, no TP2)

## Upstream

- **Radiance:** https://codeberg.org/StillDeadcode/radiance (v0.5.7)
- **vLLM:** https://github.com/vllm-project/vllm (v0.26.x / v0.1.dev1)
- **ROCm:** 7.14, gfx1100
- **AITER:** 0.1.17 — AMD Intel Triton Enhanced Runtime

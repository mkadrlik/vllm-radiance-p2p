# vLLM + Radiance TP2 P2P on AMD RDNA3 (gfx1100)

**Status:** Production · **Date:** 2026-08-11 · **Host:** big-chungus (TRX50 WS, 2× RX 7900 XTX gfx1100)

**Radiance:** [StillDeadcode/radiance](https://codeberg.org/StillDeadcode/radiance) v0.5.7 — custom vLLM fork with GDN hybrid linear-attention, MTP self-spec, dynamic draft, and ROCm-native TP2 P2P.

## Quick Reference

| Model | Decode | Config | Notes |
|-------|--------|--------|-------|
| Qwen3.6-27B-Quark-W8A8 | **21.9 tok/s** | CUDA-graph, MTP off | 27B W8A8 INT8, GDN hybrid FLA |
| Qwen3.6-35B-A3B-AWQ | **27.4 tok/s** | Eager, P2P enabled | MoE 3B active, W4A16 compressed-tensors |
| Prior lemonade stack | 16.5 tok/s | baseline | — |

## Hardware

- **CPU:** TRX50 workstation
- **GPU:** 2× AMD Radeon RX 7900 XTX (gfx1100 / RDNA3)
- **PCIe:** IOMMU off, no ACS — direct P2P works
- **RAM:** 4× 16 GiB DDR5-4800 quad-channel
- **Power:** 300 W per card (`rocm-smi --setpoweroverdrive 300`)

## The 8-Fix Chain

These are the fixes that took to boot Radiance TP2 on gfx1100:

1. **`s_wait_storecnt` → `s_waitcnt expcnt(0)`** in `router_gemm.hip` — gfx1201-only instruction broke hipcc on gfx1100
2. **`gfx1100-GEMM-A8W8.json`** baked into image — AITER Triton W8A8 GEMM tuning config; missing = engine init death
3. **`RADIANCE_FAST_REDUCE=0`** — Radiance custom all-reduce broke TP allgather (600s timeout)
4. **`NCCL_PROTO=Simple`** — reliable for non-xGMI PCIe P2P (avoids LL/LL128 issues)
5. **`RADIANCE_RUN_BWTEST=0`** — startup P2P bandwidth sweep, run concurrently across crash-looping boots, caused SMU hangs → kernel panics → host reboots (the "machine rebooted" saga)
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

## Architecture

```
vllm-radiance-p2p/
├── README.md                    # This file
├── AGENTS.md                    # Engineering notes for agents
├── compose/
│   ├── tp2-27b-quark-radiance.yml  # Qwen3.6-27B-Quark-W8A8 (TP2)
│   └── tp2-35b-a3b-quark-radiance.yml  # Qwen3.6-35B-A3B-AWQ (TP2)
├── scripts/
│   ├── run-vllm35b-awq-p2p.sh    # Qwen3.6-35B-A3B entrypoint (lemonade-tq image)
│   ├── radiance_build_state.sh   # Build monitor script
│   └── radiance_deploy_notice.sh # Deploy completion notice
├── data/
│   ├── radiance-cache-27b-quark/    # Compile cache (triton/aiter/inductor/vllm)
│   └── radiance-cache-35b-a3b-quark/
├── results/
│   ├── p2p_13313.json          # BetterBench: 35B-A3B AWQ TP2 P2P
│   └── p2p_13313_ram2.json     # BetterBench: 35B-A3B AWQ TP2 P2P (run 2)
├── docs/
│   └── vllm-radiance-27b-deploy.md  # 8-fix chain documentation
└── vllm-rocm-tq/              # AWQ inference wrapper (non-radiance)
    ├── Dockerfile
    ├── docker-compose.yml
    ├── entrypoint.sh
    ├── README.md
    └── AGENTS.md
```

## Runtime Environment

### Image: `vllm-radiance:gfx1100`
- Source build from StillDeadcode/radiance v0.5.7
- PyTorch 2.11.0+rocm7.14, Triton 3.6.0, AITER 0.1.17
- gfx1100-specific GEMM tuning baked in

### Image: `lemonade-tq` (35B-AWQ)
- vLLM v0.1.dev1+gf2069b005.rocm724
- Host ROCm 7.14 — handles hipIpcGetMemHandle correctly on gfx1100
- Stock container images (nightly, rocm7.14) have a regression that breaks P2P

### Key Environment Variables

```bash
HIP_VISIBLE_DEVICES="0,1"              # or ROCR_VISIBLE_DEVICES
NCCL_PROTO=Simple                       # P2P over PCIe, not xGMI
RADIANCE_FAST_REDUCE=0                  # Disable custom all-reduce
RADIANCE_RUN_BWTEST=0                   # Disable startup bandwidth sweep
GPU_MAX_HW_QUEUES=1                     # gfx1100 scheduler quirk
HSA_OVERRIDE_GFX_VERSION=11.0.0         # gfx1100 detection
VLLM_WORKER_MULTIPROC_METHOD=spawn     # Required for TP2
```

## Benchmarks

### Qwen3.6-35B-A3B-AWQ (TP2 P2P)
- **Decode:** 27.4 tok/s single-stream
- **TTFT:** ~60-85ms
- **Prefill:** ~1300-1800 tok/s (pp2048)
- **Source:** BetterBench, 20 runs per category, 5 concurrency levels

### Qwen3.6-27B-Quark-W8A8 (TP2 CUDA-graph)
- **Decode:** 21.9 tok/s (CUDA-graph mode)
- **Prefill:** ~2000 tok/s
- **MTP:** OFF (19.3 vs 19.5 t/s, no win)

## Deployment

### 27B Quark (Radiance)
```bash
docker compose -f compose/tp2-27b-quark-radiance.yml up -d
```

### 35B-A3B AWQ (lemonade-tq)
```bash
docker compose -f compose/tp2-35b-a3b-quark-radiance.yml up -d
```

### Boot Time
~7-9 minutes to `/health` 200 (compile cached in `data/radiance-cache-*`).

### Troubleshooting
- **Boot dies mid-autotune:** `find <cache> -name '*.json' -size 0 -delete` then restart
- **TP0/TP1 race on shared cache:** Use separate cache dirs per container
- **GDN prefill warmup warnings:** Normal IF first inference still works; if warmup fails, wipe Triton cache

## Rollback

- 27B: `docker-compose.tp2-27b-quark.yml` (lemonade image, P2P enabled, 16.5 tok/s)
- 35B: `docker-compose.yml` in `vllm-rocm-main/` (source build, NCCL_P2P_DISABLE=1)

## Upstream

- **Radiance:** https://codeberg.org/StillDeadcode/radiance (v0.5.7)
- **vLLM:** https://github.com/vllm-project/vllm (v0.26.x / v0.1.dev1)
- **ROCm:** 7.14, gfx1100
- **AITER:** 0.1.17 — AMD Intel Triton Enhanced Runtime

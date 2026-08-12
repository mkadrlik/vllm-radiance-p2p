# AGENTS.md — vLLM + Radiance TP2 P2P Engineering Notes

## Context

This repository documents the deployment and troubleshooting of vLLM with Radiance custom fork for tensor parallel (TP2) P2P on AMD RDNA3 GPUs (RX 7900 XTX, gfx1100).

## Key Engineering Findings

### P2P on gfx1100 — What Actually Works

**Working config:**
```bash
NCCL_PROTO=Simple
RADIANCE_FAST_REDUCE=0
# NCCL_P2P_DISABLE is NOT needed (IOMMU off, no ACS)
```

**Why it works:**
- PCIe P2P works on TRX50 WS with IOMMU off and no ACS barriers
- `NCCL_PROTO=Simple` is reliable for non-xGMI PCIe P2P
- `RADIANCE_FAST_REDUCE=1` was the actual hang source (custom all-reduce), NOT P2P itself

### The "Machine Rebooted" Saga

`RADIANCE_RUN_BWTEST=1` caused:
1. Startup P2P bandwidth sweep
2. Concurrent across crash-looping boots
3. SMU hangs → kernel panics → host reboots

**Fix:** `RADIANCE_RUN_BWTEST=0`

### Triton Cache Corruption

**Symptom:** `JSONDecodeError` on boot, TP0/TP1 race
**Cause:** 0-byte JSON files from crash-loop era in shared cache
**Fix:**
```bash
find <cache> -name '*.json' -size 0 -delete
# Use separate cache dirs per container (TP0/TP1 race)
```

### CUDA Graph Memory Sizing

**Problem:** 512-token CUDA graph buckets = 5.9 GiB graph pools, doesn't fit 24 GiB
**Fix:** `--compilation-config={"cudagraph_capture_sizes":[1,2,4,8,16,32,64,128],"max_cudagraph_capture_size":128}`
**Result:** Graph pools 5.9→0.5 GiB, KV cache -0.81→+4.46 GiB

### MTP (Multi-Token Prediction)

**Result:** 19.3 vs 19.5 t/s — no win on Quark-W8A8
**Reason:** Draft acceptance low on this model family
**Note:** Community 60-95 t/s MTP numbers are llama.cpp + Q4 GGUF

## Deployment Patterns

### Radiance Image (27B Quark)
- Source build from StillDeadcode/radiance v0.5.7
- gfx1100-specific GEMM tuning baked in
- AITER GEMM (linear) via Triton gemm_a8w8
- FP8 paths OFF (RDNA3 has no native FP8)

### lemonade-tq Image (35B-A3B AWQ)
- vLLM v0.1.dev1+gf2069b005.rocm724
- Host ROCm 7.14 — handles hipIpcGetMemHandle correctly
- Stock images have P2P regression on gfx1100

## Common Pitfalls

1. **`--disable-torch-compile`** — not valid in vLLM 0.26, causes instant CLI death
2. **`NCCL_PROTO=LL` or `LL128`** — unreliable on non-xGMI PCIe P2P
3. **Shared Triton cache** — TP0/TP1 race on corrupt cache files
4. **ViT 128 GiB OOM** — always use `--language-model-only` for text-only models
5. **GPU scheduler contention** — `GPU_MAX_HW_QUEUES=1` required for gfx1100

## Performance Baselines

| Config | Decode | Notes |
|--------|--------|-------|
| 27B Quark (CUDA-graph) | 21.9 tok/s | MTP off, 128-token cap |
| 35B-A3B AWQ (eager) | 27.4 tok/s | TP2 P2P enabled |
| Prior lemonade stack | 16.5 tok/s | baseline |

## Git Structure

```
vllm-radiance-p2p/
├── README.md              # Deployment guide
├── AGENTS.md              # This file (engineering notes)
├── compose/               # Docker Compose files
├── scripts/               # Entry points and monitors
├── data/                  # Compile caches (gitignored)
├── results/               # Benchmark data
└── docs/                  # Additional documentation
```

## Tech Debt

- [ ] **Gitea Actions secrets not externalized** — GITHUB_TOKEN must be injected via Gitea secret store, not hardcoded. When Gitea secrets API supports encrypted values properly, the TODO in `.gitea/workflows/mirror-and-push.yml` can be resolved.

## Upstream Links

- Radiance: https://codeberg.org/StillDeadcode/radiance
- vLLM: https://github.com/vllm-project/vllm
- ROCm: 7.14, gfx1100
- AITER: 0.1.17

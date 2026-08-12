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

### Radiance Image (27B/35B Quark)
- Source build from StillDeadcode/radiance v0.5.7
- gfx1100-specific GEMM tuning baked in
- AITER GEMM (linear) via Triton gemm_a8w8
- FP8 paths OFF (RDNA3 has no native FP8)

### AWQ Image
- vllm/vllm-openai-rocm:v0.24.0 base
- Single GPU, no TP2

## 27B vs 35B Gotcha

The 35B-A3B profile uses `--max-model-len=32768` while 27B uses `--max-model-len=65537`. **Do not set 65k on 35B** — it OOMs at TP2. The 35B-A3B is a larger model; 32k is the stable ceiling. Both share identical build context, Dockerfile, env vars, and all other arguments.

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

## Upstream Links

- Radiance: https://codeberg.org/StillDeadcode/radiance
- vLLM: https://github.com/vllm-project/vllm
- ROCm: 7.14, gfx1100
- AITER: 0.1.17

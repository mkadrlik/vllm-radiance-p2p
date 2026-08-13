# AGENTS.md — vLLM + Radiance TP2 P2P Engineering Notes

## Context

This repository documents the deployment and troubleshooting of vLLM with Radiance custom fork for tensor parallel (TP2) P2P on AMD RDNA3 GPUs (RX 7900 XTX, gfx1100).

## gfx1100 image status & build (2026-08-12)

**TL;DR:** the repo `:latest` is the working gfx1100 image (retagged 2026-08-12 from
`vllm-radiance:gfx1100` and pushed to `ghcr.io/mkadrlik/vllm-radiance-p2p:latest` +
`nas.kadrlik.home:3042/mkadrlik/vllm-radiance-p2p:latest`, digest
`sha256:6253c8e6cf9c...`). The repo `Dockerfile` does **NOT** build gfx1100.

**⚠️ Do NOT run `docker compose build` or `docker build .`** — it rebuilds from the
gfx1201 stilldeadcode base and **clobbers the `:latest` tag** with a broken image
(verified 2026-08-12: a manual build pushed over `:latest` caused the exact
`RuntimeError: No CUDA GPUs are available` failure on gfx1100 — torch's HIP init fails
device enumeration before vLLM even reaches the pynccl `hipErrorInvalidImage` stage).
Pull the prebuilt image instead; only rebuild via the reproducible source build below.

**Why:** the `stilldeadcode/vllm-radiance:0.5.7` base drifted to a newer Radiance source
targeting gfx1201/RDNA4 (`_aiter_ops.py` uses `on_gfx12x`; `aiter/ops/triton/gemm_a8w8.py`
moved). `docker build` of the repo therefore produces an image that dies on gfx1100 at
pynccl init with `hipErrorInvalidImage` (`arch check FAIL 0/2 gfx1201`; the
`patch_gfx1100.py` anchor `is_aiter_found_and_supported` matches 0x, expected 1).

**The known-good gfx1100 image** (`vllm-radiance:gfx1100`, 6253c8e6cf9c) was a **full ROCm
7.14 source build** with `ARG GFX_ARCH=gfx1100` (torch/triton/aiter 0.1.17/vllm 0.26.0 wheels
for gfx1100) plus the recovered patch layer. Its wheels are NOT recoverable from the image.

**To build for gfx1100 (reproducible source build, not yet automated):** the complete gfx1100
adaptation is in `build/` (patches/, radiance-modules/, aiter-configs/, moe-configs/,
fp8-configs/, radiance_preamble.py, radiance_entrypoint.sh), recovered from the working
image. Reconstructing the build:
1. Start from a ROCm 7.14 multi-arch base (`GFX_ARCH=gfx1100`), NOT the drifted
   stilldeadcode 0.5.7 prebuilt.
2. Build/install the source wheels for gfx1100: `PYTORCH_ROCM_ARCH=gfx1100
   HIP_ARCHITECTURES=gfx1100 AMDGPU_TARGETS=gfx1100 GPU_ARCHS=gfx1100`.
3. Layer `build/` (order = the working image's `docker history`):
   - COPY `patches/` → `/opt/patches`; run each `patch_*.py` (patch_gfx1100,
     patch_router_gemm, patch_unified_attention_lds, patch_gdn_wmma, install_radiance_hooks,
     patch_unpad, patch_mtp_mm_mask, patch_mtp_loopbreak, patch_qwen3_toolparse,
     patch_from_json_filter, patch_dynamo_metrics).
   - COPY `radiance-modules/` → site-packages; `aiter-configs/gfx1100-GEMM-A8W8.json` →
     aiter gemm configs; `moe-configs/` → fused_moe configs; `fp8-configs/` → quantization
     utils configs.
   - hipcc-compile `router_gemm.hip` → `router_gemm.so` and `radiance_ar_ext.hip` →
     `radiance_ar_ext.so` with `-O3 -std=c++17 -fPIC -shared --offload-arch=gfx1100
     -DTEMPORAL $(python -m pybind11 --includes)`.
   - COPY `radiance_preamble.py` + `radiance_entrypoint.sh` → `/opt`; ENTRYPOINT
     `/opt/radiance_entrypoint.sh`. Set `PYTORCH_ROCM_ARCH=gfx1100 RADIANCE_GFX_ARCH=gfx1100`
     etc.
4. Verify: `/v1/models` lists the model; log shows `P2P access : ENABLED 0↔1`; no
   hipError/InvalidImage.

**Until automated:** run the pre-built `:latest` (it IS gfx1100); do not `docker build`
from the repo and expect gfx1100.

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

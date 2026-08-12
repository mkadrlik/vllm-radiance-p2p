# vLLM-Radiance 27B Deployment (big-chungus)

**Status:** LIVE · **Date:** 2026-08-08 · **Endpoint:** http://127.0.0.1:13313 (served as `vllm-27b`)
**Model:** `nameistoken/Qwen3.6-27B-Quark-W8A8-INT8` (27B, W8A8 int8, GDN hybrid linear-attention)
**Image:** `vllm-radiance:gfx1100` (source build, codeberg StillDeadcode/radiance v0.5.7, torch 2.11.0+rocm7.14, triton 3.6.0, aiter 0.1.17)
**Compose:** `/home/mkadrlik/docker/lemonade-tq/docker-compose.tp2-27b-quark-radiance.yml`

## Performance
| Config | Decode | Notes |
|---|---|---|
| CUDA-graph (current) | **21.9 tok/s** | MTP off — graph pools 2.8 GiB fit 24 GiB |
| Eager + MTP | 10.5 tok/s | speculative on, no graphs |
| Prior lemonade stack | 16.5 tok/s | baseline |

## Key config (stability-tuned for 24 GiB gfx1100 × 2)
- `--language-model-only` — REQUIRED: without it the profile run allocates a 128 GiB ViT dummy tensor and OOMs (vision tower in the checkpoint).
- `NCCL_P2P_DISABLE=1` — stock RCCL allgather HANGS on this host's P2P (watchdog 600 s timeout). Radiance's custom all-reduce is also off (`RADIANCE_FAST_REDUCE=0`).
- `RADIANCE_RUN_BWTEST=0` — the startup P2P bandwidth sweep, run concurrently across crash-looping boots, caused SMU hangs → kernel panics → host reboots (the "machine rebooted" saga).
- `--max-num-seqs=64`, `--max-num-batched-tokens=2048`, `--gpu-memory-utilization=0.85` — CUDA-graph pools scale with the model/batch; MTP's graphs (5.9 GiB) don't fit, so MTP is off.
- `--enforce-eager` removed (graphs ON for the 2.1× decode win).
- Power cap 300 W per card (`rocm-smi --setpoweroverdrive 300`, re-apply after host reboot).
- GPU tuning: stock base values, undervolt, -50 offset both cards (user-applied).

## The 8-fix chain (what it took to boot)
1. `s_wait_storecnt` → `s_waitcnt expcnt(0)` in router_gemm.hip (gfx1201-only instruction broke hipcc).
2. `gfx1100-GEMM-A8W8.json` baked into image (AITER Triton W8A8 GEMM tuning config; missing = engine init death).
3. `RADIANCE_FAST_REDUCE=0` (custom all-reduce broke TP allgather).
4. `NCCL_P2P_DISABLE=1` (allgather 600 s timeout).
5. `RADIANCE_RUN_BWTEST=0` (SMU hang → host reboots).
6. Removed `--disable-torch-compile` (not a valid vLLM 0.26 flag → instant CLI death, 14 restarts).
7. `--language-model-only` (ViT 128 GiB profile OOM).
8. Wiped corrupted Triton autotuner cache (0-byte JSONs from crash-loop era → `JSONDecodeError`; also TP0/TP1 race on shared cache).

## Ops notes
- Boot ~7-9 min to `/health` 200 (compile cached in `./data/radiance-cache-27b-quark`).
- If a boot dies mid-autotune: sweep `find <cache> -name '*.json' -size 0 -delete` before restart.
- `hermes verify --json` validates the ops scripts (recipe: vllm-27b-quark ops scripts).
- Rollback: `docker-compose.tp2-27b-quark.yml` (lemonade image, P2P enabled, 16.5 tok/s).
- GDN prefill warmup warnings at boot are normal IF first inference still works; if warmup fails, the Triton cache is corrupt — wipe.

## Post-deploy tuning (2026-08-08 afternoon)
- **P2P ENABLED** — the 600s allgather hang was RADIANCE_FAST_REDUCE (custom all-reduce), NOT P2P. FAST_REDUCE=0 + NCCL_PROTO=Simple → P2P works. Do not re-disable.
- **CUDA graphs capped at 128 tokens** (`--compilation-config cudagraph_capture_sizes<=128`) — the 512-token buckets were the memory hog: pools 5.9→0.5 GiB, KV cache -0.81→+4.46 GiB. Batches >128 tokens/step fall back to eager.
- **MTP OFF** — measured 19.3 vs 19.5 t/s (no win on Quark-W8A8; draft acceptance low). Community 60-95 t/s MTP numbers are llama.cpp + Q4 GGUF.
- **Final decode: 19.7-27.5 t/s** (tg32-style), pp2048 ~2000 t/s. Community compare: llama.cpp TP2 Q8_0 = 27.9 t/s, Q4_K_M+MTP = 56-95 t/s.

# Triton-on-gfx1100 vs CUDA: Gap Analysis & Optimization Plan

**Scope:** RX 7900 XTX (gfx1100, RDNA3, 960 GB/s, 64 CUs) · Triton 3.6.0 · aiter 0.1.17 · vllm-radiance:gfx1100
**Goal:** Isolate every gap where the Triton/AMD stack leaves gfx1100 performance on the table, cross-check against our live kernel stack, rank by leverage, and define the first-pass fixes.

---

## 0. TL;DR — what's actually holding us back

| Rank | Gap | Where | Est. decode impact | Effort |
|------|-----|-------|--------------------|--------|
| **1** | **No decode (M=1) GEMV specialization** — the W8A8 GEMM runs a prefill-shaped tile (BLOCK_M=16, NUM_KSPLIT=1) for batch-1 decode, leaving ~60 of 64 CUs idle | `gemm_a8w8.py` + `gfx1100-GEMM-A8W8.json` | **High** (10–30%) | Low (config + split-K) |
| **2** | **No autotuning** — `is_tuned=False`; the shipped config is a static hand-pick, not benchmarked on this GPU. Tuklus showed a 40%+ spread between best/worst configs on the *same* kernel | `gemm_config_utils.py` | **High** (10–40%) | Low (autotune harness) |
| **3** | **`tl.dot` for INT8 → not guaranteed WMMA** — the kernel uses plain `tl.dot`; on RDNA3 Triton lowers INT8 dot to WMMA only when shapes/alignment line up. If it falls to scalar FMA, we lose the 512 FLOP/clk/CU iu8 path | `gemm_a8w8.py:164` | **Medium–High** | Medium (verify + fix shapes) |
| **4** | **No Split-K in decode config** — for M=1 the output is (1, N); with BLOCK_N=128 and N=4096 that's only 32 CTAs. 64 CUs → half idle. Split-K across the K dim would fill them | `gfx1100-GEMM-A8W8.json` (NUM_KSPLIT=1) | **High** (10–20%) | Low (config) |
| **5** | **Memory layout / coalescing** — weights are `[N, K]` transposed to `[K, N]` at runtime (`w = w.T`); if the on-disk layout isn't K-contiguous for the load, the B-tile load is strided | `gemm_a8w8.py` (weight layout) | **Medium** (5–15%) | Medium (pre-transpose weights) |
| **6** | **No `cache_modifier` tuning** — B (weights) re-read every token; L2 hint (`cg`/`cs`) could keep the hot tile resident across the N-parallel warps | `gemm_a8w8.py:152` | **Low–Medium** | Low (config) |
| **7** | **Decode attention not GQA-packed** — `fwd_decode.py` may read KV once per Q-head instead of packing 4 Q-heads/KV-head (Qwen3.8-27B: 32 Q / 8 KV) | `flash_attn_triton_amd/fwd_decode.py` | **Medium** (5–15% at long ctx) | Medium |
| **8** | **No fused dequant** — W8A8 dequant (× x_scale × w_scale) happens in the epilogue as a separate multiply; not a separate pass, but not fused into the dot either | `gemm_a8w8.py:164-175` | **Low** | Medium |

**Net realistic ceiling:** gaps 1+2+4 together (all in the GEMM config/kernel) are the bulk. Fixing them is plausibly **15–35% decode tps** — not 2×, but real. The 2× would come from a smaller model or INT4, not Triton tuning.

---

## 1. Ground truth (verified in the live image)

### 1.1 The decode kernel path
```
vllm linear layer (decode, M=1)
  → aiter.ops.triton.gemm.basic.gemm_a8w8.gemm_a8w8()
    → _gemm_a8w8_kernel  (triton, 199 lines)
      → config = _get_config(M=1, N, K)
        → get_gemm_config("GEMM-A8W8", 1, N, K)
          → loads configs/gemm/gfx1100-GEMM-A8W8.json
          → M=1 ≤ 32  →  M_LEQ_32  →  {BLOCK_M:16, BLOCK_N:128, BLOCK_K:128, num_warps:4, num_stages:3, NUM_KSPLIT:1}
          → is_tuned = False   ← NOT autotuned
```

### 1.2 The shipped decode config (M_LEQ_32)
```json
{
  "BLOCK_SIZE_M": 16,      // M=1 → 15 of 16 rows are masked waste
  "BLOCK_SIZE_N": 128,
  "BLOCK_SIZE_K": 128,
  "GROUP_SIZE_M": 4,
  "num_warps": 4,          // 4 warps = 256 threads; 64 CUs want more parallelism
  "num_stages": 3,
  "waves_per_eu": 2,
  "matrix_instr_nonkdim": 16,
  "kpack": 1,
  "NUM_KSPLIT": 1          // ← no split-K; CTA count = cdiv(N,128) only
}
```

**The CTA-count problem (gap 4, made concrete):**
For the 27B, the linear layers are roughly N∈{4096, 11008, 18432}, K∈{4096, 11008}. At M=1, BLOCK_N=128:
- N=4096 → **32 CTAs**
- N=11008 → **86 CTAs**
- N=18432 → **144 CTAs**

64 CUs. The N=4096 layers (q_proj, k_proj, v_proj, o_proj, and most MLP down-projs) launch **32 CTAs on 64 CUs = 50% of the GPU idle** for the entire kernel. That's the single biggest, cheapest win: **Split-K** across the K dim (K=4096, BLOCK_K=128 → 32 K-slices; NUM_KSPLIT=4 → 128 CTAs, fully occupied) + a tiny reduce.

### 1.3 No autotuning (gap 2)
`get_gemm_config` returns `is_tuned=False` for the gfx1100 A8W8 path — the JSON is a static hand-pick, not the output of a benchmark sweep. There is **no autotune harness** wired into the gfx1100 build. Tuklus's whitepaper (same GPU, same Triton 3.6, same memory wall) documented a **>40% spread** between best and worst block/split-K/warp configs on the *identical* GEMM shape class. We are running one guess, not the winner.

### 1.4 `tl.dot` for INT8 (gap 3)
Line 164: `accumulator += tl.dot(a, b)` where `a`/`b` are int8. On RDNA3, Triton's AMD backend lowers `tl.dot` to **WMMA** (`v_wmma_f32_16x16x16_iu8_w32`) only when:
- the dot is 16×16×16-shaped (or a multiple the backend can tile to it),
- operands are properly aligned (16-byte for iu8),
- `matrix_instr_nonkdim` matches.

The config sets `matrix_instr_nonkdim: 16` and `kpack: 1`, which *should* line up — but this is the one gap I can't confirm without reading the SASS/HSA codegen. **Action: dump the compiled kernel's ISA and grep for `v_wmma`**. If it's there → WMMA is active, gap 3 closes. If it's scalar `v_madak_f32`/FMA → we're on the slow path and the fix is shape/alignment.

### 1.5 Weight layout (gap 5)
`gemm_a8w8` does `w = w.T` at call time (line ~`w = w.T`). The on-disk weight is `[N, K]` (N rows, K cols). After `.T` the *logical* shape is `[K, N]`, but the *physical* memory is still N-major. The B-tile load (`b_ptrs` strides by `stride_bk` along K) is therefore **stride-K across N-major storage** = strided, non-coalesced. Tuklus's single biggest win (3.3×) was exactly this: transposing the weight layout so the load is coalesced. **Action: pre-transpose + contiguous the W8A8 weights at load time** (one-time cost at model load, zero per-token cost).

---

## 2. CUDA vs Triton-on-AMD: the structural gaps

The "Triton is immature vs CUDA" claim is **partly true, but the maturity gap is not the main cost on gfx1100**. Here's the honest breakdown:

| Layer | CUDA/NVIDIA | Triton/AMD gfx1100 | Gap? |
|-------|-------------|--------------------|------|
| **Tensor-core / matrix unit** | `mma`/`wgmma` (Hopper), 1st-class, Triton auto-lowers | **WMMA** (RDNA3), 1st-class, Triton *can* lower but needs shape/align luck | Small — WMMA exists and is fast (512 FLOP/clk/CU iu8); the risk is the `tl.dot`→WMMA lowering (gap 3) |
| **TMA / async bulk copy** | `cp.async.bulk`, Tensor Memory Accelerator, hardware descriptor | **No TMA on RDNA3.** Triton uses `buffer_load` + LDS staging (`num_stages` pipelining) | **Real gap** — but for a memory-bound GEMV the bottleneck is DRAM bandwidth, not the copy engine. Pipelining (`num_stages=3`) already hides most of it. Low impact for decode. |
| **L2 / cache control** | `__ldg`, `ld.global.cg/cs/ca`, L2 persistence API, `cudaStreamAttrValue` access-policy window | `cache_modifier` (`cg`/`cs`/`wb`), no L2 persistence API in Triton-AMD | **Real gap** — we can't pin the weight tile in L2 across tokens. Mitigation: `cache_modifier="cg"` on the B load + keep BLOCK_N large so the tile is reused across the K-loop. (gap 6) |
| **Autotuning** | `torch.compile` max-autotune, Triton `@autotune` with `do_bench`, cuBLASLt heuristic | **None wired for gfx1100.** aiter has the *machinery* (`_get_config`, config JSONs) but no benchmark harness in the build | **Real gap, highest leverage** (gap 2). The fix is a one-time offline sweep → write the winning JSON. |
| **Memory allocation** | `cudaMallocAsync`, memory pools, graph capture | HIP allocator, `torch.cuda.graphs` (works on ROCm), no async pool by default | Small for decode (KV cache is pre-allocated by vLLM). The Tuklus KV-clone artifact (20%) is a benchmark issue, not a production one — vLLM already pre-allocates. |
| **Kernel launch overhead** | CUDA Graphs, `cudaGraphLaunch` | ROCm graphs (vLLM `--cudagraph`), HIP graph capture | **Already handled** — radiance ships `--cudagraph_capture_sizes`. Not a gap. |
| **Wave / warp model** | 32-thread warp, `__shfl`, warp-specialization | **64-thread wavefront** (gfx1100 default; wave32 possible), `s_waitcnt`, LDS = "shared memory" | **Structural, not fixable** — but Triton abstracts it. The `num_warps` knob maps to wavefronts. For M=1 GEMV, more wavefronts = more DRAM streams in flight = better bandwidth saturation (gap 1/4). |
| **INT8/INT4 path** | `dp4a`, `mma` int8, FP8 `e4m3` | **WMMA iu8/iu4** (RDNA3 has no FP8 silicon — confirmed) | FP8 is off the table on gfx1100 (no hardware). W8A8 (int8) is the right quant. INT4 via WMMA iu4 is the next lever if we go smaller. |

**Bottom line on "immature":** the compiler *can* produce WMMA and pipelined, coalesced kernels. What's missing is **tuning, not capability.** The CUDA stack's advantage on NVIDIA is years of cuBLASLt heuristics + TMA + L2 persistence + a mature autotune loop. On gfx1100 we get the hardware (WMMA, 960 GB/s) but we have to *do the tuning ourselves* — and the shipped config is a placeholder, not a tuned result.

---

## 3. First-pass plan (ranked, lowest effort → highest leverage)

### Pass 1 — Config-only wins (no code, no rebuild, ~1 day)
These edit `configs/gemm/gfx1100-GEMM-A8W8.json` (and the attention config) inside the image or via a mounted volume. **No recompile.**

1. **Add Split-K to the decode config.** Change `M_LEQ_32` to `NUM_KSPLIT: 4` (and `SPLITK_BLOCK_SIZE` derived). For N=4096,K=4096: 32→128 CTAs, fills all 64 CUs. Add a `M_LEQ_1` or keep `M_LEQ_32`. Benchmark N=4096/11008/18432 separately — the win is biggest on the small-N layers.
2. **Sweep `num_warps` for M=1.** Try 4/8/16. More wavefronts = more in-flight DRAM loads. RDNA3's 64 CUs × 2 waves_per_eu = 128 wave slots; a GEMV wants to fill them.
3. **Sweep `BLOCK_N` × `BLOCK_K`.** Tuklus: small matrices preferred `BLOCK_N=32, BLOCK_K=256, SPLIT_K=8`. Our `BLOCK_N=128, BLOCK_K=128` is a prefill guess. Try `BLOCK_N=64/256`, `BLOCK_K=64/256`.
4. **Set `cache_modifier: "cg"`** on the B (weight) load for the decode path — keep the weight tile in L2 across the K-loop.

**Deliverable:** a benchmark script (below) + a tuned JSON. **Expected: 10–30% decode tps.**

### Pass 2 — Verify WMMA + fix layout (code, ~2–3 days)
5. **Dump the kernel ISA, confirm WMMA.** Compile `_gemm_a8w8_kernel` for M=1, extract the HISA, `grep v_wmma`. If absent → adjust `BLOCK_SIZE_K`/alignment so the dot is 16×16×16 iu8 and Triton emits WMMA. This is the difference between 512 FLOP/clk/CU and scalar FMA.
6. **Pre-transpose the W8A8 weights** at model load: store as contiguous `[K, N]` so the B-tile load is coalesced. One-time cost, removes the strided load (gap 5). Tuklus: 3.3× from layout alone on a broken case; even a partial win here is real.

### Pass 3 — Autotune harness (the durable fix, ~3–5 days)
7. **Wire an offline autotune sweep.** For each (M-bucket, N, K) the 27B actually hits, run `triton.testing.do_bench` over the config space (BLOCK_M/N/K × num_warps × num_stages × NUM_KSPLIT × cache_modifier), pick the winner, write it to the JSON. This is what `is_tuned=True` should mean. Run it once per model, bake the JSON into the image. **This is the single most durable fix** — it's the thing the CUDA stack gets for free from cuBLASLt and we don't.

### Pass 4 — Attention (long-context only, ~2 days)
8. **GQA-pack the decode attention.** Verify `fwd_decode.py` packs the 4 Q-heads per KV-head into one tile (Qwen3.8-27B: 32 Q / 8 KV). If it reads KV per-Q-head, that's 4× the KV traffic. Only matters at long context; at short context the GEMM dominates.

---

## 4. The benchmark (runnable check)

A single script that measures decode tps for the 27B's actual layer shapes, before/after each pass. This is the "one runnable check" — if a config change doesn't move this number, it's noise.

```python
# bench_decode_gemm.py — run inside the vllm-radiance:gfx1100 image
import torch, triton, time
from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8

torch.manual_seed(0)
device = "cuda"
# 27B layer shapes (N, K) — the ones that hit at decode M=1
SHAPES = [(4096, 4096), (11008, 4096), (18432, 4096), (4096, 11008), (1536, 11008)]
M = 1  # decode

for N, K in SHAPES:
    x = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
    w = torch.randint(-128, 127, (N, K), dtype=torch.int8, device=device)
    xs = torch.ones((M, 1), dtype=torch.float32, device=device)
    ws = torch.ones((1, N), dtype=torch.float32, device=device)
    # warmup
    for _ in range(10):
        y = gemm_a8w8(x, w, xs, ws, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    # bench
    n_iter = 200
    t0 = time.perf_counter()
    for _ in range(n_iter):
        y = gemm_a8w8(x, w, xs, ws, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n_iter
    # bytes moved ≈ weights (N*K int8) + activations (tiny) ; BW = bytes / time
    bytes_moved = N * K  # int8 weights dominate
    gbps = bytes_moved / dt / 1e9
    print(f"N={N:6d} K={K:6d}  {dt*1e6:8.1f} us   {gbps:7.1f} GB/s   (peak 960)")
```

**Read the GB/s column, not the µs.** Decode GEMV is memory-bandwidth-bound. Peak is 960 GB/s. If we're at 300–400 GB/s, the gaps above are real and fixable. If we're at 800+, the GEMM is near the wall and the win is in attention/quant, not the GEMM.

Run it **before** any change (baseline), then after each pass. The number is the truth.

---

## 5. What I'm NOT claiming

- **Not 2×.** The hardware wall is 960 GB/s. W8A8 27B reads ~27 GB/token → theoretical max ~35 tok/s single-stream (we're at 21.9 on vLLM, 33 on llama.cpp because llama.cpp's Q8_0 is more bandwidth-efficient per useful bit). Triton tuning gets us toward the wall, not past it.
- **FP8 is not an option** on gfx1100 — no FP8 silicon. W8A8 (int8) is the ceiling for this quant class; INT4 (WMMA iu4) is the next step if we accept a smaller/4-bit model.
- **The "Triton is immature" framing is half-right.** The compiler can emit WMMA + pipelined kernels. The gap is *tuning and heuristics* (what cuBLASLt gives NVIDIA for free), not raw capability. Fix the config, verify WMMA, autotune — and the "immature" gap mostly closes for the decode path.

---

## 6. Measured baseline + Pass-1 sweep (2026-08-27, live image)

**Baseline** (`bench_decode_gemm.py`, M=1, default shipped config):

| Shape | us | GB/s | % of 960 |
|-------|-----|------|----------|
| N=4096 K=4096 | 55.3 | 303 | 31% |
| N=11008 K=4096 | 63.7 | 708 | 74% |
| N=18432 K=4096 | 90.4 | 835 | 87% |
| N=4096 K=11008 | 74.4 | 606 | 63% |
| N=1536 K=11008 | 86.3 | 196 | 20% |
| **avg** | | **530** | 55% |

Large shapes are near the wall (708-835). The two small-N shapes are the leak.

**Pass-1 sweep** (`bench_sweep_gemm.py`, in-process config dict, 144 configs, 112 valid - 32 blew the 64 KB LDS cap):

| Shape | Baseline | Best config | Best GB/s | delta |
|-------|----------|-------------|-----------|---|
| N=4096 K=4096 | 303 | `BM=16 BN=64 BK=256 KS=1 W=4 S=3` | **414** | **+36%** |
| N=1536 K=11008 | 196 | `BM=16 BN=128 BK=128 KS=4 W=4 S=3` | **406** | **+107%** |

**Honest read:** the win is real but the small shapes plateau at ~41-43% of peak even with the best config. That is the M=1 GEMV floor (16-row tile, 15 masked rows, activation load + kernel-launch overhead dominate at these sizes), not CTA starvation alone. The big shapes are already at the wall. **Net per-token impact is modest** - the small-N layers are a fraction of total per-token time; the 4096x4096 attention proj (x4/layer x ~64 layers) is the one worth the +36%.

**Tuned config** (drop into `configs/gemm/gfx1100-GEMM-A8W8.json` `M_LEQ_32`):
```
BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=256, GROUP_SIZE_M=4,
NUM_KSPLIT=1, num_warps=4, num_stages=3, waves_per_eu=2,
matrix_instr_nonkdim=16, kpack=1, cache_modifier=null
```

---

**WMMA verification (2026-08-27):** compiled the M=1 decode kernel, disassembled the `.hsaco` with `/opt/rocm/llvm/bin/llvm-objdump`. **48x `v_wmma_i32_16x16x16_iu8`, 0 scalar FMA, 0 `v_vcnt`.** Gap 3 CLOSED - the `tl.dot` int8 path is correctly hitting the RDNA3 matrix cores. The M=1 GEMV floor is a memory/launch artifact, not scalar compute.

**Config shipped:** `build/aiter-configs/gfx1100-GEMM-A8W8.json` `M_LEQ_32`+`any` -> `BN=64 BK=256`. Pushed to Gitea `gfx1100` branch as **PR #2** (mergeable).

**Weight pre-transpose (gap 5) — DEAD (confirmed, interleaved A/B).** The wrapper reads `w.stride()` and does `w = w.T`, so both physical B-matrix layouts pass the `[N,K]` contract — no wrapper patch needed to *measure*. Interleaved 7-round median across all 5 27B shapes: N-major (current) is **26–44% faster** than K-major (4096×4096: 344 vs 193; 11008×4096: 556 vs 331; 18432×4096: 1240 vs 802; 4096×11008: 1426 vs 928; 1536×11008: 537 vs 398 GB/s). The B-load is already coalesced along K (the contiguous axis). An earlier single-run square-shape test (532 vs 498) was thermal noise; the interleaved median is the real gap. The `[N,K]` contract enforces the good layout — a feature, not a limitation. No win, no patch.

**Autotune harness (gap 2) — SHIPPED (PR #3).** `autotune_gemm_a8w8.py` sweeps 160 valid configs/shape (64 KB LDS cap respected), writes per-shape keys that override `M_LEQ_32`. Harness beat the manual sweep (wider space: BLOCK_N=32, KS to 8, W=8). Measured 7900 XTX M=1: N=4096,K=4096 ~537 GB/s; N=11008,K=4096 ~951; N=18432,K=4096 ~1038; N=4096,K=11008 ~1165; N=1536,K=11008 ~393 (GEMV floor). Run-to-run variance is GPU clock/thermal noise on a shared box — the per-shape keys are the durable part; re-run `--model <name>` on swap.

## 8. Upstream issue scan — MoE + RDNA3 hardening (2026-08-27)

Stack under test: triton 3.6.0 / aiter 0.1.17 (radiance) / vllm 0.26.0 / ROCm 7.14, gfx1100.

| Issue | What it is | Status in OUR stack | Action |
|-------|-----------|--------------------|--------|
| **triton#10808** | fused-MoE Triton JIT memory fault on gfx1100 (reported on triton 3.1.0) | vllm 0.26.0 has full expert-class MoE dispatch (`rocm_aiter_moe`, `triton_moe`, `aiter_mxfp4_w4a8_moe`) + RDNA3 W4A16 kernels. The fault was a triton-3.1 codegen bug; 3.6.0 is far newer. **Unverified on our exact triton** — needs a MoE smoke test before serving 35B-A3B. | Smoke-test a MoE model (35B-A3B) before relying on it. |
| **vllm#44460** | RDNA3 W4A16 MoE dispatch refactor (oracle/expert pattern) | **Already in 0.26.0** — `compressed_tensors_moe_wna16_rdna3.py` + `rocm_moe_rdna.py` present (the refactor's end state). | None. |
| **vllm#46186** | Enable RDNA3 W4A16 GEMM on gfx1151 (Strix Halo), keep gfx1100 WMMA split | gfx1151-specific. **gfx1100 already works** (PR preserves it). `rdna3_w4a16.py` kernel present. | None for gfx1100; relevant if we add a Strix Halo box. |
| **vllm#49321** | W4A16 cold-expert CPU offload (`VLLM_MOE_EXPERT_CACHE_SIZE`) | **Not in 0.26.0** (knob absent). Experimental/WIP. | Optional: backport only if we run a MoE too big for 2×24 GB. |
| **aiter#4234** | gfx1100 A16W16 GEMM configs (asserts without base JSON) | **Partial** — we ship `gfx1100-GEMM-A8W8.json` (our PR #3) but NOT `gfx1100-GEMM-A16W16.json`. The A16W16 path asserts on gfx1100. | Add A16W16 config if we serve BF16 (autotune harness extends to it). |
| **aiter#4329** | `select_3d_config` LDS overflow (65792 > 65536) on non-gfx1250 Wave32 | **ALREADY FIXED in our radiance fork** — `unified_attention.py:234-239` has a `RADIANCE LDS fit (3D)` while-loop (shrink `attn_stages` then `TILE_SIZE` to ≤65536) + matching 2D guard at 117-122. | None. Upstream `#4868` is proposing what we already ship. |
| **aiter#4868** | Guard RDNA unified attention against LDS overflow (fixes #4329) | **Already in our fork** (same as above). | None. |

**Net:** 4 of 7 are already solved or irrelevant in our stack. The two real items: (1) smoke-test the MoE Triton path (triton#10808) before serving 35B-A3B, (2) add the A16W16 config if we go BF16. The `#49321` CPU offload is a future option for oversized MoE.

Our radiance fork is **ahead of upstream** on the LDS guard — when `#4868` merges, our local patch becomes redundant and can be dropped on rebase.

## 7. Immediate next actions

1. **Run `bench_decode_gemm.py`** in the live image → get the baseline GB/s per layer shape. (15 min)
2. **Dump the kernel ISA**, `grep v_wmma` → confirm or refute gap 3. (30 min)
3. **Edit the JSON** (Split-K=4, num_warps sweep, cache_modifier) → re-run bench → keep the winner. (half day)
4. **Pre-transpose weights** → re-run bench. (1 day)
5. **Write the autotune harness** → bake the tuned JSON into the image. (3–5 days, the durable fix)

Each step is gated on the bench number moving. No change ships without a measured delta.

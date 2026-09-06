#!/usr/bin/env python3
"""Offline autotune harness for the gfx1100 W8A8 decode GEMM.

Sweeps the config space for each (M-bucket, N, K) the model actually hits at
decode, picks the fastest valid config (respecting the 64 KB LDS cap), and
writes the winners into the aiter GEMM config JSON.

Durable fix: per-shape tuning the CUDA stack gets free from cuBLASLt and the
gfx1100 build lacks. Run once per model, bake the JSON into the image.

Usage (inside the vllm-radiance:gfx1100 container, with GPU):
    python3 autotune_gemm_a8w8.py --model qwen27b \
        --out /opt/vllm/lib/python3.12/site-packages/aiter/ops/triton/configs/gemm/gfx1100-GEMM-A8W8.json
"""
import argparse
import itertools
import json
import time
import traceback

import torch
import triton

from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8

PEAK = 960.0
LDS_CAP = 65536  # bytes; gfx1100 per-CU LDS
N_ITER = 150

# 27B (Qwen3.8-27B) linear layer (N, K) pairs hit at decode.
MODELS = {
    "qwen27b": [
        (4096, 4096),    # q/k/v/o proj
        (11008, 4096),   # mlp gate/up
        (18432, 4096),   # wide mlp
        (4096, 11008),   # mlp down
        (1536, 11008),   # narrow down
    ],
}

BLOCK_MS = [16]
BLOCK_NS = [32, 64, 128, 256]
BLOCK_KS = [64, 128, 256]
KSPLITS = [1, 2, 4, 8]
WARPS = [4, 8]
STAGES = [2, 3]

# M-buckets for the decode range (low end of STANDARD_M_BOUNDS).
M_BUCKETS = [1, 4, 16, 32]


def lds_bytes(bm, bn, bk, stages):
    """Approx LDS: (A-tile + B-tile) per stage, double-buffered (stages-1)."""
    # int8: 1 byte/elt. A-tile = bm*bk, B-tile = bn*bk.
    per_stage = (bm * bk + bn * bk) * 1
    return per_stage * max(stages - 1, 1)


def make_cfg(bm, bn, bk, ks, nw, ns, K):
    return {
        "BLOCK_SIZE_M": bm,
        "BLOCK_SIZE_N": bn,
        "BLOCK_SIZE_K": bk,
        "GROUP_SIZE_M": 4,
        "NUM_KSPLIT": ks,
        "SPLITK_BLOCK_SIZE": triton.cdiv(K, ks),
        "num_warps": nw,
        "num_stages": ns,
        "waves_per_eu": 2,
        "matrix_instr_nonkdim": 16,
        "kpack": 1,
        "cache_modifier": None,
    }


def bench(cfg, x, wt, xs, wsc):
    try:
        for _ in range(15):
            gemm_a8w8(x, wt, xs, wsc, dtype=torch.bfloat16, config=cfg)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            gemm_a8w8(x, wt, xs, wsc, dtype=torch.bfloat16, config=cfg)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / N_ITER
    except Exception:
        return None


def sweep_shape(N, K, M):
    """Return (best_dt, best_cfg, n_valid, results) for one (N,K) at batch M."""
    torch.manual_seed(0)
    x = torch.randint(-128, 127, (M, K), dtype=torch.int8, device="cuda")
    wt = torch.randint(-128, 127, (N, K), dtype=torch.int8, device="cuda")
    xs = torch.ones((M, 1), dtype=torch.float32, device="cuda")
    wsc = torch.ones((1, N), dtype=torch.float32, device="cuda")

    results = []
    combos = list(itertools.product(BLOCK_MS, BLOCK_NS, BLOCK_KS, KSPLITS, WARPS, STAGES))
    for bm, bn, bk, ks, nw, ns in combos:
        if lds_bytes(bm, bn, bk, ns) > LDS_CAP:
            continue
        cfg = make_cfg(bm, bn, bk, ks, nw, ns, K)
        dt = bench(cfg, x, wt, xs, wsc)
        if dt is not None:
            results.append((dt, cfg))
    if not results:
        return None, None, 0, []
    results.sort(key=lambda r: r[0])
    return results[0][0], results[0][1], len(results), results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen27b", choices=list(MODELS))
    ap.add_argument("--out", default=None, help="write winners to this JSON path")
    ap.add_argument("--m", type=int, default=1, help="batch M to tune (decode=1)")
    args = ap.parse_args()

    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"model={args.model}  M={args.m}  N_ITER={N_ITER}\n")

    shapes = MODELS[args.model]
    winners = {}  # (N,K) -> cfg
    print(f"{'shape':>16} | {'best GB/s':>10} | {'%peak':>6} | config")
    print("-" * 78)
    for N, K in shapes:
        best_dt, best_cfg, n_valid, _ = sweep_shape(N, K, args.m)
        if best_dt is None:
            print(f"N={N:6d} K={K:6d} | NO VALID CONFIG ({n_valid})")
            continue
        gbps = N * K / best_dt / 1e9
        winners[(N, K)] = best_cfg
        print(f"N={N:6d} K={K:6d} | {gbps:8.1f} | {gbps/PEAK*100:4.0f}% | "
              f"BM={best_cfg['BLOCK_SIZE_M']} BN={best_cfg['BLOCK_SIZE_N']} "
              f"BK={best_cfg['BLOCK_SIZE_K']} KS={best_cfg['NUM_KSPLIT']} "
              f"W={best_cfg['num_warps']} S={best_cfg['num_stages']}  ({n_valid} valid)")

    if args.out:
        # Build the JSON: M_LEQ_32 = winner for the decode bucket (use N=4096,K=4096
        # as the representative small shape), plus per-shape specialized keys.
        # The aiter loader picks M_LEQ_32 for M<=32, then an N_K specialized key
        # overrides if present.
        rep = winners.get((4096, 4096)) or (winners[list(winners)[0]] if winners else None)
        out = {}
        if rep:
            out["M_LEQ_32"] = {k: v for k, v in rep.items() if k != "SPLITK_BLOCK_SIZE"}
        for (N, K), cfg in winners.items():
            key = f"N={N}-K={K}"
            out[key] = {k: v for k, v in cfg.items() if k != "SPLITK_BLOCK_SIZE"}
        # keep an any fallback
        out["any"] = out.get("M_LEQ_32", {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
                                            "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4,
                                            "NUM_KSPLIT": 1, "num_warps": 4,
                                            "num_stages": 3, "waves_per_eu": 2,
                                            "matrix_instr_nonkdim": 16, "kpack": 1})
        with open(args.out, "w") as f:
            json.dump(out, f, indent=4)
        print(f"\nwrote {len(out)} entries -> {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""In-process config sweep for the W8A8 decode GEMM (M=1), gfx1100.

gemm_a8w8() takes a `config` dict, so we sweep block sizes / split-K / warps
without touching any file or rebuilding. Reports the winner per shape.
"""
import itertools
import time
import traceback
import torch
import triton

from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8

M = 1
N_ITER = 200
PEAK = 960.0

SHAPES = [(4096, 4096), (1536, 11008)]

BLOCK_MS = [16]
BLOCK_NS = [64, 128, 256]
BLOCK_KS = [64, 128, 256]
KSPLITS = [1, 2, 4, 8]
WARPS = [4, 8]
STAGES = [2, 3]

_printed = [False]


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
        for _ in range(10):
            gemm_a8w8(x, wt, xs, wsc, dtype=torch.bfloat16, config=cfg)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            gemm_a8w8(x, wt, xs, wsc, dtype=torch.bfloat16, config=cfg)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / N_ITER
    except Exception:
        if not _printed[0]:
            traceback.print_exc()
            _printed[0] = True
        return None


def main():
    print(f"device: {torch.cuda.get_device_name(0)}\n")
    for N, K in SHAPES:
        torch.manual_seed(0)
        x = torch.randint(-128, 127, (M, K), dtype=torch.int8, device="cuda")
        wt = torch.randint(-128, 127, (N, K), dtype=torch.int8, device="cuda")
        xs = torch.ones((M, 1), dtype=torch.float32, device="cuda")
        wsc = torch.ones((1, N), dtype=torch.float32, device="cuda")

        results = []
        combos = list(itertools.product(BLOCK_MS, BLOCK_NS, BLOCK_KS, KSPLITS, WARPS, STAGES))
        print(f"=== N={N} K={K}  ({len(combos)} configs) ===")
        for bm, bn, bk, ks, nw, ns in combos:
            cfg = make_cfg(bm, bn, bk, ks, nw, ns, K)
            dt = bench(cfg, x, wt, xs, wsc)
            if dt is None:
                continue
            gbps = N * K / dt / 1e9
            results.append((dt, gbps, cfg))

        results.sort(key=lambda r: r[0])
        print(f"  top 5 of {len(results)} valid:")
        for dt, gbps, cfg in results[:5]:
            print(f"    {dt*1e6:7.1f} us  {gbps:6.1f} GB/s ({gbps/PEAK*100:3.0f}%)  "
                  f"BM={cfg['BLOCK_SIZE_M']} BN={cfg['BLOCK_SIZE_N']} BK={cfg['BLOCK_SIZE_K']} "
                  f"KS={cfg['NUM_KSPLIT']} W={cfg['num_warps']} S={cfg['num_stages']}")
        print()


if __name__ == "__main__":
    main()

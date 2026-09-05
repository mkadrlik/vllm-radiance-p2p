#!/usr/bin/env python3
"""A/B: W8A8 decode GEMM with N-major [N,K] (current) vs K-major [K,N] (pre-transposed).

The wrapper reads w.stride() directly and does w = w.T, so passing an
already-contiguous [K,N] weight is a pure data-layout change - the B-tile
load becomes coalesced. Measures GB/s for both layouts at M=1.
"""
import time
import torch
import triton

from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8

M = 1
N_ITER = 300
PEAK = 960.0
SHAPES = [(4096, 4096), (11008, 4096), (18432, 4096), (4096, 11008), (1536, 11008)]

# tuned decode config (from the merged PR #2)
CFG = {
    "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 256,
    "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 3,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "kpack": 1,
    "cache_modifier": None,
    "SPLITK_BLOCK_SIZE": 4096,
}


def run(wt, x, xs, wsc):
    for _ in range(20):
        gemm_a8w8(x, wt, xs, wsc, dtype=torch.bfloat16, config=CFG)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        gemm_a8w8(x, wt, xs, wsc, dtype=torch.bfloat16, config=CFG)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / N_ITER


def main():
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"{'shape':>18} | {'N-major [N,K]':>16} | {'K-major [K,N]':>16} | {'delta':>8}")
    print("-" * 66)
    for N, K in SHAPES:
        torch.manual_seed(0)
        x = torch.randint(-128, 127, (M, K), dtype=torch.int8, device="cuda")
        xs = torch.ones((M, 1), dtype=torch.float32, device="cuda")
        # N-major: weights stored [N, K] (the on-disk / current layout)
        wn = torch.randint(-128, 127, (N, K), dtype=torch.int8, device="cuda")
        # K-major: pre-transposed contiguous [K, N]
        wk = wn.t().contiguous()
        wsc = torch.ones((1, N), dtype=torch.float32, device="cuda")

        dt_n = run(wn, x, xs, wsc)
        dt_k = run(wk, x, xs, wsc)
        gbps_n = N * K / dt_n / 1e9
        gbps_k = N * K / dt_k / 1e9
        delta = (gbps_k / gbps_n - 1) * 100
        print(f"N={N:6d} K={K:6d} | {gbps_n:8.1f} GB/s | {gbps_k:8.1f} GB/s | {delta:+6.1f}%")


if __name__ == "__main__":
    main()

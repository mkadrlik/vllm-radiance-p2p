#!/usr/bin/env python3
"""Corrected A/B: physical layout of the B-matrix (weights) for the W8A8 decode GEMM.

Both layouts pass the wrapper's [N,K] contract (it reads w.stride()):
  A (N-major, current): w=[N,K] contiguous          -> B=w.T has stride_bk=1, stride_bn=K
  B (K-major):          w=[N,K] strides (1,N)       -> B=w.T has stride_bk=N, stride_bn=1

Interleaved rounds cancel thermal/clock drift on the shared box. Median of rounds.
"""
import statistics
import time
import torch
import triton

from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8

N_ITER = 40      # iters per layout per round
ROUNDS = 7       # interleaved rounds
PEAK = 960.0
SHAPES = [(4096, 4096), (11008, 4096), (18432, 4096), (4096, 11008), (1536, 11008)]

CFG = {
    "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 256,
    "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 3,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "kpack": 1,
    "cache_modifier": None, "SPLITK_BLOCK_SIZE": 4096,
}


def timeit(w, x, xs, wsc):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        gemm_a8w8(x, w, xs, wsc, dtype=torch.bfloat16, config=CFG)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / N_ITER


def main():
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"{'shape':>16} | {'A N-major':>11} | {'B K-major':>11} | {'B vs A':>8}")
    print("-" * 60)
    for N, K in SHAPES:
        torch.manual_seed(0)
        x = torch.randint(-128, 127, (1, K), dtype=torch.int8, device="cuda")
        xs = torch.ones((1, 1), dtype=torch.float32, device="cuda")
        wA = torch.randint(-128, 127, (N, K), dtype=torch.int8, device="cuda")
        wB = wA.t().contiguous().t()  # [N,K] with strides (1, N) -> K-major B
        wsc = torch.ones((1, N), dtype=torch.float32, device="cuda")
        # warmup both
        for w in (wA, wB):
            gemm_a8w8(x, w, xs, wsc, dtype=torch.bfloat16, config=CFG)
        torch.cuda.synchronize()

        a_rounds, b_rounds = [], []
        for _ in range(ROUNDS):
            a_rounds.append(timeit(wA, x, xs, wsc))
            b_rounds.append(timeit(wB, x, xs, wsc))
        a_med = statistics.median(a_rounds)
        b_med = statistics.median(b_rounds)
        ga = N * K / a_med / 1e9
        gb = N * K / b_med / 1e9
        delta = (gb / ga - 1) * 100
        print(f"N={N:6d} K={K:6d} | {ga:8.1f} | {gb:8.1f} | {delta:+6.1f}%")


if __name__ == "__main__":
    main()

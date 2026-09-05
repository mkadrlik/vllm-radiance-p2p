#!/usr/bin/env python3
"""Decode GEMV bandwidth bench for gfx1100 — W8A8 Triton GEMM (M=1).

Run inside the vllm-radiance:gfx1100 container with GPU access:
  python3 bench_decode_gemm.py

Reads the GB/s column, not the us. Decode GEMV is memory-bandwidth-bound;
peak on the 7900 XTX is 960 GB/s. If we're at 300-400, the config/WMMA gaps
are real. If 800+, the GEMM is near the wall.
"""
import time
import torch

# 27B (Qwen3.8-27B) linear layer shapes (N, K) hit at decode M=1.
# q/k/v/o proj ~4096x4096, MLP up/gate ~11008x4096, down ~4096x11008.
SHAPES = [
    (4096, 4096),    # attention proj
    (11008, 4096),   # MLP gate/up
    (18432, 4096),   # wide MLP
    (4096, 11008),   # MLP down
    (1536, 11008),   # narrow down
]
M = 1  # decode
N_ITER = 300


def bench_one(N: int, K: int):
    from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8

    torch.manual_seed(0)
    x = torch.randint(-128, 127, (M, K), dtype=torch.int8, device="cuda")
    w = torch.randint(-128, 127, (N, K), dtype=torch.int8, device="cuda")
    xs = torch.ones((M, 1), dtype=torch.float32, device="cuda")
    ws = torch.ones((1, N), dtype=torch.float32, device="cuda")

    # warmup (JIT compile + allocator)
    for _ in range(20):
        y = gemm_a8w8(x, w, xs, ws, dtype=torch.bfloat16)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(N_ITER):
        y = gemm_a8w8(x, w, xs, ws, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / N_ITER

    bytes_moved = N * K  # int8 weights dominate; activations are M=1 (negligible)
    gbps = bytes_moved / dt / 1e9
    print(f"N={N:6d} K={K:6d}  {dt * 1e6:8.1f} us   {gbps:7.1f} GB/s   (peak 960)")
    return dt, gbps


def main():
    print(f"torch {torch.__version__}  cuda available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"M={M}  N_ITER={N_ITER}\n")
    total = 0.0
    for N, K in SHAPES:
        try:
            dt, gbps = bench_one(N, K)
            total += gbps
        except Exception as e:
            print(f"N={N:6d} K={K:6d}  FAILED: {e}")
    print(f"\navg GB/s across shapes: {total / len(SHAPES):.1f}")


if __name__ == "__main__":
    main()

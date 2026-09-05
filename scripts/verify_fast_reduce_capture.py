#!/usr/bin/env python3
"""Minimal repro + verification for the RADIANCE_FAST_REDUCE MTP-capture deadlock.

Two processes (torchrun --nproc_per_node=2, one GPU each), TP=2, using the real
radiance_allreduce.RadianceAllreduce + radiance_ar_ext kernel.

Default (verification) mode — all must pass with the capture-gate fix:
  t1 eager gate      — should_custom_ar() is False outside graph capture and
                       custom_all_reduce() returns None -> callers fall through to RCCL
                       (pre-fix code launched the spin kernel eagerly here).
  t2 capture+replay  — the gate is True inside stream capture; the kernel embedded in a
                       CUDA graph replays and stays in seq lockstep across ranks.
  t3 multi-graph     — interleaved replays of graphs captured at different sizes (shared
                       seq counters) keep producing bit-identical results.
  t4 decode shapes   — bs 1/8/64 x hidden 5120 graphs == RCCL bit-for-bit.

--danger mode — reproduces the OLD deadlock (pre-fix behaviour) as a class:
  rank0 launches ONE eager AR kernel while rank1 skips it (any rank-divergent eager
  all_reduce during warmup does this), then both do what torch.cuda.graph()
  capture_begin does (device-wide sync). rank0's stream wedges on the peer-flag spin
  (RADIANCE_SPIN_MAX uncached PCIe reads ~ 4000 s). Run it under an external timeout:

    timeout -s KILL 90 torchrun --nproc_per_node=2 scripts/verify_fast_reduce_capture.py --danger
    echo $?   # 124 (+ "DANGER: SYNC-BLOCKED" printed, no "PASSED") == deadlock reproduced

  Leaves a spinning kernel on rank0's GPU until the process is killed; do NOT run while
  production inference uses these GPUs.
"""
import argparse
import datetime
import os
import sys
import time

import torch
import torch.distributed as dist

import radiance_allreduce as R


def log(msg):
    rank = os.environ.get("RANK", "?")
    print(f"[rank {rank}] {msg}", flush=True)


def rccl_sum(x):
    y = x.clone()
    dist.all_reduce(y)
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--danger", action="store_true",
                    help="reproduce the pre-fix eager-divergence deadlock "
                         "(run under `timeout -s KILL 90`; see docstring)")
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=60))
    cpu_group = dist.new_group(backend="gloo")  # vLLM passes the gloo cpu_group too

    comm = R.RadianceAllreduce(cpu_group, dev)
    assert not comm.disabled, "RadianceAllreduce failed to install in this env"

    if args.danger:
        x = torch.randn(4096, dtype=torch.bfloat16, device=dev)
        if rank == 0:
            nbytes = x.numel() * x.element_size()
            log("DANGER: launching ONE eager AR kernel on rank0 only (pre-fix behaviour)")
            comm._ext.all_reduce_mb(
                comm._peer_scratch, comm._scratch, comm._peer_flags, comm._flags,
                comm._seq.data_ptr(), comm.slot16, x.data_ptr(), x.data_ptr(),
                x.numel(), 0, torch.cuda.current_stream().cuda_stream,
                comm._nblocks(nbytes // 16), comm.nt, comm.drain, comm.acq)
        dist.barrier()  # gloo: both hosts reach here; rank0's kernel is still spinning
        log("DANGER: SYNC-BLOCKED (attempting device-wide sync, as capture_begin does)")
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _ = torch.add(x, x)
        torch.cuda.synchronize()
        log("DANGER: sync PASSED (unexpected — deadlock class not reproduced)")
        if rank == 1:
            time.sleep(3600)  # keep the job alive so the external timeout observes rank0
        dist.destroy_process_group()
        raise SystemExit(1)

    ok = True

    # ---- t1: eager gate -----------------------------------------------------
    x = torch.randn(4096, dtype=torch.bfloat16, device=dev)
    eager = comm.should_custom_ar(x)
    ret = comm.custom_all_reduce(x)
    passed = (not eager) and (ret is None)
    log(f"t1 eager gate: should_custom_ar={eager} custom_all_reduce->"
        f"{'None' if ret is None else 'tensor'} {'PASS' if passed else 'FAIL'}")
    ok = ok and passed

    # ---- t2: capture + replay -----------------------------------------------
    torch.cuda.synchronize()
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        assert comm.should_custom_ar(x), "gate must be True inside stream capture"
        out = comm.custom_all_reduce(x)
    assert out is not None
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    dist.barrier()
    same2 = torch.equal(out.float(), rccl_sum(x).float())
    log(f"t2 capture+replay x3 bit-identical to RCCL: {'PASS' if same2 else 'FAIL'}")
    ok = ok and same2

    # ---- t3: multi-graph interleaved replay (shared seq counters) -----------
    graphs, refs = [], []
    for numel in (1024, 8192, 32768):
        xi = torch.randn(numel, dtype=torch.bfloat16, device=dev)
        gi = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gi):
            oi = comm.custom_all_reduce(xi)
        assert oi is not None
        graphs.append((gi, xi, oi))
        refs.append(rccl_sum(xi))
    for i in (2, 0, 1, 1, 2, 0, 0, 2, 1):
        graphs[i][0].replay()
    torch.cuda.synchronize()
    dist.barrier()
    same3 = all(torch.equal(gr[2].float(), r.float()) for gr, r in zip(graphs, refs))
    log(f"t3 multi-graph interleaved replay (9 replays, 3 graphs): "
        f"{'PASS' if same3 else 'FAIL'}")
    ok = ok and same3

    # ---- t4: decode-shaped messages byte-identity (27B hidden=5120) ---------
    same4 = True
    for bs in (1, 8, 64):
        xd = torch.randn(bs * 5120, dtype=torch.bfloat16, device=dev)
        gd = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gd):
            od = comm.custom_all_reduce(xd)
        assert od is not None
        gd.replay()
        torch.cuda.synchronize()
        dist.barrier()
        same4 = same4 and torch.equal(od.float(), rccl_sum(xd).float())
    log(f"t4 decode-shape byte-identity (bs 1/8/64): {'PASS' if same4 else 'FAIL'}")
    ok = ok and same4

    log(f"RESULT: {'ALL PASS' if ok else 'FAILURES'}")
    dist.destroy_process_group()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

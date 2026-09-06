#!/usr/bin/env python3
"""Phase 0 microbench driver: peer-write flush primitives on gfx1100 PCIe.

torchrun --nproc_per_node=2 microbench/phase0_bench.py [--quick]
Run inside the gfx1100 container with ROCR_VISIBLE_DEVICES pinned to the prod pair
(1,2). Every device spin is bounded by a globaltimer abort; safe co-resident with prod.

Outputs JSONL to /tmp/phase0_rank{N}.jsonl; rank 0 also echoes JSON0 lines.
Aggregate with phase0_report.py.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ar_phase0 as ext  # noqa: E402  (built in-container, see build_phase0.sh)

VAR_NAMES = {
    0: "plain", 1: "atomic_rel_sys", 2: "tfs_then_store", 3: "buffer_sc0",
    4: "buffer_sc1", 5: "buffer_sc1_readflush", 6: "store+vmcnt0", 7: "store+vmcnt0_excnt0",
    8: "packed8",
}

SCRATCH_BYTES = 4096
ABORT_NS = 500_000_000        # 500 ms per-iteration abort (real RTT is us; generous)
# counters tensor: int32[8]; [0]=aborts [1]=stale [2]=never [3..]=spare
C_ABORT, C_STALE, C_NEVER = 0, 1, 2


def log(rank, msg):
    print(f"[p0 {rank}] {msg}", flush=True)


def stats(arr):
    """arr: numpy uint64 samples; ~0ull (>1e15) marks aborted iterations."""
    a = arr.astype(np.float64)
    bad = int((a > 1e15).sum())
    good = a[a <= 1e15]
    if good.size == 0:
        return dict(n=0, aborted=bad, p50=None, p90=None, p99=None, max=None, mean=None)
    return dict(n=int(good.size), aborted=bad,
                p50=float(np.percentile(good, 50)), p90=float(np.percentile(good, 90)),
                p99=float(np.percentile(good, 99)), max=float(good.max()),
                mean=float(good.mean()))


def xbar():
    dist.barrier()
    torch.cuda.synchronize()
    dist.barrier()


def sbar():
    """Rendezvous without draining OTHER streams (the hammer must stay in flight)."""
    dist.barrier()
    torch.cuda.current_stream().synchronize()
    dist.barrier()


def cptr(counters, idx):
    return counters.data_ptr() + 4 * idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    iters = 300 if args.quick else args.iters

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)  # ROCR_VISIBLE_DEVICES=1,2 -> cuda:0=GPU1, cuda:1=GPU2
    dev = torch.device(f"cuda:{rank}")
    dist.init_process_group("gloo")  # host-side handle exchange only

    ptr, h = ext.alloc_shared(SCRATCH_BYTES, True)
    hs = [None, None]
    dist.all_gather_object(hs, h)
    peer_ptr = ext.open_shared(hs[1 - rank])
    log(rank, f"scratch me=0x{ptr:x} peer=0x{peer_ptr:x}")

    st = torch.cuda.current_stream().cuda_stream
    samples = torch.zeros(iters, dtype=torch.int64, device=dev)
    counters = torch.zeros(8, dtype=torch.int32, device=dev)
    # hammer: 4 dependent uncached PCIe reads/iter (~1.5 us/iter) x 8 blocks x 1024 thr.
    # 200k iters ≈ 300 ms of sustained read storm — covers the handshake window;
    # torch.cuda.synchronize() at the end of each config drains the remainder.
    hammer_iters = 200_000

    # ---- s_timer tick-rate calibration (gfx1100 has no ns-accurate globaltimer) ----
    cal = torch.zeros(3, dtype=torch.int64, device=dev)
    import time as _t
    w0 = _t.monotonic_ns()
    ext.run_calib(cal.data_ptr(), st)
    torch.cuda.synchronize()
    w1 = _t.monotonic_ns()
    xbar()
    c0, c1 = int(cal[0]), int(cal[1])
    ticks_per_ns = max(0.001, (c1 - c0) / max(1.0, (w1 - w0) * 0.9))  # wall incl launch; scale 0.9 conservative
    log(rank, f"s_timer calib: {c1-c0} ticks / {(w1-w0)/1e6:.1f} ms wall -> {ticks_per_ns:.6f} ticks/ns")
    ABORT_TICKS = int(ABORT_NS * ticks_per_ns)

    out = open(f"/tmp/phase0_rank{rank}.jsonl", "w")

    def emit(rec):
        rec["rank"] = rank
        rec["ticks_per_ns"] = ticks_per_ns
        out.write(json.dumps(rec) + "\n")
        out.flush()
        if rank == 0:
            print("JSON0 " + json.dumps(rec), flush=True)

    def pair(fn0, fn1, sync=xbar):
        sync()
        (fn0 if rank == 0 else fn1)()
        torch.cuda.current_stream().synchronize()
        sync()

    # ================= (a) ping-pong RTT per store variant =================
    for var in (0, 1, 2, 3, 4, 5, 6, 7):
        counters.zero_(); samples.zero_()
        pair(lambda v=var: ext.run_pingpong(peer_ptr, ptr, samples.data_ptr(),
                                            cptr(counters, C_ABORT), 0, iters, v, 0, ABORT_TICKS, 0, st),
             lambda v=var: ext.run_pingpong(peer_ptr, ptr, samples.data_ptr(),
                                            cptr(counters, C_ABORT), 1, iters, v, 0, ABORT_TICKS, 0, st))
        emit(dict(test="pingpong", variant=VAR_NAMES[var], iters=iters,
                  aborts=int(counters[C_ABORT]), **stats(samples.cpu().numpy().view(np.uint64))))

    # ================= (b) data-before-flag ordering =================
    # data posted with dv, flag with fv; reader checks payload on flag arrival.
    for dv in (0, 4, 8):
        for fv in (1, 6, 7, 2, 5):
            counters.zero_(); samples.zero_()
            pair(lambda: ext.run_order(peer_ptr, ptr, samples.data_ptr(),
                                       cptr(counters, C_STALE), cptr(counters, C_NEVER),
                                       0, iters, dv, fv, 0, ABORT_TICKS, st),
                 lambda: ext.run_order(peer_ptr, ptr, samples.data_ptr(),
                                       cptr(counters, C_STALE), cptr(counters, C_NEVER),
                                       1, iters, dv, fv, 0, ABORT_TICKS, st))
            c = counters.cpu().numpy()
            emit(dict(test="order", data_variant=VAR_NAMES[dv], flag_variant=VAR_NAMES[fv],
                      iters=iters, stale=int(c[C_STALE]), never=int(c[C_NEVER]),
                      **stats(samples.cpu().numpy().view(np.uint64))))

    # ================= (c) LL packed 8B round-trip + tearing =================
    counters.zero_(); samples.zero_()
    pair(lambda: ext.run_ll(peer_ptr, ptr, samples.data_ptr(), cptr(counters, C_STALE),
                            cptr(counters, C_ABORT), 0, iters, 0, ABORT_TICKS, 0, st),
         lambda: ext.run_ll(peer_ptr, ptr, samples.data_ptr(), cptr(counters, C_STALE),
                            cptr(counters, C_ABORT), 1, iters, 0, ABORT_TICKS, 0, st))
    c = counters.cpu().numpy()
    emit(dict(test="ll_packed", iters=iters, tears=int(c[C_STALE]),
              aborts=int(c[C_ABORT]), **stats(samples.cpu().numpy().view(np.uint64))))

    # ================= (d) contention: in-kernel hammer blocks + backoff ========
    # Blocks 1..7 of the SAME kernel storm the peer's scratch with uncached reads
    # (GPU_MAX_HW_QUEUES=1 serializes streams; same-kernel workgroups stay concurrent).
    for nops_cyc in (0, 200, 1000, 5000):
        nops = max(0, nops_cyc // 63)
        for test in ("pingpong", "ll"):
            counters.zero_(); samples.zero_()
            if test == "pingpong":
                pair(lambda: ext.run_pingpong(peer_ptr, ptr, samples.data_ptr(),
                                              cptr(counters, C_ABORT), 0, iters, 1, nops, ABORT_TICKS, hammer_iters, st),
                     lambda: ext.run_pingpong(peer_ptr, ptr, samples.data_ptr(),
                                              cptr(counters, C_ABORT), 1, iters, 1, nops, ABORT_TICKS, hammer_iters, st))
            else:
                pair(lambda: ext.run_ll(peer_ptr, ptr, samples.data_ptr(), cptr(counters, C_STALE),
                                        cptr(counters, C_ABORT), 0, iters, nops, ABORT_TICKS, hammer_iters, st),
                     lambda: ext.run_ll(peer_ptr, ptr, samples.data_ptr(), cptr(counters, C_STALE),
                                        cptr(counters, C_ABORT), 1, iters, nops, ABORT_TICKS, hammer_iters, st))
            emit(dict(test="cont_" + test, backoff_cycles=nops_cyc,
                      variant=VAR_NAMES[1] if test == "pingpong" else VAR_NAMES[8],
                      iters=iters, **stats(samples.cpu().numpy().view(np.uint64))))

    # ================= (e) one-way visibility (LL mbox, echo RTT) =============
    counters.zero_(); samples.zero_()
    pair(lambda: ext.run_oneway(peer_ptr, ptr, samples.data_ptr(), cptr(counters, C_STALE),
                                cptr(counters, C_ABORT), 0, iters, 0, ABORT_TICKS, st),
         lambda: ext.run_oneway(peer_ptr, ptr, samples.data_ptr(), cptr(counters, C_STALE),
                                cptr(counters, C_ABORT), 1, iters, 0, ABORT_TICKS, st))
    emit(dict(test="oneway_ll", iters=iters,
              **stats(samples.cpu().numpy().view(np.uint64))))

    out.close()
    dist.destroy_process_group()
    log(rank, "DONE")


if __name__ == "__main__":
    main()

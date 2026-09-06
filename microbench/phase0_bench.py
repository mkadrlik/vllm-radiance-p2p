#!/usr/bin/env python3
"""Phase 0 microbench driver: peer-write flush primitives on gfx1100 PCIe.

torchrun --nproc_per_node=2 phase0_bench.py [--quick]
Run inside the gfx1100 container with ROCR_VISIBLE_DEVICES=1,2 (prod pair,
co-resident; prod stays up).

gfx1100 has NO shader-readable wall clock (s_timer*/s_memrealtime/clock64 all fail
the assembler from a live kernel — see probe2/probe3). Device timings are therefore
SPIN-POLL COUNTS, converted to us with a per-run calibration of the dependent
uncached PCIe read cost (run_read_calib, host-timed). Wall time of each handshake
burst is also recorded (host clock) as a cross-check.

JSONL to /tmp/phase0_rank{N}.jsonl (rank 0 echoes JSON0). Latencies in us.
"""
import argparse
import json
import os
import sys
import time

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
POLL_NS_MAX = 5000          # assume worst 5 us/poll when sizing the abort cap
# counters int32[8]: [0]=aborts [1]=stale/tears [2]=never
C_ABORT, C_STALE, C_NEVER = 0, 1, 2


def log(rank, msg):
    print(f"[p0 {rank}] {msg}", flush=True)


def stats(arr, poll_us):
    """arr: uint64 spin-poll counts; ~0ull (>1e15) = aborted. Converts to us."""
    a = arr.astype(np.float64)
    bad = int((a > 1e15).sum())
    good = a[a <= 1e15]
    if good.size == 0:
        return dict(n=0, aborted=bad, p50_us=None, p90_us=None, p99_us=None,
                    max_us=None, mean_us=None)
    return dict(n=int(good.size), aborted=bad,
                p50_us=float(np.percentile(good, 50)) * poll_us,
                p90_us=float(np.percentile(good, 90)) * poll_us,
                p99_us=float(np.percentile(good, 99)) * poll_us,
                max_us=float(good.max()) * poll_us,
                mean_us=float(good.mean()) * poll_us)


def xbar():
    dist.barrier()
    torch.cuda.synchronize()
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
    rsp = torch.zeros(iters, dtype=torch.int64, device=dev)
    counters = torch.zeros(8, dtype=torch.int32, device=dev)
    sink = torch.zeros(1, dtype=torch.int32, device=dev)
    hammer_iters = 20_000

    # ---- poll-cost calibration: N dependent uncached PCIe reads, host-timed ----
    N = 20000
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    xbar()
    ev0.record()
    ext.run_read_calib(peer_ptr, N, sink.data_ptr(), st)
    ev1.record()
    torch.cuda.synchronize()
    xbar()
    poll_ns = ev0.elapsed_time(ev1) * 1e6 / N
    poll_us = poll_ns / 1000.0
    log(rank, f"uncached peer read: {poll_ns:.1f} ns/poll")
    CAP = max(2000, int(200_000_000 / max(POLL_NS_MAX, poll_ns)))  # ~200ms abort budget

    out = open(f"/tmp/phase0_rank{rank}.jsonl", "w")

    def emit(rec):
        rec["rank"] = rank
        rec["poll_ns"] = round(poll_ns, 1)
        out.write(json.dumps(rec) + "\n")
        out.flush()
        if rank == 0:
            print("JSON0 " + json.dumps(rec), flush=True)

    def clean():
        """Zero LOCAL scratch + counters. Stale seq from the previous config would
        false-pass the next one (both ranks must zero their own, then rendezvous)."""
        ext.memzero(ptr, SCRATCH_BYTES)
        counters.zero_()
        samples.zero_()
        rsp.zero_()
        xbar()

    def pair(fn0, fn1, name, extra=None):
        clean()
        t0 = time.monotonic_ns()
        (fn0 if rank == 0 else fn1)()
        torch.cuda.current_stream().synchronize()
        wall_us = (time.monotonic_ns() - t0) / 1000.0
        xbar()
        rec = dict(test=name, iters=iters, wall_us_per_iter=round(wall_us / iters, 3))
        if extra:
            rec.update(extra())
        emit(rec)

    # ================= (a) ping-pong RTT per store variant =================
    for var in (0, 1, 2, 3, 4, 5, 6, 7):
        def ex(v=var):
            s0 = samples.cpu().numpy().view(np.uint64)
            s1 = rsp.cpu().numpy().view(np.uint64)
            d = dict(aborts=int(counters[C_ABORT]))
            d.update({("rtt_" + k): val for k, val in stats(s0, poll_us).items()})
            d.update({("oneway_" + k): val for k, val in stats(s1, poll_us).items()})
            return d
        pair(lambda v=var: ext.run_pingpong(peer_ptr, ptr, samples.data_ptr(), rsp.data_ptr(),
                                            cptr(counters, C_ABORT), 0, iters, v, 0, CAP, 0, st),
             lambda v=var: ext.run_pingpong(peer_ptr, ptr, samples.data_ptr(), rsp.data_ptr(),
                                            cptr(counters, C_ABORT), 1, iters, v, 0, CAP, 0, st),
             "pingpong_" + VAR_NAMES[var], ex)

    # ================= (b) data-before-flag ordering =================
    for dv in (0, 4, 8):
        for fv in (1, 6, 7, 2, 5):
            def ex():
                s0 = samples.cpu().numpy().view(np.uint64)
                s1 = rsp.cpu().numpy().view(np.uint64)
                c = counters.cpu().numpy()
                good = s1[s1 <= 1e15]
                fresh = int((good >> 63).sum()) if good.size else 0
                d = dict(stale=int(c[C_STALE]), never=int(c[C_NEVER]),
                         fresh_on_first_read=fresh, reads=len(good))
                d.update({("rtt_" + k): val for k, val in stats(s0, poll_us).items()})
                return d
            pair(lambda: ext.run_order(peer_ptr, ptr, samples.data_ptr(), rsp.data_ptr(),
                                       cptr(counters, C_STALE), cptr(counters, C_NEVER),
                                       0, iters, dv, fv, 0, CAP, 0, st),
                 lambda: ext.run_order(peer_ptr, ptr, samples.data_ptr(), rsp.data_ptr(),
                                       cptr(counters, C_STALE), cptr(counters, C_NEVER),
                                       1, iters, dv, fv, 0, CAP, 0, st),
                 f"order_d{VAR_NAMES[dv]}_f{VAR_NAMES[fv]}", ex)

    # ================= (c) LL packed 8B round-trip + tearing =================
    def ex():
        s0 = samples.cpu().numpy().view(np.uint64)
        s1 = rsp.cpu().numpy().view(np.uint64)
        c = counters.cpu().numpy()
        d = dict(tears=int(c[C_STALE]), aborts=int(c[C_ABORT]))
        d.update({("rtt_" + k): val for k, val in stats(s0, poll_us).items()})
        d.update({("oneway_" + k): val for k, val in stats(s1, poll_us).items()})
        return d
    pair(lambda: ext.run_ll(peer_ptr, ptr, samples.data_ptr(), rsp.data_ptr(),
                            cptr(counters, C_STALE), cptr(counters, C_ABORT),
                            0, iters, 0, CAP, 0, st),
         lambda: ext.run_ll(peer_ptr, ptr, samples.data_ptr(), rsp.data_ptr(),
                            cptr(counters, C_STALE), cptr(counters, C_ABORT),
                            1, iters, 0, CAP, 0, st),
         "ll_packed", ex)

    # ================= (d) contention: in-kernel hammer blocks + backoff ========
    for nops_cyc in (0, 200, 1000, 5000):
        nops = max(0, nops_cyc // 63)
        for test in ("pingpong", "ll"):
            def ex():
                s0 = samples.cpu().numpy().view(np.uint64)
                s1 = rsp.cpu().numpy().view(np.uint64)
                d = {}
                d.update({("rtt_" + k): val for k, val in stats(s0, poll_us).items()})
                d.update({("oneway_" + k): val for k, val in stats(s1, poll_us).items()})
                return d
            if test == "pingpong":
                pair(lambda: ext.run_pingpong(peer_ptr, ptr, samples.data_ptr(), rsp.data_ptr(),
                                              cptr(counters, C_ABORT), 0, iters, 1, nops, CAP, hammer_iters, st),
                     lambda: ext.run_pingpong(peer_ptr, ptr, samples.data_ptr(), rsp.data_ptr(),
                                              cptr(counters, C_ABORT), 1, iters, 1, nops, CAP, hammer_iters, st),
                     f"cont_pingpong_b{nops_cyc}", ex)
            else:
                pair(lambda: ext.run_ll(peer_ptr, ptr, samples.data_ptr(), rsp.data_ptr(),
                                        cptr(counters, C_STALE), cptr(counters, C_ABORT),
                                        0, iters, nops, CAP, hammer_iters, st),
                     lambda: ext.run_ll(peer_ptr, ptr, samples.data_ptr(), rsp.data_ptr(),
                                        cptr(counters, C_STALE), cptr(counters, C_ABORT),
                                        1, iters, nops, CAP, hammer_iters, st),
                     f"cont_ll_b{nops_cyc}", ex)

    out.close()
    dist.destroy_process_group()
    log(rank, "DONE")


if __name__ == "__main__":
    main()

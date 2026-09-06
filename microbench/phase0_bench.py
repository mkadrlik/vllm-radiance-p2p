#!/usr/bin/env python3
"""Phase 0 microbench driver: peer-write flush primitives on gfx1100 PCIe.

torchrun --nproc_per_node=2 phase0_bench.py [--quick] [--only pp,ppb,order,ll,cont,vis]
Run inside the gfx1100 container with ROCR_VISIBLE_DEVICES=1,2 (prod pair,
co-resident; prod stays up).

gfx1100 has NO shader-readable wall clock (see ar_phase0.hip header). Device
latencies = spin-poll counts; ns/poll is DERIVED per config from host wall time
divided by total observed spins (cross-checked vs the read_calib kernel).
Responder-side stats are transported into rank 0's scratch and read back via
read_words (rank 1's local tensors are invisible to the report — rev-1 bug).

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
LD_NAMES = {0: "plain_ld", 1: "atomic_acq_sys", 2: "buf_ld_sc0",
            3: "buf_ld_sc1", 4: "buf_ld_sc1_wait"}

SCRATCH_BYTES = 4096
S_RSPT_U32 = 96          # responder transport line (u32 index) — must match .hip
C_ABORT, C_STALE = 0, 1  # counters int32[8] roles


def log(rank, msg):
    print(f"[p0 {rank}] {msg}", flush=True)


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
    ap.add_argument("--only", default=None,
                    help="comma list: pp,ppb,order,ll,cont,vis")
    args = ap.parse_args()
    iters = 300 if args.quick else args.iters
    only = set(args.only.split(",")) if args.only else None

    def want(name):
        return only is None or name in only

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
    scratch_host = torch.zeros(SCRATCH_BYTES // 4, dtype=torch.int32, device=dev)
    sink = torch.zeros(1, dtype=torch.int32, device=dev)
    hammer_iters = 20_000
    BASE_B = 2000

    # ---- poll-cost calibration (cross-check; per-config derivation is truth) ----
    N = 200000
    best = 1e18
    for _ in range(3):
        xbar()
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        ev0.record()
        ext.run_read_calib(ptr + 64, N, BASE_B // 63, sink.data_ptr(), st)
        ev1.record()
        torch.cuda.synchronize()
        xbar()
        best = min(best, ev0.elapsed_time(ev1) * 1e6 / N)
    calib_ns = float(best)
    log(rank, f"calib poll (b{BASE_B}): {calib_ns:.1f} ns")
    CAP = max(2000, int(200_000_000 / max(calib_ns, 1.0)))

    out = open(f"/tmp/phase0_rank{rank}.jsonl", "w")

    def emit(rec):
        rec["rank"] = rank
        rec["calib_ns"] = round(calib_ns, 1)
        out.write(json.dumps(rec) + "\n")
        out.flush()
        if rank == 0:
            print("JSON0 " + json.dumps(rec), flush=True)

    def clean():
        ext.memzero(ptr, SCRATCH_BYTES)
        counters.zero_()
        samples.zero_()
        torch.cuda.current_stream().synchronize()
        xbar()

    def pair(fn0, fn1, name, extra):
        clean()
        t0 = time.monotonic_ns()
        (fn0 if rank == 0 else fn1)()
        torch.cuda.current_stream().synchronize()
        wall_us = (time.monotonic_ns() - t0) / 1000.0
        xbar()
        rec = dict(test=name, iters=iters, wall_us=round(wall_us, 1))
        rec.update(extra(wall_us))
        emit(rec)

    def read_scratch():
        ext.read_words(ptr, SCRATCH_BYTES // 4, scratch_host.data_ptr())
        return scratch_host.cpu().numpy().astype(np.uint32)

    def init_part(wall_us, rsp_line=14):
        s0 = samples.cpu().numpy().view(np.uint64)
        good = s0[s0 <= 1e15]
        total = float(good.sum())
        poll_ns = (wall_us * 1000.0 / total) if total > 0 else calib_ns
        d = dict(init_n=int(good.size), init_aborted=int(s0.size - good.size),
                 poll_ns_derived=round(poll_ns, 1))
        if good.size:
            p = np.percentile(good, [50, 90, 99])
            d.update(init_p50_us=round(float(p[0]) * poll_ns / 1000, 3),
                     init_p90_us=round(float(p[1]) * poll_ns / 1000, 3),
                     init_p99_us=round(float(p[2]) * poll_ns / 1000, 3),
                     init_max_us=round(float(good.max()) * poll_ns / 1000, 3),
                     init_mean_us=round(float(good.mean()) * poll_ns / 1000, 3))
        c = counters.cpu().numpy()
        d["aborts"] = int(c[C_ABORT])
        if rank == 0:
            w = read_scratch()
            rsp = w[S_RSPT_U32:S_RSPT_U32 + 13].astype(np.float64)
            nz = rsp[rsp > 0]
            if nz.size:
                d["rsp_p50_us"] = round(float(np.median(nz)) * poll_ns / 1000, 3)
                d["rsp_max_us"] = round(float(nz.max()) * poll_ns / 1000, 3)
            d["rsp_tail"] = int(w[S_RSPT_U32 + rsp_line])
        return d

    def order_part(wall_us):
        """order_test transport: u64 samples (spins|fresh<<63) at u32 0..13,
        stale count at +14, fresh count at +15."""
        d = init_part(wall_us, rsp_line=14)
        if rank == 0:
            w = read_scratch()
            u64 = w[S_RSPT_U32:S_RSPT_U32 + 14].astype(np.uint64)
            spins = u64[:7] & np.uint64((1 << 32) - 1)
            spins = spins[spins > 0].astype(np.float64)
            fresh_bits = ((u64[:7] >> np.uint64(63)) & np.uint64(1)).sum()
            d["stale"] = int(w[S_RSPT_U32 + 14])
            d["fresh"] = int(w[S_RSPT_U32 + 15])
            d["rsp_fresh_first7"] = int(fresh_bits)
            if spins.size:
                poll_ns = d["poll_ns_derived"]
                d["rsp_arrive_p50_us"] = round(float(np.median(spins)) * poll_ns / 1000, 3)
                d["rsp_arrive_max_us"] = round(float(spins.max()) * poll_ns / 1000, 3)
        return d

    # ================= (a) ping-pong RTT per store variant =================
    if want("pp"):
        for var in (0, 1, 2, 3, 4, 5, 6, 7):
            pair(lambda v=var: ext.run_pingpong(peer_ptr, ptr, samples.data_ptr(),
                                                cptr(counters, C_ABORT), 0, iters, v,
                                                BASE_B // 63, CAP, 0, st),
                 lambda v=var: ext.run_pingpong(peer_ptr, ptr, samples.data_ptr(),
                                                cptr(counters, C_ABORT), 1, iters, v,
                                                BASE_B // 63, CAP, 0, st),
                 "pingpong_" + VAR_NAMES[var],
                 lambda w_: init_part(w_))

    # ================= (a2) backoff sweep, no hammer: the starvation curve ======
    if want("ppb"):
        for nops_cyc in (0, 500, 2000, 5000, 10000, 20000, 40000):
            nops = nops_cyc // 63
            cap_n = max(2000, int(200_000_000 / max(calib_ns * nops_cyc / BASE_B, 1.0)))
            for var in (0, 1):
                pair(lambda v=var, n=nops, cn=cap_n: ext.run_pingpong(
                        peer_ptr, ptr, samples.data_ptr(),
                        cptr(counters, C_ABORT), 0, iters, v, n, cn, 0, st),
                     lambda v=var, n=nops, cn=cap_n: ext.run_pingpong(
                        peer_ptr, ptr, samples.data_ptr(),
                        cptr(counters, C_ABORT), 1, iters, v, n, cn, 0, st),
                     f"ppb_{VAR_NAMES[var]}_b{nops_cyc}", lambda w_: init_part(w_))
            pair(lambda: ext.run_ll(peer_ptr, ptr, samples.data_ptr(),
                                    cptr(counters, C_STALE), cptr(counters, C_ABORT),
                                    0, iters, nops, cap_n, 0, st),
                 lambda: ext.run_ll(peer_ptr, ptr, samples.data_ptr(),
                                    cptr(counters, C_STALE), cptr(counters, C_ABORT),
                                    1, iters, nops, cap_n, 0, st),
                 f"ppb_packed8_b{nops_cyc}", lambda w_: init_part(w_))

    # ================= (b) data-before-flag ordering =================
    if want("order"):
        for dv in (0, 4, 8):
            for fv in (1, 6, 7, 2, 5):
                pair(lambda: ext.run_order(peer_ptr, ptr, samples.data_ptr(),
                                           cptr(counters, C_STALE), cptr(counters, C_ABORT),
                                           0, iters, dv, fv, BASE_B // 63, CAP, 0, st),
                     lambda: ext.run_order(peer_ptr, ptr, samples.data_ptr(),
                                           cptr(counters, C_STALE), cptr(counters, C_ABORT),
                                           1, iters, dv, fv, BASE_B // 63, CAP, 0, st),
                     f"order_d{VAR_NAMES[dv]}_f{VAR_NAMES[fv]}",
                     lambda w_: order_part(w_))

    # ================= (c) LL packed 8B round-trip + tearing =================
    if want("ll"):
        pair(lambda: ext.run_ll(peer_ptr, ptr, samples.data_ptr(),
                                cptr(counters, C_STALE), cptr(counters, C_ABORT),
                                0, iters, BASE_B // 63, CAP, 0, st),
             lambda: ext.run_ll(peer_ptr, ptr, samples.data_ptr(),
                                cptr(counters, C_STALE), cptr(counters, C_ABORT),
                                1, iters, BASE_B // 63, CAP, 0, st),
             "ll_packed", lambda w_: init_part(w_))

    # ================= (d) contention: in-kernel hammer blocks + backoff ========
    if want("cont"):
        for nops_cyc in (0, 2000, 10000, 20000, 40000):
            nops = nops_cyc // 63
            cap_n = max(2000, int(200_000_000 / max(calib_ns * nops_cyc / BASE_B, 1.0)))
            pair(lambda: ext.run_pingpong(peer_ptr, ptr, samples.data_ptr(),
                                          cptr(counters, C_ABORT), 0, iters, 1, nops, cap_n, hammer_iters, st),
                 lambda: ext.run_pingpong(peer_ptr, ptr, samples.data_ptr(),
                                          cptr(counters, C_ABORT), 1, iters, 1, nops, cap_n, hammer_iters, st),
                 f"cont_pingpong_b{nops_cyc}", lambda w_: init_part(w_))
            pair(lambda: ext.run_ll(peer_ptr, ptr, samples.data_ptr(),
                                    cptr(counters, C_STALE), cptr(counters, C_ABORT),
                                    0, iters, nops, cap_n, hammer_iters, st),
                 lambda: ext.run_ll(peer_ptr, ptr, samples.data_ptr(),
                                    cptr(counters, C_STALE), cptr(counters, C_ABORT),
                                    1, iters, nops, cap_n, hammer_iters, st),
                 f"cont_ll_b{nops_cyc}", lambda w_: init_part(w_))

    # ================= (f) store x load visibility matrix ======================
    if want("vis"):
        VC_CAP = 100_000  # ~100 ms window at backoff(15)
        probe_i = 0
        for sv in (0, 1, 2, 3, 4, 5, 6, 7):
            for rv in range(5):
                probe_i += 1
                tag = 0x10000 + probe_i * 251
                clean()
                t0 = time.monotonic_ns()
                ext.run_vis_probe(peer_ptr, ptr, 0, cptr(counters, C_ABORT),
                                  rank, sv, rv, VC_CAP, tag, st)
                torch.cuda.current_stream().synchronize()
                wall_ms = (time.monotonic_ns() - t0) / 1e6
                xbar()
                emit(dict(test="vis", store_variant=VAR_NAMES[sv], load_variant=LD_NAMES[rv],
                          seen=int(counters[C_ABORT]), wall_ms=round(wall_ms, 2)))

    # ================= (h) delivery under production-shape spin ================
    # Writer posts tags (~1 us apart); reader spins the SAME line with N threads,
    # backoff b. Measures inbound-write delivery latency under hot local polling —
    # the V0 livelock hypothesis, isolated from CUDA graphs.
    if want("deliver"):
        DCAP = 300_000  # writer repost cap ~300 ms
        di = 0
        for nthr in (64, 1024):
            for b_cyc in (0, 100, 500, 2000, 10000):
                di += 1
                b = max(0, b_cyc // 63)
                tag = 0x20000 + di * 4096
                clean()
                t0 = time.monotonic_ns()
                ext.run_deliver(peer_ptr, ptr, rank, b, DCAP, tag, nthr, st)
                torch.cuda.current_stream().synchronize()
                wall_ms = (time.monotonic_ns() - t0) / 1e6
                xbar()
                w = read_scratch()
                emit(dict(test="deliver", threads=nthr, backoff_cycles=b_cyc,
                          writer_reposts=int(w[S_RSPT_U32 + 13]),
                          reader_spins=int(w[S_RSPT_U32 + 14]),
                          wall_ms=round(wall_ms, 2)))

    out.close()
    dist.destroy_process_group()
    log(rank, "DONE")


if __name__ == "__main__":
    main()

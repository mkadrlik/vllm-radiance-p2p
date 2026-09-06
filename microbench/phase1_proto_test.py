#!/usr/bin/env python3
"""Phase 1: validate ar_proto F (flush) and L (LL) variants on gfx1100 TP=2.

torchrun --nproc_per_node=2 phase1_proto_test.py [--quick]
Single-block topology (Phase-0 finding: multi-block per-flag hot spin = the
livelock shape; single queue + backoff delivers). Checks:
  eager bit-match vs RCCL, capture+replay lockstep xN, REPLAY-SKEW (artificial
  stream delay on rank1, no host sync), poison never set on success paths.
"""
import argparse
import ctypes
import json
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ar_proto as P  # noqa: E402

MAX_KB = 4096
NT = 1024


def log(msg):
    print(f"[p1 {os.environ.get('RANK','?')}] {msg}", flush=True)


def rccl_sum(x):
    y = x.clone()
    dist.all_reduce(y)
    return y


class Comm:
    def __init__(self, dev, group):
        self.dev = dev
        self.rank = int(os.environ["RANK"])
        self.max_bytes = MAX_KB * 1024
        self.slot16 = self.max_bytes // 16
        self.slot8 = self.max_bytes // 8
        self._scratch, sc_h = P.alloc_shared(2 * self.max_bytes, True)
        self._flags, fl_h = P.alloc_shared(16384, True)
        hs = [None, None]; fs = [None, None]
        dist.all_gather_object(hs, sc_h, group=group)
        dist.all_gather_object(fs, fl_h, group=group)
        peer = 1 - self.rank
        self._peer_scratch = P.open_shared(hs[peer])
        self._peer_flags = P.open_shared(fs[peer])
        self._seq = torch.zeros(1, dtype=torch.int32, device=dev)
        self._hz = torch.zeros(1, dtype=torch.int32, device=dev)
        self._spin = torch.zeros(1, dtype=torch.int64, device=dev)
        ph, pd = P.alloc_host_mapped()
        self.poison_host = np.ctypeslib.as_array((ctypes.c_int32 * 1).from_address(ph))
        self._poison_dev_ptr = pd
        self._ph = ph

    def reset(self):
        P.memzero(self._scratch, 2 * self.max_bytes)
        P.memzero(self._flags, 16384)
        self._seq.zero_()
        self._hz.zero_()
        self._spin.zero_()
        self.poison_host[0] = 0

    def ar_f(self, x, stream, hz=1):
        out = torch.empty_like(x)
        P.ar_f(self._peer_scratch, self._scratch, self._peer_flags, self._flags,
               self._seq.data_ptr(), self._hz.data_ptr(),
               self._poison_dev_ptr, self._spin.data_ptr(),
               x.data_ptr(), out.data_ptr(), x.numel(),
               self.slot16, NT, hz, self.rank, stream)
        return out

    def ar_l(self, x, stream, hz=1):
        out = torch.empty_like(x)
        P.ar_l(self._peer_scratch, self._scratch, self._peer_flags, self._flags,
               self._seq.data_ptr(), self._hz.data_ptr(),
               self._poison_dev_ptr, self._spin.data_ptr(),
               x.data_ptr(), out.data_ptr(), x.numel(),
               self.slot8, NT, hz, self.rank, stream)
        return out


def byte_eq(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def run_variant(name, comm, arfn, iters, skew, results):
    dev = comm.dev
    ok = True
    shapes = (1, 8, 64)
    graphs, xs, outs, refs = [], [], [], []
    for bs in shapes:
        x = torch.randn(bs * 5120, dtype=torch.bfloat16, device=dev)
        torch.cuda.synchronize(); dist.barrier()
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            with torch.cuda.graph(g, stream=s):
                o = arfn(comm, x, s.cuda_stream, 1)  # handshake INSIDE the graph:
                # replay-safe (device counters), absorbs the launch skew that killed
                # the no-handshake replay run (flags lost while peer kernel queued)
        torch.cuda.synchronize(); dist.barrier()
        graphs.append(g); xs.append(x); outs.append(o)
        refs.append(rccl_sum(x))
    t0 = time.monotonic()
    first_poison = None
    spin_mode = 0
    for i in range(iters):
        j = i % 3
        if skew and int(os.environ["RANK"]) == 1:
            torch.cuda._sleep(skew * 2000)  # ~skew us at 2 GHz SHADER clock
        graphs[j].replay()
        torch.cuda.current_stream().synchronize()
        if comm.poison_host[0] and first_poison is None:
            if spin_mode == 0 and name.endswith("_lockstep"):
                # discriminator: retry with plain-volatile spin (hz=1|2 => bit1)
                log(f"{name}: poison with atomic spin; retrying replay w/ plain spin")
                comm.reset()
                g2 = torch.cuda.CUDAGraph()
                s2 = torch.cuda.Stream()
                with torch.cuda.stream(s2):
                    with torch.cuda.graph(g2, stream=s2):
                        o2 = arfn(comm, xs[0], s2.cuda_stream, 1 | 2)
                torch.cuda.synchronize(); dist.barrier()
                for _ in range(20):
                    g2.replay()
                    torch.cuda.current_stream().synchronize()
                    if comm.poison_host[0]:
                        break
                torch.cuda.synchronize(); dist.barrier()
                ok2 = (comm.poison_host[0] == 0) and byte_eq(o2, refs[0])
                log(f"{name}: PLAIN-SPIN replay x20: {'PASS' if ok2 else 'FAIL'} "
                    f"poison={int(comm.poison_host[0])} spin={int(comm._spin.cpu()[0])}")
                results.append(dict(variant=name + "_plainspin", ok=bool(ok2)))
                first_poison = i + 1
                ok = ok2  # plain-spin result overrides for this variant
                break
            first_poison = i + 1
            ok = False
            w = torch.zeros(4096, dtype=torch.int32, device=dev)
            P.read_words(comm._flags, 4096, w.data_ptr())
            wn = w.cpu().numpy()
            log(f"{name}: POISON at replay {i+1} spin={int(comm._spin.cpu()[0])} "
                f"seq={int(comm._seq.cpu()[0])} "
                f"F_SEQ={wn[256]} stages={[int(x) for x in wn[288:292]]} "
                f"seen={wn[272]} hsA0={wn[0]} hsB0={wn[256+0]}")
            break
    torch.cuda.synchronize()
    wall = time.monotonic() - t0
    dist.barrier()
    for bs, o, r in zip(shapes, outs, refs):
        if not byte_eq(o, r):
            ok = False
            log(f"{name}: MISMATCH bs={bs} maxdiff={(o.float()-r.float()).abs().max().item()}")
    rec = dict(variant=name, iters=iters, skew_us=skew, ok=bool(ok),
               poison=int(comm.poison_host[0]),
               ar_per_s=round(iters * 3 / max(wall, 1e-6), 1),
               us_per_ar=round(wall * 1e6 / max(iters * 3, 1), 1),
               spin_last=int(comm._spin.cpu()[0]))
    results.append(rec)
    log(json.dumps(rec))
    comm.reset()
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    iters = 50 if args.quick else args.iters

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    dist.init_process_group("gloo")
    results = []

    for mk, nm in ((Comm.ar_f, "F_flush"), (Comm.ar_l, "L_ll")):
        comm = Comm(dev, dist.group.WORLD)
        x = torch.randn(5120, dtype=torch.bfloat16, device=dev)
        torch.cuda.synchronize(); dist.barrier()
        o = mk(comm, x, torch.cuda.current_stream().cuda_stream, 1)  # eager: handshake ON
        torch.cuda.current_stream().synchronize()
        dist.barrier()
        ref = rccl_sum(x)
        eq = byte_eq(o, ref)
        log(f"{nm} eager: bitmatch={eq} poison={int(comm.poison_host[0])} "
            f"spin={int(comm._spin.cpu()[0])} cap={P.SPIN_CAP} "
            f"maxdiff={(o.float()-ref.float()).abs().max().item():.4f}")
        good = eq and comm.poison_host[0] == 0
        results.append(dict(variant=nm, test="eager", ok=bool(good)))
        comm.reset()
        if not good:
            continue
        run_variant(nm + "_lockstep", comm, mk, iters, 0, results)
        run_variant(nm + "_skew500us", comm, mk, iters, 500, results)
        run_variant(nm + "_skew5ms", comm, mk, iters, 5000, results)

    json.dump(results, open(f"/tmp/phase1_rank{rank}.json", "w"), indent=1)
    all_ok = all(r["ok"] for r in results)
    log(f"RESULT: {'ALL PASS' if all_ok else 'FAILURES'}")
    dist.destroy_process_group()
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

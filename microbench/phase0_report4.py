#!/usr/bin/env python3
"""Report for rev4 phase0 logs (init/rsp/vis tables)."""
import json
import sys

recs = []
for path in sys.argv[1:]:
    for line in open(path):
        if line.startswith("JSON0 "):
            recs.append(json.loads(line[6:]))


def row(r):
    t = r["test"]
    return (f"{t:34s} p50={r.get('init_p50_us'):>9} p90={r.get('init_p90_us'):>10} "
            f"p99={r.get('init_p99_us'):>10} max={r.get('init_max_us'):>10} "
            f"ab={r.get('init_aborted',0):>3} n={r.get('init_n',0):>4} "
            f"poll={r.get('poll_ns_derived'):>7} rsp_p50={r.get('rsp_p50_us')}/"
            f"{r.get('rsp_max_us')} tail={r.get('rsp_tail')}")


print("== (a) pingpong RTT, backoff 2000 cyc ==")
for r in recs:
    if r["test"].startswith("pingpong_"):
        print(row(r))
print("\n== (a2) backoff sweep, no hammer ==")
for r in recs:
    if r["test"].startswith("ppb_"):
        print(row(r))
print("\n== (b) order (stale>0 => flag outran data) ==")
for r in recs:
    if r["test"].startswith("order_"):
        print(f"{r['test']:44s} stale={r.get('stale')} fresh={r.get('fresh')} "
              f"f7={r.get('rsp_fresh_first7')} p50={r.get('init_p50_us'):>9} "
              f"p99={r.get('init_p99_us'):>10} ab={r.get('init_aborted',0)} "
              f"never={r.get('aborts')} poll={r.get('poll_ns_derived')}")
print("\n== (c) ll_packed ==")
for r in recs:
    if r["test"] == "ll_packed":
        print(row(r))
print("\n== (d) contention (8 blocks x 1024 thr x 20k uncached reads) ==")
for r in recs:
    if r["test"].startswith("cont_"):
        print(row(r))
print("\n== (f) visibility matrix (seen=1 => reader rv observed writer sv) ==")
seen = {}
for r in recs:
    if r["test"] == "vis":
        seen[(r["store_variant"], r["load_variant"])] = r["seen"]
loads = ["plain_ld", "atomic_acq_sys", "buf_ld_sc0", "buf_ld_sc1", "buf_ld_sc1_wait"]
stores = ["plain", "atomic_rel_sys", "tfs_then_store", "buffer_sc0", "buffer_sc1",
          "buffer_sc1_readflush", "store+vmcnt0", "store+vmcnt0_excnt0"]
print(f"{'store':24s} " + " ".join(f"{l[:14]:>15s}" for l in loads))
for s in stores:
    print(f"{s:24s} " + " ".join(f"{seen.get((s, l), '?'):>15}" for l in loads))

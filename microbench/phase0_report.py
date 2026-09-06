#!/usr/bin/env python3
"""Aggregate phase0 JSONL into the PHASE0_RESULTS tables. Usage:
python3 phase0_report.py /tmp/phase0_full2.log [more logs...]"""
import json
import sys

recs = []
for path in sys.argv[1:]:
    for line in open(path):
        if line.startswith("JSON0 "):
            recs.append(json.loads(line[6:]))

def g(r, *keys):
    for k in keys:
        if k in r and r[k] is not None:
            return r[k]
    return ""

print(f"== pingpong RTT (us), backoff 2000 cyc ==")
print(f"{'variant':28s} {'p50':>8} {'p90':>8} {'p99':>9} {'max':>10} {'abrt':>5} {'wall/it':>9}")
for r in recs:
    if r["test"].startswith("pingpong_"):
        print(f"{r['test'][9:]:28s} {g(r,'rtt_p50_us'):>8} {g(r,'rtt_p90_us'):>8} "
              f"{g(r,'rtt_p99_us'):>9} {g(r,'rtt_max_us'):>10} {g(r,'rtt_aborted'):>5} "
              f"{r['wall_us_per_iter']:>9}")

print("\n== order (data-before-flag): stale>0 = flag outran data ==")
print(f"{'d/f':40s} {'stale':>6} {'never':>6} {'fresh':>6} {'p50':>8} {'p99':>9} {'max':>10} {'ab':>4}")
for r in recs:
    if r["test"].startswith("order_") or r["test"].startswith("cont_order"):
        print(f"{r['test']:40s} {r['stale']:>6} {r['never']:>6} "
              f"{r['fresh_on_first_read']}/{r['reads']:>4} {g(r,'rtt_p50_us'):>8} "
              f"{g(r,'rtt_p99_us'):>9} {g(r,'rtt_max_us'):>10} {g(r,'rtt_aborted'):>4}")

print("\n== ll_packed ==")
for r in recs:
    if r["test"] == "ll_packed":
        print(json.dumps({k: v for k, v in r.items() if k not in ("rank",)}, indent=1))

print("\n== contention (in-kernel hammer, 8 blocks x 1024 thr x 20k reads) ==")
print(f"{'test':28s} {'p50':>8} {'p90':>8} {'p99':>9} {'max':>10} {'ab':>4} {'wall/it':>9}")
for r in recs:
    if r["test"].startswith("cont_") and not r["test"].startswith("cont_order"):
        print(f"{r['test']:28s} {g(r,'rtt_p50_us'):>8} {g(r,'rtt_p90_us'):>8} "
              f"{g(r,'rtt_p99_us'):>9} {g(r,'rtt_max_us'):>10} {g(r,'rtt_aborted'):>4} "
              f"{r['wall_us_per_iter']:>9}")

print("\n== visibility matrix (store -> load seen? spins) ==")
seen = {}
for r in recs:
    if r["test"] == "vis":
        seen[(r["store_variant"], r["load_variant"])] = (r["seen"], r["spins"], r["wall_ms"])
loads = ["plain_ld", "atomic_acq_sys", "buf_ld_sc0", "buf_ld_sc1", "buf_ld_sc1_wait"]
stores = ["plain", "atomic_rel_sys", "tfs_then_store", "buffer_sc0", "buffer_sc1",
          "buffer_sc1_readflush", "store+vmcnt0", "store+vmcnt0_excnt0"]
print(f"{'store':24s} " + " ".join(f"{l[:14]:>16s}" for l in loads))
for s in stores:
    row = []
    for l in loads:
        v = seen.get((s, l))
        row.append("MISS" if not v or not v[0] else f"{v[1]}")
    print(f"{s:24s} " + " ".join(f"{x:>16s}" for x in row))

print("\n== RCCL baseline ==")
for r in recs:
    if r["test"] == "rccl_allreduce":
        print(f"numel={r['numel']:>8}  us_per_op={r['us_per_op']}")

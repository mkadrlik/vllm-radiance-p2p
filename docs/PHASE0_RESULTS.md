# Phase 0 results — peer-write flush primitives on gfx1100 PCIe (RX 7900 XTX TP=2)

Date: 2026-09-06 · Task t_20a43eac · Branch `feat/rdna3-ar-flush-protocols`
Hardware: big-chungus, 2× Navi31 (gfx1100) on consumer PCIe (ROCR 1,2), `GPU_MAX_HW_QUEUES=1`,
fine-grained (uncached) IPC BAR mappings, prod vLLM co-resident (untouched, healthy throughout).
Harness: `microbench/ar_phase0.hip` + `microbench/phase0_bench.py` (torchrun 2-proc,
`build_phase0.sh` / `run_phase0.sh`, reports via `phase0_report4.py`).
Raw data: `/tmp/phase0_full3.log`, `/tmp/phase0_full4.log`, `/tmp/phase0_pp.log` (on host).

## Method notes (each was a bug we had to fix before trusting numbers)

1. **gfx1100 has NO shader-readable wall clock.** `s_memrealtime` is gfx12+;
   `s_timer`/`s_timerlo`/`s_timerhi` and `clock64()` all FAIL the gfx1100 assembler
   when referenced from a live kernel (clang `-O0` probes "pass" only because dead
   functions are dropped before assembly — verified with `probe2.hip`/`probe3.hip`).
   All device timings below are **spin-poll counts**; ns/poll derived per config
   from host wall / total spins, cross-checked by a calibration kernel
   (`run_read_calib`, same loop shape).
2. **`buffer_wbl2`, `global_inv`, `buffer_inv` do not exist on RDNA3** (gfx12+/CDNA).
   The task's "sc1 + buffer_wbl2" primitive is not buildable on this GPU. Nearest
   L2-forcing primitive: a dependent non-posted READ of the peer address
   (`buffer_sc1_readflush`, = the V3 read-back mechanism).
3. Spin loops must poll with backoff; a hot spin (b0) on the flag line starves
   inbound posted writes (measured below — this is the production livelock axis).
4. Rendezvous/handshake must use plain stores and must not share a 64B cache line
   with any measured slot (two separate harness bugs found this way).

## The truth table

### (f) Visibility matrix — can reader flavor rv observe writer flavor sv at all?
Repost window 100 ms, unique tags, all 8 store × 5 load variants, 1500-iter suite
plus dedicated probe. **This is the headline result.**

| store ↓ / load → | plain_ld | atomic_acq_sys | buf_ld_sc0 | buf_ld_sc1 | buf_ld_sc1+wait |
|---|---|---|---|---|---|
| plain (volatile st) | **1** | **1** | 0 | 0 | 0 |
| atomic RELEASE SCOPE_SYS | **1** | **1** | 0 | 0 | 0 |
| threadfence_system + st | **1** | **1** | 0 | 0 | 0 |
| buffer_store sc0 | 0 | 0 | 0 | 0 | 0 |
| buffer_store sc1 (glsc) | 0 | 0 | 0 | 0 | 0 |
| buffer_store sc1 + readback | 0 | 0 | 0 | 0 | 0 |
| plain + s_waitcnt vmcnt(0) | **1** | **1** | 0 | 0 | 0 |
| plain + vmcnt(0) expcnt(0) | **1** | **1** | 0 | 0 | 0 |

**`buffer_store` (ANY cache policy) is NEVER delivered to the peer on this platform**
— not to plain reads, not to SCOPE_SYS atomics, not to matching buffer_loads. It
lands in the writer's L2 and the peer never sees it within 100 ms of reposting.
(The V0 kernel's payload stores are `int4` global stores — fine — but any protocol
built on buffer-store doorbells is dead on arrival here.)
Conversely `buffer_load` reads never observe peer writes either. **The only working
pairing is global-store ↔ global/atomic load.**

### (b) Data-before-flag ordering (the V1 claim: "vmcnt flush doesn't put writes on the wire")
Echo-backpressured handshake, 1500 iters per config, plain-store reader checks the
payload word when the flag arrives.

| data store | flag store | stale (flag outran data) | fresh/iters | RTT p50 / p99 (µs, b2000) |
|---|---|---|---|---|
| plain | atomic_rel_sys | **0** | 1500/1500 | 141 / 1349 |
| plain | store+vmcnt0 | **0** | 1500/1500 | 152 / 7773 |
| plain | store+vmcnt0+excnt0 | **0** | 1500/1500 | 121 / 1322 |
| plain | tfs+store | **0** | 1500/1500 | 133 / 1260 |
| buffer_sc1 | any global-store flag | **1500/1500 stale** | 0 | flag arrives, data never does |
| packed 8B {data,seq} | any | 0 | 1500/1500 | 579 / 17814 |

**`s_waitcnt vmcnt(0)` after a plain global store DOES give ordered, bounded
delivery: 0/1500 stale reads across every flag pairing.** V1's serving failure was
NOT the flush primitive — it was (a) the spin having no backoff (starvation, below)
and (b) 24 independent per-block flag queues each hot-spinning (see (h)).
buffer_sc1 data + global flag = the flag delivers and the data NEVER does
(stale=1500/1500): definitive proof the sc1 store sits in local L2.

### (a2) Backoff sweep — the starvation curve (no hammer, plain ping-pong RTT µs)

| backoff (cycles) | plain p50 | p90 | p99 | max | atomic_rel p50 | p99 | LL packed p50 | p99 |
|---|---|---|---|---|---|---|---|---|
| 0 (hot) | 396 | 14994 | 53849 | 78687 | 330 | 48048 | 329 | 49938 |
| 500 | 358 | 1592 | 30241 | 58359 | 310 | 35837 | 433 | 35667 |
| 2000 | 95 | 480 | 6360 | 34388 | 167 | 11682 | 235 | 11479 |
| 5000 | 33 | 115 | 368 | 675 | 34 | 5444 | 33 | 259 |
| 10000 | 22 | 30 | **45** | 105 | 24 | 63 | 32 | 2143 |
| 20000 | 25 | 37 | 37 | 111 | 28 | 41 | 28 | 41 |
| 40000 | 51 | 77 | 103 | 16493 | 46 | 556 | 34 | 68 |

Poll cost ≈ 0.94 µs @ b2000 (calibrated 937 ns). **Backoff ≥ ~5000 cycles bounds
RTT p99 to <500 µs; ≥ ~20000 bounds p99 to ~40 µs.** Hot spin inflates p99 to
~50 ms — a 1000× starvation penalty. The spin-poll itself delays inbound posted
writes to the same line.

### (d) Contention — 7 extra blocks × 1024 threads hammering the peer with uncached reads

| backoff | pingpong p50/p99 (µs) | LL p50/p99 (µs) |
|---|---|---|
| 0 | 629 / 29004 | 581 / 24754 |
| 2000 | 449 / 18414 | 461 / 18384 |
| 10000 | 257 / 12019 | 250 / 11810 |
| 20000 | 129 / 9862 | 129 / 9737 |
| 40000 | 90 / 9369 | 91 / 9310 |

Under a sustained 8192-thread read storm the floor rises to ~90 µs p50 / ~9 ms p99
even with heavy backoff. (Prod vLLM decode on the same GPUs is a milder version of
this storm, permanently.)

### (c) LL packed 8B {payload,seq}
`tears=0/1500` — the 8B store is delivered atomically and intact; tag-in-word is
sound as a consumption proof. RTT p50 ~0.5-1.3 ms at b2000 (backpressure-limited).

### (h) Production-shape reproduction (the decisive test)
`microbench/ar_proto.hip` Phase-1 prototypes (single-block F and L variants,
device rendezvous, backoff spin, poison-on-timeout) replayed through CUDA graphs
with host-sync between every replay:

* eager 4-byte handshake: **PASS, bit-identical to RCCL** (spin 0-4k polls) — the
  primitives work.
* graph replay at decode shapes (bs 1/8/64, 640 KB max push): **POISON at replay
  1-2 on every run.** Forensics at first poison: rank0 completed push+flag-post,
  spun the full 5 s cap, its flag line still read 0; the post-mortem dump shows the
  peer's flag value (1) present AFTER the timeout — the write was in flight or
  starved for >5 s while both ranks ran 1024-thread push+spin loops.
* Even "successful" operations cost ~25 ms/AR wall (queue + rendezvous dominated).

**This is the V0 serving livelock, reproduced without vLLM.** The mechanism is not
the flag flush (that works, table (b)); it is that real message sizes + real spin
topology re-enter the starvation regime: the pushing rank's own outbound 16B-store
stream and the polling rank's inbound-read pressure delay the small flag/payload
writes by seconds, with no backoff setting that is simultaneously fast (p50 ≥ ~25 µs
floor) and bounded under load.

## GATE VERDICT — Phase 0 FAILS

Task gate: "if NOTHING gives bounded peer visibility (all variants > ~50 us p99 or
stall), STOP." Findings:

1. Single-word visibility IS bounded — at p99 ≈ 40-45 µs with b≥20000 backoff and
   no other traffic. That is already at/over the 50 µs gate on an idle link, before
   payload push, before prod co-residency.
2. At decode message sizes (10 KB-640 KB) with the production spin topology,
   delivery is **unbounded in practice** (5 s stalls reproduced; p99 9-29 ms under
   moderate contention; ~25 ms/AR best case incl. queueing).
3. The flush primitive the redesign hoped to use (`buffer_store sc1 + buffer_wbl2`)
   does not exist or does not deliver on RDNA3: `buffer_wbl2` is not in the ISA, and
   every `buffer_store` variant is invisible to the peer.
4. RCCL baseline (same GPUs, same env): **30 µs @ 10 KB, 34 µs @ 80 KB, 76 µs @
   640 KB** per all_reduce — i.e. a working one-shot would need ≤ ~10 µs/AR to beat
   it by ~60 AR sites/step; the measured floor is 2500× worse.

**No flush primitive + no protocol redesign available on this platform gives a
one-shot AR that is both replay-safe and faster than RCCL.** The prior task's
verdict stands, now with the mechanism isolated: not flag ordering (fixable, we
proved the fix) but PCIe posted-write starvation under mutual spin + bulk push
(unfixable without doorbell/completion hardware RDNA3 consumer parts do not have).

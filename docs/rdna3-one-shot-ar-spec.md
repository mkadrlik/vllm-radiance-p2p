# RDNA3 (gfx1100) one-shot all-reduce — decision spec

Date: 2026-09-06 · Task t_20a43eac · Branch `feat/rdna3-ar-flush-protocols` (NOT merged)
Verdict: **NOT VIABLE — evidence below.** `RADIANCE_FAST_REDUCE` stays 0.

Builds on `docs/fast-reduce-mtp-deadlock.md` (V0–V3) and `docs/PHASE0_RESULTS.md`
(the hardware truth table; read it first). This doc records what a redesign WOULD
require, why the platform cannot give it, and what remains if anyone revisits.

## 1. Phase 0 measurements (hardware truth — see PHASE0_RESULTS.md for tables)

| Question | Answer |
|---|---|
| Which flush primitive gives bounded peer visibility? | `plain global store + s_waitcnt vmcnt(0)`. Ordering holds: 0/1500 stale data-then-flag reads across all pairings. |
| `buffer_store` sc0/sc1 (+readback)? | **Never delivered to the peer, ever** (all 5 reader flavors, 100 ms repost window). `buffer_wbl2`/`global_inv` don't exist in the gfx1100 ISA. |
| `s_waitcnt vmcnt(0)` after store observable in bounded time? | YES for global stores (V1's failure was spin starvation + topology, not the flush). |
| LL 8B {payload,seq} single store? | Delivered intact (tears 0/1500), visible to plain + SCOPE_SYS loads. |
| Single-word RTT, idle link, best backoff (b≥20000)? | p50 ≈ 25 µs, p99 ≈ 40–45 µs. At the task's 50 µs gate edge with ZERO payload. |
| Same under prod-realistic contention (8 blocks × 1024 thr read storm)? | p50 ≈ 90 µs, p99 ≈ 9–29 ms. |
| Hot spin (b=0) effect — the starvation hypothesis? | CONFIRMED and quantified: p99 inflates ~1000× (50 ms vs 45 µs). Spin polls delay inbound posted writes to the same line. |
| Production-shape replay (Phase 1 prototype, CUDA graph, decode sizes)? | **5 s spin stalls reproduced with poison-forensics; eager path passes, replay path fails at replay 1–2.** The V0 serving livelock, isolated from vLLM. |
| RCCL baseline to beat (same GPUs/env)? | 30 µs @10 KB · 34 µs @80 KB · 76 µs @640 KB per all_reduce. |

## 2. Protocol choice: none viable — and why the two candidates die

**F (flush) protocol** — plain push + `vmcnt(0)` + single flag + backoff spin:
ordering is sound (§1 rows 1–3) and it PASSES eager. It dies at graph replay on
message sizes: the pushing rank streams 10–640 KB of outbound 16B stores while the
peer polls; the small flag write behind that stream is starved for seconds
(Phase 1 forensics: reader's own-line plain read returns 0 for the full 5 s spin
while the post-mortem host copy of the same line shows the flag present —
delivery/visibility under mutual spin+push is unbounded). Backoff alone cannot fix
it because the starvation here is caused by the reader's own polling AND the
writer's bulk push on the same link; both are required for the algorithm.

**L (LL) protocol** — 8B {payload,seq}, tag = consumption proof: sound in the
microbench (tears=0, visible), and it removes the separate-flag ordering problem
entirely. It inherits the same failure: under replay with real payloads the tag
word delivery is exactly as starvable as any other peer write (Phase 1: L variant
poisons at the same point as F).

The missing hardware capability is a **doorbell with completion semantics** (a
write that is guaranteed on the wire / an inbound-write flush primitive). CDNA and
gfx12+ have variants of it (`buffer_wbl2`, `s_memrealtime`-observable coherence);
RDNA3 consumer parts do not expose one to the shader, and the host driver does not
offer it through IPC BAR mappings. Without it, no store-side ordering trick bounds
delivery against a spinning + pushing peer.

## 3. What WAS disproven from the V0 post-mortem (corrections worth keeping)

1. Flag ordering was never the bug: `vmcnt(0)` flush + plain stores give ordered
   delivery (0/1500 stale across 15 combos). V1's "flags invisible until timeout"
   is explained by spin starvation + launch-skew wedge, not flush failure.
2. `RADIANCE_SPIN_MAX`-style fire-and-forget is still wrong, but the fix class
   (bounded spin + poison + host recovery) is implementable — see §4; it is simply
   moot while delivery is unbounded.
3. A single-block, single-queue handshake with device rendezvous DOES deliver
   small messages eagerly at ~µs spins. Multi-block per-flag hot-spin topology
   (V0's shape) is the starvation amplifier. Any future attempt starts from the
   Phase-1 `ar_proto.hip` topology, not V0's.

## 4. Kernel design (for the record — what would ship IF a future platform gives bounded delivery)

Memory layout (per TP pair, fine-grained IPC, every slot its own 64B line):
```
flags (8KB):  [256] F_SEQ  (or LL: none)
              [272] F_SEEN  [288+...] stages      (forensics)
              [32*16 .. 96*16)  hs_a ring 64 lines (rank0 -> rank1)
              [96*16 .. 160*16) hs_b ring 64 lines (rank1 -> rank0)
scratch (2 x max_bytes): slot = seq & 1
seq: device-resident u32, atomicAdd by block 0 (SPMD-identical, replay-safe)
poison: hipHostMalloc mapped int32; device sets on ANY spin timeout; kernel
        NEVER reduces after timeout (out left untouched)
```
Ordering sequence (F): push `int4` plain stores → `s_waitcnt vmcnt(0) expcnt(0)` →
`__syncthreads` → lane0 posts flag (plain + flush) → spin peer flag with
`load_acq` + `31 x s_nop 63` backoff (~2000 cyc/poll, measured sweet spot) →
timeout ⇒ poison+return, success ⇒ local reduce. Rendezvous ring (exact-match on
`MAGIC+seq`-free monotone values) absorbs launch skew inside the graph — it is
captured and replayed, so it costs one extra PCIe RTT per AR, no host sync.
Integration: `radiance_allreduce.py` keeps the capture gate (correct regardless);
`custom_all_reduce()` returns `None` permanently once `poison_host[0]` is seen —
the per-step eager check hooks in `CudaCommunicator.all_reduce` wrapper (one
cached-numpy int read, ~100 ns, amortized over ~60 sites/step ⇒ negligible).
Poison ⇒ comm disabled for the process lifetime, fall through to RCCL. This is the
recovery path V0 lacked; it converts a wedge into a ~5 s one-time stall + degraded
(fast→RCCL) continuation, never silent corruption.

## 5. Test plan (Phase 3 protocol — NOT authorized; recorded for completeness)

Bounded prod window via `/tmp/ab_fastreduce.sh` pattern: stop prod → start test
compose w/ `RADIANCE_FAST_REDUCE=1` + prototype overlay image → health gate 14 min
→ `bench27b` (gate: ≥ 20.4 tok/s) → greedy byte-match vs `/tmp/byte_match_prod.json`
→ restore trap on EXIT (PROD_RESTORED=healthy). Replay-skew stress in the harness
(`phase1_proto_test.py`: `torch.cuda._sleep` delay on rank1, 500 µs / 5 ms modes,
N=1000) must pass FIRST — it is the test V0–V3 and the toy t1–t4 suite all lacked,
and it is what caught the failure here at replay 1 instead of at request 1.

## 6. Expected-win arithmetic (why "bounded" must mean ≤10 µs, not ≤50 µs)

Decode step ≈ 49 ms @ 20.4 tok/s with MTP=2 ⇒ ~2 steps/token. AR sites/step ≈ 60
(41 layers × ~1.4 + draft). RCCL measured: 30–76 µs/op ⇒ 60 × ~32 µs ≈ 1.9 ms/step
≈ 3.9 % of step time — the entire theoretical ceiling. A one-shot AR must beat
30 µs/op INCLUDING payload push; the idle-link single-word floor alone is 25 µs
p50 (no payload), so even a perfect doorbell protocol nets ≈ 0–2 % decode gain.
The measured platform floor is 1000× above that under any realistic co-resident
load. There is no win band to chase on this hardware even setting aside safety.

## 7. Decision

NOT VIABLE on gfx1100 PCIe. Evidence: (1) buffer-store doorbells never deliver;
(2) no L2-flush/completion ISA exists on RDNA3; (3) single-word visibility is
bounded only at the 50 µs gate edge on an idle link — above the whole RCCL budget;
(4) production-shape replay stalls 5 s with reader-side non-observation of a
peer write that is verifiably present in the reader's own memory post-mortem
(F_SEEN=0 during the spin, 1 after; the eager path with identical load/store code
passes bit-exactly every run) — and the failure is load-flavor independent:
swapping the spin poll from `load_acq` (SCOPE_SYS atomic) to plain volatile reads
fails identically (Phase 1 discriminator run, `PLAIN-SPIN replay x20: FAIL`), so
no read-side ISA choice recovers visibility inside the spinning kernel;
(5) the theoretical win is ≤ 2 % of decode time. `RADIANCE_FAST_REDUCE` default
stays 0. The capture-gate fix (PR #1) stays — it is a strict precondition for any
future protocol work.

If this is ever revisited, the entry points are: a platform with inbound-write
completion (`buffer_wbl2` class), xGPU xGMI links, or a design that never spins
while pushing (staged double-buffer with the push drained before any poll starts —
the Phase-1 topology already does this per-block; it is the cross-rank coupling
that fails).

READY FOR PHASE 3 DISPATCH: **no.** Close as NOT VIABLE.

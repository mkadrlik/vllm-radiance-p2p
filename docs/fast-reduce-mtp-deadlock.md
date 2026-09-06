# RADIANCE_FAST_REDUCE MTP-capture deadlock — root cause + fix

Date: 2026-09-05 · Task: t_9567234e · Branch: `fix/fast-reduce-mtp-capture`

## Symptom (big-chungus, gfx1100 TP=2, Qwen3.8-27B W8A8, MTP=2, FULL_AND_PIECEWISE)

- `RADIANCE_FAST_REDUCE=0`: boots healthy, graphs capture 5/5 + 6/6, 20.4 tok/s.
- `RADIANCE_FAST_REDUCE=1`: `[radiance] custom all-reduce INSTALLED (rank=0/1 ...)` then hang at
  `Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 0/5`. 600 s later BOTH ranks:
  NCCL watchdog `WorkNCCL(SeqNum=6, OpType=_ALLGATHER_BASE)` in the MTP drafter's
  `dummy_run` → workers killed → crash-loop.

## Root cause

The one-shot P2P all-reduce kernel (`radiance_ar_ext.hip`) is a fire-and-forget spin
handshake: it waits for the peer's IPC flag with **no host-side rendezvous**
(`RADIANCE_SPIN_MAX` = 4e9 uncached PCIe reads ≈ 4000 s — longer than the 600 s NCCL
watchdog, so the spin never breaks in time).

Any **rank-divergent eager all_reduce** — dynamo/inductor warmup asymmetry, profiling
runs, rank-0-only work — leaves rank A's stream holding a kernel that spins on a flag
rank B will never post. The next **device-wide sync** then blocks forever on rank A.
`torch.cuda.graph()` capture_begin performs exactly such a sync. Hence the hang on the
*first* graph capture, right after asymmetric warmup, and the watchdog firing on the
next NCCL collective on both ranks (the peer whose AR never launched waits on its
allgather; the wedged rank never reaches it).

Contributing factors:

1. `RadianceAllreduce.capture()` (the no-op yield) was **dead code**: vLLM's
   `parallel_state.graph_capture()` wraps `ca_comm.capture()` (vLLM's CustomAllreduce),
   never `radiance_comm.capture()`. The class had zero awareness of capture state.
2. vLLM's own `CustomAllreduce` avoids this entire class: inside its capture context,
   warmup calls return `empty_like` **without launching**, and its eager path
   rendezvous via `register_graph_buffers`' `all_gather_object`.

## Fix

`should_custom_ar()` returns False unless `torch.cuda.is_current_stream_capturing()`:

- Eager all_reduces (warmup, profiling, prefill > max capture size) → RCCL. Symmetric,
  deadlock-free.
- The fast kernel runs **only from graph replay**, where the design's SPMD lockstep
  assumption actually holds: captured kernels don't execute at capture time, and vLLM's
  graph dispatch is driven by the broadcast scheduler output, so both ranks replay the
  identical graph sequence and per-block `seq_ctr[b]` stays in lockstep.
- Decode hot path (bs 1..64 graphs) keeps the fast kernel — no performance regression
  by construction; confirm with bench after deploy.

**Behavior change:** with `--enforce-eager` (no graphs) the fast path is now inert — every
all_reduce takes RCCL. That mode was already the slow path on radiance (graphs are the
recommended config); document it in the PR.

## Verification (real kernel, gfx1100 TP=2, 2026-09-05)

`scripts/verify_fast_reduce_capture.py` (torchrun, 2 procs, GPUs ROCR 1,2):

```
t1 eager gate: should_custom_ar=False custom_all_reduce->None PASS
t2 capture+replay x3 bit-identical to RCCL: PASS
t3 multi-graph interleaved replay (9 replays, 3 graphs): PASS
t4 decode-shape byte-identity (bs 1/8/64): PASS
RESULT: ALL PASS
```

`--danger` mode reproduces the pre-fix wedge: one divergent eager AR on rank0, then
both ranks pin 99% CPU with zero progress at the capture sync — the production failure
class, reproduced on demand.

Unit gate check: `scripts/test_ar_capture_gate.py` (6/6 PASS).

## Serving-time regression (found by live A/B, 2026-09-05, t_e12d6d90)

The capture gate fixed startup, but `RADIANCE_FAST_REDUCE=1` on `:fix-ar` still
kills the engine on the FIRST decode step: stream opens, `sample_tokens` RPC
times out at 300 s, EngineDeadError, every later request 500s. bs=1,
kv_cache_usage 0.045 — a plain decode replay wedged.

Instrumented diagnosis (image `:fix-ar-diag`: device-side spin-abort log +
side-stream seq/flag/done sampler + py-spy): both ranks' main threads block in
`radiance_draft._prepare_match_gpu` (first device sync of the step) while both
GPUs spin. The abort log shows both ranks at the SAME seq on ALL blocks with
`my_flags == seq` — the data handshake completed, the peer's kernel never
finished. This is a mutual-spin livelock inside graph replay, not a seq
divergence and not eager-path asymmetry.

Protocol variants tried and rejected with evidence:

| Variant | Harness (toy, torchrun) | Serving (real model) |
|---|---|---|
| V0 capture-gate only (`:fix-ar`, 95c9424) | 4/4 PASS | wedge at first decode, seq≈132 |
| V1 wire-flush (`s_waitcnt vmcnt+excnt` after flag store) | ALL PASS (once) | wedge at seq 3, flags invisible until timeout |
| V2 + done-guard, fixed partition, `atomicMax` peer posts | t4 FAIL/hang (rc=137) — remote RMW atomics stall on gfx1100 PCIe | not run |
| V3 + flag read-back flush (posted-write ordering) | wholesale corruption (every element differs) | not run |

Root problem: the one-shot kernel's correctness depends on both ranks' spin
loops observing each other's flags with bounded latency. On this platform
(RDNA3 + consumer PCIe, uncached fine-grained BAR, `GPU_MAX_HW_QUEUES=1`)
posted peer writes are not observably on the wire before the writer spins, and
non-posted RMW atomics stall outright. Inside a captured graph there is no
recovery path — a timed-out spin silently corrupts the reduction, so even the
"bounded spin + fallback" direction cannot be made safe without protocol
support the hardware does not give (doorbell flush / programmatic completion).
vLLM's own CustomAllreduce avoids this by registering graph buffers and using
a different handshake on NVLink-class platforms; it is inert on ROCm for a
reason.

**Decision: `RADIANCE_FAST_REDUCE` defaults to 0** (this commit): the env-gate
in `install_custom_ar`, the banner in `radiance_preamble.py`, and the
`Dockerfile.gfx1100` ENV all flip to off. Production compose already runs 0.
The capture-gate fix (95c9424) stays — it is strictly better than pre-fix —
but the fast path is inert by default until someone proves a replay-safe
handshake on real gfx1100 PCIe with the full model. Decode baseline stays
20.4 tok/s on RCCL; the fast kernel's theoretical win does not justify an
engine that dies on request one.

## Deploy vehicle

`Dockerfile.fix-ar` — overlay of the fixed `radiance_allreduce.py` onto the known-good
gfx1100 runtime image (`ghcr.io/mkadrlik/vllm-radiance-p2p:gfx1100`, 6253c8e6cf9c), so
the live A/B is exactly the one-file change. (Full `Dockerfile.gfx1100` source build is
separately blocked by the stilldeadcode:0.5.7 base drifting to gfx1201 — `patch_gfx1100`
anchor matches 0x; unrelated to this fix.)

## Upstream note (codeberg.org/StillDeadcode/radiance)

Any custom collective with a device-side flag spin and no host rendezvous must gate
itself to graph capture, or reset+barrier at capture entry. The "both ranks run
identical graph sequences" invariant holds for **replay**, not for **eager warmup** —
the eager path is where inductor/dynamo asymmetry lives.

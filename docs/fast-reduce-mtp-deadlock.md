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

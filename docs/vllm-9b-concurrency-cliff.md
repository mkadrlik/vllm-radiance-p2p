# vLLM 0.26 decode cliff on TP1 gfx1100: ≥5 concurrent streams (Ornith-1.5-9B)

**Status:** mitigated with `--max-num-seqs 4` in the `ornith-9b` profile. Root cause
is inside vLLM's TP1 decode path at ≥5 resident sequences; upstream issue drafted
(`vllm-decode-cliff-issue.md` below has the repro). Not a radiance bug.

## Signature

| concurrent decode streams | per-stream | aggregate |
|---|---|---|
| 1 | 81 t/s | 81 |
| 4 | 49.5 | 197 |
| **5** | **13.7** | **69** ← aggregate DROPS |
| 8 | 13.2 | 106 |
| 16 | 12.4 | 199 |
| 24 | 12.4 | 317 |

Step time plateaus at ~73–76 ms and becomes batch-size-independent from 5
streams onward. Uniform across all streams. Deterministic to 0.1 t/s across
repeats and restarts.

## Ruled out (each by direct experiment on this box)

- CUDA graphs: `--enforce-eager` has the same plateau (~76 ms/step); it only
  shifts the boundary 5→6 (graphs pad batch 5→8 capture size).
- Async scheduling: on or off, cliff identical (async is +7% at batch 1).
- Prefix caching / Mamba `align` mode: cliff identical with
  `--no-enable-prefix-caching`. (`--mamba-cache-mode none` is silently
  overridden to `align` whenever prefix caching is on.)
- Thermal/power: 42 °C junction during plateau, full clocks.
- Host contention: single container on its own GPU, second GPU pair idle.
- Weight/KV bandwidth: impossible — 5 streams × 9 GB cannot cost 4× more than
  4 streams.

## What remains

py-spy during the plateau: EngineCore MainThread parked in
`vllm/v1/worker/gpu_model_runner.py` `_bookkeeping_sync → _to_list`
(the pinned-buffer copy + `transfer_event.synchronize()` sampling path, itself
a mitigation for vllm#22754). With async scheduling that branch is skipped and
the plateau persists, so the fixed cost is deeper (suspect: GDN/paged-attn
Triton kernel behavior past 4 resident sequences on RDNA3 — the engine logs
`Cannot use ROCm custom paged attention kernel, falling back to Triton` on
gfx1100; the TP2 sibling model shows NO cliff, so it's TP1/uniproc-specific).

## Practical consequence

On TP1 gfx1100, cap `--max-num-seqs 4`. Queued arrivals beyond that still run
at 49–51 t/s/stream in waves (batch 8 offered → two waves of 4 → ~410 t/s
effective aggregate, 5.4 s extra TTFT for wave 2). A 4-stream cap that runs at
50 t/s beats an uncapped engine that collapses to 13.

## Repro

`scripts/batch_curve.py <url> <model> 1 4 5 8 16` — parallel clients released
on a gate; reports per-stream median and aggregate. See also
`vllm-decode-cliff-issue.md` for the upstream-ready report.

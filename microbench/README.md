# microbench/ — Phase 0/1 evidence for the RDNA3 one-shot AR decision

Run order / provenance:
- `build_phase0.sh` + `run_phase0.sh` — Phase 0 flush-primitive microbench
  (`ar_phase0.hip`, `phase0_bench.py`); reports via `phase0_report4.py`.
- `build_proto.sh` + `run_phase1.sh` — Phase 1 prototype kernels
  (`ar_proto.hip`: F flush + L LL variants, single-block topology, host-mapped
  poison) validated by `phase1_proto_test.py` (eager, graph-replay lockstep,
  replay-skew 500 µs / 5 ms).
- `probe_isa.hip` / `probe2.hip` / `probe3.hip` (+ `probe*.sh`) — gfx1100 ISA
  capability probes: which timer/flush/store builtins actually assemble in a live
  kernel. Findings baked into docs/PHASE0_RESULTS.md.
- `results/` — raw JSONL-bearing logs of the runs cited in
  `docs/PHASE0_RESULTS.md` and `docs/rdna3-one-shot-ar-spec.md`
  (phase0_full3/full4 = truth tables; phase0_pp = pp+deliver; phase1_run* =
  prototype replay failures incl. the plain-spin discriminator in run4).

Everything runs inside `vllm-radiance-gfx1100:fix-ar-diag` on ROCR 1,2 with
`timeout -s KILL` and device-side bounded spins; prod stays up.

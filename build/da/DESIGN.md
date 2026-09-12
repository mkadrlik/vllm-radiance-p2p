# DA-P3: Declarative Attention block-table masking in vLLM (radiance)

Paper: arXiv:2609.02737 App. B. Client-side protocol already shipped in Hermes
(`plugins/context_engine/declarative_attention/`, P0/P1 evidence in
`/home/hermes-bot/da-eval/`). This is the serving-side mask: the whole mechanism
(P1 without it is DA-nm, which the paper shows costs *more*).

## Why compaction is position-safe (the load-bearing fact)

vLLM applies RoPE at KV **write** time; the paged kernel dot-products stored,
already-rotated keys. The block table is a pure gather list. So we may reorder /
compact kept blocks to the front of a request's row and shrink that row's
`seq_len` to the kept token count: the query's rotation comes from its own
position (scheduler state, untouched), keys keep theirs. No kernel change.
Causal masking inside the kernel (query_len>1, incl. MTP verify steps) stays
consistent because every kept block precedes the response and the response is
fully kept.

## Regions kept attended, every mode (App. B)

1. **Sink**: prompt tokens [0, 16) — engine guarantees content-free scaffold there.
2. **Question window**: last `da_window` tokens of the prompt (default 512),
   covering question + DA instruction.
3. **Response**: every block of generated tokens so far (always, in every mode).
4. `<focus K>` additionally keeps segment K's blocks; `<local>` keeps only 1–3.

Mask granularity: block-aligned, rounded **outward** (≤ b−1 extra tokens per
edge; b=16 on our stack, negligible vs 2048-token segments).

## Wire format (client → engine)

`vllm_xargs` (exists in served 0.26.0,
`entrypoints/openai/chat_completion/protocol.py:461` → lands in
`SamplingParams.extra_args`, visible worker-side via
`model_runner.requests[req_id].sampling_params`):

```json
{"da": {"segs": [[start_tok, end_tok], ...], "win": 512, "sink": 16}}
```

`segs` = token spans of the addressable segments in the rendered prompt, in
magic-chunk order (the Hermes engine builds the transcript, so it knows them
structurally). Absent `da` xarg → request is untouched (zero behavior change for
every other workload).

## State machine (server-side)

Per decode step, re-parse only the **response tail** as text (O(generated), not
O(N)): feed newly sampled token ids through the same incremental DA parser as
P0/P1 (`da-eval/state_machine.py` port; tolerant: malformed `<focus>` → global).
Track (mode, kept segment set). Tag vocabulary is decoded from the response
span only — cheap at any context length.

## Hook surface (served image: vllm 0.26.0, ROCM_ATTN backend)

Two monkeypatches, installed from a new radiance module via the existing
`radiance_kernels.install_all()` plugin entry (runs in every process incl. TP
workers before model load; env-gated `VLLM_DA_ATTN=1`, default off):

1. `GPUModelRunner._update_states` (or a wrapper on `_build_attention_metadata`
   input side): maintain `req_id -> DAState` (segment spans from extra_args,
   parser, last parsed response length).
2. `RocmAttentionMetadataBuilder.build` (v1/attention/backends/rocm_attn.py:114):
   after `attn_metadata` is constructed, for each decode row with an active DA
   state: compute keep-block list, rewrite `block_table` row **in place**
   (compacted kept ids first, tail filled with the last kept id — pointer
   validity), and write masked value into `seq_lens` **GPU buffer slot for that
   row** before returning. `slot_mapping` is already materialized upstream from
   the original block table — writes are unaffected. In-place edits of the
   persistent (max_reqs, max_blocks) buffer are CUDA-graph-safe (contents only,
   no shape/pointer change); `build()` runs eager per step and rewrites each
   time, so graph replay reads whatever the last build wrote — same invariant
   vLLM already relies on for seq_lens.

Hybrid model: hook attaches to `RocmAttentionMetadataBuilder` only. Qwen3.8's
GDN layers use a Mamba-style builder (kv_cache_interface.py:690 `MambaSpec`) —
untouched, exactly the paper's "global attention layers only" scope.

## Correctness invariants (must hold; tests enforce)

- I1: `da` xarg absent → byte-identical behavior to today (hook no-ops).
- I2: prefill/chunked-prefill rows (query spans > 1 new token AND no response
  yet) are never masked — the model cannot have emitted a focus tag yet.
- I3: kept set always ⊇ {sink blocks, response blocks, question-window blocks};
  the row's last scheduled write block is always kept.
- I4: masked `seq_lens[i]` = (number of kept blocks)·b capped at real kept-token
  span; never exceeds the true seq_len.
- I5: on parser confusion mid-stream (tag straddling steps), state is monotone
  within a step; recovery = global (superset) — wrong-mode is a perf issue,
  dropped-response is a correctness issue.
- I6: prefix caching: masking only shrinks *reads* of this request's own blocks;
  no other request's blocks are ever referenced. Kept ids are a subset of the
  row's own ids → safe.

## Kill switches

- `VLLM_DA_ATTN=0` (default): hooks not installed at all.
- Per-request opt-in via xarg: a bad client cannot affect other traffic.
- Any exception inside the DA path → log once, fall back to unmasked for that
  request permanently. The serving path must never 500 because of DA.

## Deploy vehicle

Overlay on the pinned known-good image (same pattern as Dockerfile.fix-ar):
COPY `da_attention.py` → site-packages, one patch script adding the
`install_da_hook()` call into `radiance_kernels.install_all`. One-edit
one-restart swap; gated on the P2 sweep finishing (never drop in-flight work).

## Honest expectations (measure, don't assert)

- P2 emask (placeholder-withhold) measures the *prefill* side; P3 removes
  *decode* KV reads. On bs1 PCIe 7900 XTX, decode at 42K context is
  KV-read-dominated (roofline, spec §Phase-3), but the +15–35% extra decode
  steps tax is real — P0's brevity effect (−29% completion tokens, measured)
  offsets it. Net must be measured end-to-end on this box.
- MTP spec decode (num_spec=2, enabled in prod) interacts: verify steps get
  masked too (safe by the response-always-kept invariant) but rejection
  resampling re-reads — no correctness risk, perf unknown.

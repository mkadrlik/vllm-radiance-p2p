#!/usr/bin/env python3
"""DA vLLM integration (arXiv:2609.02737, App. B) — radiance overlay module.

Env-gated (VLLM_DA_ATTN=1, default off), installed from
radiance_kernels.install_all() so it runs in every process (API server +
engine-core + TP workers) before model load.

Mechanism
---------
The client (Hermes declarative_attention engine) sends per-request segment
geometry via vllm_xargs (FLAT — the field is typed
dict[str, str|int|float|list[str|int|float]]; nested dicts fail pydantic,
verified against served 0.26.0):

    vllm_xargs = {"da_segs": [s0,e0, s1,e1, ...], "da_win": 512, "da_sink": 16}

spans are TOKEN offsets of the addressable magic-chunk segments inside the
rendered prompt (1-based chunk k -> segs[k-1]). A per-request state machine
(da_state_machine.py) re-parses only the response tail every step; the mode
decides which context segments stay attended:

    global -> all segments          (mask is a no-op; the common early state)
    focus  -> named segments only
    local  -> no segments (scaffold + response only)

Always attended in every mode (App. B): attention sink (first `sink` prompt
tokens), question window (last `win` prompt tokens), the entire response.

Masking compacts the kept blocks to the front of the request's block-table ROW
and shrinks seqused_k for that row to the exact kept-token count. Position-safe
because vLLM stores RoPE-rotated keys: the block table is a pure gather list;
the query's rotation comes from its own position in scheduler state we never
touch.

Hook point + CUDA-graph reasoning
---------------------------------
Wrap GPUModelRunner._build_attention_metadata: after it returns, EVERY kv
group's builder (incl. GDN in align mode, which consumes seq_lens at build
time) has materialized its derived values from the TRUE buffers. We then
mutate the PERSISTENT buffers in place (block_table.gpu rows + runner.seq_lens
slots). Content-only mutation of persistent buffers is the exact pattern vLLM
uses for its own per-step updates and is visible to:
  - eager attention (md.seq_lens / md.block_table are slice views of the same
    storage), and
  - FULL cudagraph replay (the captured graph reads the same pointers).
Cloning would be WRONG here: a captured graph re-reads original pointers, so a
clone silently no-ops under full-graph decode.

Per-step truth: _prepare_inputs runs before the metadata build every step and
restores both buffers from scheduler state (commit_block_table copies the CPU
row; seq_lens is recomputed as num_computed + scheduled). So the mask is
recomputed fresh every step with no read-after-write hazard, and re-widening
(next focus tag closes -> global -> no mask) is exact. KV writes are
unaffected: slot_mapping was materialized upstream from the ORIGINAL row, and
masked-away physical blocks still belong to this request (the scheduler owns
them via block_ids), so their content is never lost — a later step's restore
re-reads them.

Failure policy: any exception anywhere -> log once, disable DA masking for
the process, serve unmasked. A request without a valid flat xarg is never
touched (I1). Batches using cascade attention are skipped (I7: the shared
prefix region bypasses per-row seqused_k).
"""
from __future__ import annotations

import os
import sys

_ENABLED = os.environ.get("VLLM_DA_ATTN", "0") == "1"
_LOGGED: set[str] = set()
_STEP_EVERY = int(os.environ.get("VLLM_DA_LOG_EVERY", "200"))


def _log(msg: str) -> None:
    if msg not in _LOGGED:
        _LOGGED.add(msg)
        sys.stderr.write(f"[da-attn] {msg}\n")
        sys.stderr.flush()


class _DAState:
    def __init__(self):
        self.states: dict = {}       # req_id -> DARequestState
        self.runner = None
        self.tok = None
        self.block_size = 16
        self.fa_gid: int | None = None   # kv-cache group index of full attention
        self.dead = False
        self.masked_rows = 0
        self.attended_kept = 0
        self.attended_total = 0
        self.steps = 0
        self.engaged = 0

    def disable_all(self, why: str) -> None:
        if not self.dead:
            self.dead = True
            _log(f"DA masking disabled for this process: {why}")


_DA = _DAState()


def install_da_hook() -> None:
    """Wrap GPUModelRunner._build_attention_metadata with the DA mask pass."""
    if not _ENABLED:
        return
    if getattr(install_da_hook, "_done", False):
        return

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_radiance_da_wrapped", False):
        install_da_hook._done = True
        return

    _orig_build = GPUModelRunner._build_attention_metadata

    def _build(self, *args, **kwargs):
        out = _orig_build(self, *args, **kwargs)
        if not _DA.dead:
            try:
                _da_pass(self, out)
            except Exception as e:  # never break serving because of DA
                import traceback
                _log(f"DA pass failed ({e!r}); unmasked\n"
                     + traceback.format_exc(limit=8))
                _DA.disable_all("pass-exception")
        return out

    GPUModelRunner._build_attention_metadata = _build
    GPUModelRunner._radiance_da_wrapped = True
    install_da_hook._done = True
    _log("installed on GPUModelRunner._build_attention_metadata")


def _init(runner) -> None:
    """One-time discovery: full-attention group, block size, tokenizer."""
    _DA.runner = runner
    try:
        from vllm.v1.kv_cache_interface import FullAttentionSpec
        for gid, g in enumerate(runner.kv_cache_config.kv_cache_groups):
            if isinstance(g.kv_cache_spec, FullAttentionSpec):
                _DA.fa_gid = gid
                _DA.block_size = int(g.kv_cache_spec.block_size)
                break
        if _DA.fa_gid is None:
            _DA.disable_all("no full-attention kv group")
            return
        # kernel-block != allocator-block remaps ids per row; unsupported here
        bt = runner.input_batch.block_table[_DA.fa_gid]
        if getattr(bt, "use_hybrid_blocks", False):
            _DA.disable_all("hybrid kernel/allocator block sizes")
            return
    except Exception as e:
        _DA.disable_all(f"group discovery failed: {e!r}")
        return

    tok = getattr(runner, "tokenizer", None)
    if tok is None:
        try:
            from vllm.tokenizers import get_tokenizer
            mc = runner.model_config
            tok = get_tokenizer(
                mc.tokenizer, tokenizer_mode=mc.tokenizer_mode,
                trust_remote_code=mc.trust_remote_code,
                revision=mc.tokenizer_revision,
            )
        except Exception as e:
            _DA.disable_all(f"tokenizer unavailable: {e!r}")
            return
    _DA.tok = tok
    _log(f"ready: fa_gid={_DA.fa_gid} block_size={_DA.block_size}")


def _query_len(runner, i: int) -> int:
    """Scheduled query tokens for row i (includes spec-decode drafts; the
    response span is always kept, so drafts need no special handling)."""
    try:
        return int(runner.num_scheduled_tokens.np[i])
    except Exception:
        return 1


def _da_pass(runner, build_out) -> None:
    import torch

    if _DA.tok is None:
        _init(runner)
        if _DA.dead:
            return

    if torch.cuda.is_current_stream_capturing():
        return  # capture step: replay reads live buffer contents per step — fine

    attn_metadata = build_out[0] if isinstance(build_out, tuple) else build_out
    if not isinstance(attn_metadata, dict):
        return  # pooling / unexpected shape: hands off

    # I7: cascade batches read a shared prefix outside per-row seqused_k
    for md in attn_metadata.values():
        if getattr(md, "use_cascade", False):
            return

    states = _DA.states
    b = _DA.block_size
    gid = _DA.fa_gid
    num_reqs = runner.input_batch.num_reqs
    req_ids = list(runner.input_batch.req_ids[:num_reqs])

    # sweep finished requests cheaply
    if len(states) > 4 * max(num_reqs, 1):
        live = set(req_ids)
        for rid in [r for r in states if r not in live]:
            states.pop(rid, None)

    from da_state_machine import DARequestState, kept_tokens_and_blocks, parse_da_xarg

    row_updates = []
    for i, req_id in enumerate(req_ids):
        req_state = runner.requests.get(req_id)
        if req_state is None:
            continue
        st = states.get(req_id)
        if st is None:
            sp = req_state.sampling_params
            spec = parse_da_xarg(getattr(sp, "extra_args", None)) if sp else None
            if spec is None:
                continue
            st = DARequestState(spec, req_state.num_prompt_tokens)
            states[req_id] = st
            _DA.engaged += 1
            _log(f"req {req_id[:24]}: DA engaged ({st.n_segs} segments)")

        resp = req_state.output_token_ids
        # I2: never mask prefill / chunked-prefill rows
        if not resp or req_state.num_computed_tokens < req_state.num_prompt_tokens:
            continue
        st.advance(_DA.tok, resp)
        if not st.should_mask():
            continue

        prompt_len = req_state.num_prompt_tokens
        qlen = _query_len(runner, i)
        total = prompt_len + len(resp)
        resp_start = prompt_len + max(0, len(resp) - qlen)
        # +b slack covers optimistic spec-decode drift in output_token_ids
        spans = st.kept_token_spans() + [(resp_start, total + b)]
        blocks, _ = kept_tokens_and_blocks(spans, total + b, b)
        true_ids = req_state.block_ids[gid]
        # Derive n_valid from the token count, NOT len(true_ids): under hybrid
        # align-mode the id list can carry padding beyond the request's real
        # blocks, which would make last_valid hugely negative (int32 overflow).
        n_valid = min(len(true_ids), (total + b - 1) // b)
        if n_valid <= 0:
            continue
        blocks = [bl for bl in blocks if bl < n_valid]
        if not blocks or len(blocks) >= n_valid or blocks[0] != 0:
            continue  # no-op or unexpected (sink must anchor block 0)

        kept_ids = [int(true_ids[bl]) for bl in blocks]
        # seqused_k: just past the last REAL token in the compacted row. The
        # last kept block is always the response's last block (the response
        # span reaches `total`), whose valid content is total-(n_valid-1)*b;
        # earlier kept blocks are full. Edge leak <= b-1 tokens per span edge
        # is the paper's block alignment (App. B).
        last_valid = total - (n_valid - 1) * b
        seqused = (len(blocks) - 1) * b + max(1, min(last_valid, b))
        if not (0 < seqused <= 2**31 - 1):
            continue
        row_updates.append((i, kept_ids, seqused))
        _DA.attended_kept += seqused
        _DA.attended_total += total

    if not row_updates:
        return

    block_table_gpu = runner.input_batch.block_table[gid].block_table.gpu
    seq_lens = runner.seq_lens
    for i, kept_ids, seqused in row_updates:
        n = len(kept_ids)
        block_table_gpu[i, :n] = torch.tensor(
            kept_ids, dtype=block_table_gpu.dtype, device=block_table_gpu.device
        )
        block_table_gpu[i, n:] = kept_ids[-1]  # stale fill, never read (seqused_k)
        seq_lens[i] = seqused
        _DA.masked_rows += 1

    _DA.steps += 1
    if _DA.steps % _STEP_EVERY == 0:
        pct = (1 - _DA.attended_kept / max(_DA.attended_total, 1)) * 100
        sys.stderr.write(
            f"[da-attn] steps={_DA.steps} masked_rows={_DA.masked_rows} "
            f"engaged={_DA.engaged} attended_reduction={pct:.1f}%\n"
        )
        sys.stderr.flush()
        _DA.attended_kept = 0
        _DA.attended_total = 0

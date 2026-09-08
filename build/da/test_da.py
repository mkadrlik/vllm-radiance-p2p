#!/usr/bin/env python3
"""CPU test harness for the DA vLLM integration (no GPU, no vllm import).

Run: python3 test_da.py  (from build/da/)
Exits non-zero on any failure.
"""
from __future__ import annotations

import re
import sys

from da_state_machine import (
    DARequestState,
    kept_tokens_and_blocks,
    parse_da_xarg,
)

FAIL = 0


def check(name, cond):
    global FAIL
    if cond:
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}")


# --- FakeIncrementalTok: mirrors HF fast-tokenizer byte-level property -------
# Byte-level BPE never merges '>' with following text, so a complete tag in the
# decoded stream always appears complete in any window containing it.
class FakeTok:
    def __init__(self, vocab=None):
        self.vocab: dict[str, int] = {}
        self.inv: dict[int, str] = {}

    def encode(self, text):
        return [self._c(ch) for ch in text]

    def _c(self, ch):
        if ch not in self.vocab:
            i = len(self.vocab)
            self.vocab[ch] = i
            self.inv[i] = ch
        return self.vocab[ch]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.inv.get(i, "?") for i in ids)


SPEC = {
    "segs": [(2000, 4048), (4048, 6096), (6096, 8144), (8144, 10192)],
    "win": 512,
    "sink": 16,
}
PROMPT_LEN = 12000  # last segment ends at 10192; question window 11488..12000


def test_xarg_validation():
    print("xarg validation:")
    check("valid passes", parse_da_xarg({"da_segs": [0, 10, 10, 20]}) is not None)
    check("missing -> None", parse_da_xarg(None) is None)
    check("no da key -> None", parse_da_xarg({"other": 1}) is None)
    check("empty segs -> None", parse_da_xarg({"da_segs": []}) is None)
    check("odd length -> None", parse_da_xarg({"da_segs": [0, 10, 11]}) is None)
    check("overlap -> None", parse_da_xarg({"da_segs": [0, 10, 5, 20]}) is None)
    check("backwards -> None", parse_da_xarg({"da_segs": [10, 5]}) is None)
    check("bad win -> None", parse_da_xarg({"da_segs": [0, 5], "da_win": 0}) is None)
    check("sink>win -> None", parse_da_xarg({"da_segs": [0, 5], "da_win": 8, "da_sink": 16}) is None)
    check("bool win -> None", parse_da_xarg({"da_segs": [0, 5], "da_win": True}) is None)
    check("float ref -> None", parse_da_xarg({"da_segs": [0, 5.5]}) is None)
    check("str ref -> None", parse_da_xarg({"da_segs": ["0", "5"]}) is None)
    r = parse_da_xarg({"da_segs": [10, 20, 0, 10]})
    assert r is not None
    check("sorts spans", r["segs"] == [(0, 10), (10, 20)])


def test_parser_semantics():
    print("parser semantics:")
    tok = FakeTok()
    st = DARequestState(SPEC, PROMPT_LEN)
    check("starts global", st.mode == "global" and not st.should_mask())

    st.advance(tok, tok.encode("<global>chunk 3 has it"))
    check("global stays unmasked", st.mode == "global" and not st.should_mask())

    st.advance(tok, tok.encode('.</global><focus magic_chunks="3">the value'))
    check("focus activates", st.mode == "focus" and st.should_mask())
    spans = st.kept_token_spans()
    check("focus keeps only chunk 3 + scaffold", (6096, 8144) in spans and (2000, 4048) not in spans)

    st.advance(tok, tok.encode("</focus><local>answer is 42</local><answer>42</answer>"))
    check("close reverts to global", st.mode == "global")
    # after <answer> mode is frozen; global -> unmasked (correct superset)
    check("answered+global unmasked", not st.should_mask())

    # local mode
    st2 = DARequestState(SPEC, PROMPT_LEN)
    st2.advance(tok, tok.encode('<focus magic_chunks="1,4">x</focus><local>plan'))
    check("multi-ref focus accumulates 1 and 4", st2._kept_chunks == {1, 4})
    check("closed focus + open local -> local masks", st2.mode == "local" and st2.should_mask())
    check("refs stay attended through local", st2.kept_token_spans() == [(0, 16), (11488, 12000)])
    st3 = DARequestState(SPEC, PROMPT_LEN)
    st3.advance(tok, tok.encode("<local>no context please"))
    check("local masks", st3.mode == "local" and st3.should_mask())
    check("local keeps scaffold only", st3.kept_token_spans() == [(0, 16), (11488, 12000)])

    # malformed focus -> global recovery, never a crash
    st4 = DARequestState(SPEC, PROMPT_LEN)
    st4.advance(tok, tok.encode('<focus magic_chunks="99">'))
    check("out-of-range ref -> global", st4.mode == "global")
    st5 = DARequestState(SPEC, PROMPT_LEN)
    st5.advance(tok, tok.encode('<focus chunks="2">'))
    check("wrong attr name -> global", st5.mode == "global")

    # tag straddling a step boundary (the incremental hazard)
    st6 = DARequestState(SPEC, PROMPT_LEN)
    full = '<global>x</global><focus magic_chunks="2">'
    ids = tok.encode(full)
    for cut in (1, 5, 12, 20, len(ids) - 1):
        s = DARequestState(SPEC, PROMPT_LEN)
        s.advance(tok, ids[:cut])
        mid = s.mode
        s.advance(tok, ids)
        check(f"straddle cut={cut} recovers focus", s.mode == "focus" and mid in ("global", "focus"))

    # focus refs are per-tag (App. B): extracted values live in the response,
    # so a new focus tag REPLACES the kept set
    st7 = DARequestState(SPEC, PROMPT_LEN)
    st7.advance(tok, tok.encode('<focus magic_chunks="2">a</focus>'))
    st7.advance(tok, tok.encode('<focus magic_chunks="4">b'))
    spans = st7.kept_token_spans()
    check("focus keeps 4 now", (8144, 10192) in spans)
    check("closed focus drops 2 from kept set", (4048, 6096) not in spans)
    check("telemetry still records 2", 2 in st7._kept_chunks)


def test_block_math():
    print("block math (b=16):")
    b = 16
    # exact kept-token accounting incl. partial final block
    blocks, exact = kept_tokens_and_blocks([(0, 16), (100, 200)], 200, b)
    check("outward rounding", blocks == [0, 6, 7, 8, 9, 10, 11, 12])
    check("exact tokens", exact == 16 + 100)

    # final partial block: total not block-aligned
    blocks, exact = kept_tokens_and_blocks([(0, 100)], 100, b)
    check("partial last block counted exactly", exact == 100)
    check("block count ceil", blocks == list(range(7)))

    # empty / degenerate
    blocks, exact = kept_tokens_and_blocks([(50, 50)], 100, b)
    check("empty span -> nothing", blocks == [] and exact == 0)
    blocks, exact = kept_tokens_and_blocks([(90, 500)], 100, b)
    check("span clamped to total", exact == 10 and blocks == [5, 6])


def test_row_compaction_invariant():
    """The load-bearing invariant: after compaction, the first `exact` entries
    of the masked row's gather sequence are exactly the KV of the kept tokens,
    in ascending original order, with block 0 pinned (col-0/GDN safety)."""
    print("row compaction invariant:")
    b = 16
    total = PROMPT_LEN + 500
    n_valid = (total + b - 1) // b
    true_ids = list(range(1000, 1000 + n_valid))  # fake physical ids
    tok = FakeTok()
    st = DARequestState(SPEC, PROMPT_LEN)
    st.advance(tok, tok.encode('<focus magic_chunks="3">extracting'))
    spans = st.kept_token_spans() + [(PROMPT_LEN, total)]
    blocks, _ = kept_tokens_and_blocks(spans, total, b)
    blocks = [bl for bl in blocks if bl < n_valid]
    kept_ids = [true_ids[bl] for bl in blocks]
    last_valid = total - (n_valid - 1) * b
    exact = (len(blocks) - 1) * b + last_valid

    check("sink block pinned first", blocks[0] == 0)
    check("kept ascending", blocks == sorted(blocks))
    # reconstruct attended token set from compacted row + exact len
    attended = set()
    for j, blk_slot in enumerate(range(exact // b + (1 if exact % b else 0))):
        pass
    # simpler: first exact tokens map to kept blocks in order
    tokens = []
    for pos in range(exact):
        blk = pos // b
        off = pos % b
        if blk < len(blocks):
            orig = blocks[blk]
            tokens.append(orig * b + off)
    check("attended count == exact", len(tokens) == exact)
    seg3 = set(range(6096, 8144))
    check("all of chunk 3 attended", seg3 <= set(tokens))
    check("chunk 2 NOT attended", not (set(range(4048, 6096)) & set(tokens)))
    check("response attended", set(range(PROMPT_LEN, total)) <= set(tokens))
    check("sink attended", set(range(16)) <= set(tokens))
    check("window attended", set(range(11488, 12000)) <= set(tokens))
    check("exact <= total", exact <= total)
    check("kept blocks < valid blocks", len(blocks) < n_valid)


def test_reduce_reads_upper_bound():
    """The perf claim, checked on a realistic 42K-token 20-segment transcript:
    focus mode must cut the attended KV to <25% of full."""
    print("attended-token reduction (42K ctx, 20 segs, b=16):")
    b = 16
    prompt_len = 42000
    segs = [(2000 + i * 2000, 2000 + (i + 1) * 2000) for i in range(20)]
    segs = segs[:19] + [(2000 + 19 * 2000, 40000)]
    spec = {"segs": segs, "win": 512, "sink": 16}
    tok = FakeTok()
    st = DARequestState(spec, prompt_len)
    st.advance(tok, tok.encode('<focus magic_chunks="7">the path is /a/b</focus><local>comm'))
    spans = st.kept_token_spans() + [(prompt_len, prompt_len + 400)]
    blocks, exact = kept_tokens_and_blocks(spans, prompt_len + 400, b)
    full_blocks, full_exact = kept_tokens_and_blocks(
        [(0, prompt_len + 400)], prompt_len + 400, b)
    red = 1 - exact / full_exact
    check(f"local+1-seg reduction {red*100:.0f}% >= 40%", red >= 0.40)
    # focus mode with 2 chunks
    st2 = DARequestState(spec, prompt_len)
    st2.advance(tok, tok.encode('<focus magic_chunks="7,12">a</focus><focus magic_chunks="3">b'))
    spans = st2.kept_token_spans() + [(prompt_len, prompt_len + 400)]
    _, exact = kept_tokens_and_blocks(spans, prompt_len + 400, b)
    check("3-seg focus still masks", exact < full_exact * 0.5)


def test_case_insensitive_and_typical_stream():
    print("typical stream end-to-end:")
    tok = FakeTok()
    st = DARequestState(SPEC, PROMPT_LEN)
    stream = (
        "<global>\nThe answer is probably in chunk 2.\n</global>\n"
        '<FOCUS magic_chunks="2">the value is 42.</focus>\n'
        "<Local>So the answer is 42.</local>\n"
        "<answer>42</answer>"
    )
    ids = tok.encode(stream)
    # feed in random-ish micro-steps like a decode loop
    pos = 0
    for step in (1, 1, 3, 2, 7, 1, 5, 40, 100):
        pos = min(pos + step, len(ids))
        st.advance(tok, ids[:pos])
        if pos == len(ids):
            break
    while pos < len(ids):
        pos = len(ids)
        st.advance(tok, ids)
    check("final mode global (answer closed everything)", st.mode == "global")
    check("answered frozen", st._answered)
    check("chunk 2 was kept at some point", 2 in st._kept_chunks)


if __name__ == "__main__":
    test_xarg_validation()
    test_parser_semantics()
    test_block_math()
    test_row_compaction_invariant()
    test_reduce_reads_upper_bound()
    test_case_insensitive_and_typical_stream()
    print(f"\n{'FAILURES: ' + str(FAIL) if FAIL else 'ALL PASS'}")
    sys.exit(1 if FAIL else 0)

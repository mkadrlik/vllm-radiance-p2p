#!/usr/bin/env python3
"""DA request state machine for the vLLM engine side (arXiv:2609.02737 App. B).

Incremental, text-level parser over the RESPONSE token ids only — O(response)
per step worst case, never touches the prompt. Same transition semantics as
the client-side parsers (da-eval/state_machine.py, Hermes plugin): mode flips
on an opener tag, reverts to global on the matching closer; a malformed
<focus> recovers as global (never drop the turn).

A tag can straddle a step boundary (partial "<foc"), so each advance()
re-decodes the response tail with a left-context window and commits a tag only
once its closing '>' has been seen. The scan watermark sits at the START of
the last committed tag so a straddling tag is re-seen whole.
"""
from __future__ import annotations

import re
from typing import Any

SINK_TOKENS = 16
DEFAULT_WINDOW = 512
REDECODE_SLACK = 64  # response tokens re-decoded per step (tags are short)

_TAG_RE = re.compile(
    r"<focus\b[^<>]*>|</focus\s*>|<local\s*>|</local\s*>|<global\s*>|<answer\s*>",
    re.IGNORECASE,
)
_CHUNKS_ATTR = re.compile(r'magic_chunks\s*=\s*["\']([^"\']*)["\']')

_MODE_BY_CODE = {0: "global", 1: "focus", 2: "local"}


def parse_da_xarg(extra_args: dict[str, Any] | None) -> dict | None:
    """Extract + validate the DA spec from SamplingParams.extra_args.

    Wire format is flat because vllm_xargs is typed
    dict[str, str|int|float|list[str|int|float]] — nested dicts fail pydantic
    (verified against the served 0.26.0):

        vllm_xargs = {"da_segs": [s0,e0, s1,e1, ...], "da_win": 512,
                      "da_sink": 16, "da_mode": 1, "da_refs": [7]}

    da_mode/da_refs are OPTIONAL (default 0/empty): the initial mask state at
    the first response token. Needed when the client prefills a DA opener as a
    guided continuation — the server parser only sees the response, so the
    opener's mode must be declared. 0=global 1=focus 2=local.

    Returns {"segs": [(start,end)...], "win": int, "sink": int,
             "init_mode": str, "init_refs": [int]} or None.
    Validation is strict: a malformed spec means "not a DA request" (I1 —
    zero behavior change for every other workload).
    """
    if not extra_args or "da_segs" not in extra_args:
        return None
    raw = extra_args.get("da_segs")
    if not isinstance(raw, list) or len(raw) < 2 or len(raw) % 2:
        return None
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in raw):
        return None
    norm = [(raw[i], raw[i + 1]) for i in range(0, len(raw), 2)]
    if any(a < 0 or a >= b for a, b in norm):
        return None
    norm.sort()
    for (_, a1), (b0, _) in zip(norm, norm[1:]):
        if b0 < a1:
            return None  # overlapping spans: malformed spec
    win = extra_args.get("da_win", DEFAULT_WINDOW)
    sink = extra_args.get("da_sink", SINK_TOKENS)
    if (
        not isinstance(win, int) or isinstance(win, bool)
        or not isinstance(sink, int) or isinstance(sink, bool)
        or win <= 0 or not 0 <= sink <= win
    ):
        return None
    mode_code = extra_args.get("da_mode", 0)
    refs = extra_args.get("da_refs", [])
    if not isinstance(refs, list) or not all(
        isinstance(v, int) and not isinstance(v, bool) for v in refs
    ):
        return None
    init_mode = _MODE_BY_CODE.get(mode_code)
    if init_mode is None:
        return None
    return {"segs": norm, "win": win, "sink": sink,
            "init_mode": init_mode, "init_refs": list(refs)}


def kept_tokens_and_blocks(spans: list[tuple[int, int]],
                           total_tokens: int,
                           b: int) -> tuple[list[int], int]:
    """Token spans -> (sorted kept block indices, exact kept token count).

    Blocks are rounded OUTWARD (App. B): an edge block counts only the tokens
    inside the kept spans, so the seqused_k we write is the EXACT attended
    length of the compacted row — no stale KV ever enters the read set.
    The final (response) block is partial in reality: its valid token count is
    total_tokens - blk*b, clamped to the block's intersection with kept spans.
    """
    per_block: dict[int, int] = {}
    for s, e in spans:
        s = max(0, s)
        e = min(e, total_tokens)
        if s >= e:
            continue
        for blk in range(s // b, -(-e // b)):
            lo = max(s, blk * b)
            hi = min(e, (blk + 1) * b)
            if hi > lo:
                per_block[blk] = per_block.get(blk, 0) + (hi - lo)
    blocks = sorted(per_block)
    return blocks, sum(per_block.values())


class DARequestState:
    """Per-request DA mask state.

    advance(tok, response_ids) every step; should_mask() gates the row
    rewrite; kept_token_spans() answers what stays attended (response span
    appended by the caller — it grows every step).
    """

    def __init__(self, spec: dict, prompt_len: int):
        self.segs: list[tuple[int, int]] = spec["segs"]
        self.win: int = spec["win"]
        self.sink: int = spec["sink"]
        self.prompt_len = prompt_len
        self.n_segs = len(self.segs)
        self.mode = "global"
        self._active_refs: list[int] = []    # refs of the currently-open focus
        self._kept_chunks: set[int] = set()  # ever validly opened (telemetry)
        self._answered = False
        self._scan_upto = 0                  # response tokens fully parsed
        init_mode = spec.get("init_mode", "global")
        init_refs = [k for k in spec.get("init_refs", []) if 1 <= k <= self.n_segs]
        if init_mode == "focus" and init_refs:
            self.mode = "focus"
            self._active_refs = init_refs
            self._kept_chunks.update(init_refs)
        elif init_mode == "local":
            self.mode = "local"

    def advance(self, tok, response_ids: list[int]) -> None:
        """Re-scan the response tail; commit only complete tags.

        Async spec-decode leaves -1 placeholders in output_token_ids (later
        corrected); decode runs of real ids only — a placeholder can never be
        part of a tag, so splitting the decode there is safe.
        """
        if self._answered:
            return
        n = len(response_ids)
        if n == 0:
            return
        lo = max(0, min(self._scan_upto, n) - REDECODE_SLACK)
        if lo >= n and self._scan_upto >= n:
            return
        ids = response_ids[lo:]
        parts: list[str] = []
        run: list[int] = []
        for t in ids:
            if isinstance(t, int) and t >= 0:
                run.append(t)
            else:
                if run:
                    parts.append(tok.decode(run, skip_special_tokens=False))
                    run = []
                parts.append(" ")
        if run:
            parts.append(tok.decode(run, skip_special_tokens=False))
        text = "".join(parts)
        watermark = self._scan_upto
        for m in _TAG_RE.finditer(text):
            self._handle(m.group(0))
            cand = m.start() + lo
            if cand > watermark:
                watermark = cand
        self._scan_upto = min(max(watermark, self._scan_upto), n)

    def _handle(self, tag: str) -> None:
        low = tag.lower()
        if low.startswith("<focus"):
            am = _CHUNKS_ATTR.search(tag)
            refs: list[int] = []
            if am:
                for part in am.group(1).split(","):
                    part = part.strip()
                    if part.isdigit():
                        k = int(part)
                        if 1 <= k <= self.n_segs:
                            refs.append(k)
            if refs:
                # App. B: focus keeps the *named* segments of this tag. Values
                # extracted from earlier focus blocks live in the response
                # (always attended), so refs do NOT accumulate.
                self.mode = "focus"
                self._active_refs = refs
                self._kept_chunks.update(refs)
            else:
                self.mode = "global"  # malformed -> tolerant recovery
                self._active_refs = []
        elif low.startswith("</focus") or low.startswith("</local"):
            self.mode = "global"
            self._active_refs = []
        elif low.startswith("<local"):
            self.mode = "local"
            self._active_refs = []
        elif low.startswith("<global"):
            self.mode = "global"
            self._active_refs = []
        elif low.startswith("<answer"):
            self._answered = True  # mode frozen; answer region is response-only

    def kept_token_spans(self) -> list[tuple[int, int]]:
        """Context token spans kept attended in the CURRENT mode (excludes the
        response span; caller appends it).
        Invariant I3: sink + question window always present."""
        scaffold = [
            (0, self.sink),
            (max(self.sink, self.prompt_len - self.win), self.prompt_len),
        ]
        if self.mode == "focus":
            return scaffold + [self.segs[k - 1] for k in sorted(set(self._active_refs))]
        if self.mode == "local":
            return scaffold
        return scaffold + list(self.segs)

    def should_mask(self) -> bool:
        """global keeps everything — the rewrite would be a no-op; skip it."""
        return self.mode in ("focus", "local")

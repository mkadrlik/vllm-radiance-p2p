# Qwen3.8-27B on gfx1100 TP2: W8A8-INT8 vs AWQ-INT4 — measured 2026-09-05/06

TL;DR: **AWQ is the right config, but not for the reason expected.** Swapping
`Avesed/Qwen3.8-27B-INT8-W8A8` → `cyankiwi/Qwen3.8-27B-AWQ-INT4` did NOT
double single-stream decode (~20 → ~24-29 t/s, +18-25%), because single-stream
on TP2 is bound by the all-reduce path, not weight bandwidth. It DID nearly
double batch throughput and grew the KV pool +80%.

## Why quant choice barely moves single-stream

Per decode step at batch 1, the 27B (hybrid GDN, dense 27B params) reads:

| | W8A8 (39.4 GB file) | AWQ-INT4 (21.0 GB file) |
|---|---|---|
| weights/step/card (TP2) | ~19.7 GB | ~10.5 GB |
| expected at ~1.5 TB/s effective | ~13 ms | ~7 ms |
| **measured step time** | **~49 ms** (20.4 t/s × MTP 2.4) | **~92 ms raw / ~131 ms pre-MTP-fix** (24-29 t/s × MTP 2.3) |

The floor is TP2 all-reduce + launch overhead: ~128 RCCL hops/token
(`disable_custom_all_reduce=True` — the custom AR livelocks on consumer PCIe,
see the RDNA3 one-shot AR investigation, verdict: NOT VIABLE on gfx1100;
`GPU_MAX_HW_QUEUES=1`). Weight streaming is a minority of the step, so
halving it recovers only the residual: ~20%.

The same model family on **one card with zero all-reduce** (Ornith-1.5-9B
AWQ, `ornith-9b` profile) runs at **80+ t/s single-stream** — that's the TP
tax, and it dwarfs any quant effect. If single-stream latency is the problem,
remove the hop (smaller model, one card), don't re-quant the big one.

## What AWQ actually bought (measured, same-protocol batch curves)

Protocol: `/v1/chat/completions`, 256 tok, temp 0, `enable_thinking:false`,
parallel clients released together; per-stream = tokens/(t_last − t_first).
Engine was quiet (27B is the only prod worker; verify `num_requests_running`
before trusting batch=1 numbers — a contended engine read 4.0 t/s).

| batch | W8A8 | AWQ |
|---|---|---|
| 1 | 20.4 t/s (radiance A/B baseline) | 23.8–28.9 t/s |
| 8 | — | 12.2/stream, ~97 agg |
| 16 | — | 11.1/stream, ~174 agg |
| 32 | — | 10.3/stream, **~382 agg** |
| 64 | — | 9.4/stream, ~600 agg (TTFT 12 s, queue-bound) |

- **KV pool: 90K → 163,840 tokens** (weights 39 GB → 21 GB at 0.85 util).
  Two full 65K agent sessions or a dozen 20K ones resident; prefix cache hit
  rate 84-87% on real agent traffic (shared ~48K tool schema prefix).
- MTP (`qwen3_5_mtp`, depth 2) survives the swap: the AWQ rep ships MTP
  tensors; acceptance 65-74%, mean length 2.30 — engine-verified under load.
  Depth 4 is −9% (radiance A/B); keep 2.
- `min_p`/`logprods` caveats: vLLM warns `min_p and logit_bias won't work
  with speculative decoding` — acceptable for agent traffic.

## Config deltas (W8A8 → AWQ) — the whole swap was this

1. `--model Avesed/Qwen3.8-27B-INT8-W8A8` → `cyankiwi/Qwen3.8-27B-AWQ-INT4`
   (both `--model` and inside `--speculative-config`; quant flag
   `compressed-tensors` is UNCHANGED — both reps use it: W8A8 = weight+act
   int8, AWQ = pack-quantized W4 g32)
2. `--default-chat-template-kwargs '{"enable_thinking": false}'` — added.
   Required because the ContextForge-style gateways strip per-request
   `chat_template_kwargs`, so thinking-off must be a server default or every
   aux call silently pays a hidden reasoning block (observed: 50 completion
   tokens for a 4-token answer, `\n\n` residue in `content`).
3. Everything else identical (TP2, ROCM_ATTN, 0.85 util, graphs
   FULL_AND_PIECEWISE, NCCL env, shm 16G).

Rollback: the W8A8 compose is preserved in
`inference-engines/qwen/qwen3.8-27b/vllm-radiance/archive/docker-compose-27b.yml.bak-w8a8`.

## Benchmarking traps found the hard way

- **Thinking-on benches lie.** A bench that counts only `content` deltas with
  the reasoner enabled measured 4.4 t/s "decode" while the engine generated
  17.9 t/s total. Count `content + reasoning_content`, or disable thinking.
- **Idle ≠ idle.** `rocm-smi --showuse 99%` with no visible process was the
  27B serving another agent's stream. Check `vllm:num_requests_running` in
  `/metrics` before and during any measurement.
- Container logs for engine-core crashes can be empty (silent exit, restart
  loop). Reproduce with `docker run --rm -e PYTHONFAULTHANDLER=1` and
  file-redirected output to get the traceback.

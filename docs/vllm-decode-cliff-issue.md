# [Bug]: Decode throughput cliff at ≥5 concurrent streams — gfx1100 (RDNA3), TP1, hybrid GDN model (vLLM 0.26.0)

**Component:** v1 engine / gpu_model_runner sampling path
**Severity:** high (aggregate throughput *drops* when adding concurrency; ~4× loss)

## Summary

On a single RX 7900 XTX (gfx1100, ROCm 7.14, PyTorch 2.11+rocm), serving a quantized Qwen3.5-family hybrid GDN model (linear-attention + full-attention), decode throughput collapses uniformly across ALL streams when the number of simultaneously-decoding requests exceeds 4:

| concurrent decode streams | per-stream decode | aggregate decode |
|---|---|---|
| 1 | 81 t/s | 81 t/s |
| 4 | 49.5 t/s | 197 t/s |
| **5** | **13.7 t/s** | **69 t/s** |
| 8 | 13.2 t/s | 106 t/s |
| 16 | 12.4 t/s | 199 t/s |
| 24 | 12.4 t/s | 317 t/s |

Adding the 5th concurrent stream makes the engine ~3× *slower in aggregate*. Step time plateaus at ~73–76 ms and becomes nearly batch-size-independent (measured identically at 5, 8, 16, 24 streams), which is not explicable by weight or KV bandwidth.

## Key observations

1. **The plateau is a hard mode-switch, not a gradual bandwidth wall.** Staggered arrivals that keep simultaneous-decode count ≤4 run at full speed even at batch 8–16 offered (per-stream 48–75 t/s). Once ≥5 streams co-reside, *every* stream drops to 12–13 t/s.
2. **py-spy during the plateau** shows EngineCore MainThread parked in
   `gpu_model_runner.py` `_bookkeeping_sync → _to_list` (line ~7783, the `transfer_event.synchronize()` + pinned-buffer `tolist()` path) on essentially every step.
3. **Not CUDA-graph related:** `--enforce-eager` reproduces the same plateau (~76 ms/step); it only shifts the boundary from 5 to 6 streams (graphs pad batch 5→8 capture size).
4. **Not async-scheduling related (with a caveat):** `--no-async-scheduling` (sync path, where the `_to_list` bookkeeping branch lives) and async scheduling **both** plateau at ~13.5 t/s/stream at batch 5+. Async slightly *worsens* batch-1 (86→81 with graphs) and does not remove the cliff. So the fixed ~73 ms cost is not solely the sync `_to_list` — but the stack sample consistently lands there.
5. **Not prefix caching / Mamba align mode:** cliff identical with `--no-enable-prefix-caching` (and `--mamba-cache-mode none` is silently overridden to `align` when prefix caching is on, so we tested with caching fully off).
6. **Not thermal/power:** junction 42 °C during plateau; clocks at full; no other processes on the GPU; second GPU pair on the same host provably idle.
7. **Not host CPU contention:** 24c/48t Threadripper, single Python process pinned path.
8. **Same model family on TP2 (2× gfx1100, RCCL all-reduce, ~131 ms/step baseline) shows NO cliff** — smooth curve from batch 1→64 (per-stream 24→9.4, aggregate rising monotonically to ~600 t/s). So this appears specific to the **TP1/uniproc_executor** decode path.
9. Deterministic to 0.1 t/s across repeats and container restarts.

## Environment

- vLLM 0.26.0 (V1 engine), ROCm 7.14.1, PyTorch 2.11+rocm, gfx1100 (RX 7900 XTX 24 GB), `HSA_OVERRIDE_GFX_VERSION=11.0.0`
- Model: `cyankiwi/Ornith-1.5-9B-AWQ-INT4` (`Qwen3_5ForConditionalGeneration`, 24/32 linear GDN layers, 8 full-attention), `--quantization compressed-tensors` (W4A16 pack-quantized g32)
- Attention: `--attention-backend ROCM_ATTN` (logs expected `Cannot use ROCm custom paged attention kernel, falling back to Triton` on gfx1100)
- `--max-num-batched-tokens 8192`, `--gpu-memory-utilization 0.90`, `--max-model-len 262144`, CUDA graphs FULL_AND_PIECEWISE (and independently `--enforce-eager`)
- Repro also expected with any Qwen3.5/3.6 hybrid (Qwen3Next-style) on consumer AMD; we have not tested NVIDIA.

## Reproduction script

```python
#!/usr/bin/env python3
# batch_cliff.py — run against a TP1 gfx1100 server, then: python3 batch_cliff.py 4 5 8
import json, time, urllib.request, threading, sys
URL = "http://HOST:PORT/v1/chat/completions"; MODEL = "MODEL_NAME"
P = ("Explain how tensor parallelism works for transformer inference on dual-GPU "
     "PCIe systems: sharding strategies, communication patterns, latency tradeoffs. ")
def w(i, res, gate):
    b = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": P + f"(v{i})"}],
        "max_tokens": 256, "temperature": 0, "stream": True,
        "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(URL, data=b, headers={"Content-Type": "application/json"})
    while not gate.is_set(): time.sleep(0.005)
    t0 = time.perf_counter(); ttft = None; n = 0
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            if not line.startswith(b"data: "): continue
            s = line[6:].strip()
            if s == b"[DONE]": break
            c = json.loads(s)["choices"][0]["delta"].get("content")
            if c:
                if ttft is None: ttft = time.perf_counter() - t0
                n += 1
    res[i] = (ttft, n, time.perf_counter() - t0)
for batch in [int(x) for x in sys.argv[1:]]:
    res = {}; gate = threading.Event()
    ts = [threading.Thread(target=w, args=(i, res, gate)) for i in range(batch)]
    [t.start() for t in ts]; gate.set(); [t.join() for t in ts]
    per = [v[1] / (v[2] - v[0]) for v in res.values()]
    agg = sum(v[1] for v in res.values()) / max(v[2] - v[0] for v in res.values())
    print(f"batch={batch:3d} per-stream med={sorted(per)[len(per)//2]:5.1f} t/s  aggregate={agg:6.1f} t/s")
    time.sleep(3)
```

Expected (broken): batch 4 ≈ 50/197, batch 5 ≈ 13.7/69. Expected (fixed): aggregate monotonic, per-stream graceful.

## Asks

1. Is the ~73 ms fixed cost at ≥5 resident sequences a known ROCm/Triton kernel issue (e.g. GDN `chunk_gated_delta_rule` autotune falling back to a batch-capacity-dependent tile, or the paged-attention Triton fallback recompiling per batch bucket)?
2. `_to_list` already notes it exists to avoid a broader stream-sync problem (issue #22754). Should the ≥5-stream case avoid `transfer_event.synchronize()` per step entirely?
3. Happy to run instrumented builds — the host is a 3×7900XTX box and we can reproduce on demand.

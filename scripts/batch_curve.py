#!/usr/bin/env python3
"""Batch-scaling curve bench for any OpenAI-compatible vLLM endpoint.

Usage:
    python3 batch_curve.py <base_url> <model> [batch sizes...]
    python3 batch_curve.py http://localhost:13305/v1/chat/completions Qwen3.8-27B 1 4 8 16 32

Counts content + reasoning_content deltas (thinking tokens included) — a bench
that counts only `content` against a reasoning model under-reports badly.
Reports per-stream median and aggregate decode t/s, and TTFT p50.
Check `vllm:num_requests_running` on the target first: a contended engine
produces garbage numbers.
"""
import json, time, urllib.request, threading, sys

def worker(url, model, maxtok, results, idx, gate):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content":
            "Explain how tensor parallelism works for transformer inference on "
            "dual-GPU PCIe systems: sharding strategies, communication patterns, "
            f"latency tradeoffs. Be thorough. (variant {idx})"}],
        "max_tokens": maxtok, "temperature": 0.0, "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    while not gate.is_set():
        time.sleep(0.005)
    t0 = time.perf_counter(); ttft = None; n = 0
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            for line in r:
                if not line.startswith(b"data: "): continue
                s = line[6:].strip()
                if s == b"[DONE]": break
                d = json.loads(s)
                delta = d["choices"][0]["delta"]
                tok = delta.get("content") or delta.get("reasoning_content")
                if tok:
                    if ttft is None: ttft = time.perf_counter() - t0
                    n += 1
    except Exception as e:
        print(f"  req {idx} error: {e}", file=sys.stderr); return
    results[idx] = (ttft or 0, n, time.perf_counter() - t0)

def run(url, model, batch, maxtok=256):
    results = {}
    gate = threading.Event()
    threads = [threading.Thread(target=worker, args=(url, model, maxtok, results, i, gate))
               for i in range(batch)]
    for t in threads: t.start()
    gate.set()
    for t in threads: t.join()
    if not results: return
    ttfts = [v[0] for v in results.values()]
    per = [v[1] / (v[2] - v[0]) for v in results.values() if v[2] - v[0] > 1]
    if not per: return
    per.sort()
    agg = sum(v[1] for v in results.values()) / max(v[2] - v[0] for v in results.values())
    print(f"batch={batch:3d}  TTFT p50={sorted(ttfts)[len(ttfts)//2]*1000:6.0f}ms  "
          f"per-stream med={per[len(per)//2]:5.1f} t/s  aggregate={agg:6.1f} t/s",
          flush=True)

if __name__ == "__main__":
    url, model = sys.argv[1], sys.argv[2]
    for b in [int(x) for x in (sys.argv[3:] or [1, 4, 8, 16, 32])]:
        run(url, model, b)
        time.sleep(3)

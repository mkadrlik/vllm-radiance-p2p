#!/usr/bin/env python3
"""Quick inference smoke test for vllm-rocm-tq.

Usage:
    python3 scripts/test-inference.py [--port PORT] [--model NAME]
"""
import argparse
import json
import sys
import urllib.request

def test_inference(port: int, model: str) -> bool:
    """Send a chat completion request and verify the response."""
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 10,
        "temperature": 0,
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            content = data["choices"][0]["message"]["content"]
            print(f"✅ Inference OK: {content!r}")
            return True
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return False

def test_health(port: int) -> bool:
    """Check server health endpoint."""
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=13309)
    parser.add_argument("--model", default="validation")
    args = parser.parse_args()

    if not test_health(args.port):
        print(f"❌ Server not healthy on port {args.port}")
        sys.exit(1)
    print(f"✅ Server healthy on port {args.port}")

    if not test_inference(args.port, args.model):
        sys.exit(1)
#!/bin/bash
# Radiance deploy completion notice (Discord). One-shot.
echo "✅ vllm-radiance deployment COMPLETE — Qwen3.6-27B-Quark-W8A8 (TP2) live on :13313
• Decode: 21.9 tok/s (CUDA-graph mode, +33% vs lemonade 16.5 baseline; 2.1x eager)
• Stack: vllm-radiance 0.5.7, AITER GEMM, GDN FLA, stock NCCL (P2P off)
• Stability: bwtest off, fast-reduce off, power cap 300W, language-model-only
• Fixed: SMU-hang reboots (sweep pileup), ViT 128GiB OOM, corrupted triton cache
• Trade: MTP off (5.9GiB graph pools didn't fit 24GiB; 2.8GiB without)"

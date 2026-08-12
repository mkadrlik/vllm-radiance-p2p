###############################################################################
# vllm-awq
# Thin layer on vllm/vllm-openai-rocm:v0.24.0 for AWQ inference on AMD RDNA3.
#
# Eliminates: sitecustomize_vllm.py patches, custom vLLM wheel, Triton/gcc
# workarounds. The official v0.24.0 image has DiffusionGemma + Gemma-4 support,
# ROCm 7.2.3, PyTorch 2.11+HIP — everything we need.
#
# Build:
#   docker build -t ghcr.io/[username]/vllm-awq:latest .
#   # Gitea CI: docker build --build-arg REGISTRY=registry.example.com:5000 .
###############################################################################

ARG REGISTRY=docker.io
FROM ${REGISTRY}/vllm/vllm-openai-rocm:v0.24.0

# Install curl for healthcheck (already present, but explicit)
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

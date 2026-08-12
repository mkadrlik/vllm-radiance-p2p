# vLLM + Radiance TP2 P2P on AMD RDNA3 (gfx1100)
# Uses pre-built stilldeadcode/vllm-radiance image as base

FROM stilldeadcode/vllm-radiance:0.5.7

# Copy application code
COPY . /app

# Set environment variables
ENV ROCM_PATH=/opt/rocm
ENV HIP_PATH=/opt/rocm
ENV LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH:-}

# Expose vLLM API port
EXPOSE 13313

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=1800s \
    CMD curl -sf http://localhost:13313/health || exit 1

# Default command
CMD ["vllm", "serve", "--host", "0.0.0.0", "--port", "13313"]

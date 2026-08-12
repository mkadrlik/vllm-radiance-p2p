# vLLM + Radiance TP2 P2P on AMD RDNA3 (gfx1100)
# Dockerfile for building the vllm-radiance image

FROM rocm/dev-ubuntu-22.04:latest

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV ROCM_PATH=/opt/rocm
ENV HIP_PATH=/opt/rocm
ENV LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# Expose vLLM API port
EXPOSE 13313

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=1800s \
    CMD curl -sf http://localhost:13313/health || exit 1

# Default command
CMD ["vllm", "serve", "--host", "0.0.0.0", "--port", "13313"]

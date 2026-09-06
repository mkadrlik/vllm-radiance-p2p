#!/bin/bash
# Run the Phase 0 bench on GPUs 1,2 (prod pair, co-resident; prod stays up).
# Usage: bash microbench/run_phase0.sh [--quick]
# Hard safety: whole run under timeout -s KILL; every device spin self-aborts <1s.
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMG="${P0_IMAGE:-vllm-radiance-gfx1100:fix-ar-diag}"
docker run --rm --entrypoint bash \
  --device=/dev/kfd --device=/dev/dri \
  --group-add "$(stat -c %g /dev/dri/renderD128)" \
  --group-add "$(stat -c %g /dev/kfd)" \
  --shm-size=16gb \
  -e ROCR_VISIBLE_DEVICES=1,2 \
  -e HIP_VISIBLE_DEVICES=0,1 \
  -e GPU_MAX_HW_QUEUES=1 \
  -v "$REPO/microbench:/mnt/mb" \
  "$IMG" -lc "
set -eu
cd /mnt/mb
timeout -s KILL 900 torchrun --nproc_per_node=2 --master_port=29617 phase0_bench.py $*
"

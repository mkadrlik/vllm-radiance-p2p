#!/bin/bash
# Run Phase 1 prototype validation on GPUs 1,2 (prod pair, co-resident; prod up).
# Hard safety: bounded timeout; every device spin self-poisons <50 ms.
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
  -e NCCL_PROTO=Simple \
  -e NCCL_DEBUG=WARN \
  -v "$REPO/microbench:/mnt/mb" \
  "$IMG" -lc "
set -eu
cd /mnt/mb
timeout -s KILL 1500 torchrun --nproc_per_node=2 --master_port=29619 phase1_proto_test.py $*
"

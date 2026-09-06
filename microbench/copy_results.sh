#!/bin/bash
# Copy the /tmp/phase0_rank*.jsonl from a throwaway container run into the repo.
set -u
cd "$(dirname "$0")"
docker run --entrypoint bash -v "$PWD:/m" vllm-radiance-gfx1100:fix-ar-diag -lc '
cp /tmp/phase0_rank0.jsonl /m/phase0_rank0_full2.jsonl 2>/dev/null && echo copied0
cp /tmp/phase0_rank1.jsonl /m/phase0_rank1_full2.jsonl 2>/dev/null && echo copied1
wc -l /m/phase0_rank*_full2.jsonl 2>/dev/null
' 2>/dev/null | tail -5

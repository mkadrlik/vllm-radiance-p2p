#!/bin/bash
# Build ar_phase0.so for gfx1100 inside the known-good ROCm 7.14 container image.
# Usage: bash microbench/build_phase0.sh   (from repo root, on big-chungus host)
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMG="${P0_IMAGE:-vllm-radiance-gfx1100:fix-ar-diag}"
docker run --rm --entrypoint bash \
  -v "$REPO/microbench:/mnt/mb" \
  "$IMG" -lc '
set -eu
cd /mnt/mb
PY_INC=$(python3 -c "import sysconfig; print(sysconfig.get_path(\"include\"))")
INC=$(python3 -m pybind11 --includes)
SUF=$(python3 -c "import sysconfig; print(sysconfig.get_config_var(\"EXT_SUFFIX\"))")
hipcc -O3 -std=c++17 -fPIC -shared --offload-arch=gfx1100 \
  $INC -I"$PY_INC" \
  ar_phase0.hip -o "ar_phase0${SUF}" 2>&1 | tee build.log
grep -q . build.log || true
test -f "ar_phase0${SUF}" && echo BUILD_OK || { echo BUILD_FAIL; exit 1; }
'

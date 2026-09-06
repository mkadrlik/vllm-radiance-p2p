#!/bin/bash
# Build ar_proto.so (Phase 1 prototypes) for gfx1100 inside the ROCm 7.14 container.
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMG="${P0_IMAGE:-vllm-radiance-gfx1100:fix-ar-diag}"
docker run --rm --entrypoint bash \
  -v "$REPO/microbench:/mnt/mb" \
  "$IMG" -lc '
set -eu
cd /mnt/mb
INC=$(python3 -m pybind11 --includes)
PY_INC=$(python3 -c "import sysconfig; print(sysconfig.get_path(\"include\"))")
SUF=$(python3 -c "import sysconfig; print(sysconfig.get_config_var(\"EXT_SUFFIX\"))")
hipcc -O3 -std=c++17 -fPIC -shared --offload-arch=gfx1100 \
  $INC -I"$PY_INC" \
  ar_proto.hip -o "ar_proto${SUF}.new" 2>&1 | tee build_proto.log
if grep -q "error:" build_proto.log; then echo BUILD_FAIL; exit 1; fi
mv -f "ar_proto${SUF}.new" "ar_proto${SUF}"
echo BUILD_OK
'

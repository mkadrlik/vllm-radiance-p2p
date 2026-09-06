#!/bin/bash
# Compile-probe2: whole file; clang errors name whichever builtins/asm don't exist.
set -u
cd "$(dirname "$0")"
docker run --rm --entrypoint bash -v "$PWD:/m" vllm-radiance-gfx1100:fix-ar-diag -lc '
cd /m
hipcc -c -O3 -std=c++17 --offload-arch=gfx1100 probe2.hip -o /tmp/probe2.o 2>&1
echo RC=$?
'

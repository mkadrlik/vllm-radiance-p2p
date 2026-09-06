#!/bin/bash
# Compile-probe: whole file; clang errors name whichever builtins/asm don't exist.
set -u
cd "$(dirname "$0")"
docker run --rm --entrypoint bash -v "$PWD:/m" vllm-radiance-gfx1100:fix-ar-diag -lc '
cd /m
hipcc -c -std=c++17 --offload-arch=gfx1100 probe_isa.hip -o /tmp/probe.o 2>&1
echo RC=$?
'

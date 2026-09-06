#!/bin/bash
# Disassemble the pingpong spin loop to find why spins==0.
set -u
cd "$(dirname "$0")"
docker run --rm --entrypoint bash -v "$PWD:/m" vllm-radiance-gfx1100:fix-ar-diag -lc '
cd /m
/opt/rocm/lib/llvm/bin/llvm-objdump --disassemble-symbols=pingpong -d ar_phase0.cpython-312-x86_64-linux-gnu.so > pp.s 2>&1 || /opt/rocm/lib/llvm/bin/llvm-objdump -d ar_phase0.cpython-312-x86_64-linux-gnu.so > pp.s
grep -n "pingpong" pp.s | head -3
wc -l pp.s
'

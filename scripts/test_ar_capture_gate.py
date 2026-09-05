#!/usr/bin/env python3
"""Unit-check the capture-gate in radiance_allreduce.should_custom_ar (no dist needed)."""
import types

import torch

import radiance_allreduce as R


class Fake:
    disabled = False
    max_bytes = 32768 * 1024


Fake.should_custom_ar = types.MethodType(R.RadianceAllreduce.should_custom_ar, Fake)

x = torch.zeros(1024, dtype=torch.bfloat16)
ok = True

def check(name, got, want):
    global ok
    print(f"{'PASS' if got == want else 'FAIL'}  {name}: got {got} want {want}")
    ok = ok and (got == want)

check("eager gate (must be False)", Fake.should_custom_ar(x), False)

_real = torch.cuda.is_current_stream_capturing
torch.cuda.is_current_stream_capturing = lambda: True
try:
    check("capturing gate (expect True)", Fake.should_custom_ar(x), True)
    check("bad dtype (fp64)", Fake.should_custom_ar(torch.zeros(1024, dtype=torch.float64)), False)
    check("unaligned", Fake.should_custom_ar(torch.zeros(2, dtype=torch.bfloat16)), False)
    check("too big", Fake.should_custom_ar(torch.zeros(40 * 1024 * 1024, dtype=torch.bfloat16)), False)
finally:
    torch.cuda.is_current_stream_capturing = _real

check("eager again after restore", Fake.should_custom_ar(x), False)
raise SystemExit(0 if ok else 1)

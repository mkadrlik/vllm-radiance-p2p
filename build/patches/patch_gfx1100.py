#!/usr/bin/env python3
"""gfx1100 (7900 XTX / RDNA3) logic patches for vLLM on ROCm. Idempotent string replacement on the
installed site-packages copies; re-running is safe. Modeled on patch_gfx1201.py (vllm-radiance).

gfx1100-specific differences vs the gfx1201 original:
- AITER is gated on MI3xx upstream; gfx1100 is RDNA3 and needs the Triton (WMMA) path.
- The W8A8 GEMM: vLLM's AiterInt8ScaledMMLinearKernel routes through gemm_a8w8_CK (XDL/CDNA-only).
  On gfx1100 it must use AITER's Triton gemm_a8w8 (non-gluon) -- the aiter#4604 port.
- AITER's compiled sampler kernels and the CK path do not build on RDNA3; the sampler stays native.
"""
import sysconfig
from pathlib import Path
from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])


def main():
    # A0. _get_gcn_arch: honor RADIANCE_GFX_ARCH env (amdsmi reports an empty
    #     target_graphics_version for RDNA on some ROCm levels; the env is deterministic).
    apply(
        SP / "vllm/platforms/rocm.py",
        "    try:\n        return _query_gcn_arch_from_amdsmi()",
        '    import os as _os\n'
        '    _env = _os.environ.get("RADIANCE_GFX_ARCH") or _os.environ.get("VLLM_ROCM_GCN_ARCH")\n'
        '    if _env:\n'
        '        return _env\n'
        "    try:\n        return _query_gcn_arch_from_amdsmi()",
        '_env = _os.environ.get("RADIANCE_GFX_ARCH")',
        "honor RADIANCE_GFX_ARCH env",
    )
    # A. AITER enablement: vLLM gates AITER on MI3xx; gfx1100 (RDNA3) has the
    #    Triton/WMMA implementation and must be allowed too (on_gfx11 covers the gfx1100 family).
    apply(
        SP / "vllm/_aiter_ops.py",
        "        from vllm.platforms.rocm import on_mi3xx\n\n        return on_mi3xx()",
        "        from vllm.platforms.rocm import on_gfx11, on_mi3xx\n\n"
        "        return on_mi3xx() or on_gfx11()",
        "on_mi3xx() or on_gfx11()",
        "is_aiter_found_and_supported: allow gfx11 (gfx1100)",
    )
    # B. Triton HIPDriver.is_active(): stock gates on torch.cuda.is_available(), which is False in
    #    vLLM's GPU-less inspection subprocess where aiter touches the driver at import. A ROCm torch
    #    build always targets HIP, so gate on torch.version.hip.
    apply(
        SP / "triton/backends/amd/driver.py",
        "            return torch.cuda.is_available() and (torch.version.hip is not None)",
        "            return torch.version.hip is not None",
        "            return torch.version.hip is not None",
        "Triton HIPDriver.is_active: gate on torch.version.hip",
    )
    # C. AITER sampler gate: VLLM_ROCM_USE_AITER=1 also selects AITER's top-k/top-p sampler, whose
    #    C++/HIP kernel does not build on RDNA. Gate to MI3xx; gfx1100 uses the native sampler.
    apply(
        SP / "vllm/v1/sample/ops/topk_topp_sampler.py",
        '            logprobs_mode not in ("processed_logits", "processed_logprobs")\n'
        "            and rocm_aiter_ops.is_enabled()\n"
        "        ):",
        '            logprobs_mode not in ("processed_logits", "processed_logprobs")\n'
        "            and rocm_aiter_ops.is_enabled()\n"
        "            # gfx1100: AITER's sampler C++/HIP kernel does not build on RDNA3.\n"
        "            # Gate to MI3xx; gfx1100 uses the native sampler.\n"
        '            and __import__("vllm.platforms.rocm", fromlist=["on_mi3xx"]).on_mi3xx()\n'
        "        ):",
        "AITER's sampler C++/HIP kernel does not build on RDNA3",
        "topk_topp_sampler: gate AITER sampler to MI3xx",
    )
    # D. W8A8 GEMM route: the CK path (gemm_a8w8_CK) is XDL/CDNA-only. On gfx1100
    #    use AITER's Triton gemm_a8w8 (non-gluon), the aiter#4604 port.
    apply(
        SP / "vllm/_aiter_ops.py",
        "    from aiter import gemm_a8w8_CK\n\n"
        "    # gemm_a8w8_CK(a, b, scale_a, scale_b, bias) expects",
        "    from vllm.platforms.rocm import on_gfx11\n"
        "    if on_gfx11():\n"
        "        from aiter.ops.triton.gemm_a8w8 import gemm_a8w8\n"
        "        return gemm_a8w8(A, B, As, Bs, bias, output_dtype)\n"
        "    from aiter import gemm_a8w8_CK\n\n"
        "    # gemm_a8w8_CK(a, b, scale_a, scale_b, bias) expects",
        "if on_gfx11():\n        from aiter.ops.triton.gemm_a8w8 import gemm_a8w8",
        "_rocm_aiter_w8a8_gemm_impl: Triton gemm_a8w8 on gfx11",
    )
    # E. aiter JIT arch allow-list: aiter's cpp_itfs/utils.py gates JIT-compiled kernels to a
    #    known arch list. gfx1100 support landed around 0.1.16/0.1.17; patch it in if absent.
    aiter_utils = SP / "aiter_meta/csrc/cpp_itfs/utils.py"
    if aiter_utils.exists():
        s = aiter_utils.read_text()
        if "gfx1100" not in s and "gfx942" in s:
            s2 = s.replace('"gfx942"', '"gfx942", "gfx1100"', 1)
            if s2 != s:
                aiter_utils.write_text(s2)
                print("  OK    aiter utils: added gfx1100 to arch list")
            else:
                print("  WARN  aiter utils: gfx942 anchor not found, manual check needed")
    else:
        print("  WARN  aiter_meta/csrc/cpp_itfs/utils.py not found (check aiter layout)")


if __name__ == "__main__":
    main()

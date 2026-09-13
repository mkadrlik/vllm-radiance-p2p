# Troubleshooting: `vllm serve` segfaults / crashes on startup with the radiance image

Work through these in order of likelihood. Each item gives the exact signature
to look for in `docker compose logs` (or `docker inspect` exit code) and the fix.

## 1. Wrong GPU architecture (gfx1100-only image)

The radiance image is built with `PYTORCH_ROCM_ARCH=gfx1100` — it targets
**RX 7900 XTX / RX 7900 XT (RDNA3) only**. On RDNA4, MI300, or NVIDIA it dies
at startup, usually before the API server binds:

```
hipErrorInvalidImage
RuntimeError: No CUDA GPUs are available        # torch HIP init fails enumeration
torch.AcceleratorError: CUDA error: device kernel image is invalid
```

Check what you actually have:

```bash
rocminfo | grep -m2 "gfx"          # expect gfx1100 on the radiance profiles
lspci -d 1002:                     # 1002 = AMD; look for 7900 XTX/XT device ids
```

Fix: run the radiance profiles only on gfx1100. Other archs need their own
build (`PYTORCH_ROCM_ARCH=<arch>` through `Dockerfile.gfx1100`'s arg) — the
published image will not run there.

## 2. Stock ROCm 7.2.x RCCL is broken for gfx1100 TP>1

[ROCm issue #6074](https://github.com/ROCm/ROCm/issues/6074): amdclang on
7.2.x generates bad kernel code objects for RCCL, so tensor-parallel init
looks healthy and then detonates on first collective:

```
NCCL Init COMPLETE                          # then, on the first all_reduce:
HIP failure: the operation cannot be performed in the present state
pfn_hsa_system_get_info 4107
```

**The radiance image already ships the fix**: RCCL 2.27.7 taken from ROCm
7.1.1, a `ncclCommDump` symbol stub, and the jemalloc `LD_PRELOAD` chain —
all baked into the entrypoint. If you swap `librccl.so` out, mount a host
RCCL over it, or rebuild the image on a newer ROCm base, you reintroduce the
crash. **Do not replace RCCL.** Verify the shipped copy is intact:

```bash
docker compose run --rm --entrypoint bash <service> -lc 'ls -l /opt/rocm/lib/librccl.so*; /opt/rocm/libexec/rccl/rccl-symlinks 2>/dev/null || true'
```

## 3. `ROCR_VISIBLE_DEVICES` index trap (multi-GPU hosts)

The KFD **node index** reported by `rocminfo` is *not* the HIP enumeration
order used by `ROCR_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES`. They agree on
homogeneous hosts and disagree on hosts with mixed or differently-wired
cards. A wrong index gives you a segfault, an OOM on the wrong card, or a
silently single-GPU run.

Pin by **BDF or UUID**, not by index:

```bash
rocminfo | grep -E "Name:|Location:|UUID"   # shows gfx arch + BDF + UUID per node
rocm-smi --showuniqueid
```

Then set `ROCR_VISIBLE_DEVICES=0000:03:00.0,0000:83:00.0` (BDF form) or use
`HIP_VISIBLE_DEVICES` with UUIDs. On hosts where the two cards differ, verify
which physical GPU you actually grabbed — `rocm-smi --showpids` while the
engine is up shows per-card PIDs.

## 4. Missing `--language-model-only` → 128 GiB ViT dummy OOM

Multimodal Qwen checkpoints allocate a ~128 GiB dummy vision-tensor during
`profile_run` even when you only serve text. The kernel OOM-kills the process
mid-init — no Python traceback:

```bash
docker inspect --format '{{.State.ExitCode}}' <container>   # 137 = SIGKILL/OOM
dmesg | tail                                                 # oom-kill lines
```

Fix: add `--language-model-only` to the serve flags. It is required for
text-only serving on these checkpoints (the vision profile is also the fix
path if you *do* need images — see
[`docs/vllm-vision-gfx1100.md`](./vllm-vision-gfx1100.md) for the
`--limit-mm-per-prompt` + `--mm-processor-kwargs` combo that keeps the ViT
dummy allocation under budget).

## 5. Poisoned Triton / inductor cache (crash-loop)

Any boot killed **mid-autotune** leaves 0-byte `*.json` files in the Triton /
torch.compile caches. Every later boot then dies with `JSONDecodeError` — and
recreates more poison on its way down, so the container crash-loops forever
even though the original cause is long gone.

```
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

Fix (run inside the container or against the mounted cache dir):

```bash
find /root/.triton /tmp/torchinductor_root <your TRITON_CACHE_DIR> \
     -name '*.json' -size 0 -delete
```

Prevention: **never kill a boot mid-compile** — stop the container only after
`Uvicorn running` or after the compile phase has passed. With TP2, give each
rank its own cache dir (the two ranks race on the same cache files).

## 6. `RADIANCE_FAST_REDUCE=1` + MTP drafter + CUDA graphs = capture deadlock

Not a segfault but looks like one (process alive, zero progress, then a
watchdog abort on both ranks). The custom fast-reduce path breaks capture
symmetry when a speculative (MTP) drafter is active:

```
Capturing CUDA graphs 0/5        # hangs here
... watchdog timeout on _ALLGATHER_BASE after 600s on both ranks
```

Fix: keep `RADIANCE_FAST_REDUCE=0` whenever a drafter is configured (compose
default). Related: `RADIANCE_RUN_BWTEST=1` can wedge the SMU and **hard-reboot
the host** — keep it `0`. Details and the root-cause writeup:
[`docs/fast-reduce-mtp-deadlock.md`](./fast-reduce-mtp-deadlock.md).

## 7. AITER master toggle off (silent perf cliff, not a crash)

`VLLM_ROCM_USE_AITER=1` must be set or every tuned int8 kernel silently never
loads — the engine boots fine and runs 2× slow. Proof-of-engagement line in
the boot log:

```
Selected AiterInt8ScaledMMLinearKernel
```

If that line is absent, check the env var before profiling anything.

## 8. Compose `command:` form — flags only

The image `ENTRYPOINT` is `/opt/radiance_entrypoint.sh`, which execs
`vllm serve "$@"`. So the compose `command:` must be **flags only** (model as
`--model` or the first positional argument). Writing the old invocation:

```yaml
command: python3 -m vllm.entrypoints.openai.api_server --model ...   # WRONG
```

becomes trailing argv to `vllm serve` and dies instantly:

```
error: unrecognized arguments: python3
```

## 9. `request_memory` ValueError at startup = VRAM already held

```
ValueError: Could not allocate ... requested memory ...
```

This is not a crash bug — another process (often a co-resident model that did
not shut down cleanly) still holds the VRAM. Find the hog:

```bash
rocm-smi --showmeminfo vram
rocm-smi --showpids
```

Stop the other container or free the process, then start again.

---

Agent-readable condensed form (symptom-signature → fix) lives in
[AGENTS.md](../AGENTS.md) → *Segfault / crash triage*.

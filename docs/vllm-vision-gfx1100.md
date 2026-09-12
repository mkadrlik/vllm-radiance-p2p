# vLLM multimodal (vision) on gfx1100 — Qwen3.5-family ViT boot recipe

**Verified 2026-09-06** with `cyankiwi/Ornith-1.5-9B-AWQ-INT4`
(`Qwen3_5ForConditionalGeneration`, 333 BF16 visual tensors in the checkpoint)
on RX 7900 XTX, TP1, radiance 0.5.7 / vLLM 0.26.0.

## The failure mode

Dropping `--language-model-only` to enable vision produces a silent
crash-loop: EngineCore dies right after weight load, container restarts
forever, **no traceback anywhere in `docker logs`**. Symptom only becomes
legible via one-shot reproduction:

```bash
docker run --rm --entrypoint /bin/bash -e PYTHONFAULTHANDLER=1 \
  <same mounts/devices/env> <image> \
  -lc 'timeout -s KILL 240 vllm serve <args> > /out/crash.log 2>&1'
```

which shows:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 256.00 GiB.
  ... in F.scaled_dot_product_attention   (during determine_available_memory)
```

## Root cause

vLLM's dummy multimodal profiling runs the vision encoder at the *processor's
maximum* input size — and the Qwen3.5 image processor's default max image
size is unbounded. SDPA attention over that token count scales quadratically:
256 GB. The **video** modality is the worst offender; the image path also
needs a cap.

`RADIANCE_VIT_FLASH=1` (radiance's head_dim-72 ViT flash attention) does NOT
fix this — it was on during the 11-restart loop. The flash kernel still can't
allocate a 256 GB intermediate.

## The fix — both flags required

```yaml
- --limit-mm-per-prompt
- '{"image": 4, "video": 0}'     # video:0 — video profiling is the 256 GB one
- --mm-processor-kwargs
- '{"max_pixels": 1003520}'      # ~1024×979 cap on image profiling + serving
```

Result: boots HEALTHY first try, ~2 min; image round-trip correct (synthetic
PNG → "Red", 0.4 s, 87 prompt tokens for a 4×4 image); text decode unchanged
(86 t/s, TTFT 208 ms); KV pool 379K → 274K tokens (ViT weights + encoder
budgets cost ~1.5 GB — account for this when sizing `--gpu-memory-utilization`).

## Notes

- `--language-model-only` is the "text-only" switch; when present, vLLM logs
  `All limits of multimodal modalities supported by the model are set to 0,
  running in text-only mode` and skips ViT entirely.
- If you need video on this hardware, raise `video` and lower `max_pixels`
  until the profile fits — budget VRAM = frames × tokens/frame × quadratic
  attention per ViT block.
- Gateway caveat (ContextForge-class proxies): the proxy may register the
  model as text-only and 422 image content-parts. Route vision direct to the
  engine (Hermes `auxiliary.vision` uses `provider: custom`, bypassing CF).
- `RADIANCE_VIT_FLASH=1` is left ON in the `ornith-9b` profile (it's the
  faster encoder path once sizing is fixed), but it is not load-bearing for
  boot success.

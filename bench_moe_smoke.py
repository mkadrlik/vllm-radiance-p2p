"""MoE smoke test — triton#10808 (fused-MoE Triton JIT memory fault on gfx1100).
Drives vllm's fused_experts on our exact stack (triton 3.6.0 / aiter 0.1.17 /
vllm 0.26.0) with a small MoE and verifies: (1) no JIT memory fault, (2) output
is finite, (3) result matches a naive torch reference.
"""
import os, sys, torch

device = "cuda:0"

# small MoE: 8 experts, top-2, hidden 1024, inter 2048, 64 tokens
E, TOPK, H, I, T = 8, 2, 1024, 2048, 64
torch.manual_seed(0)

x = torch.randn(T, H, device=device, dtype=torch.float16)
w1 = torch.randn(E, 2 * I, H, device=device, dtype=torch.float16) * 0.02  # [E, 2I, H]
w2 = torch.randn(E, H, I, device=device, dtype=torch.float16) * 0.02        # [E, H, I]

# router: pick a deterministic topk
logits = torch.randn(T, E, device=device, dtype=torch.float32)
topk_ids = torch.topk(logits, TOPK, dim=-1).indices.to(torch.int32)   # [T, TOPK]
topk_w = torch.softmax(logits, dim=-1)
topk_weights = torch.gather(topk_w, 1, topk_ids.long()).to(torch.float32)

from vllm.model_executor.layers.fused_moe import fused_experts, MoEActivation

# Run 1: JIT compile + first call (this is where #10808 faults)
out = fused_experts(
    hidden_states=x, w1=w1, w2=w2,
    topk_weights=topk_weights, topk_ids=topk_ids,
    activation=MoEActivation.SILU,
    global_num_experts=E,
)
print(f"[run1] shape={tuple(out.shape)} finite={torch.isfinite(out).all().item()}")

# Run 2: repeated (determinism / no accumulation fault)
out2 = fused_experts(
    hidden_states=x, w1=w1, w2=w2,
    topk_weights=topk_weights, topk_ids=topk_ids,
    activation=MoEActivation.SILU,
    global_num_experts=E,
)
print(f"[run2] finite={torch.isfinite(out2).all().item()} "
      f"run1==run2 exact={torch.equal(out, out2)} "
      f"max|d|={(out-out2).abs().max().item():.2e}")

# Naive torch reference (same topk, silu, sum)
ref = torch.zeros(T, H, device=device, dtype=torch.float32)
for t in range(T):
    for k in range(TOPK):
        e = int(topk_ids[t, k]); w = float(topk_weights[t, k])
        gate = x[t].float() @ w1[e].float().T          # [2I]
        g, u = gate[:I], gate[I:]
        act = torch.nn.functional.silu(g) * u          # [I]
        ref[t] += w * (act @ w2[e].float().T)          # [H]
err = (out.float() - ref).abs().max().item()
rel = err / ref.abs().max().item()
print(f"[ref ] max|err|={err:.3e}  rel={rel:.3e}")
assert torch.isfinite(out).all(), "output has NaN/Inf"
assert rel < 5e-2, f"MoE output diverges from reference: rel={rel:.3e}"
print("PASS: MoE Triton path compiles, runs, finite, matches reference (triton#10808 not triggered)")

"""Diagnostic (not a bar): where does the da/db parity error come from? The hook_parity_S1024 inputs, seed 0 and 1.
Prints: rel err of dqs (grad wrt the scaled queries, max-norm); per head, the conditioning of da,
kappa = sum|t| / |sum t| with t = dqs * q * ln(i+1); da recomputed in float64 from the hook's fp32 dqs vs the reference."""
import math, sys, json
import torch
import torch.nn.functional as F
sys.path.insert(0, "C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad/phase_k/K1/chase")
import resolvent_hook as hk
R = hk.R
torch.backends.cuda.matmul.allow_tf32 = False


def qkv(x, wq, H):
    B, T, Dm = x.shape
    q, k, v = F.linear(x, wq).split(Dm, 2)
    return [t.view(B, T, H, Dm // H).transpose(1, 2) for t in (q, k, v)]


for seed in (0, 1):
    S = 1024
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, S, 128, generator=g, dtype=torch.float64).float()
    wq = (torch.randn(384, 128, generator=g, dtype=torch.float64) / math.sqrt(128)).float()
    wo = (torch.randn(128, 128, generator=g, dtype=torch.float64) / math.sqrt(128)).float()
    gy = torch.randn(2, S, 128, generator=g, dtype=torch.float64).float()
    out = {}
    for dt in (torch.float32, torch.float64):
        xx, ww, oo, gg = (t.cuda().to(dt) for t in (x, wq, wo, gy))
        q, k, v = qkv(xx, ww, 2)
        a = torch.ones(2, device="cuda", dtype=dt, requires_grad=True)
        b = torch.zeros(2, device="cuda", dtype=dt, requires_grad=True)
        s = hk.logit_scale(a, b, S)
        qs = (q * s[None, :, :, None]).detach().requires_grad_()
        y = hk.read(qs, k, v) if dt == torch.float32 else R.resolvent(qs, k, v, 0.999, path="dense")
        F.linear(y.transpose(1, 2).reshape(2, S, 128), oo).backward(gg)
        out[dt] = (q.double(), qs.grad.double())
    q64, dqs64 = out[torch.float64]
    _, dqs32 = out[torch.float32]
    ln = torch.log(torch.arange(1, S + 1, device="cuda", dtype=torch.float64))[None, None, :, None]
    t64 = dqs64 * q64 * ln
    t32 = dqs32 * q64 * ln
    da64, da_from32 = t64.sum((0, 2, 3)), t32.sum((0, 2, 3))
    kappa = t64.abs().sum((0, 2, 3)) / da64.abs()
    rel_dqs = float((dqs32 - dqs64).abs().max() / dqs64.abs().max())
    print(json.dumps({"seed": seed, "rel_dqs": rel_dqs, "kappa_a": kappa.tolist(),
                      "rel_da_from_fp32_dqs_fp64_sum": float((da_from32 - da64).abs().max() / da64.abs().max()),
                      "normwise_da": float(((da_from32 - da64).abs() / t64.abs().sum((0, 2, 3))).max())}))

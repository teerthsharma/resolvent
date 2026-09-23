"""Diagnostic (not a bar): localize the fp32 dq error of backward_blocked at g = .999 (hook_parity seed 1 inputs).
rel = max|dq - dq64| / max|dq64|, dq wrt the scaled queries. Variants:
 V1 fp32 forward x32 + fp32 backward_blocked (the hook)
 V2 x32, backward_blocked in float64            (backward rounding removed; forward error kept)
 V3 x64 cast to fp32, fp32 backward_blocked     (forward error removed)
 V4 u from fp32 backward_blocked, dz dense in float64 with x32   (the solve's u error only)
 V5 u64 exact, dz dense float64 with x32                        (x error only, through dP)"""
import math, sys, json
import torch
import torch.nn.functional as F
sys.path.insert(0, "C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad/phase_k/K1/chase")
import resolvent_hook as hk
R = hk.R
torch.backends.cuda.matmul.allow_tf32 = False
G = 0.999

for seed in (0, 1):
    S, H = 1024, 2
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, S, 128, generator=g, dtype=torch.float64).float()
    wq = (torch.randn(384, 128, generator=g, dtype=torch.float64) / math.sqrt(128)).float()
    wo = (torch.randn(128, 128, generator=g, dtype=torch.float64) / math.sqrt(128)).float()
    gy = torch.randn(2, S, 128, generator=g, dtype=torch.float64).float()
    q, k, v = (t.view(2, S, H, 64).transpose(1, 2).contiguous() for t in F.linear(x.cuda(), wq.cuda()).split(128, 2))
    qs = (q * hk.logit_scale(torch.ones(2, device="cuda"), torch.zeros(2, device="cuda"), S)[None, :, :, None]).contiguous()
    gx = (gy.cuda() @ wo.cuda()).view(2, S, H, 64).transpose(1, 2).contiguous()
    d = lambda t: t.double()
    x32 = hk.read(qs, k, v)
    x64 = R._dense_fwd(d(qs), d(k), d(v), G) if hasattr(R, "_dense_fwd") else hk._dense_fwd(d(qs), d(k), d(v), G)
    dq64, _, dv64 = R.backward_blocked(d(qs), d(k), x64, d(gx), G)
    u64 = dv64 / (1 - G)

    def dz_dense(u, xx):
        W = R.weights(d(qs), d(k))
        dP = G * (d(u) @ d(xx).transpose(-1, -2))
        dz = W * (dP - (W * dP).sum(-1, keepdim=True))
        return dz @ d(k) / 8.0

    rel = lambda a: float((d(a) - dq64).abs().max() / dq64.abs().max())
    dq1, _, dv1 = R.backward_blocked(qs, k, x32, gx, G)
    dq2 = R.backward_blocked(d(qs), d(k), d(x32), d(gx), G)[0]
    dq3 = R.backward_blocked(qs, k, x64.float(), gx, G)[0]
    u32 = dv1 / (1 - G)
    out = {"seed": seed, "V1_hook": rel(dq1), "V2_bwd_f64": rel(dq2), "V3_exact_x": rel(dq3),
           "V4_u32_only": rel(dz_dense(u32, x64)), "V5_x32_only": rel(dz_dense(u64, x32)),
           "rel_x32": float((d(x32) - x64).abs().max() / x64.abs().max()),
           "rel_u32": float((d(u32) - u64).abs().max() / u64.abs().max()),
           "max_u_over_max_gx": float(u64.abs().max() / d(gx).abs().max())}
    print(json.dumps(out), flush=True)

# V6 (added after V1-V5 ran): the hook's backward_blocked (x centred in dP), fp32, x32
for seed in (0, 1):
    S, H = 1024, 2
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, S, 128, generator=g, dtype=torch.float64).float()
    wq = (torch.randn(384, 128, generator=g, dtype=torch.float64) / math.sqrt(128)).float()
    wo = (torch.randn(128, 128, generator=g, dtype=torch.float64) / math.sqrt(128)).float()
    gy = torch.randn(2, S, 128, generator=g, dtype=torch.float64).float()
    q, k, v = (t.view(2, S, H, 64).transpose(1, 2).contiguous() for t in F.linear(x.cuda(), wq.cuda()).split(128, 2))
    qs = (q * hk.logit_scale(torch.ones(2, device="cuda"), torch.zeros(2, device="cuda"), S)[None, :, :, None]).contiguous()
    gx = (gy.cuda() @ wo.cuda()).view(2, S, H, 64).transpose(1, 2).contiguous()
    d = lambda t: t.double()
    x32 = hk.read(qs, k, v)
    x64 = hk._dense_fwd(d(qs), d(k), d(v), G)
    dq64 = R.backward_blocked(d(qs), d(k), x64, d(gx), G)[0]
    rel = lambda a: float((d(a) - dq64).abs().max() / dq64.abs().max())
    print(json.dumps({"seed": seed, "V6_centred_x32": rel(hk.backward_blocked(qs, k, x32, gx, G)[0]),
                      "V6b_centred_exact_x": rel(hk.backward_blocked(qs, k, x64.float(), gx, G)[0])}), flush=True)

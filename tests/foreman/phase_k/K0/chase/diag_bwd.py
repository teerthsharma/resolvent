# diagnostic (not a bar): fused vs dense-fp32 error sources, fwd and bwd, S=1024
import sys, json, math, torch
sys.path.insert(0, "."); import resolvent as R
from test_chase_k0 import _qkv, rel
q, k, v = _qkv(1024, seed=2); gout = _qkv(1024, seed=3)[0]
cn = R._cost_n()
def eff_exact(qq, kk, vv):   # materialized fp32 replacement for the fused kernel inside fs5
    z = (qq @ kk.transpose(-1, -2)) * cn.SCALE
    lse = torch.logsumexp(z, -1)
    return torch.exp(z - lse[..., None]) @ vv, lse
orig = cn._eff
for a, b in ((0.0, 1.0), (1.0, 0.0)):
    qs = R.ssmax_q(q, torch.tensor(a, device="cuda"), torch.tensor(b, device="cuda"))
    g64 = [t.double().clone().requires_grad_() for t in (qs, k, v)]
    x64 = R.resolvent(*g64, 0.99, path="dense"); x64.backward(gout.double())
    row = dict(a=a, b=b)
    for path, patch in (("fused", False), ("fused_exact_eff", True), ("dense", False)):
        cn._eff = eff_exact if patch else orig
        t = [x.clone().requires_grad_() for x in (qs, k, v)]
        x = R.resolvent(*t, 0.99, path="fused" if path != "dense" else "dense"); x.backward(gout)
        row[path] = dict(fwd=rel(x, x64), dq=rel(t[0].grad, g64[0].grad), dk=rel(t[1].grad, g64[1].grad), dv=rel(t[2].grad, g64[2].grad))
    cn._eff = orig
    print(json.dumps(row))

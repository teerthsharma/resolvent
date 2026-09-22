# diagnostic (not a bar): per-block fused forward error and kernel lse error, S=1024, a=1
import sys, json, math, torch
sys.path.insert(0, "."); import resolvent as R
from test_chase_k0 import _qkv
print("tf32", torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision())
q, k, v = _qkv(1024, seed=2)
qs = R.ssmax_q(q, torch.tensor(1.0, device="cuda"), torch.tensor(0.0, device="cuda"))
with torch.no_grad():
    x64 = R.resolvent(qs.double(), k.double(), v.double(), 0.99, path="dense")
    x = R.fused_forward(qs, k, v, 0.99)
    err = (x.double() - x64).abs().amax(dim=(0, 1, 3)) / x64.abs().max()
    print("fwd err per 256-block", [float(err[i:i+256].max()) for i in range(0, 1024, 256)])
    _, lse, _, _ = torch.ops.aten._scaled_dot_product_efficient_attention(qs, k, v, None, True, 0.0, True)
    z64 = R._logits(qs.double(), k.double())
    l64 = torch.logsumexp(z64, -1)
    print("lse eff err max", float((lse[..., :1024].double() - l64).abs().max()))
    l32 = torch.logsumexp(R._logits(qs, k), -1)
    print("lse torch fp32 err max", float((l32.double() - l64).abs().max()))
cn = R._cost_n()
def eff_exact(qq, kk, vv):
    z = (qq.double() @ kk.double().transpose(-1, -2)) * cn.SCALE
    lse = torch.logsumexp(z, -1)
    return (torch.exp(z - lse[..., None]) @ vv.double()).float(), lse.float()
with torch.no_grad():
    for name, f in (("kernel", cn._eff), ("exact64", eff_exact)):
        cn._eff = f
        x = R.fused_forward(qs, k, v, 0.99); torch.cuda.synchronize()
        err = (x.double() - x64).abs().amax(dim=(0, 1, 3)) / x64.abs().max()
        print(name, "fwd err per block", [float(err[i:i+256].max()) for i in range(0, 1024, 256)])
    # ungraded single-stream check: fs4 (no side stream)
    cn._eff = eff_exact.__globals__["cn"]._eff
    import importlib

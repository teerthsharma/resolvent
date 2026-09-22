# diagnostic: where does the fp32 forward error at S=1024 come from? (not a bar)
import sys, json, math, torch
sys.path.insert(0, "."); import resolvent as R
from test_chase_k0 import _qkv, rel
q, k, v = _qkv(1024)
for a, b in ((0.0, 1.0), (0.5, 0.0), (1.0, 0.0)):
    qs = R.ssmax_q(q, torch.tensor(a, device="cuda"), torch.tensor(b, device="cuda"))
    with torch.no_grad():
        ref = R.resolvent(qs.double(), k.double(), v.double(), 0.99, path="dense")
        fused = rel(R.resolvent(qs, k, v, 0.99, path="fused"), ref)
        dense32 = rel(R.resolvent(qs, k, v, 0.99, path="dense"), ref)
        # floor: float64 solve on W built from fp32-rounded logits (the only fp32 error is the logit)
        z32 = R._logits(qs, k).double()
        W = torch.softmax(z32, -1); S = 1024
        x = torch.linalg.solve_triangular(torch.eye(S, dtype=torch.float64, device="cuda") - 0.99 * W, 0.01 * v.double(), upper=False)
        floor = rel(x, ref)
        zmax = float(R._logits(qs.double(), k.double()).masked_fill(torch.ones(S,S,dtype=bool,device='cuda').triu(1), 0).abs().max())
    print(json.dumps(dict(a=a, b=b, fused=fused, dense32=dense32, logit_rounding_floor=floor, max_abs_logit=zmax)))

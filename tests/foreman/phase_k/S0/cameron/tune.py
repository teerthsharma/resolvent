"""Pick FWD/BWD block configs for flite at the 11.2M attention shape (B8 H6 S1024 D64), fwd+bwd ms.
Also times SDPA (twin) on the same tensors as the control."""
import sys, itertools, torch, triton, torch.nn.functional as F
import flite
B, H, S, D = 8, 6, 1024, int(sys.argv[1]) if len(sys.argv) > 1 else 64
g = torch.Generator(device="cuda").manual_seed(0)
q, k, v, do = (torch.randn(B, H, S, D, device="cuda", generator=g) for _ in range(4))
u = torch.randn(B, S, device="cuda", generator=g) + 1.0
leaves = [t.requires_grad_() for t in (q, k, v, u)]
beta = torch.tensor(1.0, device="cuda", requires_grad=True)
def step_flite():
    o = flite.flite_attention(q, k, v, u, beta); o.backward(do)
def step_sdpa():
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True); o.backward(do)
print("sdpa_ms", round(triton.testing.do_bench(step_sdpa), 3), flush=True)
best = {}
flite.PREC = sys.argv[2] if len(sys.argv) > 2 else "tf32x3"
for which in ("FWD", "BWD"):
    res = []
    for bm, bn, w, st in itertools.product((32, 64), (32, 64), (4, 8), (1, 2)):
        setattr(flite, which, (bm, bn, w, st))
        try:
            ms = triton.testing.do_bench(step_flite)
        except Exception as e:
            continue
        res.append((ms, (bm, bn, w, st)))
    res.sort(); best[which] = res[0][1]; setattr(flite, which, res[0][1])
    print(which, [(round(m, 3), c) for m, c in res[:5]], flush=True)
print("flite_ms", round(triton.testing.do_bench(step_flite), 3), "best", best)

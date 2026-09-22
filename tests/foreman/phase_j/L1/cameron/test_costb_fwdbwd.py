"""COST-B' forward+backward. IMPL=author (costb_impl.py eager, RED expected) | fused.
  grad  -- d(sum(out*R))/d{x, w_head, q, k, v} vs f64 autograd through the contract
           fold (sequential G_ij order, no prefix trick), max rel err <= 1e-3
  ratio -- fwd+bwd wall clock / SDPA fwd+bwd (default dispatch), S=4096 fp32 B1 H8 D64, <= 1.3
Writes costb_fwdbwd_<IMPL>.json.
"""
import json, os, sys, time
import torch
import torch.nn.functional as F
import test_costb_fused as T

IMPL = T.IMPL


def ref_fwd(x, w, q, k, v):
    """Differentiable f64 contract reference: Pi_i = q_i Pi_{i-1} sequentially."""
    B, H, S, D = q.shape
    qn = F.normalize(x @ w.T, dim=-1)
    acc, Ps = torch.tensor([1.0, 0, 0, 0], dtype=x.dtype).expand(B, 4), []
    for i in range(S):
        acc = T.qmul(qn[:, i], acc)
        Ps.append(acc)
    P = torch.stack(Ps, 1)  # (B,S,4)
    Pc = P * torch.tensor([1.0, -1, -1, -1], dtype=x.dtype)
    rot = lambda t, p: T.qmul(p[:, None, :, None, :].expand(B, H, S, D // 4, 4),
                              t.reshape(B, H, S, D // 4, 4)).reshape(B, H, S, D)
    s = (q @ k.transpose(-1, -2) / D ** 0.5).masked_fill(torch.triu(torch.ones(S, S, dtype=torch.bool), 1), float("-inf"))
    return rot(torch.softmax(s, -1) @ rot(v, Pc), P)


def impl_fwd(x, w, q, k, v):
    if IMPL == "author":
        return T.author_fwd(x, w, q, k, v)
    import costb_fused as cf
    return cf.mode_bprime_train(x, w, q, k, v)


def grads(fn, args, R):
    args = [a.detach().clone().requires_grad_(True) for a in args]
    (fn(*args) * R).sum().backward()
    return [a.grad for a in args]


def check_grad():
    args = T.inputs(1, 2, 64, 64, seed=3)
    R = torch.randn(1, 2, 64, 64, device="cuda", generator=torch.Generator(device="cuda").manual_seed(4))
    g_ref = grads(ref_fwd, [a.double().cpu() for a in args], R.double().cpu())
    g_imp = grads(impl_fwd, args, R)
    return max(((gi.double().cpu() - gr).abs().max() / gr.abs().max()).item() for gi, gr in zip(g_imp, g_ref))


def fb_ms(fn, args):
    args = [a.detach().clone().requires_grad_(True) for a in args]
    def step():
        fn(*args).sum().backward()
        for a in args:
            a.grad = None
    return T.cuda_ms(step, iters=10, reps=5)


def check_ratio():
    x, w, q, k, v = T.inputs(1, 8, 4096, 64)
    sd = lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)
    ts, tm = [], []
    for _ in range(3):
        ts.append(fb_ms(sd, [q, k, v]))
        tm.append(fb_ms(impl_fwd, [x, w, q, k, v]))
    s, m = sorted(ts)[1], sorted(tm)[1]
    return m / s, s, m


if __name__ == "__main__":
    gerr = check_grad()
    ratio, s, m = check_ratio()
    res = dict(impl=IMPL, grad_max_rel_err_vs_contract_f64=gerr, ms_sdpa_fwd_bwd=s,
               ms_mode_bprime_fwd_bwd=m, ratio_fwd_bwd=ratio,
               shape="B1 H8 S4096 D64 fp32 causal, loss=out.sum()", t=time.strftime("%Y-%m-%d %H:%M:%S"))
    print(json.dumps(res, indent=1))
    with open(os.path.join(T.HERE, "costb_fwdbwd_%s.json" % IMPL), "w") as f:
        json.dump(res, f, indent=1)
    fails = []
    if not gerr <= 1e-3:
        fails.append("grad: max rel err vs contract f64 = %.3e > 1e-3" % gerr)
    if not ratio <= 1.3:
        fails.append("ratio: fwd+bwd %.3f ms / SDPA %.3f ms = %.3fx > 1.3x" % (m, s, ratio))
    if fails:
        print("RED " + IMPL + ": " + " | ".join(fails))
        sys.exit(1)
    print("GREEN " + IMPL + ": grad %.2e, fwd+bwd ratio %.3fx" % (gerr, ratio))

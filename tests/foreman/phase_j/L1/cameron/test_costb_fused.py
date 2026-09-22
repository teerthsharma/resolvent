"""COST-B' route test. IMPL=author runs Foreman's costb_impl.py as it stands
(RED expected); IMPL=fused runs costb_fused.py. Three checks:
  fold  -- out_i == sum_j p_ij (q_i ... q_{j+1}) v_j, the contract's G_ij, vs f64 direct path products
  scan  -- Pi at S=4096 fp32 vs f64 sequential contract-order product, <= 1e-4 (Q1 bound)
  ratio -- forward wall clock / SDPA default dispatch, S=4096 fp32 B=1 H=8 D=64, <= 1.3
Run: python test_costb_fused.py   (env IMPL=author|fused). Writes costb_route_<IMPL>.json.
"""
import json, os, sys, time
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
SPJ = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, SPJ)
IMPL = os.environ.get("IMPL", "fused")
DEV = "cuda"


def qmul(p, q):
    w1, x1, y1, z1 = p.unbind(-1)
    w2, x2, y2, z2 = q.unbind(-1)
    return torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def contract_ref(x, w_head, q, k, v):
    """f64 direct: G_ij = q_i ... q_{j+1} by sequential products, no prefix trick."""
    x, w_head, q, k, v = (t.double().cpu() for t in (x, w_head, q, k, v))
    B, H, S, D = q.shape
    qt = F.normalize(x @ w_head.T, dim=-1)  # (B,S,4)
    out = torch.zeros(B, H, S, D, dtype=torch.float64)
    p = torch.softmax((q @ k.transpose(-1, -2) / D ** 0.5).masked_fill(
        torch.triu(torch.ones(S, S, dtype=torch.bool), 1), float("-inf")), -1)
    for b in range(B):
        G = torch.zeros(S, 4, dtype=torch.float64)  # G[j] = path product from j to current i
        for i in range(S):
            G[:i] = qmul(qt[b, i].expand(i, 4), G[:i])
            G[i] = torch.tensor([1.0, 0, 0, 0], dtype=torch.float64)
            gv = qmul(G[: i + 1, None, :].expand(i + 1, D // 4, 4),
                      v[b, :, : i + 1].reshape(H, i + 1, D // 4, 4))  # (H,i+1,nb,4)
            out[b, :, i] = torch.einsum("hj,hjc->hc", p[b, :, i, : i + 1],
                                        gv.reshape(H, i + 1, D))
    return out


def author_fwd(x, w_head, q, k, v):
    import costb_impl as ci
    B, H, S, _ = q.shape
    quat = x @ w_head.T
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    Pi = ci.hillis_steele_prefix_quat(quat.unsqueeze(1).expand(B, H, S, 4))
    out = F.scaled_dot_product_attention(q, k, ci.rotate_blocks(v, ci.qconj(Pi)), is_causal=True)
    return ci.rotate_blocks(out, Pi)


def author_scan(r):
    import costb_impl as ci
    return ci.hillis_steele_prefix_quat(F.normalize(r, dim=-1))


def fused_fwd(x, w_head, q, k, v):
    import costb_fused as cf
    return cf.mode_bprime_fused(x, w_head, q, k, v)


def fused_scan(r):
    import costb_fused as cf
    return cf.prefix_scan(r)


FWD = {"author": author_fwd, "fused": fused_fwd}[IMPL]
SCAN = {"author": author_scan, "fused": fused_scan}[IMPL]


def inputs(B, H, S, D, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(B, S, H * D, device=DEV, generator=g)
    w = torch.randn(4, H * D, device=DEV, generator=g) * 0.1
    q, k, v = (torch.randn(B, H, S, D, device=DEV, generator=g) for _ in range(3))
    return x, w, q, k, v


def check_fold():
    args = inputs(1, 2, 64, 64, seed=1)
    ref = contract_ref(*args)
    out = FWD(*args).double().cpu()
    return (out - ref).abs().max().item()


def check_scan():
    g = torch.Generator(device=DEV).manual_seed(2)
    r = torch.randn(1, 4096, 4, device=DEV, generator=g)
    P = SCAN(r).double().cpu()[0]
    qt = F.normalize(r.double().cpu()[0], dim=-1)
    ref = torch.empty_like(qt)
    acc = torch.tensor([1.0, 0, 0, 0], dtype=torch.float64)
    for i in range(qt.shape[0]):
        acc = qmul(qt[i], acc)  # Pi_i = q_i * Pi_{i-1}
        ref[i] = acc
    return (P - ref).abs().max().item()


def cuda_ms(fn, iters=30, reps=7):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) / iters)
    ts.sort()
    return ts[len(ts) // 2]


def sdpa_backend_ms(q, k, v):
    from torch.nn.attention import sdpa_kernel, SDPBackend
    res = {}
    for name in ("MATH", "EFFICIENT_ATTENTION", "FLASH_ATTENTION", "CUDNN_ATTENTION"):
        try:
            with sdpa_kernel([getattr(SDPBackend, name)]):
                res[name] = cuda_ms(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
        except Exception as e:
            res[name] = "unavailable: " + str(e).splitlines()[0][:120]
    return res


def check_ratio():
    x, w, q, k, v = inputs(1, 8, 4096, 64)
    with torch.no_grad():
        per_backend = sdpa_backend_ms(q, k, v)
        t_sdpa, t_impl = [], []
        for _ in range(3):  # interleave to cancel clock drift
            t_sdpa.append(cuda_ms(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)))
            t_impl.append(cuda_ms(lambda: FWD(x, w, q, k, v)))
    s, m = sorted(t_sdpa)[1], sorted(t_impl)[1]
    return m / s, s, m, per_backend


if __name__ == "__main__":
    fold_err = check_fold()
    scan_err = check_scan()
    ratio, t_s, t_m, per_backend = check_ratio()
    res = dict(impl=IMPL, device=torch.cuda.get_device_name(), torch=torch.__version__,
               fold_max_abs_err_vs_contract_f64=fold_err, scan_max_abs_err_fp32_vs_f64_S4096=scan_err,
               ms_sdpa_default=t_s, ms_mode_bprime_fwd=t_m, ratio_fwd=ratio,
               sdpa_ms_by_backend=per_backend, shape="B1 H8 S4096 D64 fp32 causal",
               t=time.strftime("%Y-%m-%d %H:%M:%S"))
    print(json.dumps(res, indent=1))
    with open(os.path.join(HERE, "costb_route_%s.json" % IMPL), "w") as f:
        json.dump(res, f, indent=1)
    fails = []
    if not fold_err <= 1e-4:
        fails.append("fold: max |out - contract G_ij fold| = %.3e > 1e-4" % fold_err)
    if not scan_err <= 1e-4:
        fails.append("scan: max |Pi_fp32 - Pi_f64| = %.3e > 1e-4" % scan_err)
    if not ratio <= 1.3:
        fails.append("ratio: Mode B' fwd %.3f ms / SDPA %.3f ms = %.3fx > 1.3x" % (t_m, t_s, ratio))
    if fails:
        print("RED " + IMPL + ": " + " | ".join(fails))
        sys.exit(1)
    print("GREEN " + IMPL + ": fold %.2e, scan %.2e, ratio %.3fx" % (fold_err, scan_err, ratio))

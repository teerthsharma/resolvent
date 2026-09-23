"""Chase K2 bars, third registration (2026-09-23, after run2 on hook 98259dff and before these run). RED first on
resolvent_sp_stub.py. Helpers, inputs, board line and exit rule as test_chase_k2.py (imported as T).

Why:
 - sp_fwd_parity_S1024 (run2): every numeric row passed, but its mutant row read NaN (support + 1 reaches a -inf logit in
   the first rows, so tau = -inf and W = inf - inf). "rel > 1e-6" is False for NaN, so a mutant that destroys the read
   was scored as matching. v2 predicate: the mutant must NOT satisfy rel <= 1e-6. Everything else is as v1.
 - sp_mem_S16384 (run2): (a) peak reserved 6.36 GiB at H4 D32 against 0.68 GiB allocated. The hook's row blocks widen with
   hi, so the caching allocator kept every narrower freed segment. The hook now writes the logits into one workspace
   and runs the sort pass widest-first (hook sha in the run header). The memory lines are unchanged.
   (b) resid_last256 read 1.10e-5 at H2 D64 against 1e-5: the residual of the fp32 x under float64 W. At S 16384 the
   logits reach ~40 in magnitude, and the fp32 rounding of z and tau is ~1e-6 absolute. Not bound; hypothesis only.
   Replaced by: the hook in float64 at S 16384 satisfies the equation on the last 256 rows (residual <= 1e-12 under
   float64 W from R.sparsemax), and the fp32 run agrees with it within 1e-4. That 1e-4 is a wrong-block line (a wrong
   block errs at O(1)). It was set after run2's 1.1e-5 residual, and it is not a precision claim: the 1e-5 fp32
   precision line is claimed at S 1024 only (sp_fwd_parity_S1024_v2).

BARS
 sp_fwd_parity_S1024_v2  sp_fwd_parity_S1024's rows and lines, unchanged (f64 <= 1e-10; f32 x <= 1e-5, dq/dk/dv <= 1e-4;
                       pad S 1000 <= 1e-10; module: autocast on == off bitwise, fp32, y <= 1e-5, dx/dw/da/db <= 1e-4),
                       and the mutant (_tau support size + 1, in1, g .999, f64) must fail: not (rel <= 1e-6).
 sp_mem_S16384_v2      per shape (H2 D64) [rdepth] and (H4 D32), B1 S16384, g .999, c 256, q, k, v, gx ~ N(0,1) seed 2,
                       a = 1, b = 0:
                         fp32 CUDA read() fwd + bwd after empty_cache + reset_peak_memory_stats: max_memory_allocated
                         < 6 GiB and max_memory_reserved < 6 GiB; x, dq, dk, dv finite; prefix x[..., :1024, :] vs read()
                         at S 1024 rel <= 1e-6;
                         float64 CUDA read() forward on the same inputs: residual on rows 16128..16383,
                         max|x_i - g sum_j W_ij x_j - (1-g) v_i| / max|x| <= 1e-12, W = R.sparsemax(R._logits_rows(...))
                         in float64; and rel(fp32 x, float64 x) <= 1e-4.
                       Control (must fail the memory bar): densec fwd + bwd at H2 D64 raises OutOfMemoryError or peaks
                       >= 6 GiB.
"""
import hashlib, json, math, os, sys
sys.dont_write_bytecode = True
os.environ.setdefault("CHASE_HOOK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "resolvent_sp.py"))
import test_chase_k2 as T
import torch
import torch.nn.functional as F

R, rel, inputs, densec, GiB = T.R, T.rel, T.inputs, T.densec, T.GiB


@T.bar("sp_fwd_parity_S1024_v2")
def t_parity_v2():
    hk = T.load(T.HOOK, "hook_under_test")
    out, ok = {}, True
    for nm, (seed, a, b) in T.INS.items():
        qs, k, v, gx = inputs(seed, a, b, 1024)
        for g in (0.5, 0.999):
            xr, gr = densec(qs, k, v, g, gx)
            x64 = hk.read(qs, k, v, g, 256)
            ts = [t.float().cuda().requires_grad_() for t in (qs, k, v)]
            x32 = hk.read(*ts, g, 256)
            x32.backward(gx.float().cuda())
            row = {"f64_x": rel(x64, xr), "f32_x": rel(x32.detach(), xr),
                   **{"f32_" + n: (rel(t.grad, r) if t.grad is not None else float("inf")) for n, t, r in zip(("dq", "dk", "dv"), ts, gr)}}
            ok &= row["f64_x"] <= 1e-10 and row["f32_x"] <= 1e-5 and all(row["f32_" + n] <= 1e-4 for n in ("dq", "dk", "dv"))
            out[f"{nm}_g{g}"] = row
    qs, k, v, _ = inputs(0, (1.0, 1.0), (0.0, 0.0), 1000)
    out["pad_S1000_f64_x"] = rel(hk.read(qs, k, v, 0.999, 256), densec(qs, k, v, 0.999)[0])
    ok &= out["pad_S1000_f64_x"] <= 1e-10
    g = torch.Generator().manual_seed(3)
    x = torch.randn(1, 1024, 128, generator=g, dtype=torch.float64).float()
    wq = (torch.randn(384, 128, generator=g, dtype=torch.float64) / math.sqrt(128)).float()
    wo = (torch.randn(128, 128, generator=g, dtype=torch.float64) / math.sqrt(128)).float()
    gy = torch.randn(1, 1024, 128, generator=g, dtype=torch.float64).float()
    i64 = [t.double().requires_grad_() for t in (x, wq, wo, torch.ones(2), torch.zeros(2))]
    xx, w1, w2, a64, b64 = i64
    q, kk, vv = F.linear(xx, w1).split(128, 2)
    q, kk, vv = (t.view(1, 1024, 2, 64).transpose(1, 2) for t in (q, kk, vv))
    s = a64[:, None] * torch.log(torch.arange(1, 1025, dtype=torch.float64))[None] + b64[:, None]
    y64 = F.linear(R.resolvent(q * s[None, :, :, None], kk, vv, 0.999, kind="sparsemax", path="densec")
                   .transpose(1, 2).reshape(1, 1024, 128), w2)
    y64.backward(gy.double())
    ys, grads = [], []
    for amp in (False, True):
        m = hk.ResolventAttention(128, 2).cuda()
        with torch.no_grad():
            m.qkv.weight.copy_(wq); m.out.weight.copy_(wo)
        xi = x.cuda().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            y = m(xi)
        y.backward(gy.cuda())
        ys.append(y.detach())
        grads.append([xi.grad, m.qkv.weight.grad, m.out.weight.grad, m.a.grad, m.b.grad])
    mod = {"amp_equal_bitwise": bool(torch.equal(ys[0], ys[1])) and all(
               p is not None and q_ is not None and torch.equal(p, q_) for p, q_ in zip(*grads)),
           "dtype": str(ys[1].dtype), "y": rel(ys[1], y64.detach())}
    for n, gh, i in zip(("dx", "dw_qkv", "dw_out", "da", "db"), grads[1], range(5)):
        mod[n] = rel(gh, i64[i].grad) if gh is not None else float("inf")
    ok &= mod["amp_equal_bitwise"] and mod["dtype"] == "torch.float32" and mod["y"] <= 1e-5 and \
        all(mod[n] <= 1e-4 for n in ("dx", "dw_qkv", "dw_out", "da", "db"))
    out["module"] = mod
    qs, k, v, _ = inputs(0, (1.0, 1.0), (0.0, 0.0), 1024)
    orig = hk._tau
    hk._tau = T.mut_tau_plus1
    try:
        out["mutant_tau+1_f64_x"] = rel(hk.read(qs, k, v, 0.999, 256), densec(qs, k, v, 0.999)[0])
    finally:
        hk._tau = orig
    ok &= not (out["mutant_tau+1_f64_x"] <= 1e-6)
    return ok, out


@T.bar("sp_mem_S16384_v2")
def t_mem_v2():
    hk = T.load(T.HOOK, "hook_under_test")
    out, ok = {}, True
    S, g = 16384, 0.999
    for H, D in ((2, 64), (4, 32)):
        qs, k, v, gx = inputs(2, [1.0] * H, [0.0] * H, S, H, D)
        ts = [t.float().cuda().requires_grad_() for t in (qs, k, v)]
        gxc = gx.float().cuda()
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        x = hk.read(*ts, g, 256)
        x.backward(gxc)
        torch.cuda.synchronize()
        pa, pr = torch.cuda.max_memory_allocated() / GiB, torch.cuda.max_memory_reserved() / GiB
        x = x.detach()
        fin = bool(torch.isfinite(x).all()) and all(t.grad is not None and bool(torch.isfinite(t.grad).all()) for t in ts)
        xp = hk.read(ts[0].detach()[:, :, :1024], ts[1].detach()[:, :, :1024], ts[2].detach()[:, :, :1024], g, 256)
        pre = rel(x[:, :, :1024], xp)
        del ts, gxc, xp
        torch.cuda.empty_cache()
        q64, k64, v64 = qs.cuda(), k.cuda(), v.cuda()
        with torch.no_grad():
            x64 = hk.read(q64, k64, v64, g, 256)
        lo = S - 256
        W = R.sparsemax(R._logits_rows(q64, k64, lo, S))
        resid = float((x64[:, :, lo:] - g * (W @ x64) - (1 - g) * v64[:, :, lo:]).abs().max() / x64.abs().max())
        f32_vs_f64 = rel(x, x64)
        row = {"peak_alloc_gib": pa, "peak_reserved_gib": pr, "finite": fin, "prefix_rel": pre,
               "f64_resid_last256": resid, "f32_vs_f64": f32_vs_f64}
        ok &= pa < 6 and pr < 6 and fin and pre <= 1e-6 and resid <= 1e-12 and f32_vs_f64 <= 1e-4
        out[f"H{H}_D{D}"] = row
        del x, q64, k64, v64, x64, W
        torch.cuda.empty_cache()
    qs, k, v, gx = inputs(2, [1.0] * 2, [0.0] * 2, S, 2, 64)
    ts = [t.float().cuda().requires_grad_() for t in (qs, k, v)]
    gxc = gx.float().cuda()
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        R.resolvent(*ts, g, kind="sparsemax", path="densec").backward(gxc)
        torch.cuda.synchronize()
        ctl = {"oom": False, "peak_alloc_gib": torch.cuda.max_memory_allocated() / GiB}
    except torch.OutOfMemoryError:
        ctl = {"oom": True, "peak_alloc_gib_before_oom": torch.cuda.max_memory_allocated() / GiB}
    del ts, gxc
    torch.cuda.empty_cache()
    ctl["fails_bar"] = ctl["oom"] or ctl["peak_alloc_gib"] >= 6
    ok &= ctl["fails_bar"]
    out["control_densec_H2_D64"] = ctl
    return ok, out


if __name__ == "__main__":
    print(json.dumps({"hook": T.HOOK, "sha256": hashlib.sha256(open(T.HOOK, "rb").read()).hexdigest()}), flush=True)
    t_parity_v2()
    t_mem_v2()
    print(json.dumps(T.results), flush=True)
    sys.exit(0 if all(s == "green" for s in T.results.values()) else 1)

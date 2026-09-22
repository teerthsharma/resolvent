"""Chase K0 bars. Registered 2026-09-23 before any measurement (stub RED run first).
Each bar appends {"t":"test","agent":"Chase","name":..,"status":"red"|"green"} to the board.
Exit code non-zero if any bar is red.

BARS
 fwd_softmax_S1024_vs_f64   fused forward (Cameron fs5 reuse, c=256) fp32, S=1024, B1 H8 D64, gamma .99,
                            SSMax logit scale a=1,b=0: max|x - x64| / max|x64| <= 1e-5 (x64 = float64 dense solve)
 fwd_sparsemax_S1024_vs_f64 dense sparsemax forward fp32 same shape: same metric <= 1e-5
 gradcheck_softmax_S64_f64  torch.autograd.gradcheck (float64, S=64, B1 H2 D8) through the adjoint backward,
                            inputs q, k, v and the SSMax parameters a, b: passes
 gradcheck_sparsemax_S64_f64 same for sparsemax (inputs q, k, v, a, b): passes
 fused_bwd_S1024_vs_f64     fused path backward (adjoint blocked trsm + efficient-attention backward) fp32 S=1024:
                            each of dq, dk, dv: max|g - g64| / max|g64| <= 1e-4 (g64 = float64 dense adjoint)
 y1_rebuild                 banded solve of (-Lap + m^2)G = delta, polyfit r in [20,60): |kappa - arccosh(1+m^2/2)| <= 5e-7
                            and round(kappa, 6) == contract section 9 value, m = .1/.5/1
 y2_closed_form             R-RANGE instrument on the Toeplitz head (n 3000, rows 2000..2900 step 100, r in [30,400)):
                            |m_fit - ln(g + (1-g)e^lam)| <= 1e-4 (contract) AND relative <= 1e-4 (Chase), lam .5/1, g .9/.99/.999
 y2_vs_pinned               same m_fit vs the pinned yukawa.out 5-digit values: |m_fit - printed| <= 5e-6
 gamma_rho_lt_1             every softmax head the layer uses (SSMax a in {0,.5,1,2,4,8}, b in {0,2}, S 4096, random q,k,
                            plus ALiBi bias) in fp32: gamma * max_i W_ii < 1 (spectral radius of a triangular W)
 cost_report                forward+backward at S 4096 and 8192 measured (finite) vs one SDPA fwd+bwd; REPORTED, NOT SCORED
                            (R-COST is Cameron's at K4). Written by cost.py rows, checked here for presence only.
BARS ADDED 2026-09-23 after fwd_softmax / fused_bwd came back RED (registered before fs5c / backward_blocked were measured):
 fwd_softmax_S1024_vs_f64_fs5c  fs5 with the complementary-mass pivot (resolvent.fs5c), same shape/metric, SSMax a=1 b=0,
                                inputs seed 0 AND seed 2: max over both <= 1e-5
 fusedc_bwd_S1024_vs_f64        fs5c forward + exact materialized blocked backward (resolvent.backward_blocked),
                                seed 2 inputs / seed 3 grad: each of dq, dk, dv rel <= 1e-4
BAR ADDED after fwd_sparsemax came back RED (registered before densec was measured):
 fwd_sparsemax_S1024_vs_f64_pivot  dense sparsemax fp32 with the complementary-mass pivot (path densec), a=1 b=0,
                                   seeds 0 and 2: max <= 1e-5
BAR ADDED before the graphed path (fusedcg) was measured; the cost rows use it:
 fusedcg_matches_fusedc_S1024      CUDA-graphed fs5c forward + graphed backward_blocked (static buffers) vs eager fusedc,
                                   S=1024, two consecutive calls with DIFFERENT inputs (seeds 2 then 5, so a stale
                                   static buffer is caught): x, dq, dk, dv each max rel diff <= 1e-6
BAR ADDED after breaks.py B2 showed head_mass < 0 on content-bearing heads (registered before head_mass_tail was run):
 y2_tail_instrument                tail-mass instrument rrange.head_mass_tail on the Y2 Toeplitz head (same n, rows, r window):
                                   |m - ln(g + (1-g)e^lam)| <= 1e-4 AND relative <= 1e-4, all six (lam, g)
BAR ADDED before it was run (the training path's backward, not only the dense one, under gradcheck):
 gradcheck_blocked_adjoint_S64_f64 float64 dense forward + resolvent.backward_blocked (c=16, 4 row blocks) as the backward,
                                   torch.autograd.gradcheck on q, k, v, a, b at S=64 (B1 H2 D8, softmax, SSMax): passes
"""
import json, math, os, sys, traceback
import numpy as np

SP = "C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad"
HERE = os.path.dirname(os.path.abspath(__file__))
BOARD = "C:/Users/seal/Desktop/New folder (32)/house-events.jsonl"
PINNED = "C:/Users/seal/Desktop/New folder (32)/tests/foreman/phase_k/instances/yukawa.out"
CONTRACT_Y1 = {0.1: 0.099958, 0.5: 0.494933, 1.0: 0.962424}
sys.path.insert(0, HERE)
sys.path.insert(0, SP)

results = {}


def board(name, status, value):
    with open(BOARD, "a") as f:
        f.write(json.dumps({"t": "test", "agent": "Chase", "name": "chase.k0." + name, "status": status,
                            "value": value}) + "\n")


def bar(name):
    def deco(fn):
        def run():
            try:
                ok, val = fn()
            except Exception as e:
                ok, val = False, "EXC " + type(e).__name__ + ": " + str(e)[:200]
                traceback.print_exc()
            st = "green" if ok else "red"
            print(("GREEN " if ok else "RED ") + name + ": " + json.dumps(val), flush=True)
            results[name] = st
            if os.environ.get("CHASE_BOARD", "1") == "1":
                board(name, st, val)
        run.__name__ = name
        return run
    return deco


def rel(a, b):
    return float((a.double() - b.double()).abs().max() / b.double().abs().max())


def _qkv(S, H=8, D=64, B=1, seed=0, dev="cuda", dtype=None):
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed)
    t = [torch.randn(B, H, S, D, generator=g, dtype=torch.float64) for _ in range(3)]
    return [x.to(dev, dtype or torch.float32) for x in t]


@bar("fwd_softmax_S1024_vs_f64")
def t_fwd_softmax():
    import torch, resolvent as R
    q, k, v = _qkv(1024)
    qs = R.ssmax_q(q, torch.tensor(1.0, device="cuda"), torch.tensor(0.0, device="cuda"))
    with torch.no_grad():
        x = R.resolvent(qs, k, v, 0.99, kind="softmax", path="fused")
        ref = R.resolvent(qs.double(), k.double(), v.double(), 0.99, kind="softmax", path="dense")
    e = rel(x, ref)
    return e <= 1e-5, {"rel_err": e}


@bar("fwd_sparsemax_S1024_vs_f64")
def t_fwd_sparsemax():
    import torch, resolvent as R
    q, k, v = _qkv(1024)
    qs = R.ssmax_q(q, torch.tensor(1.0, device="cuda"), torch.tensor(0.0, device="cuda"))
    with torch.no_grad():
        x = R.resolvent(qs, k, v, 0.99, kind="sparsemax", path="dense")
        ref = R.resolvent(qs.double(), k.double(), v.double(), 0.99, kind="sparsemax", path="dense")
    e = rel(x, ref)
    return e <= 1e-5, {"rel_err": e}


def _gc(kind):
    import torch, resolvent as R
    q, k, v = _qkv(64, H=2, D=8, dev="cpu", dtype=torch.float64, seed=1)
    a = torch.tensor(0.7, dtype=torch.float64)
    b = torch.tensor(0.3, dtype=torch.float64)
    ins = [t.clone().requires_grad_() for t in (q, k, v, a, b)]
    f = lambda q, k, v, a, b: R.resolvent(R.ssmax_q(q, a, b), k, v, 0.99, kind=kind, path="dense")
    ok = torch.autograd.gradcheck(f, ins, eps=1e-6, atol=1e-6, rtol=1e-4, raise_exception=False)
    return bool(ok), {"gradcheck": bool(ok), "S": 64, "dtype": "float64"}


@bar("gradcheck_softmax_S64_f64")
def t_gc_soft():
    return _gc("softmax")


@bar("gradcheck_sparsemax_S64_f64")
def t_gc_sparse():
    return _gc("sparsemax")


@bar("fused_bwd_S1024_vs_f64")
def t_fused_bwd():
    import torch, resolvent as R
    q, k, v = _qkv(1024, seed=2)
    qs = R.ssmax_q(q, torch.tensor(1.0, device="cuda"), torch.tensor(0.0, device="cuda"))
    gout = _qkv(1024, seed=3)[0]
    ins32 = [t.clone().requires_grad_() for t in (qs, k, v)]
    R.resolvent(*ins32, 0.99, kind="softmax", path="fused").backward(gout)
    ins64 = [t.double().clone().requires_grad_() for t in (qs, k, v)]
    R.resolvent(*ins64, 0.99, kind="softmax", path="dense").backward(gout.double())
    errs = {n: rel(a.grad, b.grad) for n, a, b in zip(("dq", "dk", "dv"), ins32, ins64)}
    return max(errs.values()) <= 1e-4, errs


@bar("y1_rebuild")
def t_y1():
    import rrange
    out = {}
    ok = True
    for m in (0.1, 0.5, 1.0):
        kap = rrange.y1_decay(m)
        pred = float(np.arccosh(1 + m * m / 2))
        out[str(m)] = {"kappa": kap, "pred": pred, "abs": abs(kap - pred), "contract": CONTRACT_Y1[m]}
        ok &= abs(kap - pred) <= 5e-7 and round(kap, 6) == CONTRACT_Y1[m]
    return ok, out


def _y2_rows():
    import rrange
    rows = {}
    for lam in (0.5, 1.0):
        for g in (0.9, 0.99, 0.999):
            m = rrange.toeplitz_mass(lam, g)
            pred = float(np.log(g + (1 - g) * np.exp(lam)))
            rows[(lam, g)] = (m, pred)
    return rows


Y2 = {}


@bar("y2_closed_form")
def t_y2():
    Y2.update(_y2_rows())
    out, ok = {}, True
    for (lam, g), (m, pred) in Y2.items():
        a, r = abs(m - pred), abs(m - pred) / pred
        out[f"{lam}/{g}"] = {"m_fit": m, "pred": pred, "abs": a, "rel": r}
        ok &= a <= 1e-4 and r <= 1e-4
    return ok, out


@bar("y2_vs_pinned")
def t_y2_pinned():
    import re
    if not Y2:
        Y2.update(_y2_rows())
    pinned = {}
    for line in open(PINNED):
        mm = re.match(r"Y2\s+bare mass ([\d.]+) .*gamma ([\d.]+): dressed mass measured ([\d.]+)", line)
        if mm:
            pinned[(float(mm.group(1)), float(mm.group(2)))] = float(mm.group(3))
    out, ok = {}, len(pinned) == 6
    for key, (m, _) in Y2.items():
        d = abs(m - pinned[key])
        out[f"{key[0]}/{key[1]}"] = {"m_fit": m, "pinned": pinned[key], "abs": d}
        ok &= d <= 5e-6
    return ok, out


@bar("gamma_rho_lt_1")
def t_rho():
    import torch, resolvent as R
    q, k, _ = _qkv(4096, seed=4)
    worst = 0.0
    for alibi in (False, True):
        for a in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
            for b in (0.0, 2.0):
                qs = R.ssmax_q(q, torch.tensor(a, device="cuda"), torch.tensor(b, device="cuda"))
                d = R.diag_weights(qs, k, alibi=alibi)
                worst = max(worst, 0.99 * float(d.max()))
    return worst < 1.0, {"max_gamma_Wii_at_gamma_.99": worst}


def _stub():
    if os.environ.get("CHASE_STUB") == "1":
        raise NotImplementedError("stub")


@bar("fwd_softmax_S1024_vs_f64_fs5c")
def t_fwd_c():
    import torch, resolvent as R
    _stub()
    out = {}
    for seed in (0, 2):
        q, k, v = _qkv(1024, seed=seed)
        qs = R.ssmax_q(q, torch.tensor(1.0, device="cuda"), torch.tensor(0.0, device="cuda"))
        with torch.no_grad():
            x = R.resolvent(qs, k, v, 0.99, path="fusedc")
            ref = R.resolvent(qs.double(), k.double(), v.double(), 0.99, path="dense")
        out["seed%d" % seed] = rel(x, ref)
    return max(out.values()) <= 1e-5, out


@bar("fusedc_bwd_S1024_vs_f64")
def t_bwd_c():
    import torch, resolvent as R
    _stub()
    q, k, v = _qkv(1024, seed=2)
    qs = R.ssmax_q(q, torch.tensor(1.0, device="cuda"), torch.tensor(0.0, device="cuda"))
    gout = _qkv(1024, seed=3)[0]
    ins32 = [t.clone().requires_grad_() for t in (qs, k, v)]
    R.resolvent(*ins32, 0.99, path="fusedc").backward(gout)
    ins64 = [t.double().clone().requires_grad_() for t in (qs, k, v)]
    R.resolvent(*ins64, 0.99, path="dense").backward(gout.double())
    errs = {n: rel(a.grad, b.grad) for n, a, b in zip(("dq", "dk", "dv"), ins32, ins64)}
    return max(errs.values()) <= 1e-4, errs


@bar("fwd_sparsemax_S1024_vs_f64_pivot")
def t_fwd_sp_c():
    import torch, resolvent as R
    _stub()
    out = {}
    for seed in (0, 2):
        q, k, v = _qkv(1024, seed=seed)
        qs = R.ssmax_q(q, torch.tensor(1.0, device="cuda"), torch.tensor(0.0, device="cuda"))
        with torch.no_grad():
            x = R.resolvent(qs, k, v, 0.99, kind="sparsemax", path="densec")
            ref = R.resolvent(qs.double(), k.double(), v.double(), 0.99, kind="sparsemax", path="dense")
        out["seed%d" % seed] = rel(x, ref)
    return max(out.values()) <= 1e-5, out


@bar("fusedcg_matches_fusedc_S1024")
def t_cg():
    import torch, resolvent as R
    _stub()
    out = {}
    for seed in (2, 5):
        q, k, v = _qkv(1024, seed=seed)
        qs = R.ssmax_q(q, torch.tensor(1.0, device="cuda"), torch.tensor(0.0, device="cuda"))
        gout = _qkv(1024, seed=seed + 1)[0]
        res = []
        for path in ("fusedcg", "fusedc"):
            ins = [t.clone().requires_grad_() for t in (qs, k, v)]
            x = R.resolvent(*ins, 0.99, path=path); x.backward(gout)
            res.append([x.detach()] + [t.grad for t in ins])
        for n, a, b in zip(("x", "dq", "dk", "dv"), *res):
            out["seed%d_%s" % (seed, n)] = rel(a, b)
    return max(out.values()) <= 1e-6, out


@bar("y2_tail_instrument")
def t_y2_tail():
    import rrange
    _stub()
    out, ok = {}, True
    for lam in (0.5, 1.0):
        for g in (0.9, 0.99, 0.999):
            m = rrange.toeplitz_mass_tail(lam, g)
            pred = float(np.log(g + (1 - g) * np.exp(lam)))
            out[f"{lam}/{g}"] = {"m_fit": m, "pred": pred, "abs": abs(m - pred), "rel": abs(m - pred) / pred}
            ok &= abs(m - pred) <= 1e-4 and abs(m - pred) / pred <= 1e-4
    return ok, out


@bar("gradcheck_blocked_adjoint_S64_f64")
def t_gc_blocked():
    import torch, resolvent as R
    _stub()

    class F64(torch.autograd.Function):
        @staticmethod
        def forward(ctx, qs, k, v):
            x = R._Dense.forward(ctx, qs, k, v, 0.99, "softmax")
            ctx.save_for_backward(qs, k, x)
            return x

        @staticmethod
        def backward(ctx, gx):
            qs, k, x = ctx.saved_tensors
            return R.backward_blocked(qs, k, x, gx.contiguous(), 0.99, c=16)

    q, k, v = _qkv(64, H=2, D=8, dev="cpu", dtype=torch.float64, seed=1)
    ins = [t.clone().requires_grad_() for t in (q, k, v, torch.tensor(0.7, dtype=torch.float64),
                                                 torch.tensor(0.3, dtype=torch.float64))]
    f = lambda q, k, v, a, b: F64.apply(R.ssmax_q(q, a, b), k, v)
    ok = torch.autograd.gradcheck(f, ins, eps=1e-6, atol=1e-6, rtol=1e-4, raise_exception=False)
    return bool(ok), {"gradcheck": bool(ok), "S": 64, "c": 16, "dtype": "float64"}


@bar("cost_report")
def t_cost():
    p = os.path.join(HERE, "cost_rows.jsonl")
    rows = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
    got = {r["S"]: r for r in rows if math.isfinite(r.get("ratio_fwd_bwd", float("nan")))}
    return {4096, 8192} <= set(got), {S: got[S]["ratio_fwd_bwd"] for S in got}


if __name__ == "__main__":
    only = sys.argv[1:]
    gpu = [t_fwd_softmax, t_fwd_sparsemax, t_fused_bwd, t_rho, t_fwd_c, t_bwd_c, t_fwd_sp_c, t_cg]
    tests = [t_fwd_softmax, t_fwd_sparsemax, t_gc_soft, t_gc_sparse, t_fused_bwd, t_y1, t_y2, t_y2_pinned,
             t_rho, t_fwd_c, t_bwd_c, t_fwd_sp_c, t_cg, t_y2_tail, t_gc_blocked, t_cost]
    tests = [t for t in tests if not only or t.__name__ in only]
    if any(t in gpu for t in tests) and os.environ.get("CHASE_NOPOLL") != "1":
        import design4x5
        if not design4x5.poll_until_free(timeout_s=600):
            print("RED gpu: card not free in 600 s"); sys.exit(2)
    for t in tests:
        t()
    reds = [n for n, s in results.items() if s == "red"]
    print("SUMMARY", json.dumps(results))
    sys.exit(1 if reds else 0)

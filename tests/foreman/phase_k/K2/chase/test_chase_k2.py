"""Chase K2 bars for the blocked sparsemax resolvent hook (K2.1 arm f_R_sp). Registered 2026-09-23 before any
measurement; RED run first against resolvent_sp_stub.py (CHASE_HOOK=...stub). Each bar appends
{"t":"test","agent":"Chase","name":"chase.k2.<bar>","status":..} to the board (CHASE_BOARD=0 disables).
Exit code 1 if any bar is red. The hook under test is $CHASE_HOOK (default resolvent_sp.py next to this file).
Reference: K0 densec = R.resolvent(qs, k, v, g, kind="sparsemax", path="densec") (dense sparsemax W, pivot by the
complementary mass, gradchecked in K0). rel(h, r) = max|h - r| / max|r|. TF32 off.

Shape: rdepth.py's fR resolvent layer is d_model 128, 2 heads of 64 (D = 128, H = 2 in rdepth.py); eval runs B 1.

BARS
 sp_fwd_parity_S1024   read(qs, k, v, g, c=256) vs densec float64 (CPU). B1 H2 D64 S1024. Inputs: in1 = seed 0, a = 1, b = 0;
                       in2 = seed 1, a = (1.5, .5), b = (.5, -.3); q, k, v, gx ~ N(0,1) drawn in float64, rounded to fp32;
                       qs = q * (a ln(i+1) + b). g in {0.5, 0.999}. Rows (every row must pass):
                         f64: hook in float64 on CPU, x rel <= 1e-10;
                         f32: hook in fp32 on CUDA vs densec f64 on the same fp32-rounded inputs: x rel <= 1e-5,
                              and dq, dk, dv (backward of <x, gx>) rel <= 1e-4 against densec's float64 gradients;
                         pad: S 1000, in1, g .999, f64, x rel <= 1e-10 (the zero-pad path);
                         module: ResolventAttention(128, 2) on CUDA (a = 1, b = 0 at init), x ~ N(0,1) B1 S1024 seed 3,
                              weights ~ N(0,1)/sqrt(128): output under torch.autocast(cuda, bfloat16) equals output with
                              autocast off bitwise, is fp32, and y rel <= 1e-5 vs a float64 densec layer; dx, dw_qkv, dw_out,
                              da, db rel <= 1e-4.
                       Mutant: _tau replaced by a support-size+1 version; the f64 in1 g .999 row must then exceed 1e-6.
 sp_support_S1024      weights_blocked(qs, k, c=256) (the forward's per-block weights) vs R.weights(qs, k, "sparsemax") in
                       float64 on in1 and in2: the sets {W > 0} are equal exactly; every row sums to 1 within 1e-12;
                       nontrivial: some row has support >= 2 and some causal entry is an exact 0.0.
                       Mutant: _tau(+1) must make the sets differ on in1.
 sp_gradcheck_S64_f64  torch.autograd.gradcheck (eps 1e-6, atol 1e-6, rtol 1e-4) of layer(x, wq, wo, a, b, 2, g=g, c=16) in
                       float64 on CPU; B1, d_model 16, 2 heads; x ~ N(0,1) seed 7, wq, wo ~ N(0,1)/4 (same generator);
                       a = (.7, 1.1), b = (.3, -.2); rows (S 64: 4 blocks, S 60: padded) x (g .5, g .999). Precondition per
                       row: min causal |z - tau| > 1e-4 (reference arithmetic), so eps cannot cross a kink. All rows pass,
                       AND each mutant of _jac fails at S 64 g .999: (a) the support-mean term dropped, (b) support = W >= 0.
 sp_mem_S16384         read() forward + backward, fp32 CUDA, g .999, c 256, B1 S16384 at (H2 D64) [rdepth] and (H4 D32);
                       q, k, v, gx ~ N(0,1) seed 2, a = 1, b = 0. Per shape: after reset_peak_memory_stats (inputs already
                       allocated), max_memory_allocated < 6 GiB and max_memory_reserved < 6 GiB; x, dq, dk, dv finite;
                       prefix: x[..., :1024, :] vs read() at S 1024 on the same prefix, rel <= 1e-6; residual on rows
                       16128..16383: max|x_i - g sum_j W_ij x_j - (1-g) v_i| / max|x| <= 1e-5, W from R.sparsemax on float64
                       logits of those rows (R._logits_rows). Control (must fail the memory bar): densec fwd + bwd at the
                       H2 D64 shape raises OutOfMemoryError or peaks >= 6 GiB.
 sp_rdepth_smoke_50    repo tests/foreman/phase_k/K1/cameron/rdepth.py --arm fR --hook $CHASE_HOOK:ResolventAttention --steps 50
                       --seed 0 --lr 0.003 --warmup 200 --bed kpf --emb frozen --par_init orth --eval_beds 8, two runs:
                         ga:   --gamma_sched 0.5:2000,0.9:4000,0.99:6000,0.999 --eval_ns 1024,16384 (K1-b schedule);
                         g999: no schedule (the hook's own gamma), --eval_ns 1024.
                       Each: rc 0; step-1 and step-50 losses in log.jsonl finite and loss(50) < loss(1); no abort record;
                       config hook_sha256 = sha256 of $CHASE_HOOK; every eval acc finite; result.json peak_mem_gib < 6;
                       model.pt holds attn.a / attn.b only under blocks.3 and max|a - 1| + max|b| > 0.
REPORT (not scored; no board line): cost of K1 FR read (repo K1/chase/resolvent_hook.py read(), sha 3855288d) and this read,
fp32 CUDA B1 H2 D64 g .999, S 4096 and 16384: fwd+bwd and fwd-only ms, median of 5 after 2 warm-up calls.
"""
import hashlib, importlib.util, json, math, os, shutil, statistics, subprocess, sys, time, traceback
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "C:/Users/seal/Desktop/New folder (32)"
BOARD = REPO + "/house-events.jsonl"
RDEPTH = REPO + "/tests/foreman/phase_k/K1/cameron/rdepth.py"
FRHOOK = REPO + "/tests/foreman/phase_k/K1/chase/resolvent_hook.py"
HOOK = os.environ.get("CHASE_HOOK", os.path.join(HERE, "resolvent_sp.py"))
TAG = "stub" if "stub" in os.path.basename(HOOK) else "hook"
sys.path.insert(0, REPO + "/tests/foreman/phase_k/K0/chase")
import torch
import torch.nn.functional as F
import resolvent as R                                       # K0 lane (densec)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
GiB = 2 ** 30
results = {}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def bar(name):
    def deco(fn):
        def run():
            try:
                ok, val = fn()
            except Exception as e:
                ok, val = False, "EXC " + type(e).__name__ + ": " + str(e)[:300]
                traceback.print_exc()
            st = "green" if ok else "red"
            print(("GREEN " if ok else "RED ") + name + ": " + json.dumps(val), flush=True)
            results[name] = st
            if os.environ.get("CHASE_BOARD", "1") == "1":
                with open(BOARD, "a") as f:
                    f.write(json.dumps({"t": "test", "agent": "Chase", "status": st, "name": "chase.k2." + name}) + "\n")
        run.__name__ = name
        return run
    return deco


def rel(a, b):
    return float((a.double().cpu() - b.double().cpu()).abs().max() / b.double().cpu().abs().max())


def scale(a, b, S, dt):
    a, b = torch.as_tensor(a, dtype=dt), torch.as_tensor(b, dtype=dt)
    return a[:, None] * torch.log(torch.arange(1, S + 1, dtype=dt))[None, :] + b[:, None]


def inputs(seed, a, b, S, H=2, D=64):
    """fp32-rounded q, k, v, gx (float64 tensors holding fp32 values) and qs = q * scale (formed in float64)."""
    g = torch.Generator().manual_seed(seed)
    q, k, v, gx = (torch.randn(1, H, S, D, generator=g, dtype=torch.float64).float().double() for _ in range(4))
    qs = (q * scale(a, b, S, torch.float64)[None, :, :, None]).float().double()
    return qs, k, v, gx


INS = {"in1": (0, (1.0, 1.0), (0.0, 0.0)), "in2": (1, (1.5, 0.5), (0.5, -0.3))}


def densec(qs, k, v, g, gx=None):
    ts = [t.clone().requires_grad_(gx is not None) for t in (qs, k, v)]
    x = R.resolvent(*ts, g, kind="sparsemax", path="densec")
    if gx is None:
        return x.detach(), None
    x.backward(gx)
    return x.detach(), [t.grad for t in ts]


def mut_tau_plus1(z):
    """_tau with support size + 1 (a wrong threshold): the parity and support mutant."""
    zs, _ = torch.sort(z, dim=-1, descending=True)
    kk = torch.arange(1, z.shape[-1] + 1, device=z.device, dtype=z.dtype)
    cs = zs.cumsum(-1)
    ksz = ((1 + kk * zs) > cs).sum(-1, keepdim=True).clamp(max=z.shape[-1] - 1) + 1
    return (cs.gather(-1, ksz - 1) - 1) / ksz.to(z.dtype)


@bar("sp_fwd_parity_S1024")
def t_parity():
    hk = load(HOOK, "hook_under_test")
    out, ok = {}, True
    for nm, (seed, a, b) in INS.items():
        qs, k, v, gx = inputs(seed, a, b, 1024)
        for g in (0.5, 0.999):
            xr, gr = densec(qs, k, v, g, gx)
            x64 = hk.read(qs, k, v, g, 256)
            r64 = rel(x64, xr)
            ts = [t.float().cuda().requires_grad_() for t in (qs, k, v)]
            x32 = hk.read(*ts, g, 256)
            x32.backward(gx.float().cuda())
            row = {"f64_x": r64, "f32_x": rel(x32.detach(), xr),
                   **{"f32_" + n: (rel(t.grad, r) if t.grad is not None else float("inf")) for n, t, r in zip(("dq", "dk", "dv"), ts, gr)}}
            ok &= row["f64_x"] <= 1e-10 and row["f32_x"] <= 1e-5 and all(row["f32_" + n] <= 1e-4 for n in ("dq", "dk", "dv"))
            out[f"{nm}_g{g}"] = row
    qs, k, v, _ = inputs(0, (1.0, 1.0), (0.0, 0.0), 1000)
    out["pad_S1000_f64_x"] = rel(hk.read(qs, k, v, 0.999, 256), densec(qs, k, v, 0.999)[0])
    ok &= out["pad_S1000_f64_x"] <= 1e-10
    # module row
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
    # mutant
    qs, k, v, _ = inputs(0, (1.0, 1.0), (0.0, 0.0), 1024)
    orig = hk._tau
    hk._tau = mut_tau_plus1
    try:
        out["mutant_tau+1_f64_x"] = rel(hk.read(qs, k, v, 0.999, 256), densec(qs, k, v, 0.999)[0])
    finally:
        hk._tau = orig
    ok &= out["mutant_tau+1_f64_x"] > 1e-6
    return ok, out


@bar("sp_support_S1024")
def t_support():
    hk = load(HOOK, "hook_under_test")
    out, ok = {}, True
    tri = R._mask(1024, "cpu")
    for nm, (seed, a, b) in INS.items():
        qs, k, _, _ = inputs(seed, a, b, 1024)
        Wb = hk.weights_blocked(qs, k, 256)
        Wr = R.weights(qs, k, "sparsemax")
        same = bool(torch.equal(Wb > 0, Wr > 0))
        rs = float((Wb.sum(-1) - 1).abs().max())
        nsupp = (Wb > 0).sum(-1)
        causal_zero = int(((Wb == 0) & ~tri).sum())
        out[nm] = {"sets_equal": same, "rowsum_err": rs, "max_support": int(nsupp.max()), "rows_support_ge2": int((nsupp >= 2).sum()),
                   "causal_exact_zeros": causal_zero, "nnz": int((Wb > 0).sum())}
        ok &= same and rs <= 1e-12 and int(nsupp.max()) >= 2 and causal_zero > 0
    qs, k, _, _ = inputs(0, (1.0, 1.0), (0.0, 0.0), 1024)
    orig = hk._tau
    hk._tau = mut_tau_plus1
    try:
        out["mutant_tau+1_sets_equal"] = bool(torch.equal(hk.weights_blocked(qs, k, 256) > 0, R.weights(qs, k, "sparsemax") > 0))
    finally:
        hk._tau = orig
    ok &= not out["mutant_tau+1_sets_equal"]
    return ok, out


def _margin(x, wq, a, b, H):
    B, T, Dm = x.shape
    q, k, _ = F.linear(x, wq).split(Dm, 2)
    q, k = (t.view(B, T, H, Dm // H).transpose(1, 2) for t in (q, k))
    z = R._logits(q * scale(a, b, T, x.dtype)[None, :, :, None], k)
    W = R.sparsemax(z)
    tau = torch.where(W > 0, z - W, torch.full_like(z, float("nan"))).nanmean(-1, keepdim=True)
    return float((z - tau).abs().masked_fill(R._mask(T, "cpu"), float("inf")).min())


@bar("sp_gradcheck_S64_f64")
def t_gradcheck():
    hk = load(HOOK, "hook_under_test")
    a = torch.tensor([0.7, 1.1], dtype=torch.float64)
    b = torch.tensor([0.3, -0.2], dtype=torch.float64)
    out, ok = {}, True

    def case(S, g):
        gen = torch.Generator().manual_seed(7)
        x = torch.randn(1, S, 16, generator=gen, dtype=torch.float64)
        wq = torch.randn(48, 16, generator=gen, dtype=torch.float64) / 4
        wo = torch.randn(16, 16, generator=gen, dtype=torch.float64) / 4
        ins = [t.clone().requires_grad_() for t in (x, wq, wo, a, b)]
        f = lambda x, wq, wo, a, b: hk.layer(x, wq, wo, a, b, 2, g=g, c=16)
        return _margin(x, wq, a, b, 2), lambda: bool(torch.autograd.gradcheck(f, ins, eps=1e-6, atol=1e-6, rtol=1e-4,
                                                                               raise_exception=False))
    for S in (64, 60):
        for g in (0.5, 0.999):
            m, gc = case(S, g)
            passed = gc()
            out[f"S{S}_g{g}"] = {"margin": m, "gradcheck": passed}
            ok &= m > 1e-4 and passed
    muts = {"a_no_support_mean": lambda W, dW: (W > 0).to(dW.dtype) * dW,
            "b_support_ge0": lambda W, dW: (lambda s: s * (dW - (dW * s).sum(-1, keepdim=True) / s.sum(-1, keepdim=True)))((W >= 0).to(dW.dtype))}
    orig = hk._jac
    _, gc = case(64, 0.999)
    for n, mf in muts.items():
        hk._jac = mf
        try:
            out["mutant_" + n + "_passes"] = gc()
        finally:
            hk._jac = orig
        ok &= not out["mutant_" + n + "_passes"]
    return ok, out


@bar("sp_mem_S16384")
def t_mem():
    hk = load(HOOK, "hook_under_test")
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
        lo = S - 256
        W = R.sparsemax(R._logits_rows(qs.cuda(), k.cuda(), lo, S))                   # float64, rows lo..S-1
        x64 = x.double()
        res = x64[:, :, lo:] - g * (W @ x64) - (1 - g) * v.cuda()[:, :, lo:]
        resid = float(res.abs().max() / x64.abs().max())
        row = {"peak_alloc_gib": pa, "peak_reserved_gib": pr, "finite": fin, "prefix_rel": pre, "resid_last256": resid}
        ok &= pa < 6 and pr < 6 and fin and pre <= 1e-6 and resid <= 1e-5
        out[f"H{H}_D{D}"] = row
        del ts, gxc, x, xp, W, x64, res
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


def _smoke(name, extra):
    d = os.path.join(HERE, "runs", f"{TAG}_smoke_{name}")
    shutil.rmtree(d, ignore_errors=True)
    cmd = [sys.executable, RDEPTH, "--arm", "fR", "--hook", HOOK + ":ResolventAttention", "--steps", "50", "--seed", "0",
           "--lr", "0.003", "--warmup", "200", "--bed", "kpf", "--emb", "frozen", "--par_init", "orth", "--eval_beds", "8",
           "--out", d] + extra
    p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    os.makedirs(d, exist_ok=True)
    with open(d + ".stdout.log", "w") as f:
        f.write(p.stdout)
    with open(d + ".stderr.log", "w") as f:
        f.write(p.stderr)
    recs = [json.loads(l) for l in open(os.path.join(d, "log.jsonl"))] if os.path.exists(os.path.join(d, "log.jsonl")) else []
    st = {r["step"]: r["loss"] for r in recs if r.get("t") == "step"}
    cfg = [r for r in recs if r.get("t") == "config"]
    val = {"rc": p.returncode, "loss1": st.get(1), "loss50": st.get(50), "abort": any(r.get("t") == "abort" for r in recs),
           "hook_sha_ok": bool(cfg) and cfg[0].get("hook_sha256") == hashlib.sha256(open(HOOK, "rb").read()).hexdigest()}
    ok = p.returncode == 0 and val["loss1"] is not None and val["loss50"] is not None and \
        math.isfinite(val["loss1"]) and math.isfinite(val["loss50"]) and val["loss50"] < val["loss1"] and not val["abort"] and val["hook_sha_ok"]
    rp = os.path.join(d, "result.json")
    if os.path.exists(rp):
        res = json.load(open(rp))
        val["eval_acc"] = {n: e["acc"] for n, e in res["eval"].items()}
        val["peak_mem_gib"] = res["peak_mem_gib"]
        val["train_s"] = res["train_s"]
        ok &= all(math.isfinite(x) for x in val["eval_acc"].values()) and res["peak_mem_gib"] < 6
    else:
        ok = False
    mp = os.path.join(d, "model.pt")
    if os.path.exists(mp):
        sd = torch.load(mp, map_location="cpu")
        ka = sorted(x for x in sd if x.endswith(".attn.a"))
        kb = sorted(x for x in sd if x.endswith(".attn.b"))
        val["a_keys"], val["b_keys"] = ka, kb
        if ka == ["blocks.3.attn.a"] and kb == ["blocks.3.attn.b"]:
            val["moved"] = float((sd[ka[0]] - 1).abs().max() + sd[kb[0]].abs().max())
            ok &= val["moved"] > 0
        else:
            ok = False
    else:
        ok = False
    return ok, val


@bar("sp_rdepth_smoke_50")
def t_smoke():
    oka, va = _smoke("ga", ["--gamma_sched", "0.5:2000,0.9:4000,0.99:6000,0.999", "--eval_ns", "1024,16384"])
    okb, vb = _smoke("g999", ["--eval_ns", "1024"])
    return oka and okb, {"ga": va, "g999": vb}


def cost_report():
    hk = load(HOOK, "hook_under_test")
    fr = load(FRHOOK, "k1_fr_hook")
    rep = {"fr_sha256": hashlib.sha256(open(FRHOOK, "rb").read()).hexdigest()}
    for S in (4096, 16384):
        qs, k, v, gx = inputs(4, (1.0, 1.0), (0.0, 0.0), S)
        for nm, fn in (("FR", lambda a, b, c: fr.read(a, b, c, 0.999)), ("SP", lambda a, b, c: hk.read(a, b, c, 0.999, 256))):
            ts = [t.float().cuda().requires_grad_() for t in (qs, k, v)]
            gxc = gx.float().cuda()
            tf, tfb = [], []
            for i in range(7):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                with torch.no_grad():
                    fn(*ts)
                torch.cuda.synchronize(); t1 = time.perf_counter()
                fn(*ts).backward(gxc)
                torch.cuda.synchronize(); t2 = time.perf_counter()
                if i >= 2:
                    tf.append((t1 - t0) * 1e3); tfb.append((t2 - t1) * 1e3)
            rep[f"{nm}_S{S}"] = {"fwd_ms": statistics.median(tf), "fwd_bwd_ms": statistics.median(tfb)}
            del ts, gxc
            torch.cuda.empty_cache()
        rep[f"ratio_SP_over_FR_S{S}"] = {k_: rep[f"SP_S{S}"][k_] / rep[f"FR_S{S}"][k_] for k_ in ("fwd_ms", "fwd_bwd_ms")}
    print("REPORT cost (not scored): " + json.dumps(rep), flush=True)


if __name__ == "__main__":
    print(json.dumps({"hook": HOOK, "sha256": hashlib.sha256(open(HOOK, "rb").read()).hexdigest(), "torch": torch.__version__,
                      "device": torch.cuda.get_device_name(0)}), flush=True)
    only = os.environ.get("CHASE_ONLY")
    for t in (t_parity, t_support, t_gradcheck, t_mem, t_smoke):
        if not only or t.__name__ in only.split(","):
            t()
    if os.environ.get("CHASE_COST", "0") == "1":
        try:
            cost_report()
        except Exception:
            traceback.print_exc()
    print(json.dumps(results), flush=True)
    sys.exit(0 if all(s == "green" for s in results.values()) else 1)

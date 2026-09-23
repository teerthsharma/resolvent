"""Chase K1 hook bars. Registered 2026-09-23 before any measurement; RED run first against the stub
(CHASE_HOOK=resolvent_hook_stub.py). Each bar appends {"t":"test","agent":"Chase","name":"chase.k1.<bar>","status":..}
to the board. Exit code 1 if any bar is red. The hook under test is $CHASE_HOOK (default resolvent_hook.py).

BARS
 hook_parity_S1024     ResolventAttention(d_model 128, 2 heads, D 64 = the R0 resolvent layer) on CUDA, its default
                       training path (fs5c + backward_blocked, graphed, fp32 outside autocast), vs a float64 reference
                       built from K0 resolvent.py (per-head R.ssmax_q + R.resolvent path="dense") with the same
                       fp32-rounded inputs. B 2; x ~ N(0,1); w_qkv, w_out ~ N(0,1)/sqrt(128); a = 1, b = 0 (init);
                       rows: (S 1024, seed 0), (S 1024, seed 1), (S 1000, seed 0: the zero-pad path), each run with
                       autocast off AND inside torch.autocast("cuda", bfloat16) (so a leak of autocast into the layer
                       is caught). Metric max|h - r| / max|r|. Bar: y <= 1e-5 and each of dx, dw_qkv, dw_out, da, db
                       <= 1e-4 on all 6 rows.
 hook_gradcheck_S64_f64  torch.autograd.gradcheck (eps 1e-6, atol 1e-6, rtol 1e-4) of the hook's layer() in float64 on
                       CPU (dense forward + backward_blocked with c = 16, 4 row blocks), inputs x, w_qkv, w_out, a, b;
                       B 1, S 64, d_model 16, 2 heads, a = (.7, 1.1), b = (.3, -.2), g = 0.999: passes, AND each of
                       three mutants of backward_blocked (dq x 1.01, dk x 1.01, dv x 1.01) fails gradcheck.
 hook_train_ladder_R0_50  K0 train_ladder.py --rung R0 --attn $CHASE_HOOK:FR --steps 50 --ctx 1024 --batch 8
                       --eval_batches 2 --seed 1337 --no_poll (bf16 autocast default, FineWeb-Edu shards): exit 0;
                       50 step losses all finite; final eval finite; mean(loss steps 41-50) < mean(loss steps 1-10);
                       the checkpoint has exactly one layer with attn.a / attn.b and it is blocks.3 (L-1 ALiBi + 1
                       resolvent last); that layer's a, b moved from init (max|a - 1| + max|b| > 0).
 hook_deterministic_R0_50  the same command with --deterministic, run twice into fresh dirs: 50/50 step losses equal
                       bitwise, final model state_dicts equal bitwise, and a, b moved from init.
"""
import hashlib, importlib.util, json, math, os, shutil, subprocess, sys, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = "C:/Users/seal/Desktop/New folder (32)"
BOARD = REPO + "/house-events.jsonl"
TLPATH = REPO + "/tests/foreman/phase_k/K0/wilson/train_ladder.py"
HOOK = os.environ.get("CHASE_HOOK", os.path.join(HERE, "resolvent_hook.py"))
TAG = "stub" if "stub" in os.path.basename(HOOK) else "hook"
results = {}


def load_hook():
    spec = importlib.util.spec_from_file_location("hook_under_test", HOOK)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


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
                with open(BOARD, "a") as f:
                    f.write(json.dumps({"t": "test", "agent": "Chase", "status": st, "name": "chase.k1." + name}) + "\n")
        run.__name__ = name
        return run
    return deco


def rel(a, b):
    return float((a.double() - b.double()).abs().max() / b.double().abs().max())


def ref64(x, w_qkv, w_out, a, b, H):
    import torch, torch.nn.functional as F
    sys.path.insert(0, REPO + "/tests/foreman/phase_k/K0/chase")
    import resolvent as R
    B, T, Dm = x.shape
    q, k, v = F.linear(x, w_qkv).split(Dm, 2)
    q, k, v = (t.view(B, T, H, Dm // H).transpose(1, 2) for t in (q, k, v))
    qs = torch.stack([R.ssmax_q(q[:, h], a[h], b[h]) for h in range(H)], 1)
    y = R.resolvent(qs, k, v, 0.999, kind="softmax", path="dense")
    return F.linear(y.transpose(1, 2).reshape(B, T, Dm), w_out)


@bar("hook_parity_S1024")
def t_parity():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    hk = load_hook()
    out, ok = {}, True
    for S, seed in ((1024, 0), (1024, 1), (1000, 0)):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(2, S, 128, generator=g, dtype=torch.float64).float()
        wq = (torch.randn(384, 128, generator=g, dtype=torch.float64) / math.sqrt(128)).float()
        wo = (torch.randn(128, 128, generator=g, dtype=torch.float64) / math.sqrt(128)).float()
        gy = torch.randn(2, S, 128, generator=g, dtype=torch.float64).float()
        ins64 = [t.double().cuda().requires_grad_() for t in (x, wq, wo, torch.ones(2), torch.zeros(2))]
        y64 = ref64(*ins64, 2)
        y64.backward(gy.double().cuda())
        for amp in (False, True):
            m = hk.ResolventAttention(128, 2).cuda()
            with torch.no_grad():
                m.qkv.weight.copy_(wq); m.out.weight.copy_(wo)
            xi = x.cuda().requires_grad_()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                y = m(xi)
            y.float().backward(gy.cuda())
            row = {"y": rel(y.detach(), y64.detach())}
            for n, gh, i in (("dx", xi.grad, 0), ("dw_qkv", m.qkv.weight.grad, 1), ("dw_out", m.out.weight.grad, 2),
                             ("da", m.a.grad, 3), ("db", m.b.grad, 4)):
                row[n] = rel(gh, ins64[i].grad) if gh is not None else float("inf")
            ok &= row["y"] <= 1e-5 and all(row[n] <= 1e-4 for n in ("dx", "dw_qkv", "dw_out", "da", "db"))
            out[f"S{S}_seed{seed}_amp{int(amp)}"] = row
    return ok, out


@bar("hook_gradcheck_S64_f64")
def t_gradcheck():
    import torch
    hk = load_hook()
    g = torch.Generator().manual_seed(7)
    x = torch.randn(1, 64, 16, generator=g, dtype=torch.float64)
    wq = torch.randn(48, 16, generator=g, dtype=torch.float64) / 4
    wo = torch.randn(16, 16, generator=g, dtype=torch.float64) / 4
    a = torch.tensor([0.7, 1.1], dtype=torch.float64)
    b = torch.tensor([0.3, -0.2], dtype=torch.float64)
    ins = [t.clone().requires_grad_() for t in (x, wq, wo, a, b)]
    f = lambda x, wq, wo, a, b: hk.layer(x, wq, wo, a, b, 2, c=16)
    gc = lambda: bool(torch.autograd.gradcheck(f, ins, eps=1e-6, atol=1e-6, rtol=1e-4, raise_exception=False))
    base = gc()
    orig = hk.backward_blocked
    muts = {}
    for i, n in enumerate(("dq", "dk", "dv")):
        def mut(*args, i=i, **kw):
            gr = list(orig(*args, **kw))
            gr[i] = gr[i] * 1.01
            return tuple(gr)
        hk.backward_blocked = mut
        try:
            muts[n + "x1.01_passes"] = gc()
        finally:
            hk.backward_blocked = orig
    return base and not any(muts.values()), {"gradcheck": base, **muts}


def _train(name, extra):
    d = os.path.join(HERE, "runs", f"{TAG}_{name}")
    shutil.rmtree(d, ignore_errors=True)
    cmd = [sys.executable, TLPATH, "--rung", "R0", "--attn", HOOK + ":FR", "--steps", "50", "--ctx", "1024",
           "--batch", "8", "--eval_batches", "2", "--seed", "1337", "--no_poll", "--out_dir", d] + extra
    p = subprocess.run(cmd, capture_output=True, text=True)
    with open(d + ".stderr.log", "w") as f:
        f.write(p.stderr)
    recs = [json.loads(l) for l in open(os.path.join(d, "log.jsonl"))] if os.path.exists(os.path.join(d, "log.jsonl")) else []
    return p.returncode, recs, d


def _ckpt(d):
    import torch
    return torch.load(os.path.join(d, "ckpt.pt"), map_location="cpu", weights_only=False)["model"]


def _moved(sd):
    ka = sorted(k for k in sd if k.endswith(".attn.a"))
    kb = sorted(k for k in sd if k.endswith(".attn.b"))
    if ka != ["blocks.3.attn.a"] or kb != ["blocks.3.attn.b"]:
        return False, {"a_keys": ka, "b_keys": kb}
    mv = float((sd[ka[0]] - 1).abs().max() + sd[kb[0]].abs().max())
    return mv > 0, {"a": sd[ka[0]].tolist(), "b": sd[kb[0]].tolist(), "moved": mv}


@bar("hook_train_ladder_R0_50")
def t_train():
    rc, recs, d = _train("default", [])
    steps = [r["loss"] for r in recs if r.get("t") == "step"]
    ev = [r["val_loss"] for r in recs if r.get("t") == "eval"]
    cfg = [r for r in recs if r.get("t") == "config"]
    val = {"rc": rc, "n_steps": len(steps), "first10": sum(steps[:10]) / 10 if len(steps) >= 10 else None,
           "last10": sum(steps[-10:]) / 10 if len(steps) >= 10 else None, "eval": ev,
           "attn_sha256": cfg[0]["attn_sha256"] if cfg else None}
    ok = rc == 0 and len(steps) == 50 and all(math.isfinite(s) for s in steps) and len(ev) >= 1 \
        and all(math.isfinite(e) for e in ev) and val["last10"] < val["first10"]
    if rc == 0:
        mok, mv = _moved(_ckpt(d))
        ok &= mok
        val.update(mv)
    else:
        ok = False
    return ok, val


@bar("hook_deterministic_R0_50")
def t_det():
    import torch
    runs = [_train(n, ["--deterministic"]) for n in ("detA", "detB")]
    if any(rc != 0 for rc, _, _ in runs):
        return False, {"rc": [rc for rc, _, _ in runs]}
    L = [[r["loss"] for r in recs if r.get("t") == "step"] for _, recs, _ in runs]
    sA, sB = _ckpt(runs[0][2]), _ckpt(runs[1][2])
    same_sd = set(sA) == set(sB) and all(torch.equal(sA[k], sB[k]) for k in sA)
    mok, mv = _moved(sA)
    ok = len(L[0]) == 50 and L[0] == L[1] and same_sd and mok
    return ok, {"n": [len(l) for l in L], "losses_equal": L[0] == L[1], "state_dict_equal": same_sd,
                "loss50": [l[-1] if l else None for l in L], **mv}


if __name__ == "__main__":
    only = sys.argv[1:]
    tests = [t_parity, t_gradcheck, t_train, t_det]
    tests = [t for t in tests if not only or t.__name__ in only]
    print("HOOK", HOOK, "sha256", hashlib.sha256(open(HOOK, "rb").read()).hexdigest(), flush=True)
    gpu = [t for t in tests if t in (t_parity, t_train, t_det)]
    for t in tests:
        if t not in gpu:
            t()
    if gpu:
        from gpulock import gpu_lock
        with gpu_lock("chase.k1 hook bars " + TAG):
            for t in gpu:
                t()
    print("SUMMARY", json.dumps(results))
    sys.exit(1 if any(s == "red" for s in results.values()) else 0)

# R-JEPA / Cameron round 2: fewer predictor calls at Bayes quality (cascade / prune) + LIN1 rebind on new seeds.
# Bar: BAR.md in this folder (registered before any run). Bed and D-JEPA-spec operator reused from ../rj.py.
#   python r2.py tune   -> tune.json  (seed 9 only; grid over cascade, frozen choice)
#   python r2.py eval   -> eval.json  (seeds 3, 4, 5; all arms, DJ training, frozen cascade)
import hashlib, itertools, json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
if len(sys.argv) > 1 and sys.argv[1] in ("evnp", "eval", "tune", "tune2"):   # Amendment R2: numpy-only phases never load torch (host commit)
    import types
    _t = types.ModuleType("torch"); _t.nn = types.ModuleType("torch.nn"); _t.nn.Module = object
    _t.nn.functional = types.ModuleType("torch.nn.functional"); _t.no_grad = lambda: (lambda f: f)
    sys.modules.update({"torch": _t, "torch.nn": _t.nn, "torch.nn.functional": _t.nn.functional})
import rj
from scipy.special import ndtr

K, R, M = rj.K, rj.R, 256
BAND = [12, 16, 20, 24, 32]
EVAL_SEEDS, TUNE_SEED = [3, 4, 5], 9
NEV, NTR = 10000, 20000
GRID2 = [(zc, np.inf, c, bs, tau, eps) for zc in (3.0, 5.0, np.inf) for tau in ((R / 2, 0.2, 0.05) if zc < np.inf else (R / 2,))
         for c in (0.5, 1.0, 1.5, 2.0) for bs in (8, 16, 32) for eps in (0.0, 0.02, 0.05)]   # Amendment B2: bern radius
GRID = list(itertools.product([2.0, 3.0, 5.0, np.inf], [np.inf, 2 * np.pi, 25.0], [0.2, 0.35, 0.5, 0.75, 1.0], [8, 16, 32]))


def rng_for(seed, T, tag):
    return np.random.default_rng([seed, T, {"eval": 0, "train": 1, "arm": 2}[tag]])


def jac_traj(x, T):
    """Final tangent-linear Jacobian and the per-step stretch ||J_t||_2, t = 1..T (full 2x2 tangent: 2 JVP passes)."""
    A = np.broadcast_to(np.eye(2), x.shape[:-1] + (2, 2)).copy(); st = []
    for _ in range(T):
        c = rj.KS * np.cos(x[..., 0])
        Js = np.empty(A.shape); Js[..., 0, 0] = 1 + c; Js[..., 0, 1] = 1; Js[..., 1, 0] = c; Js[..., 1, 1] = 1
        A = Js @ A; x = rj.step(x); st.append(np.linalg.norm(A, ord=2, axis=(-2, -1)))
    return A, np.stack(st, -1)


def batch(rng, N, T, delta=1e-3, chunk=200):
    """rj.make_batch's draws in the same order, keeping every member hit (N, K, M) and the stretch trajectory."""
    x0 = rng.uniform(0, rj.TWO_PI, (N, 2))
    u = rng.uniform(-1, 1, (N, K, 2))
    y = (x0 + delta * rng.standard_normal((N, 2))) % rj.TWO_PI
    succ = rj.tdist(rj.roll(x0[:, None] + u, T), rj.G) < R
    zh = rj.roll(y[:, None] + u, T); d = rj.tdist(zh, rj.G)
    J, st = jac_traj(y[:, None] + u, T)
    b = dict(T=T, delta=delta, y=y, u=u, succ=succ, zh=zh, d=d, J=J, stretch=delta * st)
    hits = np.empty((N, K, M), bool)
    for i in range(0, N, chunk):
        xm = y[i:i + chunk, None] + delta * rng.standard_normal((min(chunk, N - i), M, 2))
        hits[i:i + chunk] = rj.tdist(rj.roll(xm[:, None] + u[i:i + chunk, :, None], T), rj.G) < R
    b["hits"] = hits; b["cs"] = hits.cumsum(-1, dtype=np.int16)           # computed once (host commit is shared)
    n = rj.wrap(zh - rj.G) / d[..., None]
    b["s1"] = delta * np.linalg.norm(np.einsum("nkab,nka->nkb", J, n), axis=-1)
    b["lin1"] = ndtr((R - d) / b["s1"])
    return b


def cached_batch(seed, T):
    """batch() for (seed, T, eval), cached on disk (hits bit-packed) and retried on host MemoryError (shared commit)."""
    f = HERE / "cache" / f"ev_s{seed}_T{T}.npz"
    if f.exists():
        z = np.load(f); b = {k: z[k] for k in z.files}
        b["hits"] = np.unpackbits(b.pop("hits_p"), -1)[..., :M].astype(bool); b["T"] = int(b["T"]); b["delta"] = float(b["delta"])
        b["cs"] = b["hits"].cumsum(-1, dtype=np.int16); return b
    for i in range(40):
        try:
            b = batch(rng_for(seed, T, "eval"), NEV, T); break
        except MemoryError:
            print("MemoryError, retry", i, flush=True); time.sleep(15)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, **{k: v for k, v in b.items() if k not in ("hits", "cs")}, hits_p=np.packbits(b["hits"], -1))
    return b


def pick_bayes(b):
    return rj._tb(b["hits"].mean(-1), b["d"])


def prune(b, m):
    """LIN1 on all K (forward + 1 VJP = 2KT calls), all M members on the top-m only."""
    T, d = b["T"], b["d"]
    top = np.argsort(-(b["lin1"] - 1e-4 * d), 1, kind="stable")[:, :m]
    P = np.take_along_axis(b["hits"].mean(-1), top, 1)
    pick = np.take_along_axis(top, (P - 1e-4 * np.take_along_axis(d, top, 1)).argmax(1)[:, None], 1)[:, 0]
    return pick, np.full(len(d), 2 * K * T + m * M * T)


def cascade(b, zc, kappa, c, bsz, w=1, tau=R / 2, eps=0.0, bern=False):
    """Stage 0 (centre + full tangent, 3T per candidate, stopped at the mixing step), settle by the LIN1 z-score,
    race the rest on the shared member draws. w = calls per tangent step (sensitivity)."""
    T, d = b["T"], b["d"]; N = len(d)
    over = b["stretch"] > kappa
    mixed = over.any(-1)
    tstop = np.where(mixed, over.argmax(-1) + 1, T)
    s = b["s1"]
    settled = ~mixed & (s < tau) & (np.abs((R - d) / s) > zc)
    s1, s0 = settled & (d < R), settled & (d >= R)
    has1 = s1.any(1)
    alive = ~mixed & ~settled & ~has1[:, None]
    virt = mixed.any(1) & ~has1
    cs = b["cs"]
    spent = np.zeros((N, K), int); p = np.full((N, K), 0.5)          # 0.5: a sole unraced survivor beats settled-0
    active = alive.sum(1) + virt >= 2
    for n in range(bsz, M + 1, bsz):
        if not active.any():
            break
        live = alive & active[:, None]
        spent[live] = n
        p = np.where(live, cs[..., n - 1] / n, p)
        rad = c * np.sqrt((p * (1 - p) + 1 / n) / n) if bern else c / np.sqrt(n)
        lo = np.where(live, p - rad, -np.inf); lead = lo.argmax(1); lcb = lo.max(1)
        lcb = np.where(virt & active, np.maximum(lcb, 1 / K), lcb)
        alive &= ~(live & (p + rad < lcb[:, None]))
        if eps > 0:                                                      # epsilon-good stop (PAC best arm)
            hi = np.where(live & alive, p + rad, -np.inf); hi[np.arange(N), lead] = -np.inf
            done = active & ~virt & (lcb >= hi.max(1) - eps)
            alive[done] = False; alive[done, lead[done]] = True
        virt &= ~(active & (np.where(alive & active[:, None], p - rad, -np.inf).max(1) > 1 / K))
        active &= alive.sum(1) + virt >= 2
    sc = np.full((N, K), -np.inf)
    sc[alive] = p[alive]; sc[mixed] = 1 / K; sc[s0] = 0.0; sc[s1] = 2.0
    deff = np.where(mixed, 10.0, d)
    pick = (sc - 1e-4 * deff).argmax(1)
    calls = ((1 + 2 * w) * tstop).sum(1) + T * spent.sum(1)
    return pick, calls


def ns(s, blind, bayes):
    return (s - blind) / (bayes - blind)


def cfg_key(cfg):
    return dict(zc=cfg[0], kappa=cfg[1], c=cfg[2], b=cfg[3], **(dict(tau=cfg[4], eps=cfg[5]) if len(cfg) > 4 else {}))


def tune(grid=None, bern=False):
    grid = grid or GRID; rows = []
    for T in BAND:
        ev = cached_batch(TUNE_SEED, T)
        bl, by = ev["succ"].mean(), rj.score(ev, pick_bayes(ev))
        for cfg in grid:
            pk, calls = cascade(ev, *cfg[:4], tau=cfg[4] if len(cfg) > 4 else R / 2, eps=cfg[5] if len(cfg) > 5 else 0.0, bern=bern)
            rows.append(dict(T=T, cfg=cfg_key(cfg), ns=float(ns(rj.score(ev, pk), bl, by)), calls=float(calls.mean())))
        print("tune", T, time.strftime("%H:%M:%S"), flush=True); del ev
    res = []
    for i, cfg in enumerate(grid):
        r = rows[i::len(grid)]
        res.append(dict(cfg=cfg_key(cfg), min_ns=min(x["ns"] for x in r), calls=sum(x["calls"] for x in r),
                        ratio={x["T"]: K * (M + 1) * x["T"] / x["calls"] for x in r}))
    ok = [x for x in res if x["min_ns"] >= 0.995]
    chosen = min(ok, key=lambda x: x["calls"])["cfg"] if ok else None
    return dict(seed=TUNE_SEED, rows=rows, summary=res, chosen=chosen)


def eval_seed(seed, cfg, cfg2):
    out = []
    zc, kappa, c, bsz = cfg["zc"], cfg["kappa"], cfg["c"], cfg["b"]
    for T in BAND:
        t0 = time.time()
        ev = cached_batch(seed, T); ra = rng_for(seed, T, "arm")
        picks = dict(floor=rj.pick_floor(ev), bayes=pick_bayes(ev), ens3=rj._tb(ev["hits"][..., :3].mean(-1), ev["d"]),
                     lin1=rj._tb(ev["lin1"], ev["d"]))
        pc, calls = cascade(ev, zc, kappa, c, bsz); picks["cascade"] = pc
        _, calls2 = cascade(ev, zc, kappa, c, bsz, w=2)
        k2 = dict(tau=cfg2["tau"], eps=cfg2["eps"], bern=True)
        picks["cascade2"], c2 = cascade(ev, cfg2["zc"], cfg2["kappa"], cfg2["c"], cfg2["b"], **k2)
        _, c2w = cascade(ev, cfg2["zc"], cfg2["kappa"], cfg2["c"], cfg2["b"], w=2, **k2)
        row = dict(T=T, blind=float(ev["succ"].mean()), cascade_calls=float(calls.mean()), cascade_calls_w2=float(calls2.mean()),
                   cascade_calls_p90=float(np.percentile(calls, 90)), full_calls=K * (M + 1) * T,
                   cascade2_calls=float(c2.mean()), cascade2_calls_w2=float(c2w.mean()), cascade2_calls_p90=float(np.percentile(c2, 90)))
        for m in (1, 2, 3, 4):
            picks[f"prune{m}"], cm = prune(ev, m); row[f"prune{m}_calls"] = float(cm[0])
        for a, pk in picks.items():
            row[a] = float(rj.score(ev, pk))
        perm = ra.permuted(np.tile(np.arange(K), (NEV, 1)), axis=1)
        sh = np.take_along_axis(ev["succ"], perm, 1); shm = sh.mean()
        row["null_max_dev"] = float(max(abs(sh[np.arange(NEV), picks[a]].mean() - shm) for a in ("floor", "bayes", "lin1", "ens3", "cascade")))
        row["cascade_null"] = float(sh[np.arange(NEV), pc].mean() - shm + row["blind"])
        row["cascade2_null"] = float(sh[np.arange(NEV), picks["cascade2"]].mean() - shm + row["blind"])
        row["agree_bayes"] = float((pc == picks["bayes"]).mean())
        row["wall_s"] = round(time.time() - t0, 1)
        out.append(row); del ev
        print(seed, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    return out


def dj_seed(seed):
    """Phase 2 (torch): the D-JEPA-spec arms, learn gate at T = 2 and dj/djl scores per band lead, on the cached eval batches."""
    import torch
    torch.set_num_threads(2)
    gate, sc = {}, {}
    tr = rj.make_batch(rng_for(seed, 2, "train"), NTR, 2, need_post=True)
    for name, bs in (("dj", 8), ("djl", 256)):
        v, base = rj.tokens(tr, "DJ")
        net = rj.train_dj(v, base, tr["succ"], 1000, bs, seed)
        gate[name] = float(ns(rj.score(tr, rj.pick_dj(net, v, base)), tr["succ"].mean(), rj.score(tr, rj.pick_bayes(tr))))
    del tr
    for T in BAND:
        ev = cached_batch(seed, T); sc[str(T)] = {}
        tr = rj.make_batch(rng_for(seed, T, "train"), NTR, T, need_post=False)
        for name, bs in (("dj", 8), ("djl", 256)):
            v, base = rj.tokens(tr, "DJ")
            net = rj.train_dj(v, base, tr["succ"], 1000, bs, seed)
            ve, be = rj.tokens(ev, "DJ"); sc[str(T)][name] = float(rj.score(ev, rj.pick_dj(net, ve, be)))
        del tr, ev
        print("dj", seed, T, sc[str(T)], flush=True)
    return gate, sc


if __name__ == "__main__":
    t0 = time.time(); mode = sys.argv[1]
    meta = dict(bar_sha256=hashlib.sha256((HERE / "BAR.md").read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256((HERE / "r2.py").read_bytes()).hexdigest())
    if mode in ("tune", "tune2"):
        res = dict(meta, **(tune() if mode == "tune" else tune(GRID2, bern=True)))
        print("chosen", res["chosen"])
        (HERE / f"{mode}.json").write_text(json.dumps(res, indent=1, default=float))
    else:
        cfg = json.loads((HERE / "tune.json").read_text())["chosen"]
        cfg2 = json.loads((HERE / "tune2.json").read_text())["chosen"]
        # Amendment R1/R2: per-seed checkpoints; evnp = numpy arms (no torch), evdj = D-JEPA arms, eval = merge.
        fnp, fdj = HERE / "eval_np.json", HERE / "eval_dj.json"
        NP = json.loads(fnp.read_text()) if fnp.exists() else {}
        DJ = json.loads(fdj.read_text()) if fdj.exists() else {}
        for s in EVAL_SEEDS:
            if mode == "evnp" and str(s) not in NP:
                NP[str(s)] = eval_seed(s, cfg, cfg2); fnp.write_text(json.dumps(NP, default=float))
            if mode == "evdj" and str(s) not in DJ:
                DJ[str(s)] = dj_seed(s); fdj.write_text(json.dumps(DJ, default=float))
        if mode != "eval":
            print("wall", round(time.time() - t0, 1)); sys.exit(0)
        seeds = {s: [dict(r, **DJ[s][1][str(r["T"])]) for r in NP[s]] for s in NP}
        gate = {s: DJ[s][0] for s in DJ}
        res = dict(meta, cascade_config=cfg, cascade2_config=cfg2, seeds=seeds, gate=gate)
        res["wall_s"] = round(time.time() - t0, 1)
        (HERE / "eval.json").write_text(json.dumps(res, indent=1, default=float))
    print("wall", round(time.time() - t0, 1))

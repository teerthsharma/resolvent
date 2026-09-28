# R-JEPA / Cameron round 3: B2 re-derived on tuning seed 10 (B2'), lead-dependent eps (B3), LIN1 vs ENS3 at T 40/48.
# Bar: BAR.md in this folder (registered before any seed-10 / 6 / 7 / 8 run). Arms imported read-only from ../r2/r2.py.
#   python r3.py tune     -> tune.json     (seed 10, r2.GRID2, round-2 selection rule)
#   python r3.py leadeps  -> leadeps.json  (seed 10, eps_T per lead, B2' otherwise frozen)
#   python r3.py eval     -> eval.json     (seeds 6/7/8, band; resumes per seed)
#   python r3.py long     -> long.json     (seeds 6/7/8, T 40/48; resumes per seed)
#   python r3.py budget   -> budget.json   (Amendment G: total-member budget race, fresh seeds 11/12/13 + 6/7/8)
#   python r3.py sharp    -> sharp.json    (Amendment H: T = 32 re-scored by expected success under 1,024 fresh members)
#   python r3.py cap      -> cap.json      (Amendment E': B3 with member cap 80; seed 10 descriptive, 6/7/8 deciding)
import hashlib, json, sys, time, types
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
# numpy-only everywhere: stub torch before rj imports it (the cu126 import alone commits ~2.7 GB on this host)
_t = types.ModuleType("torch"); _t.nn = types.ModuleType("torch.nn"); _t.nn.Module = object
_t.nn.functional = types.ModuleType("torch.nn.functional"); _t.no_grad = lambda: (lambda f: f)
sys.modules.setdefault("torch", _t); sys.modules.setdefault("torch.nn", _t.nn); sys.modules.setdefault("torch.nn.functional", _t.nn.functional)
sys.path.insert(0, str(HERE.parent / "r2")); sys.path.insert(0, str(HERE.parent))
import r2, rj

K, M, R, BAND, LONG = rj.K, r2.M, rj.R, r2.BAND, [40, 48]
TUNE_SEED, EVAL_SEEDS, NEV = 10, [6, 7, 8], r2.NEV
EPS_GRID = [0.05, 0.075, 0.10, 0.125, 0.15, 0.20]
FULL = lambda T: K * (M + 1) * T


def cache_path(seed, T):
    return HERE / "cache" / f"ev_s{seed}_T{T}.npz"


def cached_batch(seed, T):
    """r2.batch for (seed, T, eval), cached under r3/cache (hits bit-packed), retried on host MemoryError."""
    f = cache_path(seed, T)
    if f.exists():
        z = np.load(f); b = {k: z[k] for k in z.files}
        b["hits"] = np.unpackbits(b.pop("hits_p"), -1)[..., :M].astype(bool); b["T"] = int(b["T"]); b["delta"] = float(b["delta"])
        b["cs"] = b["hits"].cumsum(-1, dtype=np.int16); return b
    for i in range(40):
        try:
            b = r2.batch(r2.rng_for(seed, T, "eval"), NEV, T); break
        except MemoryError:
            print("MemoryError, retry", i, flush=True); time.sleep(15)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, **{k: v for k, v in b.items() if k not in ("hits", "cs")}, hits_p=np.packbits(b["hits"], -1))
    return b


r2.cached_batch = cached_batch        # r2.tune reads batches through this name; r2's own cache dir is never written


def pick_eps(rows):
    """Registered rule: smallest eps with NS >= 0.995 and p90 ratio >= 3.3; else best p90 among NS >= 0.995."""
    ok = [r for r in rows if r["ns"] >= 0.995]
    hit = [r for r in ok if r["p90"] >= 3.3]
    return min(hit, key=lambda r: r["eps"])["eps"] if hit else max(ok, key=lambda r: r["p90"])["eps"]


def run_cfg(b, cfg, eps=None):
    return r2.cascade(b, cfg["zc"], cfg["kappa"], cfg["c"], cfg["b"], tau=cfg["tau"], eps=cfg["eps"] if eps is None else eps, bern=True)


def b3(b, cfg, lead_eps):
    return run_cfg(b, cfg, lead_eps[b["T"]])


def b3cap(b, cfg, lead_eps, nmax):
    """Amendment E': B3 with the race stopped at n = nmax members (pick = leader). r2.cascade reads the member
    count from the module global M, so it is set for the call and restored."""
    old = r2.M; r2.M = nmax
    try:
        return b3(b, cfg, lead_eps)
    finally:
        r2.M = old


def race_budget(b, c, bsz, eps, B):
    """Amendment G: r2.cascade's Bernstein eps-good race (z_c = kappa = inf) with a total member budget B per decision:
    a row stops before any round that would push its total past B; pick = leader among survivors, ties to floor."""
    T, d, cs = b["T"], b["d"], b["cs"]; N = len(d); I = np.arange(N)
    alive = np.ones((N, K), bool); spent = np.zeros((N, K), int); p = np.full((N, K), 0.5); active = np.ones(N, bool)
    for n in range(bsz, M + 1, bsz):
        active &= spent.sum(1) + alive.sum(1) * bsz <= B
        if not active.any():
            break
        live = alive & active[:, None]; spent[live] = n; p = np.where(live, cs[..., n - 1] / n, p)
        rad = c * np.sqrt((p * (1 - p) + 1 / n) / n)
        lo = np.where(live, p - rad, -np.inf); lead = lo.argmax(1); lcb = lo.max(1)
        alive &= ~(live & (p + rad < lcb[:, None]))
        hi = np.where(live & alive, p + rad, -np.inf); hi[I, lead] = -np.inf
        done = active & (lcb >= hi.max(1) - eps); alive[done] = False; alive[done, lead[done]] = True
        active &= alive.sum(1) >= 2
    return (np.where(alive, p, -np.inf) - 1e-4 * d).argmax(1), 3 * T * K + T * spent.sum(1)


def budget_seed(seed, cfg, B=640):
    out = []
    for T in BAND:
        ev = cached_batch(seed, T)
        picks = dict(floor=rj.pick_floor(ev), bayes=r2.pick_bayes(ev), ens3=rj._tb(ev["hits"][..., :3].mean(-1), ev["d"]),
                     lin1=rj._tb(ev["lin1"], ev["d"]))
        picks["b4"], c = race_budget(ev, cfg["c"], cfg["b"], cfg["eps"], B)
        nl = _null(ev, r2.rng_for(seed, T, "arm"), picks)
        row = dict(T=T, blind=float(ev["succ"].mean()), b4_calls=float(c.mean()), b4_calls_p90=float(np.percentile(c, 90)))
        for a, pk in picks.items():
            row[a] = float(rj.score(ev, pk)); row[a + "_null"] = nl[a]
        out.append(row); del ev
        print("budget", seed, T, {a: round(r2.ns(row[a], row["blind"], row["bayes"]), 4) for a in picks},
              round(FULL(T) / row["b4_calls"], 2), round(FULL(T) / row["b4_calls_p90"], 2), flush=True)
    return out


def escore(P, pick):
    return P[np.arange(len(pick)), pick].mean()


def p_ref(ev, seed, Mref=1024, chunk=25):
    """Amendment H: hit fraction of Mref fresh posterior members per candidate, rng stream [seed, T, 3] (unseen by arms)."""
    rng = np.random.default_rng([seed, ev["T"], 3]); y, u = ev["y"], ev["u"]; N = len(y); P = np.empty((N, K))
    for i in range(0, N, chunk):
        xm = y[i:i + chunk, None] + ev["delta"] * rng.standard_normal((min(chunk, N - i), Mref, 2))
        P[i:i + chunk] = (rj.tdist(rj.roll(xm[:, None] + u[i:i + chunk, :, None], ev["T"]), rj.G) < R).mean(-1)
    return P


def sharp_seed(seed, cfg, lead_eps, T=32):
    ev = cached_batch(seed, T); t0 = time.time(); P = p_ref(ev, seed); h, d = ev["hits"], ev["d"]
    picks = dict(bayes=r2.pick_bayes(ev), b2p=run_cfg(ev, cfg)[0], b4=race_budget(ev, cfg["c"], cfg["b"], cfg["eps"], 640)[0],
                 b3cap=b3cap(ev, cfg, lead_eps, 80)[0], half=rj._tb(h[..., :128].mean(-1), d), ens80=rj._tb(h[..., :80].mean(-1), d),
                 lin1=rj._tb(ev["lin1"], d), ens3=rj._tb(h[..., :3].mean(-1), d), floor=rj.pick_floor(ev))
    bl, by = P.mean(), escore(P, picks["bayes"])
    row = dict(T=T, blind_ref=float(bl), bayes_ref=float(by), wall_s=0.0)
    for a, pk in picks.items():
        row["ns_" + a] = float((escore(P, pk) - bl) / (by - bl))
        row["ns_realised_" + a] = float(r2.ns(rj.score(ev, pk), ev["succ"].mean(), rj.score(ev, picks["bayes"])))
        row["agree_" + a] = float((pk == picks["bayes"]).mean())
    row["wall_s"] = round(time.time() - t0, 1)
    print("sharp", seed, {a: round(row["ns_" + a], 4) for a in picks}, row["wall_s"], flush=True)
    return row


def cap_seed(seed, cfg, lead_eps, nmax=80):
    out = []
    for T in BAND:
        ev = cached_batch(seed, T); pk, c = b3cap(ev, cfg, lead_eps, nmax); by = r2.pick_bayes(ev)
        row = dict(T=T, blind=float(ev["succ"].mean()), bayes=float(rj.score(ev, by)), b3cap=float(rj.score(ev, pk)),
                   b3cap_calls=float(c.mean()), b3cap_calls_p90=float(np.percentile(c, 90)),
                   b3cap_null=_null(ev, r2.rng_for(seed, T, "arm"), dict(x=pk))["x"])
        out.append(row); del ev
        print("cap", seed, T, round(r2.ns(row["b3cap"], row["blind"], row["bayes"]), 4),
              round(FULL(T) / row["b3cap_calls"], 2), round(FULL(T) / row["b3cap_calls_p90"], 2), flush=True)
    return out


def leadeps(cfg):
    out = {}
    for T in BAND:
        ev = cached_batch(TUNE_SEED, T); bl, by = ev["succ"].mean(), rj.score(ev, r2.pick_bayes(ev)); rows = []
        for e in EPS_GRID:
            pk, c = run_cfg(ev, cfg, e)
            rows.append(dict(eps=e, ns=float(r2.ns(rj.score(ev, pk), bl, by)), ratio=FULL(T) / c.mean(), p90=FULL(T) / np.percentile(c, 90)))
        out[T] = dict(rows=rows, eps=pick_eps(rows)); del ev
        print("leadeps", T, out[T]["eps"], [(r["eps"], round(r["ns"], 4), round(r["p90"], 2)) for r in rows], flush=True)
    return dict(seed=TUNE_SEED, cfg=cfg, table=out, eps={T: v["eps"] for T, v in out.items()})


def _null(ev, ra, picks):
    perm = ra.permuted(np.tile(np.arange(K), (NEV, 1)), axis=1)
    sh = np.take_along_axis(ev["succ"], perm, 1); shm = sh.mean()
    return {a: float(sh[np.arange(NEV), pk].mean() - shm + ev["succ"].mean()) for a, pk in picks.items()}


def eval_seed(seed, cfg2p, cfg2, lead_eps):
    out = []
    for T in BAND:
        t0 = time.time(); ev = cached_batch(seed, T)
        picks = dict(floor=rj.pick_floor(ev), bayes=r2.pick_bayes(ev), ens3=rj._tb(ev["hits"][..., :3].mean(-1), ev["d"]),
                     lin1=rj._tb(ev["lin1"], ev["d"]))
        row = dict(T=T, blind=float(ev["succ"].mean()), full_calls=FULL(T))
        for a, (pk, c) in dict(b2p=run_cfg(ev, cfg2p), b3=b3(ev, cfg2p, lead_eps), b2=run_cfg(ev, cfg2)).items():
            picks[a] = pk; row[a + "_calls"] = float(c.mean()); row[a + "_calls_p90"] = float(np.percentile(c, 90))
        nl = _null(ev, r2.rng_for(seed, T, "arm"), picks)
        for a, pk in picks.items():
            row[a] = float(rj.score(ev, pk)); row[a + "_null"] = nl[a]
        row["null_max_dev"] = max(abs(nl[a] - row["blind"]) for a in ("floor", "bayes", "lin1", "ens3"))
        row["wall_s"] = round(time.time() - t0, 1); out.append(row); del ev
        print(seed, T, {a: round(r2.ns(row[a], row["blind"], row["bayes"]), 4) for a in picks},
              {a: (round(FULL(T) / row[a + "_calls"], 2), round(FULL(T) / row[a + "_calls_p90"], 2)) for a in ("b2p", "b3", "b2")}, flush=True)
    return out


def long_seed(seed):
    out = []
    for T in LONG:
        t0 = time.time(); ev = cached_batch(seed, T)
        picks = dict(floor=rj.pick_floor(ev), bayes=r2.pick_bayes(ev), ens3=rj._tb(ev["hits"][..., :3].mean(-1), ev["d"]),
                     ens2=rj._tb(ev["hits"][..., :2].mean(-1), ev["d"]), lin1=rj._tb(ev["lin1"], ev["d"]))
        nl = _null(ev, r2.rng_for(seed, T, "arm"), picks)
        row = dict(T=T, blind=float(ev["succ"].mean()))
        for a, pk in picks.items():
            row[a] = float(rj.score(ev, pk)); row[a + "_null"] = nl[a]
        row["null_max_dev"] = max(abs(nl[a] - row["blind"]) for a in picks)
        row["wall_s"] = round(time.time() - t0, 1); out.append(row); del ev
        print(seed, T, {a: round(r2.ns(row[a], row["blind"], row["bayes"]), 4) for a in picks}, flush=True)
    return out


if __name__ == "__main__":
    t0 = time.time(); mode = sys.argv[1]
    meta = dict(bar_sha256=hashlib.sha256((HERE / "BAR.md").read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256((HERE / "r3.py").read_bytes()).hexdigest())
    J = lambda n: json.loads((HERE / n).read_text())
    if mode == "tune":
        r2.TUNE_SEED = TUNE_SEED
        res = dict(meta, **r2.tune(r2.GRID2, bern=True)); print("chosen", res["chosen"])
        (HERE / "tune.json").write_text(json.dumps(res, indent=1, default=float))
    elif mode == "cap":                                   # Amendment E': seed 10 descriptive, then 6/7/8
        cfg, le = J("tune.json")["chosen"], {int(k): v for k, v in J("leadeps.json")["eps"].items()}
        res = dict(meta, nmax=80, cfg=cfg, lead_eps=le, tune10=cap_seed(TUNE_SEED, cfg, le), seeds={})
        for s in EVAL_SEEDS:
            res["seeds"][str(s)] = cap_seed(s, cfg, le)
        (HERE / "cap.json").write_text(json.dumps(res, indent=1, default=float))
    elif mode == "budget":                                # Amendment G: fresh seeds 11/12/13 deciding, 6/7/8 descriptive
        cfg = J("tune.json")["chosen"]; assert cfg["zc"] == np.inf and cfg["kappa"] == np.inf
        f = HERE / "budget.json"; res = J(f.name) if f.exists() else dict(meta, B=640, cfg=cfg, seeds={}, seeds_678={})
        for s in [11, 12, 13, 6, 7, 8]:
            key = "seeds" if s > 10 else "seeds_678"
            if str(s) not in res[key]:
                res[key][str(s)] = budget_seed(s, cfg); f.write_text(json.dumps(res, indent=1, default=float))
    elif mode == "sharp":                                 # Amendment H: T = 32, expected success under 1,024 fresh members
        cfg, le = J("tune.json")["chosen"], {int(k): v for k, v in J("leadeps.json")["eps"].items()}
        f = HERE / "sharp.json"; res = J(f.name) if f.exists() else dict(meta, M_ref=1024, seeds={})
        for s in [6, 7, 8, 11, 12, 13]:
            if str(s) not in res["seeds"]:
                res["seeds"][str(s)] = sharp_seed(s, cfg, le); f.write_text(json.dumps(res, indent=1, default=float))
    elif mode == "leadeps":
        (HERE / "leadeps.json").write_text(json.dumps(dict(meta, **leadeps(J("tune.json")["chosen"])), indent=1, default=float))
    else:
        f = HERE / f"{mode}.json"
        res = J(f.name) if f.exists() else dict(meta, seeds={})
        if mode == "eval":
            cfg2p, le = J("tune.json")["chosen"], {int(k): v for k, v in J("leadeps.json")["eps"].items()}
            res.update(b2p_config=cfg2p, b2_config=json.loads((HERE.parent / "r2" / "tune2.json").read_text())["chosen"], lead_eps=le)
        for s in EVAL_SEEDS:
            if str(s) not in res["seeds"]:
                res["seeds"][str(s)] = eval_seed(s, res["b2p_config"], res["b2_config"], res["lead_eps"]) if mode == "eval" else long_seed(s)
                res["script_sha256_" + str(s)] = meta["script_sha256"]
                f.write_text(json.dumps(res, indent=1, default=float))
    print("wall", round(time.time() - t0, 1))

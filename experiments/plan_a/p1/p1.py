# P1: hyperbolic vs Euclidean vs order embeddings on consequence sets, far band held out.
# Bar: sun/canon/p1/BAR.md (sha256 recorded in the output). CPU, float64.
import hashlib, json, random, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).parent
D0, STEPS, BATCH, LRS, SEEDS = 2, 4000, 4096, (0.01, 0.03, 0.1), (0, 1, 2)
ARMS = {"E1": ("euc", D0), "E2": ("euc", 2 * D0), "E8": ("euc", 8 * D0), "H1": ("hyp", D0), "O1": ("ord", D0),
        "E1-exp": ("eexp", D0), "E2-exp": ("eexp", 2 * D0)}


def world(seed, dag, depth=8, cap=1500):
    r = random.Random(f"p1|{seed}")
    par, lvl, fr = [[]], [0], [0]
    for d in range(1, depth + 1):
        nx = []
        for p in fr:
            for _ in range(r.randint(1, 4)):
                if len(par) >= cap:
                    break
                par.append([p]); lvl.append(d); nx.append(len(par) - 1)
        fr = nx
    n = len(par)
    skipped = 0
    if dag:                                              # A1.7: second parent from lower levels, not already an ancestor
        r2 = random.Random(f"p1dag|{seed}")
        anc = [set() for _ in range(n)]
        for v in range(n):                               # index order is level order, parents come first
            for p in par[v]:
                anc[v] |= anc[p] | {p}
            if lvl[v] >= 2 and r2.random() < 0.25:
                cand = [w for w in range(v) if lvl[w] < lvl[v] and w not in anc[v]]
                if cand:
                    w = r2.choice(cand); par[v].append(w); anc[v] |= anc[w] | {w}
                else:
                    skipped += 1
    kids = [[] for _ in range(n)]
    for v in range(n):
        for p in par[v]:
            kids[p].append(v)
    hop = np.zeros((n, n), np.int16)                     # 0 = not a descendant
    for s in range(n):
        dist, q = {s: 0}, [s]
        for u in q:
            for w in kids[u]:
                if w not in dist:
                    dist[w] = dist[u] + 1; q.append(w)
        for v, dd in dist.items():
            hop[s, v] = dd
    world.skipped = skipped
    return np.array(lvl), hop


def split(hop, seed):
    n = hop.shape[0]
    A = hop > 0
    neg = ~A & ~np.eye(n, dtype=bool)
    coin = np.random.default_rng(seed + 1000).random((n, n)) < 0.5
    return dict(A=A, near=(hop >= 1) & (hop <= 2), far=hop >= 3, trneg=neg & coin, honeg=neg & ~coin)


class Emb(torch.nn.Module):
    def __init__(self, n, d, kind, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.kind = kind
        self.w = torch.nn.Parameter(0.1 * torch.randn(n, d, generator=g, dtype=torch.float64))
        self.a = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.b = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.c = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    def forward(self, u, v):
        x, y = self.w[u], self.w[v]
        if self.kind == "ord":
            return self.b - self.a * torch.relu(x - y).pow(2).sum(-1)
        if self.kind == "eexp":                          # A1.4: Euclidean with H1's radial growth, x = (e^|w| - 1) w/|w|
            nx, ny = (x.pow(2).sum(-1) + 1e-15).sqrt(), (y.pow(2).sum(-1) + 1e-15).sqrt()
            x, y = torch.expm1(nx)[:, None] * x / nx[:, None], torch.expm1(ny)[:, None] * y / ny[:, None]
        if self.kind in ("euc", "eexp"):
            dist = ((x - y).pow(2).sum(-1) + 1e-12).sqrt()
            ru, rv = x.norm(dim=-1), y.norm(dim=-1)
        else:  # hyp: Lorentz model, x = exp_0(w); ponytail: tangent-at-origin parametrisation + Adam, not Riemannian SGD
            ru, rv = (x.pow(2).sum(-1) + 1e-15).sqrt(), (y.pow(2).sum(-1) + 1e-15).sqrt()
            xs, ys = torch.sinh(ru)[:, None] * x / ru[:, None], torch.sinh(rv)[:, None] * y / rv[:, None]
            inner = torch.cosh(ru) * torch.cosh(rv) - (xs * ys).sum(-1)
            dist = torch.acosh(torch.clamp(inner, min=1 + 1e-12))
        return self.b - self.a * dist + self.c * (rv - ru)


def all_logits(m, n):
    with torch.no_grad():
        u = torch.arange(n)
        return torch.cat([m(torch.full((n,), i), u)[None] for i in range(n)]).numpy()


def per_query_f1(pred, true, cand):
    t, p = true & cand, pred & cand
    tp, nt, npred = (t & p).sum(1), t.sum(1), p.sum(1)
    keep = nt > 0
    return 2 * tp[keep] / (nt[keep] + npred[keep])


def macro_f1(pred, true, cand):
    return float(per_query_f1(pred, true, cand).mean())


def mean_ap(L, true, cand):                              # threshold-free, per query
    aps = []
    for u in range(L.shape[0]):
        c = np.nonzero(cand[u])[0]
        y = true[u, c]
        if y.sum() == 0:
            continue
        o = np.argsort(-L[u, c], kind="stable"); y = y[o]
        hits = np.cumsum(y)
        aps.append(float((hits[y] / (np.nonzero(y)[0] + 1)).mean()))
    return float(np.mean(aps))


def cands(S):
    return S["near"] | S["trneg"], S["far"] | S["honeg"]


def score(pred, S):
    A = S["A"]; cn, cf = cands(S)
    return dict(near_cons=macro_f1(pred, A, cn), far_cons=macro_f1(pred, A, cf),
                near_root=macro_f1(pred.T, A.T, cn.T), far_root=macro_f1(pred.T, A.T, cf.T))


def tuned(L, S):                                         # A1.3: threshold maximising near-band consequence F1, frozen for far
    cn, _ = cands(S)
    qs = np.unique(np.concatenate([[0.0], np.quantile(L[cn], np.linspace(0.01, 0.99, 99))]))
    best = max(qs, key=lambda th: macro_f1(L > th, S["A"], cn))
    return float(best)


CACHE = Path(r"C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/cda1e2f8-4d00-4cc7-8fc6-24a3d2e86b7b/scratchpad/p1cache")


def train(job):                                          # ponytail: per-job cache + MemoryError retry; this box runs out of RAM
    f = CACHE / ("%s_%s_%d_%g.json" % (job[0], job[1], job[2], job[3]))
    if f.exists():
        return json.loads(f.read_text())
    for attempt in range(5):
        try:
            r = _train(job); break
        except MemoryError:
            time.sleep(30)
    else:
        raise MemoryError(job)
    CACHE.mkdir(exist_ok=True); f.write_text(json.dumps(r))
    return r


def _train(job):
    torch.set_num_threads(1)
    arm, dag, seed, lr = job
    kind, d = ARMS[arm]
    lvl, hop = world(seed, dag)
    S = split(hop, seed)
    n = len(lvl)
    pos, neg = np.argwhere(S["near"]), np.argwhere(S["trneg"])
    rng = np.random.default_rng(seed)
    m = Emb(n, d, kind, seed)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    bce = torch.nn.BCEWithLogitsLoss()
    y = torch.cat([torch.ones(BATCH), torch.zeros(BATCH)]).double()
    for _ in range(STEPS):
        pb, nb = pos[rng.integers(0, len(pos), BATCH)], neg[rng.integers(0, len(neg), BATCH)]
        uv = torch.from_numpy(np.concatenate([pb, nb]))
        loss = bce(m(uv[:, 0], uv[:, 1]), y)
        opt.zero_grad(); loss.backward(); opt.step()
    L = all_logits(m, n)
    if not np.isfinite(L).all():
        L = np.nan_to_num(L, nan=-1e30, posinf=1e30, neginf=-1e30)
    th = tuned(L, S)
    _, cf = cands(S)
    out = score(L > th, S)
    return dict(arm=arm, world="dag" if dag else "tree", seed=seed, lr=lr, n=n, depth=int(lvl.max()), threshold=th,
                final_loss=float(loss.detach()), far_ap=mean_ap(L, S["A"], cf),
                far_cons_q=per_query_f1(L > th, S["A"], cf).tolist(), **out)


def floors(dag, seed):
    lvl, hop = world(seed, dag)
    S = split(hop, seed)
    near = S["near"].astype(np.int64)                    # A1.1: transitive closure of train positives
    C = near.copy()
    while True:
        C2 = ((C + (C @ near)) > 0).astype(np.int64)
        if (C2 == C).all():
            break
        C = C2
    return {"F-closure": score(C > 0, S), "F-depth3": score(lvl[None, :] >= lvl[:, None] + 3, S),
            "F-all": score(np.ones_like(S["A"]), S), "skipped_second_parents": world.skipped}


def ci(a, b, rng, reps=2000):                            # paired bootstrap over pooled (seed, query)
    d = np.asarray(a) - np.asarray(b)
    bs = d[rng.integers(0, len(d), (reps, len(d)))].mean(1)
    return float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def side(c, margin, strict=False):                       # holds if whole CI >= margin (or > when strict), fails if whole CI below
    lo, hi = c[1], c[2]
    if (lo > margin) if strict else (lo >= margin):
        return True
    if (hi <= margin) if strict else (hi < margin):
        return False
    return None


if __name__ == "__main__":
    t0 = time.time()
    with Pool(4) as p:                                   # RAM-bound on this box
        sel = p.map(train, [(a, dg, 0, lr) for a in ARMS for dg in (False, True) for lr in LRS])
        best = {}
        for r in sel:
            k = (r["arm"], r["world"])
            if k not in best or r["near_cons"] > best[k]["near_cons"]:
                best[k] = r
        rest = p.map(train, [(a, dg, s, best[(a, "dag" if dg else "tree")]["lr"]) for a in ARMS for dg in (False, True) for s in SEEDS[1:]])
    runs = list(best.values()) + rest
    fl = {w: [floors(dg, s) for s in SEEDS] for w, dg in (("tree", False), ("dag", True))}
    rng = np.random.default_rng(51)
    summary, verdict = {}, {}
    for w in ("tree", "dag"):
        summary[w] = {}
        q = {}
        for a in ARMS:
            rs = sorted([r for r in runs if r["arm"] == a and r["world"] == w], key=lambda r: r["seed"])
            q[a] = sum((r["far_cons_q"] for r in rs), [])
            summary[w][a] = dict(lr=rs[0]["lr"], learned_3of3=all(r["near_cons"] >= 0.90 for r in rs),
                                 n_learned=sum(r["near_cons"] >= 0.90 for r in rs), depth=[r["depth"] for r in rs],
                                 **{k: float(np.mean([r[k] for r in rs])) for k in ("near_cons", "far_cons", "near_root", "far_root", "far_ap")},
                                 far_cons_per_seed=[r["far_cons"] for r in rs], near_cons_per_seed=[r["near_cons"] for r in rs])
        for f in ("F-closure", "F-depth3", "F-all"):
            summary[w][f] = {k: float(np.mean([x[f][k] for x in fl[w]])) for k in fl[w][0][f]}
        summary[w]["skipped_second_parents"] = [x["skipped_second_parents"] for x in fl[w]]
        L = {a: summary[w][a]["learned_3of3"] for a in ARMS}
        cis = {"H1-E8": ci(q["H1"], q["E8"], rng), "E2-H1": ci(q["E2"], q["H1"], rng),
               "O1-H1": ci(q["O1"], q["H1"], rng), "H1-E2exp": ci(q["H1"], q["E2-exp"], rng)}
        pred, counter, house = side(cis["H1-E8"], -0.02), side(cis["E2-H1"], -0.02), side(cis["O1-H1"], -0.02)
        expg = side(cis["H1-E2exp"], 0.02, strict=True)
        if not L["E2"] and L["H1"]:
            counter = False                              # A1.6
        kill = (counter is True and L["E2"] and L["H1"]) or (house is True and L["O1"] and L["H1"])
        ok = all(L[a] for a in ("H1", "E8", "O1", "E2-exp"))
        passed = pred is True and counter is False and house is False and expg is True and ok
        verdict[w] = dict(ci=cis, prediction=pred, counter=counter, house_kill=house, exp_guard=expg,
                          verdict="KILL" if kill else ("PASS" if passed else "OPEN"))
    out = dict(bar_sha256=hashlib.sha256((HERE / "BAR.md").read_bytes()).hexdigest(),
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               summary=summary, verdict=verdict, deciding="tree", wall_clock_s=round(time.time() - t0, 1),
               runs=[{k: v for k, v in r.items() if k != "far_cons_q"} for r in runs])
    (HERE / "results_p1.json").write_text(json.dumps(out, indent=1))
    for w in summary:
        print(w, verdict[w])
        for a, v in summary[w].items():
            print("  %-8s %s" % (a, v if not isinstance(v, dict) else {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items()}))
    print("wall", out["wall_clock_s"])

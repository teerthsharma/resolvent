# R-JEPA round 2 (Foreman): the shared-error dial bed with a learned predictor at K = 63. Bar: BAR.md.
# python r2.py pred                 -> results/pred_s{0,1,2}.pt + results/pred.json (L-LEARN) + results/calib.json
# python r2.py cell F Q [F Q ...]   -> results/res_f{F}_q{Q}_s{seed}.json for seeds 0, 1, 2
import hashlib, json, math, sys, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as Fn

HERE = Path(__file__).parent
OUT = HERE / "results"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
K, D, M_ACT, H, NTOK = 63, 8, 4, 5, 12
TAU = 0.05
MB = 2048                                                                      # A2: 256 -> 2048
DECIDING = {(0.0, 1.0), (1.0, 3.0), (0.0, 3.0)}                                # A3 adds (0, 1)
torch.set_num_threads(2)                                                      # resume: host-RAM rule


# ---- true dynamics ------------------------------------------------------------------------------------------
def make_dyn(seed=12345):
    g = torch.Generator().manual_seed(seed)
    Q, R = torch.linalg.qr(torch.randn(D, D, generator=g))
    Q = Q * torch.sign(torch.diagonal(R))                     # Haar
    return dict(A=1.1 * Q, C=torch.randn(D, D, generator=g) / D ** 0.5, B=torch.randn(D, M_ACT, generator=g) / M_ACT ** 0.5)


def true_step(z, a, dyn):
    A, B, C = (dyn[k].to(z.device) for k in "ABC")
    return z @ A.T + 0.5 * torch.tanh(z @ C.T + a @ B.T)


def rollout(step, y, a):
    """y (n, D) shared start or (n, K, D) per-candidate starts (or any leading dims); a (n, K, H, m) -> (n, K, D)."""
    z = y[..., None, :].expand(*a.shape[:-3], a.shape[-3], D) if y.dim() == a.dim() - 2 else y
    for t in range(a.shape[-2]):
        z = step(z, a[..., t, :])
    return z


def make_cell(n, f, sigma, seed, dyn):
    """Generated on DEV (host RAM is the binding limit: the CPU version peaked at 1.69 GB)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    step = lambda z, u: true_step(z, u, dyn)
    y = torch.randn(n, D, generator=g, device=DEV)
    a = torch.randn(n, K, H, M_ACT, generator=g, device=DEV)
    goal = rollout(step, y, torch.randn(n, 1, H, M_ACT, generator=g, device=DEV))[:, 0]
    e = math.sqrt(f) * torch.randn(n, 1, D, generator=g, device=DEV) + math.sqrt(1 - f) * torch.randn(n, K, D, generator=g, device=DEV)
    z = rollout(step, y[:, None] - sigma * e, a)
    dist = (z - goal[:, None]).norm(dim=-1)
    return dict(y=y, a=a, g=goal, dist=dist, best=dist.argmin(1))


def hit(pick, b):
    return float((pick.to(b["best"].device) == b["best"]).float().mean())


def succ(pick, b, r):
    pick = pick.to(b["dist"].device)
    return float((b["dist"][torch.arange(len(pick), device=pick.device), pick] < r).float().mean())


@torch.no_grad()
def bayes_P(b, dyn, sigma, f, r, M=2048, seed=0, chunk=16):
    """MC over (s, u_1..u_K) with TRUE dynamics: P(k best) and marginal P(|z_k - g| < r), both (n, K)."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    Pb, Ps = [], []
    for i in range(0, len(b["y"]), chunk):
        y, a, goal = (b[k][i:i + chunk].to(DEV) for k in ("y", "a", "g"))
        c = len(y)
        e = math.sqrt(f) * torch.randn(c, M, 1, D, generator=gen, device=DEV) \
            + math.sqrt(1 - f) * torch.randn(c, M, K, D, generator=gen, device=DEV)
        z = y[:, None, None] - sigma * e
        for t in range(H):
            z = true_step(z, a[:, None, :, t], dyn)
        dist = (z - goal[:, None, None]).norm(dim=-1)            # (c, M, K)
        Pb.append(Fn.one_hot(dist.argmin(-1), K).float().mean(1))
        Ps.append((dist < r).float().mean(1))
    return torch.cat(Pb).to(b["y"].device), torch.cat(Ps).to(b["y"].device)


# ---- LIN1 -----------------------------------------------------------------------------------------------------
def lin1_feats(step, y, a, goal, sigma, chunk=250):
    """d_k = |z_hat_k - g|, s_k = sigma |J_k^T n_k|: one VJP per candidate (all candidates in one backward)."""
    ds, ss = [], []
    for i in range(0, len(y), chunk):
        yy = y[i:i + chunk].to(DEV)[:, None].expand(-1, a.shape[1], -1).clone().requires_grad_(True)
        with torch.enable_grad():
            z = rollout(step, yy, a[i:i + chunk].to(DEV))
            diff = z - goal[i:i + chunk].to(DEV)[:, None]
            d = diff.norm(dim=-1)
            (gy,) = torch.autograd.grad((z * (diff / d[..., None]).detach()).sum(), yy)
        ds.append(d.detach()); ss.append(sigma * gy.norm(dim=-1))
    return torch.cat(ds).to(y.device), torch.cat(ss).to(y.device)


def lin1_prob(d, s, r):
    return torch.special.ndtr((r - d) / s)


def lin1_pick(d, s, r):
    return (torch.special.log_ndtr((r - d) / s) - 1e-9 * d).argmax(1)       # log form: no underflow ties


def rank01(x):                                                                # 0 = smallest
    return x.argsort(1).argsort(1).float() / (x.shape[1] - 1)


def tokens(step, b, sigma, r):
    d, s = lin1_feats(step, b["y"], b["a"], b["g"], sigma)
    with torch.no_grad():
        zh = torch.cat([rollout(step, b["y"][i:i + 1000].to(DEV), b["a"][i:i + 1000].to(DEV)).to(d.device)
                        for i in range(0, len(b["y"]), 1000)])
    rr, lr = rank01(d), rank01(-(torch.special.log_ndtr((r - d) / s) - 1e-9 * d))
    v = torch.cat([Fn.layer_norm(zh - b["g"][:, None].to(d.device), (D,)), d[..., None], torch.log(s + 1e-12)[..., None],
                   rr[..., None], lr[..., None]], -1)
    return v, rr, lr, d, s


# ---- heads ----------------------------------------------------------------------------------------------------
def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, o))


WIDTH = {"point": 256, "hop1": 97}                                           # params within 10 % of dj (BAR)


class Head(nn.Module):
    """forward(v, base) -> logits, higher = better. dj kinds: 'dj' + eps code ('02' = 0.2, '4' = 4) + 'L' for LIN1 base."""

    def __init__(self, kind, nin=NTOK):
        super().__init__()
        self.kind = kind
        if kind.startswith("dj"):
            code = kind[2:].rstrip("Lf")                                     # 'f' suffix: lr x 20 (A4)
            self.eps = 0.2 if code == "02" else float(code)
            self.enc = nn.Sequential(nn.Linear(nin, 64), nn.LayerNorm(64), nn.GELU())
            layer = nn.TransformerEncoderLayer(64, 4, 128, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
            self.tf = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.down, self.up = nn.Linear(64, 8), nn.Linear(8, 1)
            nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)
            return
        W = WIDTH[kind]
        if kind == "point":
            self.net = mlp(nin, W, 1)
            return
        self.enc, self.out = mlp(nin, W, W), mlp(2 * W, W, 1)
        self.q, self.k = nn.Linear(W, W), nn.Linear(W, W)
        self.theta = nn.Parameter(torch.zeros(()))

    def forward(self, v, base):
        if self.kind.startswith("dj"):
            h = self.tf(self.enc(v))
            self.last_delta = self.eps * torch.tanh(self.up(torch.tanh(self.down(h)))).squeeze(-1)
            return -(base + self.last_delta) / TAU
        if self.kind == "point":
            return self.net(v).squeeze(-1)
        h = self.enc(v)
        Kc = v.shape[1]
        logit = (self.q(h) @ self.k(h).transpose(1, 2)) / h.shape[-1] ** 0.5
        W = torch.softmax(logit.masked_fill(torch.eye(Kc, dtype=torch.bool, device=v.device), -1e9), -1)
        m = h + 0.99 * torch.sigmoid(self.theta) * W @ h
        return self.out(torch.cat([h, m], -1)).squeeze(-1)


def base_for(kind, rr, lr):
    return lr if kind.endswith("L") else rr


def train_head(kind, v, base, best, seed=0, steps=4000, bs=64, lr=1e-3):
    torch.manual_seed(seed)
    h = Head(kind).to(DEV)
    v, base, best = v.to(DEV), base.to(DEV), best.to(DEV)
    lr = lr * 20 if kind.endswith("f") else lr
    opt = torch.optim.AdamW(h.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    g = torch.Generator(device=DEV).manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, len(v), (bs,), generator=g, device=DEV)
        o = h(v[idx], base[idx])
        loss = Fn.cross_entropy(o, best[idx])
        if kind.startswith("dj"):
            loss = loss + 0.1 * (h.last_delta ** 2).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(h.parameters(), 1.0); opt.step(); sch.step()
    return h.eval()


# ---- learned predictor ----------------------------------------------------------------------------------------
class Pred(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp(D + M_ACT, 256, D)

    def forward(self, z, a):
        return self.net(torch.cat([z, a], -1))


def _transitions(n, gen, dyn):
    z = torch.randn(n, D, generator=gen, device=DEV)
    a = torch.randn(n, H, M_ACT, generator=gen, device=DEV)
    Z, A, Z1 = [], [], []
    for t in range(H):
        z1 = true_step(z, a[:, t], dyn)
        Z.append(z); A.append(a[:, t]); Z1.append(z1); z = z1
    return torch.cat(Z), torch.cat(A), torch.cat(Z1)


def r2score(p, t):
    return float(1 - ((p - t) ** 2).sum() / ((t - t.mean(0)) ** 2).sum())


def train_predictor(seed, dyn, steps=2000, bs=2048):
    gen = torch.Generator(device=DEV).manual_seed(500 + seed)
    Z, A, Z1 = _transitions(100_000, gen, dyn)
    torch.manual_seed(seed)
    net = Pred().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        i = torch.randint(0, len(Z), (bs,), generator=gen, device=DEV)
        loss = Fn.mse_loss(net(Z[i], A[i]), Z1[i])
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    Zt, At, Z1t = _transitions(10_000, gen, dyn)                 # 50,000 held-out transitions
    with torch.no_grad():
        r_one = r2score(net(Zt, At), Z1t)
        y = torch.randn(2000, D, generator=gen, device=DEV)
        a = torch.randn(2000, 1, H, M_ACT, generator=gen, device=DEV)
        r_H = r2score(rollout(net, y, a)[:, 0], rollout(lambda z, u: true_step(z, u, dyn), y, a)[:, 0])
    return net, r_one, r_H


# ---- calibration: sigma for sigma/rho ------------------------------------------------------------------------
@torch.no_grad()
def calibrate(dyn, ratios=(1.0, 3.0, 10.0), n=4000):
    gen = torch.Generator().manual_seed(999)
    step = lambda z, u: true_step(z, u, dyn)
    y = torch.randn(n, D, generator=gen)
    a = torch.randn(n, 16, H, M_ACT, generator=gen)
    z0 = rollout(step, y, a)
    rho = float(z0.std(1).pow(2).mean(-1).sqrt().mean())
    e = torch.randn(n, 16, D, generator=gen)
    spread = lambda sg: float((rollout(step, y[:, None] - sg * e, a) - z0).std((0, 1)).pow(2).mean().sqrt())
    out = dict(rho=rho, sigma={})
    for q in ratios:
        lo, hi = 1e-3, 100.0
        for _ in range(40):
            mid = math.sqrt(lo * hi)
            lo, hi = (mid, hi) if spread(mid) < q * rho else (lo, mid)
        out["sigma"][str(q)] = math.sqrt(lo * hi)
    return out


# ---- a cell ---------------------------------------------------------------------------------------------------
def run_cell(f, q, seed, calib, dyn, n_train=60_000, n_eval=20_000):   # A3: was 20,000 / 2,000
    t0 = time.time()
    sigma = calib["sigma"][str(q)]
    net = Pred().to(DEV)
    net.load_state_dict(torch.load(OUT / f"pred_s{seed}.pt"))
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    cid = int(f * 100) * 100 + int(q * 10)
    tr = make_cell(n_train, f, sigma, 10_000 * seed + cid + 1, dyn)
    ev = make_cell(n_eval, f, sigma, 10_000 * seed + cid + 2, dyn)
    r = float(tr["dist"].min(1).values.median())
    vt, rt, lt, _, _ = tokens(net, tr, sigma, r)
    ve, re_, le, de, se = tokens(net, ev, sigma, r)
    Pb, Ps = bayes_P(ev, dyn, sigma, f, r, M=MB, seed=seed)
    del tr["a"], tr["y"], tr["g"], tr["dist"]
    bp = (Pb - 1e-9 * de).argmax(1)
    picks = {"bayes": bp, "bayes_succ": Ps.argmax(1), "dist": de.argmin(1), "lin1": lin1_pick(de, se, r),
             "bayes_eps02": (Pb - 1e-9 * de).masked_fill(re_ > re_.min(1, keepdim=True).values + 0.4, -1).argmax(1)}
    import os
    extra = os.environ.get("R2_KINDS")                                       # A4: extra arms, separate file
    kinds = extra.split(",") if extra else ["point", "hop1", "dj02", "dj4L"] + (["dj02L", "dj4", "null"] if (f, q) in DECIDING else [])
    dsat = {}
    nparams = {}
    for kd in kinds:
        lab = tr["best"]
        if kd == "null":
            lab = lab[torch.randperm(len(lab), generator=torch.Generator().manual_seed(seed))]
        k2 = "point" if kd == "null" else kd
        h = train_head(k2, vt, base_for(k2, rt, lt), lab, seed=seed)
        nparams[kd] = sum(p.numel() for p in h.parameters())
        with torch.no_grad():
            bse, out, ds_ = base_for(k2, re_, le), [], []
            for i in range(0, n_eval, 1000):                              # chunked: full-batch eval OOMed at 20,000
                out.append(h(ve[i:i + 1000], bse[i:i + 1000]).argmax(1))
                if k2.startswith("dj"):
                    ds_.append((h.last_delta.abs() / h.eps).mean())
            picks[kd] = torch.cat(out)
            if ds_:
                dsat[kd] = float(torch.stack(ds_).mean())
        del h; torch.cuda.empty_cache()
    hb = hit(bp, ev)
    res = dict(f=f, q=q, seed=seed, sigma=sigma, rho=calib["rho"], r=r, n_train=n_train, n_eval=n_eval, M=MB,
               bar_sha256=hashlib.sha256((HERE / "BAR.md").read_bytes()).hexdigest(), nparams=nparams,
               hit={k: hit(p, ev) for k, p in picks.items()}, succ={k: succ(p, ev, r) for k, p in picks.items()},
               agree_bayes={k: float((p == bp).float().mean()) for k, p in picks.items()},
               delta_over_eps=dsat, bayes_pick_rank_gt_0p4=float((re_[torch.arange(n_eval, device=DEV), bp] > 0.4).float().mean()))
    res["ns"] = {k: (h_ - 1 / K) / (hb - 1 / K) for k, h_ in res["hit"].items()}
    res["wall_s"] = round(time.time() - t0, 1)
    sfx = os.environ.get("R2_SFX", "_extra") if extra else ""                    # A5 S1: eps sweep writes _eps
    torch.save({k: p.cpu().to(torch.int8) for k, p in picks.items()} | {"best": ev["best"].cpu().to(torch.int8)},
               OUT / f"picks_f{f}_q{q}_s{seed}{sfx}.pt")
    (OUT / f"res_f{f}_q{q}_s{seed}{sfx}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("f", "q", "seed", "wall_s")} | {"ns": {k: round(v, 3) for k, v in res["ns"].items()}}), flush=True)
    return res


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    dyn = make_dyn()
    if sys.argv[1] == "table":                                                 # NS per arm per seed, paired SE vs dj02
        rows = []
        for p in sorted(OUT.glob("res_f*_s[0-9].json")):
            r = json.loads(p.read_text())
            pk = torch.load(OUT / p.name.replace("res_", "picks_").replace(".json", ".pt"))
            x = OUT / p.name.replace(".json", "_extra.json")
            if x.exists():
                r["ns"] = json.loads(x.read_text())["ns"] | r["ns"]                    # extra never overwrites a main arm
                pk = {k: v for k, v in torch.load(OUT / x.name.replace("res_", "picks_").replace(".json", ".pt")).items() if k != "best"} | pk
            den = r["hit"]["bayes"] - 1 / K
            hitv = {k: (v == pk["best"]).float() for k, v in pk.items() if k != "best"}
            se = {k: float((hitv[k] - hitv["dj02"]).std() / len(hitv[k]) ** 0.5 / den) for k in hitv if k != "dj02"}
            rows.append(dict(f=r["f"], q=r["q"], seed=r["seed"], bayes_hit=round(r["hit"]["bayes"], 4),
                             ns={k: round(v, 3) for k, v in r["ns"].items()}, se_vs_dj02={k: round(v, 3) for k, v in se.items()},
                             delta_over_eps=r.get("delta_over_eps")))
            print(json.dumps(rows[-1]))
        (OUT / "table.json").write_text(json.dumps(rows, indent=1))
    elif sys.argv[1] == "sweep":                                                 # A3 power sweep: exact model, no training
        out = []
        step = lambda z, u: true_step(z, u, dyn)
        import os
        sw_seed, sw_n = int(os.environ.get("R2_SWEEP_SEED", 77)), int(os.environ.get("R2_SWEEP_N", 1000))
        for f in (0.0, 1.0):
            for sg in [float(x) for x in sys.argv[2:]]:
                ev = make_cell(sw_n, f, sg, sw_seed, dyn)
                d, s = lin1_feats(step, ev["y"], ev["a"], ev["g"], sg)
                Pb, _ = bayes_P(ev, dyn, sg, f, 1.0, M=2048, seed=3)
                row = dict(f=f, sigma=sg, bayes=hit(Pb.argmax(1), ev), dist=hit(d.argmin(1), ev),
                           nn=float(ev["dist"].min(1).values.median()))
                out.append(row); print(row, flush=True)
        (OUT / f"sweep_s{sw_seed}.json").write_text(json.dumps(out, indent=1))
    elif sys.argv[1] == "pred":
        rep = {}
        for sd in (0, 1, 2):
            net, r1, rH = train_predictor(sd, dyn)
            torch.save(net.state_dict(), OUT / f"pred_s{sd}.pt")
            rep[sd] = dict(r2_one=r1, r2_H=rH)
            print(sd, rep[sd], flush=True)
        (OUT / "pred.json").write_text(json.dumps(rep, indent=1))
        (OUT / "calib.json").write_text(json.dumps(calibrate(dyn), indent=1))
        print(open(OUT / "calib.json").read())
    else:
        calib = json.loads((OUT / "calib.json").read_text())
        a = [float(x) for x in sys.argv[2:]]
        for f, q in zip(a[::2], a[1::2]):
            for sd in (0, 1, 2):
                sfx = __import__("os").environ.get("R2_SFX", "_extra") if __import__("os").environ.get("R2_KINDS") else ""
                if not (OUT / f"res_f{f}_q{q}_s{sd}{sfx}.json").exists():         # resume: never redo a finished seed
                    run_cell(f, q, sd, calib, dyn)

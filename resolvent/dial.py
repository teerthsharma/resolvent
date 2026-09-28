"""Bed `dial`: the shared-error dial at K = 63 with a learned predictor.

Ported from experiments/dial/r2/r2.py (bar experiments/dial/r2/BAR.md, amendments A1-A6) with the A6.1 Bayes tie-break of
experiments/dial/r3/foreman/r3.py. CPU only. Latent D = 8, action 4 per step, horizon 5, true dynamics

    z' = 1.1 Q z + 0.5 tanh(C z + B a),    Q Haar-orthogonal,

K = 63 candidates per start, realised start of candidate k  x_k = y - sigma (sqrt(f) s + sqrt(1 - f) u_k).
The label is the truly best candidate; NS = (hit - 1/K) / (hit_Bayes - 1/K), Bayes by M common draws under the
true dynamics, exact ties broken by the smallest posterior-mean true distance.

    python -m resolvent.dial F Q SEED      one cell at the source sizes (hours on CPU), JSON on stdout
"""
import json
import math
import sys
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .djepa import RelationalOperator, rank01

K, D, M_ACT, H, NTOK = 63, 8, 4, 5, 12
TAU = 0.05
SIGMA = {1.0: 0.36814846098490056, 3.0: 1.1433498263409247, 10.0: 3.976951003062853}   # source results/calib.json
KINDS = ("hop1", "dj02", "dj0.5", "dj4L", "null")                                        # round-3 arms (A6.2)
# fast cell: sign of dj4L > dj02 at f = 1 held on seeds 0, 1, 2 at these sizes (hit margin 0.0055 / 0.0105 / 0.002)
FAST = dict(n_train=4000, n_eval=2000, M=128, steps=300, pred_steps=300, lr=3e-3, kinds=("dj02", "dj4L", "null"))


# ---- true dynamics and the bed --------------------------------------------------------------------------------
def make_dyn(seed=12345):
    g = torch.Generator().manual_seed(seed)
    Q, R = torch.linalg.qr(torch.randn(D, D, generator=g))
    Q = Q * torch.sign(torch.diagonal(R))                                                 # Haar
    return dict(A=1.1 * Q, C=torch.randn(D, D, generator=g) / D ** 0.5, B=torch.randn(D, M_ACT, generator=g) / M_ACT ** 0.5)


def true_step(z, a, dyn):
    return z @ dyn["A"].T + 0.5 * torch.tanh(z @ dyn["C"].T + a @ dyn["B"].T)


def rollout(step, y, a):
    """y (n, D) shared start or per-candidate starts (n, K, D); a (..., K, H, m) -> (..., K, D)."""
    z = y[..., None, :].expand(*a.shape[:-3], a.shape[-3], D) if y.dim() == a.dim() - 2 else y
    for t in range(a.shape[-2]):
        z = step(z, a[..., t, :])
    return z


def start_error(prefix, f, gen):
    """sqrt(f) s + sqrt(1 - f) u_k, shape (*prefix, K, D): a fraction f of the variance is shared by all K."""
    return math.sqrt(f) * torch.randn(*prefix, 1, D, generator=gen) + math.sqrt(1 - f) * torch.randn(*prefix, K, D, generator=gen)


def make_cell(n, f, sigma, seed, dyn):
    g = torch.Generator().manual_seed(seed)
    step = partial(true_step, dyn=dyn)
    y = torch.randn(n, D, generator=g)
    a = torch.randn(n, K, H, M_ACT, generator=g)
    goal = rollout(step, y, torch.randn(n, 1, H, M_ACT, generator=g))[:, 0]
    z = rollout(step, y[:, None] - sigma * start_error((n,), f, g), a)
    dist = (z - goal[:, None]).norm(dim=-1)
    return dict(y=y, a=a, g=goal, dist=dist, best=dist.argmin(1))


def hit(pick, b):
    return float((pick == b["best"]).float().mean())


@torch.no_grad()
def bayes_P(b, dyn, sigma, f, r, M=2048, seed=0, chunk=16):
    """MC over the start error with the TRUE dynamics: P(k best), P(|z_k - g| < r), E|z_k - g|, each (n, K)."""
    gen = torch.Generator().manual_seed(seed)
    Pb, Ps, Ed = [], [], []
    for i in range(0, len(b["y"]), chunk):
        y, a, goal = (b[k][i:i + chunk] for k in ("y", "a", "g"))
        z = y[:, None, None] - sigma * start_error((len(y), M), f, gen)
        for t in range(H):
            z = true_step(z, a[:, None, :, t], dyn)
        dist = (z - goal[:, None, None]).norm(dim=-1)                                       # (c, M, K)
        Pb.append(Fn.one_hot(dist.argmin(-1), K).float().mean(1))
        Ps.append((dist < r).float().mean(1))
        Ed.append(dist.mean(1))
    return torch.cat(Pb), torch.cat(Ps), torch.cat(Ed)


def bayes_pick(Pb, Ed, allowed=None):
    """argmax P(k best) over allowed k; exact ties go to the smallest posterior-mean true distance (A6.1)."""
    P = Pb if allowed is None else Pb.masked_fill(~allowed, -1.0)
    top = P == P.max(1, keepdim=True).values
    return Ed.masked_fill(~top, float("inf")).argmin(1)


# ---- LIN1 and tokens ------------------------------------------------------------------------------------------
def lin1_feats(step, y, a, goal, sigma, chunk=250):
    """d_k = |z_hat_k - g|, s_k = sigma |J_k^T n_k|: one VJP per candidate (all candidates in one backward)."""
    ds, ss = [], []
    for i in range(0, len(y), chunk):
        yy = y[i:i + chunk][:, None].expand(-1, a.shape[1], -1).clone().requires_grad_(True)
        with torch.enable_grad():
            z = rollout(step, yy, a[i:i + chunk])
            diff = z - goal[i:i + chunk][:, None]
            d = diff.norm(dim=-1)
            (gy,) = torch.autograd.grad((z * (diff / d[..., None]).detach()).sum(), yy)
        ds.append(d.detach())
        ss.append(sigma * gy.norm(dim=-1))
    return torch.cat(ds), torch.cat(ss)


def lin1_prob(d, s, r):
    return torch.special.ndtr((r - d) / s)


def lin1_pick(d, s, r):
    return (torch.special.log_ndtr((r - d) / s) - 1e-9 * d).argmax(1)                     # log form: no underflow ties


def tokens(step, b, sigma, r):
    """12-d token [LN(z_hat - g); d; log s; distance rank; LIN1 rank], plus both base rankings."""
    d, s = lin1_feats(step, b["y"], b["a"], b["g"], sigma)
    with torch.no_grad():
        zh = torch.cat([rollout(step, b["y"][i:i + 1000], b["a"][i:i + 1000]) for i in range(0, len(b["y"]), 1000)])
    rr, lr = rank01(d), rank01(-(torch.special.log_ndtr((r - d) / s) - 1e-9 * d))
    v = torch.cat([Fn.layer_norm(zh - b["g"][:, None], (D,)), d[..., None], torch.log(s + 1e-12)[..., None],
                   rr[..., None], lr[..., None]], -1)
    return v, rr, lr, d, s


# ---- heads ----------------------------------------------------------------------------------------------------
def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, o))


WIDTH = {"point": 256, "hop1": 97}                                                        # params within 10 % of dj


class DialHead(nn.Module):
    """forward(v, base) -> logits, higher = better. 'dj' + eps code ('02' = 0.2, '0.5', '4') + 'L' for LIN1 base."""

    def __init__(self, kind, nin=NTOK):
        super().__init__()
        self.kind = kind
        if kind.startswith("dj"):
            code = kind[2:].rstrip("L")
            self.op = RelationalOperator(nin, 64, 0.2 if code == "02" else float(code))
            self.eps = self.op.eps
        elif kind == "point":
            self.net = mlp(nin, WIDTH[kind], 1)
        else:
            W = WIDTH[kind]
            self.enc, self.out = mlp(nin, W, W), mlp(2 * W, W, 1)
            self.q, self.k = nn.Linear(W, W), nn.Linear(W, W)
            self.theta = nn.Parameter(torch.zeros(()))

    def forward(self, v, base):
        if self.kind.startswith("dj"):
            s, self.last_delta = self.op(v, base)
            return -s / TAU
        if self.kind == "point":
            return self.net(v).squeeze(-1)
        h = self.enc(v)
        logit = (self.q(h) @ self.k(h).transpose(1, 2)) / h.shape[-1] ** 0.5
        W = torch.softmax(logit.masked_fill(torch.eye(v.shape[1], dtype=torch.bool), -1e9), -1)
        m = h + 0.99 * torch.sigmoid(self.theta) * W @ h
        return self.out(torch.cat([h, m], -1)).squeeze(-1)


def saturation(head, v, base):
    """Mean |delta| / eps of a D-JEPA head: 1 means the tanh correction runs at its bound."""
    head(v, base)
    return float(head.last_delta.abs().mean() / head.eps)


def base_for(kind, rr, lr):
    return lr if kind.endswith("L") else rr


def train_head(kind, v, base, best, seed=0, steps=4000, bs=64, lr=1e-3):
    torch.manual_seed(seed)
    h = DialHead(kind)
    opt = torch.optim.AdamW(h.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, len(v), (bs,), generator=g)
        o = h(v[idx], base[idx])
        loss = Fn.cross_entropy(o, best[idx])
        if kind.startswith("dj"):
            loss = loss + 0.1 * (h.last_delta ** 2).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(h.parameters(), 1.0)
        opt.step()
        sch.step()
    return h.eval()


# ---- learned predictor ----------------------------------------------------------------------------------------
class Pred(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp(D + M_ACT, 256, D)

    def forward(self, z, a):
        return self.net(torch.cat([z, a], -1))


def _transitions(n, gen, dyn):
    z = torch.randn(n, D, generator=gen)
    a = torch.randn(n, H, M_ACT, generator=gen)
    Z, A, Z1 = [], [], []
    for t in range(H):
        z1 = true_step(z, a[:, t], dyn)
        Z.append(z), A.append(a[:, t]), Z1.append(z1)
        z = z1
    return torch.cat(Z), torch.cat(A), torch.cat(Z1)


def train_predictor(seed, dyn, steps=2000, bs=2048):
    """MLP one-step predictor on 500,000 true transitions; frozen. Returns (net, one-step R^2, H-step R^2)."""
    gen = torch.Generator().manual_seed(500 + seed)
    Z, A, Z1 = _transitions(100_000, gen, dyn)
    torch.manual_seed(seed)
    net = Pred()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        i = torch.randint(0, len(Z), (bs,), generator=gen)
        loss = Fn.mse_loss(net(Z[i], A[i]), Z1[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
    net.eval().requires_grad_(False)

    def r2(p, t):
        return float(1 - ((p - t) ** 2).sum() / ((t - t.mean(0)) ** 2).sum())

    Zt, At, Z1t = _transitions(10_000, gen, dyn)
    with torch.no_grad():
        y, a = torch.randn(2000, D, generator=gen), torch.randn(2000, 1, H, M_ACT, generator=gen)
        return net, r2(net(Zt, At), Z1t), r2(rollout(net, y, a)[:, 0], rollout(partial(true_step, dyn=dyn), y, a)[:, 0])


@torch.no_grad()
def calibrate(dyn, ratios=(1.0, 3.0, 10.0), n=4000):
    """rho = mean candidate spread of the true rollout; sigma(q) by bisection so the start error spreads q rho."""
    gen = torch.Generator().manual_seed(999)
    step = partial(true_step, dyn=dyn)
    y, a = torch.randn(n, D, generator=gen), torch.randn(n, 16, H, M_ACT, generator=gen)
    z0 = rollout(step, y, a)
    rho = float(z0.std(1).pow(2).mean(-1).sqrt().mean())
    e = torch.randn(n, 16, D, generator=gen)

    def spread(sg):
        return float((rollout(step, y[:, None] - sg * e, a) - z0).std((0, 1)).pow(2).mean().sqrt())

    out = dict(rho=rho, sigma={})
    for q in ratios:
        lo, hi = 1e-3, 100.0
        for _ in range(40):
            mid = math.sqrt(lo * hi)
            lo, hi = (mid, hi) if spread(mid) < q * rho else (lo, mid)
        out["sigma"][str(q)] = math.sqrt(lo * hi)
    return out


# ---- a cell ---------------------------------------------------------------------------------------------------
def run_cell(f, q, seed, dyn, n_train=60_000, n_eval=20_000, M=2048, steps=4000, pred_steps=2000, kinds=KINDS, lr=1e-3):
    """One (f, sigma/rho) cell for one seed: hit and NS per arm, |delta|/eps per D-JEPA arm, closable gap in hit."""
    sigma = SIGMA[q]
    net = train_predictor(seed, dyn, steps=pred_steps)[0]
    cid = round(f * 100) * 100 + round(q * 10)
    tr = make_cell(n_train, f, sigma, 10_000 * seed + cid + 1, dyn)
    ev = make_cell(n_eval, f, sigma, 10_000 * seed + cid + 2, dyn)
    r = float(tr["dist"].min(1).values.median())
    vt, rt, lt, _, _ = tokens(net, tr, sigma, r)
    ve, re_, le, de, se = tokens(net, ev, sigma, r)
    Pb, _, Ed = bayes_P(ev, dyn, sigma, f, r, M=M, seed=seed)
    picks = {"bayes": bayes_pick(Pb, Ed), "dist": de.argmin(1), "lin1": lin1_pick(de, se, r)}
    dsat = {}
    for kd in kinds:
        lab = tr["best"]
        if kd == "null":
            lab = lab[torch.randperm(len(lab), generator=torch.Generator().manual_seed(seed))]
        k2 = "point" if kd == "null" else kd
        h = train_head(k2, vt, base_for(k2, rt, lt), lab, seed=seed, steps=steps, lr=lr)
        bse = base_for(k2, re_, le)
        with torch.no_grad():
            chunks = [(ve[i:i + 1000], bse[i:i + 1000]) for i in range(0, n_eval, 1000)]
            picks[kd] = torch.cat([h(v, b).argmax(1) for v, b in chunks])
            if k2.startswith("dj"):
                dsat[kd] = sum(saturation(h, v, b) for v, b in chunks) / len(chunks)
    hits = {k: hit(p, ev) for k, p in picks.items()}
    return dict(f=f, q=q, seed=seed, sigma=sigma, r=r, n_train=n_train, n_eval=n_eval, M=M, steps=steps, hit=hits,
                ns={k: (x - 1 / K) / (hits["bayes"] - 1 / K) for k, x in hits.items()}, delta_over_eps=dsat,
                gap_hit=hits["bayes"] - hits["dist"])


if __name__ == "__main__":
    f, q, seed = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
    print(json.dumps(run_cell(f, q, seed, make_dyn()), indent=1))

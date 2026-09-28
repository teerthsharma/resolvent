# R-JEPA / Cameron: standard-map candidate bed, exact Bayes ceiling, rankers, H0 of the candidate cloud.
# Bar: sun/rjepa/cameron/BAR.md (with Amendment A1). D-JEPA operator reimplemented from arXiv 2609.24749's
# specification (Liu et al. 2026; spec as recorded in sun/rjepa/wilson/facts.json), not from its code.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

K, KS = 8, 1.5
TWO_PI = 2 * np.pi
G = np.array([0.5, 1.0])
R = np.sqrt(np.pi / 2)                       # goal-ball area = torus area / K


def step(x):
    p = (x[..., 1] + KS * np.sin(x[..., 0])) % TWO_PI
    return np.stack([(x[..., 0] + p) % TWO_PI, p], -1)


def roll(x, T):
    for _ in range(T):
        x = step(x)
    return x


def wrap(d):
    return (d + np.pi) % TWO_PI - np.pi


def tdist(a, b):
    return np.linalg.norm(wrap(a - b), axis=-1)


def torus_dmat(Q):
    return tdist(Q[:, None], Q[None])


def h0_bars(P, metric="torus"):
    """Finite H0 deaths of the Vietoris-Rips filtration = MST edge lengths (Prim, O(n^2)), sorted."""
    D = torus_dmat(P) if metric == "torus" else np.linalg.norm(P[:, None] - P[None], axis=-1)
    n = len(P); best = D[0].copy(); used = np.zeros(n, bool); used[0] = True; out = []
    for _ in range(n - 1):
        b = np.where(used, np.inf, best); j = int(b.argmin()); out.append(b[j])
        used[j] = True; best = np.minimum(best, D[j])
    return np.sort(np.array(out))


def jac(x, T):
    """Exact tangent-linear Jacobian of F^T at x (what an autograd JVP of the predictor returns; 2 tangent passes)."""
    A = np.broadcast_to(np.eye(2), x.shape[:-1] + (2, 2)).copy()
    for _ in range(T):
        c = KS * np.cos(x[..., 0])
        Js = np.empty(A.shape); Js[..., 0, 0] = 1 + c; Js[..., 0, 1] = 1; Js[..., 1, 0] = c; Js[..., 1, 1] = 1
        A = Js @ A; x = step(x)
    return A




def make_batch(rng, N, T, delta=1e-3, need_post=True, M=256, chunk=500):
    x0 = rng.uniform(0, TWO_PI, (N, 2))
    u = rng.uniform(-1, 1, (N, K, 2))
    y = (x0 + delta * rng.standard_normal((N, 2))) % TWO_PI
    z = roll(x0[:, None] + u, T); succ = tdist(z, G) < R
    zh = roll(y[:, None] + u, T); d = tdist(zh, G)
    b = dict(T=T, delta=delta, y=y, u=u, succ=succ, zh=zh, d=d, J=jac(y[:, None] + u, T))
    if need_post:
        Pb = np.empty((N, K)); P3 = np.empty((N, K))
        for i in range(0, N, chunk):
            xm = y[i:i + chunk, None] + delta * rng.standard_normal((min(chunk, N - i), M, 2))   # exact flat posterior
            zm = roll(xm[:, None] + u[i:i + chunk, :, None], T)
            hit = tdist(zm, G) < R
            Pb[i:i + chunk] = hit.mean(-1); P3[i:i + chunk] = hit[..., :3].mean(-1)
        b["P_bayes"], b["P_ens3"] = Pb, P3
    return b


def lin_prob(b, rng, S=256, chunk=1000):
    xi = rng.standard_normal((S, 2))                 # common draws across candidates
    N = len(b["d"]); out = np.empty((N, K))
    for i in range(0, N, chunk):
        J = b["J"][i:i + chunk]; zh = b["zh"][i:i + chunk]
        zs = zh[..., None, :] + b["delta"] * np.einsum("nkab,sb->nksa", J, xi)
        out[i:i + chunk] = (tdist(zs, G) < R).mean(-1)
    return out


def lin1_prob(b):
    """Chance score with one VJP per candidate: Phi((r - d)/s), s = delta*||J^T n||, n the unit goal direction."""
    from scipy.special import ndtr
    n = wrap(b["zh"] - G) / b["d"][..., None]
    s = b["delta"] * np.linalg.norm(np.einsum("nkab,nka->nkb", b["J"], n), axis=-1)
    return ndtr((R - b["d"]) / s)


def sigma(b):
    return b["delta"] * np.linalg.norm(b["J"], ord=2, axis=(-2, -1))


def _tb(P, d):                                         # argmax P, ties to the latent-distance floor
    return (P - 1e-4 * d).argmax(1)


def pick_floor(b):
    return b["d"].argmin(1)


def pick_bayes(b):
    return _tb(b["P_bayes"], b["d"])


def pick_ens3(b):
    return _tb(b["P_ens3"], b["d"])


def pick_lin(b, rng):
    return _tb(lin_prob(b, rng), b["d"])


def pick_cert(b, rng):
    ok = sigma(b).max(1) < R / 2
    return np.where(ok, pick_floor(b), rng.integers(0, K, len(ok))), ok


def score(b, pick):
    return b["succ"][np.arange(len(pick)), pick].mean()


# ---------------- D-JEPA ordinal operator (spec reimplementation) ----------------

class DJ(nn.Module):
    def __init__(self, nin, width=64, eps=0.2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(nin, width), nn.LayerNorm(width), nn.GELU())
        layer = nn.TransformerEncoderLayer(width, 4, 128, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.down, self.up = nn.Linear(width, 8), nn.Linear(8, 1)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)
        self.eps = eps

    def forward(self, v, base):
        h = self.tf(self.enc(v))
        delta = self.eps * torch.tanh(self.up(torch.tanh(self.down(h)))).squeeze(-1)
        return base + delta, delta


def rank01(c):                                        # (rank(c) - 1)/(K - 1), rank 1 = lowest cost
    return c.argsort(1).argsort(1) / (K - 1)


def tokens(b, variant, P_lin=None):
    e = lambda z: np.concatenate([np.cos(z), np.sin(z)], -1)
    dz = torch.tensor(e(b["zh"]) - e(G), dtype=torch.float32)
    dz = F.layer_norm(dz, dz.shape[-1:])
    r = rank01(b["d"])
    if variant == "LINDJ":
        rl = rank01(-(P_lin - 1e-4 * b["d"]))
        extra, base = [rl, r], rl
    elif variant == "DJs":
        extra, base = [r, np.log(sigma(b) + 1e-12)], r
    else:
        extra, base = [r], r
    v = torch.cat([dz] + [torch.tensor(x, dtype=torch.float32)[..., None] for x in extra], -1)
    return v, torch.tensor(base, dtype=torch.float32)


def dj_loss(s, delta, y, T=0.05, margin=0.02):
    lp = torch.log_softmax(-s / T, 1)
    has = y.any(1)
    l_set = -(torch.logsumexp(lp.masked_fill(~y, -1e9), 1))[has].mean() if has.any() else s.sum() * 0
    pair = (y[:, :, None] & ~y[:, None, :]).float()                     # (pos i, neg j); K = 8 <= 16, so all candidates
    l_loc = (F.softplus((margin + s[:, :, None] - s[:, None, :]) / T) * pair).sum() / pair.sum().clamp(min=1)
    return l_set + 0.25 * l_loc + 0.1 * (delta ** 2).mean()


def train_dj(v, base, y, steps=1000, bs=8, seed=0):
    torch.manual_seed(seed)
    net = DJ(v.shape[-1])
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    y = torch.tensor(y); g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, len(v), (bs,), generator=g)
        s, dl = net(v[idx], base[idx])
        loss = dj_loss(s, dl, y[idx])
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    return net


@torch.no_grad()
def pick_dj(net, v, base):
    return net(v, base)[0].argmin(1).numpy()


def auc(x, e):
    """Mann-Whitney AUC of feature x for binary event e (ties averaged)."""
    from scipy.stats import rankdata
    rk = rankdata(x); n1 = e.sum(); n0 = len(e) - n1
    return float((rk[e].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")

"""Bed `shift`: an exact-truth latent bed whose prediction error is shared by every candidate of a start.

D = 2, K = 4, goal g = 0. The planner observes y ~ N(0, 9 I); the true start is s = y - sigma xi, so
s | y ~ N(y, sigma^2 I) exactly. Candidates a_k = -y + rho r_k aim at the goal from the estimate;
executed outcomes z_k = s + a_k + sigma_e eta_k; truth best = argmin_k |z_k|.

The start error s - y is common to all K candidates: it translates the whole predicted set. The Bayes
rule argmax_k P(k best | y) is then a function of the whole set (the Gaussian mass of each candidate's
Voronoi cell, shifted), while latent distance ranks each candidate alone. As sigma grows the Bayes
rule tends to the exterior angle of each candidate's Voronoi cell, a convex-hull property.
"""
import numpy as np
import torch
import torch.nn as nn

from .djepa import TEMP
from .heads import Head, fit, mlp

K, D, RHO, SIG_E = 4, 2, 1.0, 0.02


def make_bed(n, sigma, sigma_e=SIG_E, seed=0, mix=0.0):
    """mix = share of the error variance sigma^2 that is idiosyncratic per candidate (0 = all shared)."""
    rng = np.random.default_rng([seed, int(sigma * 1000), int(sigma_e * 1000)] + ([int(mix * 1000)] if mix else []))
    sh = sigma * np.sqrt(1 - mix)
    sid = sigma_e if mix == 0 else np.sqrt(sigma_e ** 2 + sigma ** 2 * mix)
    y = 3 * rng.standard_normal((n, D))
    s = y - sh * rng.standard_normal((n, D))
    r = rng.standard_normal((n, K, D))
    a = -y[:, None] + RHO * r
    z = s[:, None] + a + sid * rng.standard_normal((n, K, D))
    return dict(y=y, a=a, r=r, z=z, best=np.linalg.norm(z, axis=-1).argmin(1),
                zhat_true=y[:, None] + a, sigma=sh, sigma_e=sid, seed=seed)


def hit(pick, b):
    return float((pick == b["best"]).mean())


def dist_pick(zhat):
    """Latent-distance planning: the candidate predicted nearest the goal."""
    return np.linalg.norm(zhat, axis=-1).argmin(1)


def bayes_pick(b, M=1024, chunk=256, max_rank=None):
    """Top-1 Bayes rule by M posterior draws of (s, eta), common across the K candidates.
    max_rank restricts it to candidates whose plug-in distance rank is <= max_rank: with max_rank = 1
    it is the ceiling of any operator obeying D-JEPA's Cor 1 at K = 4, eps = 0.2."""
    rng = np.random.default_rng([b["seed"], 7])
    out = []
    for i in range(0, len(b["y"]), chunk):
        y, a = b["y"][i:i + chunk], b["a"][i:i + chunk]
        s = y[:, None] - b["sigma"] * rng.standard_normal((len(y), M, D))
        z = s[:, :, None] + a[:, None] + b["sigma_e"] * rng.standard_normal((len(y), M, K, D))
        w = np.linalg.norm(z, axis=-1).argmin(-1)
        P = np.stack([(w == k).mean(1) for k in range(K)], 1)
        if max_rank is not None:
            rk = np.linalg.norm(y[:, None] + a, axis=-1).argsort(1).argsort(1)
            P = np.where(rk <= max_rank, P, -1.0)
        out.append(P.argmax(1))
    return np.concatenate(out)


def hull_angle_share(P, n_dir=4096, chunk=256):
    """Share of directions u for which each candidate is argmax_k P_k . u: the exterior angle of its
    Voronoi cell, its limiting Bayes win probability as a shared isotropic error grows. 0 inside the hull."""
    th = np.linspace(0, 2 * np.pi, n_dir, endpoint=False)
    U = np.stack([np.cos(th), np.sin(th)], 1)
    P = P.reshape(-1, *P.shape[-2:])
    out = []
    for i in range(0, len(P), chunk):                      # (chunk, K, n_dir) at a time: bounded memory
        w = (P[i:i + chunk] @ U.T).argmax(1)               # the candidate furthest along each u
        out.append(np.stack([(w == k).mean(-1) for k in range(P.shape[-2])], -1))
    return np.concatenate(out)


def hull_angle_pick(b):
    return hull_angle_share(b["r"], 2048).argmax(1)


def jepa_predictor(tr, seed, epochs=4):
    """MLP f(y, a_k) -> z_hat_k trained by latent MSE on executed outcomes (target encoder = identity)."""
    X = torch.tensor(np.concatenate([np.repeat(tr["y"][:, None], K, 1), tr["a"]], -1).reshape(-1, 2 * D),
                     dtype=torch.float32)
    Y = torch.tensor(tr["z"].reshape(-1, D), dtype=torch.float32)
    f = fit(mlp(2 * D, 64, D), X, Y, nn.functional.mse_loss, epochs=epochs, seed=seed)

    def pred(b):
        with torch.no_grad():
            x = torch.tensor(np.concatenate([np.repeat(b["y"][:, None], K, 1), b["a"]], -1), dtype=torch.float32)
            return f(x).numpy()
    return pred


def run_cell(sigma, seeds=(0, 1, 2), n_train=100_000, n_eval=4000, kinds=("point", "hop1", "resolvent"), epochs=36):
    """Train a JEPA predictor, then each head by cross-entropy on the executed best index; score on held-out
    starts. closure = (head - dist) / (Bayes - dist), both read on the learned predictor's z_hat."""
    res = dict(sigma=sigma, bayes=[], dist=[], hit={k: [] for k in kinds}, agree={k: [] for k in kinds},
               closure={k: [] for k in kinds})
    for sd in seeds:
        tr, ev = make_bed(n_train, sigma, seed=100 + sd), make_bed(n_eval, sigma, seed=sd)
        pred = jepa_predictor(tr, sd)
        zt, ze = pred(tr), pred(ev)
        bp = bayes_pick(ev)
        hb, hd = hit(bp, ev), hit(dist_pick(ze), ev)
        res["bayes"].append(hb)
        res["dist"].append(hd)
        Xt, Yt = torch.tensor(zt), torch.tensor(tr["best"])
        for k in kinds:
            m = Head(k)

            def loss(o, y, m=m):                        # D-JEPA head: tempered CE + trust term on delta
                if not m.kind.startswith("djepa"):
                    return nn.functional.cross_entropy(o, y)
                return nn.functional.cross_entropy(o / TEMP, y) + 0.1 * (m.last_delta ** 2).mean()
            m = fit(m, Xt, Yt, loss, epochs=epochs, seed=sd)
            with torch.no_grad():
                p = m(torch.tensor(ze)).argmax(1).numpy()
            res["hit"][k].append(hit(p, ev))
            res["agree"][k].append(float((p == bp).mean()))
            res["closure"][k].append((hit(p, ev) - hd) / (hb - hd) if hb > hd else float("nan"))
    return res

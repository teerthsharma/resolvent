"""Bed `torus`: Chirikov standard map (Ks = 1.5) on the 2-torus, K = 8 candidates, exact flat posterior.

A start x0 is observed as y = x0 + delta xi. Candidate k perturbs the start by u_k and rolls T steps;
success = the rolled point lands in the goal ball of radius R around G (ball area = torus area / K).
The start error is shared by every candidate, and the map is chaotic, so the error grows with T.

Rankers (argmax, ties to the latent-distance floor):
    floor  latent distance of the rolled estimate y + u_k to G
    bayes  M posterior draws of the start per candidate, in-ball fraction   (the ceiling, K M T calls)
    ens3   the first 3 of those draws                                         (3 T calls per candidate)
    lin    tangent-linear Monte Carlo: z_hat + delta J xi over S common draws (shared draws across candidates)
    lin1   Phi((R - d) / (delta ||J^T n||)), n the unit goal direction        (one VJP per candidate)
"""
import numpy as np
import torch

K, KS = 8, 1.5
TWO_PI = 2 * np.pi
G = np.array([0.5, 1.0])
R = np.sqrt(np.pi / 2)


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


def jac(x, T):
    """Exact tangent-linear Jacobian of F^T at x (area-preserving: det = 1)."""
    A = np.broadcast_to(np.eye(2), x.shape[:-1] + (2, 2)).copy()
    for _ in range(T):
        c = KS * np.cos(x[..., 0])
        Js = np.empty(A.shape)
        Js[..., 0, 0], Js[..., 0, 1], Js[..., 1, 0], Js[..., 1, 1] = 1 + c, 1, c, 1
        A, x = Js @ A, step(x)
    return A


def make_batch(rng, N, T, delta=1e-3, need_post=True, M=256, chunk=200):
    x0 = rng.uniform(0, TWO_PI, (N, 2))
    u = rng.uniform(-1, 1, (N, K, 2))
    y = (x0 + delta * rng.standard_normal((N, 2))) % TWO_PI
    succ = tdist(roll(x0[:, None] + u, T), G) < R
    zh = roll(y[:, None] + u, T)
    b = dict(T=T, delta=delta, y=y, u=u, succ=succ, zh=zh, d=tdist(zh, G), J=jac(y[:, None] + u, T))
    if need_post:
        Pb, P3 = np.empty((N, K)), np.empty((N, K))
        for i in range(0, N, chunk):
            xm = y[i:i + chunk, None] + delta * rng.standard_normal((min(chunk, N - i), M, 2))
            inball = tdist(roll(xm[:, None] + u[i:i + chunk, :, None], T), G) < R
            Pb[i:i + chunk], P3[i:i + chunk] = inball.mean(-1), inball[..., :3].mean(-1)
        b["P_bayes"], b["P_ens3"] = Pb, P3
    return b


def lin_prob(b, rng, S=256, chunk=250):
    xi = rng.standard_normal((S, 2))
    out = np.empty(b["d"].shape)
    for i in range(0, len(out), chunk):
        zs = b["zh"][i:i + chunk, ..., None, :] + b["delta"] * np.einsum("nkab,sb->nksa", b["J"][i:i + chunk], xi)
        out[i:i + chunk] = (tdist(zs, G) < R).mean(-1)
    return out


def lin1_prob(b):
    """Chance of landing in the goal ball from one VJP per candidate: Phi((R - d) / s), s = delta ||J^T n||."""
    n = wrap(b["zh"] - G) / b["d"][..., None]
    s = b["delta"] * np.linalg.norm(np.einsum("nkab,nka->nkb", b["J"], n), axis=-1)
    return torch.special.ndtr(torch.from_numpy((R - b["d"]) / s)).numpy()


def _tb(P, d):
    return (P - 1e-4 * d).argmax(1)


def pick_floor(b):
    return b["d"].argmin(1)


def pick_bayes(b):
    return _tb(b["P_bayes"], b["d"])


def pick_ens3(b):
    return _tb(b["P_ens3"], b["d"])


def pick_lin(b, rng):
    return _tb(lin_prob(b, rng), b["d"])


def pick_lin1(b):
    return _tb(lin1_prob(b), b["d"])


def score(b, pick):
    return float(b["succ"][np.arange(len(pick)), pick].mean())

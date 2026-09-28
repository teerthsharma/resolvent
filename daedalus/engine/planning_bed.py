"""planning_consequence_v0: exact-truth candidate-selection bed for rankers (registry/bars_r2.json).

A COPY of Foreman's round-1 shared-error bed (sun/rjepa/foreman/rjepa.py: make_bed, bayes_pick, dist_pick,
Amendments A1-A2 of sun/rjepa/foreman/BAR.md), frozen here so edits to Foreman's working file cannot move a
registered bed. D = 2, K = 4, goal 0. y ~ N(0, 9 I); start s = y - sigma*xi (shared by every candidate);
a_k = -y + r_k, r_k ~ N(0, I); executed z_k = s + a_k + sigma_e*eta_k; truth = argmin_k |z_k|.

A ranker sees only the plug-in predicted outcomes zhat_k = y + a_k (feature view "predicted", F = 2), in
PLANNER ORDER (ascending |zhat_k|, the order a planner emits its own shortlist in), and is trained from
executed outcomes (the best index). Executed outcomes z and eval labels never leave the verifier's process.
The Bayes rule knows the generative model exactly (MC, M = 1024) and is the ceiling no honest ranker beats.
"""
from __future__ import annotations

import numpy as np

K, D, RHO, SIG_E = 4, 2, 1.0, 0.02


def make_bed(n, sigma, seed, sigma_e=SIG_E, K=K):  # bars_r4 r4_c: K is a bed parameter (default 4 = v0)
    rng = np.random.default_rng([int(seed), int(sigma * 1000), int(sigma_e * 1000)])
    y = 3 * rng.standard_normal((n, D))
    s = y - sigma * rng.standard_normal((n, D))
    r = rng.standard_normal((n, K, D))
    a = -y[:, None] + RHO * r
    z = s[:, None] + a + sigma_e * rng.standard_normal((n, K, D))
    zhat = y[:, None] + a
    order = np.linalg.norm(zhat, axis=-1).argsort(1, kind="stable")      # planner order
    take = lambda v: np.take_along_axis(v, order[..., None], 1)
    zhat, z, a = take(zhat), take(z), take(a)
    return dict(y=y, a=a, z=z, zhat=zhat, best=np.linalg.norm(z, axis=-1).argmin(1),
                sigma=sigma, sigma_e=sigma_e, seed=int(seed))


def features(b, view="predicted"):
    if view != "predicted":  # the executed view is truth: it never leaves this process
        raise ValueError(f"feature view {view!r} is not registered")
    return b["zhat"].astype(np.float32)


def bayes_pick(b, M=1024, chunk=256):
    rng = np.random.default_rng([b["seed"], 7])
    K = b["a"].shape[1]
    chunk = max(1, chunk * 4 // K)  # same memory per chunk at any K; chunking does not change the draws at K = 4
    out = []
    for i in range(0, len(b["y"]), chunk):
        y, a = b["y"][i:i + chunk], b["a"][i:i + chunk]
        s = y[:, None] - b["sigma"] * rng.standard_normal((len(y), M, D))
        z = s[:, :, None] + a[:, None] + b["sigma_e"] * rng.standard_normal((len(y), M, K, D))
        w = np.linalg.norm(z, axis=-1).argmin(-1)
        out.append(np.stack([(w == k).mean(1) for k in range(K)], 1).argmax(1))
    return np.concatenate(out)


def dist_pick(zhat):
    return np.linalg.norm(zhat, axis=-1).argmin(1)


def base_score(zhat):
    """D-JEPA's base b_i (distance rank scaled to [0, 1]) as a higher-is-better score: -rank/(K-1)."""
    r = np.linalg.norm(zhat, axis=-1).argsort(-1, kind="stable").argsort(-1, kind="stable")
    return -r / (zhat.shape[-2] - 1)


if __name__ == "__main__":  # self-check: planner order puts the distance pick at index 0; Bayes beats dist
    b = make_bed(4000, 1.5, 0)
    assert (dist_pick(b["zhat"]) == 0).all()
    assert np.allclose(base_score(b["zhat"])[:, 0], 0) and np.allclose(base_score(b["zhat"])[:, -1], -1)
    hb, hd = float((bayes_pick(b) == b["best"]).mean()), float((b["best"] == 0).mean())
    print({"bayes": hb, "dist": hd})
    assert hb > hd + 0.03

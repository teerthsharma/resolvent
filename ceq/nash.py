"""Quantal-response (logit Nash) equilibrium as the source of a token's stance.

STATUS: RETIRED AS A STANCE, KEPT AS AN INSTRUMENT (issue #2). Both kill
conditions in `tests/w7/test_w7_nash.py` fired on 5/5 training seeds: OOD NRMSE
5.2888 +- 0.6041 against the learned signed stance's 2.7333 +- 1.0235, and
4.6142 with both known defects repaired. The fixed point below is a genuine
equilibrium; it does not buy composition. Table and producer: docs/FAILS.md,
section 3.

THE PROBLEM THIS ADDRESSES. The signed path sum measured a real tier ladder --
attention 5.8198, APPNP 4.2107, signed 2.6151 OOD NRMSE over 5 seeds -- and
failed the bar that matters: every arm sat ABOVE 1.0, worse than predicting the
mean, on a held-out composition of two sign flips seen only separately.

A learned stance is a regression on magnitudes and has no reason to compose. A
best response is ORDINAL: it turns on the sign of a payoff difference, not its
size. Composition of sign flips is an ordinal operation.

THE CONSTRUCTION.

    s      = sigmoid((M s + b) / tau)      support probability, one per token
    stance = 2 s - 1  in  [-1, 1]
    A_ij   = rho * P_ij * stance_j

`M` is symmetrized, so the game is a potential game and its logit equilibrium is
a stationary point of the potential. `P` is an ordinary non-negative causal
softmax: who may talk to whom. The stance decides with what sign.

WHAT SURVIVES, BY CONSTRUCTION RATHER THAN BY LUCK.
`|A_ij| <= rho * P_ij` and `P` is row-stochastic, so `sum_j |A_ij| <= rho`. `A` is
strictly lower triangular. `CEQ.Nilpotent.pow_card_eq_zero` therefore still gives
`A^n = 0` -- its single hypothesis is `forall i j, i <= j -> A i j = 0` and it
carries no sign condition -- and `occupancy_is_exact_inverse` still gives an
exact terminating resolvent.

WHAT IS NOT NEEDED. Nash and QRE exist by Brouwer. No contraction certificate, no
Perron vector, no power iteration. The measured reducible-A failure mode --
power iteration returning `w` with exact zeros so the weighted sup norm is
undefined -- cannot arise because no `w` appears anywhere.

TEMPERATURE IS A WELL-POSEDNESS PARAMETER, NOT A HYPERPARAMETER. The map
`s -> sigmoid((Ms+b)/tau)` has Lipschitz constant `||M||_2 / (4 tau)`. Above
`tau* = ||M||_2 / 4` it is a contraction: the equilibrium is unique and the
iteration reaches it. Below `tau*` the game can have several equilibria and the
iterate need not converge to any particular one. `qre_stance` reports its own
residual rather than asserting convergence, and `safe_tau` computes the
threshold, because a stance read off a non-converged iterate is not an
equilibrium.

THE GEOMETRY. The multiplicative replicator form is natural-gradient ascent on
the simplex under the Fisher-Rao metric, which is why the iterate stays in [0,1]
with no projection step and no clamping.
"""
from __future__ import annotations

import math

import torch

DEFAULT_TAU = 1.0
DEFAULT_ITERS = 200
DEFAULT_TOL = 1e-13


def safe_tau(m: torch.Tensor, margin: float = 1.25) -> float:
    """Smallest temperature at which the QRE map is a contraction, times a margin.

    Lipschitz constant of `s -> sigmoid((Ms+b)/tau)` is `||M||_2 / (4 tau)`, since
    `sigmoid'` is bounded by 1/4. Below this the equilibrium need not be unique.
    """
    return margin * float(torch.linalg.matrix_norm(m.detach(), ord=2).max()) / 4.0


def qre_stance(m: torch.Tensor, b: torch.Tensor, tau: float = DEFAULT_TAU,
               iters: int = DEFAULT_ITERS, tol: float = DEFAULT_TOL,
               return_residual: bool = False):
    """Solve `s = sigmoid((M s + b) / tau)` by fixed-point iteration.

    Starts at the barycentre `s = 1/2`, which is the maximum-entropy point of the
    per-token simplex and the only start that privileges no token.

    Returns the residual on request rather than asserting convergence: below the
    contraction temperature the iteration can fail to converge, and a caller that
    silently accepts a non-converged iterate is reading a stance that is not an
    equilibrium. That is the same fail-open shape as a fixed point published
    unconditionally after a Banach iteration that did not converge.
    """
    def step(z):
        # m @ z works for [n,n]@[n] and for batched [...,n,n]@[...,n] alike
        return torch.sigmoid(((m @ z.unsqueeze(-1)).squeeze(-1) + b) / tau)

    s = torch.full(b.shape, 0.5, dtype=b.dtype, device=b.device)
    for _ in range(iters):
        nxt = step(s)
        done = float((nxt - s).detach().abs().max()) < tol
        s = nxt
        if done:
            break
    if return_residual:
        return s, float((s - step(s)).abs().max())
    return s


def affine_fit_ratio(m: torch.Tensor, b: torch.Tensor, star: torch.Tensor,
                     tau: float, radius: float, n_probe: int = 64,
                     seed: int = 0) -> float:
    """How far the update is from affine at `star`, as a ratio against a truly
    affine control probed identically.

    The radius sweep is not optional. A piecewise-affine update -- the max-plus
    star, for instance -- looks perfectly affine below its own cell radius, and
    scored a ratio of 5.5 there while a genuinely smooth update scored 1e5 and
    above. Reporting a single radius hides exactly the case the probe exists to
    catch.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = star.shape[-1]
    dx = torch.randn(n_probe, n, generator=g, dtype=star.dtype).to(star.device)
    dx = dx / dx.norm(dim=-1, keepdim=True) * radius

    def resid(fn) -> float:
        y = torch.stack([fn(star + d) - fn(star) for d in dx])
        j = torch.linalg.lstsq(dx, y).solution          # best affine map
        return float((dx @ j - y).norm() / (y.norm() + torch.finfo(y.dtype).tiny))

    f = lambda z: torch.sigmoid((m @ z + b) / tau)
    ctrl = lambda z: (m @ z + b) / tau                  # exactly affine
    return resid(f) / max(resid(ctrl), 1e-16)


def _causal_bool(s: int, device) -> torch.Tensor:
    return torch.ones(s, s, dtype=torch.bool, device=device).tril(-1)


def nash_operator(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                  *, rho: float = 0.9, tau: float | None = None,
                  iters: int = DEFAULT_ITERS) -> torch.Tensor:
    """A_ij = rho * P_ij * stance_j, with `stance` a logit-Nash equilibrium.

    `P` says who may talk to whom; the equilibrium stance says with what sign.
    Splitting the two is what keeps the L1 bound -- and therefore the whole Lean
    chain -- while letting the operator go negative.
    """
    s_len, d = query.shape[-2], query.shape[-1]
    scores = (query @ key.transpose(-2, -1)) / math.sqrt(d)
    m = _causal_bool(s_len, query.device)
    p = torch.softmax(scores.masked_fill(~m, torch.finfo(query.dtype).min),
                      dim=-1).masked_fill(~m, 0.0)

    g = (value @ value.transpose(-2, -1)) / math.sqrt(d)
    game = (g + g.transpose(-2, -1)) / 2.0              # symmetric => potential game
    bias = value.mean(-1)
    t = tau if tau is not None else safe_tau(game.reshape(-1, s_len, s_len))
    stance = 2.0 * qre_stance(game, bias, tau=t, iters=iters) - 1.0
    return rho * p * stance.unsqueeze(-2)

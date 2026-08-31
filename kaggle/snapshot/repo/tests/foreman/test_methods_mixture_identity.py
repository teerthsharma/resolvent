"""Binds the one algebraically-falsifiable claim in `METHODS.md` §2.

METHODS.md §2 proves the grid mixture is a supermartingale twice: once directly,
term by term, and once by the identity

    Ebar_t  =  Prod_{i<=t} ( 1 + lambar_{i-1} * d_i / B )

with `lambar_{i-1} = Sum_k w_k^{(t-1)} * lam^(k)` the posterior mean of lambda.
The second argument is the load-bearing one -- it is what licenses the sentence
"the grid mixture is exactly a single predictable-lambda process", and it is
what §4.3 leans on when it says hand re-weighting the prior destroys the
theorem. The direct term-by-term argument does not depend on it, so this test
does not gate the supermartingale property itself; it gates the REFRAME.

If the telescoping identity is wrong, the direct proof still stands and the
e-process is still valid, but METHODS.md §2's second half and §4.3's explanation
of the failure mode are both wrong and must be struck.

WHAT IS AND IS NOT TESTED HERE. This file tests an ALGEBRAIC IDENTITY between
two expressions in the same finite sequence of numbers. It is not a statistical
test, it does not simulate a null, and it does not exercise any e-process
implementation -- the must-fire null simulation (a null stream must NOT cross 20
in 10,000 replays at alpha=0.05) belongs to Chase's implementation and is not
duplicated here. A green here says the reframe is sound arithmetic; it says
nothing about whether the shipped process is calibrated.

NO THREAD DEPENDENCE. Nothing here calls BLAS; the arithmetic is pure Python
float. `torch.set_num_threads(2)` is pinned in this file anyway, per standing
policy, so that this file and not its launcher is the authority on the matter.

    python -m pytest tests/foreman/test_methods_mixture_identity.py -q
"""
from __future__ import annotations

import random

import torch

torch.set_num_threads(2)                       # pinned HERE, not by the launcher

# The grid and prior are arbitrary for an identity test -- the claim is that the
# identity holds for EVERY admissible grid and prior, so these are chosen to be
# awkward rather than round: a non-uniform prior, an endpoint at 0.0 (whose
# factor is identically 1 and must not perturb the posterior mean), and the
# boundary point 0.5.
LAM_GRID = (0.0, 0.05, 0.2, 0.5)
W_PRIOR = (0.1, 0.2, 0.3, 0.4)
B = 2.0                                        # METHODS.md §4.1's recommended C

TOL = 1e-12                                    # float64 round-off over <=64 steps


def _mixture(ds: list[float]) -> list[float]:
    """Ebar_t for t = 0..len(ds), by the DEFINITION: a fixed-weight sum."""
    out = [1.0]
    for t in range(1, len(ds) + 1):
        acc = 0.0
        for lam, w in zip(LAM_GRID, W_PRIOR):
            prod = 1.0
            for d in ds[:t]:
                prod *= 1.0 + lam * d / B
            acc += w * prod
        out.append(acc)
    return out


def _telescoped(ds: list[float]) -> tuple[list[float], list[float]]:
    """Ebar_t by METHODS.md §2's identity, plus the lambar it used at each step."""
    out = [1.0]
    lambars = []
    for t in range(1, len(ds) + 1):
        # posterior weights at t-1, from the PRIOR weights -- never re-normalised
        # toward the winning lambda (METHODS.md §4.3).
        num = []
        for lam, w in zip(LAM_GRID, W_PRIOR):
            prod = 1.0
            for d in ds[: t - 1]:
                prod *= 1.0 + lam * d / B
            num.append(w * prod)
        denom = sum(num)
        post = [n / denom for n in num]
        lambar = sum(p * lam for p, lam in zip(post, LAM_GRID))
        lambars.append(lambar)
        out.append(out[-1] * (1.0 + lambar * ds[t - 1] / B))
    return out, lambars


def test_mixture_equals_predictable_lambda_product():
    """METHODS.md §2: the two expressions agree at every t, on every stream."""
    rng = random.Random(0)
    for trial in range(200):
        n = rng.randint(1, 64)
        # |d_i| <= B is METHODS.md §4.1's hard precondition. Draw at and inside
        # the boundary, including the extremes, since that is where a factor
        # would go negative if the lambda cap were wrong.
        ds = [rng.uniform(-B, B) for _ in range(n)]
        if trial % 10 == 0:                    # exercise the boundary exactly
            ds[0] = -B
            ds[-1] = B
        direct = _mixture(ds)
        tele, lambars = _telescoped(ds)
        assert len(direct) == len(tele) == n + 1
        for t, (a, b) in enumerate(zip(direct, tele)):
            assert abs(a - b) <= TOL * max(1.0, abs(a)), (
                f"trial={trial} t={t} direct={a!r} telescoped={b!r}"
            )
        # lambar must be predictable-admissible: a convex combination of grid
        # points lies in [min grid, max grid], so it is in [0, 1/2] and the
        # single-lambda argument applies to it verbatim.
        for lb in lambars:
            assert min(LAM_GRID) - TOL <= lb <= max(LAM_GRID) + TOL, lb


def test_factors_stay_in_the_half_to_three_halves_band():
    """METHODS.md §2: |d| <= B and lam <= 1/2 put every factor in [1/2, 3/2].

    This is the nonnegativity argument, and it is the reason the cap is 1/2 and
    not 1. A factor of exactly 0 kills the process for the rest of the run.
    """
    rng = random.Random(1)
    for _ in range(2000):
        d = rng.uniform(-B, B)
        for lam in LAM_GRID:
            f = 1.0 + lam * d / B
            assert 0.5 - TOL <= f <= 1.5 + TOL, (lam, d, f)


def test_prior_weights_are_a_probability_vector():
    """§4.3's precondition, pinned by value so a later edit cannot drift it."""
    assert len(LAM_GRID) == len(W_PRIOR)
    assert abs(sum(W_PRIOR) - 1.0) <= TOL
    assert all(w >= 0.0 for w in W_PRIOR)
    assert all(0.0 <= lam <= 0.5 for lam in LAM_GRID)


def test_zero_difference_contributes_exactly_one():
    """§4.4's handling of an unusable seed: d_i = 0 must be no evidence at all.

    Not approximately one -- exactly one, at every grid point, so a recorded
    non-value cannot move the reading in either direction.
    """
    for lam in LAM_GRID:
        assert 1.0 + lam * 0.0 / B == 1.0

    rng = random.Random(2)
    ds = [rng.uniform(-B, B) for _ in range(16)]
    with_zero = ds[:8] + [0.0] + ds[8:]
    assert _mixture(ds)[-1] == _mixture(with_zero)[-1]


if __name__ == "__main__":
    test_mixture_equals_predictable_lambda_product()
    test_factors_stay_in_the_half_to_three_halves_band()
    test_prior_weights_are_a_probability_vector()
    test_zero_difference_contributes_exactly_one()
    print("METHODS.md §2 identity: OK")

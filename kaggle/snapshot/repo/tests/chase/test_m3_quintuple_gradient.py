"""CHASE Phase C - is the gradient the money run trains through actually right?

Subject: `scale/m3_quintuple.py::BatchedSettled`.

WHY THIS EXISTS. Contract 1.2 requires the implicit gradient to be gradchecked
against finite differences at rtol 1e-4 in float64, and Foreman's birth gate 2
did exactly that -- for `scale/arm_s.py::Settled`, which is the PROBABILITY
domain, per-example path. The money run trains through `BatchedSettled`, which
is the LOG domain, batched path. That is a different function and it arrived
with no gradcheck of its own. A settled cell trained through an unverified
gradient is not a measurement of settling; it is a measurement of whatever the
optimiser did with a wrong descent direction.

THE REFERENCE. Not finite differences on the fixed point -- differentiating a
fixed point by perturbing its inputs and re-solving is exactly the thing the
implicit form exists to avoid, and its own accuracy would then be the thing
under test. The reference here is the EXACT gradient obtained by unrolling the
settling loop with autograd switched on. At convergence the implicit gradient
and the unrolled gradient are the same object, so agreement is a real check and
disagreement localises to the Neumann truncation.

The must-fire at the bottom truncates the Neumann sum to one term and requires
the check to FAIL. Without it, a backward that returned the direct term alone --
ignoring (I - J)^-1 entirely -- would pass everything above.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scale import m3_quintuple as Q            # noqa: E402

BETA = 0.5
N_NEUMANN = 21


def _inputs(n=3, k=5, seed=0):
    """A well-conditioned batch of (log_gate, log_gram) in float64.

    log_gram must be a log of a POSITIVE Gram matrix; any finite matrix is a
    valid log, so a plain randn is a legitimate instance of the map's domain and
    does not smuggle in the structure the real Gram happens to have.
    """
    g = torch.Generator().manual_seed(seed)
    lg = torch.randn(n, k, generator=g, dtype=torch.float64)
    lG = torch.randn(n, k, k, generator=g, dtype=torch.float64)
    lg = lg - torch.logsumexp(lg, dim=-1, keepdim=True)
    return lg.requires_grad_(True), lG.requires_grad_(True)


def _grads(fn, lg, lG, w):
    lg = lg.detach().clone().requires_grad_(True)
    lG = lG.detach().clone().requires_grad_(True)
    out = fn(lg, lG)
    (out * w).sum().backward()
    return lg.grad.clone(), lG.grad.clone()


def _implicit(steps=400, n_neumann=N_NEUMANN):
    return lambda lg, lG: Q.BatchedSettled.apply(lg, lG, BETA, steps, n_neumann)


def _unrolled(steps=400):
    return lambda lg, lG: Q.batched_settle_log(lg, lG, BETA, steps)


def test_the_forward_reaches_a_fixed_point_at_all():
    """A gradient check at a point that is not a fixed point checks nothing."""
    lg, lG = _inputs()
    with torch.no_grad():
        la = Q.batched_settle_log(lg, lG, BETA, 400)
        nxt = Q.batched_log_alpha_step(la, lg, lG, BETA)
    assert float((nxt - la).abs().max()) < 1e-12, float((nxt - la).abs().max())


def test_implicit_gradient_matches_the_unrolled_gradient():
    """The money-run gradient against the exact one, float64, rtol 1e-4."""
    for seed in (0, 1, 2):
        lg, lG = _inputs(seed=seed)
        w = torch.randn(lg.shape, generator=torch.Generator().manual_seed(99),
                        dtype=torch.float64)
        gi = _grads(_implicit(), lg, lG, w)
        gu = _grads(_unrolled(), lg, lG, w)
        for name, a, b in (("d/d log_gate", gi[0], gu[0]),
                           ("d/d log_gram", gi[1], gu[1])):
            rel = float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))
            assert rel < 1e-4, (
                f"seed={seed} {name}: implicit gradient differs from the "
                f"unrolled one by relative {rel:.3e}; the money run's settled "
                f"cell is trained through this."
            )


def test_gradient_is_finite_and_nonzero():
    """A backward that silently returned zeros would pass a relative-error test
    against nothing, so the magnitude is pinned too."""
    lg, lG = _inputs()
    w = torch.ones_like(lg)
    gg, gG = _grads(_implicit(), lg, lG, w)
    for nm, t in (("log_gate", gg), ("log_gram", gG)):
        assert torch.isfinite(t).all(), nm
        assert float(t.abs().max()) > 1e-8, (nm, float(t.abs().max()))


def test_truncating_the_neumann_sum_to_one_term_breaks_it(monkeypatch):
    """MUST-FIRE CONTROL for every assertion above.

    N = 1 keeps only the direct term and drops (I - J)^-1 entirely. The
    truncation error bound is kappa**N/(1-kappa) = 0.5/0.5 = 1.0 at N = 1
    against 0.5**21/0.5 = 9.5367431640625e-07 at N = 21, so this MUST fail the
    1e-4 bar. If it does not, the test above is not reading the Neumann sum and
    the gradcheck is decoration.
    """
    lg, lG = _inputs()
    w = torch.randn(lg.shape, generator=torch.Generator().manual_seed(99),
                    dtype=torch.float64)
    gu = _grads(_unrolled(), lg, lG, w)
    g1 = _grads(_implicit(n_neumann=1), lg, lG, w)
    rel = float((g1[1] - gu[1]).abs().max() / gu[1].abs().max())
    assert rel > 1e-4, (
        f"a one-term Neumann sum reproduced the exact gradient to {rel:.3e}; "
        f"the N=21 agreement above therefore says nothing about the Neumann "
        f"sum being computed at all"
    )


def test_the_neumann_error_shrinks_as_N_grows():
    """Dose-response on the truncation, so agreement at N=21 is not a
    coincidence of one setting. kappa = beta = 0.5 exactly, so each extra term
    should roughly halve the error."""
    lg, lG = _inputs()
    w = torch.randn(lg.shape, generator=torch.Generator().manual_seed(99),
                    dtype=torch.float64)
    gu = _grads(_unrolled(), lg, lG, w)
    errs = []
    for n_neu in (1, 3, 6, 12, 21):
        gi = _grads(_implicit(n_neumann=n_neu), lg, lG, w)
        errs.append(float((gi[1] - gu[1]).abs().max()
                          / gu[1].abs().max()))
    assert errs == sorted(errs, reverse=True), errs
    assert errs[-1] < 1e-4 < errs[0], errs

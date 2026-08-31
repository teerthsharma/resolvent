"""The settling driver, bound to the convergence law it claims to verify.

ARM S needs something that iterates T to a fixed point and journals the Hilbert
residual per step. Foreman owns T's exact FORM; nothing owned the driver, so it is
written generic over T — any callable from the cone to the cone plugs in, and
fixing the form later changes nothing here.

The driver is calibrated against a map whose contraction ratio is known in
advance, because a settler that has never recovered a planted rate is not an
instrument. For a positive matrix acting on the cone the projective diameter is
the largest Hilbert distance between two of its COLUMNS; that characterisation is
checked here numerically against sampled pairs rather than asserted, since it is
the step where a wrong constant would hide.

The must-fire is the direction that matters: a map that does NOT contract has to
be reported as unconverged, never returned silently at the step cap with a
trajectory that looks finished.
"""
from __future__ import annotations

import itertools
import math

import pytest
import torch

from scale.hilbert import d_H, kappa_cert
from scale.settle import column_diameter, settle

torch.set_num_threads(2)


def _positive_matrix(n: int, seed: int, spread: float = 1.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, n, generator=g, dtype=torch.float64) * spread + 0.05


def _normalise(v: torch.Tensor) -> torch.Tensor:
    return v / v.sum()


# ------------------------------------------------- the diameter characterisation

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_column_diameter_is_attained_at_the_corners(seed):
    """The supremum lives on the EXTREME RAYS, and A e_i is exactly column i.

    Path A: the algebraic column formula. Path B: evaluate at the basis vectors
    directly. They must agree to machine precision, which is what makes the
    column formula the diameter rather than an estimate of it.
    """
    a = _positive_matrix(6, seed)
    eye = torch.eye(6, dtype=torch.float64)
    corners = max(d_H(a @ eye[i], a @ eye[j])
                  for i, j in itertools.combinations(range(6), 2))
    assert column_diameter(a) == pytest.approx(corners, rel=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_uniform_interior_sampling_UNDERESTIMATES_the_diameter(seed):
    """A defect in contract 1.1 as first written, bound here so it cannot return.

    1.1 defined the certified bound as `Delta_hat = max over SAMPLED pairs`. The
    supremum is attained at the corners of the cone, and uniform interior draws
    do not go near them: measured, they reach 39-53% of the true diameter, and
    the kappa implied by that is 0.31-0.33 against a truth of 0.56-0.69.

    So a sampled Delta_hat is a LOWER bound on the diameter presented as an UPPER
    bound on the contraction ratio, and it errs OPTIMISTIC — the arm looks like a
    stronger contraction than it is, and 1/(1-kappa) then under-budgets the
    Neumann truncation. The certificate must be evaluated at the extreme rays.
    """
    a = _positive_matrix(6, seed)
    truth = column_diameter(a)
    g = torch.Generator().manual_seed(seed + 500)
    sampled = 0.0
    for _ in range(4000):
        x = torch.rand(6, generator=g, dtype=torch.float64) + 1e-3
        y = torch.rand(6, generator=g, dtype=torch.float64) + 1e-3
        sampled = max(sampled, d_H(a @ x, a @ y))
    assert sampled <= truth + 1e-9, "sampling cannot exceed the true diameter"
    assert sampled < 0.7 * truth, (
        "uniform sampling got close to the corner value; the shortfall this test "
        "documents would not be reproducible and the contract repair is moot")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_near_corner_sampling_does_reach_it(seed):
    """The repair works: concentrate the draws toward the extreme rays and the
    sampled diameter recovers the algebraic one. This is the must-fire for the
    test above — without it, 'sampling underestimates' could be an artefact of a
    broken sampler rather than a fact about where the supremum lives."""
    a = _positive_matrix(6, seed)
    truth = column_diameter(a)
    g = torch.Generator().manual_seed(seed + 900)
    sampled = 0.0
    for _ in range(4000):
        x = torch.rand(6, generator=g, dtype=torch.float64) ** 40 + 1e-12
        y = torch.rand(6, generator=g, dtype=torch.float64) ** 40 + 1e-12
        sampled = max(sampled, d_H(a @ x, a @ y))
    assert sampled == pytest.approx(truth, rel=1e-6)


def test_column_diameter_is_infinite_for_a_matrix_with_a_zero():
    a = _positive_matrix(5, 3).clone()
    a[2, 4] = 0.0
    assert column_diameter(a) == math.inf


# ------------------------------------------------------ the convergence law

def test_observed_decay_respects_the_certified_ratio():
    """d_H(m_t, m*) <= kappa^t d_H(m_0, m*) — the law contract 1.1 asserts.

    The certified kappa is an upper bound, so the test is one-sided: the observed
    per-step ratio must not EXCEED it. A driver reporting faster-than-certified
    contraction is fine; one reporting slower has broken the theorem or the code.
    """
    a = _positive_matrix(8, 4)
    kappa = kappa_cert(column_diameter(a))
    assert kappa < 1.0
    m0 = _normalise(torch.rand(8, generator=torch.Generator().manual_seed(9),
                               dtype=torch.float64) + 0.01)
    res = settle(lambda m: _normalise(a @ m), m0, tol=1e-12, max_steps=400)
    assert res.converged
    assert len(res.residuals) >= 2
    ratios = [res.residuals[i + 1] / res.residuals[i]
              for i in range(len(res.residuals) - 1) if res.residuals[i] > 0]
    assert max(ratios) <= kappa + 1e-9, (
        f"observed ratio {max(ratios):.6f} exceeds certified {kappa:.6f}")


def test_predicted_step_count_bounds_the_observed_one():
    """t* = ceil(log(d_H(m_0,m*)/tol)/log(1/kappa)) must be an UPPER bound."""
    a = _positive_matrix(8, 5)
    kappa = kappa_cert(column_diameter(a))
    m0 = _normalise(torch.rand(8, generator=torch.Generator().manual_seed(11),
                               dtype=torch.float64) + 0.01)
    tol = 1e-10
    res = settle(lambda m: _normalise(a @ m), m0, tol=tol, max_steps=2000)
    assert res.converged
    predicted = math.ceil(math.log(res.residuals[0] / tol) / math.log(1.0 / kappa))
    assert res.steps <= predicted, f"took {res.steps}, theory caps at {predicted}"


def test_fixed_point_is_reached_from_different_starts():
    """Uniqueness is the whole point of Birkhoff: the limit must not depend on
    where the iteration began."""
    a = _positive_matrix(8, 6)
    T = lambda m: _normalise(a @ m)
    outs = []
    for s in (21, 22, 23):
        m0 = _normalise(torch.rand(8, generator=torch.Generator().manual_seed(s),
                                   dtype=torch.float64) + 0.01)
        outs.append(settle(T, m0, tol=1e-13, max_steps=2000).m)
    assert d_H(outs[0], outs[1]) < 1e-9
    assert d_H(outs[0], outs[2]) < 1e-9


# ---------------------------------------------------------------- MUST-FIRE

def test_must_fire_a_non_contracting_map_is_reported_unconverged():
    """A permutation is an isometry on the cone: it never contracts. The driver
    must say so rather than returning at the cap with a tidy trajectory."""
    perm = torch.tensor([1, 2, 3, 0])
    m0 = _normalise(torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64))
    res = settle(lambda m: m[perm], m0, tol=1e-12, max_steps=50)
    assert not res.converged
    assert res.steps == 50


def test_must_fire_a_map_leaving_the_cone_is_caught():
    """If T sends the iterate to the boundary the residual is +inf, and that has
    to surface as a failure rather than as a NaN nobody reads."""
    def bad(m):
        out = m.clone()
        out[0] = 0.0
        return _normalise(out)
    m0 = _normalise(torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float64))
    res = settle(bad, m0, tol=1e-12, max_steps=10)
    assert not res.converged
    assert res.left_cone


def test_must_fire_the_step_bound_can_actually_be_violated():
    """The predicted-steps test has to be capable of failing, or it proves
    nothing. A deliberately slack kappa produces a bound the run beats easily;
    a deliberately tight one produces a bound it cannot meet."""
    a = _positive_matrix(8, 7)
    d = column_diameter(a)
    tight = kappa_cert(d * 0.05)                 # far too optimistic a ratio
    m0 = _normalise(torch.rand(8, generator=torch.Generator().manual_seed(13),
                               dtype=torch.float64) + 0.01)
    res = settle(lambda m: _normalise(a @ m), m0, tol=1e-10, max_steps=2000)
    bogus = math.ceil(math.log(res.residuals[0] / 1e-10) / math.log(1.0 / tight))
    assert res.steps > bogus, "a wrong kappa must produce a violated bound"

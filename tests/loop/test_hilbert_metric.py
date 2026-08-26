"""The Hilbert projective metric, bound to its defining properties.

Round 6's entire certificate rests on this one function, and three fellows need
it simultaneously — Foreman for both kappa estimators, ARM S for settling, Chase
for the kappa back-fit from round 5's residual journal. Three independent
implementations of one formula give three independent bugs, so it is written
once here and every consumer imports it.

The property that a naive implementation gets wrong is the PROJECTIVE one:
d_H(p, q) = 0 whenever p and q are parallel, NOT only when they are equal. The
metric lives on rays through the positive cone, and the settling map T normalises
only at the end, so intermediate vectors are not on the simplex and a
scale-sensitive implementation reads a nonzero distance between two
representations of the same point.

The second trap is arithmetic. Evaluating max_j(p_j/q_j) / min_j(p_j/q_j) as a
literal ratio overflows for spreads that are ordinary in softmax rows; the
difference-of-logs form does not. This is the same class as round 5's arccos
collapse, and it is tested rather than assumed.
"""
from __future__ import annotations

import math

import pytest
import torch

from scale.hilbert import (d_H, delta_hat, kappa_cert, neumann_terms,
                          one_minus_kappa)

torch.set_num_threads(2)


def _pos(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, generator=g, dtype=torch.float64) + 0.1


# --------------------------------------------------------------- metric axioms

def test_identity_of_indiscernibles_on_rays():
    """d_H(p, p) = 0 exactly, not approximately."""
    p = _pos(64, 0)
    assert d_H(p, p) == 0.0


@pytest.mark.parametrize("alpha", [1e-8, 0.5, 1.0, 3.0, 1e8])
def test_projective_invariance(alpha):
    """THE defining property: d_H is blind to positive rescaling.

    A scale-sensitive implementation passes every other test in this file and
    fails only here, which is exactly why it is the first one written.
    """
    p = _pos(64, 1)
    assert d_H(alpha * p, p) == pytest.approx(0.0, abs=1e-12)


def test_symmetry():
    p, q = _pos(64, 2), _pos(64, 3)
    assert d_H(p, q) == pytest.approx(d_H(q, p), rel=1e-12)


def test_triangle_inequality():
    p, q, r = _pos(32, 4), _pos(32, 5), _pos(32, 6)
    assert d_H(p, r) <= d_H(p, q) + d_H(q, r) + 1e-12


def test_positivity_for_non_parallel_vectors():
    p, q = _pos(32, 7), _pos(32, 8)
    assert d_H(p, q) > 0.0


# ------------------------------------------------------- the boundary of the cone

def test_zero_entry_is_infinite_distance():
    """A zero entry leaves the open cone. This is K-A's ONLY trigger."""
    p = _pos(16, 9)
    q = _pos(16, 10).clone()
    q[3] = 0.0
    assert d_H(p, q) == math.inf


def test_negative_entry_is_infinite_distance():
    """A signed operator is not a positive map, and must read as such."""
    p = _pos(16, 11)
    q = _pos(16, 12).clone()
    q[5] = -0.3
    assert d_H(p, q) == math.inf


def test_both_zero_in_the_same_slot_is_still_infinite():
    """Neither vector is in the open cone; a 0/0 slot must not be silently
    dropped into a finite reading."""
    p, q = _pos(16, 13).clone(), _pos(16, 14).clone()
    p[2] = q[2] = 0.0
    assert d_H(p, q) == math.inf


# ------------------------------------------------- the arithmetic K-A depends on

def test_extreme_spread_does_not_overflow():
    """The literal ratio max(p/q)/min(p/q) overflows here; the log form must not.

    A spread of e^700 is not exotic for a softmax row whose smallest entry has
    underflowed relative to its largest.
    """
    p = torch.tensor([1.0, math.exp(350.0)], dtype=torch.float64)
    q = torch.tensor([math.exp(350.0), 1.0], dtype=torch.float64)
    got = d_H(p, q)
    assert math.isfinite(got)
    assert got == pytest.approx(700.0, rel=1e-12)


@pytest.mark.parametrize("delta", [10.0, 40.0, 75.0, 76.0, 100.0, 500.0, 1400.0])
def test_one_minus_kappa_survives_where_tanh_dies(delta):
    """1 - kappa = 2/(e^(Delta/2) + 1), the closed form K-A was repaired with.

    The float route 1 - tanh(Delta/4) reads exactly 0 from Delta = 100 and is
    already 7.2% high at Delta = 75. The closed form is checked here against
    50-digit Decimal, which is an independent arithmetic path rather than the
    same expression rewritten.
    """
    from decimal import Decimal, getcontext
    getcontext().prec = 60
    e = (Decimal(delta) / 2).exp()
    truth = 2 / (e + 1)
    got = one_minus_kappa(delta)
    assert got > 0.0, "the gap must stay representable, never collapse to 0"
    assert abs(Decimal(got) - truth) / truth < Decimal("1e-12")


def test_kappa_cert_saturates_and_the_helper_says_so():
    """kappa_cert is allowed to saturate; what is NOT allowed is treating that
    saturation as K-A firing. The gap function is the non-saturating witness."""
    assert kappa_cert(200.0) == 1.0            # float64 tanh saturates, expected
    assert one_minus_kappa(200.0) > 0.0        # and the gap still reads nonzero
    assert kappa_cert(math.inf) == 1.0
    assert one_minus_kappa(math.inf) == 0.0    # only a TRUE infinity gives zero


# ---------------------------------------------------------------- MUST-FIRE

def test_must_fire_a_scale_sensitive_metric_is_caught():
    """The projective test has to be able to fail. A metric that forgot the
    normalisation is constructed here and the property must reject it."""
    def scale_blind(p, q):                       # deliberately wrong: L-inf of diff
        return float((p - q).abs().max())
    p = _pos(64, 15)
    assert scale_blind(3.0 * p, p) > 1e-6, "the control cannot detect the defect"


def test_must_fire_delta_hat_sees_a_non_positive_map():
    """delta_hat over a batch containing one non-positive row must be infinite,
    or K-A can never fire for the reason it exists."""
    g = torch.Generator().manual_seed(16)
    rows = torch.rand(8, 12, generator=g, dtype=torch.float64) + 0.1
    assert math.isfinite(delta_hat(rows))
    rows[4, 7] = 0.0
    assert delta_hat(rows) == math.inf


# ------------------------------------------- the defect this module made itself

@pytest.mark.parametrize("delta", [60.0, 76.5, 100.0, 200.0, 700.0])
def test_neumann_terms_does_not_reconstruct_kappa_by_subtraction(delta):
    """Regression. `neumann_terms` originally computed math.log(1.0 - gap).

    For Delta = 76.5 the gap is ~4.9e-17, `1.0 - gap` rounds to exactly 1.0, and
    its log is 0.0 — a ZeroDivisionError one line later. The function had undone
    the K-A repair one call after it was made. It uses math.log1p(-gap) now, and
    the pairing of the closed form against a brute-force loop is what caught it.
    """
    n = neumann_terms(delta)
    assert n > 0 and math.isfinite(n)


def test_neumann_terms_reports_impossibility_rather_than_truncating():
    """At Delta = inf the conditioning does not exist; -1 says so out loud."""
    assert neumann_terms(math.inf) == -1


def test_neumann_cost_explodes_long_before_kappa_reaches_one():
    """A REAL CONSTRAINT on contract 1.2, not a defect.

    kappa < 1 is not the same as the implicit gradient being computable. At
    Delta = 20 the truncation needs 254,653 terms; by Delta = 60 it needs about
    2.3e14. Contract 1.2's Neumann route is implementable only while Delta stays
    small, and the certificate alone does not establish that.
    """
    assert neumann_terms(20.0) == 254653
    assert neumann_terms(60.0) > 10 ** 14

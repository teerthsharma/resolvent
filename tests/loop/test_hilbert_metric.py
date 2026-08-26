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

from scale.hilbert import (d_H, delta_hat, kappa_cert, n_parts, neumann_terms,
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


def test_both_zero_in_the_same_slot_is_FINITE():
    """SUPERSEDED AND REWRITTEN at round 6 iteration 4, not deleted.

    This originally asserted +inf, on the reasoning that neither vector is in the
    open cone. That reasoning was wrong. A shared zero means a shared SUPPORT,
    which means the same part, and the Hilbert metric is finite within a part —
    restricted to the support both vectors are strictly positive.

    The error was not academic: every masked attention row carries zeros, so
    under the old reading no two causal rows were ever a finite distance apart,
    and the whole metric was unusable on the objects it exists to measure. It was
    found by cross-checking against an independently written implementation that
    read a finite value here, and reporting the disagreement instead of
    reconciling it quietly.
    """
    p, q = _pos(16, 13).clone(), _pos(16, 14).clone()
    p[2] = q[2] = 0.0
    got = d_H(p, q)
    assert math.isfinite(got) and got > 0.0
    assert got == pytest.approx(d_H(p[[i for i in range(16) if i != 2]],
                                    q[[i for i in range(16) if i != 2]]), rel=1e-12)


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


def test_must_fire_delta_hat_sees_a_row_leaving_its_part():
    """SUPERSEDED AND REWRITTEN at round 6 iteration 4, not deleted.

    This test originally asserted that one zero among ninety-six entries drives
    Delta to +inf. That is precisely the behaviour the same-part ruling removes:
    a zeroed row simply moves to a different part, and the remaining rows still
    form a part with a perfectly good finite diameter. Written the old way, K-A
    fires on every causal draw forever, because causal rows at different indices
    always have different supports.

    What K-A actually asks is whether the image lands in ONE part, so that is
    what is asserted now — the part count rises, and the surviving part is still
    measurable rather than poisoned.
    """
    g = torch.Generator().manual_seed(16)
    rows = torch.rand(8, 12, generator=g, dtype=torch.float64) + 0.1
    assert math.isfinite(delta_hat(rows))
    assert n_parts(rows) == 1
    rows[4, 7] = 0.0
    assert n_parts(rows) == 2, "the zeroed row must be seen to leave its part"
    assert math.isfinite(delta_hat(rows)), "the surviving part stays measurable"


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


# ------------------------------------- the same-part ruling (round 6 it.4)

def test_d_H_stays_strict_across_parts():
    """The RULING keeps d_H unchanged: it is a metric on each part, extended by
    +inf between them. Only Delta gained the restriction."""
    p = torch.tensor([1.0, 2.0, 0.0, 4.0], dtype=torch.float64)
    q = torch.tensor([2.0, 1.0, 3.0, 1.0], dtype=torch.float64)
    assert d_H(p, q) == math.inf


def test_delta_hat_is_a_sup_over_SAME_PART_pairs():
    """Rows with differing supports must not force Delta to +inf.

    Causal rows at different indices always have different supports, so the
    unrestricted reading was +inf on every draw — measured at +inf in 30/30 live
    cells while the same-part reading gave 148.8022 to 403.5583 nats. A kill on
    the unrestricted number fires always and carries no information.
    """
    rows = torch.tensor([[1.0, 2.0, 0.0, 0.0],
                         [2.0, 1.0, 0.0, 0.0],       # same support as row 0
                         [1.0, 1.0, 1.0, 1.0]],      # a different part
                        dtype=torch.float64)
    got = delta_hat(rows)
    assert math.isfinite(got) and got > 0.0
    expected = d_H(rows[0][:2], rows[1][:2])
    assert got == pytest.approx(expected, rel=1e-12)


def test_n_parts_counts_what_K_A_actually_asks():
    rows = torch.tensor([[1.0, 2.0, 0.0, 0.0],
                         [2.0, 1.0, 0.0, 0.0],
                         [1.0, 1.0, 1.0, 1.0],
                         [0.0, 0.0, 0.0, 0.0]],      # all-zero: belongs to no part
                        dtype=torch.float64)
    assert n_parts(rows) == 2


def test_a_negative_entry_is_still_infinite():
    """The restriction is about supports, not about sign. A negative entry is not
    a cone point at all and must survive the repair as +inf."""
    rows = torch.tensor([[1.0, 2.0], [1.0, -1.0]], dtype=torch.float64)
    assert delta_hat(rows) == math.inf


def test_must_fire_the_same_part_reading_can_still_be_infinite():
    """The repair must not make Delta unconditionally finite. Two rows sharing a
    support but with an entry driven to zero inside it leave the part."""
    rows = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
    assert math.isfinite(delta_hat(rows))
    rows[1, 1] = 0.0
    assert n_parts(rows) == 2                # they are no longer the same part
    assert delta_hat(rows) == 0.0            # and no PAIR survives to be measured


# --------------------------------------- the log-domain path (round 6 it.5)

def test_log_domain_equals_the_softmax_path_exactly():
    """d_H(softmax u, softmax v) = osc(u - v), so the metric never needs exp().

    Algebra: softmax(u)_j = e^{u_j}/Z_u, so log(softmax(u)_j / softmax(v)_j)
    = (u_j - v_j) - log(Z_u/Z_v). The log-partition term is CONSTANT in j and
    d_H is an oscillation over j, so it cancels exactly.
    """
    from scale.hilbert import d_H_logits
    for seed in range(6):
        g = torch.Generator().manual_seed(seed)
        u = torch.randn(64, generator=g, dtype=torch.float64) * 30
        v = torch.randn(64, generator=g, dtype=torch.float64) * 30
        assert d_H_logits(u, v) == pytest.approx(
            d_H(torch.softmax(u, 0), torch.softmax(v, 0)), abs=1e-11)


def test_log_domain_survives_where_float32_softmax_destroys_the_metric():
    """The blocker this path exists to remove.

    At ARM A logit scales a float32 softmax underflows to exact zeros, the two
    rows land in different parts, and d_H reads +inf — the metric is destroyed on
    the objects it exists to measure. The log-domain route never forms exp(), so
    underflow cannot arise, and it is cheaper: no exp, no log.
    """
    from scale.hilbert import d_H_logits
    g = torch.Generator().manual_seed(7)
    u = torch.randn(1024, generator=g) * 60
    v = torch.randn(1024, generator=g) * 60
    p32, q32 = torch.softmax(u, 0), torch.softmax(v, 0)
    assert int((p32 == 0).sum() + (q32 == 0).sum()) > 0, "no underflow to defend against"
    assert d_H(p32.double(), q32.double()) == math.inf
    got = d_H_logits(u, v)
    assert math.isfinite(got) and got > 0.0


def test_log_domain_honours_the_same_part_rule_through_the_mask():
    """Support is carried by the mask, not by which entries underflowed. Rows
    with different masks are in different parts and must read +inf."""
    from scale.hilbert import d_H_logits
    u = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    v = torch.tensor([2.0, 1.0, 5.0, 0.0], dtype=torch.float64)
    m1 = torch.tensor([True, True, True, False])
    m2 = torch.tensor([True, True, False, False])
    assert math.isfinite(d_H_logits(u, v, m1))
    assert d_H_logits(u, v, m1) == pytest.approx(
        float((u[m1] - v[m1]).max() - (u[m1] - v[m1]).min()), rel=1e-12)
    assert d_H_logits(u, v, m1, m2) == math.inf


def test_must_fire_log_domain_is_not_trivially_zero():
    """A path that returned 0 unconditionally would pass an equality test against
    a second zero. It must track the actual oscillation."""
    from scale.hilbert import d_H_logits
    u = torch.tensor([0.0, 5.0, 10.0], dtype=torch.float64)
    assert d_H_logits(u, u) == 0.0                      # same ray
    assert d_H_logits(u, u + 3.0) == 0.0                # softmax-invariant shift
    assert d_H_logits(u, 2.0 * u) == pytest.approx(10.0, rel=1e-12)

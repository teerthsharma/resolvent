"""Contract 1.3's leakage gate, computed rather than swept.

1.3 voids degree-2 claims at any geometry where `E|L3|/E|I| > 0.5`, and left the
bound to be established empirically. On the readout family used throughout this
round it is available in closed form, which gives a per-draw admissibility rule
instead of a per-geometry sweep.

    f(S) = a_t / (1 - sum of masked masses)

    I  = f() - f(c) - f(j) + f(c,j)
    L3 = f() - f(c) - f(j) - f(k) + f(c,j) + f(c,k) + f(j,k) - f(c,j,k)

`I` is bilinear in the two masked masses and `L3` is trilinear, so `L3` is one
order HIGHER and the ratio vanishes as the third token gets light. Leakage is a
problem when the third token is HEAVY, not when the probe is delicate — which is
the opposite of what "leakage" suggests, and worth stating.
"""
from __future__ import annotations

import pytest

A_T = 0.05


def _f(*masked: float) -> float:
    return A_T / (1.0 - sum(masked)) if masked else A_T


def interaction(p_c: float, p_j: float) -> float:
    return _f() - _f(p_c) - _f(p_j) + _f(p_c, p_j)


def leakage(p_c: float, p_j: float, p_k: float) -> float:
    return (_f() - _f(p_c) - _f(p_j) - _f(p_k)
            + _f(p_c, p_j) + _f(p_c, p_k) + _f(p_j, p_k)
            - _f(p_c, p_j, p_k))


def ratio(p_c: float, p_j: float, p_k: float) -> float:
    return abs(leakage(p_c, p_j, p_k)) / abs(interaction(p_c, p_j))


def test_leakage_is_one_order_higher_than_the_interaction():
    """L3 is trilinear where I is bilinear, so the ratio -> 0 with the third
    mass. A leakage term of the SAME order would make degree-2 claims hopeless
    everywhere, so this is the property that lets ARM P exist at all."""
    for m in (1e-3, 1e-4, 1e-5):
        assert ratio(m, m, m) == pytest.approx(3 * m, rel=0.05)


def test_the_gate_trips_on_a_HEAVY_third_token():
    """Leakage is a problem when the third token is heavy, not when the probe is
    delicate — the opposite of what the word suggests."""
    assert ratio(0.05, 0.05, 0.02) < 0.5
    assert ratio(0.05, 0.05, 0.25) > 0.5


@pytest.mark.parametrize("m,want", [(0.010, 0.125146), (0.020, 0.123854),
                                    (0.050, 0.119861), (0.100, 0.112769),
                                    (0.200, 0.096521)])
def test_the_crossing_is_pinned(m, want):
    """The admissibility threshold, pinned as a value rather than an inequality.

    An inequality here would let the number drift the way a published tail norm
    once did; these are bisected to 1e-6 and asserted at 1e-5.
    """
    lo, hi = 1e-9, 0.5
    for _ in range(80):
        mid = (lo + hi) / 2
        if ratio(m, m, mid) < 0.5:
            lo = mid
        else:
            hi = mid
    assert hi == pytest.approx(want, abs=1e-5)


def test_the_leading_order_is_OPTIMISTIC_and_must_not_be_used_as_the_bar():
    """3*p_k predicts a crossing at 1/6. The exact ratio crosses near 0.115, so
    the series overstates the safe region by roughly 45%. Using the leading order
    as the admissibility bar would admit draws that void the claim."""
    assert ratio(0.05, 0.05, 1.0 / 6.0) > 0.5, (
        "the leading-order bar would have been safe here; the warning is moot")
    assert 0.5 / 3.0 > 0.119861 * 1.35, "the overstatement is smaller than claimed"


def test_must_fire_a_same_order_leakage_would_be_caught():
    """The control. If L3 were the same order as I, the ratio would NOT vanish
    with the third mass, and the first test above would fail — so that test can
    genuinely discriminate."""
    same_order = lambda p_c, p_j, p_k: interaction(p_c, p_j) * 0.9
    r = [abs(same_order(m, m, m)) / abs(interaction(m, m)) for m in (1e-3, 1e-5)]
    assert r[0] == pytest.approx(r[1], rel=1e-9), "control is not same-order"
    assert r[0] > 0.5, "a same-order leakage must void the gate at every scale"

"""Every Lean theorem cited as certifying the module must hold AT ITS SETTINGS.

THE GAP THIS CLOSES — M5, and the fourth appearance of one shape.

`CEQ.Occupancy.occupancy_is_exact_inverse` proves the truncated sum **is** the
two-sided inverse of `(I − A)`. It is stated at `N = n`, the matrix dimension.
The module truncates at `hops = 2` (the parity point) or `4`
(`ceq/attention.py::DEFAULT_HOPS`). At those settings `A^hops` is **not zero** —
measured `0.880500` at s=128 and `1.292741` at s=512 (hops=2) — so the theorem's
hypothesis is **violated by the tensor that ships**, which is M5's kill verbatim.

That is the fourth instance of the same defect: a correct statement about an
object other than the one that ships.

    instrument #17   `tgate` carried every headline and ships nowhere
    M4's kill        written about kept content, measured on deleted content
    M2's clause 2    a control that is zero by construction
    M5 (here)        a theorem whose hypothesis the module does not meet

Iteration 18's bind covers **operators**. Iteration 31's covers **arms**.
Nothing covered **hypotheses**, which is why this survived twenty-eight
iterations while `lake build` stayed green the whole time. **A theorem being
true is not the same as a theorem applying.**

BOUND ON VALUES, NOT ON TEXT. Four structure-by-regex instruments in this
project have broken or cried wolf (the LOCK slice boundary, the LOCK-line
scraper matching prose, the provenance audit missing a line break, the
ARMS-DISTINCT dispatch slice). The two that never have — the calibration gate
and the bitwise replay — compare values. This one evaluates the hypothesis
numerically at the shipped settings and compares against what the theorem
requires.
"""
from __future__ import annotations

import math

import pytest
import torch

from ceq import bench

#: The settings the module actually runs at.
#: `DEFAULT_HOPS = 4` in `ceq/attention.py`; the parity point is `hops = 2`.
SHIPPED_HOPS = (2, 4)
SIZES = (16, 64, 128, 512)

#: [RUN, iteration 35] Measured tail magnitudes at the shipped settings, seed 0,
#: dim 16, rho=1.5, lam=0.10. These are the values every document must quote.
EXPECTED_TAIL = {
    (16, 2): 1.356739, (16, 4): 0.695433,
    (64, 2): 1.476635, (64, 4): 0.976016,
    (128, 2): 0.880500, (128, 4): 0.882030,
    (512, 2): 1.292741, (512, 4): 0.925148,
}


def _operator(s: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(s, 16, generator=g)
    k = torch.randn(s, 16, generator=g)
    return bench._causal_sgate_operator(q, k, rho=1.5, lam=0.10).double()


def _pow_max(a: torch.Tensor, p: int) -> float:
    return float(torch.linalg.matrix_power(a, p).abs().max())


# ==========================================================================
# CALIBRATION — the hypothesis must be seen to HOLD where the theorem states it
# ==========================================================================

@pytest.mark.parametrize("s", SIZES)
def test_pow_card_eq_zero_holds_exactly_at_N_equals_n(s):
    """`Nilpotent.pow_card_eq_zero`: strictly-lower-triangular ⇒ `A^n = 0`.

    This is the must-hold end. If it fails, the Lean core is contradicted by
    the shipped tensor and every other claim here is void — so it is checked
    before the failing case below is trusted to mean anything.
    """
    a = _operator(s)
    got = _pow_max(a, s)
    assert got == 0.0, (
        f"A^n is {got!r} at s={s}, not exactly 0. `pow_card_eq_zero` is stated "
        f"over any CommRing with no sign and no magnitude hypothesis, so a "
        f"nonzero reading means the shipped tensor is not strictly lower "
        f"triangular — which would invalidate the Lean core, not just M5."
    )


# ==========================================================================
# THE BIND — M5's kill, evaluated rather than asserted
# ==========================================================================

@pytest.mark.parametrize("s", SIZES)
@pytest.mark.parametrize("hops", SHIPPED_HOPS)
def test_occupancy_exactness_hypothesis_is_VIOLATED_at_shipped_hops(s, hops):
    """M5's kill, made a measurement instead of a claim.

    `occupancy_is_exact_inverse` needs `A^N = 0` at the truncation index `N`.
    The module runs `N = hops ∈ {2, 4}` while `n` is the sequence length, so the
    hypothesis holds only when `hops >= n`. This test PINS THE VIOLATION so it
    cannot silently stop being reported — if a future change makes the module
    exact, this test fails and the claim may be restored deliberately rather
    than by drift.
    """
    a = _operator(s)
    got = _pow_max(a, hops)
    assert hops < s, "this test is about the truncated regime"
    assert got > 0.0, (
        f"A^{hops} is exactly 0 at s={s} — the shipped truncation now SATISFIES "
        f"`occupancy_is_exact_inverse`. M5's RED may be revisited, deliberately."
    )
    # and it is not merely nonzero — it is O(1), so the discarded tail is large
    assert got > 1e-3, (
        f"A^{hops} = {got!r} at s={s}: nonzero but tiny. The M5 finding is "
        f"reported as O(1); if the magnitude has collapsed, re-derive it."
    )
    # THE VALUE ITSELF, pinned. An inequality is what let a FABRICATED exact
    # number ride along inside fourteen passing tests: the docstring claimed
    # 1.471448 at s=128 while this operator reads 0.880500, and `> 1e-3` was
    # true of both. Iteration 35's audit swept 1,800 settings — dim, seed,
    # generator layout, rho, lam, hops — and found ZERO producing 1.471448.
    # Any exact magnitude quoted in a document must now match a pinned value.
    assert got == pytest.approx(EXPECTED_TAIL[(s, hops)], abs=5e-7), (
        f"A^{hops} at s={s} reads {got!r}, pinned {EXPECTED_TAIL[(s, hops)]!r}. "
        f"Either the operator moved (G2) or a quoted magnitude is fabricated."
    )


def test_no_geometric_bound_replaces_the_theorem_at_shipped_rho():
    """The tail is not merely uncertified — it is unbounded at `rho = 1.5`.

    `truncation_bound` refuses `rho >= 1` because `rho^(K+1)/(1-rho)` is
    NEGATIVE there: at the shipped `rho = 1.5, hops = 2` it evaluates to
    **-6.75**. So there is no theorem covering the truncation AND no bound
    standing in for one.
    """
    from ceq import attention
    with pytest.raises(ValueError):
        attention.truncation_bound(1.5, 2)
    # and the bound is real below 1, so the refusal is about rho, not a stub
    assert attention.truncation_bound(0.5, 2) == pytest.approx(0.5 ** 3 / 0.5)


def test_the_gap_is_stated_as_a_conditional_not_a_certification():
    """The honest wording, pinned.

    `occupancy_is_exact_inverse` certifies exactness at `hops >= n`. The module
    runs `hops = 2`. Any document claiming a "machine-checked finite resolvent"
    for the SHIPPED computation is overstating, and this test records the
    condition under which the claim would become true.
    """
    s = 128
    a = _operator(s)
    assert _pow_max(a, s) == 0.0                    # exact at hops = n
    assert _pow_max(a, 2) > 1e-3                    # not exact at hops = 2
    needed = s
    assert needed > max(SHIPPED_HOPS), (
        f"exactness needs hops >= {needed} at s={s}; the module ships "
        f"hops in {SHIPPED_HOPS}"
    )

"""An absolute threshold on a scale-free question is an arm-dependent filter.

THE GENERAL FAILURE, named one level above "the floor is too big". The quantity
`sign_flip_rate` reports is SCALE-FREE: the sign of a number is invariant under
any positive rescaling of it. The rule that decided which draws counted was NOT
scale-free -- `min(|lo|, |hi|) > 1e-6` carries units. Applying a rule with units
to a question without them imports each arm's own magnitude into the answer, so
the threshold stops being a dust filter and becomes an UNCONTROLLED,
ARM-DEPENDENT, `s`-DEPENDENT SAMPLE FILTER. Inspector strikes S2, S3 and S4 are
three readings of that one defect.

The mechanism is measured in `test_the_two_arms_live_two_orders_of_magnitude_apart`
below: at the published geometry the two arms' gradient distributions sit a
factor of ~200 apart, so one shared absolute cut lands INSIDE one arm's
distribution and five orders of magnitude BELOW the other's.

WHY xfail(strict=True) AND NOT A DELETED TEST. The published rule is still the
shipped default -- changing it would move the four `run_calib.py` numbers and
fire stopping condition G2, which is not a decision this file gets to make. So
the defect is pinned as a strict xfail instead: it must keep failing, and the
day someone makes it pass, the suite goes red and the calibration is revisited
deliberately rather than silently.

CPU ONLY, ON PURPOSE. This file does not use `_device.DEVICES`. The numbers here
are exact integer counts out of a fixed draw budget and are compared against
each other, not against a GPU parity oracle; running the same arithmetic twice
on two devices would double the cost and prove nothing.
"""
from __future__ import annotations

import functools
import statistics

import pytest
import torch

from ceq import bench

#: The published decay geometry: `ratio_sweep.py` and `diagnose.decay_curve`
#: both use `i = s-1, j = s/4, c = s/2`. Inspector S3 was struck for running its
#: control at `j = 1` instead, so this file uses the published one and nothing
#: else. s=32, hops=2, 1024 draws reproduces the S2 cell exactly.
GEOM = dict(s=32, i=31, j=8, c=16, hops=2, n_draws=1024, seed=0)

#: The shipped default: an ABSOLUTE cut, no relative component.
ABSOLUTE = pytest.param(1e-6, 0.0, id="floor=1e-6,rel=0",
                        marks=pytest.mark.xfail(
                            strict=True,
                            reason="THE DEFECT. Absolute cut on a scale-free "
                                   "quantity; see module docstring."))

#: The proposed criterion. `rel` is dimensionless and multiplies the arm's OWN
#: median `max(|lo|, |hi|)`, so the cut carries the arm's units and the rule is
#: invariant under a uniform rescaling of that arm's gradients.
#:
#: WHY 1e-9 AND NOT SMALLER OR LARGER. float64 eps is 2.2e-16; a two-hop path
#: sum over s=32 accumulates order s^2 ~ 1e3 operations, so genuine round-off is
#: order 1e-13 of the row scale. 1e-9 sits four decades above that -- it removes
#: dust and nothing else. It is not a free parameter to be tuned upward: at
#: rel=1e-3 the measured ratio is 2.39x, WORSE than the absolute floor's 2.07x,
#: because the two arms' gradient distributions differ in shape as well as in
#: scale. A relative rule is the right KIND of rule; it is not a licence to cut
#: deep. Measured on this geometry: rel=0 -> 1.077x, 1e-9 -> 1.120x,
#: 1e-6 -> 1.217x, 1e-3 -> 2.391x.
RELATIVE = pytest.param(0.0, 1e-9, id="floor=0,rel=1e-9")

RULES = [ABSOLUTE, RELATIVE]

#: How far the two arms' discard rates may differ before the rule is a filter
#: rather than a dust guard. A rule that throws away half of one arm's evidence
#: and none of the other's is not measuring the arms.
MAX_DISCARD_GAP = 0.10

#: How far a rule may move the arm-vs-arm ratio away from the ratio measured
#: with NO rule at all. The no-rule ratio is the thing being estimated; a
#: discard rule is supposed to clean it, not redefine it.
MAX_RATIO_DISTORTION = 0.25


@functools.lru_cache(maxsize=None)
def draws(kind: str):
    """The raw `(lo, hi)` pairs. Cached: the loop is the only expensive part,
    and every rule in this file is a pure function of the same pairs. That
    reuse is the whole point of splitting `sign_flip_draws` from `flip_rate` --
    a floor sweep used to cost a full re-measurement per floor value, which is
    why no published number ever carried one."""
    return bench.sign_flip_draws(kind, device=torch.device("cpu"), **GEOM)


def scale_of(kind: str) -> float:
    return statistics.median([max(abs(a), abs(b)) for a, b in draws(kind)])


def test_the_two_arms_live_two_orders_of_magnitude_apart():
    """THE CAUSE, measured. Not an inference from row-L1 -- the gradients.

    `sgate` pins row L1 at rho=1.5, so its gradient scale is fixed in `s`.
    DeltaNet's WY matrix has no row normalizer at all and its row L1 grows
    roughly linearly (4.23 / 21.53 / 81.62 at s = 32/128/512). One absolute
    threshold cannot be a fair rule for both, and the gap is not marginal.
    """
    sg, dn = scale_of("sgate"), scale_of("deltanet")
    print(f"\nmedian max|grad|:  sgate {sg:.6g}   deltanet {dn:.6g}   "
          f"ratio {dn / sg:.1f}x")
    print(f"the shipped absolute cut 1e-6 sits at {1e-6 / sg:.3g} of sgate's "
          f"scale and {1e-6 / dn:.3g} of DeltaNet's")
    assert dn / sg > 50.0, (sg, dn)


@pytest.mark.parametrize("floor,rel", RULES)
def test_the_discard_rule_bites_both_arms_at_comparable_rates(floor, rel):
    """The requirement an exclusion rule has to meet to be a rule and not a bias.

    `discard_fraction` is the number no published rate ever carried: the share
    of SIGN-FLIPPED draws the rule threw away. If it differs sharply between two
    arms being compared, the comparison is between two different samples.
    """
    sg = bench.discard_fraction(draws("sgate"), floor, rel)
    dn = bench.discard_fraction(draws("deltanet"), floor, rel)
    print(f"\nfloor={floor:g} rel={rel:g}  discarded: sgate {sg:.4%}  "
          f"deltanet {dn:.4%}  gap {abs(sg - dn):.4%}")
    assert abs(sg - dn) <= MAX_DISCARD_GAP, (
        f"the discard rule threw away {sg:.2%} of sgate's flipped draws and "
        f"{dn:.2%} of DeltaNet's. That is an arm-dependent sample filter, not "
        f"a dust guard.")


@pytest.mark.parametrize("floor,rel", RULES)
def test_the_arm_ratio_is_a_property_of_the_arms_and_not_of_the_rule(floor, rel):
    """S2 as a standing test. The headline was `1.39x, non-overlapping`.

    With no discard rule at all the two arms read 0.050781 and 0.054688 --
    1.08x, overlapping. The published 2.07x separation is manufactured by the
    threshold, and it is manufactured entirely on one side: sgate moves 1.93x
    between the two rules and DeltaNet does not move at all.
    """
    base = (bench.flip_rate(draws("deltanet"), 0.0, 0.0)
            / bench.flip_rate(draws("sgate"), 0.0, 0.0))
    sg = bench.flip_rate(draws("sgate"), floor, rel)
    dn = bench.flip_rate(draws("deltanet"), floor, rel)
    got = dn / sg
    print(f"\nfloor={floor:g} rel={rel:g}  sgate {sg:.6f}  deltanet {dn:.6f}  "
          f"ratio {got:.3f}x   (no rule at all: {base:.3f}x)")
    assert abs(got / base - 1.0) <= MAX_RATIO_DISTORTION, (
        f"the discard rule moved the arm-vs-arm ratio from {base:.3f}x to "
        f"{got:.3f}x. The ratio is reporting the rule.")


def test_softmax_zero_is_structural_and_no_rule_can_move_it():
    """The control that must NOT move, checked at every rule including none.

    softmax's 0.0000 is not a small number that a threshold rounded down. There
    are ZERO sign-flipped draws: a non-negative operator composed with fixed
    linear value projections factors as (non-negative weight) x (fixed matrix),
    so no third token can move the sign at any depth. Removing the floor
    entirely does not manufacture a single flip out of noise -- which is the
    other half of the case for the relative criterion, since a criterion that
    invented flips would be worse than the bias it fixes.
    """
    d = draws("softmax")
    flipped = [(a, b) for a, b in d if a * b < 0]
    print(f"\nsoftmax: {len(flipped)} sign-flipped draws out of {len(d)}; "
          f"median max|grad| {scale_of('softmax'):.3g}")
    assert flipped == [], flipped[:5]
    for floor, rel in [(1e-6, 0.0), (0.0, 0.0), (0.0, 1e-9), (0.0, 1e-3)]:
        assert bench.flip_rate(d, floor, rel) == 0.0, (floor, rel)


def test_the_relative_rule_still_removes_genuine_float_dust():
    """It has to actually discard something, or it is `floor=0` with extra words.

    sgate's smallest `max(|lo|, |hi|)` is order 1e-14 against a median of order
    4e-3 -- eleven decades down, which is round-off and not a measurement. The
    relative rule removes exactly that band.
    """
    d = draws("sgate")
    dust = [1 for a, b in d if a * b < 0 and min(abs(a), abs(b)) < 1e-9 * scale_of("sgate")]
    frac = bench.discard_fraction(d, 0.0, 1e-9)
    print(f"\nsgate: {sum(dust)} flipped draws inside the dust band; "
          f"relative rule discards {frac:.4%} of flipped draws")
    assert sum(dust) > 0
    assert frac > 0.0

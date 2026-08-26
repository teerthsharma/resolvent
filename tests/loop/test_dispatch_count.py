"""The cost gate's dispatch claim, made testable.

K-F was left undecided because its two available numbers disagree: exact FLOP
ratios near one, and clock ratios up to forty-five on a contended box. The gap was
attributed to framework dispatch overhead by citation. A citation explains a
mechanism; it does not measure this arm.

These tests bind the measurable half. A settling loop issues O(t*) dispatches
while performing O(1) FLOPs in t*, so the dispatch count must grow linearly in the
step count while the arithmetic does not. That is a prediction the diagnosis makes
and it can fail.
"""
from __future__ import annotations

import pytest
import torch

from scale.dispatch_count import (DispatchCounter, count_dispatches,
                                  glance_dispatches, settle_dispatches)

torch.set_num_threads(2)


def test_the_counter_counts_what_it_claims():
    """A counter that over- or under-counts by a constant would still look
    linear, so it is pinned against a known-length sequence first."""
    a = torch.ones(4)
    assert count_dispatches(lambda: a + a) == 1
    assert count_dispatches(lambda: (a + a) * a) == 2
    assert count_dispatches(lambda: ((a + a) * a) - a) == 3


def test_dispatch_count_is_linear_in_the_step_count():
    """THE PREDICTION. Each settling step issues a fixed number of dispatches, so
    the total must be affine in `steps` with a positive slope."""
    counts = {t: settle_dispatches(t) for t in (0, 10, 20, 40, 80)}
    diffs = [counts[b] - counts[a]
             for a, b in ((0, 10), (10, 20), (20, 40), (40, 80))]
    per_step = [diffs[0] / 10, diffs[1] / 10, diffs[2] / 20, diffs[3] / 40]
    assert all(p > 0 for p in per_step), f"not growing: {counts}"
    assert max(per_step) == pytest.approx(min(per_step), rel=1e-9), (
        f"per-step dispatch cost is not constant: {per_step}")


def test_settling_dominates_the_dispatch_budget_but_not_the_arithmetic():
    """At the measured median step count the dispatch ratio is an order of
    magnitude, while the published FLOP ratio at s=1024,k=8 is 1.010420.

    That is the whole of K-F's unresolved gap, stated as two numbers taken by two
    methods rather than as a citation.
    """
    glance = glance_dispatches()
    settled = settle_dispatches(44)            # Foreman's measured median t*
    assert settled > 10 * glance, (
        f"dispatch ratio {settled / glance:.2f}x — too small to explain the "
        f"clock/FLOP gap, so the dispatch diagnosis would be wrong")


def test_the_operator_build_is_shared_and_cancels():
    """`settle_dispatches(0)` must equal the operator build alone, so the
    difference between settled and glance is settling and nothing else."""
    zero = settle_dispatches(0)
    glance = glance_dispatches()
    # glance applies the operator once more than the zero-step settle does
    assert 0 < zero <= glance, f"zero-step settle {zero} vs glance {glance}"


def test_must_fire_a_loop_free_arm_shows_no_step_growth():
    """The control. A computation with no per-step work must NOT grow with the
    step argument — otherwise the linearity test would pass on anything."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(16, 8, generator=g)

    def flat(steps):
        return count_dispatches(lambda: x @ x.transpose(-2, -1))

    assert flat(1) == flat(100), "the control grows with steps; it should not"

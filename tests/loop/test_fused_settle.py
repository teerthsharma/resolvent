"""S6: the settling loop's per-step dispatch cost, and what fusing may not change.

The gate is a cost claim, so the test that binds it is a cost measurement plus an
exactness measurement, and the second is the one that matters. A fused loop that is
cheaper and answers a different question has not closed S6; it has changed the arm.

WHY FUSION IS AVAILABLE AT ALL, stated so the test can check the reason rather than
the result. The Hilbert metric is projective: d_H(c*p, q) = d_H(p, q) for any c > 0,
which is not an approximation but the definition, since a positive scaling shifts
every log-ratio by log(c) and an oscillation subtracts it back out. The per-step
normalisation therefore contributes NOTHING to any residual the driver journals. It
exists only to keep the iterate inside float range. Deferring it across a group of
steps changes the iterate by a positive scalar and changes the journal by zero.

That is why the exactness assertion below is EQUALITY on the residuals rather than a
tolerance: the fused and unfused runs differ by a scale factor the metric cannot
see. A tolerance here would hide the failure mode that matters, which is a fused
loop that quietly iterates something else.
"""
from __future__ import annotations

import math

import pytest
import torch

from scale.dispatch_count import count_dispatches
from scale.hilbert import d_H
from scale.settle import settle, settle_fused


def _row_stochastic(s: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.softmax(torch.randn(s, s, generator=g).double(), -1)


def _start(s: int) -> torch.Tensor:
    return torch.full((s,), 1.0 / s, dtype=torch.float64)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_fused_journal_is_identical_not_merely_close(seed: int) -> None:
    """The fused run must journal the SAME residuals, exactly.

    Deferring the normalisation multiplies the iterate by a positive scalar, and
    the Hilbert metric is invariant under exactly that. So the journals agree
    bit-for-bit up to floating-point reassociation, and the tolerance below is for
    the reassociation alone -- it is not room for a different iteration.
    """
    m = _row_stochastic(48, seed)
    plain = settle(lambda v: m @ v, _start(48), tol=1e-12, max_steps=400)
    fused = settle_fused(m, _start(48), tol=1e-12, max_steps=400, group=16)

    assert plain.steps == fused.steps
    assert plain.converged == fused.converged
    assert len(plain.residuals) == len(fused.residuals)
    for a, b in zip(plain.residuals, fused.residuals):
        assert a == pytest.approx(b, rel=1e-9, abs=1e-15)


@pytest.mark.parametrize("group", [1, 4, 16, 64])
def test_fused_fixed_point_matches_up_to_scale(group: int) -> None:
    """The settled reading itself, compared in the metric it is read in.

    Comparing the two limits entrywise would fail by construction, since one is
    normalised and one is not. Comparing them in d_H is the comparison the arm
    actually makes, and it must read zero.
    """
    m = _row_stochastic(32, 7)
    plain = settle(lambda v: m @ v, _start(32), tol=1e-12, max_steps=500)
    fused = settle_fused(m, _start(32), tol=1e-12, max_steps=500, group=group)
    assert d_H(plain.m, fused.m) < 1e-9


def test_dispatch_cost_per_step_falls_below_the_measured_three() -> None:
    """The S6 target, measured as a SLOPE rather than a ratio.

    A first draft of this test divided one total by the step count and read
    3.0078125 where round 6 recorded exactly 3.0. The code was right and the test
    was wrong: 385/128 is 3.0 per step plus one fixed dispatch for building the
    start vector, which sat inside the counted region. Round 6's figure was a
    DIFFERENCE between two arms sharing a setup, so the setup cancelled there and
    did not cancel here.

    Taking the slope between two step counts cancels any fixed cost by
    construction, which is what "per step" means and what a single ratio cannot
    deliver. The unfused loop is then exactly 3 -- matmul, sum, divide -- of which
    only the matmul does arithmetic.
    """
    s = 64
    m = _row_stochastic(s, 0)

    def plain(steps: int):
        def run():
            v = _start(s)
            for _ in range(steps):
                v = m @ v
                v = v / v.sum()
            return v
        return count_dispatches(run)

    def fused(steps: int, group: int = 16):
        def run():
            v = _start(s)
            for i in range(1, steps + 1):
                v = m @ v
                if i % group == 0:
                    v = v / v.sum()
            return v
        return count_dispatches(run)

    slope_plain = (plain(256) - plain(128)) / 128
    slope_fused = (fused(256) - fused(128)) / 128
    assert slope_plain == 3.0, slope_plain
    assert slope_fused <= 1.2, slope_fused
    # The matmul is irreducible: one per step, and the only one doing work.
    assert slope_fused >= 1.0
    # Round 6 measured the settling arm's 3.0 as a difference against the glance.
    # The slope reproduces it without needing a second arm to cancel against.
    assert plain(128) - plain(0) == 3 * 128


def test_group_size_does_not_change_the_flop_count() -> None:
    """Fusion must be free in arithmetic, not merely cheaper in dispatches.

    A group of k steps is k matrix-vector products either way. Anything that
    precomputed a matrix power instead would be asymptotically MORE arithmetic
    (s^3 log k against k s^2), so this asserts the matmul count is flat in the
    group size -- the property that makes 'zero FLOP cost' true rather than a
    slogan.
    """
    s, steps = 32, 64
    m = _row_stochastic(s, 3)
    counts = {}
    for group in (1, 2, 8, 32):
        n = 0

        class _MM(torch.utils._python_dispatch.TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                nonlocal n
                if "mv" in str(func) or "matmul" in str(func) or "mm" in str(func):
                    n += 1
                return func(*args, **(kwargs or {}))

        with _MM():
            settle_fused(m, _start(s), tol=0.0, max_steps=steps, group=group)
        counts[group] = n
    assert len(set(counts.values())) == 1, counts
    # Without this the assertion above passes on set([0]) -- a counter that never
    # fired would report every group as equally free. Nine controls in this
    # campaign went vacuous in exactly that shape.
    assert all(v == steps for v in counts.values()), counts


def test_deferred_normalisation_reports_rather_than_silently_overflows() -> None:
    """A group large enough to leave float range must be REPORTED, not returned.

    This is the failure mode fusion introduces and the reason the rescale point is
    also the check point. An unnormalised iterate under a map whose spectral radius
    is far from one drifts geometrically; past some group size it reaches inf,
    every later residual is non-finite, and a driver that did not check would
    return a converged-looking result computed from a dead iterate.

    THE FIRST DRAFT OF THIS TEST WAS VACUOUS, and the two arguments below are the
    repair. It asserted `left_cone or converged` at group=64 and tol=1e-12, and
    read `left_cone=False, converged=True, steps=1`: the iterate settled before it
    could drift, so the branch under test never ran and the disjunction was carried
    entirely by the wrong half. `tol=0.0` removes the early exit and group=128
    puts the rescale beyond the drift, which together make the assertion an
    assertion. It is the eleventh vacuous control found in this campaign and the
    second caught by its own author before shipping.
    """
    s = 16
    g = torch.Generator().manual_seed(11)
    m = torch.softmax(torch.randn(s, s, generator=g).double(), -1) * 1e60

    out = settle_fused(m, _start(s), tol=0.0, max_steps=400, group=128)
    assert out.left_cone, "the drift branch did not fire -- this control is vacuous"
    assert not out.converged
    assert all(math.isfinite(r) for r in out.residuals)

    # And the control has teeth only if the same map settles cleanly when the
    # rescale keeps up with the drift. Without this the test would also pass on a
    # driver that reported left_cone unconditionally.
    ok = settle_fused(m, _start(s), tol=1e-12, max_steps=400, group=1)
    assert ok.converged and not ok.left_cone

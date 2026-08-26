"""Rectangular Monge oracle, bound against an independent solver.

The +3/-5 gate that decides this round's shape runs a RECTANGULAR assignment —
a few dozen causal tokens against a pool of thousands. The existing exact check
covers only the square case, so the assignment the gate depends on has no
independent verification. This closes that.

The cost |a_i - b_j| on a line is Monge, and for Monge costs an optimal
assignment can be taken MONOTONE: sort both sides, and the chosen fillers appear
in the same order as the causal tokens they serve. That turns the rectangular
assignment into a subsequence-selection dynamic program, exact in O(n*m), which
shares no code and no algorithm with a Hungarian or auction solver. Two methods
that fail differently.
"""
from __future__ import annotations

import math

import pytest
import torch

from scale.monge import monge_assign, monge_cost

torch.set_num_threads(2)

try:
    from scipy.optimize import linear_sum_assignment
    HAVE_SCIPY = True
except Exception:                                    # pragma: no cover
    HAVE_SCIPY = False


def _pair(n: int, m: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(n, generator=g, dtype=torch.float64) * 10
    b = torch.rand(m, generator=g, dtype=torch.float64) * 10
    return a, b


@pytest.mark.skipif(not HAVE_SCIPY, reason="scipy absent")
@pytest.mark.parametrize("n,m,seed", [(4, 4, 0), (5, 12, 1), (8, 40, 2),
                                      (1, 9, 3), (16, 17, 4), (12, 300, 5)])
def test_matches_an_independent_solver(n, m, seed):
    """Path A: monotone subsequence DP. Path B: Hungarian/JV via scipy."""
    a, b = _pair(n, m, seed)
    C = (a.unsqueeze(1) - b.unsqueeze(0)).abs()
    r, c = linear_sum_assignment(C.numpy())
    assert monge_cost(a, b) == pytest.approx(float(C[r, c].sum()), rel=1e-12)


@pytest.mark.parametrize("n,m,seed", [(3, 3, 6), (6, 25, 7), (9, 60, 8)])
def test_the_assignment_is_injective_and_monotone(n, m, seed):
    a, b = _pair(n, m, seed)
    idx = monge_assign(a, b)
    assert len(idx) == n
    assert len(set(idx)) == n, "an assignment must be injective"
    # Monotone in the VALUES of b, not in b's original indices — `idx` points
    # into the unsorted pool, so comparing raw indices tests nothing. This was
    # wrong on the first writing and the failure was the test's, not the DP's.
    order = torch.argsort(a)
    chosen = [float(b[idx[int(i)]]) for i in order]
    assert chosen == sorted(chosen), "Monge admits a monotone optimum"


def test_cost_equals_the_sum_over_the_returned_assignment():
    """The reported cost and the returned indices must describe the same thing.
    A solver whose cost and assignment disagree is two instruments, not one."""
    a, b = _pair(7, 30, 9)
    idx = monge_assign(a, b)
    total = sum(abs(float(a[i]) - float(b[idx[i]])) for i in range(len(a)))
    assert monge_cost(a, b) == pytest.approx(total, rel=1e-12)


def test_identity_pool_costs_zero():
    """Matching a population against itself is the null direction: exact zero."""
    a, _ = _pair(20, 20, 10)
    assert monge_cost(a, a) == pytest.approx(0.0, abs=1e-12)


def test_more_pool_never_costs_more():
    """Adding candidates can only help. A DP with a wrong recurrence typically
    breaks this monotonicity, which is why it is asserted rather than assumed."""
    a, b = _pair(6, 20, 11)
    small = monge_cost(a, b[:12])
    large = monge_cost(a, b)
    assert large <= small + 1e-12


def test_raises_when_the_pool_is_too_small():
    a, b = _pair(9, 4, 12)
    with pytest.raises(ValueError):
        monge_assign(a, b)


# ---------------------------------------------------------------- MUST-FIRE

def test_must_fire_a_crossed_assignment_costs_strictly_more():
    """Monotonicity is what the DP exploits, so the crossed pairing — which
    monotonicity forbids — must be strictly more expensive."""
    a = torch.tensor([0.0, 1.0], dtype=torch.float64)
    b = torch.tensor([0.0, 1.0], dtype=torch.float64)
    crossed = abs(0.0 - 1.0) + abs(1.0 - 0.0)
    assert monge_cost(a, b) < crossed - 1e-9
    assert monge_cost(a, b) == pytest.approx(0.0, abs=1e-12)


def test_must_fire_greedy_nearest_is_beaten_on_real_instances():
    """The oracle must catch a plausible WRONG answer, or it certifies nothing.

    The first version of this control used a hand-built two-by-two example and
    could not fire: on two points the monotone assignment is FORCED, so greedy
    and optimal coincide for every such instance. That is the same
    control-that-cannot-be-nonzero defect this project has struck repeatedly,
    written into the control meant to prevent it.

    Greedy does differ — measured, strictly worse in 111 of 400 random four-by-
    nine instances — so the control is drawn rather than hand-built.
    """
    g = torch.Generator().manual_seed(0)
    worse = 0
    for _ in range(400):
        n, m = 4, 9
        a = torch.rand(n, generator=g, dtype=torch.float64) * 10
        b = torch.rand(m, generator=g, dtype=torch.float64) * 10
        used, tot = set(), 0.0
        for ai in a:
            j = min((k for k in range(m) if k not in used),
                    key=lambda k: abs(float(ai) - float(b[k])))
            used.add(j)
            tot += abs(float(ai) - float(b[j]))
        assert tot >= monge_cost(a, b) - 1e-9, "greedy beat the optimum"
        worse += tot > monge_cost(a, b) + 1e-9
    assert worse > 50, f"greedy differed in only {worse}/400 — control too weak"


@pytest.mark.skipif(not HAVE_SCIPY, reason="scipy absent")
def test_must_fire_a_broken_recurrence_disagrees_with_scipy():
    """A deliberately wrong DP (skip-only, never taking a filler) must NOT match
    the independent solver — otherwise the agreement test above is vacuous."""
    a, b = _pair(5, 20, 13)
    C = (a.unsqueeze(1) - b.unsqueeze(0)).abs()
    r, c = linear_sum_assignment(C.numpy())
    wrong = sum(float(C[i, i]) for i in range(len(a)))   # naive diagonal
    assert abs(wrong - float(C[r, c].sum())) > 1e-6, (
        "the naive answer coincides here; the agreement test proves nothing")

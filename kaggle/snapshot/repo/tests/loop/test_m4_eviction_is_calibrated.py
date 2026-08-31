"""M4's kill, calibrated at both ends — the must-fire arm it never had.

THE DEFECT. M4 claims *"removing a token renormalizes survivors exactly; settled
state of unrelated content moves by 0.0"*, and its headline is
`0.000000e+00` over 24/24 draws. But the protocol perturbs a token chosen by
`lowest_salience_token(x, rho, exclude=keep)` — which **excludes `keep` by
construction** — and then measures `settle_evicted`, which reads `x[keep]` and
nothing else (`ceq/eviction.py:118-121`).

So `y[keep] == x[keep]` bitwise and the difference is **identically zero by
arithmetic**. The headline reduces to *"you deleted it, so it stopped
mattering."* That is the M2 `not_in_P` defect on a tensor: a control that cannot
be nonzero is not a control, and a kill that cannot fire is a defect.

WHAT THIS FILE ADDS. Not a different claim — the **missing half of the
calibration**. Per the standing doctrine, *calibrate every instrument against a
case where it must fire and one where it must not, before believing it*:

    MUST FIRE     perturb a token INSIDE `keep` -> settle_evicted MUST move.
                  If it does not, the instrument is dead and the 0.0 on crushed
                  tokens is worth nothing.
    MUST NOT FIRE perturb a CRUSHED token -> settle_evicted must not move.
                  This is M4's claim, and it is only evidence once the arm above
                  has been shown capable of moving.

The distinction the original protocol lost is that **0.0 from an instrument that
can move is a result; 0.0 from an instrument that cannot is a tautology.**

The gating comparison is kept and sharpened. Gating leaves the crushed token in
the softmax denominator, so it moves under BOTH perturbations — which is exactly
why gating was never the like-for-like control it was reported as.
"""
from __future__ import annotations

import pytest
import torch

from ceq import eviction as ev

KEEP, RHO, S, D = 6, 0.9, 16, 8


def _context(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(1000 + seed)
    return torch.randn(S, D, generator=g)


def _edit(x: torch.Tensor, idx: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(7000 + seed)
    y = x.clone()
    y[idx] = torch.randn(D, generator=g)
    return y


def _moved(x, y, keep, fn) -> float:
    return float((fn(y, keep, RHO) - fn(x, keep, RHO)).abs().max())


# ==========================================================================
# MUST FIRE — without this the 0.0 below is a tautology
# ==========================================================================

def test_evicted_settle_moves_when_a_KEPT_token_is_perturbed():
    """The calibration M4 never had.

    A token inside `keep` IS read by `settle_evicted`. Perturbing it must move
    the settled state, or the instrument cannot detect change at all and its
    zeros carry no information.
    """
    moved, worst = 0, 0.0
    for s in range(8):
        x = _context(s)
        keep = ev.keep_indices(x, KEEP, RHO)
        target = int(keep[len(keep) // 2])          # deliberately INSIDE keep
        y = _edit(x, target, s)
        d = _moved(x, y, keep, ev.settle_evicted)
        worst = max(worst, d)
        moved += int(d > 1e-12)
    assert moved == 8, (
        f"settle_evicted moved in only {moved}/8 draws when a KEPT token was "
        f"perturbed (worst {worst!r}). The instrument cannot register change, "
        f"so M4's 0.000000e+00 on crushed tokens is a tautology, not a result."
    )


# ==========================================================================
# MUST NOT FIRE — M4's actual claim, now interpretable
# ==========================================================================

def test_evicted_settle_does_not_move_when_a_CRUSHED_token_is_perturbed():
    """M4's claim. Only evidence because the arm above is shown able to move.

    Recorded honestly: this zero is STRUCTURAL — `settle_evicted` never reads
    outside `keep`. It is worth asserting as a regression guard, and it is NOT
    worth reporting as a measurement of exactness.
    """
    worst = 0.0
    for s in range(8):
        x = _context(s)
        keep = ev.keep_indices(x, KEEP, RHO)
        crushed = ev.lowest_salience_token(x, RHO, exclude=keep)
        assert not bool((keep == crushed).any()), (
            "lowest_salience_token returned a KEPT token; the structural "
            "argument below no longer holds and this test must be re-derived"
        )
        worst = max(worst, _moved(x, _edit(x, crushed, s), keep, ev.settle_evicted))
    assert worst == 0.0, f"expected a structural zero, got {worst!r}"


# ==========================================================================
# THE COMPARISON, SHARPENED — why gating was never like-for-like
# ==========================================================================

@pytest.mark.parametrize("where", ["kept", "crushed"])
def test_gating_moves_under_both_perturbations_and_eviction_does_not(where):
    """Gating keeps the token in the softmax denominator, so it moves under BOTH.

    That is the real asymmetry, and it is a fact about the DENOMINATOR, not
    evidence that eviction is exact. Reporting `0.000000e+00` against
    `2.154868e-05` as a like-for-like comparison overstated it: on the crushed
    perturbation the two arms are not measuring the same thing at all.
    """
    gated_moves = 0
    for s in range(8):
        x = _context(s)
        keep = ev.keep_indices(x, KEEP, RHO)
        idx = (int(keep[len(keep) // 2]) if where == "kept"
               else ev.lowest_salience_token(x, RHO, exclude=keep))
        y = _edit(x, idx, s)
        gated_moves += int(_moved(x, y, keep, ev.settle_gated) > 1e-12)
    assert gated_moves == 8, (
        f"gated settle moved in only {gated_moves}/8 draws under a {where} "
        f"perturbation; the denominator asymmetry is the whole point of the "
        f"comparison and it must be visible in both placements"
    )

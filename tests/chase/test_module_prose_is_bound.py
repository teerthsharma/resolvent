"""Two numbers a module docstring states and no test has ever checked.

A number in a docstring is a claim with no instrument behind it. Both of the
claims bound here were written in an earlier round, read by every later reader as
established, and had no failing case anywhere in the repository.

WHAT IS BOUND, AND THE RED FOR EACH.

1. `ceq/attention.py`'s three-tier header. `grep -rn 0.376137914` over the whole
   working tree returns exactly two hits: the docstring itself, and one
   `"t": "finding"` line in `house-events-round1.jsonl`. That JSONL is a
   transcript of what an agent SAID, not a run log -- no seed, no shape, no
   script, nothing rerunnable. So the numbers licensing the entire architecture
   ("both baselines sit in tier 2; the bar is tier 3") rested on a quoted
   sentence. RED: no test in `tests/` mentioned any of the three figures.
   They are replaced below by figures this file produces, and the docstring now
   names this test.

2. `ceq/lm.py`'s committed globals are NOT the parity point. `RHO, HOPS = 0.9, 3`
   and `SGATE_LAM = 1.0` sit at module scope; the 1.0334 median was measured at
   `rho=1.5, lam=0.10, hops=2`. The docstring inside `Attention.operator` says so
   in prose. RED: nothing failed if that prose were deleted, and nothing measured
   how far apart the two operators actually are -- a reader who imports the
   defaults gets a different operator than the number describes and no signal
   that anything is wrong.

WHY THE GLOBALS ARE NOT SIMPLY CORRECTED HERE. Every measurement already on
record in `ceq/lm.py` was taken at those values, and `ceq/bench.py` -- which the
live calibration probe reads through `run_calib.py` -- is another agent's file
this round. Moving a module global to make a docstring true would invalidate the
recorded numbers to fix the prose. The divergence is BOUND instead: the test
fails if either side moves without the other.
"""
from __future__ import annotations

import math
import pathlib

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Shape of every draw below. Small on purpose: the tier claims are structural,
#: not asymptotic, and a claim that needs a big tensor to show up is a different
#: claim. float64 throughout so "invariant" means invariant rather than "below
#: fp32 resolution".
S, D, HOPS = 8, 16, 2
DTYPE = torch.float64


def _mask(device):
    return torch.ones(S, S, dtype=torch.bool, device=device).tril(-1)


def _softmax_row(w, m):
    neg = torch.finfo(w.dtype).min
    return torch.softmax(w.masked_fill(~m, neg), -1).masked_fill(~m, 0.0)


def _operators(q, k, m, *, rho=1.5, lam=0.10):
    """(non-negative, signed) from ONE set of logits.

    The two arms differ in exactly one term -- the subtracted `lam * softmax(-w)`
    -- so any difference downstream is a statement about signedness and not about
    a draw.
    """
    w = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    pp = _softmax_row(w, m)
    pm = _softmax_row(-w, m)
    return pp, rho * (pp - lam * pm) / (1.0 + lam)


def _influence(a, hops=HOPS):
    """`d(out)/d(v)` for `out = v + Av + ... + A^K v`, which is `I + A + ... + A^K`.

    Written out rather than autograd'd because it is exactly the matrix the tier
    claim is about, and because `torch.eye` makes the strict upper triangle's
    exact zero visible instead of an epsilon.
    """
    out = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
    ah = out.clone()
    for _ in range(hops):
        ah = a @ ah
        out = out + ah
    return out


# ------------------------------------------------ tier 1: pairwise similarity

def test_a_third_token_cannot_move_the_ratio_of_two_softmax_weights(device):
    """TIER 1, and the decidable criterion the whole taxonomy rests on: a
    similarity weighting is pairwise, so `p_ij / p_il = exp(w_ij - w_il)` and no
    perturbation of a third key `c` can appear in it.

    Measured, seed 0, S=8 D=16 float64, row i=7, pair (j,l)=(1,3), perturbed
    key c=5 shifted by +3.0:

        pair ratio  0.200257204381 -> 0.200257204381    |delta| 1.110e-16
        same row    max |dp| 5.349e-01

    The control is the second line. Without it this test would pass on a
    perturbation that did nothing at all.
    """
    g = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(S, D, dtype=DTYPE, generator=g).to(device)
    k = torch.randn(S, D, dtype=DTYPE, generator=g).to(device)
    m = _mask(device)
    i, j, l, c = 7, 1, 3, 5

    p0, _ = _operators(q, k, m)
    k2 = k.clone()
    k2[c] += 3.0
    p1, _ = _operators(q, k2, m)

    moved = (p1[i] - p0[i]).abs().max().item()
    assert moved > 1e-2, ("the perturbation did not move the row at all, so the "
                          "invariance below is vacuous: {}".format(moved))

    r0 = (p0[i, j] / p0[i, l]).item()
    r1 = (p1[i, j] / p1[i, l]).item()
    assert abs(r1 - r0) <= 8 * torch.finfo(DTYPE).eps * abs(r0), (
        "a third token moved a pairwise similarity ratio: {} -> {}".format(r0, r1))


# --------------------------------- tier 2 vs tier 3: the ratio moves either way

def test_multi_hop_lets_a_third_token_move_the_ratio_in_both_the_signed_and_the_non_negative_arm(device):
    """TIER 2's first half. Multi-hop is what BUYS the ratio move, and it buys it
    for the non-negative operator too -- which is precisely why the ratio test
    does not separate the two arms and a second criterion is needed.

    Measured, seed 0, same draw as above, influence `I + A + A^2`:
    non-negative ratio 0.846220 -> 0.425642, a 49.70% move; the signed arm on
    the same draw moves 0.797063 -> 0.077354, 90.30%. Both move, so the ratio
    is not the discriminator -- the sign of the minimum entry is.
    """
    g = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(S, D, dtype=DTYPE, generator=g).to(device)
    k = torch.randn(S, D, dtype=DTYPE, generator=g).to(device)
    m = _mask(device)
    i, j, l, c = 7, 1, 3, 5

    p0, _ = _operators(q, k, m)
    k2 = k.clone()
    k2[c] += 3.0
    p1, _ = _operators(q, k2, m)

    j0 = _influence(p0)
    j1 = _influence(p1)
    r0 = (j0[i, j] / j0[i, l]).item()
    r1 = (j1[i, j] / j1[i, l]).item()
    assert abs(r1 - r0) / abs(r0) > 1e-3, (
        "the non-negative multi-hop influence ratio did not move; tier 2 is not "
        "distinguished from tier 1 by this draw: {} -> {}".format(r0, r1))


def test_only_the_signed_arm_reaches_a_negative_influence_entry(device):
    """TIER 2's second half and TIER 3, as one measurement with its own control.

    The entry that matters is the MINIMUM of the influence Jacobian. For a
    non-negative operator the Kleene star of a non-negative matrix is
    non-negative in any ordered semiring, so the minimum is the strict upper
    triangle's structural zero and never anything less. For the signed operator
    it is reachable.

    Measured over 200 draws, S=8 D=16 hops=2 float64, `sgate` at rho=1.5
    lam=0.10, the SAME logits feeding both arms:

        non-negative   0/200 draws with a negative entry, global min  0.000000e+00
        sgate        107/200 draws with a negative entry, global min -1.377561e-01

    A 0/200 control is what makes the 107/200 mean anything. If the control ever
    goes non-zero, the tier taxonomy is wrong, not the operator.
    """
    m = _mask(device)
    neg_nonneg = neg_signed = 0
    min_nonneg = min_signed = 0.0
    for seed in range(200):
        g = torch.Generator(device="cpu").manual_seed(seed)
        q = torch.randn(S, D, dtype=DTYPE, generator=g).to(device)
        k = torch.randn(S, D, dtype=DTYPE, generator=g).to(device)
        p, a = _operators(q, k, m)
        n_min = _influence(p).min().item()
        s_min = _influence(a).min().item()
        min_nonneg = min(min_nonneg, n_min)
        min_signed = min(min_signed, s_min)
        neg_nonneg += n_min < 0
        neg_signed += s_min < 0

    assert (neg_nonneg, min_nonneg) == (0, 0.0), (
        "a NON-NEGATIVE operator produced a negative influence entry in {} of 200 "
        "draws (min {:.6e}); the tier-2 claim is that this is impossible"
        .format(neg_nonneg, min_nonneg))
    assert neg_signed >= 80, (
        "the signed operator reached a negative influence entry in only {} of 200 "
        "draws; tier 3 is the only thing that distinguishes this operator"
        .format(neg_signed))
    assert min_signed < -0.05, min_signed


def test_the_attention_docstring_cites_this_test_rather_than_a_bare_number(device):
    """The bind. `ceq/attention.py`'s tier table now names this file, so deleting
    the tests silently un-evidences the header, and changing the header without
    the tests is a diff a reviewer can see."""
    src = (ROOT / "ceq" / "attention.py").read_text(encoding="utf-8")
    assert "test_module_prose_is_bound" in src, (
        "ceq/attention.py's tier claims name no test")
    for orphan in ("0.376137914", "0.787332", "0.808386"):
        assert orphan not in src, (
            "{} is back in ceq/attention.py; it appears in no run log, only in "
            "house-events-round1.jsonl as a quoted agent sentence".format(orphan))


# ------------------------------------------ ceq/lm.py's globals vs the parity point

def test_the_research_modules_committed_globals_are_not_the_parity_point(device):
    """RECORDED, NOT FIXED. `ceq/lm.py` is where the 1.0334 run happened, and its
    module globals are the PRE-campaign values. A reader who imports them gets
    `rho=0.9, lam=1.0, hops=3`, which is an operating point with no published
    number attached to it.

    This test fails in BOTH directions on purpose: if the globals are moved to
    the parity point, every number already recorded in that module was taken at
    the old ones and the move has to be a deliberate, documented change rather
    than a silent one.
    """
    from ceq import lm
    from ceq.hf.configuration_ceq import PARITY_POINT

    assert (lm.RHO, lm.SGATE_LAM, lm.HOPS) == (0.9, 1.0, 3), (
        "ceq/lm.py's globals moved to {}; every measurement recorded in that "
        "module was taken at (0.9, 1.0, 3), so this is a change that has to be "
        "made deliberately".format((lm.RHO, lm.SGATE_LAM, lm.HOPS)))
    assert (lm.RHO, lm.SGATE_LAM, lm.HOPS) != PARITY_POINT
    assert PARITY_POINT == (1.5, 0.10, 2), PARITY_POINT


def test_lm_py_says_out_loud_that_its_globals_are_not_the_parity_point(device):
    """The prose that stops a reader trusting the defaults has to be present.
    Grep-bound for the same reason `.tril(-1)` is."""
    src = (ROOT / "ceq" / "lm.py").read_text(encoding="utf-8")
    assert "THE MODULE GLOBALS BELOW ARE NOT THE PARITY POINT" in src, (
        "the warning that ceq/lm.py's defaults are not the measured operating "
        "point has been removed from the module")


def test_the_two_operating_points_are_measurably_different_operators(device):
    """How far apart, as a number, so "different point" is not a shrug.

    Same logits, `sgate` at the committed globals against `sgate` at the parity
    point. Measured seed 0, S=8 D=16 float64: max |A_parity - A_committed| =
    1.2273, and 2.9701 in relative Frobenius norm. Row sums separate too: the
    committed `lam = 1.0` makes every row sum EXACTLY zero (max |row sum|
    8.327e-17) while the parity `lam = 0.10` leaves rho*(1-lam)/(1+lam) =
    1.227273 of positive mass per row.
    """
    from ceq import lm
    from ceq.hf.configuration_ceq import PARITY_POINT

    g = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(S, D, dtype=DTYPE, generator=g).to(device)
    k = torch.randn(S, D, dtype=DTYPE, generator=g).to(device)
    m = _mask(device)

    _, committed = _operators(q, k, m, rho=lm.RHO, lam=lm.SGATE_LAM)
    _, parity = _operators(q, k, m, rho=PARITY_POINT[0], lam=PARITY_POINT[1])

    gap = (parity - committed).abs().max().item()
    assert gap > 0.1, ("the two operating points produce the same operator to "
                       "{:.3e}; then the divergence is cosmetic".format(gap))

    #: lam = 1.0 annihilates the constant vector; lam = 0.10 does not. This is
    #: the structural difference, not just a magnitude one.
    assert committed[1:].sum(-1).abs().max().item() < 1e-15, committed.sum(-1)
    per_row = PARITY_POINT[0] * (1 - PARITY_POINT[1]) / (1 + PARITY_POINT[1])
    assert parity[1:].sum(-1).min().item() == pytest.approx(per_row, rel=1e-9)

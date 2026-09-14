"""The clean null L-NULL requires, with the refused coordinates PINNED.

The null in tests/curvature/test_pi_jepa_assignment_work.py shuffles the whole
nine-vector, and refusals travel with the shuffle. A refused coordinate is
dropped from the read, so every row of that null also re-selects which seven of
the nine encoder coordinates the read sees at all. Replaying its generator at
PERM_SEED=20260914 gives twelve of twelve rows refusing a pair other than
{6, 7}. That null varies two things and names one, so under L-NULL it is void
and the claim it licensed -- that the corner assignment carries no information
-- is unbound.

This file builds the null that isolates the variable.

  VARIED, and it is the only thing varied: which of the seven assigned
  positions carry the softmax corner.
  PINNED, by name: the refused coordinates stay at {6, 7} and stay refused; the
  corner multiset stays at four softmax and three linear; the encoder weights,
  the predictor weights, the seed, the bed and the held-out draw are the shipped
  ones.

The space is C(7, 4) = 35 arrangements and is ENUMERATED WHOLE, so there is no
sampling fraction to report and no seed behind the null itself. The assigned
mask is one of the thirty-five, which makes its rank the statistic.

RUN: python -m pytest tests/curvature/test_clean_permutation_null.py -v
"""

import itertools

import numpy as np
import pytest

import ceqjepa.pi_jepa as pj

#: The assigned mask must land in this fraction of the enumerated null to count
#: as carrying information about which coordinate deserves which corner.
RANK_BAR = 0.25


def _assigned():
    """The shipped mask, and the index sets the null must hold fixed."""
    beta = pj.beta_vector()
    refused = tuple(i for i, b in enumerate(beta) if pj.is_refusal(b))
    assigned = tuple(i for i, b in enumerate(beta) if not pj.is_refusal(b))
    ones = tuple(i for i in assigned if float(beta[i]) == 1.0)
    return beta, refused, assigned, ones


def _mask_with_ones_at(beta, assigned, ones_at):
    """A mask differing from `beta` ONLY in which assigned positions carry 1.0.

    Refusals are copied through at their own indices, so the read sees the same
    seven columns in every row of the null.
    """
    out = list(beta)
    for i in assigned:
        out[i] = 1.0 if i in ones_at else 0.0
    return out


def _enumerate(beta, refused, assigned, ones):
    """All C(7, 4) arrangements, refusals pinned. The assigned mask is one."""
    k = len(ones)
    return [tuple(c) for c in itertools.combinations(assigned, k)]


def test_the_null_holds_the_refused_coordinates_fixed():
    """MUST-FIRE. The defect that voided the previous null, asserted absent."""
    beta, refused, assigned, ones = _assigned()
    assert refused == (6, 7), refused
    rows = _enumerate(beta, refused, assigned, ones)
    assert len(rows) == 35, "expected C(7,4) = 35 arrangements, got %d" % len(rows)
    for ones_at in rows:
        m = _mask_with_ones_at(beta, assigned, ones_at)
        got = tuple(i for i, b in enumerate(m) if pj.is_refusal(b))
        assert got == refused, (
            "an arrangement moved the refused coordinates from %s to %s: this "
            "null would confound corner assignment with column selection, which "
            "is the defect it exists to remove" % (refused, got))
        n_ones = sum(1 for b in m if not pj.is_refusal(b) and float(b) == 1.0)
        assert n_ones == len(ones), (
            "an arrangement changed the corner multiset: %d softmax columns "
            "against the assigned mask's %d" % (n_ones, len(ones)))


def test_the_enumeration_contains_the_assigned_mask_exactly_once():
    """Without this the rank statistic is comparing against a null it is not in."""
    beta, refused, assigned, ones = _assigned()
    rows = _enumerate(beta, refused, assigned, ones)
    assert rows.count(ones) == 1, (
        "the assigned arrangement %s appears %d times in the enumeration"
        % (ones, rows.count(ones)))


@pytest.mark.slow
def test_the_assigned_arrangement_beats_the_clean_null():
    """The binding this round owes: the claim, against a null that isolates it."""
    beta, refused, assigned, ones = _assigned()
    rows = _enumerate(beta, refused, assigned, ones)
    scored, refusals = {}, {}
    for ones_at in rows:
        m = _mask_with_ones_at(beta, assigned, ones_at)
        model = pj.build(pj.SEED, beta=m)
        for p in model.online.parameters():
            p.requires_grad_(False)
        try:
            fit = pj.fit(model, steps=pj.N_STEPS, seed=pj.SEED)
        except AssertionError as exc:
            refusals[ones_at] = str(exc)
            continue
        held = pj._held_out_scores(fit["model"], pj.SEED + 1)
        scored[ones_at] = float(held["nrmse"])

    assert ones in scored, (
        "the assigned arrangement itself refused: %s" % refusals.get(ones))
    mine = scored[ones]
    order = sorted(scored.values())
    rank = order.index(mine) + 1
    frac = rank / len(order)
    print("\n  clean null, refusals pinned at {6, 7}, C(7,4) = %d enumerated whole"
          % len(rows))
    print("  scored %d, refused %d" % (len(scored), len(refusals)))
    print("  assigned %.4f  best %.4f  median %.4f  worst %.4f"
          % (mine, order[0], float(np.median(order)), order[-1]))
    print("  rank %d of %d (%.3f)" % (rank, len(order), frac))
    assert frac <= RANK_BAR, (
        "the corner assignment does not beat a null that isolates it: assigned "
        "%.4f ranks %d of %d (%.3f) against a %.2f bar, over the whole C(7,4) = "
        "%d space with the refused coordinates pinned at %s. best %.4f, median "
        "%.4f, worst %.4f, %d arrangements refused. This null varies only which "
        "assigned position carries the softmax corner, so the claim it tests is "
        "the claim the rule makes"
        % (mine, rank, len(order), frac, RANK_BAR, len(rows), refused,
           order[0], float(np.median(order)), order[-1], len(refusals)))

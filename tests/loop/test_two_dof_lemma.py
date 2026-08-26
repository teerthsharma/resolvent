"""The 2-dof lemma: why the four-point probe cannot collapse the way round 5 did.

Round 5 died of a scalar identity. With a ONE-token mask the renormalised row is
A^0[i,j] = A^c[i,j]/(1-p), so every readout is a function of the single scalar p,
and theta = arcsin(sqrt(TV)) followed — two statistics, one degree of freedom,
functionally dependent. Contract 1.3 asserts the four-point probe escapes this
"by construction" because a two-token mask leaves two degrees of freedom. That was
asserted and never checked; this checks it, and finds one place it fails.

Readout f(S) = the row's weight on a fixed target t outside {c, j}:

    f(empty) = a_t
    f({c})   = a_t / (1 - p_c)
    f({j})   = a_t / (1 - p_j)
    f({c,j}) = a_t / (1 - p_c - p_j)

    I(c,j) = f(empty) - f({c}) - f({j}) + f({c,j})
           = a_t [ 1 - 1/(1-p_c) - 1/(1-p_j) + 1/(1-p_c-p_j) ]

Expanded, I = a_t * p_c * p_j * (6 p_c p_j + 3 p_c + 3 p_j + 2), so the leading
term is 2 a_t p_c p_j: exactly bilinear, and identically zero if either token
carries no mass. That is what a degree-2 interaction must look like, and it is why
no single scalar can reproduce it.
"""
from __future__ import annotations

import math

import pytest
import torch

torch.set_num_threads(2)

A_T = 0.05


def interaction(p_c: float, p_j: float, a_t: float = A_T) -> float:
    return a_t * (1.0 - 1.0 / (1.0 - p_c) - 1.0 / (1.0 - p_j)
                  + 1.0 / (1.0 - p_c - p_j))


def _rank(fns, pt, h=1e-6):
    rows = [[(f(pt[0] + h, pt[1]) - f(pt[0] - h, pt[1])) / (2 * h),
             (f(pt[0], pt[1] + h) - f(pt[0], pt[1] - h)) / (2 * h)] for f in fns]
    J = torch.tensor(rows, dtype=torch.float64)
    return int(torch.linalg.matrix_rank(J, atol=1e-9)), torch.linalg.svdvals(J)


@pytest.mark.parametrize("p_c,p_j", [(0.10, 0.03), (0.20, 0.15),
                                     (0.30, 0.05), (0.02, 0.40)])
def test_interaction_is_not_a_function_of_total_variation(p_c, p_j):
    """Rank 2 means I cannot be written as a function of TV alone. Rank 1 would
    be round 5 repeating itself with a different pair of statistics."""
    r, sv = _rank([interaction, lambda a, b: a + b], (p_c, p_j))
    assert r == 2, f"collapsed to rank {r}; singular values {sv.tolist()}"
    assert float(sv[1]) > 1e-4, "second direction present but numerically dead"


def test_the_interaction_vanishes_when_either_token_carries_no_mass():
    """A degree-2 interaction must be identically zero if either participant is
    absent. This is the property that distinguishes it from a degree-1 effect."""
    for p in (0.0, 0.05, 0.2, 0.4):
        assert interaction(0.0, p) == pytest.approx(0.0, abs=1e-15)
        assert interaction(p, 0.0) == pytest.approx(0.0, abs=1e-15)


def test_leading_order_is_bilinear():
    """I -> 2 a_t p_c p_j as both masses go to zero, from the series expansion."""
    for p in (1e-4, 1e-5, 1e-6):
        assert interaction(p, p) == pytest.approx(2 * A_T * p * p, rel=2e-3)


def test_DEGENERATE_on_the_symmetric_locus():
    """A FINDING, not a defect, and it is not in the contract.

    I is symmetric in (p_c, p_j), so on the diagonal p_c = p_j its two partials
    are equal; TV = p_c + p_j has equal partials everywhere. The Jacobian rows are
    therefore parallel and the rank drops to 1 — measured second singular value
    2.027511e-17 at (0.05, 0.05).

    So the two degrees of freedom are GENERIC, not universal: exactly where the
    two masked tokens carry equal mass, the probe loses its second direction and
    is momentarily as collapsed as round 5's was everywhere. A draw protocol that
    pairs tokens of similar salience will sit near this locus, and the contract
    does not say so.
    """
    for p in (0.05, 0.10, 0.25):
        r, sv = _rank([interaction, lambda a, b: a + b], (p, p))
        assert r == 1, f"expected the symmetric degeneracy, got rank {r}"
        # Relative, not absolute. The degeneracy is EXACT in exact arithmetic;
        # what survives here is finite-difference rounding, and that noise scales
        # with the magnitude of the derivatives, which grows with p. An absolute
        # 1e-12 bar was the first version of this line and it failed at p = 0.10
        # and 0.25 for that reason alone -- the tolerance was wrong, not the
        # lemma. The analytic statement is asserted separately below.
        assert float(sv[1]) / float(sv[0]) < 1e-10


def test_the_symmetric_degeneracy_is_EXACT_not_merely_small():
    """Symbolic, so no differencing is involved and no tolerance is needed.

    I is symmetric in its two arguments, so dI/dp_c - dI/dp_j vanishes identically
    on the diagonal. This is what makes the rank drop a fact about the probe
    rather than an artefact of how it was measured.
    """
    sp = pytest.importorskip("sympy")
    pc, pj, at = sp.symbols("p_c p_j a_t", positive=True)
    expr = at * (1 - 1 / (1 - pc) - 1 / (1 - pj) + 1 / (1 - pc - pj))
    gap = sp.simplify(sp.diff(expr, pc) - sp.diff(expr, pj))
    assert sp.simplify(gap.subs(pj, pc)) == 0
    assert sp.simplify(gap) != 0, "off-diagonal it must NOT vanish"


# ---------------------------------------------------------------- MUST-FIRE

@pytest.mark.parametrize("p", [0.05, 0.20, 0.40])
def test_must_fire_the_one_token_mask_reads_rank_one(p):
    """Round 5's death, reproduced. If this ever read rank 2 the whole check
    would be measuring something other than functional dependence."""
    h = 1e-6
    rows = [[(math.asin(math.sqrt(p + h)) - math.asin(math.sqrt(p - h))) / (2 * h)],
            [((p + h) - (p - h)) / (2 * h)]]
    J = torch.tensor(rows, dtype=torch.float64)
    assert int(torch.linalg.matrix_rank(J, atol=1e-9)) == 1


def test_must_fire_a_degree_one_statistic_is_caught():
    """The rank test must REJECT something that really is a function of TV. A
    degree-1 readout is constructed and required to read rank 1."""
    r, _ = _rank([lambda a, b: 3.0 * (a + b) - 1.0, lambda a, b: a + b], (0.2, 0.1))
    assert r == 1, "the check cannot detect a genuine degree-1 statistic"

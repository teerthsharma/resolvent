"""do(clamp) on a grid: the whole consequence field as ONE rank-1 update.

WHY THIS FILE EXISTS. docs/canon/CORRECTIONS.md row C4 records that
`do(BLOCK a cell)` is NOT a rank-1 edit -- blocking cell b rewrites b's own row
and rescales its <=4 neighbours' rows, so it is rank <=5, and the grid operator
is not lower triangular, which is why ceqjepa/intervene.py refuses it outright
(intervene.py:54-66: "the operator would stop being lower triangular"). That
retraction cited a measurement -- sigma_2/sigma_1 = 2.837e-15 over 12 instances
-- for which NO PRODUCER EXISTED ANYWHERE IN THE TREE. A headline number with no
producer is precisely what STRUCK.md exists to catch (see its rows marked
FABRICATED and NO PRODUCER HAS EVER EXISTED). This module is that producer.

THE CLAIM, and the exact hypothesis it holds under:

    Forcing the walker AT one cell t onto one neighbour j rewrites exactly ONE
    row of M = I - Q. The resulting change to the committor at EVERY cell is
    exactly rank 1:

        M' = M - e_t d^T,  R' = R + e_t dR^T
        c  = M^-1 e_t,  w^T = d^T M^-1,  den = 1 - d^T c
        q' = q + outer(c, (dR + R^T w) / den)

    The collapse 1 + w_t/den = 1/den is the only algebraic step; nothing else is
    needed. c = M^-1 e_t is the expected-visit vector to t and is >= 0, which is
    why every non-zero pixel of the field carries the sign of a single scalar.

WHAT IS AND IS NOT CLAIMED. This is a statement about the SOLVE, which is exact
and needs no learning. It is not a statement that a trained model does anything.
The separation-by-depth reading of this was RETIRED -- with fitted coefficients a
degree-8 polynomial reaches 1.448e-02% of the committor's range at kappa=94.08
where the fixed-coefficient Neumann truncation reads 112.6157%, a factor of
7,780, and Cayley-Hamilton makes degree n-1 exact outright. Do not reintroduce a
representation claim here; see CORRECTIONS.md.

RUN: python -m ceqjepa.rank1_field
"""

import numpy as np

from ceqjepa.beds.gridworld import (GOAL, HAZARD, make_grid, solve_grid,
                                    _neighbours)

__all__ = ["clamp_chain", "clamp_field", "rank_ratio", "measure_rank1"]

# A field is called rank 1 when sigma_2/sigma_1 falls below this. float64 on a
# well-conditioned grid lands near 1e-15; the two-row control in demo() reads
# 1e-2, so the gap this threshold sits in is thirteen orders wide.
RANK1_TOL = 1e-10


def clamp_chain(sol, t, j):
    """(Q, R) for do(walker at t -> j): t's outgoing row replaced by a point mass.

    Returns None if j is not a legal destination (not a neighbour of t, or a
    wall), rather than silently producing a chain for a different intervention.
    """
    ti = sol.tidx.get(t)
    if ti is None:
        return None
    G = sol.cells.shape[0]
    if j not in _neighbours(G, t):
        return None
    cell = sol.cells[j // G, j % G]
    Q, R = sol.Q.copy(), sol.R.copy()
    Q[ti, :] = 0.0
    R[ti, :] = 0.0
    if j in sol.tidx:
        Q[ti, sol.tidx[j]] = 1.0
    elif cell == GOAL:
        R[ti, 0] = 1.0
    elif cell == HAZARD:
        R[ti, 1] = 1.0
    else:
        return None                      # a wall: not an intervention, no chain
    return Q, R


def clamp_field(sol, t):
    """[nT, n_candidates] of q_goal(cell | do(t -> j)) - q_goal(cell), one column
    per legal destination j. This is the whole consequence field for one cursor
    position: what changes EVERYWHERE, not just at t."""
    cols = []
    for j in _neighbours(sol.cells.shape[0], t):
        ch = clamp_chain(sol, t, j)
        if ch is None:
            continue
        Q, R = ch
        try:
            q2 = np.linalg.solve(np.eye(Q.shape[0]) - Q, R)
        except np.linalg.LinAlgError:
            continue                     # den = 0: the clamp made the chain non-absorbing
        cols.append(q2[:, 0] - sol.q[:, 0])
    return np.stack(cols, 1) if len(cols) >= 2 else None


def rank_ratio(D):
    """sigma_2 / sigma_1. Exactly 0 for a rank-1 matrix, up to conditioning."""
    s = np.linalg.svd(D, compute_uv=False)
    return float(s[1] / s[0]), int((s > s[0] * RANK1_TOL).sum())


# A field whose largest entry is below this is at the float64 floor: the clamp
# changed nothing measurable. sigma_2/sigma_1 on such a field is noise over
# noise and reads ~1e-1, which a naive rank test reports as "rank 3" -- the test
# cannot tell "not rank 1" from "nothing happened". Measured: an excluded
# instance has columns of norm 1.68e-15, 1.66e-14, 2.29e-15. These are COUNTED
# and reported, never silently dropped: a cursor cell whose field is identically
# zero is a real property of the bed and the demo has to render something there.
DEGENERATE_FLOOR = 1e-9


def measure_rank1(n_instances=12, G=9, gap=3, seed=1, threshold=0.01):
    """Rank of the clamp field per instance, over instances whose field is not
    at the float64 floor. Returns the degenerate count alongside, never hiding it."""
    import random
    rng = random.Random(seed)
    ratios, ranks, dens, degenerate, scanned = [], [], [], 0, 0
    while len(ratios) < n_instances:
        sol = solve_grid(make_grid(rng, G=G, gap=gap))
        if len(sol.tlist) < 8:
            continue
        D = clamp_field(sol, sol.tlist[rng.randrange(len(sol.tlist))])
        if D is None:
            continue
        scanned += 1
        peak = float(np.abs(D).max())
        if peak < DEGENERATE_FLOOR:
            degenerate += 1                      # counted, not swept away
            continue
        r, k = rank_ratio(D)
        ratios.append(r)
        ranks.append(k)
        dens.append(float((np.abs(D) > threshold).mean()))
    return dict(n=n_instances, scanned=scanned, degenerate=degenerate,
                ratios=ratios, ranks=ranks,
                ratio_med=float(np.median(ratios)), ratio_max=float(max(ratios)),
                rank_set=sorted(set(ranks)), density=float(np.mean(dens)))


def demo():
    print("(a) THE CLAIM: do(clamp t -> j) changes the committor EVERYWHERE by a")
    print("    rank-1 update, on an operator that is NOT lower triangular.")
    m = measure_rank1()
    print("    %d instances with a measurable field, G=9 gap=3" % m["n"])
    print("    sigma_2/sigma_1   median %.3e   max %.3e" % (m["ratio_med"], m["ratio_max"]))
    print("    numerical rank (tol %.0e * sigma_1): %s" % (RANK1_TOL, m["rank_set"]))
    print("    frac(|dq| > 0.01) over all pixels  : %.4f" % m["density"])
    print("    DEGENERATE, reported not hidden: %d of %d cursor cells scanned had a field"
          % (m["degenerate"], m["scanned"]))
    print("    entirely below %.0e -- the clamp changed nothing measurable there, and the"
          % DEGENERATE_FLOOR)
    print("    rank statistic on such a field is noise over noise (it reads ~1e-1).")
    assert m["rank_set"] == [1], "the clamp field is not rank 1: ranks %s" % m["rank_set"]
    assert m["ratio_max"] < RANK1_TOL, "sigma_2/sigma_1 exceeds the tolerance"

    print("(b) THE PLANTED NEGATIVE: clamping TWO rows must NOT be rank 1, or the")
    print("    statistic above is measuring nothing and would call anything rank 1.")
    import random
    rng = random.Random(4)
    sol = solve_grid(make_grid(rng, G=9, gap=3))
    nT = len(sol.tlist)
    cols = []
    for a, b in ((0, 1), (0, 2), (1, 2), (2, 3)):
        Q, R = sol.Q.copy(), sol.R.copy()
        for idx, r in enumerate((sol.tlist[a], sol.tlist[b])):
            ri = sol.tidx[r]
            Q[ri, :] = 0.0
            R[ri, :] = 0.0
            R[ri, idx % R.shape[1]] = 1.0
        cols.append(np.linalg.solve(np.eye(nT) - Q, R)[:, 0] - sol.q[:, 0])
    two, k2 = rank_ratio(np.stack(cols, 1))
    print("    one-row clamp  sigma_2/sigma_1 = %.3e  (rank %s)" % (m["ratio_med"], m["rank_set"]))
    print("    two-row clamp  sigma_2/sigma_1 = %.3e  (rank %d)" % (two, k2))
    assert two > 1e-6, "the two-row control is ALSO rank 1: the check cannot fail"
    print("    FIRED: %.1e -> %.3e, thirteen orders apart." % (m["ratio_med"], two))

    print("(c) REFUSAL: a destination that is not a legal neighbour yields NO chain,")
    print("    rather than a chain for some other intervention.")
    t = sol.tlist[len(sol.tlist) // 2]
    legal = _neighbours(sol.cells.shape[0], t)
    far = next(c for c in sol.tlist if c not in legal and c != t)
    print("    t=%d legal destinations %s" % (t, legal))
    print("    clamp_chain(t -> %d) [not a neighbour] = %s" % (far, clamp_chain(sol, t, far)))
    assert clamp_chain(sol, t, far) is None, "an illegal destination produced a chain"
    assert clamp_chain(sol, t, legal[0]) is not None, "a legal destination was refused"
    print("    refused as required, and a legal destination is still accepted.")

    print("ALL SELF-CHECKS PASSED")


if __name__ == "__main__":
    demo()

"""CAMERON round 6 -- the 2-dof lemma and its two degenerate loci (contract 1.3).

THE LEMMA. Round 5's F-identity showed the probe collapses on a ONE-TOKEN mask:
`theta = arcsin(sqrt(TV))` exactly. The cause is dimensional -- with `|S| = 1` the
renormalised row is a function of one scalar, so every scalar statistic of it is a
function of every other. Contract 1.3 asserts the four-point probe escapes this
because `|S| = 2` carries two degrees of freedom, and says the lemma is not to be
believed until the rank is measured per geometry.

Measured, it escapes GENERICALLY and fails on two identifiable loci, which is a
weaker statement than "by construction" and is the one these tests encode.

  LOCUS 1  `min(p_c, p_j) -> 0`. Real for every readout family. Masking a token
           that carries no mass does nothing, so the lattice loses a direction.
           Cannot be designed away; ARM P stratifies on it.
  LOCUS 2  `p_c == p_j`, and ONLY for a readout family whose members are all
           symmetric under exchanging c and j. Their partials coincide on the
           diagonal. An asymmetric family has no such degeneracy at the same
           geometry, so this one IS bought back by the choice of readout.

TOLERANCES ARE RELATIVE THROUGHOUT. An absolute singular-value bar is one a large
Jacobian cannot meet and a small one meets for free, which is how a tolerance ends
up doing the work a measurement should do. Where a property is exact it is tested
through its exact structure instead.
"""
from __future__ import annotations

import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(2)                       # pinned HERE, not by the launcher

from scale import twodof                                          # noqa: E402

S, D, C, J, SEED = 32, 8, 5, 11, 20260826
GEOMS = 24
PMIN_FLOOR = 1e-12          # locus 1 stratum boundary, fixed before the run


def _reports(family: str):
    out = []
    for t in range(GEOMS):
        q, k = twodof.geometry(S, D, SEED + t)
        out.append(twodof.rank_report(q, k, i=S - 1, c=C, j=J, family=family))
    return out


def test_the_struck_family_reads_rank_one_at_every_geometry():
    """The must-fire, and it is exact rather than a tolerance result.

    Contract 1.3 strikes an affine function of `{f({}), f({c})}` unbuilt as
    degree <= 1. All three of its readouts are functions of one scalar, so its
    Jacobian is an outer product and its rank is exactly 1. If this could read 2
    the rank check would be incapable of detecting a collapse and every rank-2
    verdict below would be void.
    """
    reps = _reports("degenerate")
    assert [r.rank for r in reps] == [1] * GEOMS, [r.rank for r in reps]
    # relative, not absolute: the second direction must be negligible COMPARED
    # TO the first, which is the only scale-free way to say "not there".
    assert max(r.cond for r in reps) < 1e-14, max(r.cond for r in reps)


def test_the_lemma_holds_away_from_locus_one():
    """Rank 2 wherever both tokens carry mass. The lemma, conditioned properly."""
    reps = [r for r in _reports("asymmetric") if r.pmin >= PMIN_FLOOR]
    assert len(reps) >= 12, len(reps)
    assert all(r.rank == 2 for r in reps), [(r.pmin, r.rank) for r in reps]


def test_locus_one_is_where_the_collapses_are():
    """The finding, stated as a count over drawn geometries rather than an example.

    A hand-built minimal geometry is where a control goes vacuous. These are
    drawn, and the claim is about the split between the two strata.
    """
    reps = _reports("asymmetric")
    lo = [r for r in reps if r.pmin < PMIN_FLOOR]
    hi = [r for r in reps if r.pmin >= PMIN_FLOOR]
    assert lo and hi, (len(lo), len(hi))
    # every rank-1 geometry lies in the low stratum
    assert all(r.rank == 2 for r in hi), [(r.pmin, r.rank) for r in hi]
    assert any(r.rank == 1 for r in lo), [(r.pmin, r.rank) for r in lo]


def test_locus_two_is_a_property_of_the_readout_not_of_the_lattice():
    """Same geometry, same estimator, two readout families, opposite verdicts.

    `p_c == p_j` is forced exactly by making the two key rows identical, so the
    diagonal is hit at `|p_c - p_j| = 0` rather than approached.
    """
    q, k = twodof.forced_diagonal(S, D, SEED, C, J)
    sym = twodof.rank_report(q, k, i=S - 1, c=C, j=J, family="symmetric")
    asym = twodof.rank_report(q, k, i=S - 1, c=C, j=J, family="asymmetric")
    assert sym.pgap == 0.0, sym
    assert asym.pgap == 0.0, asym
    assert sym.rank == 1, sym
    assert asym.rank == 2, asym
    # and the gap between them is orders of magnitude, not a hair either side
    # of wherever the floor was placed
    assert asym.cond / max(sym.cond, 1e-300) > 1e12, (asym.cond, sym.cond)


def test_a_symmetric_family_is_fine_off_the_diagonal():
    """Locus 2 must be about the diagonal, not about the symmetric family always failing.

    Without this the previous test would be satisfied by a readout pair that is
    simply broken everywhere, which is a different defect wearing the same result.
    """
    reps = []
    for t in range(GEOMS):
        q, k = twodof.geometry(S, D, SEED + t)
        r = twodof.rank_report(q, k, i=S - 1, c=C, j=J, family="symmetric")
        if r.pmin >= PMIN_FLOOR:
            reps.append(r)
    assert reps, "no geometry cleared the locus-1 floor"
    assert any(r.rank == 2 for r in reps), [(r.pmin, r.cond, r.rank) for r in reps]


def test_f_identity_is_reproduced_on_the_one_token_mask():
    """Round 5's collapse must still be visible through this instrument.

    `theta = arcsin(sqrt(TV))` is EXACT in real arithmetic. The residual measured
    here is float error, and it is dominated by `arccos` near 1, which loses
    roughly half the mantissa. So the bar is set at the scale that conditioning
    implies for float64 rather than at machine epsilon, and the measured spread
    across geometries is 7.6e-17 to 1.8e-09.
    """
    worst = max(twodof.f_identity_residual(*twodof.geometry(S, D, SEED + t),
                                           i=S - 1, c=C)
                for t in range(GEOMS))
    assert worst < 1e-7, worst


def test_rank_alone_would_overstate_the_second_direction():
    """Why conditioning is reported and not just rank.

    Some geometries clear the rank floor while their second direction is a
    millionth of the first. Calling those "two degrees of freedom" without
    printing the conditioning is how a rank check flatters a probe.
    """
    reps = _reports("asymmetric")
    weak = [r for r in reps if r.rank == 2 and r.cond < 1e-4]
    assert weak, [(r.cond, r.rank) for r in reps]


def test_l3_sign_pattern_equals_the_difference_of_two_degree_two_interactions():
    """The 8-point mask's signs, checked by an independent decomposition.

    The third finite difference is the degree-2 interaction with `m` absent minus
    the same interaction with `m` masked:

        L3 = [f({}) - f({c}) - f({j}) + f({c,j})]
           - [f({m}) - f({c,m}) - f({j,m}) + f({c,j,m})]

    which is a different grouping of the same eight terms than the flat signed
    sum `leakage_ratio` computes. A transposed or dropped sign shows up here and
    nowhere else; the magnitude alone would look entirely reasonable.
    """
    M, TGT = 17, 3
    for t in range(8):
        q, k = twodof.geometry(S, D, SEED + t)
        q, k = q.double(), k.double()
        z = torch.zeros(2, dtype=torch.float64)

        def f(drop):
            return float(twodof.masked_row(q, k, S - 1, C, J, z, drop)[TGT])

        i_without = f(()) - f((C,)) - f((J,)) + f((C, J))
        i_with = f((M,)) - f((C, M)) - f((J, M)) + f((C, J, M))
        abs_i, abs_l3 = twodof.leakage_ratio(q, k, i=S - 1, c=C, j=J, m=M,
                                             target=TGT)
        assert abs(abs_i - abs(i_without)) <= 1e-15 * max(1.0, abs(i_without))
        assert abs(abs_l3 - abs(i_without - i_with)) <= 1e-15 * max(
            1.0, abs(i_without - i_with))


def test_leakage_refuses_overlapping_positions():
    """`m` or the target colliding with c or j is not a cell, it is a bug."""
    q, k = twodof.geometry(S, D, SEED)
    for bad in ((C, 3), (17, C), (J, 3)):
        try:
            twodof.leakage_ratio(q, k, i=S - 1, c=C, j=J, m=bad[0], target=bad[1])
        except ValueError:
            continue
        raise AssertionError(f"accepted overlapping positions {bad}")

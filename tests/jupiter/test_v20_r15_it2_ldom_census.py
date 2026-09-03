"""V20 R15 it.2 -- JUPITER. The two nodes the Inspector struck as UNBOUND (C21, C20).

C21: the L-DOM census -- "Thm 13.10's hypotheses are satisfied by 0 of the
campaign's registered beds" -- had a producer (`scripts/v20_m14_cheeger.py::
domain_census`) that returns a formatted STRING and is written to a `.txt`. A
string is not an assertion. These nodes assert the census as numbers.

C20: `test_the_repo_crude_phi_is_an_upper_bound_on_the_exact_minimum` asserts
`crude >= exact - TOL` -- ONE-SIDED. It passes identically if crude were 10x
exact. The node below is TWO-SIDED and carries its own planted negative.

TOLERANCES.
  * `TOL = 1e-11` is inherited unchanged from `test_m14_cheeger.py:72`, whose
    docstring (`:37-46`) derives it as >5x the dominating error source on these
    same matrices. Reusing it means this file cannot be accused of picking a
    bar that admits its own reading.
  * The census assertions are EXACT integer comparisons -- counts, shapes and
    boolean predicates. No float tolerance applies and none is used.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.v20_m14_cheeger import (  # noqa: E402
    domain_census_facts,
    exact_min_conductance,
    repo_chain_weights,
    sweep_cut_conductance,
)

TOL = 1e-11


# ---------------------------------------------------------------------------
# C21 -- the L-DOM census, asserted
# ---------------------------------------------------------------------------
def test_ldom_census_bed_m_satisfies_the_hypotheses_on_zero_of_384_draws():
    f = domain_census_facts()["BED-M"]
    assert f["n_draws"] == 384
    assert f["H1_square_matrix_over_states"] is False
    assert f["draws_satisfying_H1_H4"] == 0


def test_ldom_census_the_ternary_object_satisfies_them_on_zero_of_2048_draws():
    f = domain_census_facts()["ternary"]
    assert f["n_draws"] == 2048
    assert set(f["value_support"]) <= {-1.0, 0.0, 1.0}
    assert f["negative_entries"] > 0          # so no pi >= 0 can exist
    assert f["H1_square_matrix_over_states"] is False
    assert f["draws_satisfying_H1_H4"] == 0


def test_ldom_census_bed_k_is_strictly_lower_triangular_and_nilpotent_on_all_8():
    f = domain_census_facts()["BED-K"]
    assert f["n_kernels_swept"] == 8
    assert f["strictly_lower_triangular"] is True
    assert f["nilpotent"] is True             # K^n == 0 => NOT irreducible => H3 fails
    assert f["row_sums_are_one"] is False     # => H2 fails independently
    assert f["draws_satisfying_H1_H4"] == 0


def test_ldom_census_verdict_zero_registered_beds_and_the_rips_graph_is_a_different_corpus():
    c = domain_census_facts()
    registered = ("BED-M", "ternary", "BED-K")
    assert sum(c[k]["draws_satisfying_H1_H4"] for k in registered) == 0
    assert c["registered_beds_satisfying_H1_H4"] == 0
    # the one object that DOES satisfy them is not a registered bed
    assert c["Rips_chain"]["draws_satisfying_H1_H4"] == 2
    assert c["Rips_chain"]["is_a_registered_bed"] is False


def test_planted_negative_the_census_predicate_can_return_nonzero():
    """V-16: a check that cannot fail is not a check.

    The predicate is `H1 and H2 and H3 and H4`. Feed it the object the census
    itself reports as SATISFYING all four (the Rips chain) and the same predicate
    returns a nonzero count. The zero is therefore a measurement, not a constant.
    """
    c = domain_census_facts()
    assert c["Rips_chain"]["draws_satisfying_H1_H4"] > 0
    assert c["_predicate_is_the_same_function"] is True


# ---------------------------------------------------------------------------
# C20 -- the two-sided node
# ---------------------------------------------------------------------------
def _crude_and_exact():
    W, meta = repo_chain_weights("LargestJoin_S2Rips_64")
    return float(meta["phi_bridge"]), float(exact_min_conductance(W)["phi"]), W


def test_the_repo_crude_phi_EQUALS_the_exact_minimum_two_sided():
    """The struck claim, bound in both directions -- by ONE reachable assertion.

    `crude >= exact - TOL` alone admits crude = 10*exact (it.1 C20). The it.2
    repair added `crude <= exact + TOL` and a ratio bound, three assertions.

    it.4: the two one-sided lines were DELETED as unreachable, not as noise.
    DERIVED: on this instance `exact = 1/27 = 0.037037...`, so
    `|crude/exact - 1| <= TOL` is `|crude - exact| <= TOL*exact = 3.70e-13`,
    while either one-sided line is `|crude - exact| <= TOL = 1e-11`. The ratio
    bound is 27x tighter on BOTH sides, so no mutation exists that a one-sided
    line catches and the ratio does not. Deleting either one-sided line removes
    nothing the node still asserts -- which is why the Inspector's C7 probe
    ("delete the upper-side assert and confirm the planted negative fails")
    could not have discriminated against them either. What remains is one line
    that is two-sided, strictly stronger than the deleted pair, and reachable:
    see `test_planted_negative_...`, which now invokes THIS function.
    """
    crude, exact, _W = _crude_and_exact()
    assert crude / exact == pytest.approx(1.0, abs=TOL), (crude, exact, crude / exact)


def test_the_repo_crude_phi_is_the_planted_rational_one_over_twentyseven():
    crude, exact, _W = _crude_and_exact()
    assert exact == pytest.approx(1.0 / 27.0, abs=TOL), (exact, 1.0 / 27.0)
    assert crude == pytest.approx(1.0 / 27.0, abs=TOL), (crude, 1.0 / 27.0)


@pytest.mark.parametrize("name,factor,offset", [
    # C20's own counterexample: "would pass identically if crude were 10x exact".
    ("crude_x10", 10.0, 0.0),
    # The discriminating mutation: +TOL/2 clears the DELETED upper-side line
    # (`crude <= exact + TOL`) and still fires the ratio bound, which is 27x
    # tighter. This is the mutation that shows the surviving line is the one
    # doing the work, and that the deleted pair was dominated, not load-bearing.
    ("crude_plus_half_tol", 1.0, TOL / 2.0),
])
def test_planted_negative_the_two_sided_node_fires_when_crude_is_inflated(
        monkeypatch, name, factor, offset):
    """V-16 for the two-sided node, repaired for it.2 C7.

    C7: the it.2 form re-declared the assertion on a LOCAL and never invoked
    `test_the_repo_crude_phi_EQUALS_the_exact_minimum_two_sided`, so deleting
    that node's assertion left this negative passing. It guarded a copy of the
    text it was supposed to guard.

    THE REPAIR: monkeypatch the module-global `_crude_and_exact` the real node
    calls, then CALL THE REAL NODE. Nothing is re-declared here. Delete the
    surviving assertion from that node and this negative stops raising and
    FAILS -- verified by doing exactly that, `V20_R15_IT4_JUPITER.md` TASK B.
    """
    crude, exact, W = _crude_and_exact()
    monkeypatch.setitem(
        globals(), "_crude_and_exact",
        lambda: (crude * factor + offset, exact, W))
    with pytest.raises(AssertionError):
        test_the_repo_crude_phi_EQUALS_the_exact_minimum_two_sided()


def test_planted_negative_the_sweep_path_is_not_hardwired_to_the_exact_path():
    """Both paths report 1/27 on this instance. Show they are two computations:
    a NAMED permutation of the vertex labels -- `numpy.random.default_rng(20152)
    .permutation(n)` -- relabels W, and both paths must still agree, while the
    sweep's reported ordering must CHANGE. A hardwired constant would not move.
    """
    _crude, exact, W = _crude_and_exact()
    n = W.shape[0]
    perm = np.random.default_rng(20152).permutation(n)
    Wp = W[np.ix_(perm, perm)]
    assert not np.array_equal(perm, np.arange(n)), "seed 20152 must not be identity"
    assert sweep_cut_conductance(Wp)["order"] != sweep_cut_conductance(W)["order"]
    assert exact_min_conductance(Wp)["phi"] == pytest.approx(exact, abs=TOL)
    assert sweep_cut_conductance(Wp)["phi"] == pytest.approx(
        sweep_cut_conductance(W)["phi"], abs=TOL)

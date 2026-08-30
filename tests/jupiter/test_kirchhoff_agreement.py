"""The instrument law: two independent oracles for the same Dirichlet solution.

`scale/e4_harmonic.py` computes the harmonic measure by an absorbing-chain solve,
`B = N R` with `N = (I - Q)^-1`. `scale/kirchhoff.py` computes the same numbers
from the matrix-tree theorem, as a ratio of spanning-forest counts read off the
grounded Laplacian. The two share the graph and nothing else: one inverts a
sub-stochastic transient block indexed by the transient nodes only, the other
solves a symmetric Laplacian system on the whole component in which the first
absorbing endpoint is an ordinary interior column. A transposition, a wrong
degree normalisation or an off-by-one in either index map moves one and not the
other, which is the entire reason for keeping both.

The last tests in this file plant such an off-by-one in a scratch copy of the
absorbing-chain oracle and require the agreement check to fire on it. An
agreement test that has never seen a disagreement measures nothing.
"""
import numpy as np
import pytest

from scale import kirchhoff as K
from scale.e4_harmonic import (SHIPPED_CASE, SMALL_CASE, absorbing_chain,
                                case_graph, fixed_point)


def drawn_graphs(count=6, n=9, p=0.45, seed=0):
    """Connected Erdos-Renyi draws. Drawn, never hand-built."""
    rng = np.random.RandomState(seed)
    out = []
    while len(out) < count:
        adjacency = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if rng.rand() < p:
                    adjacency[i].add(j)
                    adjacency[j].add(i)
        if len(K.component_of(adjacency, 0)) == n:
            out.append(adjacency)
    return out


def test_the_laplacian_cofactor_is_the_spanning_tree_count():
    """Kirchhoff, against brute-force enumeration of spanning trees."""
    hits = 0
    for adjacency in drawn_graphs(count=4, n=7, p=0.5, seed=1):
        nodes = sorted(range(len(adjacency)))
        lap, _index = K.laplacian(adjacency, nodes)
        by_determinant = K.spanning_tree_count(lap)
        by_enumeration = K.brute_force_spanning_trees(adjacency, nodes)
        assert by_enumeration > 0
        assert abs(by_determinant - by_enumeration) <= 1e-6 * by_enumeration
        hits += 1
    assert hits == 4


def test_the_two_forest_count_is_the_double_minor_determinant():
    """`det(L with rows/cols a and b deleted)` counts spanning 2-forests
    separating `a` from `b`."""
    for adjacency in drawn_graphs(count=3, n=7, p=0.5, seed=2):
        nodes = sorted(range(len(adjacency)))
        lap, index = K.laplacian(adjacency, nodes)
        a, b = 0, len(nodes) - 1
        by_determinant = K.separating_forest_count(lap, index[a], index[b])
        by_enumeration = K.brute_force_separating_forests(adjacency, nodes, a, b)
        assert by_enumeration > 0
        assert abs(by_determinant - by_enumeration) <= 1e-6 * by_enumeration


def test_the_harmonic_measure_is_the_spanning_forest_ratio():
    """`omega_v = F(v,a | b) / F(a | b)`, the right side counted by enumeration
    and the left side solved from the grounded Laplacian."""
    for adjacency in drawn_graphs(count=3, n=7, p=0.5, seed=3):
        nodes = sorted(range(len(adjacency)))
        a, b = 0, len(nodes) - 1
        omega, transient = K.harmonic_measure(adjacency, nodes, (a, b))
        denominator = K.brute_force_separating_forests(adjacency, nodes, a, b)
        assert denominator > 0
        for slot, v in enumerate(transient):
            numerator = K.brute_force_separating_forests(
                adjacency, nodes, a, b, together=v)
            assert abs(omega[slot] - numerator / denominator) < 1e-9


def test_effective_resistance_agrees_with_the_determinant_ratio():
    """`R(x,y) = det(L_xy) / det(L_x)` against the grounded-inverse form."""
    for adjacency in drawn_graphs(count=3, n=8, p=0.5, seed=4):
        nodes = sorted(range(len(adjacency)))
        lap, index = K.laplacian(adjacency, nodes)
        grounded = K.grounded_inverse(lap, 0)
        for x, y in ((1, 2), (0, 3), (2, 7)):
            by_inverse = K.resistance_from_grounded(grounded, index[x], index[y])
            by_determinant = (K.separating_forest_count(lap, index[x], index[y])
                              / K.spanning_tree_count(lap))
            assert by_inverse > 0.0
            assert abs(by_inverse - by_determinant) < 1e-8 * max(1.0, by_inverse)


def test_the_two_oracles_agree_on_drawn_erdos_renyi_instances():
    """The instrument law, on drawn graphs. Also asserts the label is
    non-degenerate, so the agreement is not agreement on a constant."""
    discriminating = 0
    for adjacency in drawn_graphs(count=6, n=9, p=0.45, seed=5):
        nodes = sorted(range(len(adjacency)))
        bridge = (0, len(nodes) - 1)
        gap, omega = K.oracle_gap(adjacency, nodes, bridge)
        assert gap < K.AGREEMENT_TOL, f"gap {gap:.3e}"
        assert omega.std() > 1e-3, "label is constant; agreement says nothing"
        assert 0.0 < omega.mean() < 1.0
        discriminating += 1
    assert discriminating == 6


@pytest.mark.parametrize("case", [SMALL_CASE, SHIPPED_CASE])
def test_the_two_oracles_agree_on_the_drawn_rips_dirichlet_instance(case):
    """The instrument law on the real E4' substrate, at both sizes the module
    ships. The shipped case is the one the gates are read on, so it is the one
    the law has to hold at; the small case is kept because a law that only ever
    runs at one size has never been shown to scale."""
    _name, n, degree, seed = case
    adjacency, nodes, bridge = case_graph(n, degree, seed)
    gap, omega = K.oracle_gap(adjacency, nodes, bridge)
    assert gap < K.AGREEMENT_TOL, f"gap {gap:.3e}"
    assert omega.std() > 1e-3
    assert 0.0 < omega.mean() < 1.0


def test_the_agreement_check_fires_on_a_planted_off_by_one():
    """THE MUST-FIRE. A scratch copy of the absorbing-chain oracle carrying one
    planted off-by-one in its transient index map. The check must reject it on
    every drawn instance, and the count of instances on which it discriminates
    is asserted rather than assumed."""
    caught = 0
    total = 0
    for adjacency in drawn_graphs(count=6, n=9, p=0.45, seed=5):
        nodes = sorted(range(len(adjacency)))
        bridge = (0, len(nodes) - 1)
        clean, omega = K.oracle_gap(adjacency, nodes, bridge)
        assert clean < K.AGREEMENT_TOL
        total += 1
        defective = K.scratch_chain_with_off_by_one(adjacency, nodes, bridge)
        if float(np.max(np.abs(omega - defective))) >= K.AGREEMENT_TOL:
            caught += 1
    assert total == 6
    assert caught == 6, f"the check fired on only {caught}/{total} planted defects"


def test_the_planted_defect_is_reachable_through_the_shipped_entry_point():
    """The defect must be caught by `assert_oracles_agree`, the function the
    instrument law actually calls, not only by a hand comparison."""
    adjacency = drawn_graphs(count=1, n=9, p=0.45, seed=5)[0]
    nodes = sorted(range(len(adjacency)))
    bridge = (0, len(nodes) - 1)
    K.assert_oracles_agree(adjacency, nodes, bridge)          # clean: no raise
    with pytest.raises(AssertionError):
        K.assert_oracles_agree(adjacency, nodes, bridge,
                               chain=K.scratch_chain_with_off_by_one)


def test_the_kirchhoff_oracle_refuses_a_damped_walk():
    """The forest identity is the identity for the undamped walk. A killed walk
    is a different operator and the law must decline rather than mislead."""
    adjacency = drawn_graphs(count=1, n=9, p=0.45, seed=5)[0]
    nodes = sorted(range(len(adjacency)))
    with pytest.raises(ValueError):
        K.assert_oracles_agree(adjacency, nodes, (0, 8), kill=1 / 16)


def test_agreement_against_the_shipped_fixed_point_function():
    """The Kirchhoff oracle is compared against `e4_harmonic.fixed_point`
    itself, not against a local reimplementation of it."""
    for adjacency in drawn_graphs(count=3, n=9, p=0.45, seed=5):
        nodes = sorted(range(len(adjacency)))
        bridge = (0, len(nodes) - 1)
        q, r, transient = absorbing_chain(adjacency, nodes, bridge, kill=0.0)
        shipped = fixed_point(q, r)
        omega, mine = K.harmonic_measure(adjacency, nodes, bridge)
        assert mine == transient
        assert float(np.max(np.abs(shipped - omega))) < K.AGREEMENT_TOL

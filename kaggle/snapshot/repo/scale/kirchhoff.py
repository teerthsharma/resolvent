"""A second exact oracle for the E4' harmonic measure, from the matrix-tree theorem.

WHAT THIS IS. `scale/e4_harmonic.py` labels its Dirichlet instances with an
absorbing-chain solve: partition the transition matrix, form the fundamental
matrix `N = (I - Q)^-1`, and read `B = N R`. That is one computation of the
harmonic measure `omega_v = P_v(tau_a < tau_b)`. This module computes the same
numbers a second time by a route that shares the graph and nothing else, and the
two are then required to agree. The repo has already lost a producer that
acquired a defect after publication with nobody noticing; two derivations of the
same number rarely break identically, and that redundancy is the point.

THE DERIVATION, IN FOUR STEPS.

1. KIRCHHOFF. For the unweighted graph `G` on `n` nodes with Laplacian
   `L = D - A`, every cofactor of `L` equals `tau(G)`, the number of spanning
   trees. Concretely `det(L_g) = tau(G)` where `L_g` deletes row and column `g`.

2. TWO-FOREST COUNT. Deleting two rows and columns gives
   `det(L_{xy}) = F(x | y)`, the number of spanning 2-forests in which `x` and
   `y` lie in different trees. This is the all-minors form of step 1.

3. EFFECTIVE RESISTANCE. Ground at `x`. By Cramer, `M = (L_x)^-1` has
   `M_yy = det(L_{xy}) / det(L_x) = F(x | y) / tau(G)`, and `M_xx = M_xy = 0`
   because the grounded row and column are gone. Since effective resistance is
   `R(x,y) = (e_x - e_y)^T L^+ (e_x - e_y)` and `M` differs from `L^+` only along
   the all-ones direction, which cancels in the difference,

       R(x, y) = M_xx + M_yy - 2 M_xy = F(x | y) / tau(G).

   Effective resistance is a ratio of spanning-forest counts. That is the
   classical corollary chain, and it is exact.

4. HARMONIC MEASURE. Injecting unit current from `a` to `b` gives potential
   `v = L^+ (e_a - e_b)`, and polarisation on
   `(e_x - e_b) - (e_a - e_b) = e_x - e_a` yields

       v(x) - v(b) = (R(x,b) + R(a,b) - R(x,a)) / 2.

   Normalising by `v(a) - v(b) = R(a,b)` gives the harmonic function with
   boundary values 1 at `a` and 0 at `b`, which is exactly
   `omega_x = P_x(tau_a < tau_b)`. Now GROUND AT `b`. Then `R(x,b) = M_xx`,
   `R(a,b) = M_aa` and `R(x,a) = M_xx + M_aa - 2 M_xa`, so the three-term
   expression collapses with no cancellation at all:

       omega_x = M_xa / M_aa = F(x, a | b) / F(a | b).

   The harmonic measure is a ratio of spanning-forest counts, and computing it
   needs ONE solve of `L_b z = e_a` -- never a full inverse and never a
   subtraction of two large resistances.

WHY THIS IS INDEPENDENT OF THE CHAIN SOLVE, AND NOT A RESTATEMENT OF IT. The
chain path inverts `I - Q`, where `Q` is the sub-stochastic transient block
`D^-1 A` restricted to the nodes that are neither endpoint, indexed by a map
that omits both endpoints. The Kirchhoff path solves a SYMMETRIC, unnormalised
`L_b = D - A` on the whole component, in which `a` is an ordinary interior
column and only `b` is removed, and it reads a single column of the answer. The
two systems differ in their matrix, in their normalisation, in their right-hand
side and in their index map. A transposed adjacency, a degree counted one too
high, or a slot shifted by one moves one of them and not the other. `L_T` and
`I - Q` are diagonally related -- `I - Q = D^-1 L_T` -- which is exactly why the
Kirchhoff route deliberately does NOT form `L_T`.

THE INSTRUMENT LAW. `assert_oracles_agree` is called by
`e4_harmonic.measure()` on every Dirichlet instance it builds, and requires
`max |omega_kirchhoff - omega_chain| < AGREEMENT_TOL`. MEASURED gaps, this
machine, numpy 1.26.4, float64: `2.220446e-16` over the six drawn Erdos-Renyi
instances of `tests/jupiter/test_kirchhoff_agreement.py`, `8.992806e-15` on
`SMALL_CASE` (62 merged nodes, 60 transient) and `9.636736e-14` on
`SHIPPED_CASE` (1202 merged nodes, 1200 transient). The gap grows with size, as
a conditioning argument says it must, and is still three orders below the
tolerance at the largest instance the corpus draws.

THE PLANTED DEFECT. `scratch_chain_with_off_by_one` is a scratch copy of
`absorbing_chain` carrying one off-by-one -- the walk divides by
`len(neighbours) + 1`, the bug you get by counting the node itself among its own
neighbours. It is a scratch copy on purpose: nothing in the shipped codepath
imports it. `tests/jupiter/test_kirchhoff_agreement.py` requires the agreement
check to reject it on every drawn instance, because an agreement test that has
never seen a disagreement is a control that has never been shown to
discriminate.

SCOPE. The forest identity is the identity for the UNDAMPED walk. A killed walk
(`e4_harmonic.KILL_RATE > 0`) is a different operator whose harmonic measure is
not a spanning-forest ratio of `G`, so `assert_oracles_agree` raises rather than
comparing two different objects and reporting the difference as a defect.
"""
from __future__ import annotations

import itertools

import numpy as np

__all__ = ["AGREEMENT_TOL", "component_of", "laplacian", "spanning_tree_count",
           "separating_forest_count", "grounded_inverse",
           "resistance_from_grounded", "harmonic_measure", "chain_oracle",
           "scratch_chain_with_off_by_one", "oracle_gap", "assert_oracles_agree",
           "brute_force_spanning_trees", "brute_force_separating_forests"]

#: The instrument law's tolerance. Set from the measured gaps quoted in the
#: module docstring, not chosen after seeing a result: the largest observed
#: disagreement between two CORRECT oracles is `9.636736e-14` at the shipped
#: size, and the smallest disagreement the planted defect produces over the
#: drawn battery is `1.749951e-01` -- twelve orders of magnitude of daylight.
#: `1e-10` is the round number inside that window, three orders above the
#: largest clean gap and nine below the smallest defect.
AGREEMENT_TOL = 1e-10


def component_of(adjacency, root: int) -> list[int]:
    """The connected component containing `root`, sorted. Used to draw test
    instances and to check a caller's `nodes` really is one component."""
    seen = {root}
    stack = [root]
    while stack:
        u = stack.pop()
        for v in adjacency[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return sorted(seen)


def laplacian(adjacency, nodes) -> tuple[np.ndarray, dict[int, int]]:
    """`(L, index)` with `L = D - A` on the induced subgraph, unit conductances.

    `nodes` must be a full connected component of `adjacency`; the degree used
    is the node's degree in `adjacency`, which coincides with its degree in the
    induced subgraph exactly when no edge leaves `nodes`. That coincidence is
    what lets this Laplacian be compared against a chain built from
    `1 / len(adjacency[v])`, so it is checked rather than assumed.
    """
    index = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    lap = np.zeros((n, n))
    for v in nodes:
        i = index[v]
        for u in adjacency[v]:
            if u not in index:
                raise ValueError(f"edge {v}-{u} leaves the node set; `nodes` is "
                                 "not a union of components")
            lap[i, index[u]] -= 1.0
        lap[i, i] = float(len(adjacency[v]))
    return lap, index


def spanning_tree_count(lap: np.ndarray, ground: int = 0) -> float:
    """`tau(G) = det(L_ground)`. Kirchhoff's matrix-tree theorem.

    Returned as a float. The count overflows any integer type well before the
    sizes this repo draws -- a 1202-node Rips component has on the order of
    `10^900` spanning trees -- and every use here is a RATIO of two such counts,
    where the overflow cancels. `harmonic_measure` therefore never forms either
    count; it is this function that exists to be checked against enumeration.
    """
    keep = [i for i in range(len(lap)) if i != ground]
    return float(np.linalg.det(lap[np.ix_(keep, keep)]))


def separating_forest_count(lap: np.ndarray, i: int, j: int) -> float:
    """`F(i | j) = det(L with rows and columns i and j deleted)` -- the number of
    spanning 2-forests putting `i` and `j` in different trees."""
    if i == j:
        raise ValueError("a 2-forest separating a node from itself is not defined")
    keep = [k for k in range(len(lap)) if k not in (i, j)]
    if not keep:
        return 1.0
    return float(np.linalg.det(lap[np.ix_(keep, keep)]))


def grounded_inverse(lap: np.ndarray, ground: int) -> np.ndarray:
    """`(L_ground)^-1`, padded back to full size with zeros in the ground's row
    and column. Used only by the resistance checks; `harmonic_measure` needs one
    column of this and solves for it instead."""
    n = len(lap)
    keep = [i for i in range(n) if i != ground]
    inv = np.linalg.inv(lap[np.ix_(keep, keep)])
    out = np.zeros((n, n))
    out[np.ix_(keep, keep)] = inv
    return out


def resistance_from_grounded(grounded: np.ndarray, i: int, j: int) -> float:
    """`R(i,j) = M_ii + M_jj - 2 M_ij`, valid for any choice of ground."""
    return float(grounded[i, i] + grounded[j, j] - 2.0 * grounded[i, j])


def harmonic_measure(adjacency, nodes, bridge) -> tuple[np.ndarray, list[int]]:
    """THE KIRCHHOFF ORACLE. `omega_v = P_v(tau_a < tau_b)` for every transient
    `v`, as `M_va / M_aa` with `M` the `b`-grounded inverse Laplacian.

    Returns `(omega, transient)` with `transient` in the SAME order
    `e4_harmonic.absorbing_chain` produces it, so the two oracles' outputs are
    directly comparable without a reordering step that could itself be wrong.

    One solve, `L_b z = e_a`, never a full inverse: only column `a` of `M` is
    read. `z_a = M_aa = R(a,b) > 0` on a connected component, so the division is
    safe and needs no guard beyond the connectivity `laplacian` already checks.
    """
    a, b = int(bridge[0]), int(bridge[1])
    if a == b:
        raise ValueError("the bridge endpoints must differ")
    lap, index = laplacian(adjacency, nodes)
    n = len(nodes)
    keep = [i for i in range(n) if i != index[b]]
    rhs = np.zeros(len(keep))
    rhs[keep.index(index[a])] = 1.0
    column = np.linalg.solve(lap[np.ix_(keep, keep)], rhs)
    potential = np.zeros(n)
    potential[keep] = column
    scale = potential[index[a]]
    transient = [v for v in nodes if v not in (a, b)]
    omega = np.array([potential[index[v]] / scale for v in transient])
    return omega, transient


def chain_oracle(adjacency, nodes, bridge, kill: float = 0.0) -> np.ndarray:
    """THE SHIPPED ORACLE, called through `e4_harmonic` itself rather than
    reimplemented here. Comparing against a local copy would compare this
    module to itself."""
    from scale.e4_harmonic import absorbing_chain, fixed_point
    q, r, _transient = absorbing_chain(adjacency, nodes, bridge, kill=kill)
    return fixed_point(q, r)


def scratch_chain_with_off_by_one(adjacency, nodes, bridge,
                                  kill: float = 0.0) -> np.ndarray:
    """A SCRATCH COPY OF `absorbing_chain` CARRYING ONE PLANTED OFF-BY-ONE.

    The single change from `e4_harmonic.absorbing_chain` is `len(neighbours) + 1`
    in place of `len(neighbours)` -- the degree miscount you get by including the
    node itself among its own neighbours. It is deliberately the survivable kind
    of bug: every row stays strictly sub-stochastic, so `I - Q` remains strictly
    diagonally dominant and the solve succeeds and returns plausible numbers in
    `[0, 1]`. Nothing downstream would notice it. The agreement check does.

    NOTHING IMPORTS THIS FROM THE SHIPPED PATH. It exists so that
    `tests/jupiter/test_kirchhoff_agreement.py` can show the instrument law
    firing on a defect, on drawn instances, with the count of instances on which
    it discriminates asserted.
    """
    absorbing = set(bridge)
    transient = [v for v in nodes if v not in absorbing]
    index = {v: i for i, v in enumerate(transient)}
    m = len(transient)
    q = np.zeros((m, m))
    r = np.zeros(m)
    for v in transient:
        neighbours = adjacency[v]
        step = (1.0 - kill) / (len(neighbours) + 1)          # <-- planted defect
        for u in neighbours:
            if u == bridge[0]:
                r[index[v]] += step
            elif u != bridge[1]:
                q[index[v], index[u]] += step
    return np.linalg.solve(np.eye(m) - q, r)


def oracle_gap(adjacency, nodes, bridge, *, kill: float = 0.0,
               chain=chain_oracle) -> tuple[float, np.ndarray]:
    """`(max |omega_kirchhoff - omega_chain|, omega_kirchhoff)`.

    `chain` is injectable so the planted defect can be pushed through the same
    entry point the shipped oracle uses. It defaults to the shipped one.
    """
    if kill != 0.0:
        raise ValueError("the spanning-forest identity is the identity for the "
                         "undamped walk; kill must be 0.0")
    omega, _transient = harmonic_measure(adjacency, nodes, bridge)
    other = chain(adjacency, nodes, bridge, kill=kill)
    if other.shape != omega.shape:
        raise ValueError(f"oracles disagree on shape: {other.shape} vs {omega.shape}")
    return float(np.max(np.abs(omega - other))), omega


def assert_oracles_agree(adjacency, nodes, bridge, *, kill: float = 0.0,
                         tol: float = AGREEMENT_TOL, chain=chain_oracle) -> float:
    """THE INSTRUMENT LAW. Runs both oracles and returns the gap, raising if it
    is at or above `tol`. Called by `e4_harmonic.measure` on every instance."""
    gap, _omega = oracle_gap(adjacency, nodes, bridge, kill=kill, chain=chain)
    assert gap < tol, (
        f"the two harmonic-measure oracles disagree by {gap:.6e} >= {tol:.0e} on "
        f"a {len(nodes)}-node component with bridge {tuple(bridge)}. One of the "
        "two codepaths is wrong; do not read a label off either until it is found.")
    return gap


# --------------------------------------------------------------------------
# Brute force. These exist to tie the determinants of the first two steps of the
# derivation to the combinatorial objects they are claimed to count, on graphs
# small enough to enumerate. They are exponential and are never called on a
# drawn Rips instance.
# --------------------------------------------------------------------------

def _edge_list(adjacency, nodes) -> list[tuple[int, int]]:
    keep = set(nodes)
    return sorted({(min(v, u), max(v, u))
                   for v in nodes for u in adjacency[v] if u in keep})


def _forest_components(nodes, chosen) -> list[set[int]] | None:
    """The components of `chosen`, or None if `chosen` contains a cycle."""
    parent = {v: v for v in nodes}

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for x, y in chosen:
        rx, ry = find(x), find(y)
        if rx == ry:
            return None
        parent[rx] = ry
    groups: dict[int, set[int]] = {}
    for v in nodes:
        groups.setdefault(find(v), set()).add(v)
    return list(groups.values())


def brute_force_spanning_trees(adjacency, nodes) -> int:
    """`tau(G)` by enumerating every `|V| - 1` subset of edges."""
    edges = _edge_list(adjacency, nodes)
    return sum(1 for chosen in itertools.combinations(edges, len(nodes) - 1)
               if _forest_components(nodes, chosen) is not None)


def brute_force_separating_forests(adjacency, nodes, a: int, b: int,
                                   together: int | None = None) -> int:
    """Spanning 2-forests with `a` and `b` in different trees.

    With `together=v`, additionally requires `v` in the same tree as `a` -- the
    numerator `F(v, a | b)` of the harmonic-measure ratio.
    """
    edges = _edge_list(adjacency, nodes)
    total = 0
    for chosen in itertools.combinations(edges, len(nodes) - 2):
        groups = _forest_components(nodes, chosen)
        if groups is None:
            continue
        side_a = next(g for g in groups if a in g)
        if b in side_a:
            continue
        if together is not None and together not in side_a:
            continue
        total += 1
    return total

"""N2: the Gromov delta-hyperbolicity instrument, contract v10.1 section N2.

WHAT IS BEING REGISTERED. `scale.hyperbolic.gromov_delta` -- the delta-hyperbolicity
measurable the campaign adds for its corpus graphs. The operative definition is the
Gromov-product form: with `(y|z)_x = 1/2(d(x,y)+d(x,z)-d(y,z))`, a metric space is
delta-hyperbolic iff for all quadruples `(w,x,y,z)` of DISTINCT points with `x != y`,

    (x|y)_w >= min((x|z)_w, (z|y)_w) - delta,

and the instrument returns the optimal

    delta_hat = max over those quadruples of [min((x|z)_w, (z|y)_w) - (x|y)_w].

Trees read delta_hat = 0 EXACTLY (in a tree the meeting points satisfy
`(x|y)_w >= min((x|z)_w, (z|y)_w)` identically -- the classical lemma), and a cycle
C_n reads delta_hat = n/4 (girth/4). The full derivation, including the equivalent
four-point-sum form `delta_hat = max(sigma_top - sigma_second)` over unordered
quadruples and why repeated indices are excluded (with the y=x counterexample), is
documented in the module docstring of scale/hyperbolic.py. DERIVED evidence class.

WHY RED-FIRST. House law. Every planted control below is written before the module
exists, confirmed RED, then made GREEN. Kill switch K-R8c binds: if the instrument
fails ANY planted control, every delta_hat number in results/delta_hyperbolicity.* is
VOID until repaired. The controls:

    random trees (several sizes)      delta_hat == 0.0 exactly
    cycles C_n, 4 | n                 |delta_hat - n/4| <= 1e-9
    odd cycle C_13                    0 < delta_hat <= 13/2 (sanity only)
    complete graphs                   delta_hat == 0.0 (derived in the test)
    E4' motif (two cliques + bridge)  printed, bounded by the diameter
    brute-force cross-reference       exact agreement to 1e-12 on random graphs
    sampled mode                      never exceeds exact; controls hold

NON-DEGENERACY (PASS-half law): tree-zero alone proves nothing -- a broken constant
zero would pass it. Each zero-claiming control is therefore PAIRED with a strictly
positive counterpart of the same size (tree vs cycle), and the brute-force control
pins the formula itself against an independent literal implementation.
"""
from __future__ import annotations

import importlib
import importlib.util
import itertools
import math
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_MODULE = "scale.hyperbolic"
_REQUIRED = ("gromov_delta", "hop_distances")


def _require_hyperbolic():
    """Every test funnels through here so that RED is a plain AssertionError
    naming the missing symbol, never an ImportError raised mid-collection."""
    spec = importlib.util.find_spec(_MODULE)
    assert spec is not None, (
        f"{_MODULE} missing: the contract v10.1 section N2 instrument "
        f"(gromov_delta) is not implemented yet")
    mod = importlib.import_module(_MODULE)
    missing = [attr for attr in _REQUIRED if not hasattr(mod, attr)]
    assert not missing, f"{_MODULE} incomplete, missing: {missing}"
    return mod


# --------------------------------------------------------------- generators --
def _random_tree(n: int, seed: int):
    """Random recursive tree on n nodes: node k attaches to a uniform earlier
    node. Connected by construction, and every tree, however lopsided, must
    read delta_hat == 0."""
    rng = np.random.default_rng(seed)
    return [(int(k), int(rng.integers(0, k))) for k in range(1, n)]


def _cycle(n: int):
    return [(i, (i + 1) % n) for i in range(n)]


def _complete(n: int):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _two_cliques_bridge(m: int, k: int):
    """The E4' shape at toy scale: two cliques joined by ONE bridge edge."""
    edges = _complete(m) + [(a + m, b + m) for a, b in _complete(k)]
    return edges + [(0, m)], m + k


def _connected_random_graph(n: int, p_extra: float, seed: int):
    """Spanning path (guarantees connectivity) plus Bernoulli extra edges."""
    rng = np.random.default_rng(seed)
    edges = set(tuple(sorted(e)) for e in _cycle(n))
    for i in range(n):
        for j in range(i + 2, n):
            if rng.random() < p_extra:
                edges.add((i, j))
    return sorted(edges)


# ------------------------------------------------------------ brute force ref --
def _brute_force_delta(dist: np.ndarray) -> float:
    """Literal transcription of the documented definition: ordered DISTINCT
    quadruples (w, x, y, z) with x != y; term = min((x|z)_w, (z|y)_w) - (x|y)_w.
    O(n^4) python; used ONLY on tiny graphs to pin the vectorized instrument."""
    n = dist.shape[0]
    best = -math.inf
    for w, x, y, z in itertools.permutations(range(n), 4):
        if x == y:
            continue
        p_xz = 0.5 * (dist[w, x] + dist[w, z] - dist[x, z])
        p_zy = 0.5 * (dist[w, z] + dist[w, y] - dist[z, y])
        p_xy = 0.5 * (dist[w, x] + dist[w, y] - dist[x, y])
        best = max(best, min(p_xz, p_zy) - p_xy)
    return 0.0 if best == -math.inf else float(best)


# ------------------------------------------------------------------ 0. RED gate --
def test_the_instrument_module_and_entry_points_exist():
    _require_hyperbolic()


# ------------------------------------------------------- 1. formula pinning --
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_exact_mode_agrees_with_literal_brute_force(seed):
    """THE formula control. Three random connected graphs, brute-forced by the
    independent literal implementation above; any disagreement voids every
    downstream delta_hat number (K-R8c)."""
    mod = _require_hyperbolic()
    n = 9 + seed
    edges = _connected_random_graph(n, p_extra=0.25, seed=seed)
    dist = mod.hop_distances(n, edges)
    got = mod.gromov_delta(dist)
    want = _brute_force_delta(dist)
    assert abs(got - want) <= 1e-12, (got, want)


# ------------------------------------------------------------- 2. tree control --
@pytest.mark.parametrize("n,seed", [(8, 11), (16, 12), (33, 13), (65, 14)])
def test_random_trees_read_delta_exactly_zero(n, seed):
    """Trees are 0-hyperbolic with NO slack: the lemma
    (x|y)_w >= min((x|z)_w, (z|y)_w) holds identically, so the supremum of the
    violation is exactly the scalar 0.0 -- asserted with ==, not a tolerance.
    NON-DEGENERACY: the same-size cycle control below reads strictly positive,
    so this is a discriminating zero, not a broken constant zero."""
    mod = _require_hyperbolic()
    dist = mod.hop_distances(n, _random_tree(n, seed))
    assert mod.gromov_delta(dist) == 0.0


def test_tree_zero_is_discriminating_not_degenerate():
    """PASS-half non-degeneracy: on the SAME size, the tree reads exactly 0.0
    while the cycle reads strictly positive. Together the pair proves the
    instrument discriminates rather than constants out."""
    mod = _require_hyperbolic()
    n = 32
    tree = mod.hop_distances(n, _random_tree(n, 15))
    cyc = mod.hop_distances(n, _cycle(n))
    d_tree = mod.gromov_delta(tree)
    d_cycle = mod.gromov_delta(cyc)
    assert d_tree == 0.0, d_tree
    assert d_cycle > 0.0, d_cycle


# ------------------------------------------------------------ 3. cycle control --
@pytest.mark.parametrize("n", [12, 16, 40])
def test_cycles_read_girth_over_four(n):
    """For C_n with 4 | n the optimal violation is attained by anchoring at one
    node and placing x, y, z at arc offsets n/4, 3n/4, n/2, which reads exactly
    n/4 (worked in the scale/hyperbolic.py docstring); TOLERANCE: all pairwise
    distances are integers <= n/2, every Gromov product is a multiple of 1/2,
    and binary floating point represents those exactly, so the only error is
    representation noise around 2^-53-scale ulps -- 1e-9 is six orders of
    magnitude above it and still far below any structural deviation."""
    mod = _require_hyperbolic()
    dist = mod.hop_distances(n, _cycle(n))
    got = mod.gromov_delta(dist)
    assert abs(got - n / 4.0) <= 1e-9, (n, got)


def test_odd_cycle_sanity_only():
    """C_13 has no antipodal quarter-mark, so the exact optimum depends on floor
    effects the module docstring documents; assert only the sanity envelope
    0 < delta_hat <= diam = floor(13/2)."""
    mod = _require_hyperbolic()
    n = 13
    dist = mod.hop_distances(n, _cycle(n))
    got = mod.gromov_delta(dist)
    assert 0.0 < got <= n / 2.0, got


# -------------------------------------------------------- 4. complete graph --
@pytest.mark.parametrize("n", [8, 25])
def test_complete_graph_reads_exactly_zero_derived_bound(n):
    """Derived expectation: in K_n every off-diagonal distance is 1, so EVERY
    matching sum sigma = d(..)+d(..) over any quadruple equals exactly 2, the
    top-two gap is 0, and delta_hat = 0 EXACTLY -- the sharpest bound consistent
    with its diameter-<=1 geometry (an equidistant space is a scaled simplex,
    whose products are all 1/2 and violate nothing)."""
    mod = _require_hyperbolic()
    dist = mod.hop_distances(n, _complete(n))
    assert mod.gromov_delta(dist) == 0.0


# ------------------------------------------- 5. the E4' motif, printed, sane --
def test_e4prime_two_cliques_bridge_motif_printed():
    """The E4' shape at toy scale: K_6 and K_7 joined by ONE bridge edge.
    PRINTED for the journal, asserted only against sanity bounds: 0 <= d <=
    diameter = 2. No equality claim: whether a tree-of-blocks motif reads 0
    (block graphs are ptolemaic, hence 0-hyperbolic) is a MEASURED statement
    about the motif, not an assumption to bake into a test."""
    mod = _require_hyperbolic()
    edges, n = _two_cliques_bridge(6, 7)
    dist = mod.hop_distances(n, edges)
    got = mod.gromov_delta(dist)
    print(f"E4' motif K6-bridge-K7 (n={n}): delta_hat={got!r}")
    assert 0.0 <= got <= 2.0, got


# --------------------------------------------------------- 6. sampled mode --
def test_sampled_mode_never_exceeds_exact_and_holds_controls():
    """Sampled mode maximises over a subset of anchors/bases, so it is a MONOTONE
    LOWER BOUND on the exact value (semantics documented in the module). On the
    controls it must reproduce them: exactly 0 on a tree, strictly inside
    (0, exact] on a cycle with the default budget, and deterministic under a
    seeded rng."""
    mod = _require_hyperbolic()
    n = 40
    rng = np.random.default_rng(20260826)
    tree = mod.hop_distances(n, _random_tree(n, 16))
    cyc = mod.hop_distances(n, _cycle(n))
    s_tree = mod.gromov_delta(tree, sampled=True, rng=np.random.default_rng(7))
    s_cyc_a = mod.gromov_delta(cyc, sampled=True, rng=np.random.default_rng(7))
    s_cyc_b = mod.gromov_delta(cyc, sampled=True, rng=np.random.default_rng(7))
    e_cyc = mod.gromov_delta(cyc)
    assert s_tree == 0.0, s_tree
    assert 0.0 < s_cyc_a <= e_cyc + 1e-9, (s_cyc_a, e_cyc)
    assert s_cyc_a == s_cyc_b, "sampled mode is not reproducible under a seed"

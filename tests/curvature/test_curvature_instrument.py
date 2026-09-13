"""RED-FIRST tests for ceqjepa/curvature.py (PHASE C it.1, Ollivier-Ricci).

WHAT THESE TESTS ARE FOR. The instrument under test claims to read the geometry
of a context graph: dense/coherent regions positive, bridges between phases
negative. Every one of the five tests below is a way for that claim to be FALSE
while the code still returns numbers:

  1. W1 IS NOT TRANSPORT. An earlier round of this project shipped a W1 proxy and
     it was graded F3. A proxy returns a plausible number on every input, so no
     amount of downstream plausibility catches it. The only thing that catches it
     is a case where the answer is known independently: a closed form, and an
     EXHAUSTIVE enumeration of the vertices of the transportation polytope,
     implemented HERE, in the test, with no call into the module under test.
  2. THE PLANT DOES NOT FIRE. A bridge that does not read negative.
  3. THE PLANT FIRES ON NOTHING. A single isotropic cloud -- one phase, no bridge
     -- reading bridges anyway. That is the instrument measuring k, not geometry,
     and it is the negative control that makes test 2 mean anything.
  4. THE SIGN IS A FUNCTION OF k. If "bridges are negative" only survives at one k,
     the claim is about the neighbour count and not about the data.
  5. A REFUSAL IS SWALLOWED. A disconnected pair, an isolated node and a duplicate
     point each have NO curvature; returning NaN, 0.0, or raising into a caller's
     except clause all lose the distinction between "no answer" and "the answer".

THE IMPORT IS GUARDED ON PURPOSE. A module-level `import ceqjepa.curvature` turns
the RED phase into a single collection error with no test names in it. Each test
must be seen to FAIL BY NAME before the module exists, so the import failure is
carried into each test body instead.

RUN: python -m pytest tests/curvature/test_curvature_instrument.py -v
"""

import itertools
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

try:
    from ceqjepa import curvature as cv
except Exception as _exc:                      # noqa: BLE001 -- RED phase carries it
    cv = None
    _IMPORT_ERROR = _exc


def _cv():
    """The module under test, or a named failure saying it is not there yet."""
    if cv is None:
        raise AssertionError(
            "ceqjepa/curvature.py did not import: %r" % (_IMPORT_ERROR,))
    return cv


# ---------------------------------------------------------------------------
# Independent oracles. Nothing below this line calls the module under test.
# ---------------------------------------------------------------------------

def brute_force_w1(mu, nu, C):
    """EXACT W1 by exhaustive enumeration of the transportation polytope's vertices.

    A basic feasible solution of an m x n transportation problem has at most
    m + n - 1 nonzero cells, and the cells it uses form a spanning forest of the
    complete bipartite graph on (rows, columns). So: enumerate every subset of
    m + n - 1 cells, keep those whose bipartite graph is a SPANNING TREE (connected,
    acyclic, touching all m + n nodes), solve the marginal equations on it (the
    system has a unique solution there), keep the solution if it is non-negative.
    The minimum cost over those vertices is W1, by the fundamental theorem of
    linear programming. No solver, no tolerance on optimality -- a finite min.

    Combinatorial in the support size and therefore only usable at the 3x3 the
    owner's spec asks it for, where the subset count is small enough to enumerate
    exhaustively. A basis has one fewer cell than the marginals have rows.
    """
    mu, nu, C = np.asarray(mu, float), np.asarray(nu, float), np.asarray(C, float)
    m, n = C.shape
    cells = [(i, j) for i in range(m) for j in range(n)]
    best = math.inf
    for basis in itertools.combinations(range(len(cells)), m + n - 1):
        chosen = [cells[b] for b in basis]
        # spanning tree test on the bipartite graph: rows 0..m-1, cols m..m+n-1
        parent = list(range(m + n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        acyclic = True
        for (i, j) in chosen:
            ri, rj = find(i), find(m + j)
            if ri == rj:
                acyclic = False
                break
            parent[ri] = rj
        if not acyclic:
            continue
        if len({find(a) for a in range(m + n)}) != 1:
            continue                                   # not spanning
        # solve the marginals restricted to the basis cells
        A = np.zeros((m + n, len(chosen)))
        for c, (i, j) in enumerate(chosen):
            A[i, c] = 1.0
            A[m + j, c] = 1.0
        b = np.concatenate([mu, nu])
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        if np.abs(A @ x - b).max() > 1e-12:
            continue                                   # infeasible on this basis
        if x.min() < -1e-12:
            continue                                   # not a vertex of the polytope
        cost = sum(xc * C[i, j] for xc, (i, j) in zip(x, chosen))
        best = min(best, cost)
    return best


def mean_ci(vals, z=1.96):
    """(mean, ci_lo, ci_hi, n). Normal-approximation CI on the MEAN, n reported."""
    v = np.asarray(vals, float)
    n = v.size
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    se = v.std(ddof=1) / math.sqrt(n) if n > 1 else 0.0
    return (float(v.mean()), float(v.mean() - z * se), float(v.mean() + z * se), int(n))


# ---------------------------------------------------------------------------
# 1. EXACTNESS OF W1
# ---------------------------------------------------------------------------

def test_w1_matches_closed_form():
    """Two point masses at distance d give W1 = d; a uniform block shifted by s on
    a line gives W1 = s (the monotone coupling is optimal in 1-D). Equality to
    1e-12, not 'close': an entropic or proxy W1 misses these by percent, not by eps.
    """
    m = _cv()

    for d in (1.0, 2.5, 7.25, 1e-3):
        got = m.w1_exact([1.0], [1.0], [[d]])
        assert abs(got - d) < 1e-12, "two point masses at %g read W1=%.17g" % (d, got)

    for npts, shift in ((4, 1), (4, 3), (6, 2), (8, 5)):
        xs = np.arange(npts, dtype=float)
        ys = xs + shift
        C = np.abs(xs[:, None] - ys[None, :])
        mu = np.full(npts, 1.0 / npts)
        got = m.w1_exact(mu, mu, C)
        assert abs(got - shift) < 1e-12, \
            "uniform block shifted by %d read W1=%.17g" % (shift, got)


def test_w1_matches_brute_force_vertex_enumeration():
    """3x3, exhaustive over the vertices of the transportation polytope.

    The brute force lives in this file and shares no code with the module. If the
    module's W1 is a solver's optimum, the two agree to machine precision on every
    instance; if it is a proxy, it disagrees on the first one.
    """
    m = _cv()
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(25):
        C = rng.random((3, 3)) * 5.0
        mu = rng.random(3)
        mu /= mu.sum()
        nu = rng.random(3)
        nu /= nu.sum()
        got = m.w1_exact(mu, nu, C)
        ref = brute_force_w1(mu, nu, C)
        worst = max(worst, abs(got - ref))
        assert abs(got - ref) < 1e-12, \
            "LP %.17g vs exhaustive vertex min %.17g (diff %.3e)" % (got, ref, abs(got - ref))
    assert worst < 1e-12


def test_w1_is_not_a_proxy_in_the_source():
    """The F3 lesson, encoded so it cannot be re-shipped quietly.

    A regularised or proxy W1 is not detectable from its return value alone once
    the temperature is small, so the source is checked for the machinery as well
    as the answer. Not a substitute for the two tests above -- an addition.
    """
    m = _cv()
    src = Path(m.__file__).read_text(encoding="utf-8")
    body = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    for banned in ("sinkhorn", "entropic", "logsumexp", "reg=", "epsilon_scaling"):
        assert banned not in body.lower(), \
            "banned approximation machinery %r appears in %s" % (banned, m.__file__)
    assert re.search(r"linprog|linear_sum_assignment|min_cost_flow|network_simplex",
                     body), "no transport solver is called: where does W1 come from?"


# ---------------------------------------------------------------------------
# 2. T-BRIDGE MUST-FIRE
# ---------------------------------------------------------------------------

def test_bridge_reads_negative_and_intra_reads_positive():
    """Both planted beds. Bridge edges negative, intra edges positive, each group
    with a CI on its mean and its EDGE COUNT beside it.

    Two beds, because they answer different questions. The GRAPH bed (two dense
    blocks joined by one cross edge) is the spec's planted two-cluster graph and
    carries a genuine cut edge. The POINT-CLOUD bed is the kNN construction PHASE
    C actually uses, where the narrowest possible cut is k edges wide around an
    isthmus VERTEX. An instrument that fires on the first and not the second is
    not usable on embeddings, so both are asserted here.

    The owner's lineage numbers (intra +0.072 +/- 0.205, bridge -0.732) are
    orientation only and are deliberately NOT hardcoded: this asserts the SIGNS
    and the separation and prints what these beds actually measure.
    """
    m = _cv()

    W, labels = m.two_block_bed()
    resg = m.curvature_from_graph(W)
    gi, gb = m.split_by_plant(resg, labels)
    gi_mean, gi_lo, gi_hi, gi_n = mean_ci(gi)
    gb_mean, gb_lo, gb_hi, gb_n = mean_ci(gb)
    print("\n  GRAPH bed")
    print("    intra  mean %+.4f  CI [%+.4f, %+.4f]  edges %d"
          % (gi_mean, gi_lo, gi_hi, gi_n))
    print("    bridge mean %+.4f  CI [%+.4f, %+.4f]  edges %d"
          % (gb_mean, gb_lo, gb_hi, gb_n))
    print("    refusals %r  total edges %d" % (resg["refusals"], resg["n_edges"]))
    assert gi_n > 20 and gb_n >= 1, "graph bed degenerate: %d / %d" % (gi_n, gb_n)
    assert gi_lo > 0.0, "graph intra is not positive: CI [%+.4f, %+.4f]" % (gi_lo, gi_hi)
    assert gb_hi < 0.0, "graph bridge is not negative: CI [%+.4f, %+.4f]" % (gb_lo, gb_hi)

    X, plabels = m.two_cluster_bed()
    res = m.curvature_from_points(X)
    intra, bridge = m.split_by_plant(res, plabels)
    i_mean, i_lo, i_hi, i_n = mean_ci(intra)
    b_mean, b_lo, b_hi, b_n = mean_ci(bridge)
    print("  POINT-CLOUD bed, k = %d" % m.K_NEIGHBORS)
    print("    intra  mean %+.4f  CI [%+.4f, %+.4f]  edges %d" % (i_mean, i_lo, i_hi, i_n))
    print("    bridge mean %+.4f  CI [%+.4f, %+.4f]  edges %d" % (b_mean, b_lo, b_hi, b_n))
    print("    refusals %r  total edges %d" % (res["refusals"], res["n_edges"]))
    assert i_n > 20 and b_n >= 3, "point bed degenerate: %d / %d" % (i_n, b_n)
    assert i_lo > 0.0, "point intra is not positive: CI [%+.4f, %+.4f]" % (i_lo, i_hi)
    assert b_hi < 0.0, "point bridge is not negative: CI [%+.4f, %+.4f]" % (b_lo, b_hi)
    assert b_mean < i_mean, "bridge is not below intra"


# ---------------------------------------------------------------------------
# 3. PLANTED NEGATIVE
# ---------------------------------------------------------------------------

def test_one_phase_cloud_reads_no_bridges():
    """A single isotropic Gaussian cloud has no bridge. If it reads one, the
    instrument is measuring k and not the geometry.

    The threshold is generated FROM THE SAME CLOUD FAMILY: pool the edge curvatures
    of replicate clouds (fresh seeds, same n, dim, sigma) and take a low quantile.

    The graph bed gets its own planted negative, which is the sharper one: a
    SINGLE block of the same size and density, bisected by index. Same
    label-crossing definition of a bridge edge, no bottleneck behind it. If that
    reads negative, the instrument is measuring the labelling.

    And the LIMIT is asserted, not hoped: on the point cloud the null tail reaches
    below the least negative planted bridge edge, so the point-cloud read is a
    group statement and a single negative edge is not a bridge. On the graph bed
    the same overlap is zero. A change in either direction fails here.
    """
    m = _cv()
    null = m.one_phase_null(n_replicates=4)
    thresh = float(np.quantile(null, 0.01))

    X = m.one_phase_bed(seed=1000)
    res = m.curvature_from_points(X)
    kap = np.asarray(res["kappa"], float)
    frac_below = float((kap < thresh).mean())
    print("\n  one-phase cloud: %d edges, min %+.4f; null %d edges, 1%% point %+.4f, "
          "fraction below %.4f" % (kap.size, kap.min(), null.size, thresh, frac_below))
    assert kap.size > 50, "one-phase bed has too few edges to say anything"
    assert frac_below <= 0.05, \
        "one-phase cloud puts %.1f%% of its edges below its own 1%% null" % (100 * frac_below)

    gnull = m.one_block_null(n_replicates=2)
    W, glabels = m.two_block_bed()
    _, gbridge = m.split_by_plant(m.curvature_from_graph(W), glabels)
    g_mean, g_lo, g_hi, g_n = mean_ci(gnull)
    print("  graph null (one block, bisected by index): mean %+.4f CI [%+.4f, %+.4f] "
          "min %+.4f edges %d" % (g_mean, g_lo, g_hi, gnull.min(), g_n))
    print("  graph planted bridge max %+.4f over %d edges" % (max(gbridge), len(gbridge)))
    assert g_lo > 0.0, "an arbitrary bisection of one block reads negative"
    assert max(gbridge) < gnull.min(), \
        ("the graph null reaches %+.4f, at or below the planted bridge %+.4f"
         % (gnull.min(), max(gbridge)))

    Xp, plabels = m.two_cluster_bed()
    _, pbridge = m.split_by_plant(m.curvature_from_points(Xp), plabels)
    overlap = float(np.mean([(null < v).mean() for v in pbridge]))
    g_overlap = float(np.mean([(gnull < v).mean() for v in gbridge]))
    print("  per-edge overlap with the null: point cloud %.4f, graph %.4f"
          % (overlap, g_overlap))
    assert max(pbridge) > null.min(), \
        ("the point-cloud null tail no longer reaches past the plant: the recorded "
         "group-only limit has stopped reproducing and must be re-measured")
    assert overlap < 0.10, "point-cloud per-edge overlap rose to %.4f" % overlap
    assert g_overlap == 0.0, "the graph bed lost its per-edge separation"


# ---------------------------------------------------------------------------
# 4. K-GAMING CONTROL
# ---------------------------------------------------------------------------

def test_sign_separation_survives_a_k_sweep():
    """Sweep k. The claim 'bridges are negative' is admissible only over the k
    range where it actually holds, and the range is reported whether or not it is
    all of the swept values.
    """
    m = _cv()
    ks = m.K_SWEEP
    assert len(ks) >= 4, "the sweep must cover at least 4 values of k"

    held = []
    print()
    for k in ks:
        X, labels = m.two_cluster_bed()
        res = m.curvature_from_points(X, k=k)
        intra, bridge = m.split_by_plant(res, labels)
        i_mean, i_lo, i_hi, i_n = mean_ci(intra)
        b_mean, b_lo, b_hi, b_n = mean_ci(bridge)
        ok = (i_n > 0 and b_n > 0 and i_lo > 0.0 and b_hi < 0.0)
        held.append(ok)
        print("  k=%-3d intra %+.4f [%+.4f,%+.4f] n=%-4d | bridge %+.4f [%+.4f,%+.4f] "
              "n=%-3d | separated=%s" % (k, i_mean, i_lo, i_hi, i_n,
                                         b_mean, b_lo, b_hi, b_n, ok))

    holding = [k for k, ok in zip(ks, held) if ok]
    print("  separation holds at k in %r out of %r" % (holding, list(ks)))
    assert m.K_NEIGHBORS in holding, \
        "the PINNED k=%d does not separate: holding range %r" % (m.K_NEIGHBORS, holding)
    assert len(holding) >= max(2, len(ks) // 2), \
        "separation survives only %d of %d swept k values: %r" % (len(holding), len(ks), holding)
    assert ks[0] not in holding, \
        ("the smallest swept k = %d now separates too: when measured, the isthmus "
         "carried %d edges there and the CI spanned zero, so the recorded k floor "
         "must be re-measured" % (ks[0], ks[0]))

    # The cut-width sweep is the knob that exists on the GRAPH bed, and it is the
    # other half of the same question: a bottleneck stops being one once it is
    # wide enough, and the read must track that rather than the label.
    print("  graph bed, cut-width sweep:")
    for n_cross in m.CUT_SWEEP:
        Wc, lc = m.two_block_bed(n_cross=n_cross)
        _, bc = m.split_by_plant(m.curvature_from_graph(Wc), lc)
        c_mean, c_lo, c_hi, c_n = mean_ci(bc)
        print("    cross edges %-2d bridge %+.4f CI [%+.4f, %+.4f] edges %d"
              % (n_cross, c_mean, c_lo, c_hi, c_n))
        assert c_hi < 0.0, "the bridge stopped reading negative at cut width %d" % n_cross


# ---------------------------------------------------------------------------
# 5. REFUSAL IS A VALUE
# ---------------------------------------------------------------------------

def test_refusals_are_values_with_distinct_reasons():
    """Disconnected pair, isolated node, duplicate point (d = 0), self-loop.

    Each returns a refusal carrying a reason; no exception, no NaN, and the four
    reasons are DISTINCT -- a single "bad input" reason would make the refusal
    unactionable, which is the same defect as swallowing it.
    """
    m = _cv()
    reasons = {}

    # (a) two disjoint triangles: 0-1-2 and 3-4-5. d(0,3) = inf.
    W = np.zeros((6, 6))
    for a, b in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]:
        W[a, b] = W[b, a] = 1.0
    D = m.graph_metric(W)
    r = m.kappa_edge(W, D, 0, 3)
    assert m.is_refusal(r), "a disconnected pair returned a number: %r" % (r,)
    reasons["disconnected"] = r.reason

    # (b) an isolated node: node 3 has no neighbours at all.
    W2 = np.zeros((4, 4))
    for a, b in [(0, 1), (1, 2), (0, 2)]:
        W2[a, b] = W2[b, a] = 1.0
    D2 = m.graph_metric(W2)
    r2 = m.kappa_edge(W2, D2, 0, 3)
    assert m.is_refusal(r2), "an isolated node returned a number: %r" % (r2,)
    reasons["isolated"] = r2.reason

    # (c) duplicate point: an edge of length 0 in the metric.
    W3 = np.array([[0.0, 0.0, 1.0],
                   [0.0, 0.0, 1.0],
                   [1.0, 1.0, 0.0]])
    D3 = m.graph_metric(W3)
    D3 = D3.copy()
    D3[0, 1] = D3[1, 0] = 0.0                     # coincident embeddings
    r3 = m.kappa_edge(W3, D3, 0, 1)
    assert m.is_refusal(r3), "a zero-distance pair returned a number: %r" % (r3,)
    reasons["duplicate"] = r3.reason

    # (d) self-loop.
    r4 = m.kappa_edge(W, D, 1, 1)
    assert m.is_refusal(r4), "a self-loop returned a number: %r" % (r4,)
    reasons["self_loop"] = r4.reason

    print("\n  " + "\n  ".join("%-12s -> %s" % (k, v) for k, v in reasons.items()))
    assert len(set(reasons.values())) == 4, \
        "refusal reasons are not distinct: %r" % (reasons,)
    for name, why in reasons.items():
        assert isinstance(why, str) and len(why) > 10, "%s has no usable reason" % name
        assert "nan" not in why.lower()

    # and a refusal must never be silently truthy-as-a-number
    for r_ in (r, r2, r3, r4):
        assert not isinstance(r_, float), "refusal is a float: %r" % (r_,)


# ---------------------------------------------------------------------------
# HOUSE INVARIANTS: pinned constants, edge counts beside every number
# ---------------------------------------------------------------------------

def test_k_and_alpha_are_pinned_and_stated_in_the_docstring():
    m = _cv()
    doc = (m.__doc__ or "")
    assert isinstance(m.K_NEIGHBORS, int) and m.K_NEIGHBORS > 0
    assert isinstance(m.ALPHA, float) and 0.0 < m.ALPHA < 1.0
    assert str(m.K_NEIGHBORS) in doc, "the pinned k is not stated in the module docstring"
    assert str(m.ALPHA) in doc, "the pinned alpha is not stated in the module docstring"


def test_every_curvature_number_carries_its_edge_count():
    m = _cv()
    X, _ = m.two_cluster_bed()
    res = m.curvature_from_points(X)
    assert res["n_edges"] == len(res["kappa"]) == len(res["edges"])
    st = m.group_stats(res["kappa"])
    for field in ("mean", "ci_lo", "ci_hi", "n_edges"):
        assert field in st, "group_stats drops %s" % field
    assert st["n_edges"] == res["n_edges"]
    assert m.group_stats([])["n_edges"] == 0


# ---------------------------------------------------------------------------
# 6. THE PINNED METRIC IS PINNED FOR A MEASURED REASON
# ---------------------------------------------------------------------------

def test_euclid_metric_is_worse_than_the_pinned_hop_metric():
    """"hop is pinned" is a claim, and until this test it was asserted nowhere.

    demo() printed the euclid numbers beside the hop numbers and nothing checked
    them, so a silent change to the euclid control failed no test -- exactly the
    shape of drift the docstring test below is for.

    The claim, in the only form that survives being written down: under euclid
    weighting the negative control gets DEEPER while the planted signal gets
    WEAKER, so the per-edge overlap between them is strictly larger. All three
    are asserted, because the first two alone could both move the same way and
    leave discriminability unchanged.
    """
    m = _cv()
    ctl = m.metric_control()
    hop, euc = ctl["hop"], ctl["euclid"]
    for name, c in (("hop", hop), ("euclid", euc)):
        print("\n  %-7s intra %+.4f [%+.4f, %+.4f] n=%d | bridge %+.4f [%+.4f, %+.4f] "
              "n=%d | null min %+.4f n=%d | overlap %.4f"
              % (name, c["intra"]["mean"], c["intra"]["ci_lo"], c["intra"]["ci_hi"],
                 c["intra"]["n_edges"], c["bridge"]["mean"], c["bridge"]["ci_lo"],
                 c["bridge"]["ci_hi"], c["bridge"]["n_edges"], c["null"]["min"],
                 c["null"]["n_edges"], c["overlap"]))

    assert euc["null"]["min"] < hop["null"]["min"], \
        ("the euclid negative control no longer has the deeper tail: euclid %+.4f "
         "vs hop %+.4f" % (euc["null"]["min"], hop["null"]["min"]))
    assert euc["bridge"]["mean"] > hop["bridge"]["mean"], \
        ("the euclid planted signal is no longer the weaker one: euclid %+.4f vs "
         "hop %+.4f" % (euc["bridge"]["mean"], hop["bridge"]["mean"]))
    assert euc["overlap"] > hop["overlap"], \
        ("euclid no longer overlaps its null more than hop does: euclid %.4f vs "
         "hop %.4f -- the pinning of the hop metric has lost its measured reason"
         % (euc["overlap"], hop["overlap"]))


# ---------------------------------------------------------------------------
# 7. NO NUMBER IN A DOCSTRING THAT NO RUN PRINTS
# ---------------------------------------------------------------------------

#: ANY decimal, and NO exemption list. The narrow earlier version of this pattern
#: required a sign, a percent, an "x" or an exponent, so a bare decimal in prose
#: walked straight through it -- which is exactly the form all three of Cameron's
#: struck numbers took, including a hardcoded timing printed inside a line
#: labelled RUN. An exemption list is how a check like this gets defanged, so a
#: legitimate constant that fails here (a DOI, a year, a pinned parameter) is
#: fixed by PRINTING it in the run, never by excusing it.
_MEASURED = re.compile(
    r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?"        # 40,317 and 40,317.5
    r"|[+-]?\d+(?:\.\d+)?[eE][+-]?\d+"           # 1e-12, 2.220e-16, 0.000e+00
    r"|[+-]?\d+\.\d+"                            # 0.1363
    r"|[+-]?\d+"                                  # 590
)


def _norm(tok):
    """Canonical form of a number token: drop a leading + and any thousands commas.

    So a docstring may write a bare decimal where the run prints a signed one,
    or a grouped integer where the run prints an ungrouped one, without either
    being a false alarm. A MINUS sign is never dropped: a negative and a positive
    of the same magnitude are different numbers, and the guard must say so.
    """
    return tok.lstrip("+").replace(",", "")


def _printed_missing(docs, out):
    """Which docstring numbers do NOT appear in the run output.

    This is the whole guard, factored out so it can be attacked directly by
    test_the_docstring_guard_is_not_vacuous below. A guard that is only ever
    exercised on the real file cannot be shown to reject anything.
    """
    printed = {_norm(tok) for tok in _MEASURED.findall(out)}
    return [(where, tok) for where, doc in docs
            for tok in _MEASURED.findall(doc) if _norm(tok) not in printed]


def _own_docstrings():
    """Every docstring in THIS FILE, by the same rule the module gets.

    A measured number written into a test docstring is exactly as unbound as one
    written into the module's, and until this was added a plant here was caught
    by nothing. Anything in this file that is not a module measurement is
    reworded to drop the numeral rather than exempted: a numeral nobody is
    prepared to bind should not be written down.
    """
    mod = sys.modules[__name__]
    docs = [("test module", mod.__doc__ or "")]
    for name, obj in sorted(vars(mod).items()):
        if getattr(obj, "__module__", None) != __name__:
            continue
        doc = getattr(obj, "__doc__", None)
        if isinstance(doc, str) and doc.strip():
            docs.append(("test:" + name, doc))
    return docs


def _module_docstrings(m):
    """(name, docstring) for the module and everything DEFINED in it.

    Filtered on __module__ so numpy's and scipy's docstrings do not come along;
    not filtered on __all__, because demo() is public, is the thing that prints
    every number, and is absent from __all__ -- so an __all__-scoped scan misses
    precisely the function whose output the rest of the check is against.
    """
    docs = [("module", m.__doc__ or "")]
    for name, obj in sorted(vars(m).items()):
        if getattr(obj, "__module__", None) != m.__name__:
            continue
        doc = getattr(obj, "__doc__", None)
        if isinstance(doc, str) and doc.strip():
            docs.append((name, doc))
    return docs


def test_the_docstring_guard_is_not_vacuous():
    """Attack the guard directly, on synthetic input, rather than trusting that
    a green run on the real file means it can reject anything.

    Four ways a number can fail to be bound, and the guard must catch all four:

      COLLISION   a token that is a SUBSTRING of a printed number: drop the
                  leading digit of the printed DOI and the remainder is printed
                  by nothing, yet a bare `tok in out` finds it inside the DOI and
                  passes it. The plants live in the test body, never in this
                  docstring, precisely because the run must NOT print them. This is
                  the one that was struck: substring matching makes the guard
                  vacuous against exactly the numbers most likely to be wrong,
                  the ones that look almost right.
      ABSENT      a number nowhere in the run at all.
      BARE INT    an integer, with or without a thousands separator. Worthless
                  under substring matching (every short integer is inside some
                  longer number) and live under token matching, which is why the
                  order of these two repairs is not free.
      PRESENT     the control: a number the run does print must PASS, or the
                  guard is merely noisy rather than strict.
    """
    out = ("DOI 10.1016/j.jfa.2008.11.001 elapsed 83.42 s intra +0.1363 "
           "worst 2.220e-16 rows 40,317 edges 590")
    for label, doc in (("collision", "the value 0.1016"),
                       ("collision", "the value 40.31"),
                       ("absent", "the value 77.31"),
                       ("absent int", "some 4031 rows"),
                       ("sign flip", "the value -0.1363")):
        assert _printed_missing([("plant", doc)], out), \
            "%s: %r passed a guard that only the run's own numbers should pass" % (label, doc)

    for label, doc in (("printed decimal", "elapsed 83.42 s"),
                       ("printed signed", "intra +0.1363 and 0.1363"),
                       ("printed exponent", "worst 2.220e-16"),
                       ("printed doi", "DOI 10.1016/j.jfa.2008.11.001"),
                       ("printed grouped", "rows 40,317 and 40317"),
                       ("printed integer", "edges 590")):
        assert not _printed_missing([("plant", doc)], out), \
            "%s: %r is in the run but the guard flagged it" % (label, doc)


def test_every_measured_number_in_a_docstring_is_printed_by_the_demo(capsys):
    """The check whose absence let six reported numbers come from no run.

    Every decimal in the module docstring, and in the docstring of everything
    defined in this module, must appear verbatim in the output of `python -m
    ceqjepa.curvature`. A number that no run prints cannot be checked by anyone,
    including its author, and drifts silently from the moment the bed changes.
    """
    m = _cv()
    m.demo()
    out = capsys.readouterr().out

    docs = _module_docstrings(m)
    own = _own_docstrings()
    missing = _printed_missing(docs + own, out)
    checked = sum(len(_MEASURED.findall(doc)) for _, doc in docs)
    checked_own = sum(len(_MEASURED.findall(doc)) for _, doc in own)
    print("\n  %d numbers across %d module docstrings, %d across %d test "
          "docstrings, %d missing from the run"
          % (checked, len(docs), checked_own, len(own), len(missing)))
    assert checked >= 60, "only %d numbers found: the regex is not biting" % checked
    assert not missing, ("these docstring numbers are printed by no run:\n    "
                         + "\n    ".join("%s: %s" % (w, t) for w, t in missing))

    # THE COVERAGE FIGURE IS ITSELF A NUMBER IN A REPORT, so it is bound the same
    # way every other number here is: the module counts its own docstrings, the
    # demo prints that count, and this test -- which counted them independently,
    # above -- requires the two to agree. A coverage figure quoted from a run that
    # did not print it is the exact drift class this whole test exists to stop,
    # and it caught me: "19 docstrings" came from the RED run, before demo() had
    # a docstring of its own, while the shipped run covers 20.
    n_docs, decimals = m.docstring_numbers()
    assert n_docs == len(docs), \
        ("the module counts %d docstrings and this test counts %d: the two scans "
         "disagree, so neither figure can be quoted" % (n_docs, len(docs)))
    assert len(decimals) == checked, \
        ("the module counts %d numbers and this test counts %d"
         % (len(decimals), checked))
    assert "%d docstrings" % n_docs in out, \
        "the run does not print its own docstring count, so the figure in any " \
        "report of it is unbound"
    assert "%d numbers" % len(decimals) in out, \
        "the run does not print its own number count"


# ---------------------------------------------------------------------------
# 8. A REFUSED EDGE REACHES A CONSUMER AS NaN, AND THE INDEX DOES NOT SHIFT
# ---------------------------------------------------------------------------

def test_refused_edges_arrive_as_nan_and_the_edge_index_is_stable():
    """A refusal scored as 0.0 curvature is a silent wrong answer.

    A residual ||kappa - kappa_target|| reads a refused edge substituted by 0.0
    as PERFECT AGREEMENT wherever the target is also near zero, which is the one
    place a refusal must never be allowed to land. And dropping the refused edge
    instead is no better for a consumer differencing two blocks: it shifts every
    index after it, so the residual subtracts mismatched edges.

    So: `all_edges` carries every edge whether or not it refused, `kappa_full`
    carries NaN at the refused positions, `edge_index` maps an edge to its
    position, and an aggregate over kappa_full comes out NaN rather than
    averaging around the hole.
    """
    m = _cv()
    W = np.zeros((4, 4))
    for a, b in [(0, 1), (1, 2), (0, 2), (2, 3)]:
        W[a, b] = W[b, a] = 1.0

    clean = m.curvature_from_graph(W)
    D = m.graph_metric(W)
    D[0, 1] = D[1, 0] = 0.0                     # duplicate embeddings on a real edge
    res = m.curvature_from_graph(W, D=D)

    assert res["n_refused"] == 1, "the planted refusal did not fire: %r" % (res["refusals"],)
    assert list(res["all_edges"]) == list(clean["all_edges"]), \
        "the edge index shifted when an edge refused: a consumer differencing two " \
        "blocks would subtract mismatched edges"
    assert len(res["all_edges"]) == 4, "all_edges dropped the refused edge"

    kf = np.asarray(res["kappa_full"], dtype=float)
    pos = res["edge_index"][(0, 1)]
    print("\n  kappa_full = %r, refused position %d, reason %r"
          % (list(np.round(kf, 4)), pos, res["refusal_at"][pos].code))
    assert np.isnan(kf[pos]), \
        "a refused edge arrived as %r rather than NaN" % (kf[pos],)
    assert kf[pos] != 0.0
    assert not np.isnan(np.delete(kf, pos)).any(), "an answerable edge came back NaN"

    st = m.group_stats(kf)
    assert np.isnan(st["mean"]), "the aggregate averaged around a refusal: %r" % (st,)
    assert st["n_edges"] == 4, "the aggregate silently dropped the refused edge"

    for (i, j), value in zip(res["edges"], res["kappa"]):
        assert kf[res["edge_index"][(i, j)]] == value, \
            "kappa_full and the numeric-only lists disagree at edge (%d,%d)" % (i, j)


# ---------------------------------------------------------------------------
# 9. THE SECOND MEASURE, and the guarantee that adding it moved nothing
# ---------------------------------------------------------------------------

def _triangles_with_a_bridge():
    """Two triangles {0,1,2} and {3,4,5} joined by the edge (2,3)."""
    W = np.zeros((6, 6))
    for a, b in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]:
        W[a, b] = W[b, a] = 1.0
    return W


def test_measures_agree_on_an_unweighted_graph():
    """Uniform weights make the two definitions the SAME definition.

    (1 - alpha) * w_xz / sum_z w_xz collapses to (1 - alpha) / deg(x) the moment
    every w is equal, so disagreement here is an implementation error and not a
    modelling choice. This is the test that says the new argument is wired to the
    weights rather than to something else that happens to co-vary with them.
    """
    m = _cv()
    W = _triangles_with_a_bridge()
    D = m.graph_metric(W)
    worst = 0.0
    for (i, j) in [(2, 3), (0, 1), (1, 2), (4, 5)]:
        u = m.kappa_edge(W, D, i, j, measure="uniform")
        p = m.kappa_edge(W, D, i, j, measure="weight-proportional")
        worst = max(worst, abs(u - p))
        assert abs(u - p) < 1e-12, \
            "edge (%d,%d): uniform %.17g vs weight-proportional %.17g" % (i, j, u, p)
    print("\n  unweighted graph, worst |uniform - weight-proportional| = %.3e" % worst)

    X, _ = m.two_cluster_bed()
    a = m.curvature_from_points(X)
    b = m.curvature_from_points(X, measure="weight-proportional")
    assert a["all_edges"] == b["all_edges"]
    assert np.array_equal(np.asarray(a["kappa"]), np.asarray(b["kappa"])), \
        "the pinned bed is unweighted, so the two measures must read it identically"
    print("  pinned bed (%d edges): the two measures agree BITWISE" % a["n_edges"])


def test_measures_disagree_on_a_weighted_graph():
    """If they do not disagree where the weights differ, the argument does nothing."""
    m = _cv()
    W = _triangles_with_a_bridge()
    W[0, 1] = W[1, 0] = 4.0
    W[2, 3] = W[3, 2] = 3.0
    D = m.graph_metric(W)

    gaps = {}
    for (i, j) in [(2, 3), (1, 2), (0, 2)]:
        u = m.kappa_edge(W, D, i, j, measure="uniform")
        p = m.kappa_edge(W, D, i, j, measure="weight-proportional")
        gaps[(i, j)] = abs(u - p)
        print("\n  edge (%d,%d): uniform %+.6f  weight-proportional %+.6f  gap %.6f"
              % (i, j, u, p, abs(u - p)))
    assert max(gaps.values()) > 1e-3, \
        "the two measures differ by at most %.3e on a weighted graph: the new " \
        "argument is inert" % max(gaps.values())


def test_heavy_edge_moves_only_the_weight_proportional_reading():
    """The planted negative for the second measure, with the METRIC held fixed.

    Raising one edge's weight changes two things at once in general -- the mass
    split AND the shortest-path metric -- so this holds D at the unweighted
    metric and varies only W. Then the uniform reading cannot move by
    construction (it never looks at w), and the weight-proportional reading must
    move in the direction the heavy edge implies: mass pulled onto the far
    endpoint of the bridge is mass that no longer has to be transported across
    it, so W1 falls and kappa RISES.

    The uniform half is the planted negative: if it moves, the argument is
    leaking into the default path and the two downstream agents are broken.
    """
    m = _cv()
    W = _triangles_with_a_bridge()
    D = m.graph_metric(W)                       # metric of the UNWEIGHTED graph
    heavy = W.copy()
    heavy[2, 3] = heavy[3, 2] = 5.0

    u_before = m.kappa_edge(W, D, 2, 3, measure="uniform")
    u_after = m.kappa_edge(heavy, D, 2, 3, measure="uniform")
    p_before = m.kappa_edge(W, D, 2, 3, measure="weight-proportional")
    p_after = m.kappa_edge(heavy, D, 2, 3, measure="weight-proportional")
    print("\n  bridge (2,3), metric held fixed at the unweighted hop metric")
    print("    uniform             %+.6f -> %+.6f   (delta %.3e)"
          % (u_before, u_after, u_after - u_before))
    print("    weight-proportional %+.6f -> %+.6f   (delta %+.6f)"
          % (p_before, p_after, p_after - p_before))

    assert u_after == u_before, \
        "the uniform reading moved when only an edge WEIGHT changed: %.17g -> %.17g" \
        % (u_before, u_after)
    assert p_after > p_before + 1e-3, \
        ("a heavy bridge edge did not raise the weight-proportional curvature: "
         "%+.6f -> %+.6f" % (p_before, p_after))


def test_the_default_measure_path_is_bitwise_unchanged():
    """ricci_flow.py and chess_steps.py import this module and are green or
    mid-flight. The default path must be what it was before the measure argument
    existed, bitwise, and must still reproduce the published table."""
    m = _cv()
    X, labels = m.two_cluster_bed()
    default = m.curvature_from_points(X)
    explicit = m.curvature_from_points(X, measure="uniform")
    assert default["all_edges"] == explicit["all_edges"]
    assert np.array_equal(np.asarray(default["kappa"]), np.asarray(explicit["kappa"]))
    assert default["n_edges"] == explicit["n_edges"] == 590

    intra, bridge = m.split_by_plant(default, labels)
    si, sb = m.group_stats(intra), m.group_stats(bridge)
    print("\n  default path: intra %+.4f (edges %d), bridge %+.4f (edges %d)"
          % (si["mean"], si["n_edges"], sb["mean"], sb["n_edges"]))
    assert abs(si["mean"] - 0.1363) < 5e-5, "intra moved off the published +0.1363"
    assert abs(sb["mean"] - (-0.3479)) < 5e-5, "bridge moved off the published -0.3479"
    assert si["n_edges"] == 582 and sb["n_edges"] == 8

    for key in ("edges", "kappa", "n_edges", "n_refused", "refusals", "alpha",
                "W", "D", "all_edges", "kappa_full", "edge_index", "refusal_at"):
        assert key in default, "the result dict lost the key %r" % key

    W, glabels = m.two_block_bed()
    assert np.array_equal(
        np.asarray(m.curvature_from_graph(W)["kappa"]),
        np.asarray(m.curvature_from_graph(W, measure="uniform")["kappa"]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))

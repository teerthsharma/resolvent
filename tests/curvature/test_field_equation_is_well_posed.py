"""Is kappa_target(e) = lambda*T(e) - Lambda a well-posed field equation on a graph?

THE OBJECT UNDER EXAMINATION (PHASE C spec, verbatim):

    T(i,j) := |dq(i) - dq(j)| under the read's interventions
    kappa_target(e) = lambda * T(e) - Lambda
    w_e <- w_e * (1 - eta (kappa_e - kappa_target(e)))
    FIXED POINT <=> kappa == kappa_target <=> "the phase held by the tensor"

SIX PLACES IT CAN FAIL SILENTLY, one test group each:

  1. ATTAINABILITY.  kappa = 1 - W1/d and W1 >= 0, so kappa <= 1 ALWAYS. T is an
     absolute difference, bounded below by 0, so kappa_target >= -Lambda and
     kappa_target > 1 for every T > (1+Lambda)/lambda. A target above the ceiling is
     a fixed point that CANNOT EXIST. The flow chases it forever and the printed
     residual looks merely unconverged.
  2. THE FLOW.  w_e <- w_e(1 - eta r_e) crosses zero at eta r_e = 1. A weight of zero
     deletes the edge; a negative weight is not a reweighting, it makes the
     shortest-path distance undefined (every negative undirected edge is a negative
     cycle), so kappa does not exist rather than being merely wrong.
  3. GAUGE AND STABILITY.  kappa depends on d, d depends on w, the flow changes w.
     kappa is invariant under a global rescaling of w, so the fixed-point SET is a ray
     and the Jacobian is singular along w. Freezing d makes the residual converge to a
     number the recomputed curvature contradicts. And with the weight-proportional
     measure the spectrum of diag(w)J is mixed-sign, so the fixed point is unstable for
     every eta and both sign conventions.
  4. THE MEASURE CONVENTION DECIDES 3, AND THE SPEC DOES NOT NAME IT.  Uniform lazy
     measure: single-signed positive spectrum, the spec's own flow contracts. Weight-
     proportional: mixed-sign, nothing contracts. Whichever a curvature module picks,
     it settles the field equation's well-posedness without the field equation saying
     so.
  5. THE CITED FLOW IS A DIFFERENT OPERATOR.  Ni, Lin, Luo & Gao 2019 (arXiv
     1907.03993) Eq. 10 is w^(k+1)_ij = (1 - kappa^(k)_ij) * d^(k)(i,j) -- an
     ASSIGNMENT of a curvature-scaled shortest-path distance, with no eta and no
     target. The spec's update rescales the existing weight. They agree only where
     every edge is its own shortest path.
  6. FREEZE.  lambda and Lambda are fitted on a planted bed and then frozen. A refit
     after data is struck by the owner's rule, so freezing has to be a check that
     fires, not a promise.

EVERY INSTRUMENT HERE CARRIES A PLANTED NEGATIVE. An attainability checker that
passes everything is worthless, so each checker is shown a case it must flag and a
case it must not.

THE CURVATURE USED HERE IS HAND-BUILT, ON PURPOSE. `ceqjepa.curvature` is being
written in parallel and is not imported. `_curvature` below computes the exact
Ollivier-Ricci curvature by solving the transport LP, and
`test_the_hand_built_curvature_reproduces_the_closed_forms` is its control: it
reproduces kappa(K_n) = (n-2)/(n-1) for the alpha=0 walk and kappa(K_2) = 1-|2a-1|
for the a-lazy walk, both closed forms, before anything else in this file runs.

INTERFACE THIS FILE WOULD NEED FROM ceqjepa.curvature TO DROP THE HAND-BUILT ONE:

    ollivier_ricci(weights: np.ndarray[n,n], alpha: float, *,
                   measure: "uniform" | "proportional" = ...,
                   length: "weight" | "reciprocal" = "weight",
                   metric: np.ndarray[n,n] | None = None) -> np.ndarray[n,n]

    * `weights[i,j] > 0` iff the edge exists; symmetric; zero diagonal.
    * `alpha` is the idleness of the lazy walk.
    * `measure` picks m_x = alpha*delta_x + (1-alpha) * (uniform over N(x)) against
      m_x = alpha*delta_x + (1-alpha) * w[x,.]/sum(w[x,.]). Group 4 shows this is not a
      detail: it decides whether the field equation's flow converges.
    * `length` says which role w plays in the GROUND METRIC: "weight" is the spec's
      identification (edge length = w, Ni et al.'s convention), "reciprocal" is edge
      length = 1/w. Group 3 shows the two are not interchangeable and that the spec's
      choice is the unstable one under the proportional measure.
    * `metric` overrides the ground metric with a FROZEN distance matrix instead of
      recomputing shortest paths from `weights`. Group 3 needs both branches.
    * exact W1 (a transport LP), not a Sinkhorn approximation. The measurements this
      file makes run down to a 1.3989e-14 flow residual and a 3.8311e-11 frozen-metric
      residual, both far below any entropic bias. (The 5.625e-07 gauge defect is NOT a
      curvature error -- it is the central-difference floor of `field_eq.jacobian` at
      h = 1e-05; the invariance itself is exact, and is checked directly to 1e-10 in
      `test_curvature_is_invariant_under_a_global_rescaling_so_the_fixed_point_is_a_ray`.)
"""
from __future__ import annotations

import collections
import contextlib
import inspect
import io
import itertools
import pathlib
import re
import sys

import numpy as np
import pytest
from scipy.optimize import linprog
from scipy.sparse.csgraph import NegativeCycleError, shortest_path

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ceqjepa import field_eq  # noqa: E402  -- the module under test


# --------------------------------------------------------------------------
# the hand-built curvature (see the interface note above)
# --------------------------------------------------------------------------

def _lengths(w, edges, n, length):
    L = np.full((n, n), np.inf)
    for (i, j), v in zip(edges, w):
        ell = v if length == "weight" else 1.0 / v
        L[i, j] = L[j, i] = ell
    np.fill_diagonal(L, 0.0)
    return L


def _metric(w, edges, n, length="weight"):
    return shortest_path(_lengths(w, edges, n, length), method="D", directed=False)


def _measure(P, x, alpha, measure="proportional"):
    """The lazy walk's measure at x.

    TWO CONVENTIONS, BOTH DEFENSIBLE, AND THE SPEC NAMES NEITHER:
      "proportional" -- m_x(y) is proportional to w_xy, the Ni et al. 2019 convention
                        and the one "edge weights = attention" implies;
      "uniform"      -- m_x(y) = 1/deg(x), Ollivier's original combinatorial measure,
                        under which the weights reach kappa ONLY through the metric.
    `test_well_posedness_flips_with_the_unstated_measure_convention` shows the choice
    decides whether the spec's flow is stable at all.
    """
    m = np.zeros(len(P))
    nb = np.flatnonzero(P[x] > 0)
    if measure == "uniform":
        m[nb] = (1.0 - alpha) / len(nb)
    else:
        p = P[x, nb]
        m[nb] = (1.0 - alpha) * p / p.sum()
    m[x] += alpha
    return m


def _w1(mu, nu, d):
    """Exact Wasserstein-1 with ground metric d, by the transport LP."""
    n = len(mu)
    rows = [np.zeros((n, n)) for _ in range(2 * n)]
    for i in range(n):
        rows[i][i, :] = 1.0
        rows[n + i][:, i] = 1.0
    A = np.array([r.reshape(-1) for r in rows])
    res = linprog(d.reshape(-1), A_eq=A, b_eq=np.concatenate([mu, nu]),
                  bounds=(0, None), method="highs")
    assert res.status == 0, res.message
    return float(res.fun)


def _curvature(w, edges, n, alpha=0.0, length="weight", metric=None,
               measure="proportional"):
    """Exact Ollivier-Ricci per edge. `metric` freezes the ground metric."""
    w = np.asarray(w, dtype=float)
    P = np.zeros((n, n))
    for (i, j), v in zip(edges, w):
        P[i, j] = P[j, i] = v
    D = _metric(w, edges, n, length) if metric is None else metric
    return np.array([1.0 - _w1(_measure(P, i, alpha, measure),
                               _measure(P, j, alpha, measure), D) / D[i, j]
                     for (i, j) in edges])


def _edge_d(w, edges, n, length="weight"):
    """Shortest-path distance across each edge -- Ni et al. Eq. 10 needs d, not w."""
    D = _metric(w, edges, n, length)
    return np.array([D[i, j] for (i, j) in edges])


# --------------------------------------------------------------------------
# the bed: two triangles joined by one bridge. 6 nodes, 7 edges.
# The bridge is the edge a consequence-stress source is supposed to see.
# --------------------------------------------------------------------------

BED_N = 6
BED_EDGES = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (2, 3)]
BRIDGE = 6  # index of (2,3) in BED_EDGES


def bed_kappa(length="weight", metric=None, alpha=0.0, measure="proportional"):
    """A kappa_fn(w) -> per-edge curvature, the shape field_eq.run_flow consumes."""
    return lambda w: _curvature(w, BED_EDGES, BED_N, alpha, length, metric, measure)


# ==========================================================================
# CONTROL: the instrument, before any claim is made with it
# ==========================================================================

def test_the_hand_built_curvature_reproduces_the_closed_forms():
    for n in (2, 3, 4, 5):
        edges = list(itertools.combinations(range(n), 2))
        k = _curvature(np.ones(len(edges)), edges, n, alpha=0.0)
        assert np.allclose(k, (n - 2) / (n - 1), atol=1e-9), f"K_{n}: {k}"
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        k = _curvature([1.0], [(0, 1)], 2, alpha=a)[0]
        assert abs(k - (1.0 - abs(2 * a - 1))) < 1e-9, f"K_2 alpha={a}: {k}"


# ==========================================================================
# 1. ATTAINABILITY
# ==========================================================================

def test_curvature_has_a_hard_ceiling_of_one_that_the_target_can_exceed():
    """DERIVED: kappa = 1 - W1/d, W1 >= 0, d > 0, so kappa <= 1. RUN: never exceeded."""
    rng = np.random.default_rng(0)
    worst = -np.inf
    for _ in range(40):
        w = np.exp(rng.uniform(np.log(1e-2), np.log(1e2), len(BED_EDGES)))
        worst = max(worst, _curvature(w, BED_EDGES, BED_N).max())
    assert worst <= field_eq.KAPPA_CEILING + 1e-12, worst
    assert field_eq.KAPPA_CEILING == 1.0
    # and the source walks straight past it: T >= 0 is the only bound T carries.
    lam, Lam = 0.8, 1.5
    T = np.array([0.0, 1.0, 3.125, 50.0])
    tgt = field_eq.linear_target(T, lam, Lam)
    assert tgt.min() == pytest.approx(-Lam), "T >= 0 pins the target's floor at -Lambda"
    assert (tgt > field_eq.KAPPA_CEILING).any(), "no T makes this target unreachable?"
    assert tgt[2] == pytest.approx(1.0), "T = (1+Lambda)/lambda is exactly the ceiling"


def test_the_minus_two_floor_is_not_a_bound_on_the_state_space_the_flow_explores():
    """PLANTED NEGATIVE for anyone who codes the attainable set as the box [-2, 1].

    -2(1-alpha) is the floor for UNIT edge lengths. The flow makes the graph weighted
    on its first step, and on a weighted graph kappa is unbounded below.
    """
    assert field_eq.unweighted_floor(0.0) == -2.0
    assert field_eq.unweighted_floor(0.5) == -1.0
    w = np.ones(len(BED_EDGES))
    w[BRIDGE] = 0.5
    k = _curvature(w, BED_EDGES, BED_N)[BRIDGE]
    assert k < field_eq.unweighted_floor(0.0), f"bridge kappa {k} did not break -2"
    assert k == pytest.approx(-2.8, abs=1e-6), k
    # and it is not a near miss: the same sweep the ceiling test runs also has a floor
    rng = np.random.default_rng(0)
    worst = min(_curvature(np.exp(rng.uniform(np.log(1e-2), np.log(1e2),
                                              len(BED_EDGES))), BED_EDGES, BED_N).min()
                for _ in range(60))
    assert worst < -1e3, worst


def test_an_unattainable_target_is_flagged_and_an_attainable_one_is_not():
    """PLANTED PAIR, both branches of the checker."""
    above = field_eq.classify_target(1.8)
    assert not above.attainable and "ceiling" in above.reason
    inside = field_eq.classify_target(0.5)
    assert inside.attainable, inside.reason

    # the deeper planted negative: inside the box, outside THIS edge's reachable set.
    rng = np.random.default_rng(3)
    lo, hi = field_eq.reachable_interval(
        lambda w: _curvature(w, BED_EDGES, BED_N)[0], len(BED_EDGES), rng, n=40)
    assert lo > -1.0, f"pick a tighter edge; reachable=({lo},{hi})"
    v = field_eq.classify_target(lo - 0.25, reachable=(lo, hi))
    assert not v.attainable and "reachable" in v.reason, v.reason
    v = field_eq.classify_target((lo + hi) / 2.0, reachable=(lo, hi))
    assert v.attainable, v.reason


def test_a_single_edge_graph_reaches_exactly_one_curvature_whatever_the_weight():
    """The reachable set can be a POINT. No reweighting moves it, so every other
    target in [-2, 1] is a fixed point that cannot exist on this graph."""
    rng = np.random.default_rng(11)
    lo, hi = field_eq.reachable_interval(
        lambda w: _curvature(w, [(0, 1)], 2, alpha=0.0), 1, rng, n=20)
    assert hi - lo < 1e-12 and abs(lo) < 1e-12, (lo, hi)
    assert not field_eq.classify_target(0.5, reachable=(lo, hi)).attainable


def test_an_unattainable_target_plateaus_the_residual_at_a_nonzero_floor():
    """THE SILENT FAILURE. Nothing in the trace says 'impossible'; it says 'not yet'."""
    w0 = np.ones(len(BED_EDGES))
    w0[BRIDGE] = 0.5
    tgt = _curvature(w0, BED_EDGES, BED_N).copy()
    tgt[BRIDGE] = 1.8  # above the ceiling: cannot be met by any weight vector
    tr = field_eq.run_flow(w0, bed_kappa(), tgt, eta=0.05, steps=200)
    assert tr.stopped is None, tr.stopped
    assert field_eq.is_stuck(tr.residual, window=20, tol=1e-4)
    assert field_eq.residual_floor(tr.residual) > 1.0, tr.residual[-1]
    assert tr.residual[-1] < tr.residual[0], "it must LOOK like it converged part-way"


def test_the_plateau_detector_does_not_fire_on_a_converging_flow():
    """CONTROL for the detector. Uses the rerouted flow, because the spec's flow does
    not converge even on a target that is attainable by construction -- see group 3."""
    wstar = np.ones(len(BED_EDGES))
    kfn = bed_kappa(length="reciprocal")
    tgt = kfn(wstar)
    tr = field_eq.run_flow(np.array([0.6, 1.4, 1.0, 0.9, 1.1, 1.0, 1.3]), kfn, tgt,
                           eta=1.0, steps=120, sign=+1, normalise=True)
    assert tr.stopped is None, tr.stopped
    assert tr.residual[-1] < 1e-6, tr.residual[-1]
    # it DOES plateau -- at zero. Only `is_stuck` separates the two readings.
    assert field_eq.is_plateau(tr.residual, window=20, tol=1e-4)
    assert not field_eq.is_stuck(tr.residual, window=20, tol=1e-4)


# ==========================================================================
# 2. THE FLOW'S OWN PATHOLOGIES
# ==========================================================================

def test_the_eta_threshold_is_the_reciprocal_of_the_largest_positive_gap():
    """DERIVED: w(1 - eta r) <= 0 iff eta r >= 1. Checked against a brute scan."""
    kappa = np.array([1.0, 0.2, -0.3])
    tgt = np.array([-1.5, 0.0, -0.1])
    gap = (kappa - tgt).max()
    eta_star = field_eq.eta_ceiling_from_gap(kappa, tgt)
    assert eta_star == pytest.approx(1.0 / gap)
    w = np.ones(3)
    for eta in np.linspace(0.01, 2.0, 200):
        died = bool((field_eq.flow_step_unchecked(w, kappa, tgt, eta) <= 0).any())
        assert died == (eta >= eta_star - 1e-12), (eta, eta_star)


def test_a_plausible_eta_drives_a_weight_non_positive_and_a_safe_one_does_not():
    """PLANTED PAIR. Lambda = 1.5 from a fit, eta = 0.5 from the cited flow's folklore."""
    Lam = 1.5
    assert field_eq.eta_ceiling_from_Lambda(Lam) == pytest.approx(1.0 / (1.0 + Lam))
    kappa = np.array([1.0, 0.4])          # kappa = 1 is the ceiling
    tgt = field_eq.linear_target(np.array([0.0, 0.0]), 0.8, Lam)  # T = 0 -> -Lambda
    assert (field_eq.flow_step_unchecked(np.ones(2), kappa, tgt, 0.5) <= 0).any()
    safe = 0.9 * field_eq.eta_ceiling_from_gap(kappa, tgt)
    assert (field_eq.flow_step(np.ones(2), kappa, tgt, safe) > 0).all()
    # the folklore eta < 1 is safe for the CITED flow (no target) and unsafe with one
    assert (field_eq.flow_step_unchecked(np.ones(2), kappa, np.zeros(2), 0.5) > 0).all()


def test_a_non_positive_weight_is_refused_rather_than_returned():
    with pytest.raises(field_eq.NonPositiveWeight):
        field_eq.flow_step(np.ones(2), np.array([1.0, 0.4]),
                           np.array([-1.5, -1.5]), eta=0.5)


def test_a_negative_edge_length_leaves_no_shortest_path_metric_at_all():
    """A negative weight is not a reweighting and not merely a wrong number: every
    negative undirected edge is a negative cycle, so d does not exist and neither
    does kappa = 1 - W1/d."""
    w = np.ones(len(BED_EDGES))
    w[BRIDGE] = -0.25
    L = _lengths(w, BED_EDGES, BED_N, "weight")
    with pytest.raises(NegativeCycleError):
        shortest_path(np.where(np.isinf(L), 0.0, L), method="BF", directed=False)


# ==========================================================================
# 3. GAUGE: kappa depends on d, d depends on w, the flow changes w
# ==========================================================================

def test_curvature_is_invariant_under_a_global_rescaling_so_the_fixed_point_is_a_ray():
    w = np.array([1.0, 2.0, 3.0, 1.5, 0.7, 2.2, 0.3])
    base = _curvature(w, BED_EDGES, BED_N)
    for c in (2.0, 13.7, 1000.0):
        assert np.allclose(_curvature(c * w, BED_EDGES, BED_N), base, atol=1e-10)
    J = field_eq.jacobian(bed_kappa(), w)
    rel = np.abs(J @ w).max() / (np.abs(J).max() * np.abs(w).max())
    assert rel < 1e-7, f"the gauge mode must lie in the kernel of J: {rel}"
    assert field_eq.stability(bed_kappa(), w).gauge_defect < 1e-7


def test_freezing_the_metric_converges_to_a_residual_the_true_curvature_contradicts():
    """THE MEASUREMENT THAT DECIDES QUESTION 3. Freeze d and the flow reports a
    residual of 3.8311e-11 while the recomputed curvature is 0.4306 off target -- four
    orders of magnitude apart. Silent, and green on every printed diagnostic."""
    w0 = np.ones(len(BED_EDGES))
    w0[BRIDGE] = 0.5
    tgt = _curvature(w0, BED_EDGES, BED_N).copy()
    tgt[BRIDGE] += 0.15
    d0 = _metric(w0, BED_EDGES, BED_N)
    tr = field_eq.run_flow(w0, bed_kappa(metric=d0), tgt, eta=0.3, steps=60)
    assert tr.stopped is None, tr.stopped
    assert tr.residual[-1] < 1e-8, f"frozen-d must LOOK converged: {tr.residual[-1]}"
    true_resid = float(np.abs(bed_kappa()(tr.w) - tgt).max())
    assert true_resid > 1e4 * tr.residual[-1], (tr.residual[-1], true_resid)
    assert true_resid > 0.1, true_resid


def test_the_spec_fixed_point_is_linearly_unstable_for_every_eta():
    """Even with a target attainable BY CONSTRUCTION -- it is the curvature of a real
    weight vector -- the flow leaves it. diag(w)J carries eigenvalues of BOTH signs,
    so no step size and no sign convention makes the spectral radius <= 1."""
    wstar = np.ones(len(BED_EDGES))
    st = field_eq.stability(bed_kappa(), wstar)
    assert st.gauge_defect < 1e-6, st.gauge_defect
    assert st.eigs.real.min() < -1e-3, st.eigs
    assert st.eigs.real.max() > 1e-3, st.eigs
    assert st.stable_eta_max is None, st.verdict
    for eta in (0.01, 0.3):
        assert field_eq.spectral_radius(st.eigs, eta, sign=-1) > 1.0
        assert field_eq.spectral_radius(st.eigs, eta, sign=+1) > 1.0
    # `> 1.0` is one-sided and any always-large return satisfies it -- the mutation
    # harness caught exactly that, so pin the function to its definition and to a case
    # it must call CONTRACTING. max|1 + sign*eta*mu|, nothing else.
    assert field_eq.spectral_radius(np.array([0.5]), 1.0, sign=-1) == pytest.approx(0.5)
    assert field_eq.spectral_radius(np.array([-0.25]), 2.0, sign=+1) == pytest.approx(0.5)
    assert field_eq.spectral_radius(np.array([0.5, -2.0]), 1.0, sign=-1) == pytest.approx(3.0)
    # and the divergence is real, not a linearisation artefact
    tr = field_eq.run_flow(np.array([0.6, 1.4, 1.0, 0.9, 1.1, 1.0, 1.3]),
                           bed_kappa(), bed_kappa()(wstar), eta=0.1, steps=120)
    assert min(tr.residual) < 0.06, "it approaches"
    assert tr.residual[-1] > 5 * min(tr.residual), "and then leaves"


def test_the_reroute_makes_the_spectrum_single_signed_and_recovers_the_planted_weights():
    """THE SURVIVING ROUTE, measured. w cannot be both the attention weight and the
    edge length: attention is high where nodes are CLOSE. Take length = 1/w, gauge-
    normalise, and run the update on the ATTENTION weight, where it reads
    w <- w(1 + eta r). On the LENGTH ell = 1/w the same step is ell <- ell/(1 + eta r):
    a division, not a sign flip. Writing it as a sign flip on a length variable
    diverges -- it.2 measured a weight range of 1.571e+36 and HiGHS status 15 that way.

    PARTLY WRITTEN AFTER THE MODULE. This test was in the RED run, but its body was
    later rewritten to bind `Stability.nonzero`, which did not exist then, so its
    current form is not the form that went red. The revert driven by
    `test_the_reconstructed_red_fails_exactly_what_binds_nonzero` fails this test too,
    for that reason.
    """
    wstar = np.ones(len(BED_EDGES))
    kfn = bed_kappa(length="reciprocal")
    st = field_eq.stability(kfn, wstar, sign=+1)
    assert (st.nonzero < 0).all(), st.eigs
    assert len(st.nonzero) == len(st.eigs) - 1, "exactly one gauge mode"
    assert st.stable_eta_max == pytest.approx(2.0 / np.abs(st.nonzero).max(), rel=1e-9)
    # the bound must actually BE the contraction boundary, not a number beside one
    assert field_eq.spectral_radius(st.nonzero, 0.8 * st.stable_eta_max, sign=+1) < 1.0
    assert field_eq.spectral_radius(st.nonzero, 1.1 * st.stable_eta_max, sign=+1) > 1.0
    tgt = kfn(wstar)
    w0 = np.array([0.6, 1.4, 1.0, 0.9, 1.1, 1.0, 1.3])
    inside = field_eq.run_flow(w0, kfn, tgt, eta=0.8 * st.stable_eta_max, steps=200,
                               sign=+1, normalise=True)
    assert inside.stopped is None, inside.stopped
    assert inside.residual[-1] < 1e-9, inside.residual[-1]
    assert np.abs(inside.w / wstar - 1.0).max() < 1e-6, inside.w
    # PLANTED NEGATIVE for the eta bound: past it, the flow fails.
    outside = field_eq.run_flow(w0, kfn, tgt, eta=1.1 * st.stable_eta_max, steps=200,
                                sign=+1, normalise=True)
    assert outside.stopped is not None or outside.residual[-1] > 1e-3, outside.residual[-1]


# ==========================================================================
# 4. THE MEASURE CONVENTION, WHICH DECIDES GROUP 3 AND IS NOT IN THE SPEC
# ==========================================================================

def test_well_posedness_flips_with_the_unstated_measure_convention():
    """WRITTEN AFTER THE MODULE. Its RED is a reconstruction, and is labelled so.

    This test postdates the RED run for this file: it was added once
    `ceqjepa/curvature.py` landed and its `lazy_measure` turned out to put mass
    (1-alpha)/deg on every neighbour irrespective of weight. Its RED was re-established
    afterwards rather than before, by deleting `Stability.nonzero` -- a field this test
    binds and which did not exist at the RED run -- in a scratch copy of the module.
    Under that revert the test raises

        AttributeError: 'Stability' object has no attribute 'nonzero'

    and the suite exits 1. `test_the_reconstructed_red_fails_exactly_what_binds_nonzero`
    drives that revert in-process, so the reconstruction is a run rather than a memory.

    NO FAILURE COUNT IS QUOTED HERE, deliberately. A count of failing tests is
    invalidated by adding a test, which is exactly how a stated RED comes to describe
    less coverage than the green shipped beside it -- twice, on this file. The revert
    also breaks `demo()`, which reads `.nonzero`, and that takes both number-guard tests
    down with it; so the total moves with the file and is printed by a run, never
    asserted in prose.

    NOR IS THIS THE ONLY POST-MODULE TEST, and an earlier version of this docstring
    said it was. `test_the_reroute_...` predates the module but had its body rewritten
    to bind the same field; the number-guard, mutation-identity and reconstructed-RED
    tests are all post-module by construction. The re-derivable half is that a grep for
    `.nonzero` returns exactly the two tests named above plus `demo()`. How many test
    functions the file carried at the RED run is NOT re-derivable -- tests/curvature/ is
    untracked, no git object holds that state, and the pytest cache carries a stale id.
    Any figure for it is an uncounted reconstruction and is not asserted anywhere.

    THE COUPLING NOBODY OWNS. The spec fixes the update and the source but not the
    lazy measure, and the measure decides whether its flow is stable at all.

    m_x proportional to w (Ni et al.; what "edge weights = attention" implies): the
    weights reach kappa through the transport measure AND the metric, the two channels
    pull opposite ways, the spectrum is mixed-sign and NO eta works.

    m_x uniform over N(x) (Ollivier's combinatorial measure): the weights reach kappa
    only through the metric, the spectrum is single-signed and positive, and the spec's
    own update contracts for eta < 1.174514. Well posed -- but the kernel is now TWO
    dimensional, so the fixed point is a 2-parameter family rather than a ray, and the
    gauge problem is strictly worse than under the measure that broke stability.

    Whichever a curvature module picks, it decides this question for the field equation
    without the field equation saying so.
    """
    w = np.ones(len(BED_EDGES))
    prop = field_eq.stability(bed_kappa(measure="proportional"), w)
    unif = field_eq.stability(bed_kappa(measure="uniform"), w)

    assert prop.eigs.real.min() < -1e-3 and prop.eigs.real.max() > 1e-3
    assert prop.stable_eta_max is None, prop.verdict
    assert len(prop.nonzero) == len(w) - 1, "one gauge mode under the proportional measure"

    assert (unif.nonzero > 0).all(), unif.eigs
    assert unif.stable_eta_max == pytest.approx(2.0 / unif.nonzero.max(), rel=1e-9)
    assert 1.0 < unif.stable_eta_max < 1.5, unif.stable_eta_max
    assert len(unif.nonzero) == len(w) - 2, "TWO gauge modes under the uniform measure"

    # and the run agrees with the linearisation on the side that says it should
    w0 = np.array([0.6, 1.4, 1.0, 0.9, 1.1, 1.0, 1.3])
    kfn = bed_kappa(measure="uniform")
    tr = field_eq.run_flow(w0, kfn, kfn(w), eta=0.8 * unif.stable_eta_max, steps=200,
                           normalise=True)
    assert tr.stopped is None and tr.residual[-1] < 1e-8, (tr.stopped, tr.residual[-1])


def test_gauge_normalisation_removes_the_scale_drift():
    """Without it the weights run away while the residual falls: a converged curvature
    on a weight vector that is leaving the finite part of the space."""
    w0 = np.ones(len(BED_EDGES))
    w0[BRIDGE] = 0.5
    tgt = _curvature(w0, BED_EDGES, BED_N).copy()
    tgt[BRIDGE] = 1.8
    free = field_eq.run_flow(w0, bed_kappa(), tgt, eta=0.05, steps=200)
    held = field_eq.run_flow(w0, bed_kappa(), tgt, eta=0.05, steps=200, normalise=True)
    assert abs(free.log_scale[-1]) > 1.0, free.log_scale[-1]
    assert abs(held.log_scale[-1]) < 1e-9, held.log_scale[-1]
    assert field_eq.gauge_normalise(np.array([2.0, 8.0])) == pytest.approx([0.5, 2.0])


# ==========================================================================
# 5. THE CITED FLOW IS A DIFFERENT OPERATOR
#    Ni, Lin, Luo & Gao 2019 (arXiv 1907.03993) Eq. 10:
#        w^(k+1)_ij = (1 - kappa^(k)_ij) * d^(k)(i, j)
# ==========================================================================

def test_the_spec_update_equals_the_cited_one_only_where_every_edge_is_a_geodesic():
    """Eq. 10 assigns a curvature-scaled DISTANCE; the spec rescales the WEIGHT. Where
    d(i,j) == w_ij the two coincide at eta = 1 with a zero source, and nowhere else."""
    w = np.ones(len(BED_EDGES))
    k = _curvature(w, BED_EDGES, BED_N)
    d = _edge_d(w, BED_EDGES, BED_N)
    assert np.allclose(d, w), "the unit barbell has every edge on its own geodesic"
    assert np.allclose(field_eq.ni_eq10_step(k, d),
                       field_eq.flow_step_unchecked(w, k, np.zeros_like(k), eta=1.0))

    # PLANTED NEGATIVE: a triangle with one long edge, so d(0,1) = 2 < w(0,1) = 3.
    tri, n = [(0, 1), (0, 2), (1, 2)], 3
    wt = np.array([3.0, 1.0, 1.0])
    dt = _edge_d(wt, tri, n)
    assert dt[0] == pytest.approx(2.0) and dt[0] < wt[0]
    kt = _curvature(wt, tri, n)
    assert not np.allclose(field_eq.ni_eq10_step(kt, dt),
                           field_eq.flow_step_unchecked(wt, kt, np.zeros_like(kt), 1.0))


def test_the_two_flows_do_not_share_a_fixed_point_condition():
    """Spec, zero source: kappa == 0 on every edge. Eq. 10: w == (1 - kappa) d. The
    second implies the first only when d == w, so an identity bind against Ni's
    community separation cannot pass with the spec's update as written."""
    tri, n = [(0, 1), (0, 2), (1, 2)], 3
    wt = np.array([3.0, 1.0, 1.0])
    kt = _curvature(wt, tri, n)
    dt = _edge_d(wt, tri, n)
    assert not field_eq.at_fixed_point_spec(kt, np.zeros_like(kt))
    w_ni = field_eq.ni_eq10_step(kt, dt)
    assert not np.allclose(_curvature(w_ni, tri, n), 0.0, atol=1e-6), (
        "Eq. 10's image is not the spec's kappa == 0 surface")
    # the equivalence, stated as a check rather than as a hope
    assert field_eq.flows_agree(wt, dt) is False
    assert field_eq.flows_agree(np.ones(len(BED_EDGES)),
                               _edge_d(np.ones(len(BED_EDGES)), BED_EDGES, BED_N)) is True


# ==========================================================================
# 6. FITTING lambda AND Lambda, AND MAKING THE FREEZE A CHECK
# ==========================================================================

def test_the_fit_recovers_planted_constants():
    T = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
    lam, Lam = 0.37, 1.25
    got = field_eq.fit_lambda_Lambda(T, field_eq.linear_target(T, lam, Lam))
    assert got == pytest.approx((lam, Lam), abs=1e-10)


def test_a_frozen_fit_verifies_against_its_own_recorded_inputs(tmp_path):
    T = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
    k = field_eq.linear_target(T, 0.37, 1.25) + np.array([1e-3, -1e-3, 0, 1e-3, -1e-3])
    rec = field_eq.freeze_fit(tmp_path / "fit.json", "BED-BARBELL-v1", T, k, alpha=0.0,
                              convention="measure=uniform,length=weight,metric=recomputed")
    field_eq.verify_frozen(rec)
    field_eq.verify_frozen(field_eq.load_frozen(tmp_path / "fit.json"))
    field_eq.assert_reported_at(rec.lam, rec.Lam, rec)


def test_a_refit_is_detected_on_the_paths_that_can_catch_it(tmp_path):
    """PLANTED NEGATIVES: three scenarios, and they land on TWO of the module's three
    raise sites, not three. Stated that way because the count of scenarios is not the
    count of code paths, and only the second number is coverage.

      (a) constants edited in the record   -> verify_frozen, the reproduce branch
      (b) refit on new data, cited old     -> assert_reported_at, the constants branch
      (c) constants nobody froze           -> assert_reported_at, the SAME branch as (b)

    The third raise site, verify_frozen's digest branch, is not reachable from here --
    a refit produces a record that is internally consistent. It is covered instead by
    `test_the_freeze_record_carries_everything_the_fit_depended_on`, which tampers the
    digest directly.
    """
    T = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
    k = field_eq.linear_target(T, 0.37, 1.25)
    rec = field_eq.freeze_fit(tmp_path / "fit.json", "BED-BARBELL-v1", T, k, alpha=0.0)

    # (a) the constants were edited in the record
    with pytest.raises(field_eq.RefitDetected, match="does not reproduce"):
        field_eq.verify_frozen(rec._replace(lam=0.40))
    # (b) the record was refitted on different data and re-saved consistently.
    #     Note this reaches the reporting gate, not the digest branch: `refit` verifies
    #     cleanly on its own inputs, which is exactly why a digest over the CONSTANTS
    #     could never have caught it.
    T2 = T + 0.1
    refit = field_eq.freeze_fit(tmp_path / "fit2.json", "BED-BARBELL-v1", T2,
                                field_eq.linear_target(T2, 0.44, 1.10), alpha=0.0)
    field_eq.verify_frozen(refit)          # internally consistent, and still a refit
    assert refit.digest != rec.digest
    with pytest.raises(field_eq.RefitDetected, match="digest"):
        field_eq.assert_reported_at(refit.lam, refit.Lam, rec)
    # (c) a number reported at constants nobody froze
    with pytest.raises(field_eq.RefitDetected, match="reported"):
        field_eq.assert_reported_at(0.42, 1.25, rec)


def test_the_freeze_record_carries_everything_the_fit_depended_on(tmp_path):
    """A digest over the constants alone cannot detect (b). It must cover the inputs --
    and the CONVENTIONS, because the lazy measure decides whether the flow those
    constants drive converges at all."""
    T = np.array([0.0, 1.0, 2.0])
    k = field_eq.linear_target(T, 0.3, 1.0)
    a = field_eq.freeze_fit(tmp_path / "a.json", "BED", T, k, alpha=0.0, convention="u")
    b = field_eq.freeze_fit(tmp_path / "b.json", "BED", T, k, alpha=0.5, convention="u")
    c = field_eq.freeze_fit(tmp_path / "c.json", "OTHER", T, k, alpha=0.0, convention="u")
    d = field_eq.freeze_fit(tmp_path / "d.json", "BED", T, k, alpha=0.0, convention="p")
    assert a.lam == b.lam == c.lam == d.lam and a.Lam == b.Lam == c.Lam == d.Lam
    assert len({a.digest, b.digest, c.digest, d.digest}) == 4, (
        "bed, alpha AND convention must each enter the digest")
    # PLANTED NEGATIVE: same numbers, different measure convention, must not pass as one
    with pytest.raises(field_eq.RefitDetected, match="digest"):
        field_eq.assert_reported_at(d.lam, d.Lam, a._replace(digest=d.digest))


# ==========================================================================
# THE REROUTED SOURCE: a target that cannot leave the attainable set
# ==========================================================================

def test_the_squashed_source_is_attainable_for_every_stress_the_linear_one_is_not():
    """PLANTED PAIR on the same T. Same two free parameters, same monotone response to
    consequence stress, but the codomain is the attainable interval by construction."""
    T = np.array([0.0, 1.0, 10.0, 1e3, 1e9])
    lam, Lam = 0.8, 1.5
    linear = field_eq.linear_target(T, lam, Lam)
    assert not all(field_eq.classify_target(t).attainable for t in linear)
    sq = field_eq.squashed_target(T, lam=lam, T0=1.0, floor=-2.0,
                                  ceiling=field_eq.KAPPA_CEILING)
    assert all(field_eq.classify_target(t).attainable for t in sq), sq
    assert np.all(np.diff(sq) >= 0), "still monotone in the consequence stress"
    assert np.all(np.diff(sq[:3]) > 0), "and strictly so before it saturates"
    assert sq.max() < field_eq.KAPPA_CEILING and sq.min() > -2.0


# ==========================================================================
# THE NUMBER GUARD
#
# Copied from Chase's docstring-number test and widened the way Cameron widened it:
# it matches ANY decimal, and it ships NO exemption list, because an exemption list is
# how this check gets defanged. Where a legitimate constant would fail -- an arXiv id,
# a tolerance, a version -- the repair is to PRINT it in the run, not to excuse it.
#
# WHY IT IS HERE. Every number an audit struck off this agent's work was a decimal
# sitting in prose that no run printed: a recovery figure attributed to the wrong eta,
# a test threshold quoted as a reading, and a mutant count for a harness that did not
# exist. All three are the same defect and this is the bind for it.
#
# The producer of record is `demo()` below, not a demo inside field_eq.py: field_eq
# imports no curvature by design, so it cannot recompute its own docstring numbers.
# The hand-built transport LP in this file is what produced them, so this file is
# where they have to be reproducible from.
# ==========================================================================

#: ANY number: integer or decimal, with or without an exponent, and allowing a unit
#: suffix ("16.58s") to follow it. No exemption list, because an exemption list is how
#: this check gets defanged. Where a legitimate constant would fail -- an arXiv id, a
#: tolerance, a step count -- the repair is to PRINT it in the run, not to excuse it.
#:
#: THREE WIDENINGS, each forced by a case the narrower form let through:
#:   1. `(?![\w.])` after the number excluded every decimal that ENDS A SENTENCE.
#:      Caught by this file's own planted negative.
#:   2. Requiring a point or an exponent excluded every BARE INTEGER claim -- including,
#:      when this was found, the "2 failed, 22 passed" this file had just been struck for.
#:   3. `(?!\w)` excluded every number carrying a UNIT, so "16.58s" was invisible.
#: The trailing guard is now only `(?!\.\d)`, which stops a version string splitting.
NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?!\.\d)")

#: Kept under its old name so the widening is visible rather than silent.
DECIMAL = NUMBER


def _printed_tokens(text):
    """Numbers in `text`, AS TOKENS.

    Substring matching is not good enough and this is not a theoretical worry: with it,
    "23" passed because 1.6923e-11 contains it, "22" passed inside -0.3122, and "27"
    passed inside 0.872749. A guard that accepts a number because some longer number
    contains its digits is a guard that accepts almost any short claim.
    """
    return set(NUMBER.findall(text))


def _documented_numbers():
    """Every number in a docstring of the module under test AND of this file.

    Scanning only `field_eq` left this file's own prose unguarded, which is where the
    struck reconstructed-RED counts were sitting. A guard that exempts the file it
    lives in is an exemption list with one very large entry.
    """
    found = {}
    sources = [("field_eq:<module>", field_eq.__doc__)]
    for name in field_eq.__all__:
        sources.append((f"field_eq:{name}",
                        getattr(getattr(field_eq, name), "__doc__", None)))
    here = sys.modules[__name__]
    sources.append(("tests:<module>", here.__doc__))
    for name in sorted(vars(here)):
        obj = getattr(here, name)
        if callable(obj) and getattr(obj, "__module__", None) == __name__:
            sources.append((f"tests:{name}", getattr(obj, "__doc__", None)))
    for where, doc in sources:
        for number in NUMBER.findall(doc or ""):
            found.setdefault(number, []).append(where)
    return found


def _stdout_of(fn):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    return buf.getvalue()


def test_every_decimal_in_a_docstring_is_printed_by_a_run():
    """A number in prose that no run prints is a number bound to nothing.

    TWO TIERS, because a module has to be runnable on its own. Numbers in
    `ceqjepa.field_eq`'s own docstrings must be printed by `field_eq.demo()` -- the
    module's self-check, reachable as `python -m ceqjepa.field_eq` without pytest.
    Numbers in THIS file's docstrings may come from either demo. Letting the module's
    prose be discharged by a test file's demo is how a module ends up with claims that
    nobody can exercise without the test suite.
    """
    documented = _documented_numbers()
    module_printed = _printed_tokens(_stdout_of(field_eq.demo))
    both_printed = module_printed | _printed_tokens(_stdout_of(demo))

    own = {n: w for n, w in documented.items()
           if any(s.startswith("field_eq:") for s in w) and n not in module_printed}
    assert not own, (
        f"{len(own)} of field_eq's own documented numbers are not printed by "
        f"field_eq.demo(): "
        + "; ".join(f"{n} (in {', '.join(sorted(set(w)))})" for n, w in sorted(own.items())))

    missing = {n: w for n, w in documented.items() if n not in both_printed}
    detail = "; ".join(f"{n} (in {', '.join(sorted(set(w)))})"
                       for n, w in sorted(missing.items()))
    assert not missing, (
        f"{len(missing)} of {len(documented)} documented numbers are printed by no "
        f"run: {detail}")
    assert len(documented) > 60, f"only {len(documented)} numbers found -- scan broke"


def _demo_stuck_figures():
    """The four is_stuck figures `field_eq.demo()` emitted, parsed back out of stdout.

    The module's own self-check is the producer, so this checks the numbers a reader of
    `python -m ceqjepa.field_eq` actually sees.
    """
    lines = _stdout_of(field_eq.demo).splitlines()
    conv = [ln for ln in lines if "converging run spread" in ln]
    stuck = [ln for ln in lines if "stuck run" in ln and "spread" in ln]
    assert len(conv) == 1 and len(stuck) == 1, (conv, stuck)
    return ([float(x) for x in NUMBER.findall(conv[0])[-2:]]
            + [float(x) for x in NUMBER.findall(stuck[0])[-2:]])


def test_the_demo_prints_measurements_and_not_literals():
    """PLANTED IDENTITY, and equality alone is NOT enough to close it.

    The gap was real: reverting these four figures to the literals 8.9e-08, 3.6e-08,
    8.9e-06 and 1.269 left both number-guard tests green, because a printed literal
    satisfies a guard that only asks whether a number appeared.

    Checking the printed values against a recomputation does not close it either --
    those literals ARE the right values to two significant figures, so they pass. What
    a literal cannot do is MOVE. So this perturbs the runs the figures describe, by
    scaling every residual the flow reports, and requires all four printed numbers to
    scale with them. Spreads and floors are both homogeneous of degree one in the
    residual, so the expected factor is exact, and a hardcoded number fails it whatever
    its value. The figures are printed to four significant figures for the same reason:
    at %.1e the literals 8.9e-08 and 1.269 were indistinguishable from the measurement,
    and a tolerance wide enough to accept the rounding is wide enough to accept a stale
    number.

    THE FACTOR IS 1.01, NOT 2. `field_eq.demo()` asserts its own findings as it prints
    them -- that a near-fixed-point run dips below 0.06 before it leaves, that a frozen
    metric stays under 1e-8 -- and doubling every residual falsifies those, so the
    perturbation would be measuring the self-check rather than the print. One percent
    is far outside the four-significant-figure print and far inside every bar the demo
    asserts.
    """
    base = _demo_stuck_figures()
    real_run_flow = field_eq.run_flow
    factor = 1.01

    def scaled(*a, **k):
        tr = real_run_flow(*a, **k)
        return tr._replace(residual=[factor * x for x in tr.residual])

    try:
        field_eq.run_flow = scaled
        moved = _demo_stuck_figures()
    finally:
        field_eq.run_flow = real_run_flow
    assert field_eq.run_flow is real_run_flow

    for got, was in zip(moved, base):
        assert got == pytest.approx(factor * was, rel=2e-4), (moved, base)
        assert got != was, "a printed figure did not move with its measurement"

    # and they are the figures of the traces they claim to describe, not of some other
    w1s = np.ones(len(BED_EDGES))
    w0 = np.array([0.6, 1.4, 1.0, 0.9, 1.1, 1.0, 1.3])
    wb = np.ones(len(BED_EDGES)); wb[BRIDGE] = 0.5
    kr = bed_kappa(length="reciprocal")
    c = field_eq.run_flow(w0, kr, kr(w1s), eta=1.0, steps=120, sign=+1, normalise=True)
    tg = _curvature(wb, BED_EDGES, BED_N).copy(); tg[BRIDGE] = 1.8
    s = field_eq.run_flow(wb, bed_kappa(), tg, eta=0.05, steps=200)
    want = [max(c.residual[-20:]) - min(c.residual[-20:]),
            field_eq.residual_floor(c.residual),
            max(s.residual[-20:]) - min(s.residual[-20:]),
            field_eq.residual_floor(s.residual)]
    for got, wanted in zip(base, want):
        assert got == pytest.approx(wanted, rel=2e-4), (base, want)


def test_the_reconstructed_red_fails_exactly_what_binds_nonzero():
    """The reconstructed RED, driven rather than remembered.

    `Stability.nonzero` did not exist at this file's RED run. Reverting it must break
    every test that binds it and nothing else, which is what makes the reconstruction
    worth anything. The failing TOTAL is not asserted -- it moves whenever a test is
    added, and a stated RED that rots into describing less than the green beside it is
    the defect this replaces.
    """
    me = "test_the_reconstructed_red_fails_exactly_what_binds_nonzero"
    binders = [n for n in vars(sys.modules[__name__])
               if n.startswith("test_") and n != me
               and ".nonzero" in inspect.getsource(globals()[n])]
    assert set(binders) == {
        "test_the_reroute_makes_the_spectrum_single_signed_and_recovers_the_planted_weights",
        "test_well_posedness_flips_with_the_unstated_measure_convention"}, sorted(binders)
    assert ".nonzero" in inspect.getsource(field_eq.demo), (
        "the module's own demo binds it too, and takes the guard tests down with it")

    # Only the RETURNED object loses the field. `field_eq.stability` builds a real
    # Stability internally, so swapping the class itself would break construction
    # rather than reproduce the pre-`nonzero` module.
    fields = tuple(f for f in field_eq.Stability._fields if f != "nonzero")
    Reverted = collections.namedtuple("Stability", fields)
    real_stability = field_eq.stability

    def without_nonzero(*a, **k):
        full = real_stability(*a, **k)
        return Reverted(**{f: getattr(full, f) for f in fields})

    try:
        field_eq.stability = without_nonzero
        for name in binders:
            with pytest.raises(AttributeError, match="nonzero"):
                globals()[name]()
        with pytest.raises(AttributeError, match="nonzero"):
            with contextlib.redirect_stdout(io.StringIO()):
                field_eq.demo()
    finally:
        field_eq.stability = real_stability
    assert field_eq.stability is real_stability


def test_the_number_guard_fires_on_a_decimal_no_run_prints():
    """PLANTED NEGATIVE for the guard itself. A guard that passes everything is the
    thing it was built to catch."""
    original = field_eq.classify_target.__doc__
    try:
        field_eq.classify_target.__doc__ = (original or "") + " Throughput 3.14159."
        with pytest.raises(AssertionError, match="3.14159"):
            test_every_decimal_in_a_docstring_is_printed_by_a_run()
    finally:
        field_eq.classify_target.__doc__ = original
    # and it must pass again the moment the unprinted number is gone
    test_every_decimal_in_a_docstring_is_printed_by_a_run()


# ==========================================================================
# THE MUTATION HARNESS
#
# The planted negatives above are planted INPUTS: each checker is handed a case it
# must flag and a case it must not. That is necessary and it is not sufficient -- it
# says nothing about whether the ASSERTIONS would notice an instrument that stopped
# working. This harness answers that second question: it breaks one shipped instrument
# at a time and reports which tests notice.
#
# It is not a pytest test. It runs the whole suite once per mutant, which takes minutes,
# and a suite that takes minutes stops being run. Invoke it directly:
#
#     python tests/curvature/test_field_equation_is_well_posed.py
#
# A SURVIVING mutant is a finding: an instrument no assertion binds.
# ==========================================================================

def _mutants():
    """label -> (attribute of field_eq, replacement). Each breaks one instrument."""
    import hashlib
    return {
        "classify_target always attainable":
            ("classify_target", lambda *a, **k: field_eq.Verdict(True, "reachable ceiling")),
        "is_stuck always True": ("is_stuck", lambda *a, **k: True),
        "is_stuck always False": ("is_stuck", lambda *a, **k: False),
        "is_plateau always False": ("is_plateau", lambda *a, **k: False),
        "residual_floor always 0": ("residual_floor", lambda *a, **k: 0.0),
        "eta_ceiling_from_gap always inf":
            ("eta_ceiling_from_gap", lambda *a, **k: float("inf")),
        "eta_ceiling_from_Lambda ignores Lambda":
            ("eta_ceiling_from_Lambda", lambda Lam: 1.0),
        "flow_step never refuses": ("flow_step", None),   # bound to unchecked below
        "unweighted_floor is -inf": ("unweighted_floor", lambda a: float("-inf")),
        "jacobian is the identity":
            ("jacobian", lambda f, w, h=1e-5: np.eye(len(w))),
        "stability never finds a stable eta":
            ("stability", lambda *a, **k: field_eq.Stability(
                np.array([1.0]), np.array([1.0]), 0.0, None, "mutant")),
        "spectral_radius always 2": ("spectral_radius", lambda *a, **k: 2.0),
        "gauge_normalise is the identity":
            ("gauge_normalise", lambda w: np.asarray(w, dtype=float)),
        "squashed_target is the linear law":
            ("squashed_target", lambda T_, lam, T0, floor, ceiling=1.0, margin=1e-3:
                field_eq.linear_target(T_, lam, -floor)),
        "flows_agree always True": ("flows_agree", lambda *a, **k: True),
        "ni_eq10_step rescales the weight instead of assigning the distance":
            ("ni_eq10_step", lambda kappa, d: (1.0 - np.asarray(kappa, float))
                * np.asarray(d, float) * 0 + np.asarray(d, float)),
        "verify_frozen is a no-op": ("verify_frozen", lambda *a, **k: None),
        "assert_reported_at is a no-op": ("assert_reported_at", lambda *a, **k: None),
        "the digest ignores the convention":
            ("_digest", lambda bed, al, conv, Tv, kv: hashlib.sha256(repr(
                (bed, float(al), [float(x) for x in np.ravel(Tv)],
                 [float(x) for x in np.ravel(kv)])).encode()).hexdigest()),
        "fit_lambda_Lambda returns a constant":
            ("fit_lambda_Lambda", lambda T_, k_: (0.37, 1.25)),
    }


def _run_one(name):
    """Call a test function directly. Returns None if it passed, else the error name."""
    import inspect
    import tempfile
    fn = globals()[name]
    kw = {}
    if "tmp_path" in inspect.signature(fn).parameters:
        kw["tmp_path"] = pathlib.Path(tempfile.mkdtemp())
    try:
        fn(**kw)
        return None
    except BaseException as exc:          # pytest.fail raises off Exception
        return type(exc).__name__


def instrument_tests():
    """The tests the harness scores against.

    The two number-guard tests are excluded. They bind PROSE to a run, not an
    instrument to an assertion: under any mutant the demo prints different figures, so
    they would kill everything and the harness would stop discriminating. A kill has to
    mean "an assertion noticed the instrument broke".
    """
    skip = ("test_every_decimal_in_a_docstring_is_printed_by_a_run",
            "test_the_number_guard_fires_on_a_decimal_no_run_prints",
            "test_the_demo_prints_measurements_and_not_literals",
            "test_the_reconstructed_red_fails_exactly_what_binds_nonzero",
            "test_the_mutation_counts_are_recomputed_not_asserted")
    return sorted(n for n, v in globals().items()
                  if n.startswith("test_") and callable(v) and n not in skip)


def mutation_pass(tests, verbose=False):
    """Apply every registered mutant to `tests`. Counts come from the loop's own lists.

    Returns {applied, killed, survived, n_registered}. A published mutant count has to
    be recomputed from these lists in the same run: being PRINTED and being MEASURED
    are different properties, and a hardcoded figure satisfies the first one perfectly.
    `test_the_mutation_counts_are_recomputed_not_asserted` pins the identity
    len(killed) + len(survived) == len(applied) == n_registered, which no literal can
    satisfy, and checks that the set of labels applied is the registry itself.
    """
    registry = _mutants()
    applied, killed, survived = [], {}, []
    for label, (attr, repl) in registry.items():
        if repl is None:                  # "flow_step never refuses"
            repl = field_eq.flow_step_unchecked
        original = getattr(field_eq, attr)
        setattr(field_eq, attr, repl)
        try:
            noticed = [n for n in tests if _run_one(n) is not None]
        finally:
            setattr(field_eq, attr, original)
        assert getattr(field_eq, attr) is original, f"{attr} was not restored"
        applied.append(label)
        if noticed:
            killed[label] = noticed
        else:
            survived.append(label)
        if verbose:
            head = "SURVIVED    " if not noticed else "killed by %2d" % len(noticed)
            print("  %s  %s" % (head, label))
            if noticed:
                print("%16s%s" % ("", ", ".join(n[5:] for n in noticed)))
    return {"applied": applied, "killed": killed, "survived": survived,
            "n_registered": len(registry)}


def mutation_report():
    """Full on-demand pass. Every printed count is taken from the loop's own lists."""
    tests = instrument_tests()
    baseline = [n for n in tests if _run_one(n) is not None]
    print("baseline: %d tests, %d failing %s"
          % (len(tests), len(baseline), baseline if baseline else "(clean)"))
    if baseline:
        print("REFUSING to score mutants against a red baseline")
        return baseline
    res = mutation_pass(tests, verbose=True)
    assert len(res["killed"]) + len(res["survived"]) == len(res["applied"])
    assert len(res["applied"]) == res["n_registered"]
    print("")
    print("%d mutants, %d killed, %d survived"
          % (len(res["applied"]), len(res["killed"]), len(res["survived"])))
    for s in res["survived"]:
        print("  SURVIVOR (an instrument no assertion binds): %s" % s)
    return res["survived"]


def test_the_mutation_counts_are_recomputed_not_asserted():
    """PLANTED IDENTITY. The harness's counts must come out of its own loop.

    A number printed beside a harness that ran separately is a literal, and a literal
    passes any check that only asks whether it was printed. This runs the FULL mutant
    registry against three cheap tests -- enough to prove the loop applies every
    registered mutant, counts what it applied, and restores the module afterwards --
    and pins the arithmetic as an identity that no hardcoded figure can satisfy.

    The kill verdicts here are not the published ones; the published pass is
    `mutation_report()` over the whole suite, which is minutes long and on demand.
    """
    cheap = ["test_the_fit_recovers_planted_constants",
             "test_a_non_positive_weight_is_refused_rather_than_returned",
             "test_the_eta_threshold_is_the_reciprocal_of_the_largest_positive_gap"]
    before = {a: getattr(field_eq, a) for a, _ in _mutants().values()}
    res = mutation_pass(cheap)

    assert set(res["applied"]) == set(_mutants()), "a registered mutant was not applied"
    assert len(res["applied"]) == res["n_registered"] == len(_mutants())
    assert len(res["killed"]) + len(res["survived"]) == len(res["applied"])
    assert res["n_registered"] > 15, res["n_registered"]
    # every instrument is put back exactly as it was found
    for attr, original in before.items():
        assert getattr(field_eq, attr) is original, attr
    # the three cheap tests are a real subset, so SOME mutant must have been noticed
    # and some must have escaped -- an all-killed or all-survived pass would mean the
    # loop is not actually running the tests it says it is
    assert res["killed"] and res["survived"], (res["killed"].keys(), res["survived"])


def demo():
    """The numbers THIS FILE's prose quotes, on top of the module's own self-check.

    `field_eq.demo()` is the producer of record for everything `ceqjepa.field_eq`
    claims -- run it directly as `python -m ceqjepa.field_eq`. This adds only what is
    peculiar to the test file: the figures that justify how the guard matches, and the
    two residual bounds quoted in the module-level docstring above.
    """
    field_eq.demo()

    print("== WHY THE GUARD MATCHES TOKENS, NOT SUBSTRINGS ==")
    for probe, host in (("22", "-0.3122"), ("23", "1.6923e-11"), ("27", "0.872749")):
        print("  %s is a substring of %s, so substring matching would pass it "
              "unprinted" % (probe, host))
    print("  the scale invariance is checked directly to 1e-10, not through the")
    print("  1e-05 finite difference")

    print("== THE TWO BOUNDS THIS FILE'S OWN DOCSTRING QUOTES ==")
    w1s = np.ones(len(BED_EDGES))
    w0 = np.array([0.6, 1.4, 1.0, 0.9, 1.1, 1.0, 1.3])
    wb = np.ones(len(BED_EDGES)); wb[BRIDGE] = 0.5
    kr = bed_kappa(length="reciprocal")
    st = field_eq.stability(kr, w1s, sign=+1)
    best = field_eq.run_flow(w0, kr, kr(w1s), eta=0.8 * st.stable_eta_max, steps=200,
                             sign=+1, normalise=True)
    tgt_f = _curvature(wb, BED_EDGES, BED_N).copy(); tgt_f[BRIDGE] += 0.15
    tf = field_eq.run_flow(wb, bed_kappa(metric=_metric(wb, BED_EDGES, BED_N)),
                           tgt_f, eta=0.3, steps=60)
    print("  tightest flow residual        %.4e" % best.residual[-1])
    print("  frozen-metric residual        %.4e   against a recomputed %.4f"
          % (tf.residual[-1], np.abs(bed_kappa()(tf.w) - tgt_f).max()))

    print("== WHY THE PERTURBATION FACTOR IS 1.01 ==")
    sr = field_eq.run_flow(w0, bed_kappa(), bed_kappa()(w1s), eta=0.1, steps=120)
    dip, frozen = min(sr.residual), tf.residual[-1]
    print("  field_eq.demo() asserts its own findings as it prints them. Two bars it")
    print("  would breach under a factor of 2, and clears at 1.01:")
    print("    near-fixed-point dip %.4f must stay under 0.06   -> scaled %.4f"
          % (dip, 1.01 * dip))
    print("    frozen-metric residual %.4e must stay under 1e-8 -> scaled %.4e"
          % (frozen, 1.01 * frozen))
    print("  1.01 is far outside the four-significant-figure print and far inside")
    print("  every bar the demo asserts.")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
        sys.exit(0)
    sys.exit(1 if mutation_report() else 0)

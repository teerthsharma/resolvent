"""RED-FIRST tests for ceqjepa/ricci_flow.py (PHASE C it.2, discrete Ricci flow).

WHAT THESE TESTS ARE FOR. The PHASE C spec names its weight update "the Ni et al.
2019 form" and binds it to Ni et al.'s community separation at zero source. Read
from the PDF of arXiv:1907.03993 (Ni, Lin, Luo, Gao, Sci. Rep. 9), the update in
that paper is

    w^(k+1)_ij = (1 - kappa^(k)_ij) * d^(k)(i, j)                    [their Eq. 10]

with d^(k) the metric of the weighted graph (V, E, w^(k)) -- RECOMPUTED each step
-- discretising Ollivier's d/dt d_ij = -kappa_ij d_ij [their Eq. 9]. The spec's
update is

    w_e <- w_e * (1 - eta (kappa_e - kappa_target(e)))                [the spec]

a rescale of the EXISTING weight, carrying an eta and a target. At zero source
these are two different maps, and every test below is a way for the spec's
identity claim to be false while both maps still return numbers:

  1. ONE RULE WEARING TWO NAMES. If `spec_flow_step` at eta = 1 is bitwise
     `ni_flow_step`, the comparison the whole iteration rests on is vacuous.
  2. EQ. 10 IS NOT EQ. 10. The distance factor is the RECOMPUTED METRIC d(i,j),
     not the incident weight w_ij. On a graph with a shortcut the two differ, and
     an implementation that multiplies by w_ij returns plausible numbers forever.
  3. THE NO-OPS DO NOT HOLD. eta = 0 must be BITWISE identity, and a step on a
     graph already at target must not move. Either failing means the update has a
     drift term nobody asked for.
  4. THE BIND FAILS. Ni's rule must reproduce the community separation, measured
     as ARI against ground truth, ABOVE the control of the identical cut rule with
     no flow. Without that control the cut rule could be doing the work.
  5. THE PIPELINE SEPARATES NOTHING. A single planted community must not shatter
     under either rule. That is the negative control that makes 4 mean anything.
  6. A REFUSAL IS SWALLOWED. Curvature refuses on a disconnected pair; a flow step
     that turns a refusal into NaN weights loses the distinction between "this
     edge has no curvature" and "this edge has curvature zero".

THE IMPORT IS GUARDED ON PURPOSE. A module-level import turns the RED phase into
one collection error with no test names in it; each test must be seen to fail BY
NAME before the module exists.

RUN: python -m pytest tests/curvature/test_ricci_flow.py -v
"""

import itertools
import math
import re

import numpy as np
import pytest

from ceqjepa import curvature as cv       # a finished dependency, not under test

try:
    from ceqjepa import ricci_flow as rf
except Exception as _exc:                 # noqa: BLE001 -- RED phase carries it
    rf = None
    _IMPORT_ERROR = _exc


def _rf():
    """The module under test, or a named failure saying it is not there yet."""
    if rf is None:
        raise AssertionError(
            "ceqjepa/ricci_flow.py did not import: %r" % (_IMPORT_ERROR,))
    return rf


# ---------------------------------------------------------------------------
# Independent oracles. Nothing below this line calls the module under test.
# ---------------------------------------------------------------------------

def eq10_by_hand(W):
    """Ni et al. Eq. 10 recomputed here, edge by edge, from curvature.py only.

    w'_ij = (1 - kappa_ij) * d(i, j), d the shortest-path metric of the WEIGHTED
    graph W. Deliberately written as a naive loop over the upper triangle with no
    shared code with the module under test: if ricci_flow.py has invented its own
    distance, its own laziness, or its own edge ordering, this disagrees.
    """
    W = np.asarray(W, dtype=float)
    D = cv.graph_metric(W)
    out = W.copy()
    for i, j in zip(*np.nonzero(np.triu(W, 1))):
        k = cv.kappa_edge(W, D, int(i), int(j))
        if cv.is_refusal(k):
            continue
        out[i, j] = out[j, i] = (1.0 - float(k)) * float(D[i, j])
    return out


def ari_by_hand(a, b):
    """Adjusted Rand index, from the contingency table, by the definition.

        ARI = (sum_ij C(n_ij,2) - E) / (0.5*(sum_i C(a_i,2) + sum_j C(b_j,2)) - E)
        E   = sum_i C(a_i,2) * sum_j C(b_j,2) / C(n,2)

    Hand-rolled so that the module's own ARI has something to be wrong against.
    """
    a, b = np.asarray(a), np.asarray(b)
    n = a.size
    comb2 = lambda x: x * (x - 1) / 2.0
    ua, ub = np.unique(a), np.unique(b)
    tab = np.array([[np.sum((a == x) & (b == y)) for y in ub] for x in ua],
                   dtype=float)
    sij = comb2(tab).sum()
    si = comb2(tab.sum(axis=1)).sum()
    sj = comb2(tab.sum(axis=0)).sum()
    exp = si * sj / comb2(float(n))
    denom = 0.5 * (si + sj) - exp
    return 1.0 if denom == 0.0 else float((sij - exp) / denom)


def shortcut_graph():
    """A weighted triangle-plus-tail where one edge is NOT a geodesic.

    Edge (0,1) has weight 10 but 0-2-1 costs 1 + 1 = 2, so d(0,1) = 2 != w_01.
    Eq. 10 must use 2. A rescale of w_01 cannot land on the same number.
    """
    W = np.zeros((5, 5))
    for (i, j), w in {(0, 1): 10.0, (0, 2): 1.0, (1, 2): 1.0,
                      (2, 3): 1.0, (3, 4): 1.0, (2, 4): 1.0}.items():
        W[i, j] = W[j, i] = w
    return W


# ---------------------------------------------------------------------------
# 1. ONE RULE WEARING TWO NAMES
# ---------------------------------------------------------------------------

def test_the_two_rules_are_not_the_same_map():
    """Both rules exist as separate functions and disagree on a WEIGHTED graph.

    The bed is the shortcut graph on purpose. DERIVED: on a graph whose weights
    are all 1, d(i,j) = 1 on every edge, so Eq. 10 reads (1 - kappa) * 1 and the
    spec at eta = 1 reads 1 * (1 - kappa) -- the same number. A unit-weight bed
    therefore cannot tell the two maps apart, and the next test pins that
    coincidence rather than letting it hide here.
    """
    m = _rf()
    W = shortcut_graph()
    a = np.asarray(m.ni_flow_step(W)["W"], dtype=float)
    b = np.asarray(m.spec_flow_step(W, eta=1.0, kappa_target=0.0)["W"], dtype=float)
    gap = float(np.abs(a - b).max())
    assert gap > 1e-9, (
        "ni_flow_step and spec_flow_step returned the same weights to %.3e on a "
        "weighted graph: one is silently the other, and the comparison is vacuous"
        % gap)


def test_the_two_rules_coincide_exactly_while_the_graph_is_shortcut_free():
    """THE MECHANISM, pinned: agreement is a function of the shortcut census.

    DERIVED. An edge with d(i,j) = w_ij is its own geodesic. On a graph where
    EVERY edge is (a shortcut-free graph, which a unit-weight graph is), Eq. 10's
    (1 - kappa) d and the spec's eta = 1 rescale w (1 - kappa) are the same
    floating-point product, so the two maps are BITWISE identical -- and stay so
    for as long as the flow keeps the graph shortcut-free. This test asserts the
    biconditional, not the coincidence: agreement holds at exactly the steps
    whose input graph carries no shortcut. If a future change makes them agree on
    a shortcut graph, or disagree on a shortcut-free one, that is a defect in one
    of the two rules and it fails here rather than in a docstring.
    """
    m = _rf()
    W, _ = cv.two_block_bed(n_per_block=12, p=0.35, n_cross=1)
    assert set(np.unique(W)) == {0.0, 1.0}, "the bed stopped being unit-weight"
    a = b = W.copy()
    seen = []
    for _ in range(6):
        D = cv.graph_metric(a)
        ii, jj = np.nonzero(np.triu(a, 1))
        free = int((D[ii, jj] < a[ii, jj] - 1e-12).sum()) == 0
        a = np.asarray(m.ni_flow_step(a)["W"], dtype=float)
        b = np.asarray(m.spec_flow_step(b, eta=1.0, kappa_target=0.0)["W"],
                       dtype=float)
        seen.append((free, float(np.abs(a - b).max())))
    for step, (free, gap) in enumerate(seen, 1):
        if free:
            assert gap == 0.0, (
                "step %d entered shortcut-free and the two rules still differ by "
                "%.3e: the derivation d = w has stopped holding" % (step, gap))
    assert all(free for free, _ in seen), (
        "the bed left the shortcut-free regime during the flow: the recorded "
        "census (0 shortcut edges at every step) must be re-measured")


# ---------------------------------------------------------------------------
# 2. EQ. 10 IS NOT EQ. 10
# ---------------------------------------------------------------------------

def test_ni_step_is_equation_10_verbatim():
    """(1 - kappa) * d, recomputed independently, edge by edge, to 1e-12."""
    m = _rf()
    W, _ = cv.two_block_bed(n_per_block=10, p=0.4, n_cross=2)
    got = np.asarray(m.ni_flow_step(W)["W"], dtype=float)
    want = eq10_by_hand(W)
    err = float(np.abs(got - want).max())
    assert err < 1e-12, "ni_flow_step is not Eq. 10: worst weight error %.3e" % err


def test_ni_step_uses_the_recomputed_metric_not_the_incident_weight():
    """On a graph with a shortcut, d(0,1) = 2 while w_01 = 10. Eq. 10 uses 2."""
    m = _rf()
    W = shortcut_graph()
    D = cv.graph_metric(W)
    assert D[0, 1] == pytest.approx(2.0), "the oracle graph lost its shortcut"
    k = cv.kappa_edge(W, D, 0, 1)
    assert not cv.is_refusal(k)
    want_metric = (1.0 - float(k)) * 2.0
    want_weight = (1.0 - float(k)) * 10.0
    got = float(np.asarray(m.ni_flow_step(W)["W"])[0, 1])
    assert got == pytest.approx(want_metric, abs=1e-12), (
        "ni_flow_step gave %.12g; Eq. 10 with the recomputed metric is %.12g and "
        "with the incident weight is %.12g -- the metric is not being recomputed"
        % (got, want_metric, want_weight))


# ---------------------------------------------------------------------------
# 3. THE NO-OPS
# ---------------------------------------------------------------------------

def test_eta_zero_leaves_every_weight_bitwise_unchanged():
    """Not 'close': BITWISE. A drift at eta = 0 is a term nobody asked for."""
    m = _rf()
    W, _ = cv.two_block_bed(n_per_block=12, p=0.35, n_cross=1)
    out = np.asarray(m.spec_flow_step(W, eta=0.0, kappa_target=0.0)["W"], dtype=float)
    assert out.shape == W.shape
    assert out.tobytes() == W.tobytes(), (
        "eta = 0 moved %d weights, worst by %.3e"
        % (int((out != W).sum()), float(np.abs(out - W).max())))


def test_spec_step_at_target_is_a_noop_to_1e_12():
    """kappa_target set to the measured kappa of every edge: nothing may move."""
    m = _rf()
    W, _ = cv.two_block_bed(n_per_block=12, p=0.35, n_cross=1)
    D = cv.graph_metric(W)
    target = {}
    for i, j in zip(*np.nonzero(np.triu(W, 1))):
        k = cv.kappa_edge(W, D, int(i), int(j))
        if not cv.is_refusal(k):
            target[(int(i), int(j))] = float(k)
    out = np.asarray(m.spec_flow_step(W, eta=0.5, kappa_target=target)["W"],
                     dtype=float)
    err = float(np.abs(out - W).max())
    assert err < 1e-12, (
        "a step on a graph already at target moved a weight by %.3e" % err)


# ---------------------------------------------------------------------------
# 4. THE BIND
# ---------------------------------------------------------------------------

def test_ni_flow_recovers_planted_communities_above_the_no_flow_control():
    """Ni's Eq. 10, then the cut, against the SAME cut with no flow at all."""
    m = _rf()
    W, labels = m.planted_two_community()
    flowed = m.run_flow(W, rule="ni", n_steps=m.N_STEPS)
    after = m.best_cut_ari(flowed["W"], labels)
    control = m.best_cut_ari(W, labels)
    assert after["ari"] == pytest.approx(ari_by_hand(labels, after["pred"]), abs=1e-12), \
        "the module's ARI disagrees with the hand-rolled one"
    assert control["ari"] < 0.5, (
        "the NO-FLOW control already recovers the communities at ARI %.4f: the cut "
        "rule is doing the work and the flow is proving nothing" % control["ari"])
    assert after["ari"] > control["ari"], (
        "Ni's flow (ARI %.4f, %d steps) does not beat no flow at all (ARI %.4f)"
        % (after["ari"], m.N_STEPS, control["ari"]))
    assert after["ari"] >= 0.8, (
        "Ni's community separation did not reproduce: ARI %.4f over %d nodes after "
        "%d steps" % (after["ari"], labels.size, m.N_STEPS))


def test_spec_zero_source_reproduces_the_ni_separation():
    """THE SPEC'S IDENTITY BIND, stated as the spec states it, at zero source."""
    m = _rf()
    W, labels = m.planted_two_community()
    ni = m.best_cut_ari(m.run_flow(W, rule="ni", n_steps=m.N_STEPS)["W"], labels)
    spec = m.best_cut_ari(
        m.run_flow(W, rule="spec", n_steps=m.N_STEPS, eta=m.ETA,
                   kappa_target=0.0)["W"], labels)
    assert spec["ari"] >= ni["ari"] - 0.05, (
        "zero-source spec rule ARI %.4f (eta %g, %d steps) against Ni Eq. 10 ARI "
        "%.4f: the spec's update does NOT reproduce Ni's separation"
        % (spec["ari"], m.ETA, m.N_STEPS, ni["ari"]))


# ---------------------------------------------------------------------------
# 5. THE PLANTED NEGATIVE
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule", ["ni", "spec"])
def test_single_community_does_not_separate(rule):
    """One block, no plant. The cut must not shatter it under either rule."""
    m = _rf()
    W = m.planted_one_community()
    out = m.run_flow(W, rule=rule, n_steps=m.N_STEPS, eta=m.ETA)
    frac = m.largest_component_fraction(out["W"], m.CUT_QUANTILE)
    assert frac >= 0.9, (
        "the %s rule shattered a single community: the largest component holds "
        "%.1f%% of nodes after %d steps" % (rule, 100 * frac, m.N_STEPS))


# ---------------------------------------------------------------------------
# 6. REFUSALS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule", ["ni", "spec"])
def test_a_refusal_is_a_value_and_never_becomes_a_nan_weight(rule):
    """Two disjoint triangles: a step completes, no exception, no NaN, a census.

    MEASURED and worth saying plainly: curvature.py's own refusals do NOT fire
    here. A flow step only visits EDGES, and every edge has two endpoints of
    degree >= 1 inside one component, so self-loop, degenerate-neighbourhood,
    zero-distance and unreachable-support are all unreachable from a well-formed
    weighted graph. The refusals that this module can actually produce are its
    own -- non-positive-weight (next test) and transport-lp-failed -- and the
    contract this test holds is the weaker, still load-bearing one: a
    disconnected graph is not an exception and never becomes a NaN weight.
    """
    m = _rf()
    W = np.zeros((6, 6))
    for i, j in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]:
        W[i, j] = W[j, i] = 1.0
    out = m.ni_flow_step(W) if rule == "ni" else m.spec_flow_step(W, eta=0.5)
    new = np.asarray(out["W"], dtype=float)
    assert np.isfinite(new).all(), "a flow step produced %d non-finite weights" \
                                   % int((~np.isfinite(new)).sum())
    assert (new >= 0.0).all(), "a flow step produced a negative weight"
    assert "refusals" in out, "the step reports no refusal census"
    assert isinstance(out["refusals"], dict)


def test_a_step_that_cannot_be_taken_refuses_with_a_reason():
    """A target that would drive a weight non-positive is a REFUSAL, not a crash."""
    m = _rf()
    W, _ = cv.two_block_bed(n_per_block=10, p=0.4, n_cross=1)
    out = m.spec_flow_step(W, eta=50.0, kappa_target=0.0)
    assert np.isfinite(np.asarray(out["W"], dtype=float)).all()
    assert (np.asarray(out["W"], dtype=float) >= 0.0).all(), \
        "eta = 50 drove weights negative instead of refusing"
    assert out["refusals"], "no refusal recorded for an unrepresentable step"
    for code, count in out["refusals"].items():
        assert isinstance(code, str) and count >= 1


# ---------------------------------------------------------------------------
# 7. THE RESIDUAL IS A COLUMN, NOT A SCALAR
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule", ["ni", "spec"])
def test_the_residual_is_reported_per_step(rule):
    """A final scalar hides a plateau and hides an oscillation. Keep the column."""
    m = _rf()
    W, _ = cv.two_block_bed(n_per_block=10, p=0.4, n_cross=1)
    out = m.run_flow(W, rule=rule, n_steps=5, eta=m.ETA)
    for key in ("residual_abs", "residual_gauge", "kappa_min", "kappa_med",
                "kappa_max", "kappa_spread", "n_edges", "shortcuts",
                "eta_ceiling"):
        assert key in out, "run_flow does not report %r" % key
        assert len(out[key]) == 5, \
            "%r has %d entries for 5 steps" % (key, len(out[key]))
    assert all(np.isfinite(r) and r >= 0.0 for r in out["residual_abs"])
    assert all(n > 0 for n in out["n_edges"]), "the flow lost all its edges"


# ---------------------------------------------------------------------------
# 8. NO NUMBER IN A DOCSTRING THAT NO RUN PRINTS
# ---------------------------------------------------------------------------

#: ANY decimal, and NO exemption list. Copied verbatim from Chase's guard in
#: test_curvature_instrument.py, in the form Cameron widened it to. The narrow
#: earlier version required a sign, a percent, an "x" or an exponent, so a bare
#: decimal in prose walked straight through -- which is the form nearly every
#: number struck this round took, including three hardcoded prices printed as if
#: measured. An exemption list is how a check like this gets defanged, so a
#: legitimate constant that fails here (an arXiv id, a year, a pinned parameter)
#: is fixed by PRINTING it in the run, never by excusing it.
_MEASURED = re.compile(r"[+-]?\d+\.\d+(?:[eE][+-]?\d+)?")


def _module_docstrings(m):
    """(name, docstring) for the module and everything DEFINED in it.

    Filtered on __module__ so numpy's and scipy's docstrings do not come along;
    NOT filtered on __all__, because demo() is public, is the thing that prints
    every number, and is absent from __all__ -- an __all__-scoped scan misses
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


def test_every_measured_number_in_a_docstring_is_printed_by_the_demo(capsys):
    """Every decimal in ricci_flow.py's docstrings must appear in its own run.

    This test is why the plateau-onset error was catchable at all: the sentence
    "floors at 5.5853e-02 and sits there from step 21" contained a step number
    the run never printed beside that value. A number no run prints cannot be
    checked by anyone, including its author, and drifts silently from the moment
    the bed changes.

    It costs a full demo() -- the same ~125 s the module's own RUN line budgets.
    That is the price of the guard and it is cheaper than one wrong number.
    """
    m = _rf()
    m.demo()
    out = capsys.readouterr().out

    docs = _module_docstrings(m)
    missing = [(where, tok) for where, doc in docs
               for tok in _MEASURED.findall(doc) if tok not in out]
    checked = sum(len(_MEASURED.findall(doc)) for _, doc in docs)
    print("\n  %d decimals across %d docstrings, %d missing from the run"
          % (checked, len(docs), len(missing)))
    assert checked >= 60, "only %d decimals found: the regex is not biting" % checked
    assert not missing, ("these docstring numbers are printed by no run:\n    "
                         + "\n    ".join("%s: %s" % (w, t) for w, t in missing))


# ---------------------------------------------------------------------------
# 9. A TINY RESIDUAL IS NOT A CERTIFICATE ON THE WEIGHTS
# ---------------------------------------------------------------------------

def test_weight_error_is_blind_to_the_gauge_and_exact_on_a_known_perturbation():
    """The identity test for the new metric, before anything is measured with it.

    kappa is scale-invariant, so a weight comparison that is NOT gauge-blind
    measures the gauge and not the error -- it would report a large error for a
    perfect recovery scaled by 1000. Two halves: rescaling must give exactly
    zero, and a PLANTED multiplicative perturbation of known size must be
    returned at that size and on the edge it was planted on.
    """
    m = _rf()
    W, _ = cv.two_block_bed(n_per_block=10, p=0.4, n_cross=2)
    for c in (2.0, 13.7, 1000.0):
        err, edge = m.weight_error(c * W, W)
        assert err == pytest.approx(0.0, abs=1e-12), (
            "weight_error is not gauge-blind: scaling by %g reads %.3e" % (c, err))

    ii, jj = np.nonzero(np.triu(W, 1))
    Wp = W.copy()
    i, j = int(ii[3]), int(jj[3])
    Wp[i, j] = Wp[j, i] = W[i, j] * 1.25
    err, edge = m.weight_error(Wp, W)
    # the gauge normalisation spreads the planted 25% over the geometric mean of
    # all n edges, so the surviving ratio on the planted edge is 1.25 / 1.25**(1/n)
    n = int(ii.size)
    want = 1.25 / 1.25 ** (1.0 / n) - 1.0
    assert err == pytest.approx(want, rel=1e-9), (
        "a planted 25%% perturbation on one of %d edges reads %.6e, derived %.6e"
        % (n, err, want))
    assert edge == (i, j), "the error is reported on edge %r, planted on %r" \
                           % (edge, (i, j))


def test_a_tiny_residual_does_not_certify_the_weights():
    """THE HAZARD, on this module's own bed: does converged mean the geometry?

    A flow is run to a converged gauge residual from a PERTURBED copy of its own
    fixed point. The residual says the iteration stopped moving; the weight error
    says whether it stopped in the right place. If the residual is tiny and the
    weight error is not, then "the flow converged" is a statement about kappa
    alone and every convergence number in this module needs the weight error
    printed beside it.

    MY PRIOR WAS WRONG AND THE MEASUREMENT INVERTED IT. I expected Eq. 10 itself
    to converge onto the wrong weights. It does not: it returns to its own fixed
    point, which is isolated modulo gauge and attracting. The hazard is real but
    it belongs to a DIFFERENT arm -- the spec's rule at eta = 2, which converges
    to ten digits onto a graph carrying shortcuts, and therefore onto weights
    that are not Eq. 10's. Both halves are pinned here, because either one
    drifting would put the hazard back in the wrong place.
    """
    m = _rf()
    W, _ = cv.two_block_bed(seed=0, n_per_block=8, p=0.45, n_cross=2)
    out = m.recovery_table(rules=(("ni", None), ("spec", 2.0)), n_steps=200,
                           tol=m.LONG_TOL, sigma=0.30, seed=3, W=W)
    assert out["start_err"] > 0.5, \
        "the planted perturbation is too small to be a test: %.3e" % out["start_err"]
    ni_row, spec_row = out["rows"]

    assert ni_row["resid"] < m.LONG_TOL and ni_row["w_err"] < 1e-6, (
        "Eq. 10 no longer returns to its own fixed point (resid %.3e, weight error "
        "%.4e from a planted %.4e): the attracting-fixed-point finding must be "
        "re-measured" % (ni_row["resid"], ni_row["w_err"], out["start_err"]))

    assert spec_row["resid"] < m.LONG_TOL, \
        "the spec arm did not converge: residual %.3e" % spec_row["resid"]
    assert spec_row["w_err"] > 1e-3, (
        "the spec arm converged to residual %.3e AND weight error %.3e: the hazard "
        "has stopped reproducing on this bed. That is itself a finding -- invert "
        "this assertion to pin it, with the shortcut census beside it."
        % (spec_row["resid"], spec_row["w_err"]))


def test_every_convergence_row_carries_a_weight_error():
    """A convergence number with no weight error beside it is the defect.

    Contract, not a threshold: whatever the answer turns out to be, the table
    that reports a residual must report max|w/w* - 1| in the same row, or the
    reader is being handed the half of the story that always looks good.
    """
    m = _rf()
    out = m.recovery_table(rules=(("ni", None), ("spec", 2.0)),
                           n_steps=40, tol=m.LONG_TOL, sigma=0.30, seed=3)
    assert len(out["rows"]) == 2
    for row in out["rows"]:
        for key in ("name", "steps", "resid", "w_err", "w_err_edge", "kappa_const",
                    "kappa_spread", "n_edges"):
            assert key in row, "the recovery row does not report %r" % key
        assert row["n_edges"] > 0
        assert np.isfinite(row["w_err"])


def test_the_flat_directions_of_the_fixed_point_are_counted():
    """The gauge mode is one flat direction. The question is whether it is alone.

    The log-coordinate Jacobian of the gauge-normalised step is computed at the
    fixed point. Normalisation makes the all-ones (scale) direction exactly flat,
    so J @ 1 must be 0 -- that is the check that the Jacobian is the right
    object. Any OTHER eigenvalue on the unit circle is a second direction the
    flow does not contract, and it is what lets a converged residual sit on the
    wrong weights.

    THE GATE IS PART OF THE TEST. The transport LP has a solver tolerance of its
    own, so a finite difference taken too finely returns noise wearing a
    spectrum. J @ 1 = 0 holds by construction for the gauge-normalised map, so
    its measured size is the honest gate on whether J means anything -- and it is
    checked BEFORE any eigenvalue below it is believed. It is also why the
    Jacobian is taken AT the fixed point: a linearisation somewhere else is not
    the object the flat-direction question is about.
    """
    m = _rf()
    W, _ = cv.two_block_bed(seed=0, n_per_block=6, p=0.50, n_cross=2)
    fp = m.run_flow(m.gauge_normalise(W), "ni", 200, gauge=True, tol=m.LONG_TOL)
    J, edges = m.log_jacobian(fp["W"], rule="ni")
    n = len(edges)
    assert J.shape == (n, n), "the Jacobian is %r for %d edges" % (J.shape, n)
    gate = float(np.abs(J @ np.ones(n)).max())
    assert gate < 5e-2, (
        "J @ 1 = %.3e, not ~0: the gauge direction is not annihilated, so this is "
        "noise and not the Jacobian of the gauge-normalised map" % gate)

    flat = m.flat_modes(J)
    for key in ("n_unit", "n_zero", "spectral_radius", "magnitudes", "tol"):
        assert key in flat, "flat_modes does not report %r" % key
    # The gauge is ANNIHILATED by the normalisation, so it sits at |lambda| ~ 0,
    # not at 1. A mode AT 1 would be a genuine second dimension of the fixed-point
    # set -- the thing that would let a converged residual sit on wrong weights,
    # and the only thing a second normalisation could repair.
    assert flat["n_zero"] >= 1, \
        "the gauge mode is not at zero: %r" % flat["magnitudes"][-3:]
    assert flat["n_unit"] == 0 and flat["spectral_radius"] < 1.0, (
        "a mode on the unit circle appeared (n_unit %d, rho %.6f): the fixed point "
        "is no longer isolated modulo gauge, and a second normalisation would then "
        "be the right repair -- which today's measurement says it is not"
        % (flat["n_unit"], flat["spectral_radius"]))

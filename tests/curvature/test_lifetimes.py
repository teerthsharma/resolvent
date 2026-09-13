"""RED-FIRST tests for ceqjepa/lifetimes.py (A1: every simplex declares a lifetime).

WHAT THESE TESTS ARE FOR. The module under test claims that every created simplex
carries an interval (b, d) on a NAMED filtration axis, that a read consulting it
outside that interval is REFUSED rather than answered, and that the distribution
of those intervals has a survival function whose hazard is recoverable. Each test
below is a way for that claim to be FALSE while the code still returns numbers:

  (a) THE AXIS IS DECORATION. A lifetime that can be built without an axis, or
     compared against a lifetime from another filtration, is a pair of floats
     wearing a label. That is the owner's kill and it is tested first.
  (b) INFINITY IS A BIG NUMBER. An essential class encoded as a big finite float, or as the
     filtration maximum, silently becomes the LONGEST FINITE BAR in every
     downstream statistic, and no aggregate can tell the difference afterwards.
  (c) THE BARCODE IS ASSERTED, NOT COMPUTED. Two complexes here have barcodes
     known BY INSPECTION -- a unit square under Rips, and a 12-ring coned off at
     a known time -- and both oracles are written out in this file with no call
     into the module. A third check runs two independent engines (gudhi and
     ripser) over the same points and reports the worst disagreement.
  (d) THE REFUSAL IS AN EXCEPTION OR A NaN. Either one is invisible to a scorer:
     an exception is caught by a caller's except clause and a NaN passes
     `nan < tol` silently. The refusal must be a VALUE carrying its reason.
  (e) THE POOLED RATE HIDES THE FAILURE. A criterion that refuses everything
     scores 100.00% sensitivity, and one that answers everything scores 100.00%
     specificity. Both are scored beside the real criterion, because a single
     agreement number cannot separate them.
  (f) THE ESSENTIAL CLASSES ARE DROPPED. Dropping them biases every lifetime
     downward. Kaplan-Meier with right-censoring is checked against a
     hand-computed six-observation table, and the bias from dropping is measured.
  (g) THE FILTRATION STEP IS THE ANSWER. The owner's MARS item: coarsening the
     step must MOVE the lifetimes, and the size of that move is reported rather
     than hidden.
  (h) THE NUMBER CAME FROM NO RUN. Every number in the docstrings of BOTH files
     -- decimals and bare integers alike, matched as whole TOKENS so that a
     fragment of a longer number cannot stand in for one -- must appear in the
     output of a run performed here, and every published COUNT and RATE must
     equal an expression recomputed in the same run, so that no printed literal
     satisfies the guard.

THE IMPORT IS GUARDED ON PURPOSE. A module-level `import ceqjepa.lifetimes` turns
the RED phase into one collection error with no test names in it. Each test must
be seen to FAIL BY NAME before the module exists.

RUN: python -m pytest tests/curvature/test_lifetimes.py -v
"""

import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

try:
    from ceqjepa import lifetimes as lf
except Exception as _exc:                      # noqa: BLE001 -- RED phase carries it
    lf = None
    _IMPORT_ERROR = _exc


def _lf():
    """The module under test, or a named failure saying it is not there yet."""
    if lf is None:
        raise AssertionError(
            "ceqjepa/lifetimes.py did not import: %r" % (_IMPORT_ERROR,))
    return lf


# ---------------------------------------------------------------------------
# Independent oracles. Nothing below this line calls the module under test.
# ---------------------------------------------------------------------------

SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

#: The Vietoris-Rips barcode of the unit square, BY INSPECTION. Four points, six
#: pairwise distances: four sides of length 1 and two diagonals of length
#: sqrt(2). At radius 0 there are four components; each side edge merges two of
#: them, so three H0 bars die at 1.0 and one is essential. The four sides close a
#: loop the instant the last of them enters, at 1.0; the first diagonal enters at
#: sqrt(2) and fills it with two triangles, so H1 is the single bar
#: (1.0, sqrt(2)).
SQUARE_H0 = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, math.inf)]
SQUARE_H1 = [(1.0, math.sqrt(2.0))]

#: The barcode of the growing ring, BY INSPECTION. Twelve vertices all present at
#: t = 0; edge (i, i+1) enters at t = i+1 for i = 0..10, so eleven components die
#: at t = 1..11 and one is essential. The closing edge (11, 0) enters at t = 12
#: and creates the loop; a hub vertex and all twelve triangles enter at t = 20 and
#: fill it. H1 is the single bar (12, 20).
RING_N = 12
RING_CLOSE = 12.0
RING_CONE = 20.0
RING_H0 = [(0.0, float(i)) for i in range(1, RING_N)] + [(0.0, math.inf)]
RING_H1 = [(RING_CLOSE, RING_CONE)]


def as_pairs(bars):
    """(birth, death) tuples, sorted, from anything bar-shaped."""
    out = []
    for b in bars:
        birth = getattr(b, "birth", None)
        death = getattr(b, "death", None)
        if birth is None:
            birth, death = float(b[0]), float(b[1])
        out.append((float(birth), float(death)))
    return sorted(out)


def worst_pair_gap(a, b):
    """Worst endpoint disagreement between two equal-length sorted bar lists.

    Infinite endpoints must line up with infinite endpoints; a finite endpoint
    paired against an infinity is reported as infinite disagreement rather than
    silently skipped, because that is exactly the substitution item (b) above is about.
    """
    assert len(a) == len(b), "bar counts differ: %d vs %d" % (len(a), len(b))
    worst = 0.0
    for (b0, d0), (b1, d1) in zip(a, b):
        worst = max(worst, abs(b0 - b1))
        if math.isinf(d0) != math.isinf(d1):
            return math.inf
        if math.isfinite(d0):
            worst = max(worst, abs(d0 - d1))
    return worst


def km_by_hand():
    """Kaplan-Meier on six observations, computed here by hand.

    Events at 1, 2, 2, 5; right-censored at 3 and 6. The risk set starts at 6.

        t = 1   at risk 6, 1 event   S = 5/6           = 0.8333333333333334
        t = 2   at risk 5, 2 events  S = (5/6)(3/5)    = 0.5
        t = 3   censored, no drop    S                 = 0.5
        t = 5   at risk 2, 1 event   S = (1/2)(1/2)    = 0.25
        t = 6   censored, no drop    S                 = 0.25

    A censored observation stays in the risk set up to its own time and then
    leaves without an event -- that is the whole difference between this and the
    empirical survival of the four uncensored durations, which would read
    S(5) = 0.0.
    """
    durations = [1.0, 2.0, 2.0, 5.0]
    censored = [3.0, 6.0]
    checkpoints = [(0.5, 1.0), (1.0, 5.0 / 6.0), (1.5, 5.0 / 6.0), (2.0, 0.5),
                   (3.0, 0.5), (4.9, 0.5), (5.0, 0.25), (7.0, 0.25)]
    return durations, censored, checkpoints


# ---------------------------------------------------------------------------
# 1. THE OWNER'S KILL: a lifetime without its axis is not a lifetime
# ---------------------------------------------------------------------------

def test_a_lifetime_without_an_axis_refuses_to_exist():
    """The kill, made mechanical.

    A malformed lifetime must not exist at all, so this one refusal is an
    exception and not a value: a Refusal returned from a constructor would have
    to be stored somewhere, and every consumer downstream would then need to
    re-check what it was handed. Refusals-as-values are for READS, which is
    where a scorer can count them.
    """
    m = _lf()
    with pytest.raises(TypeError):
        m.Lifetime(0.0, 1.0)                       # no axis at all
    for bad in (None, "scale", 0, ("scale", 1.0)):
        with pytest.raises(TypeError):
            m.Lifetime(0.0, 1.0, bad)
    ok = m.Lifetime(0.0, 1.0, m.SCALE)
    print("\n  %r" % (ok,))
    assert ok.axis is m.SCALE
    assert "scale" in repr(ok), "the repr hides the axis: %r" % (ok,)


def test_a_lifetime_carries_the_filtration_step_of_its_axis():
    """Every reported lifetime number must be readable beside its step.

    Without the step on the axis, the MARS finding in test 7 has nothing to be
    reported against: two barcodes with different steps look like the same
    barcode with different numbers.
    """
    m = _lf()
    for axis in m.AXES:
        assert isinstance(axis.name, str) and axis.name
        assert float(axis.step) > 0.0, "axis %r has no filtration step" % (axis,)
    names = sorted(a.name for a in m.AXES)
    print("\n  axes %r" % (names,))
    assert names == ["scale", "temperature", "time"]


def test_lifetimes_on_different_axes_refuse_to_be_compared():
    """A scale bar of 0.3 is not shorter than a time bar of 8.

    Ordering across axes is a category error that produces a perfectly ordinary
    boolean, so nothing downstream can catch it. Equality across axes is FALSE
    rather than an error, because containers call __eq__ and a raising __eq__
    breaks `in`, `set` and `dict`.
    """
    m = _lf()
    a = m.Lifetime(0.0, 0.3, m.SCALE)
    b = m.Lifetime(0.0, 8.0, m.TIME)
    for op in (lambda: a < b, lambda: a > b, lambda: a <= b, lambda: a >= b):
        with pytest.raises(m.AxisMismatch):
            op()
    assert a != b and not (a == b), "cross-axis equality must be False, not an error"
    assert len({a, b}) == 2, "two axes collapsed into one hash bucket"
    same = m.Lifetime(0.0, 0.9, m.SCALE)
    assert a < same, "same-axis ordering broke"
    print("\n  %r versus %r" % (a, b))
    print("  cross-axis comparison raised %s; same-axis order holds"
          % m.AxisMismatch.__name__)


def test_an_essential_class_is_infinite_and_never_a_large_finite_number():
    """d = infinity is representable, and is not the filtration maximum.

    An essential class written as t_max is the longest FINITE bar in every
    aggregate that follows, and once written there is no way to recover which
    bars were essential.
    """
    m = _lf()
    ess = m.Lifetime(0.0, math.inf, m.SCALE)
    fin = m.Lifetime(0.0, 2.5, m.SCALE)
    assert ess.is_essential and not fin.is_essential
    assert math.isinf(ess.death) and math.isinf(ess.duration)
    assert ess.duration > 1e300, "an essential class is comparable to a big float"
    assert ess.censored_duration(2.5) == 2.5, "censoring at t_max did not clip"
    print("\n  essential duration %r, censored at 2.5 -> %r"
          % (ess.duration, ess.censored_duration(2.5)))
    with pytest.raises(ValueError):
        m.Lifetime(1.0, 0.5, m.SCALE)              # death before birth
    with pytest.raises(ValueError):
        m.Lifetime(math.inf, math.inf, m.SCALE)    # inf - inf = NaN duration
    with pytest.raises(ValueError):
        m.Lifetime(float("nan"), 1.0, m.SCALE)


# ---------------------------------------------------------------------------
# 2. THE BARCODE IS COMPUTED, AND TWO ORACLES SAY SO
# ---------------------------------------------------------------------------

def test_the_square_barcode_matches_the_one_computed_by_inspection():
    """The unit square under Rips, against SQUARE_H0/SQUARE_H1 above."""
    m = _lf()
    bc = m.square_bed()
    assert bc.axis is m.SCALE, "the square bed is a SCALE filtration"
    for dim, oracle in ((0, SQUARE_H0), (1, SQUARE_H1)):
        got = as_pairs(bc.in_dimension(dim))
        gap = worst_pair_gap(got, sorted(oracle))
        print("\n  square H%d: %r vs inspection %r -> worst %.3e"
              % (dim, got, sorted(oracle), gap))
        assert gap < 1e-12, "H%d disagrees with inspection by %r" % (dim, gap)
    assert bc.n_simplices > 0, "the barcode reports no simplices behind it"


def test_the_ring_barcode_matches_the_one_computed_by_inspection():
    """A time filtration whose every bar endpoint was chosen by hand.

    This is the axis check the square cannot make: the ring is not a Rips complex
    of any point cloud, its filtration values are TIMES, and the loop's birth and
    death were written into the construction.
    """
    m = _lf()
    bc = m.ring_bed()
    assert bc.axis is m.TIME
    for dim, oracle in ((0, RING_H0), (1, RING_H1)):
        got = as_pairs(bc.in_dimension(dim))
        gap = worst_pair_gap(got, sorted(oracle))
        print("\n  ring H%d: %d bars, worst gap %.3e" % (dim, len(got), gap))
        assert gap < 1e-12, "H%d disagrees with inspection: %r vs %r" % (dim, got, sorted(oracle))
    loop = as_pairs(bc.in_dimension(1))[0]
    assert loop == (RING_CLOSE, RING_CONE)


def test_gudhi_and_ripser_produce_the_same_scale_barcode():
    """Two engines, one bed. The worst disagreement is the number reported.

    Agreement between two implementations of the same definition is not proof of
    correctness -- both could be wrong the same way -- which is why the two
    inspection oracles above come first. This catches the other failure: a
    barcode that is an artefact of one library's construction.
    """
    m = _lf()
    X = m.scale_bed()
    a = m.rips_barcode(X, m.SCALE_THRESH, m.SCALE)
    b = m.ripser_barcode(X, m.SCALE_THRESH, m.SCALE)
    worst = 0.0
    for dim in (0, 1):
        pa, pb = as_pairs(a.in_dimension(dim)), as_pairs(b.in_dimension(dim))
        assert len(pa) == len(pb), \
            "H%d bar counts differ: gudhi %d, ripser %d" % (dim, len(pa), len(pb))
        gap = worst_pair_gap(pa, pb)
        print("\n  H%d: %d bars each, worst endpoint gap %.3e" % (dim, len(pa), gap))
        worst = max(worst, gap)
    assert worst < 1e-5, "the two engines disagree by %r: one of them is not Rips" % worst
    assert worst > 0.0, "identical to the last bit -- is ripser actually being called?"


def test_every_bar_is_a_lifetime_on_the_barcodes_own_axis():
    """A Barcode that hands out bare pairs has lost the kill.

    Every bar must come back as a Lifetime carrying the barcode's axis, so a
    caller that mixes two barcodes gets the AxisMismatch instead of a number.
    """
    m = _lf()
    for bc in (m.square_bed(), m.ring_bed(), m.temperature_bed()):
        for dim, bar in bc.bars:
            assert isinstance(bar, m.Lifetime), "H%d bar is %r" % (dim, bar)
            assert bar.axis is bc.axis
    sq = m.square_bed().in_dimension(1)[0]
    rg = m.ring_bed().in_dimension(1)[0]
    with pytest.raises(m.AxisMismatch):
        sq < rg
    print("\n  square-H1 vs ring-H1 comparison refused across scale/time")


# ---------------------------------------------------------------------------
# 3. T-DEAD: the refusal is a value
# ---------------------------------------------------------------------------

def test_a_read_outside_the_interval_is_a_refusal_value_with_reason_expired():
    """Not an exception, not a NaN, not a zero."""
    m = _lf()
    lt = m.Lifetime(1.0, 3.0, m.SCALE)
    inside = m.read_at(lt, 2.0, m.SCALE, 42.0)
    assert inside == 42.0 and not m.is_refusal(inside)

    for tau in (0.5, 3.0, 9.0):
        got = m.read_at(lt, tau, m.SCALE, 42.0)
        assert m.is_refusal(got), "a dead read answered %r at tau=%r" % (got, tau)
        assert got.code == m.EXPIRED, "wrong reason code %r" % (got.code,)
        assert isinstance(got.reason, str) and got.reason
        with pytest.raises(TypeError):
            _ = got < 1.0                      # a threshold on a refusal must fail loudly
        assert not (isinstance(got, float) and math.isnan(got))
    print("\n  %r" % (m.read_at(lt, 9.0, m.SCALE, 42.0),))

    assert m.read_at(lt, 1.0, m.SCALE, 42.0) == 42.0, "the interval must be half-open [b, d)"
    assert m.is_refusal(m.read_at(lt, 3.0, m.SCALE, 42.0)), "d must not be inside [b, d)"


def test_a_read_on_the_wrong_axis_is_refused_and_not_silently_converted():
    """tau = 8 means nothing to a scale bar even though 8 is a fine float."""
    m = _lf()
    lt = m.Lifetime(1.0, 3.0, m.SCALE)
    got = m.read_at(lt, 8.0, m.TIME, 42.0)
    assert m.is_refusal(got) and got.code == m.AXIS_MISMATCH, \
        "a cross-axis read returned %r" % (got,)
    nul = m.read_at(lt, float("nan"), m.SCALE, 42.0)
    assert m.is_refusal(nul) and nul.code == m.NULL_QUERY, \
        "a NaN query returned %r" % (nul,)
    print("\n  wrong axis -> %r ; NaN tau -> %r" % (got, nul))


def test_the_3x2_score_reports_sensitivity_and_specificity_separately():
    """And the two trivial criteria are scored beside the real one.

    The real criterion must be exact on a bed built from the intervals
    themselves; refuse-everything must reach 100.00% sensitivity at 0.00%
    specificity and answer-everything the mirror of that. A pooled agreement rate cannot
    tell any of the three apart on a bed whose class balance is chosen freely,
    which is the whole reason both numbers are demanded.
    """
    m = _lf()
    bc = m.scale_barcode()
    cases = m.t_dead_cases(bc, seed=0)
    counts = {}
    for _, _, _, truth in cases:
        counts[truth] = counts.get(truth, 0) + 1
    print("\n  %d cases: %r" % (len(cases), sorted(counts.items())))
    for cls in (m.DEFINED, m.UNDEFINED, m.NULL):
        assert counts.get(cls, 0) > 0, "class %r is empty: it cannot discriminate" % (cls,)

    real = m.score_3x2(cases, m.read_criterion)
    assert real["sensitivity"] == 1.0 and real["specificity"] == 1.0, \
        "the read is not exact on a bed derived from its own intervals: %r" % (real,)
    assert real["table"][m.DEFINED]["refused"] == 0
    assert real["table"][m.UNDEFINED]["answered"] == 0
    assert real["table"][m.NULL]["answered"] == 0

    allr = m.score_3x2(cases, m.refuse_everything)
    alla = m.score_3x2(cases, m.answer_everything)
    print("  real  sens %.4f spec %.4f" % (real["sensitivity"], real["specificity"]))
    print("  refuse-everything sens %.4f spec %.4f" % (allr["sensitivity"], allr["specificity"]))
    print("  answer-everything sens %.4f spec %.4f" % (alla["sensitivity"], alla["specificity"]))
    assert (allr["sensitivity"], allr["specificity"]) == (1.0, 0.0)
    assert (alla["sensitivity"], alla["specificity"]) == (0.0, 1.0)
    assert "pooled" not in real or real["pooled"] is None, \
        "a pooled agreement rate is exactly what these three numbers exist to prevent"


# ---------------------------------------------------------------------------
# 4. PLANTED NEGATIVES, each seen to fire
# ---------------------------------------------------------------------------

def test_a_class_alive_across_the_whole_filtration_is_never_refused():
    """Planted negative 1: the global relation. Always evidence, at every tau."""
    m = _lf()
    lt = m.Lifetime(0.0, math.inf, m.SCALE)
    taus = np.linspace(0.0, 1e6, 501)
    refused = [t for t in taus if m.is_refusal(m.read_at(lt, float(t), m.SCALE, 1.0))]
    print("\n  essential bar refused on %d of %d queries out of to tau=1e6"
          % (len(refused), len(taus)))
    assert not refused, "an essential class was refused at %r" % (refused[:3],)


def test_a_class_dead_before_the_query_is_always_refused():
    """Planted negative 2: the expired relation. Never evidence, at any tau past d."""
    m = _lf()
    lt = m.Lifetime(0.0, 1.0, m.SCALE)
    taus = np.linspace(1.0, 500.0, 500)
    answered = [t for t in taus if not m.is_refusal(m.read_at(lt, float(t), m.SCALE, 1.0))]
    print("\n  dead bar answered on %d of %d queries" % (len(answered), len(taus)))
    assert not answered, "a dead class answered at %r" % (answered[:3],)


def test_coarsening_the_filtration_step_moves_the_lifetimes():
    """The owner's MARS item, measured rather than hidden.

    If the step could be coarsened without moving a bar, the barcode would be a
    property of the data alone and any reported lifetime would be safe to quote
    without its step. It is not, so it is not: the shift is reported beside the
    step that produced it, and this test fails if a coarsening is ever silently
    absorbed.
    """
    m = _lf()
    X = m.scale_bed()
    fine = m.rips_barcode(X, m.SCALE_THRESH, m.SCALE)
    base = as_pairs(fine.in_dimension(1))
    seen = []
    for step in m.STEP_SWEEP:
        coarse = m.rips_barcode(X, m.SCALE_THRESH, m.SCALE, step=step)
        got = as_pairs(coarse.in_dimension(1))
        shift = worst_pair_gap(got, base) if len(got) == len(base) else math.inf
        seen.append((step, len(got), shift))
        print("\n  step %.4g -> %d H1 bars, worst shift %r" % (step, len(got), shift))
    moved = [s for s in seen if s[2] > 0.0]
    assert moved, "no coarsening moved a single bar: the step sweep is not biting"
    assert max(s[2] for s in seen if math.isfinite(s[2])) > 0.01, \
        "the largest shift is under one hundredth of a unit: report a real sweep"


# ---------------------------------------------------------------------------
# 5. SURVIVAL AND HAZARD
# ---------------------------------------------------------------------------

def test_kaplan_meier_matches_the_hand_computed_table():
    """Six observations, four events, two censored -- oracle in km_by_hand()."""
    m = _lf()
    durations, censored, checkpoints = km_by_hand()
    est = m.km(durations, censored)
    print("\n  risk table the estimator used (t, at risk, events):")
    for t, nr, ne in zip(est.t, est.n_risk, est.n_event):
        print("    t = %g   at risk %d   events %d" % (t, nr, ne))
    for tau, want in checkpoints:
        got = m.s_at(est, tau)
        print("  S(%.3g) = %r, hand %r" % (tau, got, want))
        assert abs(got - want) < 1e-12, "S(%r) = %r, hand-computed %r" % (tau, got, want)
    assert m.s_at(est, 0.0) == 1.0, "S(0) must be 1"

    # The control the docstring names: the same four events with the two
    # censored observations DELETED reach S(5) = 0.0, where censoring holds
    # 0.25. That is the whole difference the estimator is carrying.
    dropped = m.s_at(m.km(durations), 5.0)
    print("  censored S(5) = %r against dropped S(5) = %r"
          % (m.s_at(est, 5.0), dropped))
    assert dropped == 0.0, "the uncensored sample did not run out: %r" % (dropped,)


def test_dropping_essential_classes_biases_every_lifetime_downward():
    """The bias is measured, not asserted away.

    Right-censoring the essential classes at t_max is the honest floor: they are
    known to last AT LEAST that long. Dropping them entirely removes the longest
    observations from the sample, so the mean can only fall.
    """
    m = _lf()
    bc = m.scale_barcode()
    with_ess = m.censoring_bias(bc)
    print("\n  censored mean %.6f, finite-only mean %.6f, bias %.6f over %d bars (%d essential)"
          % (with_ess["censored_mean"], with_ess["dropped_mean"], with_ess["bias"],
             with_ess["n_bars"], with_ess["n_essential"]))
    assert with_ess["n_essential"] > 0, "no essential class on this bed: the test is vacuous"
    assert with_ess["bias"] > 0.0, \
        "dropping the essential classes did not lower the mean: %r" % (with_ess,)
    assert with_ess["censored_mean"] > with_ess["dropped_mean"]


def test_the_hazard_integral_reproduces_the_survival():
    """S(tau) = exp(-integral_0^tau h) on this module's own sample.

    Two agreements are demanded, because they fail for different reasons. The
    product-limit against the exponential of the cumulative hazard is an
    ESTIMATOR gap, order 1/n, and it must shrink when n grows. The quadrature of
    -S'/S back to S is an ARITHMETIC gap on a smooth control where the true
    hazard is a constant that must be recovered.
    """
    m = _lf()
    small = m.identity_check(n=60, rate=1.3, seed=0)
    large = m.identity_check(n=600, rate=1.3, seed=0)
    print("\n  n=60  km-vs-exp(-H) %.6f, quadrature %.6f, rate hat %.4f"
          % (small["km_vs_na"], small["quadrature"], small["rate_hat"]))
    print("  n=600 km-vs-exp(-H) %.6f, quadrature %.6f, rate hat %.4f"
          % (large["km_vs_na"], large["quadrature"], large["rate_hat"]))
    assert large["km_vs_na"] < small["km_vs_na"], \
        "the estimator gap did not shrink with n: it is not an estimator gap"
    assert large["quadrature"] < 0.05, "the identity does not close: %r" % (large,)
    assert abs(large["rate_hat"] - 1.3) < 0.2, \
        "the planted hazard rate 1.3 was not recovered: %r" % (large["rate_hat"],)

    real = m.survival_report(m.scale_barcode())
    print("  scale bed: km-vs-exp(-H) %.6f over %d durations"
          % (real["km_vs_na"], real["n"]))
    assert real["km_vs_na"] < 0.5


# ---------------------------------------------------------------------------
# 6. THE TWO GUARDS THIS ROUND LEARNED THE HARD WAY
# ---------------------------------------------------------------------------

#: A number WITH ITS BOUNDARIES. The lookarounds are the repair for the vacuity
#: the Inspector struck: `"6.11911" in out` is TRUE the moment the run prints the
#: arXiv id 2606.11911, so a substring test passes any fabricated number that
#: happens to share digits with a real one -- which is exactly what a stale
#: number looks like. Matching TOKEN-to-TOKEN instead means a docstring number
#: must appear in the run as a whole number, not as a fragment of another.
#: The inner (?:\.\d+)* keeps a dotted version string (3.12.0, 0.6.14) as ONE
#: token on both sides rather than splitting it into a spurious 3.12.
#: The trailing guard is (?!\w)(?!\.\d) and not (?![\w.]): a sentence-final
#: period must not hide the number in front of it -- "ripser 0.6.14." printed at
#: the end of a line was missed by the stricter form, and a guard that misses the
#: run's own numbers is the same defect pointing the other way.
_MEASURED = re.compile(
    r"(?<![\w.])[+-]?\d+(?:\.\d+)*(?:[eE][+-]?\d+)?(?!\w)(?!\.\d)")


def _tokens(text):
    """Every number in `text` as a boundary-anchored token.

    A leading plus is stripped, so a docstring quoting a number bare still
    matches a run that prints it signed. A leading MINUS is not, because a
    negative number is a different number and silently matching it across the
    sign would put the defect straight back.
    """
    return {t.lstrip("+") for t in _MEASURED.findall(text)}


#: Set in the child pytest process so the own-file scan does not recurse.
_CHILD = "LIFETIMES_GUARD_CHILD"


def _own_test_run_output():
    """stdout of THIS file's own tests, so numbers in THEIR docstrings are bound.

    No guard in this round scanned the file it lives in, so a measured number
    written into a TEST docstring -- the hand-computed Kaplan-Meier table below,
    say -- was bound to nothing at all and could drift without a single failure.
    It is bound here to the run that prints it, which is this file under `-s`,
    fetched in a child process because a test cannot capture what its siblings
    printed. The env var stops the child from recursing into this test.
    """
    env = dict(os.environ)
    env[_CHILD] = "1"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(Path(__file__).resolve()),
         "-q", "-s", "-p", "no:cacheprovider"],
        capture_output=True, text=True, env=env,
        cwd=str(Path(__file__).resolve().parents[2]))
    assert proc.stdout, "the child run printed nothing: %r" % (proc.stderr[-400:],)
    return proc.stdout


def _module_docstrings(m):
    """(name, docstring) for the module and everything DEFINED in it.

    Filtered on __module__ so numpy's and gudhi's docstrings do not come along;
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
        for attr in sorted(vars(obj)) if isinstance(obj, type) else ():
            member = getattr(obj, attr, None)
            member = member.fget if isinstance(member, property) else member
            # AUTHORED docstrings only. namedtuple writes "Alias for field
            # number 4" for every field accessor and collections owns those
            # integers, not this file; the same __module__ filter that keeps
            # numpy out at module level keeps them out here. This is a scope
            # rule, not an exemption list -- no NUMBER is ever exempted.
            if getattr(member, "__module__", None) != m.__name__:
                continue
            sub = getattr(member, "__doc__", None)
            if isinstance(sub, str) and sub.strip() and attr != "__doc__":
                docs.append(("%s.%s" % (name, attr), sub))
    return docs


def test_every_measured_number_in_a_docstring_is_printed_by_the_demo(capsys):
    """No exemption list. A number no run prints cannot be checked by anyone."""
    if os.environ.get(_CHILD):
        pytest.skip("child of the own-file scan; the parent does the scanning")
    m = _lf()
    m.demo()
    out = capsys.readouterr().out
    run = _tokens(out) | _tokens(_own_test_run_output())
    assert "2606.11911" in run, "the run stopped printing the arXiv id"
    assert "6.11911" not in run, \
        "SUBSTRING VACUITY: a fragment of 2606.11911 is being read as a number"
    assert "77.31" not in run, "the absent-decimal control was found in the run"

    docs = _module_docstrings(m) + _module_docstrings(sys.modules[__name__])
    missing = [(where, tok) for where, doc in docs
               for tok in _tokens(doc) if tok not in run]
    checked = sum(len(_tokens(doc)) for _, doc in docs)
    print("\n  %d numbers across %d docstrings, %d missing from the run"
          % (checked, len(docs), len(missing)))
    assert checked >= 55, "only %d numbers found: the regex is not biting" % checked
    assert "40317" not in run, "the absent-integer control was found in the run"
    assert "36050" in run, "a measured integer stopped being matched as a token"
    assert not missing, ("these docstring numbers are printed by no run:\n    "
                         + "\n    ".join("%s: %s" % (w, t) for w, t in missing))


def test_the_published_counts_and_rates_are_recomputed_not_literals(capsys):
    """PRINTED IS NOT MEASURED.

    A literal typed into a print statement passes the docstring guard above
    trivially. So every published count and rate is pinned HERE to an expression
    that recomputes it from the module's own functions in this same run: a
    hand-typed constant fails unless it happens to equal the recomputation, at
    which point it is no longer a fabrication.
    """
    m = _lf()
    stats = m.demo()
    out = capsys.readouterr().out
    assert isinstance(stats, dict) and stats, "demo() must return what it measured"

    bc = m.scale_barcode()
    fin = [b for _, b in bc.bars if not b.is_essential]
    ess = [b for _, b in bc.bars if b.is_essential]
    assert stats["scale_n_finite"] == len(fin) > 0
    assert stats["scale_n_essential"] == len(ess) > 0
    assert stats["scale_n_simplices"] == bc.n_simplices > 0

    cases = m.t_dead_cases(bc, seed=0)
    real = m.score_3x2(cases, m.read_criterion)
    assert stats["t_dead_n_cases"] == len(cases)
    assert stats["t_dead_sensitivity"] == real["sensitivity"]
    assert stats["t_dead_specificity"] == real["specificity"]

    bias = m.censoring_bias(bc)
    assert stats["censoring_bias"] == bias["bias"]

    rep = m.survival_report(bc)
    assert stats["surviving_mass"] == m.s_at(rep["km"], bc.t_max) > 0.0

    X = m.scale_bed()
    rip = m.ripser_barcode(X, m.SCALE_THRESH, m.SCALE)
    engine = max(max(abs(u.birth - v.birth),
                     0.0 if u.is_essential else abs(u.death - v.death))
                 for dim in (0, 1)
                 for u, v in zip(bc.in_dimension(dim), rip.in_dimension(dim)))
    assert stats["engine_worst"] == engine > 0.0

    small = m.identity_check(n=60)
    big = m.identity_check(n=600)
    assert stats["identity_quadrature"] == big["quadrature"]
    assert stats["identity_rate_hat"] == big["rate_hat"]
    assert stats["fall_estimator_pct"] == 100.0 * (1.0 - big["km_vs_na"] / small["km_vs_na"])
    assert stats["fall_quadrature_pct"] == 100.0 * (1.0 - big["quadrature"] / small["quadrature"])
    assert stats["fall_quadrature_pct"] > 0.0,         "BOTH gaps fall; a published claim that one does not is prose, not a measurement"

    base = bc.in_dimension(1)
    shifts = []
    for step in m.STEP_SWEEP:
        h1 = m.rips_barcode(X, m.SCALE_THRESH, m.SCALE, step=step).in_dimension(1)
        shifts.append(max(max(abs(u.birth - v.birth), abs(u.death - v.death))
                          for u, v in zip(h1, base))
                      if len(h1) == len(base) else math.inf)
    assert stats["mars_worst_shift"] == max(shifts)
    assert stats["planted_alive_refusals"] == 0 and stats["planted_dead_answers"] == 0

    for key in ("scale_n_finite", "scale_n_essential", "scale_n_simplices",
                "t_dead_n_cases"):
        assert str(stats[key]) in out, "%s = %r is not in the run's output" % (key, stats[key])
    for key in ("censoring_bias", "surviving_mass", "identity_quadrature"):
        assert ("%.6f" % stats[key]) in out or ("%.4f" % stats[key]) in out, \
            "%s = %r is not printed" % (key, stats[key])
    assert ("%.3e" % stats["engine_worst"]) in out
    for key in ("mars_worst_shift", "identity_rate_hat"):
        assert ("%.4f" % stats[key]) in out, "%s = %r is not printed" % (key, stats[key])
    for key in ("fall_estimator_pct", "fall_quadrature_pct"):
        assert ("%.2f" % stats[key]) in out, "%s = %r is not printed" % (key, stats[key])
    print("\n  %d published values re-derived against the module in this run" % len(stats))


def test_the_demo_finishes_inside_the_budget_and_ends_with_the_exact_line(capsys):
    """The budget is 180 s on CPU, and the last line is fixed by contract."""
    m = _lf()
    t0 = time.time()
    m.demo()
    dt = time.time() - t0
    out = capsys.readouterr().out.rstrip("\n")
    print("\n  demo() took %.2f s" % dt)
    assert dt < 180.0, "demo() took %.1f s" % dt
    assert out.endswith("ALL SELF-CHECKS PASSED"), \
        "the last line is %r" % (out.splitlines()[-1] if out else "",)
    assert "RUN:" in (m.__doc__ or ""), "the module docstring carries no RUN: line"
    for name in m.__all__:
        assert hasattr(m, name), "__all__ advertises a missing name %r" % (name,)

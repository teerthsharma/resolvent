"""RED-FIRST tests for ceqjepa/verb_grade.py (A2, the three grades, and T-VERB).

WHAT THESE TESTS ARE FOR. The module under test claims a THIRD GRADE on every
token: H(next | verb), the verb's consequence entropy. Two things can be wrong
with that grade before any experiment is run, and both are fatal:

  1. IT IS ZERO BY CONSTRUCTION. On this space the successor map is a FUNCTION --
     one move leads to exactly one state -- so H(next state | move) is identically
     0 for every move. Read that way the grade is not small, it is empty. The
     module must SAY so and must not measure around it.
  2. IT IS THE OUT-DEGREE WEARING A HAT. Under a uniform policy on this space no
     two legal moves from a position collapse to the same successor, so
     H(next | position) = ln(out-degree) EXACTLY, everywhere. Any grade that is a
     monotone function of out-degree has the same rank statistics as out-degree
     against every event, identically -- so it cannot beat out-degree, and that
     is arithmetic, not a result. Only a grade over an ABSTRACTED verb, where
     many concrete moves share a word, escapes it.

So the module owes three readings side by side, all exact on the enumerated
space, and an honest statement of which one carries information:

  (a) H(next state | position) under the uniform policy -- exact, equals ln(deg),
      the degenerate-but-exact baseline.
  (b) H(state two ply out | concrete verb) -- the opponent's reply. The mover's
      own move is deterministic; what comes back is not. Also exactly ln(deg), of
      the SUCCESSOR, so degenerate in the same way one ply later.
  (c) H(next verb | verb TYPE), the verb abstracted into a class so that many
      concrete moves share a word. Fan-out comes from the abstraction, which is
      what a token-level grade looks like on language. The abstraction must be
      STATED, and it is scored at two widths so the reader can see the grade move
      with it.

The rest of the file is the usual way for a number to be wrong while the code
still prints one: the statistic is not an entropy (planted deterministic and
uniform verbs), the estimator's error against the exact value is never measured,
T-VERB has no control or an unpaired one, and a number in prose came from no run.

THE IMPORT IS GUARDED ON PURPOSE, matching tests/curvature/test_drift_null.py: a
module-level import turns the RED phase into one collection error with no test
names in it, and each test below must be seen to FAIL BY NAME.

WHICH TESTS WENT RED, AND WHICH DID NOT. Two RED phases were run and recorded:
the first contract failed 26 of 26 by name, and the revised contract -- rewritten
after the three-readings correction -- failed 21 and passed 15, 36 tests. This
file ships 41. Five tests were therefore written AFTER the last recorded RED and
never went red as pytest tests. They are marked WRITTEN-AFTER, UNBOUND in their
own docstrings and listed here so the arithmetic reconciles without being chased:

    test_a_verb_type_whose_occurrences_all_end_the_stream_is_a_refusal_not_a_zero
    test_the_component_of_the_bound_is_log_degree_only_where_the_words_are_unique
    test_identity_the_binned_rates_are_recomputed_from_the_event_counts
    test_the_coverage_figure_is_itself_bound_by_two_independent_counters
    test_the_guard_rejects_a_fragment_of_a_printed_number

The last of the five carries its own pre-repair measurement instead of a red
phase: the four fragments it plants were each measured passing the substring
guard this file shipped first, and those measurements are quoted at _MEASURED.
No RED is retro-fitted for any of the five. A red invented after the green proves
nothing, and the honest record is that these five are unbound.

RUN: python -m pytest tests/curvature/test_verb_grade.py -v
"""

import math
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]

try:
    from ceqjepa import verb_grade as vg
except Exception as _exc:                      # noqa: BLE001 -- RED phase carries it
    vg = None
    _IMPORT_ERROR = _exc


def _vg():
    """The module under test, or a named failure saying it is not there yet."""
    if vg is None:
        raise AssertionError(
            "ceqjepa/verb_grade.py did not import: %r" % (_IMPORT_ERROR,))
    return vg


#: The owner's lineage triple for the entropy grade, in nats. Quoted, never used
#: as an expected value: this bed has its own alphabets and reports its own.
OWNER_TRIPLE = (0.17, 0.69, 1.39)

#: The two abstraction widths the type-level grade is read at.
ABSTRACTIONS = ("coarse", "fine")


# ---------------------------------------------------------------------------
# 1. PLANTED NEGATIVES ON THE STATISTIC ITSELF. No bed, no enumeration: these run
#    in milliseconds and are the first thing to go red when the entropy is wrong.
# ---------------------------------------------------------------------------

def test_a_deterministic_verb_reads_exactly_zero_nats():
    """One successor, all the mass. Anything but a hard 0.0 is not an entropy."""
    m = _vg()
    for counts in ([7.0], [1], [0.0, 0.0, 5.0, 0.0]):
        h = m.verb_entropy_nats(np.asarray(counts, float))
        assert h == 0.0, ("a deterministic verb read %r nats, not 0.0, from %r"
                          % (h, counts))


def test_a_uniform_verb_over_d_successors_reads_log_d_to_1e_12():
    """The other end of the same statistic, and the one a normalisation bug moves."""
    m = _vg()
    for d in (2, 3, 4, 10, 137):
        h = m.verb_entropy_nats(np.ones(d))
        assert abs(h - math.log(d)) < 1e-12, (
            "a uniform verb over %d successors read %.15f, expected log(%d) = %.15f"
            % (d, h, d, math.log(d)))


def test_the_entropy_is_in_nats_and_not_in_bits():
    """log(2) nats, not 1.0 bit. A base error rescales every number downstream."""
    m = _vg()
    h = m.verb_entropy_nats(np.ones(2))
    assert abs(h - math.log(2.0)) < 1e-12, \
        "a fair binary verb read %.15f: this is bits, not nats" % h


def test_a_verb_with_no_live_continuation_is_a_refusal_with_a_reason():
    """Zero mass is not zero entropy. It is the absence of a distribution, and it
    has to arrive as a value carrying its cause, never as 0.0 or NaN."""
    m = _vg()
    r = m.verb_entropy_nats(np.zeros(5))
    assert m.is_refusal(r), "an all-zero row returned %r instead of a Refusal" % (r,)
    assert r.code and r.reason, "the refusal carries no code or no reason: %r" % (r,)


# ---------------------------------------------------------------------------
# 2. THE GRADE THE ADDENDUM LITERALLY ASKS FOR IS IDENTICALLY ZERO
# ---------------------------------------------------------------------------

def test_the_literal_reading_of_the_grade_is_reported_as_identically_zero():
    """A2 as written asks for H(next | verb) where the verb is a concrete move.
    The successor map on this space is a FUNCTION, so that entropy is 0 for every
    one of the 4,891,672 moves -- by construction, not by measurement. A module
    that quietly substitutes a different reading has changed the claim without
    saying so."""
    m = _vg()
    lit = m.grades()["literal_reading"]
    assert lit["max_nats"] == 0.0 and lit["min_nats"] == 0.0, \
        "the literal grade is not identically zero: [%r, %r]" % (lit["min_nats"],
                                                                 lit["max_nats"])
    assert lit["n_verbs"] == 4891672, \
        "the literal grade was checked on %d moves, not all 4,891,672" % lit["n_verbs"]
    assert "function" in lit["reason"].lower() or "determin" in lit["reason"].lower(), \
        "the module gives no structural reason for the zero: %r" % lit["reason"]


def test_no_two_legal_moves_from_a_position_collapse_to_the_same_successor():
    """The fact underneath both degeneracies. If successors ever collided,
    H(next | position) would fall BELOW ln(deg) and the identity in the next test
    would be a coincidence rather than a structure."""
    m = _vg()
    assert m.grades()["n_successor_collisions"] == 0, \
        "%d positions have two moves reaching one state: the identity is not exact" \
        % m.grades()["n_successor_collisions"]


# ---------------------------------------------------------------------------
# 3. THE THREE READINGS, SIDE BY SIDE
# ---------------------------------------------------------------------------

def test_all_three_readings_are_reported_side_by_side_with_their_levels():
    m = _vg()
    defs = {d["key"]: d for d in m.verb_report()["definitions"]}
    for key in ("a_position", "b_reply", "c_coarse", "c_fine"):
        assert key in defs, "reading %r is missing: %r" % (key, sorted(defs))
    assert defs["a_position"]["level"] == "noun"
    assert defs["b_reply"]["level"] == "occurrence"
    assert defs["c_coarse"]["level"] == defs["c_fine"]["level"] == "type"
    for d in defs.values():
        for k in ("min_nats", "median_nats", "max_nats", "mean_nats",
                  "n_values", "denominator", "abstraction"):
            assert k in d, "reading %r has no %s" % (d["key"], k)
        assert d["min_nats"] <= d["median_nats"] <= d["max_nats"]
        assert d["abstraction"].strip(), "reading %r states no abstraction" % d["key"]


def test_reading_a_is_exactly_log_out_degree_everywhere():
    """H(next | position) under the uniform policy IS ln(deg). Not close to."""
    m = _vg()
    g = m.grades()
    deg = g["out_degree"]
    live = deg > 0
    gap = float(np.abs(g["h_position"][live] - np.log(deg[live])).max())
    assert gap == 0.0, "H(next|position) differs from ln(deg) by %.3e" % gap
    assert int(live.sum()) == 367216, \
        "%d live positions, not the enumerated 367,216" % int(live.sum())
    assert deg.max() == 35, "the maximum out-degree is %d, not 35" % deg.max()


def test_reading_b_is_exactly_log_out_degree_of_the_successor():
    m = _vg()
    g = m.grades()
    d = g["b_reply"]
    assert d["max_abs_gap"] == 0.0, \
        "the reply entropy differs from ln(deg of successor) by %.3e" % d["max_abs_gap"]
    assert d["n_occurrences"] > 4000000, \
        "the identity was checked on only %d occurrences" % d["n_occurrences"]


def test_a_monotone_function_of_out_degree_cannot_beat_out_degree():
    """THE STRUCTURAL FINDING, and it has to be SEEN rather than argued. Spearman
    and AUC are rank statistics, ln is strictly increasing, so readings (a) and
    (b) score IDENTICALLY to raw out-degree against any event. Equality here must
    be exact; a difference of 1e-9 would mean one of the two is not what it says."""
    m = _vg()
    arms = {a["name"]: a for a in m.t_verb_report()["occurrence_arms"]}
    assert "H(next|position) = ln deg" in arms and "out-degree of successor" in arms, \
        "the occurrence table does not carry both sides of the identity: %r" % sorted(arms)
    a = arms["H(state 2ply | verb)"]["auc"]
    b = arms["out-degree of successor"]["auc"]
    assert a == b, ("the reply entropy scores AUC %.12f and raw out-degree %.12f: "
                    "a monotone map changed a rank statistic" % (a, b))


def test_the_type_level_abstractions_are_stated_and_differ_in_width():
    """Fan-out at the type level comes from the abstraction, so the abstraction is
    part of the result. Two widths, both named, both with their vocabulary size."""
    m = _vg()
    widths = []
    for kind in ABSTRACTIONS:
        t = m.typed_grade(kind)
        assert t["abstraction"].strip(), "abstraction %r is unnamed" % kind
        assert t["n_verb_types"] == t["entropy_nats"].size
        widths.append(t["n_verb_types"])
    assert widths[0] < widths[1], \
        "the coarse alphabet (%d) is not narrower than the fine one (%d)" % tuple(widths)
    assert widths[0] >= 5, "a %d-word alphabet is not a vocabulary" % widths[0]


def test_the_type_level_grade_is_not_a_monotone_function_of_out_degree():
    """The escape from the degeneracy, checked rather than asserted. If the type
    grade were still monotone in degree, all three readings would be one reading."""
    from scipy import stats
    m = _vg()
    t = m.typed_grade("fine")
    r = abs(float(stats.spearmanr(t["entropy_nats"], t["mean_degree"]).statistic))
    assert r < 0.999, \
        "the type grade is rank-identical to out-degree (|rho| = %.6f): still degenerate" % r


def test_the_conditional_distributions_are_exact_probability_rows():
    """Every row sums to 1 and every entry is a probability. If the bigram matrix
    is not normalised the entropy is a number about nothing."""
    m = _vg()
    for kind in ABSTRACTIONS:
        t = m.typed_grade(kind)
        rows = np.asarray(t["conditional"].sum(1)).ravel()[t["live_mask"]]
        assert np.abs(rows - 1.0).max() < 1e-10, \
            "%s conditional rows are off simplex by %.3e" % (kind,
                                                             np.abs(rows - 1.0).max())
        assert t["conditional"].data.min() > 0.0, \
            "the %s conditional carries a non-positive entry" % kind
        assert t["conditional"].data.max() <= 1.0 + 1e-12, \
            "the %s conditional carries an entry above 1" % kind


def test_a_verb_type_whose_occurrences_all_end_the_stream_is_a_refusal_not_a_zero():
    """The distinction a 0.0 would erase. On the coarse alphabet the
    queen-capture verb has no continuation at all, and eight OTHER verbs have a
    genuine, forced 0.0 -- if the first were also written 0.0 the two would be
    indistinguishable and the zero fraction would be a lie.

    WRITTEN-AFTER, UNBOUND: added after the last recorded RED, so it has
    never been seen to fail. Not retro-fitted with a red; see the file docstring.
    """
    m = _vg()
    t = m.typed_grade("coarse")
    assert t["n_refused_verb_types"] >= 1, \
        "no coarse verb type refuses, but the queen-capture verb has no reply"
    assert len(t["refusals"]) == t["n_refused_verb_types"], \
        "the refusals are counted but not carried as values"
    for r in t["refusals"]:
        assert m.is_refusal(r) and r.code and r.reason, \
            "a refusal arrives without a code or a reason: %r" % (r,)
    assert np.isnan(t["entropy_nats"][~t["live_mask"]]).all(), \
        "a refused verb type carries a number instead of a hole"
    genuine = int((t["entropy_nats"] == 0.0).sum())
    assert genuine > 0, "no coarse verb has a genuine forced zero to be confused with"


def test_the_type_entropy_is_bracketed_by_its_two_analytic_bounds():
    """H <= log(support) always, and H >= E[component entropy] because the
    conditional is a MIXTURE. Outside either bracket is a bug in the mixture, not
    a finding about chess.

    The lower bound is the mean entropy of the successor's TYPE distribution, NOT
    E[ln deg]. Those coincide only when the abstraction is injective within a
    position; a coarse word merges several legal moves, so ln(deg) overstates the
    component and the bound it would give is false. Asserted here in the correct
    form, with the wrong form asserted to actually FAIL on the coarse alphabet --
    a bound that no alphabet violates is a bound this test cannot be trusted on.
    """
    m = _vg()
    violations = {}
    for kind in ABSTRACTIONS:
        t = m.typed_grade(kind)
        keep = t["live_mask"]
        h, ls = t["entropy_nats"][keep], t["log_support"][keep]
        comp, mld = t["mean_component_entropy"][keep], t["mean_log_degree"][keep]
        assert (h <= ls + 1e-9).all(), \
            "%s: %d verbs read above log(support)" % (kind, int((h > ls + 1e-9).sum()))
        assert (h >= comp - 1e-9).all(), \
            "%s: %d verbs read below the mean COMPONENT entropy" % (
                kind, int((h < comp - 1e-9).sum()))
        violations[kind] = int((h < mld - 1e-9).sum())
    assert violations["fine"] == 0, \
        "the fine alphabet violates the ln(deg) bound: it should be injective"
    assert violations["coarse"] > 0, (
        "the coarse alphabet satisfies the ln(deg) bound on every verb, so this "
        "test cannot tell the two bounds apart and proves nothing")


def test_the_component_of_the_bound_is_log_degree_only_where_the_words_are_unique():
    """Why the two bounds differ, checked at the source. Under the fine
    abstraction no two legal moves at a position share a word, so the component
    entropy IS ln(deg) exactly; under the coarse one they do, and it is strictly
    below.

    WRITTEN-AFTER, UNBOUND: added after the last recorded RED, so it has
    never been seen to fail. Not retro-fitted with a red; see the file docstring.
    """
    m = _vg()
    fine = m.typed_grade("fine")["component_gap_vs_log_degree"]
    coarse = m.typed_grade("coarse")["component_gap_vs_log_degree"]
    assert fine < 1e-12, (
        "the fine abstraction is not injective within a position: gap %.3e is far "
        "above the float residue of summing d terms against one log" % fine)
    assert coarse > 1.0, (
        "the coarse abstraction merges almost nothing: gap %.3e. A word that never "
        "covers two legal moves is not a coarser word." % coarse)


def test_the_grade_counts_are_the_enumerated_ones():
    """Grade 0 nouns, grade 1 verb occurrences. A grade that is not counted is a
    word, and the bed's published totals are 368,452 and 4,891,672."""
    m = _vg()
    g = m.grades()
    assert g["n_nouns"] == 368452, "grade 0 has %d nouns" % g["n_nouns"]
    assert g["n_verb_occurrences"] == 4891672, \
        "grade 1 has %d occurrences" % g["n_verb_occurrences"]


# ---------------------------------------------------------------------------
# 4. THE ESTIMATOR'S ERROR AGAINST THE EXACT VALUE
# ---------------------------------------------------------------------------

def test_the_plug_in_estimator_is_measured_against_the_exact_value_at_several_sizes():
    """The number this module owes the rest of the programme.

    Every corpus that cannot be enumerated will read this grade with a plug-in
    estimator. Its bias must be reported against the EXACT value, at more than
    one sample size, with an interval over replicates.
    """
    m = _vg()
    for kind in ABSTRACTIONS:
        rows = [r for r in m.estimator_bias() if r["abstraction"] == kind]
        assert len(rows) >= 3, "%s has only %d sample sizes" % (kind, len(rows))
        for r in rows:
            assert r["bias_nats"] < 0.0 or r["ci_lo"] <= 0.0 <= r["ci_hi"], (
                "%s at N=%d reads %+.6f with CI [%+.6f, %+.6f]: a plug-in entropy is "
                "biased DOWNWARD, so a SIGNIFICANTLY positive bias means the exact "
                "value is wrong" % (kind, r["n_samples"], r["bias_nats"],
                                    r["ci_lo"], r["ci_hi"]))
            assert r["ci_lo"] < r["bias_nats"] < r["ci_hi"], \
                "%s at N=%d sits outside its own interval" % (kind, r["n_samples"])
            assert r["n_replicates"] >= 5
        sizes = [r["n_samples"] for r in rows]
        assert sizes == sorted(sizes), "%s sizes are not in order" % kind
        bias = [abs(r["bias_nats"]) for r in rows]
        if kind == "fine":
            assert bias == sorted(bias, reverse=True), \
                "the wide alphabet's bias does not shrink with sample size: %r" % bias
        else:
            # The narrow alphabet reaches zero mid-sweep, so the sequence is not
            # monotone -- only the endpoints are, and only they are claimed.
            assert bias[-1] < bias[0], \
                "the narrow alphabet's bias is no smaller at N=%d than at N=%d: %r" \
                % (sizes[-1], sizes[0], bias)


def test_the_bias_scales_with_the_alphabet_which_is_the_transferable_finding():
    """The reason both abstractions are run. The plug-in bias is set by the number
    of CELLS, so a narrow alphabet is estimable where a wide one is not -- and
    that, not the chess, is what transfers to a corpus."""
    m = _vg()
    rows = {(r["abstraction"], r["n_samples"]): r for r in m.estimator_bias()}
    biggest = max(n for _, n in rows)
    coarse = abs(rows[("coarse", biggest)]["bias_nats"])
    fine = abs(rows[("fine", biggest)]["bias_nats"])
    assert fine > 10 * coarse, (
        "at N=%d the wide alphabet is biased %.4f and the narrow one %.4f: the "
        "alphabet is not driving the bias" % (biggest, fine, coarse))


def test_a_bias_correction_is_offered_and_beats_the_raw_plug_in_where_it_can():
    """THE HIPPOCRATIC ROUTE, and its price. If the plug-in is condemned the
    report has to name what to use instead -- Miller-Madow is the cheapest
    correction -- but the claim is bounded: it must strictly help on every row
    whose bias this run can RESOLVE, and it is allowed to hurt where the plug-in
    was already inside its own interval, because there it corrects an error that
    was not there. A correction claimed everywhere would be the overclaim."""
    m = _vg()
    rows = m.estimator_bias()
    resolvable = [r for r in rows if r["ci_hi"] < 0.0]
    assert len(resolvable) >= 3, \
        "only %d rows have a resolvable bias: the sweep proves nothing" % len(resolvable)
    for r in rows:
        assert "miller_madow_bias_nats" in r, \
            "no corrected arm at %s N=%d" % (r["abstraction"], r["n_samples"])
    for r in resolvable:
        assert abs(r["miller_madow_bias_nats"]) < abs(r["bias_nats"]), (
            "%s at N=%d: the correction (%+.6f) does not beat the raw plug-in "
            "(%+.6f) where the bias is resolvable"
            % (r["abstraction"], r["n_samples"], r["miller_madow_bias_nats"],
               r["bias_nats"]))


# ---------------------------------------------------------------------------
# 5. T-VERB, WHICH ONLY EXISTS BESIDE ITS CONTROLS
# ---------------------------------------------------------------------------

def test_t_verb_scores_entropy_beside_out_degree_and_a_shuffled_arm():
    """The claim without the controls is not a claim. Out-degree is the cheaper
    predictor of the same events and log(degree) is related to H by construction,
    so both must be in the table or the comparison cannot be read."""
    m = _vg()
    t = m.t_verb_report()
    arms = {a["name"]: a for a in t["arms"]}
    for needed in ("H(next|verb)", "E[log deg]", "mean deg", "H shuffled"):
        assert needed in arms, "arm %r is missing: %r" % (needed, sorted(arms))
    for name, a in arms.items():
        assert a["ci_lo"] <= a["rho"] <= a["ci_hi"], \
            "arm %r reports a rho outside its own interval" % name


def test_the_occurrence_table_scores_every_reading_on_one_event_with_one_statistic():
    """All three readings live at different levels, so they can only be compared
    on the occurrences they share. One event, one AUC, one denominator."""
    m = _vg()
    t = m.t_verb_report()
    arms = {a["name"]: a for a in t["occurrence_arms"]}
    assert len(arms) >= 5, "only %d arms in the occurrence table" % len(arms)
    # The shared denominator: every occurrence except the ones where some arm
    # REFUSES a grade. Derived from the refusal count, never written as a literal.
    expected = 4891672 - m.typed_grade("coarse")["n_refused_occurrences"]
    for name, a in arms.items():
        assert 0.0 <= a["auc"] <= 1.0, "arm %r has AUC %r" % (name, a["auc"])
        assert a["ci_lo"] <= a["auc"] <= a["ci_hi"], \
            "arm %r reports an AUC outside its own interval" % name
        assert a["n_events"] > 0, "arm %r saw no events" % name
        assert a["n_occurrences"] == expected, (
            "arm %r is scored on %d occurrences, not the %d every arm shares"
            % (name, a["n_occurrences"], expected))
    assert len({a["n_occurrences"] for a in arms.values()}) == 1, \
        "the arms are scored on different denominators and cannot be compared"


def test_the_paired_difference_between_entropy_and_degree_carries_an_interval():
    """Two rhos side by side cannot say which won -- they share every verb. The
    difference has to be bootstrapped PAIRED, or a small gap is unreadable."""
    m = _vg()
    d = m.t_verb_report()["paired_vs_degree"]
    assert d["ci_lo"] < d["delta"] < d["ci_hi"], \
        "the paired difference sits outside its own interval"
    assert d["verdict"] in ("entropy wins", "degree wins", "tie"), \
        "the paired comparison has no stated verdict: %r" % d.get("verdict")
    sign_agrees = ((d["ci_lo"] > 0 and d["verdict"] == "entropy wins")
                   or (d["ci_hi"] < 0 and d["verdict"] == "degree wins")
                   or (d["ci_lo"] <= 0 <= d["ci_hi"] and d["verdict"] == "tie"))
    assert sign_agrees, (
        "the verdict %r does not follow from the interval [%+.4f, %+.4f]"
        % (d["verdict"], d["ci_lo"], d["ci_hi"]))


def test_shuffling_the_entropy_grade_destroys_the_association():
    """PLANTED NEGATIVE, SEEN TO FIRE. If a shuffled grade scores anywhere near
    the real one, the association is coming from the binning and not the grade."""
    m = _vg()
    arms = {a["name"]: a for a in m.t_verb_report()["arms"]}
    real, shuf = abs(arms["H(next|verb)"]["rho"]), abs(arms["H shuffled"]["rho"])
    assert shuf < 0.15, "the shuffled arm still scores |rho| = %.4f" % shuf
    assert real > 5 * shuf, \
        "the real grade (%.4f) is not clear of its shuffle (%.4f)" % (real, shuf)


def test_every_binned_rate_carries_its_bin_count_and_an_interval():
    """A rate without its denominator is a story, and a rate without an interval
    cannot be compared to the bin beside it."""
    m = _vg()
    t = m.t_verb_report()
    assert len(t["bins"]) >= 4, "only %d bins" % len(t["bins"])
    total = 0
    for b in t["bins"]:
        assert b["n_occurrences"] > 0 and b["n_verbs"] > 0, "an empty bin was reported"
        assert 0.0 <= b["ci_lo"] <= b["rate"] <= b["ci_hi"] <= 1.0, \
            "bin rate %.4f is not inside its Wilson interval" % b["rate"]
        assert b["n_events"] == int(b["n_events"]), "the event count is not an integer"
        total += b["n_occurrences"]
    assert total == m.grades()["n_verb_occurrences"], \
        "the bins cover %d occurrences of %d" % (total,
                                                 m.grades()["n_verb_occurrences"])


def test_the_abstention_events_are_counted_by_cause():
    """Refusals are values with reasons. An abstention rate that cannot be split
    into stalemate, drawn-by-defence and the captured-queen draw is one number
    standing in for three different things."""
    m = _vg()
    causes = m.t_verb_report()["abstention_by_cause"]
    assert len(causes) >= 2, "abstentions are reported under %d causes" % len(causes)
    assert sum(causes.values()) == m.grades()["n_abstentions_total"], \
        "the causes do not sum to the total abstention count"
    for code, k in causes.items():
        assert isinstance(code, str) and code.strip(), "a cause has no name"
        assert k > 0, "cause %r is reported with zero events" % code


# ---------------------------------------------------------------------------
# 6. SEEDS, AND THE LINEAGE FIGURES
# ---------------------------------------------------------------------------

def test_every_number_is_reproducible_at_the_pinned_seed():
    m = _vg()
    a = m.t_verb_report(seed=m.SEED)
    b = m.t_verb_report(seed=m.SEED)
    assert [x["rho"] for x in a["arms"]] == [x["rho"] for x in b["arms"]], \
        "two calls at the same seed disagree"
    ea = m.estimator_bias(seed=m.SEED)
    eb = m.estimator_bias(seed=m.SEED)
    assert [x["bias_nats"] for x in ea] == [x["bias_nats"] for x in eb], \
        "the estimator bias is not reproducible at the pinned seed"


def test_the_owners_lineage_triple_is_recorded_and_compared_against_all_readings():
    """0.17 / 0.69 / 1.39 nats came from another alphabet. Not an expected value
    here and never hardcoded as one -- but RECORDED, and compared against EVERY
    reading, with the nearest one named, or the two quietly get read as the same
    measurement."""
    m = _vg()
    src = (ROOT / "ceqjepa" / "verb_grade.py").read_text(encoding="utf-8")
    for v in OWNER_TRIPLE:
        assert str(v) in src, "the owner's lineage figure %s is not recorded" % v
    r = m.verb_report()
    assert r["owner_triple_nats"] == list(OWNER_TRIPLE), \
        "the recorded lineage triple is not the owner's"
    assert r["owner_nearest_reading"] in [d["key"] for d in r["definitions"]], \
        "no reading is named as the nearest to the lineage triple"
    for d in r["definitions"]:
        diff = d["owner_difference_nats"]
        ours = [d["min_nats"], d["median_nats"], d["max_nats"]]
        for got, own, delta in zip(ours, OWNER_TRIPLE, diff):
            assert abs((got - own) - delta) < 1e-9, \
                "reading %r states a difference that is not ours minus his" % d["key"]


def test_the_entropy_distribution_reports_the_zero_fraction_and_its_denominator():
    m = _vg()
    for d in m.verb_report()["definitions"]:
        assert d["n_values"] > 0, "reading %r has no denominator" % d["key"]
        assert 0.0 <= d["zero_fraction"] <= 1.0
        assert d["denominator"].strip(), "reading %r does not say what it is over" % d["key"]
    a = [d for d in m.verb_report()["definitions"] if d["key"] == "a_position"][0]
    assert a["zero_fraction"] > 0.0, \
        "reading (a) reports no forced moves, but 13,260 positions have one legal move"


# ---------------------------------------------------------------------------
# 7. PRINTED IS NOT MEASURED. Each published number is pinned to the expression
#    that recomputes it, because a literal in a print string passes every other
#    check in this file.
# ---------------------------------------------------------------------------

def _demo_out(capsys):
    _vg().demo()
    return capsys.readouterr().out


def _one(pattern, out, what):
    hits = re.findall(pattern, out)
    assert len(hits) == 1, "expected exactly one %s line, found %d" % (what, len(hits))
    return float(hits[0])


#: How close a recomputation has to land on the RETURNED value. Not the printing
#: tolerance: a literal typed to the printed precision would satisfy that and is
#: exactly mutation 2. This is tight enough that only the real expression passes.
EXACT = 1e-12


def test_identity_the_published_aggregate_entropy_is_recomputed_from_the_grades(capsys):
    """Both mutations, in one test.

    MUTATION 1, a fabricated number in the print string with the dict left
    honest, is caught by the printed-vs-returned leg. MUTATION 2, a literal
    written into the dict with the print left honest, passes the docstring guard
    and the printed-vs-returned leg both -- it is caught only by recomputing
    sum_v p(v) H(v) from the raw per-verb arrays and demanding the RETURNED value
    match to 1e-12, which a number typed to six decimals cannot do.
    """
    m = _vg()
    out = _demo_out(capsys)
    printed = _one(r"exact aggregate H\(next \| verb type\)\s+([0-9.]+) nats", out,
                   "aggregate entropy")
    t = m.typed_grade("fine")
    keep = t["live_mask"]
    recomputed = float((t["n_live"][keep] / t["n_live"].sum()
                        * t["entropy_nats"][keep]).sum())
    returned = m.verb_report()["aggregate_fine_nats"]
    assert abs(printed - returned) < 5e-7, (
        "the demo prints %.6f nats but verb_report() returns %.6f: the printed "
        "number is a literal" % (printed, returned))
    assert abs(returned - recomputed) < EXACT, (
        "verb_report() returns %.15f but the per-verb arrays recompute %.15f: the "
        "returned number is a literal" % (returned, recomputed))


def test_identity_the_published_reading_a_mean_is_recomputed_from_the_out_degrees(capsys):
    """Reading (a)'s mean, recomputed from the raw out-degree array with numpy."""
    m = _vg()
    out = _demo_out(capsys)
    printed = _one(r"a_position[^\n]*?mean\s+([0-9.]+)", out, "reading (a) mean")
    recomputed = float(np.log(np.maximum(m.grades()["out_degree"], 1)).mean())
    returned = [d for d in m.verb_report()["definitions"]
                if d["key"] == "a_position"][0]["mean_nats"]
    assert abs(printed - returned) < 5e-6, \
        "the demo prints mean %.6f, verb_report() returns %.6f" % (printed, returned)
    assert abs(returned - recomputed) < EXACT, \
        "verb_report() returns %.15f, the out-degrees recompute %.15f" % (returned,
                                                                          recomputed)


def test_identity_the_published_t_verb_rho_is_recomputed_from_the_grades(capsys):
    """The headline association, recomputed from the per-verb arrays with scipy
    directly, sharing no code path with the reporter."""
    from scipy import stats
    m = _vg()
    out = _demo_out(capsys)
    printed = _one(r"H\(next\|verb\)\s+([+-][0-9.]+)", out, "T-VERB entropy rho")
    t = m.typed_grade("fine")
    keep = t["live_mask"]
    rate = (t["n_abstentions"] / t["n_occurrences"])[keep]
    recomputed = float(stats.spearmanr(t["entropy_nats"][keep], rate).statistic)
    returned = [a for a in m.t_verb_report()["arms"]
                if a["name"] == "H(next|verb)"][0]["rho"]
    assert abs(printed - returned) < 5e-5, \
        "the demo prints rho %+.4f, t_verb_report() returns %+.4f" % (printed,
                                                                      returned)
    assert abs(returned - recomputed) < 1e-9, \
        "t_verb_report() returns %.15f, scipy recomputes %.15f" % (returned,
                                                                   recomputed)


def test_identity_the_published_estimator_bias_is_recomputed_from_the_entry_point(capsys):
    """Every row of the sweep, both legs. The exact value each bias is measured
    against is itself recomputed from the per-verb arrays, and the percentage is
    re-derived from the two, so a literal anywhere in the row fails."""
    m = _vg()
    out = _demo_out(capsys)
    for r in m.estimator_bias():
        tok = "%+.4f" % r["bias_nats"]
        assert tok in out, (
            "the demo does not print the bias %s that estimator_bias() returns for "
            "%s at N=%d" % (tok, r["abstraction"], r["n_samples"]))
        t = m.typed_grade(r["abstraction"])
        keep = t["live_mask"]
        exact = float((t["n_live"][keep] / t["n_live"].sum()
                       * t["entropy_nats"][keep]).sum())
        assert abs(r["exact_nats"] - exact) < EXACT, (
            "%s at N=%d is measured against %.15f but the arrays give %.15f"
            % (r["abstraction"], r["n_samples"], r["exact_nats"], exact))
        assert abs(r["bias_percent"] - 100.0 * r["bias_nats"] / exact) < 1e-9, \
            "%s at N=%d: the percentage is not the bias over the exact value" \
            % (r["abstraction"], r["n_samples"])


def test_identity_the_binned_rates_are_recomputed_from_the_event_counts(capsys):
    """Every published rate re-derived from its own two integers, and the
    integers re-derived from the per-verb abstention counts.

    WRITTEN-AFTER, UNBOUND: added after the last recorded RED, so it has
    never been seen to fail. Not retro-fitted with a red; see the file docstring.
    """
    m = _vg()
    t = m.t_verb_report()
    fine = m.typed_grade("fine")
    keep = fine["live_mask"]
    for b in t["bins"]:
        assert abs(b["rate"] - b["n_events"] / b["n_occurrences"]) < EXACT, \
            "bin %d's rate is not its events over its occurrences" % b["bin"]
    assert sum(b["n_events"] for b in t["bins"]) == \
        int(fine["n_abstentions"][keep].sum()), \
        "the bins' events do not sum to the per-verb abstention counts"
    assert abs(t["overall_rate"]
               - m.grades()["n_abstentions_total"]
               / m.grades()["n_verb_occurrences"]) < EXACT, \
        "the overall rate is not the total events over the total occurrences"


def test_identity_the_runtime_the_demo_prints_is_the_runtime_it_took(capsys):
    """A hardcoded timing printed inside a line labelled RUN is exactly how a
    fabricated number survived this round.

    This call is the WARM one -- every entry point is cached by now, so the demo
    takes milliseconds and an honest clock prints about zero. That is the strong
    form of the check, not the weak one: any literal large enough to look like a
    cold run fails the upper bound here immediately. The cold run is checked
    against its own subprocess clock in the self-check test below.
    """
    t0 = time.perf_counter()
    out = _demo_out(capsys)
    took = time.perf_counter() - t0
    printed = _one(r"self-check wall clock\s+([0-9.]+) s", out, "wall clock")
    assert printed <= took + 1.0, (
        "the demo printed %.1f s but took %.3f s on a warm cache: that number "
        "did not come from this run" % (printed, took))
    assert printed >= 0.5 * took - 0.1, \
        "the demo printed %.1f s but took %.3f s" % (printed, took)


# ---------------------------------------------------------------------------
# 8. NO NUMBER IN A DOCSTRING THAT NO RUN PRINTS
# ---------------------------------------------------------------------------

#: CLAUSE 1: ANY number, WITH ITS BOUNDARIES, decimals and bare integers alike --
#: bare integers in prose have been struck twice this round on top of decimals.
#:
#: The lookarounds are the repair for the SUBSTRING VACUITY the Inspector struck
#: in the form this file shipped first. `"89167" in out` is TRUE the moment the
#: run prints 4891672, so a plain `in` test accepts any fabricated number that
#: shares digits with a real one -- and a stale number looks exactly like that.
#: Measured on this module's own run before the repair: the fabricated 89167,
#: 728896, 6845 and .8530 ALL passed the substring form, against the printed
#: 4891672, 5.728896, 368452 and -0.8530. Matching TOKEN-to-TOKEN, with
#: membership against the run's token SET rather than its raw text, rejects all
#: four while still finding every real number. Pinned permanently below in
#: test_the_guard_rejects_a_fragment_of_a_printed_number, which is the vacuity
#: control: a guard whose planted collision passes is measuring nothing.
#:
#: The inner (?:\.\d+)* keeps a dotted version string as ONE token rather than
#: splitting it into a spurious first component. The trailing guard is
#: (?!\w)(?!\.\d) and not (?![\w.]) so that a sentence-final period cannot hide
#: the number in front of it -- a guard that misses the run's own numbers is the
#: same defect pointing the other way.
#:
#: CLAUSE 3: no exemption list. A legitimate constant that fails here -- a year, a
#: pinned parameter -- is fixed by PRINTING it in the run.
_MEASURED = re.compile(
    r"(?<![\w.])[+-]?\d+(?:\.\d+)*(?:[eE][+-]?\d+)?(?!\w)(?!\.\d)")


def _tokens(text):
    """Every number in `text` as a boundary-anchored token.

    A leading plus is stripped, so a docstring quoting a number bare still
    matches a run that prints it signed. A leading MINUS is not: a negative
    number is a different number, and matching across the sign would put the
    defect straight back.
    """
    return {t.lstrip("+") for t in _MEASURED.findall(text)}

#: CLAUSE 4's floor. Without one, a scope regression that silently empties the
#: scan reports "0 missing" and passes cheerfully.
MIN_DOCSTRINGS = 12
MIN_NUMBERS = 90


def _module_docstrings(m):
    """(name, docstring) for the module and everything DEFINED in it.

    CLAUSE 2: scoped by __module__, not by __all__. numpy's and scipy's
    docstrings do not come along, but demo() and every PRIVATE helper do -- a
    bare constant in a helper's prose is exactly where the __all__-scoped form of
    this guard has been walked through.

    Deliberately re-implemented here rather than calling the module's own
    docstring_numbers(): two counters that must agree is a check, one shared
    helper the module could quietly narrow is not.
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
    """The check whose absence let fabricated numbers ship this round.

    Every number in the module docstring, and in the docstring of everything
    defined in this module, must appear verbatim in the output of `python -m
    ceqjepa.verb_grade`. A number no run prints cannot be checked by anyone, its
    author included, and drifts from the moment the bed changes.
    """
    m = _vg()
    out = _demo_out(capsys)
    run = _tokens(out)
    docs = _module_docstrings(m)
    missing = [(where, tok) for where, doc in docs
               for tok in _tokens(doc) if tok not in run]
    checked = sum(len(_MEASURED.findall(doc)) for _, doc in docs)
    print("\n  %d numbers across %d docstrings, %d printed by no run, run token "
          "set %d wide" % (checked, len(docs), len(missing), len(run)))
    assert len(docs) >= MIN_DOCSTRINGS, \
        "only %d docstrings in scope: the scan has been narrowed" % len(docs)
    assert checked >= MIN_NUMBERS, \
        "only %d numbers found: the pattern is not biting" % checked
    assert not missing, ("these docstring numbers are printed by no run:\n    "
                         + "\n    ".join("%s: %s" % (w, t) for w, t in missing))


#: The permanent planted collisions. Each is a fabricated number that is a
#: FRAGMENT of a real one the run prints, and each passed the substring form this
#: file shipped first. They stay here forever: if the run stops printing the whole
#: number the control is vacuous, so both halves are asserted every time.
_FRAGMENT_PLANTS = (("89167", "4891672"), ("728896", "5.728896"),
                    ("6845", "368452"), ("48916", "4891672"))


def test_the_guard_rejects_a_fragment_of_a_printed_number(capsys):
    """THE VACUITY CONTROL, and the reason the guard above is not a substring test.

    A guard whose planted collision passes is measuring nothing, and "0 missing"
    from such a guard is a number about nothing. Each plant below is a fabricated
    value that shares digits with a real printed one. The substring form accepts
    every one of them -- asserted here, so the defect is demonstrated rather than
    described -- and the token form must reject every one while still finding the
    whole number it was cut from.

    WRITTEN-AFTER, UNBOUND: added after the last recorded RED, so it has
    never been seen to fail. Not retro-fitted with a red; see the file docstring.
    """
    out = _demo_out(capsys)
    run = _tokens(out)
    for fragment, whole in _FRAGMENT_PLANTS:
        assert whole in run, (
            "the run stopped printing %s, so the %s plant is vacuous and this "
            "control proves nothing" % (whole, fragment))
        assert fragment in out, (
            "%s is no longer a substring of the run: the plant no longer "
            "exercises the defect it was built for" % fragment)
        assert fragment not in run, (
            "SUBSTRING VACUITY: the fabricated %s is being read as present "
            "because the run prints %s. The guard is a substring test again."
            % (fragment, whole))
    absent = "40317.5"
    assert absent not in out and absent not in run, \
        "the absent-number control %s was found in the run" % absent
    print("\n  %d fragment plants rejected by the token guard and accepted by the "
          "substring form" % len(_FRAGMENT_PLANTS))


def test_the_coverage_figure_is_itself_bound_by_two_independent_counters(capsys):
    """CLAUSE 4. The coverage figure is a number in prose like any other, and a
    stale one -- counted during a RED run, before demo() had a docstring -- is
    how the first version of this guard was struck. The module counts its own
    docstrings, this test counts them again with its own scan, the two must
    agree, and BOTH figures must appear verbatim in the run.

    WRITTEN-AFTER, UNBOUND: added after the last recorded RED, so it has
    never been seen to fail. Not retro-fitted with a red; see the file docstring.
    """
    m = _vg()
    out = _demo_out(capsys)
    mine = _module_docstrings(m)
    theirs = m.docstring_numbers()
    n_mine = sum(len(_MEASURED.findall(d)) for _, d in mine)
    assert theirs["n_docstrings"] == len(mine), (
        "the module counts %d docstrings, an independent scan counts %d"
        % (theirs["n_docstrings"], len(mine)))
    assert theirs["n_numbers"] == n_mine, (
        "the module counts %d numbers in its docstrings, an independent scan "
        "counts %d" % (theirs["n_numbers"], n_mine))
    for figure in (theirs["n_docstrings"], theirs["n_numbers"]):
        assert str(figure) in out, \
            "the coverage figure %d is not printed by the run" % figure
    assert len(mine) >= MIN_DOCSTRINGS and n_mine >= MIN_NUMBERS, \
        "the floor is not met: %d docstrings, %d numbers" % (len(mine), n_mine)


# ---------------------------------------------------------------------------
# 9. THE SELF-CHECK
# ---------------------------------------------------------------------------

def test_the_self_check_runs_under_300s_and_ends_with_the_banner():
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "ceqjepa.verb_grade"],
                       cwd=str(ROOT), capture_output=True, text=True, timeout=420)
    dt = time.time() - t0
    print("\n  python -m ceqjepa.verb_grade -> exit %d in %.1f s" % (p.returncode, dt))
    assert p.returncode == 0, "the self-check failed:\n%s\n%s" % (p.stdout[-3000:],
                                                                  p.stderr[-3000:])
    assert dt < 300.0, "the self-check took %.1f s, over the 300 s budget" % dt
    assert p.stdout.strip().splitlines()[-1].strip() == "ALL SELF-CHECKS PASSED", \
        "the self-check does not end with the banner:\n%s" % p.stdout[-800:]
    # THE COLD-RUN TIMING IDENTITY. The subprocess is timed from outside; the
    # clock it printed must be that duration minus interpreter startup, so a
    # literal cannot sit in the line and no cache can flatter it.
    printed = _one(r"self-check wall clock\s+([0-9.]+) s", p.stdout, "wall clock")
    assert 0.5 * dt <= printed <= dt, (
        "the cold run took %.1f s from outside but printed %.1f s" % (dt, printed))
    print("  cold-run clock: %.1f s printed against %.1f s measured" % (printed, dt))


def test_the_verdict_on_t_verb_is_stated_in_the_module_and_not_only_in_a_report():
    """A finding that lives in a round report is a finding that gets re-shipped.
    Whatever T-VERB scored, the module has to carry it."""
    m = _vg()
    src = (ROOT / "ceqjepa" / "verb_grade.py").read_text(encoding="utf-8")
    assert re.search(r"T-VERB", src), "the module never names the claim it tests"
    v = m.t_verb_report()["paired_vs_degree"]["verdict"]
    assert v.split()[0].lower() in src.lower(), \
        "the module source does not state the verdict %r it computes" % v


def test_the_prior_art_line_is_recorded_in_the_module():
    """The grade is unoccupied ground and the module has to say why it thinks so,
    beside what it does NOT claim. An unattributed conjunction is the failure
    mode this project has already paid for."""
    m = _vg()
    src = (ROOT / "ceqjepa" / "verb_grade.py").read_text(encoding="utf-8")
    assert re.search(r"PRIOR ART|NOT CLAIMED", src), \
        "the module carries no prior-art or not-claimed section"

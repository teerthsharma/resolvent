"""RED-FIRST tests for ceqjepa/t_length.py (T-LENGTH: the length bar, both ways).

WHAT THESE TESTS ARE FOR. The module under test claims that a readout whose
NORMALISATION EXPONENT is assigned per output coordinate -- beta = 1 normalised
(a softmax mean), beta = 0 unnormalised (a linear sum) -- holds an INTENSIVE and
an EXTENSIVE bar at once at lengths it never trained on, while any single
exponent loses one of the two by a margin that grows with n. It further claims
the mask is an INPUT, so the harness does not wait for the assignment rule to be
measured. Each test below is a way for that to be FALSE while the code still
returns numbers:

  (a) THE MASK IS BAKED IN. A head that hard-codes its exponents is not a harness
     for an assignment rule, it is one more model. Tested by driving one head
     with four masks, by reading the module source for an import of the rule it
     is supposed to be independent of, and by handing pool a mask shaped like the
     TOKEN axis -- which broadcasts silently to the wrong answer in the shipped
     per-layer code and must raise here.
  (b) THE TEST RAN AT THE TRAINING LENGTH. T-LENGTH evaluated only where it
     trained measures nothing about length. That is the owner kill, so the module
     must REFUSE such a call rather than answer it.
  (c) THE TWO BARS ARE TWO BEDS. The extensive target here is n times the
     intensive one EXACTLY, recomputed in this file from the returned arrays,
     element by element, never with allclose over a whole list.
  (d) THE CALIBRATION DID NOT FIRE. The softmax arm is not a finding: it
     replicates a cited theorem, so it is the instrument that says whether the
     bed is honest. It must WIN the intensive control and LOSE the extensive
     plant, and the closed-form magnitude is pre-registered before the run. A
     softmax head that does not lose the plant leaves the theorem hypothesis
     unmet on this bed and the test VOID rather than negative; one that does not
     win the control is a broken baseline and the test is VOID as well.
  (e) THE FAILURE IS A CONSTANT. A gap that does not grow with n is a fitting
     offset, not a length failure.
  (f) THE WIN IS UNMATCHED. A win over the learned-beta head claimed without
     matched parameter counts is a tie by the owner kill rule, so both counts are
     asserted present and the published verdict must follow from the table rather
     than from prose.
  (g) N LEAKED THROUGH A POSITION. A model that can read n gets an extensive
     target for free. The census must be near zero on the feature means, must
     report the channel the extremes really do carry, and must be seen to fire on
     a planted position feature; and the n-oracle control must convert, or the
     attack is measuring nothing.
  (h) THE CLAIM OUTRAN THE CITATION. An unscoped "softmax cannot" is refuted by
     arXiv:2511.20038 in one line, so the module is required to carry the scope
     and the citations in its own docstring.
  (i) THE NUMBER CAME FROM NO RUN. Every number in the docstrings of BOTH files
     -- decimals and bare integers alike, matched as whole TOKENS -- must appear
     in a run performed here, the coverage figure is bound by two independently
     written counters against a floor, and every published value is pinned to an
     expression that recomputes it in the same run, position by position.

THE IMPORT IS GUARDED ON PURPOSE. A module-level `import ceqjepa.t_length` turns
the RED phase into one collection error with no test names in it. Each test must
be seen to FAIL BY NAME before the module exists.

RUN: python -m pytest tests/curvature/test_t_length.py -v
"""

import ast
import contextlib
import io
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

try:
    from ceqjepa import t_length as tlen
except Exception as _exc:                      # noqa: BLE001 -- RED phase carries it
    tlen = None
    _IMPORT_ERROR = _exc


def _tl():
    """The module under test, or a named failure saying it is not there yet."""
    if tlen is None:
        raise AssertionError(
            "ceqjepa/t_length.py did not import: %r" % (_IMPORT_ERROR,))
    return tlen


#: demo() trains every arm, so it is run ONCE per process and every test reads
#: the same stats and the same captured text. redirect_stdout rather than capsys
#: alone, because under `-s` the fixture captures nothing and the guard would
#: then be scanning an empty string and passing vacuously.
_DEMO = {}


def _demo(capsys):
    if not _DEMO:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            stats = _tl().demo()
        capsys.readouterr()
        _DEMO["stats"], _DEMO["out"] = stats, buf.getvalue()
        assert _DEMO["out"].strip(), "demo() printed nothing"
    return _DEMO["stats"], _DEMO["out"]


#: The floor the coverage figure is held against, for the MODULE alone. Printed
#: rather than compared against an exemption list: there is no exemption list,
#: and a number no run prints fails whatever it is.
MIN_DOCSTRINGS = 15
MIN_NUMBERS = 75

#: The owner point estimates at 4x, carried here as PROSE from the brief and
#: never as a measurement.
OWNER_4X = (7.9, 31.6)


# ---------------------------------------------------------------------------
# 1. THE MASK IS AN INPUT
# ---------------------------------------------------------------------------

def test_the_beta_mask_is_an_input_and_four_masks_drive_one_head(capsys):
    """CLAUSE (a). One pooling function, four masks, four different readouts."""
    m = _tl()
    rng = np.random.default_rng(0)
    w = rng.random((5, 7)) + 0.5
    u = rng.normal(size=(5, 7))
    out = {}
    for name, mask in (("all-softmax", (1.0, 1.0)), ("all-linear", (0.0, 0.0)),
                       ("assigned-oracle", (1.0, 0.0)), ("half", (0.5, 0.5))):
        out[name] = np.asarray(m.pool(w, u, mask))
        assert out[name].shape == (5, 2), \
            "pool returned %r for mask %r" % (out[name].shape, mask)
    for a, b in (("all-softmax", "all-linear"), ("all-softmax", "half"),
                 ("all-linear", "half")):
        assert not np.allclose(out[a], out[b]), \
            "masks %s and %s give the same readout: the exponent is baked in" % (a, b)
    for i in range(5):
        assert out["assigned-oracle"][i, 0] == out["all-softmax"][i, 0]
        assert out["assigned-oracle"][i, 1] == out["all-linear"][i, 1]
    print("\n  4 masks, 4 distinct readouts, oracle columns matched row by row")


def test_the_pooling_matches_an_independent_recomputation_element_by_element():
    """CLAUSE (a), oracle. beta = 1 is the weighted mean, beta = 0 the plain sum.

    Written here with numpy and no call into the module arithmetic, compared
    position by position so a divergence names the element.
    """
    m = _tl()
    rng = np.random.default_rng(3)
    w = rng.random((6, 9)) + 0.25
    u = rng.normal(size=(6, 9))
    got = np.asarray(m.pool(w, u, (1.0, 0.0)))
    for i in range(6):
        mean_i = float((w[i] * u[i]).sum() / w[i].sum())
        sum_i = float((w[i] * u[i]).sum())
        assert abs(got[i, 0] - mean_i) < 1e-12, \
            "row %d: normalised readout %r != %r" % (i, got[i, 0], mean_i)
        assert abs(got[i, 1] - sum_i) < 1e-12, \
            "row %d: unnormalised readout %r != %r" % (i, got[i, 1], sum_i)
        lo, hi = float(u[i].min()), float(u[i].max())
        assert lo <= got[i, 0] <= hi, \
            "row %d: the normalised readout left the convex hull" % i


def test_a_mask_shaped_like_the_token_axis_raises_instead_of_broadcasting():
    """CLAUSE (a). The live footgun in the shipped per-layer code.

    A beta shaped like the token axis broadcasts to [..., S, S] and exponentiates
    by the key rather than the query, silently and with a plausible shape. The
    mask interface here takes a vector, so the axis is asserted by name.
    """
    m = _tl()
    rng = np.random.default_rng(5)
    w, u = rng.random((3, 8)) + 0.5, rng.normal(size=(3, 8))
    with pytest.raises(ValueError):
        m.pool(w, u, np.full(8, 0.5))
    with pytest.raises(ValueError):
        m.pool(w, u, 0.5)
    with pytest.raises(ValueError):
        m.pool(w, u, np.full((2, 2), 0.5))
    ok = np.asarray(m.pool(w, u, (1.0, 0.0)))
    assert ok.shape == (3, len(m.TARGETS))


def test_the_module_imports_no_assignment_rule():
    """CLAUSE (a). The refused trade-off, checked mechanically.

    Read from the IMPORT STATEMENTS, not from the source text: the docstring
    documents the wiring with an example import line, and a substring scan calls
    that an import. Naming the interface is the deliverable; executing it today
    is what must not happen.
    """
    m = _tl()
    src = Path(m.__file__).read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    for forbidden in ("pi_assign", "hbucket"):
        offenders = sorted(x for x in imported if forbidden in x)
        assert not offenders, \
            "the module imports %r: the harness is not independent" % (offenders,)
    # RUNTIME, IN A FRESH INTERPRETER, because the static scan above cannot see
    # a DYNAMIC import and this module has one: load_measured_mask reaches the
    # rule through importlib at CALL time. What has to be true is that IMPORTING
    # this module does not reach it -- a fact about a process.
    #
    # This assertion used to read `"ceqjepa.pi_assign" not in sys.modules`, which
    # measured the SESSION rather than the module: any other test file importing
    # the rule for its own purposes turned it red, and it duly went red on
    # collection order alone (test_pi_assign.py collected first: 1 failed, the
    # three assertions above it passing in both orderings). A check that a
    # sibling file can flip is not a check on this file.
    probe = ("import sys, ceqjepa.t_length; "
             "print('ceqjepa.pi_assign' in sys.modules)")
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                          text=True,
                          cwd=str(Path(__file__).resolve().parents[2]))
    assert proc.returncode == 0, "the probe did not run: %r" % (proc.stderr[-400:],)
    assert proc.stdout.strip() == "False", (
        "importing t_length in a FRESH interpreter dragged in the assignment "
        "rule: probe printed %r" % (proc.stdout.strip(),))
    assert "pi_assign" in (m.__doc__ or ""), \
        "the wiring interface is not documented, so wiring it is not mechanical"


# ---------------------------------------------------------------------------
# 2. THE OWNER KILL: THE TRAINING LENGTH ALONE
# ---------------------------------------------------------------------------

def test_running_at_the_training_length_only_is_refused_not_answered():
    """CLAUSE (b). A refusal is a VALUE carrying its reason, not an exception.

    An exception is caught by a caller except clause; a NaN passes `nan < tol`.
    The refusal has to be visible to a scorer.
    """
    m = _tl()
    assert m.length_refusal(16, (16,)) is not None, \
        "evaluating only at the training length was not refused: that is the kill"
    assert m.length_refusal(16, (16, 8)) is not None, \
        "a shorter-than-training length was accepted as a length test"
    assert m.length_refusal(16, (64, 128)) is None, \
        "a genuine 4x/8x pair was refused"
    why = m.length_refusal(16, (16,))
    assert isinstance(why, str) and why.strip(), "the refusal carries no reason"
    rep = m.t_length_report(test_lengths=(16,), seeds=(0,))
    assert rep["refused"], "t_length_report answered a training-length-only call"
    assert "nrmse" not in rep, "a refused report still published a table"


# ---------------------------------------------------------------------------
# 3. ONE BED, TWO BARS
# ---------------------------------------------------------------------------

def test_the_two_targets_are_the_same_data_and_the_extensive_one_is_n_times_it():
    """CLAUSE (c). Independent oracle, recomputed in this file, row by row."""
    m = _tl()
    for n in (16, 64):
        X, Y = m.bed(n=n, n_instances=32, seed=1)
        assert X.shape[0] == Y.shape[0] == 32 and X.shape[1] == n, \
            "bed at n=%d returned shapes %r / %r" % (n, X.shape, Y.shape)
        assert Y.shape[1] == len(m.TARGETS) == 2
        worst = 0.0
        for i in range(32):
            worst = max(worst, abs(float(Y[i, 1]) - n * float(Y[i, 0])))
            assert 0.0 <= float(Y[i, 0]) <= 1.0, \
                "row %d: the intensive target %r is not a probability" % (i, Y[i, 0])
        assert worst < 1e-9, \
            "n=%d: the extensive target is not n x the intensive one (%r)" % (n, worst)


def test_the_intensive_target_is_length_invariant_and_the_extensive_one_is_not():
    """CLAUSE (c). The theorem hypothesis, CHECKED ON THE DATA.

    A normalised readout is a convex combination of per-token values, so it lives
    in a bounded interval independent of n. That only kills an extensive target
    if the target really grows, and only spares the intensive one if that really
    does not -- both measured here rather than assumed.
    """
    m = _tl()
    means = {}
    for n in (16, 64, 128):
        _, Y = m.bed(n=n, n_instances=256, seed=2)
        means[n] = (float(Y[:, 0].mean()), float(Y[:, 1].mean()))
    spread = max(a for a, _ in means.values()) - min(a for a, _ in means.values())
    assert spread < 0.02, \
        "the intensive target moved %r across lengths: it is not intensive" % spread
    assert means[64][1] > 3.5 * means[16][1] and means[128][1] > 7.0 * means[16][1], \
        "the extensive target does not scale with n: %r" % (means,)


# ---------------------------------------------------------------------------
# 4. THE CALIBRATION, PRE-REGISTERED, BOTH DIRECTIONS
# ---------------------------------------------------------------------------

def test_the_closed_form_prediction_is_available_before_any_run():
    """CLAUSE (d). A pre-registration that needs the answer is not one.

    The predicted magnitudes are arithmetic on (n0, n) alone -- no data, no
    model, no fit -- so they can be written down before the bed is built.
    """
    m = _tl()
    for n, want_soft, want_lin in ((64, 0.75, 3.0), (128, 0.875, 7.0)):
        pr = m.predicted_single_exponent_error(16, n)
        assert abs(pr["softmax_extensive"] - want_soft) < 1e-12
        assert abs(pr["linear_intensive"] - want_lin) < 1e-12


def test_the_softmax_head_wins_the_intensive_control(capsys):
    """CLAUSE (d), direction one. If this fails the baseline is broken and the
    whole test is VOID rather than negative."""
    rep = _demo(capsys)[0]["report"]
    losers = ("all-linear", "lookup-table", "marginal-frequency", "learned-beta")
    for L in rep["test_lengths"]:
        best_loser = min(rep["nrmse"][a][L][0] for a in losers)
        assert rep["nrmse"]["all-softmax"][L][0] < best_loser, (
            "VOID, not negative: the softmax head lost the INTENSIVE control at "
            "n=%d (%.4f against %.4f). A baseline that cannot win the bar it is "
            "supposed to win measures nothing."
            % (L, rep["nrmse"]["all-softmax"][L][0], best_loser))


def test_the_softmax_head_loses_the_extensive_plant_as_the_cited_theorem_predicts(capsys):
    """CLAUSE (d), direction two. The instrument, not the finding.

    If this fails the hypothesis of the cited theorem is unmet on this bed -- the
    extensive target is leaking, or the bar is not extensive -- and the test is
    VOID rather than negative.
    """
    m = _tl()
    rep = _demo(capsys)[0]["report"]
    soft = rep["nrmse"]["all-softmax"]
    assigned = rep["nrmse"]["assigned-oracle"]
    for L in rep["test_lengths"]:
        assert soft[L][1] > assigned[L][1] and soft[L][1] > 0.25, (
            "VOID, not negative: the softmax head did not lose the EXTENSIVE "
            "plant at n=%d (%.4f against the assigned head %.4f)"
            % (L, soft[L][1], assigned[L][1]))
        pred = m.predicted_single_exponent_error(rep["train_length"], L)
        assert abs(soft[L][1] - pred["softmax_extensive"]) < 0.05, (
            "the bed reproduces the SIGN but not the SCALE at n=%d: measured "
            "%.4f against a closed form of %.4f. Usable, but it must be said."
            % (L, soft[L][1], pred["softmax_extensive"]))


def test_the_linear_head_loses_the_intensive_bar(capsys):
    """CLAUSE (d). The mirror of the plant: one exponent cannot serve both."""
    rep = _demo(capsys)[0]["report"]
    lin = rep["nrmse"]["all-linear"]
    assigned = rep["nrmse"]["assigned-oracle"]
    for L in rep["test_lengths"]:
        assert lin[L][0] > assigned[L][0], (
            "VOID, not negative: the linear head did not lose the INTENSIVE bar "
            "at n=%d (%.4f against %.4f)" % (L, lin[L][0], assigned[L][0]))


def test_both_failures_grow_with_n(capsys):
    """CLAUSE (e). A constant gap is an offset, not a length failure."""
    rep = _demo(capsys)[0]["report"]
    four, eight = rep["test_lengths"]
    soft, lin = rep["nrmse"]["all-softmax"], rep["nrmse"]["all-linear"]
    assert soft[eight][1] > soft[four][1], (
        "the softmax extensive error did not grow: %.4f at n=%d, %.4f at n=%d"
        % (soft[four][1], four, soft[eight][1], eight))
    assert lin[eight][0] > lin[four][0], (
        "the linear intensive error did not grow: %.4f at n=%d, %.4f at n=%d"
        % (lin[four][0], four, lin[eight][0], eight))


def test_the_assigned_mask_holds_both_bars(capsys):
    """CLAUSE (d). The instance own prediction, held to a stated bar."""
    m = _tl()
    rep = _demo(capsys)[0]["report"]
    for L in rep["test_lengths"]:
        for c, name in enumerate(m.TARGETS):
            got = rep["nrmse"]["assigned-oracle"][L][c]
            assert got < 0.15, (
                "the assigned head does not hold the %s bar at n=%d: %.4f"
                % (name, L, got))


def test_the_exponent_alone_is_not_the_whole_assignment(capsys):
    """CLAUSE (d), the round correction. The bias control has to fire.

    The instance predicted exactness from the exponent alone. It is not exact,
    and the residual is the additive output constant rather than the length: the
    bias-free arm on the same mask must collapse it.
    """
    stats, _ = _demo(capsys)
    four = stats["report"]["test_lengths"][0]
    free = stats["report"]["nrmse"]["assigned-oracle"][four][1]
    assert stats["nobias_extensive_4x"] < 0.25 * free, (
        "the bias control did not fire: %.4f against %.4f, so the residual is "
        "something other than the constant"
        % (stats["nobias_extensive_4x"], free))


# ---------------------------------------------------------------------------
# 5. THE OWNER COUNTER, SCORED RATHER THAN ARGUED WITH
# ---------------------------------------------------------------------------

def test_the_learned_beta_head_is_compared_at_printed_parameter_counts(capsys):
    """CLAUSE (f). Matched counts or it is a tie, by the owner own kill rule.

    The learned head carries exactly one extra scalar per output coordinate --
    the exponent it has to discover -- and that difference is the experiment, so
    it is printed rather than hidden. The test does not require a win.
    """
    m = _tl()
    stats, out = _demo(capsys)
    rep = stats["report"]
    p_assigned = rep["n_params"]["assigned-oracle"]
    p_learned = rep["n_params"]["learned-beta"]
    assert p_assigned > 0 and p_learned > 0
    assert p_learned - p_assigned == len(m.TARGETS), (
        "the learned head carries %d extra parameters, not one exponent per "
        "coordinate: %d against %d"
        % (p_learned - p_assigned, p_learned, p_assigned))
    assert str(p_assigned) in out and str(p_learned) in out, \
        "a comparison was published without both parameter counts beside it"
    assert rep["verdict"] in ("WIN", "TIE"), rep["verdict"]
    assert isinstance(rep["verdict_because"], str) and rep["verdict_because"].strip()


def test_the_published_verdict_follows_from_the_numbers_not_from_prose(capsys):
    """CLAUSE (f). The verdict is recomputed here from the same table."""
    rep = _demo(capsys)[0]["report"]
    learned = rep["nrmse"]["learned-beta"]
    assigned = rep["nrmse"]["assigned-oracle"]
    worst_learned = max(learned[L][c] for L in rep["test_lengths"] for c in (0, 1))
    worst_assigned = max(assigned[L][c] for L in rep["test_lengths"] for c in (0, 1))
    mine = "WIN" if worst_learned > rep["tie_band"] * max(worst_assigned, 1e-12) else "TIE"
    assert mine == rep["verdict"], (
        "the module publishes %r, the same rule on the same table gives %r "
        "(worst learned %.4f, worst assigned %.4f, band %.2f)"
        % (rep["verdict"], mine, worst_learned, worst_assigned, rep["tie_band"]))


def test_the_learned_head_was_not_merely_under_trained(capsys):
    """CLAUSE (f). A win over an under-trained baseline is not a win.

    The learned head is given the best of a sweep by TRAINING loss at each seed,
    so its failure has to be at the test lengths and not on the training set.
    """
    m = _tl()
    assert len(m.LEARNED_SWEEP) >= 2, "the learned head got no sweep at all"
    rep = _demo(capsys)[0]["report"]
    assert rep["train_loss"]["learned-beta"] < 0.05, (
        "the learned head did not fit the TRAINING length (%.6f): its test "
        "failure cannot be called under-determination"
        % rep["train_loss"]["learned-beta"])
    for b in rep["per_seed_beta"]["learned-beta"]:
        assert 0.0 <= b[0] <= 1.0 and 0.0 <= b[1] <= 1.0


# ---------------------------------------------------------------------------
# 6. THE SIMPLEST BASELINES FIRST
# ---------------------------------------------------------------------------

def test_the_untrained_baselines_are_scored_before_anything_trained(capsys):
    """CLAUSE L-SIMPLE. A lookup table and a marginal predictor, first in the
    table, on the same rows as everything else."""
    rep = _demo(capsys)[0]["report"]
    arms = rep["arms"]
    assert arms[0] == "lookup-table" and arms[1] == "marginal-frequency", \
        "the untrained baselines are not first in the table: %r" % (arms,)
    for arm in ("lookup-table", "marginal-frequency"):
        assert rep["n_params"][arm] == 0
        for L in rep["test_lengths"]:
            assert rep["nrmse"][arm][L][1] > rep["nrmse"]["assigned-oracle"][L][1], \
                "%s beat the assigned head on the extensive bar at n=%d" % (arm, L)


# ---------------------------------------------------------------------------
# 7. THE MARS ATTACK: N THROUGH A POSITION
# ---------------------------------------------------------------------------

def test_the_inputs_carry_nothing_about_n_and_the_planted_leak_is_seen_to_fire(capsys):
    """CLAUSE (g). A census that cannot detect a handed leak is decoration."""
    stats = _demo(capsys)[0]
    cen = stats["census"]
    assert cen["mean_r2"] < 0.05, \
        "the per-token feature means already predict n at R2 = %.4f" % cen["mean_r2"]
    assert cen["leak_r2"] > 0.5, \
        "the planted positional leak was not detected: R2 = %.4f" % cen["leak_r2"]
    assert cen["extreme_r2"] > cen["mean_r2"], \
        "the order-statistic channel was not measured at all"


def test_the_n_oracle_control_converts_so_the_missing_quantity_is_named(capsys):
    """CLAUSE (g). The positive control for the leak channel.

    Handing the softmax head the true n must collapse the extensive bar. If it
    does not, the failure is not the factor n and the whole attack is aimed at
    the wrong thing.
    """
    stats = _demo(capsys)[0]
    four = stats["report"]["test_lengths"][0]
    soft = stats["report"]["nrmse"]["all-softmax"][four][1]
    assert stats["oracle_n_extensive_4x"] < 0.05, (
        "the n-oracle control did not fire: %.4f, against the softmax head %.4f"
        % (stats["oracle_n_extensive_4x"], soft))
    assert stats["leak_extensive_4x"] < soft, \
        "a trained positional feature did not help on the extensive bar at all"


def test_the_hull_bound_hypothesis_is_checked_on_the_models_own_values(capsys):
    """CLAUSE L-CLASS. A win counts only against a theorem whose hypotheses are
    checked on the data and printed."""
    rep = _demo(capsys)[0]["report"]
    mu = rep["max_abs_u"]["all-softmax"]
    train = rep["train_length"]
    eight = rep["test_lengths"][1]
    assert mu[eight] < 3.0 * mu[train], (
        "the head own value range grew %.2fx from n=%d to n=%d: the hull bound "
        "is not fixed and the theorem does not apply here"
        % (mu[eight] / max(mu[train], 1e-12), train, eight))
    truth = rep["truth_mean"]
    assert truth[eight][1] / max(truth[train][1], 1e-12) > 7.0, \
        "the extensive target did not grow: the hypothesis is unmet"


# ---------------------------------------------------------------------------
# 8. THE CLAIM IS SCOPED AND THE PRIOR ART IS CITED
# ---------------------------------------------------------------------------

def test_every_cannot_sentence_is_scoped_and_the_prior_art_is_named(capsys):
    """CLAUSE (h). An unscoped "softmax cannot" is refuted in one line.

    The replication is cited, the bound on the negative claim is cited, and the
    nearest prior work to the delta is cited -- in the module own docstring,
    where a reader meets them, and in the run.
    """
    m = _tl()
    doc = m.__doc__ or ""
    for arxiv in ("2410.01104", "2406.04267", "2511.20038", "2202.04643"):
        assert arxiv in doc, "the module docstring does not cite arXiv:%s" % arxiv
    out = _demo(capsys)[1]
    for arxiv in ("2410.01104", "2406.04267", "2511.20038", "2202.04643"):
        assert arxiv in out, "the run does not print arXiv:%s" % arxiv
    for phrase in ("scratchpad", "REPLICAT"):
        assert phrase.lower() in doc.lower(), \
            "the docstring carries no %r: the claim is unscoped" % phrase
    assert "sweep_expressivity.md:325-330" in doc, \
        "the in-tree [V] location of the replicated theorem is not given"


# ---------------------------------------------------------------------------
# 9. THE GUARD
# ---------------------------------------------------------------------------

#: A number WITH ITS BOUNDARIES. Token matching, not substring: `"7.9" in out`
#: is TRUE the moment a run prints 27.94, so a substring test passes any
#: fabricated number sharing digits with a real one. The inner (?:\.\d+)* keeps
#: a dotted version string as ONE token. The trailing guard is (?!\w)(?!\.\d)
#: and not (?![\w.]), because a sentence-final period must not hide the number
#: in front of it.
_MEASURED = re.compile(
    r"(?<![\w.])[+-]?\d+(?:\.\d+)*(?:[eE][+-]?\d+)?(?!\w)(?!\.\d)")


def _tokens(text):
    """Every number in `text` as a boundary-anchored token. COUNTER ONE."""
    return {t.lstrip("+") for t in _MEASURED.findall(text)}


def _tokens_scan(text):
    """The same token rule, written WITHOUT a regex. COUNTER TWO.

    Two spellings of one regex are one counter. This is a character scan, and
    the test below asserts it returns the SAME SET rather than merely the same
    count, so a divergence names the token.
    """
    out, n, i = set(), len(text), 0
    isd = str.isdigit

    def wordish(ch):
        return ch.isalnum() or ch == "_"

    while i < n:
        if not isd(text[i]):
            i += 1
            continue
        s = i
        if s > 0 and text[s - 1] in "+-":
            if s - 1 == 0 or not (wordish(text[s - 2]) or text[s - 2] == "."):
                s -= 1
        elif s > 0 and (wordish(text[s - 1]) or text[s - 1] == "."):
            while i < n and isd(text[i]):
                i += 1
            continue
        j = i
        while j < n and isd(text[j]):
            j += 1
        while j + 1 < n and text[j] == "." and isd(text[j + 1]):
            j += 1
            while j < n and isd(text[j]):
                j += 1
        if j < n and text[j] in "eE":
            k = j + 1
            if k < n and text[k] in "+-":
                k += 1
            if k < n and isd(text[k]):
                while k < n and isd(text[k]):
                    k += 1
                j = k
        ok = not (j < n and wordish(text[j]))
        if j + 1 < n and text[j] == "." and isd(text[j + 1]):
            ok = False
        if ok:
            out.add(text[s:j].lstrip("+"))
        i = j
    return out


#: Set in the child pytest process so the own-file scan does not recurse.
_CHILD = "T_LENGTH_GUARD_CHILD"


def _own_test_run_output():
    """stdout of THIS file own tests, so numbers in THEIR docstrings are bound."""
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

    Scoped by __module__, not __all__: demo() prints every number and is absent
    from __all__, so an __all__-scoped scan misses exactly the thing being
    checked. The same filter keeps numpy and torch docstrings out without any
    NUMBER ever being exempted -- a scope rule, not an exemption list.
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
            if getattr(member, "__module__", None) != m.__name__:
                continue
            sub = getattr(member, "__doc__", None)
            if isinstance(sub, str) and sub.strip() and attr != "__doc__":
                docs.append(("%s.%s" % (name, attr), sub))
    return docs


def test_every_measured_number_in_a_docstring_is_printed_by_a_run(capsys):
    """No exemption list. A number no run prints cannot be checked by anyone."""
    if os.environ.get(_CHILD):
        pytest.skip("child of the own-file scan; the parent does the scanning")
    out = _demo(capsys)[1]
    m = _tl()
    run = _tokens(out) | _tokens(_own_test_run_output())
    assert "2511.20038" in run, "the run stopped printing the bounding citation"
    assert "20038" not in run, \
        "SUBSTRING VACUITY: a fragment of an arXiv id is being read as a number"
    assert "40317" not in run, "the absent-integer control was found in the run"
    assert "77.31" not in run, "the absent-decimal control was found in the run"

    docs = _module_docstrings(m) + _module_docstrings(sys.modules[__name__])
    missing = [(where, tok) for where, doc in docs
               for tok in _tokens(doc) if tok not in run]
    checked = len({tok for _, doc in docs for tok in _tokens(doc)})
    print("\n  %d distinct numbers across %d docstrings, %d missing from the run"
          % (checked, len(docs), len(missing)))
    assert not missing, ("these docstring numbers are printed by no run:\n    "
                         + "\n    ".join("%s: %s" % (w, t) for w, t in missing))


def test_the_coverage_figure_is_bound_by_two_independently_written_counters(capsys):
    """The coverage figure is a number in prose like any other.

    The module counts its own docstrings and numbers; this file counts them again
    with a scan sharing no code with the regex; the two must return the SAME SET
    on the run text and the same counts on the docstrings; both figures must be
    printed; and both must clear the floor.
    """
    m = _tl()
    out = _demo(capsys)[1]
    a, b = _tokens(out), _tokens_scan(out)
    assert a == b, ("the two counters disagree on the run own output: %r"
                    % (sorted(a ^ b)[:12],))
    mine = _module_docstrings(m)
    theirs = m.docstring_numbers()
    n_mine = len({t for _, d in mine for t in _tokens(d)})
    assert theirs["n_docstrings"] == len(mine), (
        "the module counts %d docstrings, an independent scan counts %d"
        % (theirs["n_docstrings"], len(mine)))
    assert theirs["n_numbers"] == n_mine, (
        "the module counts %d numbers, an independent scan counts %d"
        % (theirs["n_numbers"], n_mine))
    for figure in (theirs["n_docstrings"], theirs["n_numbers"]):
        assert str(figure) in out, \
            "the coverage figure %d is not printed by the run" % figure
    assert len(mine) >= MIN_DOCSTRINGS and n_mine >= MIN_NUMBERS, \
        "the floor is not met: %d docstrings, %d numbers" % (len(mine), n_mine)


def test_the_published_values_are_recomputed_not_literals(capsys):
    """PRINTED IS NOT MEASURED.

    A literal typed into a print statement satisfies the docstring guard
    trivially. Every published value is pinned HERE to an expression that
    recomputes it from the module own table in the same run, one position at a
    time and never with allclose over a whole list.
    """
    m = _tl()
    stats, out = _demo(capsys)
    assert isinstance(stats, dict) and stats, "demo() must return what it measured"
    rep = stats["report"]
    four, eight = rep["test_lengths"]

    assert stats["head_commit"] == m.STATED_AT_COMMIT and stats["head_commit"] in out
    assert stats["machine_id"] == m.MACHINE_ID and stats["machine_id"] in out

    for arm in rep["arms"]:
        for L in rep["lengths"]:
            for c in range(len(m.TARGETS)):
                got = rep["nrmse"][arm][L][c]
                want = m.nrmse(rep["truth"][L][:, c], rep["pred"][arm][L][:, c])
                assert got == want, \
                    "%s at n=%d target %d: %r != %r" % (arm, L, c, got, want)

    assert stats["softmax_extensive_factor_4x"] == (
        rep["nrmse"]["all-softmax"][four][1] / rep["nrmse"]["assigned-oracle"][four][1])
    assert stats["linear_intensive_factor_4x"] == (
        rep["nrmse"]["all-linear"][four][0] / rep["nrmse"]["assigned-oracle"][four][0])
    assert stats["softmax_extensive_factor_8x"] == (
        rep["nrmse"]["all-softmax"][eight][1] / rep["nrmse"]["assigned-oracle"][eight][1])
    assert stats["owner_4x"] == OWNER_4X, \
        "the owner estimates were altered: %r" % (stats["owner_4x"],)
    assert stats["diff_vs_owner"] == (
        stats["softmax_extensive_factor_4x"] - OWNER_4X[0],
        stats["linear_intensive_factor_4x"] - OWNER_4X[1])

    for key in ("n_params_assigned", "n_params_learned"):
        assert str(stats[key]) in out, "%s = %r is not printed" % (key, stats[key])
    assert stats["n_params_learned"] - stats["n_params_assigned"] == len(m.TARGETS)
    assert stats["verdict"] in out and stats["verdict"] == rep["verdict"]
    for key in ("nobias_extensive_4x", "leak_extensive_4x", "oracle_n_extensive_4x"):
        assert ("%.4f" % stats[key]) in out, "%s = %r is not printed" % (key, stats[key])
    for key in ("n_docstrings", "n_numbers"):
        assert str(stats[key]) in out
    print("\n  %d published values re-derived against the module in this run"
          % len(stats))


def test_the_demo_finishes_inside_the_budget_and_ends_with_the_exact_line(capsys):
    """The budget is 300 s on CPU, and the last line is fixed by contract."""
    m = _tl()
    stats, out = _demo(capsys)
    out = out.rstrip("\n")
    print("\n  demo() took %.2f s" % stats["elapsed_s"])
    assert stats["elapsed_s"] < 300.0, "demo() took %.1f s" % stats["elapsed_s"]
    assert math.isfinite(stats["elapsed_s"])
    assert out.endswith("ALL SELF-CHECKS PASSED"), \
        "the last line is %r" % (out.splitlines()[-1] if out else "",)
    assert "RUN:" in (m.__doc__ or ""), "the module docstring carries no RUN: line"
    for name in m.__all__:
        assert hasattr(m, name), "__all__ advertises a missing name %r" % (name,)
    first = out.splitlines()[0]
    assert m.STATED_AT_COMMIT in first and m.MACHINE_ID in first, \
        "L-PROV: the first line is %r" % (first,)


# ---------------------------------------------------------------------------
# 10. THE MEASURED MASK, WIRED
# ---------------------------------------------------------------------------

def test_the_measured_mask_is_taken_as_a_vector_and_validated_at_the_boundary():
    """CLAUSE (a), the wiring. The mask crosses as DATA and is checked as data.

    The producing file emits a vector along a named axis with refused
    coordinates kept as None, precisely so the axis cannot silently shorten.
    Every one of those properties is a way for the wiring to be wrong while the
    shapes still look plausible, so each is refused by name here.
    """
    m = _tl()
    good = dict(axis="D", axis_len=3, coords=["a", "b", "c"],
                beta=[1.0, 0.0, None], alpha=[0.0, 1.0, 0.5],
                refusals={"c": "MIXED-SCALING"}, n_assigned=2)
    mask, detail = m.mask_from_frozen(good, {"intensive": "a", "extensive": "b"})
    assert mask == (1.0, 0.0), mask
    assert detail["refused"] is None

    compacted = dict(good, beta=[1.0, 0.0], coords=["a", "b"], axis_len=3)
    with pytest.raises(ValueError):
        m.mask_from_frozen(compacted, {"intensive": "a", "extensive": "b"})
    renamed = dict(good, axis="S")
    with pytest.raises(ValueError):
        m.mask_from_frozen(renamed, {"intensive": "a", "extensive": "b"})
    dropped = dict(good, beta=[1.0, 0.0, 0.5], refusals={"c": "MIXED-SCALING"})
    with pytest.raises(ValueError):
        m.mask_from_frozen(dropped, {"intensive": "a", "extensive": "b"})

    # The rule is never REQUIRED: a missing one is a reason, not a crash.
    gone = m.load_measured_mask("ceqjepa.no_such_rule_module")
    assert gone["available"] is False and gone["frozen"] is None
    assert "no_such_rule_module" in gone["reason"]

    # A refused coordinate is a REFUSAL, never a fabricated exponent.
    onto_refused = {"intensive": "a", "extensive": "c"}
    mask2, detail2 = m.mask_from_frozen(good, onto_refused)
    assert mask2 is None and detail2["refused"], detail2
    assert "MIXED-SCALING" in detail2["refused"]


def test_beta_is_derived_from_alpha_on_this_bed_not_chosen(capsys):
    """The assigned mask is the dimensionally FORCED setting, not a tuned one.

    A read carries N^(1-beta), so the exponent of the count in the output is
    1 - beta and the assignment beta = 1 - alpha is bookkeeping. Measured here on
    THIS bed's own targets by a length sweep, with no model and no fit quality
    gated on anything.
    """
    m = _tl()
    alpha = m.target_alpha()
    assert abs(alpha["intensive"]) < 0.05, \
        "the intensive target is not flat in n: alpha = %.4f" % alpha["intensive"]
    assert abs(alpha["extensive"] - 1.0) < 0.05, \
        "the extensive target does not scale like n: alpha = %.4f" % alpha["extensive"]
    derived = tuple(round(1.0 - alpha[t], 6) for t in m.TARGETS)
    assert derived == (1.0, 0.0) or all(
        abs(d - o) < 0.05 for d, o in zip(derived, m.MASKS["assigned-oracle"])), \
        "beta = 1 - alpha does not reproduce the oracle mask: %r" % (derived,)


def test_the_measured_mask_row_is_scored_beside_the_oracle(capsys):
    """The whole DR-2 claim, as a row: does the rule recover the corner per
    coordinate from data alone?

    If the measured vector equals the oracle vector entry for entry the rows are
    identical BY CONSTRUCTION, and saying so is the honest reading; the
    information is in the vector comparison, not in the row. If it differs, the
    module must name the coordinate that differs and what it costs in each cell.
    """
    m = _tl()
    stats, out = _demo(capsys)
    wired = stats["measured_mask"]
    if not wired["available"]:
        pytest.skip("the assignment rule is not in the tree: %r" % wired["reason"])
    rep = stats["report"]
    assert "assigned-measured" in rep["arms"], "the measured row was not scored"
    assert rep["n_params"]["assigned-measured"] == rep["n_params"]["assigned-oracle"]
    assert wired["axis"] == "D" and wired["axis_len"] == 9, wired
    oracle = m.MASKS["assigned-oracle"]
    same = tuple(wired["mask"]) == tuple(oracle)
    assert wired["agrees_with_oracle"] == same
    for L in rep["test_lengths"]:
        for c in range(len(m.TARGETS)):
            got = rep["nrmse"]["assigned-measured"][L][c]
            ora = rep["nrmse"]["assigned-oracle"][L][c]
            if same:
                assert got == ora, (
                    "the masks are identical but the rows are not (%r against "
                    "%r at n=%d, target %d): the pipeline is not deterministic"
                    % (got, ora, L, c))
            else:
                assert wired["differs_at"], \
                    "the masks differ and no coordinate was named"
    assert "assigned-measured" in out


def test_no_fit_quality_is_gated_on_r_squared():
    """R2 measures residual against a SPREAD a flat coordinate does not have, so
    a threshold on it refuses exact power laws for being flat. Relayed from the
    producing file, and enforced here as an absence."""
    m = _tl()
    src = Path(m.__file__).read_text(encoding="utf-8")
    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    for bad in ("r2 >", "r2 <", "r_squared", "R2_MIN", "rsq"):
        assert bad not in code, "a fit quality is gated on %r" % bad


# ---------------------------------------------------------------------------
# 11. THE REFUSAL CHANNEL, EXERCISED END TO END
# ---------------------------------------------------------------------------

def test_a_refused_coordinate_is_refused_and_never_silently_defaulted():
    """The failure this whole channel exists to prevent, planted and caught.

    A careless caller writes `b if b is not None else 1.0` and a refusal becomes
    the softmax corner with no trace. The two paths are run side by side here:
    the careless one returns a NUMBER, the harness returns a REFUSAL carrying
    the rule's own reason, and the two reasons are distinguishable.
    """
    m = _tl()
    frozen = dict(axis="D", axis_len=4, coords=["ok", "mix", "osc", "x"],
                  beta=[1.0, None, None, 0.0], alpha=[0.0, 0.5, 0.0, 1.0],
                  refusals={"mix": "MIXED-SCALING", "osc": "NOT-A-POWER-LAW"},
                  n_assigned=2)
    for coord, reason in (("mix", "MIXED-SCALING"), ("osc", "NOT-A-POWER-LAW")):
        mapping = {"intensive": "ok", "extensive": coord}
        mask, detail = m.mask_from_frozen(frozen, mapping)
        assert mask is None, "a refused coordinate produced a corner: %r" % (mask,)
        assert reason in detail["refused"], detail["refused"]
        careless = m.silently_defaulted_mask(frozen, mapping)
        assert careless is not None and careless[1] == 1.0, careless
        assert careless != mask, "the plant and the guard agree: nothing is caught"


def test_the_refusal_bed_targets_are_built_like_the_coordinates_they_map_to():
    """The mapping is a CONSTRUCTION, so it is checked, not asserted.

    The MIXED target is a running total of a zero-mean per-token quantity -- the
    same construction the rule's half_walk coordinate is planted as -- so its
    norm grows like the square root and its alpha sits between the corners. The
    NOT-A-POWER-LAW target carries an amplitude that is a symmetric bump in
    log2(n), the same construction as osc_shape, so its norm is NOT monotone in
    n and no exponent describes it.
    """
    m = _tl()
    alpha = m.refusal_target_alpha()
    assert 0.35 < alpha["mixed"] < 0.65, \
        "the mixed target is not a half-walk: alpha = %.4f" % alpha["mixed"]
    norms = m.refusal_target_norms()
    n0, four, eight = m.N0, m.TEST_LENGTHS[0], m.TEST_LENGTHS[1]
    assert norms["mixed"][eight] > norms["mixed"][four] > norms["mixed"][n0], \
        "the mixed target does not grow at all: %r" % (norms["mixed"],)
    osc = norms["not-a-power-law"]
    assert abs(osc[eight] - osc[n0]) < 0.05 * osc[n0], (
        "the oscillating target does not return to its training amplitude at 8x, "
        "so it is not the planted bump: %r" % (osc,))
    assert abs(osc[four] - osc[n0]) > 0.2 * osc[n0], \
        "the oscillating target is flat: it would be a power law after all"


def test_forcing_a_corner_on_a_refused_coordinate_is_scored_both_ways(capsys):
    """What a reader LOSES by honouring the refusal, in the same nrmse units.

    Three arms per refused reason: force the softmax corner, force the linear
    corner, and refuse. The refusing arm must report a NUMBER for the coordinate
    that has a corner and a REFUSAL for the one that does not -- filling that
    cell with a default is the substitution this channel exists to prevent.
    """
    m = _tl()
    stats = _demo(capsys)[0]
    ref = stats["refusal"]
    for reason in ("MIXED-SCALING", "NOT-A-POWER-LAW"):
        rows = ref[reason]
        for arm in ("force-softmax", "force-linear", "refuse"):
            assert arm in rows["nrmse"], "%s has no %s arm" % (reason, arm)
        for L in m.TEST_LENGTHS:
            assert isinstance(rows["nrmse"]["refuse"][L][1], str), (
                "the refusing arm published a number for a refused coordinate at "
                "n=%d: %r" % (L, rows["nrmse"]["refuse"][L][1]))
            assert "REFUSED" in rows["nrmse"]["refuse"][L][1]
            assert isinstance(rows["nrmse"]["refuse"][L][0], float), \
                "the refusing arm refused the coordinate that HAS a corner"
            for arm in ("force-softmax", "force-linear"):
                assert isinstance(rows["nrmse"][arm][L][1], float)


def test_the_two_refusal_reasons_do_not_cost_the_same(capsys):
    """MIXED-SCALING means alpha exists between the corners; NOT-A-POWER-LAW
    means alpha has no referent. They must not cost the same, and the module has
    to publish which is which rather than leaving a reader to infer it."""
    m = _tl()
    stats = _demo(capsys)[0]
    ref = stats["refusal"]
    four = m.TEST_LENGTHS[0]
    def winner_per_length(reason):
        return {L: ("solo-linear"
                    if ref[reason]["nrmse"]["solo-linear"][L][1]
                    < ref[reason]["nrmse"]["solo-softmax"][L][1]
                    else "solo-softmax") for L in m.TEST_LENGTHS}

    mixed_win = winner_per_length("MIXED-SCALING")
    nopow_win = winner_per_length("NOT-A-POWER-LAW")
    assert len(set(mixed_win.values())) == 1, (
        "MIXED-SCALING has no consistent corner either: %r" % (mixed_win,))
    assert len(set(nopow_win.values())) > 1, (
        "the NOT-A-POWER-LAW ranking does not flip, so the two reasons are not "
        "being distinguished by this table: %r" % (nopow_win,))
    assert ref["verdict"]["mixed_has_a_good_corner"] is True
    assert ref["verdict"]["nopow_corner_ranking_flips"] is True
    assert ref["verdict"]["mixed_winner"] == mixed_win
    assert ref["verdict"]["nopow_winner"] == nopow_win
    assert isinstance(ref["verdict"]["says"], str) and ref["verdict"]["says"].strip()

    # The softmax corner on a HALF-WALK follows the count law exactly, which is
    # the evidence that the exponent the rule fits is not the exponent that
    # governs the readout error.
    for L in m.TEST_LENGTHS:
        want = m.predicted_single_exponent_error(m.N0, L)["softmax_extensive"]
        got = ref["MIXED-SCALING"]["nrmse"]["solo-softmax"][L][1]
        assert abs(got - want) < 0.01, (
            "the half-walk's softmax error is %.4f against the count law %.4f at "
            "n=%d" % (got, want, L))


# ---------------------------------------------------------------------------
# L-SURFACE. A check that prints and then aborts is a FAILED check.
#
# The founding instance is this module. From e3d56cb to 391a2d0, `python -m
# ceqjepa.t_length` printed
#
#     PROVENANCE commit c9a9434 (git says 391a2d0) ... -- both verified, not copied
#
# and then raised AssertionError on the very next line, because HEAD_COMMIT was
# a literal that had to equal HEAD and never did. Every value check in this file
# routes through demo(), so the suite read `21 failed, 13 passed in 5.53s` for
# two commits while asserting nothing. Two surfaces made that invisible: the
# word PROVENANCE and two commit hashes printed BEFORE the abort, and the fact
# that no test in this file ever asserted the module's exit status. Sibling
# suites already did -- tests/curvature/test_chess_steps.py:338 and
# tests/curvature/test_drift_null.py:546 -- and this one did not.
#
# Exit codes are asserted, never assumed, and never read through a pipeline: a
# shell `python -m ceqjepa.t_length 2>&1 | tail -3; echo $?` reports tail's
# status, which is 0 whatever the module did. That mistake was made in this
# repository while diagnosing this very defect.
# ---------------------------------------------------------------------------

def _self_check_ok(returncode, stdout):
    """The predicate: BOTH the exit status AND the terminal banner.

    Either half alone is satisfiable by a run that failed, which is what the
    planted negatives below demonstrate rather than assert.
    """
    lines = [ln for ln in stdout.strip().splitlines() if ln.strip()]
    return (returncode == 0
            and bool(lines)
            and lines[-1].strip() == "ALL SELF-CHECKS PASSED")


def _run_snippet(code):
    """A child process standing in for a module with a given surface."""
    p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, timeout=60)
    return p.returncode, p.stdout


def test_a_printed_banner_does_not_stand_in_for_an_exit_code():
    """The planted negatives, entering at the predicate's front door."""
    rc, out = _run_snippet(
        "print('PROVENANCE commit deadbee (git says cafef00)')\n"
        "print('ALL SELF-CHECKS PASSED')\n"
        "raise SystemExit(1)\n")
    assert rc == 1 and "ALL SELF-CHECKS PASSED" in out
    assert not _self_check_ok(rc, out), (
        "a process that prints the banner and exits 1 passed the predicate: "
        "the exit status is not being read")

    rc, out = _run_snippet(
        "print('PROVENANCE commit deadbee (git says cafef00)')\n"
        "raise SystemExit(0)\n")
    assert rc == 0 and "ALL SELF-CHECKS PASSED" not in out
    assert not _self_check_ok(rc, out), (
        "a process that exits 0 without finishing its checks passed the "
        "predicate: the banner is not being read")

    rc, out = _run_snippet("print('ALL SELF-CHECKS PASSED')\n")
    assert _self_check_ok(rc, out), (
        "the predicate rejects a clean run, so it cannot judge the module")


def test_the_module_self_check_exits_zero_and_ends_with_the_banner():
    """The check whose absence let this module abort unnoticed for two commits."""
    root = Path(__file__).resolve().parents[2]
    p = subprocess.run([sys.executable, "-m", "ceqjepa.t_length"], cwd=str(root),
                       capture_output=True, text=True, timeout=600)
    tail = "\n".join(p.stdout.strip().splitlines()[-3:])
    print("\n  python -m ceqjepa.t_length -> exit %d\n%s" % (p.returncode, tail))
    assert _self_check_ok(p.returncode, p.stdout), (
        "exit %d, last lines:\n%s\nstderr:\n%s"
        % (p.returncode, tail, p.stderr[-3000:]))


def test_the_provenance_line_reports_drift_instead_of_aborting_on_it():
    """STATED_AT_COMMIT records where numbers were measured; it never gates."""
    m = _tl()
    prov = m.provenance()
    assert set(prov) >= {"commit", "commit_ok", "commit_matches_stated"}, prov
    assert prov["commit_ok"], "git resolved no commit: %r" % (prov["commit"],)
    src = Path(m.__file__).read_text(encoding="utf-8")
    assert 'assert prov["commit_matches_stated"]' not in src, (
        "drift from the stated commit is being asserted again: that is the "
        "self-disabling gate this module was repaired to remove")

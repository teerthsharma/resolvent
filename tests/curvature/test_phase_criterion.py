"""A3: is a kNN cluster a PHASE because binding beats thermal, or because a
threshold was fitted and then dressed in thermodynamic vocabulary?

These tests are written before ceqjepa/phase_criterion.py exists. They assert
RELATIONS -- commensurability, model ordering, freeze mechanics, refusal -- and
never a magic constant, because a test that pins a number the module also prints
is a tautology dressed as a check. The two places a number IS pinned are the
IDENTITY tests at the bottom, and there the number is pinned to the EXPRESSION
that recomputes it from the run, which is the only pinning that can fail.

The whole suite reads ONE cached sweep (`m.report()`), so the curvature cost is
paid once, not once per test.

TESTS THAT POSTDATE THE RED, AND ARE THEREFORE UNBOUND BY IT. The recorded RED
is 5 failed + 1 passed + 21 errors = 27 tests; this file now ships 32; the
difference of 5 was written afterwards and no run of this file ever saw any of
them fail for the absence of the module:

    test_published_slope_ci_is_the_interval_the_train_points_recompute
    test_published_alternative_residuals_are_recomputed_from_the_same_points
    test_published_tlife_temperatures_are_recomputed_from_their_own_legs
    test_published_landau_well_positions_and_depths_are_recomputed
    test_the_k_control_is_repeated_at_a_second_bed_size

The first four came from the coordinator's identity-guard clause, the fifth from
the guard-repair pass. An audit of this file named only the first four, against
a shipped total the guard-repair pass had already moved; that audit's own two
figures are deliberately not quoted here, because they came from a run that was
not this one and nothing here can produce them. No RED was manufactured after
the fact to cover any of these: their bite rests on the mutation battery
instead. Every count above is reconciled against a LIVE count of this file and
printed by the docstring guard below, and the same figures are on the board.
"""

import re
import subprocess
import sys
import time

import numpy as np
import pytest

import ceqjepa.curvature as cv


#: The RED recorded on the board for this file, kept here so the guard can
#: reconcile it against a LIVE count of the tests actually shipped. These three
#: are historical -- they describe a pytest run, which is not something the
#: module's demo can print -- and they are labelled as such wherever printed.
RED_FAILED, RED_PASSED, RED_ERRORS = 5, 1, 21


def _pc():
    import ceqjepa.phase_criterion as m
    return m


def _shipped_tests():
    """Count the test functions this file actually ships, at run time.

    The shipped count is MEASURED here rather than typed, so the arithmetic in
    the file docstring is reconciled against the file instead of against memory:
    an audit of this file used a count that a later pass had already moved.
    """
    mod = sys.modules[__name__]
    return sum(1 for n, o in vars(mod).items()
               if n.startswith("test_") and callable(o))


@pytest.fixture(scope="module")
def rep():
    return _pc().report()


# ---------------------------------------------------------------------------
# 1. THE CRITERION IS COMMENSURABLE, OR IT IS A FITTED THRESHOLD
# ---------------------------------------------------------------------------

def test_binding_and_thermal_carry_declared_units_and_they_match():
    """A criterion comparing two quantities with different units is not a
    criterion. The module must DECLARE the unit of every factor and the two
    sides of the comparison must land in the same one.
    """
    m = _pc()
    u = m.UNITS
    for name in ("binding", "thermal", "temperature", "entropy", "coupling",
                 "separation", "reduced_temperature"):
        assert name in u, "no declared unit for %r" % name
    assert u["binding"] == u["thermal"], (
        "binding is in %r and thermal is in %r -- the comparison is not a "
        "criterion, it is a fitted threshold with extra steps"
        % (u["binding"], u["thermal"]))
    assert u["binding"] == "1", (
        "the two sides must be dimensionless for the comparison to survive a "
        "change of length unit; got %r" % u["binding"])


def test_the_owners_four_factor_form_is_reported_as_non_commensurable():
    """coupling x density vs temperature x entropy is L^-d against L^1.

    The addendum states the criterion in four factors. The module is required to
    carry the dimensional arithmetic for that form and to report it FAILING,
    rather than quietly substituting the form that works.
    """
    m = _pc()
    d = m.FOUR_FACTOR_DIMENSIONS
    assert d["binding_exponent"] != d["thermal_exponent"], (
        "the four-factor form is being reported as commensurable; it is not")
    assert isinstance(d["verdict"], str) and "not commensurable" in d["verdict"]


def test_the_bed_is_exactly_scale_invariant_so_the_linear_form_is_forced():
    """The load-bearing test of the whole report.

    kNN ranking is scale-free and the hop metric discards length, so a bed whose
    every length is proportional to (sep, sigma) must give BITWISE identical
    curvature under a global rescale. If that holds, sep_crit = c*sigma is not a
    fitted form -- it is forced, because sep and sigma are the only two lengths
    and their ratio is the only dimensionless combination. Any other form needs a
    third length that the bed does not contain.
    """
    m = _pc()
    X1, l1 = m.two_phase_bed(seed=7, sep=6.0, sigma=1.0)
    X2, l2 = m.two_phase_bed(seed=7, sep=60.0, sigma=10.0)
    assert np.abs(10.0 * X1 - X2).max() < 1e-12, "the bed itself is not scaling"
    assert (l1 == l2).all()

    a, b = m.bed_measure(X1, l1), m.bed_measure(X2, l2)
    for key in ("U", "S", "q"):
        assert a[key] == b[key], (
            "%s moved under a pure rescale: %.17g vs %.17g. The pipeline has a "
            "hidden length in it and the linear form is no longer forced."
            % (key, a[key], b[key]))
    assert a["n_edges"] == b["n_edges"]


def test_the_shipped_two_cluster_bed_is_not_scale_invariant():
    """Why this module carries its own bed instead of importing curvature's.

    curvature.two_cluster_bed hardcodes two absolute lengths -- the 1.5 subtracted
    from the isthmus span and the 0.05 isthmus jitter -- so it is NOT a two-length
    bed and a critical line measured on it is contaminated by a third scale. This
    test READS that fact and does not repair it: curvature.py is not this module's
    to touch.
    """
    X1, _ = cv.two_cluster_bed(seed=7, sigma=0.8, sep=9.0)
    X2, _ = cv.two_cluster_bed(seed=7, sigma=8.0, sep=90.0)
    gap = float(np.abs(10.0 * X1 - X2).max())
    assert gap > 1e-6, (
        "curvature.two_cluster_bed now scales exactly (max dev %.3e); the reason "
        "this module ships its own bed has gone away and should be retired" % gap)


# ---------------------------------------------------------------------------
# 2. T-CRIT: THE ALTERNATIVES ARE FITTED ON THE SAME DATA AND SEEN TO LOSE
# ---------------------------------------------------------------------------

def test_all_three_forms_are_fitted_on_the_identical_train_points(rep):
    """A linear fit reported alone proves nothing about the other two."""
    f = rep["tcrit"]["fits"]
    assert set(f) == {"linear", "quadratic", "constant"}
    n = {k: f[k]["n"] for k in f}
    assert len(set(n.values())) == 1, ("the three forms were fitted on different "
                                       "numbers of points: %r" % n)
    assert n["linear"] >= 6, "only %d train points" % n["linear"]
    for k in f:
        for field in ("c", "ci_lo", "ci_hi", "rss", "aic", "resid_rms"):
            assert field in f[k], "%s fit has no %s" % (k, field)


def test_the_linear_form_wins_on_residual_and_on_aic(rep):
    """The owner's test says a wrong form must FAIL. That is only demonstrated
    if the wrong forms are fitted and seen to lose."""
    f = rep["tcrit"]["fits"]
    for wrong in ("quadratic", "constant"):
        assert f["linear"]["rss"] < f[wrong]["rss"], (
            "sep = c*T does not beat sep = %s on residual: %.6g vs %.6g"
            % (wrong, f["linear"]["rss"], f[wrong]["rss"]))
        assert f["linear"]["aic"] < f[wrong]["aic"], (
            "sep = c*T does not beat sep = %s on AIC: %.4f vs %.4f"
            % (wrong, f["linear"]["aic"], f[wrong]["aic"]))
    assert rep["tcrit"]["winner"] == "linear"


def test_the_slope_has_a_finite_ci_that_excludes_zero(rep):
    lin = rep["tcrit"]["fits"]["linear"]
    assert lin["ci_lo"] > 0.0, "the slope CI touches zero: [%.4f, %.4f]" % (
        lin["ci_lo"], lin["ci_hi"])
    assert lin["ci_lo"] < lin["c"] < lin["ci_hi"]
    assert np.isfinite(lin["resid_rms"])


# ---------------------------------------------------------------------------
# 3. PRE-REGISTRATION: THE DIGEST IS OVER INPUTS, AND A REFIT FIRES
# ---------------------------------------------------------------------------

def test_the_digest_is_over_inputs_not_outputs(rep):
    """A digest over the fitted parameters cannot distinguish a freeze from an
    honest refit: refit, re-digest, and the record still 'verifies'. So the
    digest must move when a MEASURED INPUT moves and must not move when a FITTED
    OUTPUT moves.
    """
    m = _pc()
    spec = rep["freeze"]["spec"]
    base = m.freeze_digest(spec)

    out = dict(spec)
    out["fitted_c"] = 999.0                       # an OUTPUT, must not count
    assert m.freeze_digest(out) == base, (
        "the digest moved when a fitted output was added: it is digesting "
        "outputs, and a refit would sail through the freeze check")

    bad = m.perturb_one_measured_value(spec, delta=1e-9)
    assert m.freeze_digest(bad) != base, (
        "a measured input moved by 1e-9 and the digest did not: the freeze is "
        "over something other than the data the line was fitted on")


def test_the_frozen_digest_in_the_source_matches_the_train_grid(rep):
    """The pre-registration itself. FROZEN_DIGEST is a literal in the source; if
    the train grid, the bed, the seeds or the measured values move, this fires
    and the shipped line is known to have been fitted on other data."""
    m = _pc()
    assert rep["freeze"]["digest"] == m.FROZEN_DIGEST, (
        "the shipped line was fitted on a grid whose digest is %s but the source "
        "pre-registers %s" % (rep["freeze"]["digest"], m.FROZEN_DIGEST))


def test_a_planted_refit_on_the_test_grid_is_caught(rep):
    """The owner strikes any line fitted after seeing the test grid. Plant that
    exact sin and require the check to fire."""
    m = _pc()
    honest = m.check_freeze(rep["freeze"]["spec"], rep["tcrit"]["fits"]["linear"]["c"])
    assert honest["ok"] is True, honest["reason"]

    cheat_spec, cheat_c = m.planted_refit_on_test_grid()
    caught = m.check_freeze(cheat_spec, cheat_c)
    assert caught["ok"] is False, (
        "a line refitted on the held-out grid passed the freeze check")
    assert "digest" in caught["reason"].lower()


def test_the_frozen_line_predicts_the_held_out_grid(rep):
    """Evaluated on a TEST grid the fit never saw, at temperatures and seeds the
    train grid does not contain."""
    h = rep["tcrit"]["heldout"]
    assert h["n"] >= 4, "only %d held-out points" % h["n"]
    assert not (set(h["temperatures"]) & set(rep["tcrit"]["train_temperatures"])), (
        "the held-out grid shares a temperature with the train grid: %r vs %r"
        % (h["temperatures"], rep["tcrit"]["train_temperatures"]))
    assert not (set(h["seeds"]) & set(rep["tcrit"]["train_seeds"])), \
        "the held-out grid shares a seed with the train grid"
    lin = rep["tcrit"]["fits"]["linear"]
    assert lin["ci_lo"] <= h["c_implied"] <= lin["ci_hi"], (
        "the held-out data implies c = %.4f, outside the frozen CI [%.4f, %.4f]"
        % (h["c_implied"], lin["ci_lo"], lin["ci_hi"]))


# ---------------------------------------------------------------------------
# 4. LANDAU: WELLS WITH POSITIONS AND DEPTHS, NOT A PICTURE
# ---------------------------------------------------------------------------

def test_the_free_energy_is_reported_as_numbers_with_two_wells(rep):
    """Positions and depths, and the second well weighed against seed noise.

    Local minima of a discrete sweep appear from sampling noise alone, so a well
    count is worthless without the noise it is being read against. The module
    must publish both and its own verdict must follow its own arithmetic.
    """
    L = rep["landau"]
    assert len(L["ladder"]) >= 6
    for row in L["ladder"]:
        for field in ("tau", "argmin_s", "F_min", "minima"):
            assert field in row
        assert np.isfinite(row["F_min"])
        for s_pos, depth in row["minima"]:
            assert np.isfinite(s_pos) and np.isfinite(depth)
    assert any(len(row["minima"]) >= 2 for row in L["ladder"]), (
        "no temperature on the ladder has two local minima: there are no wells "
        "for a minimum to move between, and F = E - T*S is decoration here")
    assert np.isfinite(L["well_noise"]) and L["n_wells_max"] >= 1
    assert (L["wells_exceed_noise"] is True) == bool(
        L["n_wells_max"] >= 2 and L["well_gap"] > L["well_noise"]), (
        "the bimodality verdict %r contradicts its own numbers: %d wells, gap %r "
        "against noise %.6f" % (L["wells_exceed_noise"], L["n_wells_max"],
                                L["well_gap"], L["well_noise"]))
    assert L["transition_order"] == ("first" if L["wells_exceed_noise"]
                                     else "second"),         "the stated transition order does not follow from the measured wells"
    assert len(L["order_verdict"]) > 60


def test_the_landau_minimum_actually_moves_between_wells(rep):
    L = rep["landau"]
    positions = [row["argmin_s"] for row in L["ladder"]]
    assert len(set(positions)) >= 2, (
        "the free-energy minimum sits at %r at every temperature" % positions[0])
    assert np.isfinite(L["tau_jump"]), "no jump temperature was located"
    assert L["jump_from"] != L["jump_to"]


def test_the_landau_jump_is_compared_to_the_criterion_crossing(rep):
    """Two routes to one number, or an honest statement that they disagree.

    The criterion's crossing and the Landau well-jump are different statements --
    a local balance at the configuration's own reduced temperature, against a
    global comparison across the family -- and the module must print the gap
    rather than let the reader assume they coincide.
    """
    L = rep["landau"]
    assert "tau_crit_from_criterion" in L and "gap_to_criterion" in L
    assert np.isfinite(L["gap_to_criterion"])
    assert abs(L["gap_to_criterion"]
               - (L["tau_jump"] - L["tau_crit_from_criterion"])) < 1e-12


# ---------------------------------------------------------------------------
# 5. T-LIFE: THE BUDGET, AND THE ROUTE THAT REFUSED
# ---------------------------------------------------------------------------

def test_t_life_reports_a_prediction_and_a_measurement_each_with_a_ci(rep):
    t = rep["tlife"]
    for side in ("predicted", "measured"):
        for field in ("mean", "ci_lo", "ci_hi", "n"):
            assert field in t[side], "%s has no %s" % (side, field)
        assert t[side]["n"] >= 3, "%s over only %d seeds" % (side, t[side]["n"])
        assert t[side]["ci_lo"] < t[side]["ci_hi"]
    assert np.isfinite(t["difference"])
    assert abs(t["difference"] - (t["measured"]["mean"] - t["predicted"]["mean"])) < 1e-12


def test_the_owners_lineage_figure_is_carried_as_his_and_not_as_ours(rep):
    """0.60 at coupling 1/4 is the owner's number from a run this module never
    made. It is reported BESIDE the measurement with the difference stated, and
    it is never the source of any value this module publishes."""
    t = rep["tlife"]
    assert t["owner_figure"] == 0.60 and t["owner_coupling"] == 0.25
    assert t["coupling"] == 0.25, "the comparison is at a different coupling"
    assert abs(t["difference_from_owner"]
               - (t["measured"]["mean"] - 0.60)) < 1e-12
    assert t["owner_figure_is_ours"] is False


def test_the_h0_barcode_route_is_refused_with_its_measured_reason(rep):
    """The single-linkage death does not resolve the gap at this bed size. A
    refusal is a value with a reason and a number, not a silent substitution."""
    b = rep["tlife"]["barcode"]
    assert cv.is_refusal(b["death"]), (
        "the barcode route is reporting a death of %r; the probe measured the "
        "A-to-B linkage sitting on top of the within-phase linkage at every "
        "separation, so a death read off it is not a death" % (b["death"],))
    assert np.isfinite(b["ratio_max"]) and b["ratio_max"] > 0
    assert isinstance(b["wanted_interface"], str) and len(b["wanted_interface"]) > 40


# ---------------------------------------------------------------------------
# 6. PLANTED NEGATIVES, EACH SEEN TO FIRE
# ---------------------------------------------------------------------------

def test_the_structureless_bed_produces_no_critical_line(rep):
    """The trap this criterion walks into on its own: U > 0 and tau -> 0 at large
    separation, so U - tau*S crosses zero on ANY bed, structure or not. The
    crossing alone is not evidence. What separates them is whether the entropy
    RESPONDS to the control parameter.
    """
    n = rep["negatives"]["null_bed"]
    assert cv.is_refusal(n["critical_ratio"]), (
        "the structureless bed returned a critical ratio of %r" % (n["critical_ratio"],))
    assert n["dS_ds_ci_lo"] <= 0.0 <= n["dS_ds_ci_hi"], (
        "the null entropy slope CI [%.5f, %.5f] excludes zero"
        % (n["dS_ds_ci_lo"], n["dS_ds_ci_hi"]))
    p = rep["negatives"]["planted_bed"]
    assert p["dS_ds_ci_hi"] < 0.0, (
        "the planted entropy slope CI [%.5f, %.5f] touches zero, so the gate "
        "that refuses the null would also refuse the plant"
        % (p["dS_ds_ci_lo"], p["dS_ds_ci_hi"]))


def test_a_bed_planted_at_a_known_line_recovers_it_within_ci(rep):
    """The planting rides on the exact scale invariance: a bed built at
    temperature g*sigma but reported at sigma has a critical line at exactly
    g times the frozen one, and the fit must find that factor and nothing else.
    """
    r = rep["negatives"]["known_line"]
    assert r["g"] != 1.0
    assert r["ci_lo"] <= r["expected_c"] <= r["ci_hi"], (
        "planted line c = %.4f not recovered; got [%.4f, %.4f]"
        % (r["expected_c"], r["ci_lo"], r["ci_hi"]))


def test_the_k_gaming_control_is_measured_and_its_verdict_is_stated(rep):
    """A line that tracks k is measuring the construction, not the physics. The
    control must be RUN and its answer printed whichever way it comes out."""
    g = rep["negatives"]["k_gaming"]
    assert len(g["rows"]) >= 3, "only %d values of k swept" % len(g["rows"])
    for row in g["rows"]:
        assert {"k", "s_star", "n_edges", "ci_lo", "ci_hi", "n_seeds"} <= set(row)
        assert row["n_seeds"] >= 3, "k=%d rests on %d seeds" % (row["k"], row["n_seeds"])
        assert row["ci_lo"] <= row["s_star"] <= row["ci_hi"]
        assert row["n_edges"] > 0, "a curvature number with no edge count behind it"
    assert isinstance(g["verdict"], str) and len(g["verdict"]) > 20
    assert np.isfinite(g["frac_move"])
    assert (g["tracks_k"] is True) == (abs(g["frac_move"]) > 0.10), (
        "the verdict and the measured movement of %.4f disagree" % g["frac_move"])
    #: Marginal CIs at each k share the seed spread, which cancels in the
    #: difference; the paired statistic is the one that can answer this.
    assert g["paired_n"] >= 3 and len(g["paired"]) == g["paired_n"]
    assert abs(g["paired_mean"] - float(np.mean(g["paired"]))) < 1e-12, (
        "the published paired difference is not the mean of its own values")
    assert isinstance(g["monotone"], bool) and isinstance(g["paired_excludes_zero"], bool)


def test_the_k_control_is_repeated_at_a_second_bed_size(rep):
    """A control run once is a number with no spread behind it.

    An earlier report of this work cited a corroborating figure from a probe
    that lived outside the tree, so nothing in the repository could produce it.
    This binds the corroboration to a run: the same control at a second bed
    size, with its own paired statistic, and the two must agree in SIGN before
    either is called corroboration.

    WRITTEN AFTER THE RED, AND THEREFORE UNBOUND BY IT. The fifth such test and
    the one that took this file past the count an audit of it used; it arrived
    with the guard-repair pass, not with the original contract. See the file
    docstring for the arithmetic.
    """
    g = rep["negatives"]["k_gaming"]
    assert len(g["beds"]) >= 2, "the k control ran at one bed size only"
    sizes = [b["n_per"] for b in g["beds"]]
    assert len(set(sizes)) == len(sizes), "the two beds are the same size: %r" % sizes
    for bed in g["beds"]:
        assert bed["n_points"] == 2 * bed["n_per"] + 1
        assert bed["paired_n"] >= 3
        assert abs(bed["paired_mean"] - float(np.mean(bed["paired"]))) < 1e-12
        for row in bed["rows"]:
            assert row["n_edges"] > 0 and row["n_seeds"] >= 3
    signs = {b["frac_move"] < 0 for b in g["beds"]}
    assert (g["corroborated"] is True) == bool(
        len(signs) == 1 and all(b["tracks_k"] for b in g["beds"])), (
        "the corroboration verdict does not follow from the two beds' own "
        "numbers: %r" % [(b["n_per"], b["frac_move"], b["tracks_k"])
                         for b in g["beds"]])
    assert g["beds"][0] is not g["beds"][1]


# ---------------------------------------------------------------------------
# 7. PRINTED IS NOT MEASURED -- IDENTITY TESTS
# ---------------------------------------------------------------------------
#: Three fabricated prices survived the docstring guard this round by being
#: literals inside demo(). So every published number is pinned to the EXPRESSION
#: that recomputes it from the run, not to a constant.

def test_published_slope_is_the_number_the_train_points_recompute(rep):
    T = np.asarray(rep["tcrit"]["train_T"], float)
    sep = np.asarray(rep["tcrit"]["train_sep_crit"], float)
    c = float((T * sep).sum() / (T * T).sum())
    assert abs(c - rep["tcrit"]["fits"]["linear"]["c"]) < 1e-12, (
        "the published slope %.10f is not the least-squares slope %.10f of the "
        "points it is published beside"
        % (rep["tcrit"]["fits"]["linear"]["c"], c))


def test_published_slope_ci_is_the_interval_the_train_points_recompute(rep):
    """A CI printed as a literal is as fabricated as a mean printed as one.

    WRITTEN AFTER THE RED, AND THEREFORE UNBOUND BY IT. The recorded RED for
    this file names fewer tests than the file now ships; this is one of the
    four added afterwards, on the coordinator's identity-guard clause, and no
    run of this file ever saw it fail for the absence of the module. Its bite
    rests on the mutation battery instead, and no RED was manufactured after
    the fact to cover it. The exact counts are on the board rather than here,
    because a count of pytest outcomes is not something this module's demo can
    print, and a number no run produces does not belong in a docstring even
    when it is true.
    """
    from scipy.stats import t as student_t
    lin = rep["tcrit"]["fits"]["linear"]
    T = np.asarray(rep["tcrit"]["train_T"], float)
    y = np.asarray(rep["tcrit"]["train_sep_crit"], float)
    n = y.size
    c = float((T * y).sum() / (T * T).sum())
    rss = float(((y - c * T) ** 2).sum())
    se = float(np.sqrt(rss / (n - 1) / float((T * T).sum())))
    tc = float(student_t.ppf(0.975, n - 1))
    assert abs(lin["ci_lo"] - (c - tc * se)) < 1e-12, (
        "published CI low %.10f, recomputed %.10f" % (lin["ci_lo"], c - tc * se))
    assert abs(lin["ci_hi"] - (c + tc * se)) < 1e-12
    assert abs(lin["rss"] - rss) < 1e-12


def test_published_alternative_residuals_are_recomputed_from_the_same_points(rep):
    """The two losing forms carry published residuals; a fabricated residual is
    how a wrong form is made to look as though it lost honestly.

    WRITTEN AFTER THE RED, AND THEREFORE UNBOUND BY IT. The recorded RED for
    this file names fewer tests than the file now ships; this is one of the
    four added afterwards, on the coordinator's identity-guard clause, and no
    run of this file ever saw it fail for the absence of the module. Its bite
    rests on the mutation battery instead, and no RED was manufactured after
    the fact to cover it. The exact counts are on the board rather than here,
    because a count of pytest outcomes is not something this module's demo can
    print, and a number no run produces does not belong in a docstring even
    when it is true.
    """
    m = _pc()
    T = np.asarray(rep["tcrit"]["train_T"], float)
    y = np.asarray(rep["tcrit"]["train_sep_crit"], float)
    for name in ("quadratic", "constant"):
        f = rep["tcrit"]["fits"][name]
        x = m._basis(name, T)
        c = float((x * y).sum() / (x * x).sum())
        resid = y - c * x
        assert abs(f["c"] - c) < 1e-12, "%s slope %.10f vs %.10f" % (name, f["c"], c)
        assert len(f["resid"]) == resid.size, "%s residual count moved" % name
        for i, (pub, re_) in enumerate(zip(f["resid"], resid)):
            assert abs(pub - re_) < 1e-12, (
                "%s residual %d published as %.12f, recomputed as %.12f -- "
                "compared POSITION by position, because one frozen value hides "
                "inside an allclose over a list whose neighbours still vary"
                % (name, i, pub, re_))
        assert abs(f["rss"] - float((resid ** 2).sum())) < 1e-12
        assert abs(f["resid_rms"]
                   - float(np.sqrt((resid ** 2).sum() / y.size))) < 1e-12


def test_published_tlife_temperatures_are_recomputed_from_their_own_legs(rep):
    """predicted = U/S at the planted configuration, measured = 1/s*. Both means
    are pinned to the per-seed values they claim to be the mean of.

    WRITTEN AFTER THE RED, AND THEREFORE UNBOUND BY IT. The recorded RED for
    this file names fewer tests than the file now ships; this is one of the
    four added afterwards, on the coordinator's identity-guard clause, and no
    run of this file ever saw it fail for the absence of the module. Its bite
    rests on the mutation battery instead, and no RED was manufactured after
    the fact to cover it. The exact counts are on the board rather than here,
    because a count of pytest outcomes is not something this module's demo can
    print, and a number no run produces does not belong in a docstring even
    when it is true.
    """
    t = rep["tlife"]
    pred = [row["U"] / row["S"] for row in t["at_s0"]]
    assert len(pred) == len(t["predicted_values"])
    for i, (re_, pub) in enumerate(zip(pred, t["predicted_values"])):
        assert abs(re_ - pub) < 1e-12, (
            "predicted value %d published as %.12f, recomputed as %.12f"
            % (i, pub, re_))
    assert abs(t["predicted"]["mean"] - float(np.mean(pred))) < 1e-12, (
        "published predicted tau_death %.10f is not the mean of %r"
        % (t["predicted"]["mean"], pred))
    meas = [1.0 / s for s in t["s_stars"]]
    assert len(meas) == len(t["measured_values"])
    for i, (re_, pub) in enumerate(zip(meas, t["measured_values"])):
        assert abs(re_ - pub) < 1e-12, (
            "measured value %d published as %.12f, recomputed as %.12f"
            % (i, pub, re_))
    assert abs(t["measured"]["mean"] - float(np.mean(meas))) < 1e-12
    for row in t["at_s0"]:
        assert row["n_edges"] > 0


def test_published_landau_well_positions_and_depths_are_recomputed(rep):
    """Not only F_min: every local minimum's POSITION and DEPTH is re-derived
    from the published family, because the well list IS the Landau claim.

    WRITTEN AFTER THE RED, AND THEREFORE UNBOUND BY IT. The recorded RED for
    this file names fewer tests than the file now ships; this is one of the
    four added afterwards, on the coordinator's identity-guard clause, and no
    run of this file ever saw it fail for the absence of the module. Its bite
    rests on the mutation battery instead, and no RED was manufactured after
    the fact to cover it. The exact counts are on the board rather than here,
    because a count of pytest outcomes is not something this module's demo can
    print, and a number no run produces does not belong in a docstring even
    when it is true.
    """
    m = _pc()
    fam = rep["landau"]["family"]
    for row in rep["landau"]["ladder"]:
        F = np.array([m.free_energy(q["U"], q["S"], row["tau"]) for q in fam])
        wells = [(fam[i]["s"], float(F[i])) for i in range(F.size)
                 if (i == 0 or F[i] <= F[i - 1])
                 and (i == F.size - 1 or F[i] <= F[i + 1])]
        assert len(wells) == len(row["minima"]), (
            "tau=%.2f publishes %d wells, the family gives %d"
            % (row["tau"], len(row["minima"]), len(wells)))
        for (s_pub, f_pub), (s_re, f_re) in zip(row["minima"], wells):
            assert abs(s_pub - s_re) < 1e-12 and abs(f_pub - f_re) < 1e-12, (
                "well published at (%.4f, %.6f), recomputed at (%.4f, %.6f)"
                % (s_pub, f_pub, s_re, f_re))
        assert abs(row["argmin_s"] - fam[int(F.argmin())]["s"]) < 1e-12


def test_published_free_energies_are_e_minus_tau_s_of_the_published_family(rep):
    m = _pc()
    fam = rep["landau"]["family"]
    for row in rep["landau"]["ladder"]:
        F = [m.free_energy(p["U"], p["S"], row["tau"]) for p in fam]
        assert abs(min(F) - row["F_min"]) < 1e-12, (
            "F_min at tau=%.3f is published as %.6f but E - tau*S over the "
            "published family gives %.6f" % (row["tau"], row["F_min"], min(F)))


def test_published_binding_is_the_mean_of_the_published_within_edges(rep):
    m = _pc()
    p = rep["identity_probe"]
    X, lab = m.two_phase_bed(seed=p["seed"], sep=p["sep"], sigma=p["sigma"])
    again = m.bed_measure(X, lab, k=p["k"], alpha=p["alpha"])
    assert abs(again["U"] - p["U"]) < 1e-12, (
        "the published binding %.10f does not survive recomputation: %.10f"
        % (p["U"], again["U"]))
    assert again["n_edges"] == p["n_edges"], "the edge count behind it moved"


def test_published_tau_crit_is_one_over_the_published_slope(rep):
    c = rep["tcrit"]["fits"]["linear"]["c"]
    assert abs(rep["landau"]["tau_crit_from_criterion"] - 1.0 / c) < 1e-12


# ---------------------------------------------------------------------------
# 8. THE WIDENED DOCSTRING-NUMBER GUARD -- ANY DECIMAL, NO EXEMPTION LIST
# ---------------------------------------------------------------------------

#: A number WITH ITS BOUNDARIES, and matched TOKEN-to-TOKEN. The lookarounds are
#: the repair for the substring vacuity struck in this file's first form: `tok in
#: out` is true for a fabricated 5.827721 the moment the run prints 15.827721, so
#: a substring test passes any invented number that shares digits with a real one
#: -- which is exactly what a stale number looks like. Re-scored under this
#: reading none of the numbers already shipped goes missing, so the defect was
#: unexploited, but unexploited is not absent and the planted collision below
#: keeps it that way.
#:
#: ANY decimal AND ANY BARE INTEGER, scoped by __module__ so demo() and the
#: private helpers are in scope, NO exemption list: a decimal-only pattern misses
#: a step count or a test count stated in prose. The inner (?:\.\d+)* keeps a
#: dotted version string as ONE token rather than splitting a spurious 3.11 out
#: of 3.11.9; the trailing (?!\w)(?!\.\d) rather than (?![\w.]) so a sentence-
#: final period does not hide the number in front of it.
_MEASURED = re.compile(
    r"(?<![\w.])[+-]?\d+(?:\.\d+)*(?:[eE][+-]?\d+)?(?!\w)(?!\.\d)")
_DECIMAL = re.compile(
    r"(?<![\w.])[+-]?\d+\.\d+(?:\.\d+)*(?:[eE][+-]?\d+)?(?!\w)(?!\.\d)")


def _tokens(text):
    """Every number in `text` as a boundary-anchored token.

    A leading plus is stripped, so a docstring quoting a number bare still
    matches a run that prints it signed. A leading MINUS is not: a negative
    number is a different number, and matching across the sign would put the
    defect back facing the other way.
    """
    return {t.lstrip("+") for t in _MEASURED.findall(text)}


def _missing_from(docs, run_text):
    """THE membership expression. Every docstring number not bound by the run.

    The guard's collision control below runs through THIS function rather than
    inspecting a token set beside it, because a control that only checks the
    tokeniser is decorative: reverting this line to `tok in run_text` -- the
    substring form already struck once -- left the collision assertion passing
    and the whole guard vacuous again, which is precisely how the defect came
    back the first time. One function, and the control goes through it.
    """
    run = _tokens(run_text)
    return [(where, tok) for where, doc in docs
            for tok in _tokens(doc) if tok not in run]


def _module_docstrings(m):
    docs = [("module", m.__doc__ or "")]
    for name, obj in sorted(vars(m).items()):
        if getattr(obj, "__module__", None) != m.__name__:
            continue
        doc = getattr(obj, "__doc__", None)
        if isinstance(doc, str) and doc.strip():
            docs.append((name, doc))
    return docs


def test_every_decimal_in_every_docstring_is_printed_by_the_demo(capsys):
    """No exemption list. A number no run prints cannot be checked by anyone,
    including its author, and a number copied from ANOTHER agent's run can never
    be checked by any run of this module at all."""
    m = _pc()
    m.demo()
    out = capsys.readouterr().out

    #: The RED/shipped arithmetic in this file's docstring is reconciled HERE
    #: and printed, so those numerals are bound to a run like every other one.
    #: The shipped count is measured off the module; the RED components are
    #: historical and are printed as such, because a count of pytest outcomes is
    #: not something the module's demo can produce.
    shipped = _shipped_tests()
    recorded = RED_FAILED + RED_PASSED + RED_ERRORS
    reconciliation = (
        "\n  recorded RED (historical, from the board): %d failed + %d passed + "
        "%d errors = %d tests; shipped NOW, counted off this module: %d; "
        "written after the RED: %d"
        % (RED_FAILED, RED_PASSED, RED_ERRORS, recorded, shipped,
           shipped - recorded))
    print(reconciliation)
    assert shipped > recorded, (
        "the shipped count %d is not above the recorded RED total %d; the "
        "file docstring's arithmetic no longer describes this file"
        % (shipped, recorded))

    run_text = out + reconciliation

    #: THE PERMANENT PLANTED COLLISION, driven through _missing_from itself.
    #: 15.827721 is the constant form's RSS and the run prints it; 5.827721 is a
    #: fabrication sharing its tail, and under the substring form it passed. All
    #: three states are asserted in band -- a real printed token BOUND, a planted
    #: collision UNBOUND, an absent decimal UNBOUND -- because a guard that flags
    #: a number the run did print and a guard that passes a number the run never
    #: printed are one defect facing two directions.
    control = [("planted", "a fabricated 5.827721 and an absent 77.31 and 40317"),
               ("real", "the printed 15.827721 and 0.427621")]
    flagged = {tok for _, tok in _missing_from(control, run_text)}
    assert flagged == {"5.827721", "77.31", "40317"}, (
        "the collision control did not come back exactly as expected: %r. If "
        "5.827721 is absent from this set the membership test has slid back to "
        "the substring form and every number below is unbound; if 15.827721 or "
        "0.427621 is present the guard is flagging numbers the run did print"
        % sorted(flagged))

    #: This file's OWN docstrings are scanned too, against the same run. A
    #: measured number written into a TEST docstring was bound to nothing at all.
    docs = _module_docstrings(m)
    here = _module_docstrings(sys.modules[__name__])
    missing = _missing_from(docs + here, run_text)
    checked = sum(len(_MEASURED.findall(doc)) for _, doc in docs)
    decimals = sum(len(_DECIMAL.findall(doc)) for _, doc in docs)
    print("\n  %d numbers (%d of them decimals) across %d module docstrings "
          "and %d in this file, %d missing"
          % (checked, decimals, len(docs), len(here), len(missing)))
    assert decimals >= 20, ("only %d decimals found: widening to integers must "
                            "not dilute the decimal guard" % decimals)
    assert checked >= 40, "only %d numbers found: the regex is not biting" % checked
    assert not missing, ("these docstring numbers are printed by no run:\n    "
                         + "\n    ".join("%s: %s" % (w, t) for w, t in missing))

    #: CLAUSE 4. The coverage figure is a number in prose like any other and
    #: nothing was checking it: a stale count from a RED run, taken before
    #: demo() had a docstring of its own, reported a guard as wider than it was.
    #: The counts above were scanned by THIS file; the module scans its own and
    #: prints them. Two counters that must agree is a check.
    cov = m.docstring_numbers()
    assert cov["n_docstrings"] == len(docs), (
        "the module counts %d docstrings, an independent scan finds %d: %r"
        % (cov["n_docstrings"], len(docs),
           set(cov["names"]) ^ {w for w, _ in docs}))
    assert cov["n_numbers"] == checked, (
        "the module counts %d numbers, an independent scan finds %d"
        % (cov["n_numbers"], checked))
    assert cov["n_decimals"] == decimals, (
        "the module counts %d decimals, an independent scan finds %d"
        % (cov["n_decimals"], decimals))
    for figure in (cov["n_docstrings"], cov["n_numbers"], cov["n_decimals"]):
        assert str(figure) in out, (
            "the coverage figure %d is not printed by the run, so nothing "
            "checks it" % figure)
    assert not missing, ("these docstring numbers are printed by no run:\n    "
                         + "\n    ".join("%s: %s" % (w, t) for w, t in missing))


# ---------------------------------------------------------------------------
# 9. THE DEMO IS A RUN, UNDER BUDGET, AND SAYS SO
# ---------------------------------------------------------------------------

def test_the_demo_finishes_under_300_seconds_and_ends_with_the_exact_line():
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "ceqjepa.phase_criterion"],
                       capture_output=True, text=True, timeout=400)
    dt = time.time() - t0
    assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-4000:]
    assert dt < 300.0, "the demo took %.1f s" % dt
    assert p.stdout.rstrip().endswith("ALL SELF-CHECKS PASSED"), \
        repr(p.stdout[-300:])
    assert re.search(r"\b\d+\.\d\s*s\b", p.stdout), \
        "the demo does not state its own wall time"


def test_every_curvature_number_is_printed_beside_its_edge_count(rep):
    """The invariant this project keeps getting burned by."""
    for p in rep["landau"]["family"]:
        assert p["n_edges"] > 0 and "n_refused" in p
    for row in rep["negatives"]["k_gaming"]["rows"]:
        assert row["n_edges"] > 0

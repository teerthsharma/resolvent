"""RED-FIRST tests for ceqjepa/pi_assign.py (DR-2 R1: the assignment rule; R3:
dimensional hygiene).

WHAT THE MODULE UNDER TEST CLAIMS. Each latent coordinate of an encoder is given
a scaling exponent alpha, fitted over a context-length sweep; alpha near zero
takes the softmax corner beta=1, alpha near one takes the linear corner beta=0,
and anything else is REFUSED as a value carrying a reason. Only dimensionless
arguments enter exp. Each test below is a way for those claims to be FALSE while
numbers still come out:

  (a) THE EXPONENT HAS NO REFERENT. alpha presumes the norm IS a power law in n,
     and four sweep points are a thin fit. A coordinate whose norm is not a power
     law must be refused under its OWN reason, distinct from MIXED-SCALING, and
     the shapes this four-point test CANNOT refuse must be measured and named as
     the floor rather than left implicit.
  (b) THE REFUSAL IS A NaN OR A SILENT DEFAULT. Either is invisible to a scorer:
     a NaN passes `nan < tol` silently and a default beta assigns the coordinate
     to a corner nobody chose. The refusal must be a VALUE with a reason.
  (c) THE COORDINATE IS ASSIGNED WITHOUT ITS ALPHA. That is the owner's kill, and
     it is planted here: an assignment whose alpha has been stripped must be
     struck by a check that runs in the module, not by a reader.
  (d) exp EATS A DIMENSIONFUL ARGUMENT. The second kill. A logit whose unit
     exponents are not all zero must be refused at the exp, not caught in review.
  (e) THE CORNER IS UNRELATED TO THE SCALING. If the beta=1 read did not inherit
     its units as a MEAN and the beta=0 read as a TOTAL, the assignment rule
     would be a vocabulary and not a derivation. The output exponent must be
     measured against the derived 1-beta on the bed.
  (f) THE NAIVE BASELINE IS NEVER SHOWN. One corner for every coordinate, and a
     random corner, must be reported beside the rule, and the single-corner
     baseline must be seen to WIN inside its own class and LOSE outside it.
  (g) THE MASK IS FROZEN OVER THE OUTPUTS. A digest over fitted exponents cannot
     tell a freeze from an honest refit. The digest must be over the sweep
     INPUTS, and a planted refit must be seen to fire against it.
  (h) ALPHA IS A MEASUREMENT OF THE NORMALIZER. A layer dividing by a quantity
     that grows with n collapses every exponent toward zero and silently
     reclassifies extensive coordinates as intensive. The census that detects it
     must run automatically inside the assignment, not be remembered.
  (i) THE NUMBER CAME FROM NO RUN. Every number in the docstrings of BOTH files,
     decimals and bare integers alike, matched as whole TOKENS so a fragment of a
     longer number cannot stand in for one, must appear in the output of a run
     performed here; and every published value must equal an expression that
     recomputes it, compared position by position.

NOTHING HERE IS A CLAIM ABOUT A TRAINED MODEL. The encoder is synthetic and every
coordinate's class is planted by construction, because R2 (the JEPA build) is a
later iteration and no trained encoder exists to read.

THE IMPORT IS GUARDED ON PURPOSE. A module-level `import ceqjepa.pi_assign` turns
the RED phase into one collection error with no test names in it. Each test must
be seen to FAIL BY NAME before the module exists.

RUN: python -m pytest tests/curvature/test_pi_assign.py -v
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
    from ceqjepa import pi_assign as _pi_assign
except Exception as _exc:                     # noqa: BLE001 -- RED phase carries it
    _pi_assign = None
    _IMPORT_ERROR = _exc
else:
    _IMPORT_ERROR = None


def _pa():
    """The module under test, or a named failure instead of a collection error."""
    if _pi_assign is None:
        pytest.fail("ceqjepa.pi_assign does not import: %r" % (_IMPORT_ERROR,))
    return _pi_assign


# ---------------------------------------------------------------------------
# 1. THE ASSIGNMENT RULE, AND THE FOUR MUST-FIRES
# ---------------------------------------------------------------------------

def test_planted_intensive_coordinates_read_alpha_near_zero(capsys):
    """A mean, a probability and a ratio are intensive BY CONSTRUCTION.

    If the sweep cannot recover alpha = 0 on coordinates whose n-independence is
    arithmetic, nothing it reports about an unknown coordinate is worth reading.
    """
    m = _pa()
    r = m.report()
    rows = [(name, a) for name, a in r["assignments"].items()
            if m.PLANT[name][0] == "INTENSIVE"]
    assert len(rows) >= 3, "fewer than three intensive coordinates are planted"
    for name, a in rows:
        print("\n  %-14s alpha %+0.4f  CI [%+0.4f, %+0.4f]  R2 %.6f  beta %s"
              % (name, a.alpha, a.ci_lo, a.ci_hi, a.r2, a.beta))
        assert abs(a.alpha - 0.0) <= 0.1, \
            "%s is planted intensive and read alpha = %r" % (name, a.alpha)
        assert a.beta == 1.0, \
            "%s read alpha %.4f and did NOT take the softmax corner" % (name, a.alpha)


def test_planted_extensive_coordinates_read_alpha_near_one(capsys):
    """A count, a sum and a total are extensive BY CONSTRUCTION."""
    m = _pa()
    r = m.report()
    rows = [(name, a) for name, a in r["assignments"].items()
            if m.PLANT[name][0] == "EXTENSIVE"]
    assert len(rows) >= 3, "fewer than three extensive coordinates are planted"
    for name, a in rows:
        print("\n  %-14s alpha %+0.4f  CI [%+0.4f, %+0.4f]  R2 %.6f  beta %s"
              % (name, a.alpha, a.ci_lo, a.ci_hi, a.r2, a.beta))
        assert abs(a.alpha - 1.0) <= 0.1, \
            "%s is planted extensive and read alpha = %r" % (name, a.alpha)
        assert a.beta == 0.0, \
            "%s read alpha %.4f and did NOT take the linear corner" % (name, a.alpha)


def test_the_half_and_half_coordinate_is_refused_as_mixed_and_is_not_a_nan(capsys):
    """A sum of zero-mean tokens grows like sqrt(n): half a count, no corner.

    The refusal must be a VALUE carrying MIXED-SCALING. A NaN would pass every
    `nan < tol` downstream and a default corner would assign the coordinate to a
    corner nobody chose, so both are checked explicitly here.
    """
    m = _pa()
    r = m.report()
    names = [n for n, p in m.PLANT.items() if p[0] == "MIXED"]
    assert names, "no half-and-half coordinate is planted"
    for name in names:
        a = r["assignments"][name]
        print("\n  %-14s alpha %+0.4f  CI [%+0.4f, %+0.4f]  R2 %.6f  beta %r"
              % (name, a.alpha, a.ci_lo, a.ci_hi, a.r2, a.beta))
        assert m.is_refusal(a.beta), "%s was assigned beta = %r" % (name, a.beta)
        assert a.beta.code == m.REASON_MIXED, \
            "%s refused for %r, not MIXED-SCALING" % (name, a.beta.code)
        assert not isinstance(a.beta, float), "the refusal is a float"
        assert np.isfinite(a.alpha), "the refused coordinate carries alpha = %r" % (a.alpha,)
        assert abs(a.alpha - 0.5) <= 0.1, \
            "the sqrt(n) plant read alpha = %.4f, not 0.5" % (a.alpha,)


def test_the_non_power_law_coordinate_is_refused_under_a_different_reason(capsys):
    """An exponent fitted to something that is not a power law has no referent.

    The reason must be DISTINCT from MIXED-SCALING: a coordinate refused for
    having an exponent between the corners and a coordinate refused for having no
    exponent at all are two different findings and are reported as two. The plant
    is built so the naive route WOULD have assigned it -- its exponent sits at a
    corner -- and only the power-law test, which runs first, refuses it.
    """
    m = _pa()
    r = m.report()
    names = [n for n, p in m.PLANT.items() if p[0] == "NOT-A-POWER-LAW"]
    assert names, "no non-power-law coordinate is planted"
    assert m.REASON_NOT_POWER_LAW != m.REASON_MIXED, "the two reasons are one reason"
    for name in names:
        a = r["assignments"][name]
        print("\n  %-14s alpha %+0.4f  R2 %.6f  resid RMS %.6f  chi2 %.4g  beta %r"
              % (name, a.alpha, a.r2, a.resid_rms, a.chi2, a.beta))
        assert m.is_refusal(a.beta), "%s was assigned beta = %r" % (name, a.beta)
        assert a.beta.code == m.REASON_NOT_POWER_LAW, \
            "%s refused for %r, not NOT-A-POWER-LAW" % (name, a.beta.code)
        assert a.chi2 > m.CHI2_CRIT, "the plant passed at chi2 %.4g" % a.chi2
        assert abs(a.alpha) <= m.ALPHA_TOL, (
            "the plant's exponent is not at a corner, so the ORDER of the checks "
            "is not being tested: alpha %+.4f" % a.alpha)

    good = [r["assignments"][n].chi2 for n, p in m.PLANT.items()
            if p[0] != "NOT-A-POWER-LAW"]
    bad = [r["assignments"][n].chi2 for n in names]
    print("  power laws chi2 at worst %.4f; the plant reads %s; critical %.1f"
          % (max(good), ", ".join("%.4g" % v for v in bad), m.CHI2_CRIT))
    assert max(good) < m.CHI2_CRIT < min(bad), \
        "the critical value %r does not sit in the measured gap" % (m.CHI2_CRIT,)


def test_r_squared_cannot_be_the_power_law_test_and_the_control_fires(capsys):
    """R^2's denominator is the spread a flat coordinate does not have.

    A planted negative for the design that was NOT taken: had the refusal been
    read off R^2, the module reports how many of its own planted power laws it
    would have refused, and that count must be nonzero or the control is prose.
    """
    m = _pa()
    r = m.report()
    c = r["r2_control"]
    print("\n  R^2 < %.2f would refuse %d of the %d planted power laws: %s"
          % (c["threshold"], c["would_refuse"], c["n_true_power_laws"],
             ", ".join(c["names"]) or "none"))
    assert c["would_refuse"] > 0, "the R^2 control refuses nothing: it is vacuous"
    for name in c["names"]:
        a = r["assignments"][name]
        assert a.r2 < c["threshold"] and a.chi2 <= m.CHI2_CRIT, \
            "%s is not an example of the failure being claimed" % (name,)
        assert not m.is_refusal(a.beta), \
            "%s was refused anyway: the control names the wrong coordinate" % (name,)


def test_the_shape_this_four_point_fit_cannot_refuse_is_measured_and_named(capsys):
    """THE FLOOR. A logarithm is not a power law and four points cannot say so.

    It is planted, it passes the power-law test, and it is ASSIGNED a corner. The
    deliverable is the measured floor, not the reassurance: a coordinate inside it
    carries an exponent with no referent and the assignment cannot tell.
    """
    m = _pa()
    r = m.report()
    names = [n for n, p in m.PLANT.items() if p[0] == "FLOOR"]
    assert names, "the detection floor is not planted"
    for name in names:
        a = r["assignments"][name]
        print("\n  %-14s alpha %+0.4f  chi2 %.4f  curvature %+0.6f  floor %.6f  beta %r"
              % (name, a.alpha, a.chi2, a.curvature, a.q_floor, a.beta))
        assert a.chi2 <= m.CHI2_CRIT, \
            "the floor plant was refused at chi2 %.4f: it is no longer the floor" % a.chi2
        assert not m.is_refusal(a.beta), "the floor plant was refused, not assigned"
        assert abs(a.curvature) < a.q_floor, \
            "the plant's curvature %r is above its own floor %r" % (a.curvature, a.q_floor)
    assert r["floor"]["names"] == names
    assert r["floor"]["r2_max"] == max(r["assignments"][n].r2 for n in names)
    assert r["floor"]["q_floor"] == r["assignments"][names[0]].q_floor


# ---------------------------------------------------------------------------
# 2. THE TWO KILLS
# ---------------------------------------------------------------------------

def test_an_assignment_without_its_alpha_is_struck(capsys):
    """The owner's kill, planted so it is SEEN to fire, and in both directions."""
    m = _pa()
    r = m.report()
    for name, a in r["assignments"].items():
        assert np.isfinite(a.alpha), "%s carries alpha = %r" % (name, a.alpha)
        assert m.struck(a) is None, "a live assignment was struck: %s" % (name,)
    good = r["assignments"][[n for n, p in m.PLANT.items()
                             if p[0] == "INTENSIVE"][0]]
    dead = m.strip_alpha(good)
    verdict = m.struck(dead)
    print("\n  planted alpha-less assignment -> %r" % (verdict,))
    assert m.is_refusal(verdict), "an alpha-less assignment was NOT struck"
    assert verdict.code == m.REASON_NO_ALPHA
    assert r["kills"]["alphaless_struck"] is True


def test_exp_of_a_dimensionful_argument_is_refused(capsys):
    """The second kill. A dimensionful exponent is refused AT the exp."""
    m = _pa()
    ok = m.safe_exp(np.array([0.5, -0.5]), {"V": 0, "U": 0, "N": 0})
    assert not m.is_refusal(ok), "a dimensionless argument was refused"
    assert np.allclose(ok, np.exp([0.5, -0.5]))

    bad = m.safe_exp(np.array([0.5, -0.5]), {"V": 1, "U": 0, "N": 0})
    print("\n  exp of a V^1 argument -> %r" % (bad,))
    assert m.is_refusal(bad), "exp ate a dimensionful argument"
    assert bad.code == m.REASON_DIMENSIONFUL

    # Sum-zero is NECESSARY and NOT sufficient: V^1 N^-1 sums to zero and is
    # still dimensionful. The module must enforce the stronger all-zero form.
    sneaky = m.safe_exp(np.array([0.5]), {"V": 1, "U": 0, "N": -1})
    print("  exp of a V^1 N^-1 argument (exponents SUM to 0) -> %r" % (sneaky,))
    assert m.is_refusal(sneaky), "sum-zero was mistaken for dimensionless"
    assert m.units_sum({"V": 1, "U": 0, "N": -1}) == 0


def test_every_logit_prints_its_unit_exponents_and_they_are_all_zero(capsys):
    """R3: logits are pi groups, and the exponents are PRINTED as numbers."""
    m = _pa()
    r = m.report()
    assert r["logits"], "the module declares no logits"
    for name, row in r["logits"].items():
        print("\n  logit %-12s %s  sum %d"
              % (name, " ".join("%s^%d" % (d, row["units"][d]) for d in m.BASE_DIMS),
                 row["units_sum"]))
        assert row["units_sum"] == 0, "%s has unit exponents summing to %d" % (name, row["units_sum"])
        for d in m.BASE_DIMS:
            assert row["units"][d] == 0, \
                "%s carries %s^%d and is not a pi group" % (name, d, row["units"][d])
        assert row["exp_ok"] is True, "%s was refused at the exp" % (name,)


# ---------------------------------------------------------------------------
# 3. R3 CHECKED ON THE BED: THE MEAN, THE TOTAL, AND WHERE beta COMES FROM
# ---------------------------------------------------------------------------

def test_the_softmax_read_is_a_mean_and_the_linear_read_is_a_total(capsys):
    """Checked on the bed, not asserted in prose.

    W = e^l / Z^beta carries N^-beta, so a read over n tokens carries N^(1-beta):
    a MEAN at beta=1 and a TOTAL at beta=0. The output exponent is therefore
    1-beta, which is the derivation the assignment rule rests on, and it is
    measured over a beta sweep rather than stated.
    """
    m = _pa()
    r = m.report()
    sweep = r["beta_sweep"]
    worst = 0.0
    for row in sweep:
        print("\n  beta %.2f  predicted alpha_out %+0.4f  measured %+0.4f  gap %+0.4f  R2 %.6f"
              % (row["beta"], row["predicted"], row["alpha_out"], row["gap"], row["r2"]))
        assert row["predicted"] == 1.0 - row["beta"]
        worst = max(worst, abs(row["gap"]))
    assert worst == r["beta_sweep_worst_gap"]
    assert worst <= 0.1, "alpha_out = 1 - beta fails by %.4f" % worst
    ends = {row["beta"]: row["alpha_out"] for row in sweep}
    assert abs(ends[1.0] - 0.0) <= 0.1, "the softmax read is not a mean"
    assert abs(ends[0.0] - 1.0) <= 0.1, "the linear read is not a total"


def test_declared_units_and_measured_alpha_are_compared_per_coordinate(capsys):
    """The half coordinate declares one whole count and measures only half of one.

    Two independent routes to the same refusal: the coordinate DECLARES itself a
    total and MEASURES as half a count. A rule that only looked at alpha would
    refuse it for the right reason by luck; the disagreement is the reason.
    """
    m = _pa()
    r = m.report()
    for name, row in r["unit_vs_alpha"].items():
        print("\n  %-14s declared N^%+0.1f  measured alpha %+0.4f  disagreement %+0.4f"
              % (name, row["declared_n"], row["alpha"], row["gap"]))
        assert row["gap"] == row["declared_n"] - row["alpha"]
    for name, p in m.PLANT.items():
        if p[0] in ("INTENSIVE", "EXTENSIVE"):
            assert abs(r["unit_vs_alpha"][name]["gap"]) <= 0.1, \
                "%s declares and measures different scalings" % (name,)
        if p[0] == "MIXED":
            assert abs(r["unit_vs_alpha"][name]["gap"]) > 0.1, \
                "%s declares a total and measures like one" % (name,)


# ---------------------------------------------------------------------------
# 4. L-SIMPLE AND L-FAIL: THE NAIVE CORNER, BOTH DIRECTIONS
# ---------------------------------------------------------------------------

def test_the_simple_baselines_are_reported_beside_the_rule(capsys):
    """L-SIMPLE: one corner everywhere, and a random corner, scored on the same bed."""
    m = _pa()
    r = m.report()
    b = r["baselines"]
    for name in ("rule", "all_softmax", "all_linear", "random"):
        assert name in b, "the %s baseline is not reported" % (name,)
        print("\n  %-12s served %d/%d  broken %d  refused %d"
              % (name, b[name]["served"], b[name]["n_coords"],
                 b[name]["broken"], b[name]["refused"]))
        assert b[name]["served"] + b[name]["broken"] + b[name]["refused"] \
            == b[name]["n_coords"]
    assert b["rule"]["served"] >= b["random"]["served"], \
        "the measured rule does not beat a coin"


def test_the_single_corner_wins_inside_its_class_and_loses_outside_it(capsys):
    """L-FAIL, BOTH DIRECTIONS. If the naive corner never lost, the design is void;
    if it never won, the bed cannot tell the corners apart at all."""
    m = _pa()
    r = m.report()
    b = r["baselines"]
    inside = b["all_softmax"]["by_class"]["INTENSIVE"]
    outside = b["all_softmax"]["by_class"]["EXTENSIVE"]
    rule_in = b["rule"]["by_class"]["INTENSIVE"]
    rule_out = b["rule"]["by_class"]["EXTENSIVE"]
    print("\n  all_softmax on INTENSIVE: %d/%d served (rule: %d/%d)"
          % (inside["served"], inside["total"], rule_in["served"], rule_in["total"]))
    print("  all_softmax on EXTENSIVE: %d/%d served (rule: %d/%d)"
          % (outside["served"], outside["total"], rule_out["served"], rule_out["total"]))
    assert inside["served"] == inside["total"] == rule_in["served"], \
        "the naive corner does NOT win inside its own class: the bed is not measuring"
    assert outside["served"] == 0 < rule_out["served"], \
        "the naive corner does NOT lose outside its class: the design is void"

    lin_in = b["all_linear"]["by_class"]["EXTENSIVE"]
    lin_out = b["all_linear"]["by_class"]["INTENSIVE"]
    print("  all_linear  on EXTENSIVE: %d/%d served; on INTENSIVE: %d/%d served"
          % (lin_in["served"], lin_in["total"], lin_out["served"], lin_out["total"]))
    assert lin_in["served"] == lin_in["total"] and lin_out["served"] == 0


# ---------------------------------------------------------------------------
# 5. L-CLASS: THE THEOREM, ITS HYPOTHESES, CHECKED ON THE DATA
# ---------------------------------------------------------------------------

def test_the_theorem_hypotheses_are_checked_on_this_bed_and_printed(capsys):
    """A win counts only against a theorem whose hypotheses hold on the data.

    The theorem is three_corners_containment: at g == 0 the beta=1 setting IS
    softmax attention and the beta=0 setting IS linear attention, and
    corners_are_distinct says the two are different points of one family. Both
    hypotheses are numbers on this bed, and both are printed.
    """
    m = _pa()
    r = m.report()
    h = r["theorem"]
    print("\n  theorem %s  READ %s" % (h["name"], h["source"]))
    print("  (h1) gate g identically zero:            %s" % (h["gate_is_zero"],))
    print("  (h2) all logits finite:                  %s over %d entries"
          % (h["logits_finite"], h["n_logits"]))
    print("  (h3) beta=1 row sums:                    %.12f (max |sum-1| %.3e)"
          % (h["row_sum_beta1"], h["row_sum_beta1_err"]))
    print("  (h4) beta=0 row sums:                    %.6f  -> corners distinct: %s"
          % (h["row_sum_beta0"], h["corners_distinct"]))
    assert h["gate_is_zero"] is True
    assert h["logits_finite"] is True and h["n_logits"] > 0
    assert h["row_sum_beta1_err"] < 1e-12, \
        "the beta=1 corner is not row-stochastic here: %r" % (h["row_sum_beta1_err"],)
    assert h["corners_distinct"] is True, \
        "the two corners coincide on this bed: the containment is vacuous here"
    assert abs(h["row_sum_beta0"] - 1.0) > 0.1


# ---------------------------------------------------------------------------
# 6. THE FROZEN MASK, AND THE REFIT THAT MUST FIRE
# ---------------------------------------------------------------------------

def test_the_mask_digest_is_over_the_inputs_and_the_planted_refit_fires(capsys):
    """A digest over the fitted exponents cannot tell a freeze from a refit."""
    m = _pa()
    r = m.report()
    spec = r["mask"]["spec"]
    assert set(spec) >= {"encoder", "seed", "n_grid", "measured"}, \
        "the digest payload is missing an input: %r" % (sorted(spec),)
    assert "alpha" not in spec and "beta" not in spec, \
        "the digest covers the OUTPUTS: %r" % (sorted(spec),)
    assert m.mask_digest(spec) == r["mask"]["digest"] == m.FROZEN_MASK_DIGEST
    print("\n  frozen digest %s ... check ok=%s"
          % (r["mask"]["digest"][:16], r["mask"]["check"]["ok"]))
    assert r["mask"]["check"]["ok"] is True

    moved = m.perturb_one_measured_norm(spec, 1e-9)
    assert m.mask_digest(moved) != r["mask"]["digest"], \
        "moving a measured norm by 1e-9 left the digest unchanged"

    refit_spec, refit_digest = m.planted_refit()
    check = m.check_mask(refit_spec)
    print("  planted refit digest %s ... ok=%s" % (refit_digest[:16], check["ok"]))
    assert refit_digest != m.FROZEN_MASK_DIGEST
    assert check["ok"] is False, "the planted refit passed the freeze check"
    assert r["mask"]["planted_refit_fires"] is True


def test_the_mask_is_emitted_as_a_vector_along_a_named_axis(capsys):
    """A beta of shape [S] is silently accepted and wrong: the exponent would
    index the key rather than the query. The axis travels with the vector."""
    m = _pa()
    r = m.report()
    fm = r["mask"]["frozen"]
    print("\n  axis %r length %d, %d assigned, %d refused; beta = [%s]"
          % (fm["axis"], fm["axis_len"], fm["n_assigned"], len(fm["refusals"]),
             ", ".join("None" if b is None else "%.0f" % b for b in fm["beta"])))
    # A LITERAL, not m.MASK_AXIS. Comparing the emitted axis against the constant
    # that produced it is true under any rename, so it pins nothing; and a
    # blacklist of the one wrong name ("S") leaves every other wrong name green.
    # The emitter defends its own contract here, not the consumer downstream.
    assert fm["axis"] == "D", \
        "the mask axis is %r; the contract every consumer is written against is 'D'" \
        % (fm["axis"],)
    assert m.MASK_AXIS == "D", "MASK_AXIS is %r" % (m.MASK_AXIS,)
    assert fm["axis_len"] == len(m.COORD_NAMES) == len(fm["beta"]) == len(fm["coords"])
    assert tuple(fm["coords"]) == tuple(m.COORD_NAMES), \
        "the mask's order is not the coordinate order"
    for i, name in enumerate(fm["coords"]):
        a = r["assignments"][name]
        want = None if m.is_refusal(a.beta) else float(a.beta)
        assert fm["beta"][i] == want, \
            "mask position %d (%s): %r against the assignment's %r" \
            % (i, name, fm["beta"][i], want)
        assert fm["alpha"][i] == a.alpha, \
            "mask position %d (%s) carries alpha %r, not %r" \
            % (i, name, fm["alpha"][i], a.alpha)
    assert fm["n_assigned"] + len(fm["refusals"]) == fm["axis_len"], \
        "the refused coordinates were dropped and the axis silently shortened"
    assert '"D"' in fm["note"] or "'D'" in fm["note"], \
        "the note does not name the axis literally: %r" % (fm["note"],)
    assert fm["digest"] == r["mask"]["digest"]


def test_the_axis_named_in_the_prose_is_the_axis_the_code_emits(capsys):
    """A stale docstring beside renamed code currently ships green.

    The module docstring and frozen_mask's own docstring both TELL a reader which
    axis the mask lives on. Nothing compared either of them to the constant, so a
    rename left the prose saying one thing and the code emitting another, and the
    only thing that caught it was a consumer in another file repeating the check
    -- a pin that disappears the moment a second consumer does not repeat it.
    """
    m = _pa()
    said_module = re.search(r'axis named\s+"([A-Za-z]+)"', m.__doc__ or "")
    said_fn = re.search(r'axis="([A-Za-z]+)"', m.frozen_mask.__doc__ or "")
    assert said_module, "the module docstring does not name the axis in a parsable form"
    assert said_fn, "frozen_mask's docstring does not name the axis in a parsable form"
    emitted = m.report()["mask"]["frozen"]["axis"]
    print("\n  module docstring says %r, frozen_mask says %r, the code emits %r"
          % (said_module.group(1), said_fn.group(1), emitted))
    assert said_module.group(1) == said_fn.group(1) == emitted == m.MASK_AXIS == "D"


def test_the_nearest_prior_work_is_cited_in_the_module_itself(capsys):
    """A citation discovered by a reviewer is a citation that was not made."""
    m = _pa()
    doc = m.__doc__ or ""
    print("\n  prior art declared: %s" % (m.PRIOR_ART[:90],))
    for token in ("Bakarji", "2202.04643", "PRIOR ART"):
        assert token in doc, "the module docstring does not carry %r" % (token,)
    assert "2202.04643" in m.PRIOR_ART and "RELAYED" in m.PRIOR_ART, \
        "the relayed characterisation is presented as verified here"
    assert "not claimed as new" in doc, \
        "the docstring does not disclaim dimensional consistency as prior art"


# ---------------------------------------------------------------------------
# 7. THE ATTACK AGAINST THE RULE ITSELF
# ---------------------------------------------------------------------------

def test_a_normalizer_that_hides_n_collapses_alpha_and_the_census_fires(capsys):
    """THE ATTACK. RMS normalisation over a representation whose norm grows with n
    subtracts that growth from every exponent at once: extensive coordinates read
    intensive and are SILENTLY reclassified, and the rule becomes a measurement of
    the normalizer. The census must run inside assign(), not be remembered."""
    m = _pa()
    r = m.report()
    atk = r["attack"]
    print("\n  normalizer alpha (total norm): clean %+0.4f  attacked %+0.4f"
          % (atk["alpha_total_clean"], atk["alpha_total_attacked"]))
    for name in atk["per_coordinate"]:
        row = atk["per_coordinate"][name]
        print("  %-14s alpha %+0.4f -> %+0.4f  (%s -> %s)"
              % (name, row["alpha_clean"], row["alpha_attacked"],
                 row["class_clean"], row["class_attacked"]))
    assert atk["silently_reclassified"] > 0, \
        "the attack reclassified nothing: it is not the attack it claims to be"
    for name, row in atk["per_coordinate"].items():
        if m.PLANT[name][0] == "EXTENSIVE":
            assert abs(row["alpha_attacked"]) <= 0.1, \
                "%s did not collapse toward zero under the normalizer" % (name,)

    print("  census on the CLEAN bed:    %r" % (atk["census_clean"]["verdict"],))
    print("  census on the ATTACKED bed: %r" % (atk["census_attacked"]["verdict"],))
    assert atk["census_clean"]["fires"] is False, "the census fires on a clean bed"
    assert atk["census_attacked"]["fires"] is True, "the census missed the normalizer"
    assert atk["census_attacked"]["reason"] == m.REASON_NORMALIZER
    assert atk["census_runs_inside_assign"] is True, \
        "the census is a function nobody calls"
    assert m.is_refusal(r["attacked_assignment_verdict"]), \
        "assign() returned an assignment on a normalized bed without refusing it"
    # The reclassification count is read off what the RULE alone said, before the
    # census overrode it. Reading it off the post-census view would report zero
    # and hide the very finding the census exists because of.
    for name, row in atk["per_coordinate"].items():
        assert row["class_attacked_rule_only"] == \
            atk["rule_only_attacked"][name].klass


def test_a_naive_caller_cannot_obtain_a_corner_on_a_normalized_bed(capsys):
    """THE REFUSAL IS STRUCTURAL, NOT ADVISORY.

    A verdict beside nine populated assignments is a flag a caller can skip, and
    the caller who skips it gets beta 1.0 INTENSIVE for a planted EXTENSIVE
    coordinate with no obstacle of any kind. That is the exact wrong answer the
    census exists to prevent, so when the census fires the corners a caller can
    REACH are Refusals: arithmetic on one raises TypeError the way
    ceqjepa.curvature's Refusal is built to, instead of passing silently.

    The exponents stay populated, because they are the diagnosis.
    """
    m = _pa()
    res = m.assign(seed=m.SEED, normalize=True)
    assert m.is_refusal(res["verdict"]), "the census did not fire on the attacked bed"
    for name, a in res["assignments"].items():
        print("\n  naive read: assignments[%r].beta = %r  klass %s (planted %s)"
              % (name, a.beta, a.klass, m.PLANT[name][0]))
        assert m.is_refusal(a.beta), \
            "%s hands a naive caller %r on a normalized bed" % (name, a.beta)
        assert a.beta.code == m.REASON_NORMALIZER
        assert a.klass == "REFUSED"
        assert np.isfinite(a.alpha), \
            "%s lost its exponent: the diagnosis was thrown away with the corner" % name
        with pytest.raises(TypeError):
            _ = 1.0 - a.beta

    fm = m.frozen_mask(res["assignments"], res["census"]["verdict"])
    assert fm["n_assigned"] == 0 and len(fm["refusals"]) == fm["axis_len"], \
        "the emitted mask still carries corners on a normalized bed"

    # And the clean bed is untouched: a structural refusal that fired everywhere
    # would be the same defect wearing the opposite sign.
    clean = m.assign(seed=m.SEED, normalize=False)
    assert clean["verdict"] is None
    corners = [a.beta for a in clean["assignments"].values() if not m.is_refusal(a.beta)]
    print("  clean bed still yields %d corners" % len(corners))
    assert len(corners) == 7, "the clean bed yields %d corners" % len(corners)


# ---------------------------------------------------------------------------
# 8. THE GUARDS
# ---------------------------------------------------------------------------

#: A number WITH ITS BOUNDARIES, token to token. `"0.5" in out` is true the moment
#: the run prints 10.5 or 0.50, so a substring test passes any fabricated number
#: sharing digits with a real one. The inner (?:\.\d+)* keeps a dotted version
#: string as ONE token; the trailing guard is (?!\w)(?!\.\d) and not (?![\w.]) so
#: that a sentence-final period does not hide the number in front of it.
_MEASURED = re.compile(
    r"(?<![\w.])[+-]?\d+(?:\.\d+)*(?:[eE][+-]?\d+)?(?!\w)(?!\.\d)")

#: Two independently written counters, and a floor under each, so that a scope
#: regression reports zero missing and passes. These are the TEST's numbers over
#: BOTH files; the module writes its own over its own scope in
#: docstring_numbers() and the two must agree on the module's docstring count.
#: The floors sit about a tenth under the measured counts, so a scope that
#: quietly narrows fires here instead of reporting nothing missing.
MIN_DOCSTRINGS = 60
MIN_NUMBERS = 50

_CHILD = "PI_ASSIGN_GUARD_CHILD"


def _tokens(text):
    """Every number in `text` as a boundary-anchored token, leading + stripped.

    A leading MINUS is kept, because a negative number is a different number and
    matching across the sign would put the defect straight back.
    """
    return {t.lstrip("+") for t in _MEASURED.findall(text)}


def _own_test_run_output():
    """stdout of THIS file's own tests, so numbers in THEIR docstrings are bound."""
    env = dict(os.environ)
    env[_CHILD] = "1"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(Path(__file__).resolve()),
         "-q", "-s", "-p", "no:cacheprovider"],
        capture_output=True, text=True, env=env,
        cwd=str(Path(__file__).resolve().parents[2]))
    assert proc.stdout, "the child run printed nothing: %r" % (proc.stderr[-400:],)
    return proc.stdout


def _module_docstrings(mod):
    """(name, docstring) for the module and everything DEFINED in it.

    Filtered on __module__ so numpy's docstrings do not come along, and NOT
    filtered through __all__, because demo() is the function that prints every
    number and is absent from __all__. This is a SCOPE rule, not an exemption
    list: no number is ever exempted.
    """
    docs = [("module", mod.__doc__ or "")]
    for name, obj in sorted(vars(mod).items()):
        if getattr(obj, "__module__", None) != mod.__name__:
            continue
        doc = getattr(obj, "__doc__", None)
        if isinstance(doc, str) and doc.strip():
            docs.append((name, doc))
        for attr in sorted(vars(obj)) if isinstance(obj, type) else ():
            member = getattr(obj, attr, None)
            member = member.fget if isinstance(member, property) else member
            if getattr(member, "__module__", None) != mod.__name__:
                continue
            sub = getattr(member, "__doc__", None)
            if isinstance(sub, str) and sub.strip() and attr != "__doc__":
                docs.append(("%s.%s" % (name, attr), sub))
    return docs


def test_every_measured_number_in_a_docstring_is_printed_by_a_run(capsys):
    """No exemption list. A number no run prints cannot be checked by anyone."""
    if os.environ.get(_CHILD):
        pytest.skip("child of the own-file scan; the parent does the scanning")
    m = _pa()
    m.demo()
    out = capsys.readouterr().out
    run = _tokens(out) | _tokens(_own_test_run_output())

    # Vacuity controls, both directions and both shapes.
    assert "0.31415926" not in run, \
        "the absent-decimal control was found in the run"
    assert "70317" not in run, "the absent-integer control was found in the run"
    assert "128" in run, "a measured integer stopped being matched as a token"

    docs = _module_docstrings(m) + _module_docstrings(sys.modules[__name__])
    missing = [(where, tok) for where, doc in docs
               for tok in _tokens(doc) if tok not in run]
    checked = len({tok for _, doc in docs for tok in _tokens(doc)})
    print("\n  %d distinct numbers across %d docstrings, %d missing from the run"
          % (checked, len(docs), len(missing)))
    assert len(docs) >= MIN_DOCSTRINGS, \
        "only %d docstrings in scope: the scan has shrunk" % len(docs)
    assert checked >= MIN_NUMBERS, \
        "only %d numbers found: the regex is not biting" % checked

    # The module counts its own, independently. Two counters, or it is one --
    # and ALL THREE published figures are compared, not just the first. A floor
    # is not a counter: n_numbers frozen at its own literal, or at a wrong one,
    # passed a floor check while the demo printed the wrong figure, and
    # n_decimals was asserted by nothing at all. The recount below is over the
    # MODULE's scope only, so it is comparable to what the module reports.
    cov = m.docstring_numbers()
    mod_docs = _module_docstrings(m)
    mod_toks = {tok for _, doc in mod_docs for tok in _tokens(doc)}
    mine = (len(mod_docs), len(mod_toks), len([t for t in mod_toks if "." in t]))
    print("  the module counts %d docstrings, %d numbers, %d decimals; "
          "this file recounts %d, %d, %d"
          % (cov["n_docstrings"], cov["n_numbers"], cov["n_decimals"],
             mine[0], mine[1], mine[2]))
    assert cov["n_docstrings"] == mine[0], \
        "docstring counters disagree: module %d, test %d" % (cov["n_docstrings"], mine[0])
    assert cov["n_numbers"] == mine[1], \
        "number counters disagree: module %d, test %d" % (cov["n_numbers"], mine[1])
    assert cov["n_decimals"] == mine[2], \
        "decimal counters disagree: module %d, test %d" % (cov["n_decimals"], mine[2])
    assert not missing, ("these docstring numbers are printed by no run:\n    "
                         + "\n    ".join("%s: %s" % (w, t) for w, t in missing))


#: A floor under the recomputation below, so that deleting entries from it and
#: from report()["published"] together cannot make this guard pass by covering
#: nothing. Set under the measured count, like the two above.
MIN_PUBLISHED = 20


def _recomputed_published(m, r):
    """Every published value, rebuilt from the module's own functions HERE.

    Written independently of report(): each entry re-runs the producing function
    rather than reading the value report() stored, so a literal frozen into the
    published dict fails even when every top-level key beside it is live.
    """
    atk = m.normalizer_attack(m.SEED)
    tb = m.tiebreak_probe(m.SEED)
    base = m.baselines(r["assignments"], m.SEED)
    th = m.theorem_hypotheses(m.SEED)
    floor_name = [n for n, p in m.PLANT.items() if p[0] == "FLOOR"][0]
    fa = r["assignments"][floor_name]
    truth = [n for n, p in m.PLANT.items()
             if p[0] in ("INTENSIVE", "EXTENSIVE", "MIXED", "FLOOR")]
    return {
        "beta_sweep_worst_gap": max(abs(m.output_alpha(b, seed=m.SEED).alpha
                                        - (1.0 - b)) for b in m.BETA_SWEEP),
        "maxsub_alpha_beta0": m.output_alpha(0.0, seed=m.SEED, max_subtract=True).alpha,
        "maxsub_alpha_beta1": m.output_alpha(1.0, seed=m.SEED, max_subtract=True).alpha,
        "alpha_total_clean": atk["alpha_total_clean"],
        "alpha_total_attacked": atk["alpha_total_attacked"],
        "silently_reclassified": atk["silently_reclassified"],
        "offset_median_clean": atk["census_clean"]["offset_median"],
        "offset_median_attacked": atk["census_attacked"]["offset_median"],
        "offset_iqr_attacked": atk["census_attacked"]["offset_iqr"],
        "offset_iqr_clean": atk["census_clean"]["offset_iqr"],
        "reachable_corners_attacked": atk["reachable_corners_attacked"],
        "tiebreak_sqrt_gap": tb["sqrt_gap"],
        "tiebreak_alpha_difference": tb["alpha_difference"],
        "total_spread_clean": atk["census_clean"]["total_spread"],
        "total_spread_attacked": atk["census_attacked"]["total_spread"],
        "r2_would_refuse": len([n for n in truth
                                if r["assignments"][n].r2 < m.R2_NAIVE_MIN]),
        "floor_curvature": fa.curvature,
        "floor_q_floor": fa.q_floor,
        "row_sum_beta1": th["row_sum_beta1"],
        "row_sum_beta0": th["row_sum_beta0"],
        "rule_served": base["rule"]["served"],
        "all_softmax_served": base["all_softmax"]["served"],
        "all_linear_served": base["all_linear"]["served"],
        "random_served": base["random"]["served"],
    }


def _as_written(text, value):
    """`value` re-formatted in the SHAPE the prose wrote it in.

    A presence guard asks only whether a token appears somewhere in the run, so a
    sentence can be made factually false using a number that is genuinely printed
    -- attached to a different quantity. Comparing as strings in the prose's own
    precision is what binds the number to ITS quantity: a stated "20.0" against a
    live 20.0, a stated exponent against a live one rounded to the same figures,
    a stated count against a live count.
    """
    t = text.lstrip("+")
    if "e" in t or "E" in t:
        mant = t.lower().split("e")[0]
        k = len(mant.split(".")[1]) if "." in mant else 0
        return ("%." + str(k) + "e") % float(value)
    if "." in t:
        return ("%." + str(len(t.split(".")[1])) + "f") % float(value)
    return str(int(value))


def _claims(m, r):
    """(label, regex, live values) for every measured number stated in prose.

    One entry per quantity the module docstring asserts, each compared to the
    live measurement in the prose's own precision. The presence guard cannot do
    this: it only asks whether a token appears in the run, so a sentence rebuilt
    out of a printed token that belongs to a different quantity reads as false
    and passes. Declared constants and headline figures come first, because a
    docstring that misstates its own instrument's threshold is the worst case.
    """
    A = r["assignments"]
    atk = r["attack"]
    fl = r["floor"]
    th = r["theorem"]
    c = r["r2_control"]
    sweep = {row["beta"]: row for row in r["beta_sweep"]}
    inten = [n for n, p in m.PLANT.items() if p[0] == "INTENSIVE"]
    powerlaw = [A[n].chi2 for n, p in m.PLANT.items() if p[0] != "NOT-A-POWER-LAW"]
    notpl = [A[n].chi2 for n, p in m.PLANT.items() if p[0] == "NOT-A-POWER-LAW"]
    osc = A[[n for n, p in m.PLANT.items() if p[0] == "NOT-A-POWER-LAW"][0]]
    return [
        # -- declared constants of the instrument ---------------------------
        ("chi-square critical value",
         r"critical value (\d+(?:\.\d+)?),", [m.CHI2_CRIT]),
        ("chi-square critical value, restated in the floor derivation",
         r"refusable at chi-square (\d+(?:\.\d+)?) is q_floor = sqrt\((\d+(?:\.\d+)?)/4\)",
         [m.CHI2_CRIT, m.CHI2_CRIT]),
        ("the floor's leading coefficient",
         r"sqrt\([\d.]+/4\) \* se = (\d+(?:\.\d+)?)\*se", [math.sqrt(m.CHI2_CRIT / 4.0)]),
        ("the naive R^2 threshold", r"refusing at R\^2 < (\d+(?:\.\d+)?)", [m.R2_NAIVE_MIN]),
        ("the sweep grid", r"n in \{(\d+), (\d+), (\d+), (\d+)\}", list(m.N_GRID)),
        ("the corner tolerance, via the beta settings",
         r"across beta in \{(\d+(?:\.\d+)?), (\d+(?:\.\d+)?), (\d+(?:\.\d+)?), (\d+(?:\.\d+)?), (\d+(?:\.\d+)?)\}",
         list(m.BETA_SWEEP)),
        ("the mask axis length", r'"D", of length (\d+)', [len(m.COORD_NAMES)]),
        ("the coordinate count in the chi-square family bound",
         r"over the (\d+) coordinates\s+tested", [len(m.COORD_NAMES)]),
        # -- headline figures ------------------------------------------------
        ("beta-sweep worst gap", r"is at worst\s+(\d+(?:\.\d+)?) across beta",
         [r["beta_sweep_worst_gap"]]),
        ("floor curvature and the floor itself",
         r"carries curvature (-\d+(?:\.\d+)?) against its own floor of (\d+(?:\.\d+)?)",
         [fl["curvature"], fl["q_floor"]]),
        ("the floor plant's assigned exponent",
         r"softmax corner at alpha (\d+(?:\.\d+)?),", [A["log_shape"].alpha]),
        ("the clean-bed census offset",
         r"median (\d+(?:\.\d+)?) with IQR (\d+(?:\.\d+)?) on the clean bed",
         [atk["census_clean"]["offset_median"], atk["census_clean"]["offset_iqr"]]),
        ("the attacked-bed census offset",
         r"median (\d+(?:\.\d+)?) with IQR (\d+(?:\.\d+)?)\s*\n?after,",
         [atk["census_attacked"]["offset_median"],
          atk["census_attacked"]["offset_iqr"]]),
        ("the census's coordinate count",
         r"over the (\d+) coordinates that passed the power-law test",
         [atk["census_clean"]["n_offsets"]]),
        ("total-norm exponent, both beds",
         r"norm reads alpha (\d+(?:\.\d+)?) before and (\d+(?:\.\d+)?) after",
         [atk["alpha_total_clean"], atk["alpha_total_attacked"]]),
        ("total-norm spread, both beds",
         r"exact to (\d+(?:\.\d+)?(?:e[+-]?\d+)?) relative spread after normalisation against "
         r"(\d+(?:\.\d+)?(?:e[+-]?\d+)?) before",
         [atk["census_attacked"]["total_spread"],
          atk["census_clean"]["total_spread"]]),
        ("the R^2 control", r"would refuse (\d+) of the (\d+)\s+planted power laws",
         [c["would_refuse"], c["n_true_power_laws"]]),
        ("the silent reclassification count",
         r"all (\d+) planted extensive", [atk["silently_reclassified"]]),
        ("the reachable-corner count on the attacked bed",
         r"bed (\d+) of the (\d+) coordinates hand back a float",
         [atk["reachable_corners_attacked"], len(m.COORD_NAMES)]),
        ("the wrong corner a naive read used to return",
         r"returned beta (\d+(?:\.\d+)?) INTENSIVE for all (\d+) planted",
         [1.0, atk["silently_reclassified"]]),
        ("R^2 of the three planted intensive coordinates",
         r"read R\^2 of (\d+(?:\.\d+)?), (\d+(?:\.\d+)?) and (\d+(?:\.\d+)?)", [A[n].r2 for n in inten]),
        ("the chi-square gap", r"reads chi-square (\d+(?:\.\d+)?(?:e[+-]?\d+)?) against the eight\s+"
         r"power laws' worst of (\d+(?:\.\d+)?)", [max(notpl), max(powerlaw)]),
        ("the chi-square p-values", r"p = (\d+(?:\.\d+)?(?:e[+-]?\d+)?) per coordinate and "
         r"(\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s+over", [r["chi2_p"], r["chi2_p_family"]]),
        ("the 2-dof interval on the non-power-law coordinate",
         r"runs from (-\d+(?:\.\d+)?)\s+to (\d+(?:\.\d+)?) on the non-power-law",
         [osc.ci_lo, osc.ci_hi]),
        ("the max-subtraction control",
         r"exponent falls from (\d+(?:\.\d+)?) to (\d+(?:\.\d+)?)",
         [sweep[0.0]["alpha_out"], r["maxsub_alpha_beta0"]]),
        ("the max-subtraction control at the softmax corner",
         r"beta=1 stays\s+at (\d+(?:\.\d+)?)", [r["maxsub_alpha_beta1"]]),
    ]


def test_every_measured_number_stated_in_prose_is_pinned_to_its_own_quantity(capsys):
    """A WRONG NUMBER BUILT FROM A PRINTED TOKEN PASSES THE PRESENCE GUARD.

    Every token in "critical value 128" is printed by the run -- 128 is the last
    sweep point -- so the sentence arguing chi-square is the right test can
    misstate the instrument's own threshold and go green. The same holds for the
    census median rebuilt out of an R^2 value, and the beta-sweep gap rebuilt out
    of an alpha. The presence guard cannot see it, because presence is all it
    asks. Each such sentence is parsed here and compared to the live measurement
    in the prose's own precision, which binds the number to ITS quantity.

    The coverage this buys is PRINTED rather than claimed, because the count of
    prose numbers left on presence alone is the honest figure.
    """
    m = _pa()
    r = m.report()
    doc = m.__doc__ or ""
    bound = set()
    for label, pattern, live in _claims(m, r):
        hit = re.search(pattern, doc)
        assert hit, "%s: no longer in the docstring in a parsable form (%s)" \
            % (label, pattern)
        assert len(hit.groups()) == len(live), \
            "%s: %d groups against %d live values" % (label, len(hit.groups()), len(live))
        for i, (said, value) in enumerate(zip(hit.groups(), live)):
            want = _as_written(said, value)
            assert said.lstrip("+") == want, \
                "%s, position %d: the docstring says %r, the measurement is %r" \
                % (label, i, said, want)
            bound.add(said.lstrip("+"))

    total = _tokens(doc)
    print("\n  %d of %d distinct numbers in the module docstring are pinned to a "
          "named quantity" % (len(bound & total), len(total)))
    print("  the remaining %d are bound by presence in the run only"
          % len(total - bound))
    assert len(bound & total) >= 30, \
        "only %d prose numbers are claim-bound" % len(bound & total)


def test_the_published_values_are_not_rounded_literals(capsys):
    """PRINTED IS NOT MEASURED.

    A literal typed into a print statement satisfies the docstring guard above
    trivially. Every published value is pinned HERE to an expression that
    recomputes it from the module's own functions in this same run, and every
    vector is compared POSITION BY POSITION so that one frozen entry beside live
    neighbours cannot hide inside an aggregate tolerance.

    THE CEILING, STATED RATHER THAN IMPLIED, AND IT IS WHY THIS TEST IS NAMED FOR
    ROUNDED LITERALS AND NOT FOR LITERALS. Equality cannot tell a live
    computation from a constant typed at full precision: a value frozen at every
    digit it has passes, while the same value frozen as the figure the run PRINTS
    fails. That is the ceiling of any equality-based pin rather than a defect in
    it, and no amount of recomputation moves it -- only a different KIND of check
    would, such as perturbing an input and requiring the published value to move,
    which this module gets from the freeze digest instead. What this test does
    bind is every rounded literal, which is the shape a copied, quoted or stale
    number actually takes.
    """
    m = _pa()
    r = m.report()
    m.demo()
    out = capsys.readouterr().out

    grid, norms = m.coordinate_norms(seed=m.SEED)
    for i, n in enumerate(grid):
        assert n == m.N_GRID[i], "n grid position %d is %r" % (i, n)
    for name, a in r["assignments"].items():
        fit = m.fit_power_law(grid, norms[name])
        assert a.alpha == fit.alpha, "%s alpha %r != refit %r" % (name, a.alpha, fit.alpha)
        assert a.r2 == fit.r2 and a.resid_rms == fit.resid_rms
        assert a.ci_lo == fit.ci_lo and a.ci_hi == fit.ci_hi
        for i in range(len(grid)):
            assert a.norms[i] == norms[name][i], \
                "%s norm position %d: published %r, recomputed %r" \
                % (name, i, a.norms[i], norms[name][i])

    for i, row in enumerate(r["beta_sweep"]):
        got = m.output_alpha(row["beta"], seed=m.SEED)
        assert row["alpha_out"] == got.alpha, \
            "beta sweep position %d: published %r, recomputed %r" \
            % (i, row["alpha_out"], got.alpha)
        assert row["gap"] == got.alpha - (1.0 - row["beta"])

    assert r["beta_sweep_worst_gap"] == max(abs(x["gap"]) for x in r["beta_sweep"])
    assert r["mask"]["digest"] == m.mask_digest(r["mask"]["spec"])

    published = r["published"]
    assert published, "report() publishes no pinned values"

    # The published dict is its own surface and must be pinned SEPARATELY from the
    # top-level keys it is built from: freezing one entry of it as a literal
    # leaves every top-level check passing. Rebuilt here from the module's own
    # functions, then compared key for key.
    mine = _recomputed_published(m, r)
    assert set(mine) == set(published), \
        "published keys %r against the recomputation's %r" \
        % (sorted(set(published) - set(mine)), sorted(set(mine) - set(published)))
    assert len(mine) >= MIN_PUBLISHED, \
        "only %d published values are pinned: the recomputation has shrunk" % len(mine)
    for key in sorted(published):
        assert published[key] == mine[key], \
            "published[%r] = %r, recomputed %r" % (key, published[key], mine[key])

    for key, value in published.items():
        assert isinstance(value, (int, float)), "%s is %r" % (key, value)
        txt = str(value) if isinstance(value, int) else None
        assert (txt in out if txt is not None
                else (("%.4f" % value) in out or ("%.6f" % value) in out
                      or ("%.3e" % value) in out or ("%.2e" % value) in out
                      or ("%.2f" % value) in out)), \
            "%s = %r is not printed by the run" % (key, value)
    print("\n  %d published values re-derived against the module in this run"
          % len(published))


def test_the_demo_finishes_inside_the_budget_and_ends_with_the_exact_line(capsys):
    """The bar is 300 s and the last line is fixed by contract.

    The first line is the provenance: the producing commit and the machine id,
    both computed in the run rather than copied into it.
    """
    m = _pa()
    t0 = time.time()
    m.demo()
    dt = time.time() - t0
    out = capsys.readouterr().out.rstrip("\n")
    lines = out.splitlines()
    print("\n  demo() took %.2fs; first line %r" % (dt, lines[0]))
    assert dt < 300.0, "demo() took %.1fs against a 300 s bar" % dt
    assert out.endswith("ALL SELF-CHECKS PASSED"), \
        "the last line is %r" % (lines[-1] if lines else "",)
    prov = m.provenance()
    assert prov["commit"] in lines[0] and prov["machine"] in lines[0], \
        "the first line is not the provenance: %r" % (lines[0],)
    assert re.fullmatch(r"[0-9a-f]{12}", prov["machine"]), \
        "the machine id is not a 12-hex digest: %r" % (prov["machine"],)
    assert "RUN:" in (m.__doc__ or ""), "the module docstring carries no RUN: line"
    for name in m.__all__:
        assert hasattr(m, name), "__all__ advertises a missing name %r" % (name,)


def test_the_wall_clock_contributes_no_tokens_to_the_scanned_surface(capsys):
    """A TIMING PRINT IS A LAUNDERING CHANNEL, and it is closed by contract here.

    A wall clock printed with a space before its unit enters the "printed by a
    run" token set, and that does two bad things at once. An absent-decimal
    control anywhere in the runtime range goes red at random -- which is how this
    was found, on an unmutated tree under load. And a fabricator quoting any
    plausible small decimal is handed it for free, because some run somewhere
    printed it.

    The repair is to keep clock readings off the scanned surface entirely, by
    printing them with NO space before the unit, so the guard's trailing
    negative-lookahead on a word character swallows the token. That is asserted
    here rather than assumed, because the fix is one space wide and nothing else
    in the suite would notice it going back.
    """
    m = _pa()
    m.demo()
    out = capsys.readouterr().out
    timing = [ln for ln in out.splitlines() if "elapsed" in ln and "bar" in ln]
    assert len(timing) == 1, "expected one timing line, found %d" % len(timing)
    toks = _tokens(timing[0])
    print("\n  timing line: %r" % (timing[0].strip(),))
    print("  tokens it contributes: %s" % (sorted(toks) or "none",))
    assert toks == {"300"}, \
        "the timing line contributes %s; only the declared 300 s bar may appear" \
        % (sorted(toks),)
    # The property the format relies on, checked directly on both shapes.
    for sample in ("0.03s", "0.18s", "123.45s", "0.00s"):
        assert _tokens(sample) == set(), "%r still tokenises" % (sample,)
    for sample in ("0.03 s", "0.18 s"):
        assert _tokens(sample), "%r should tokenise: the control is backwards" % (sample,)


def test_the_declared_over_measured_tiebreak_is_refused_with_its_killer(capsys):
    """A rule that looks obvious, bound as a hypothesis rather than implemented.

    The proposal turns a MIXED-SCALING refusal into the linear corner whenever
    the DECLARED count exponent is a corner. It is killed by a coordinate that
    declares a corner and whose correct corner is neither -- and the kill only
    counts if that coordinate is INDISTINGUISHABLE from the one the tiebreak was
    meant to rescue on every input the tiebreak reads. Both halves are asserted.
    """
    m = _pa()
    r = m.report()
    tb = r["tiebreak"]
    print("\n  walk   declares N^%+0.1f measures %+0.4f" % (tb["declared_walk"], tb["walk_alpha"]))
    print("  sqrt   declares N^%+0.1f measures %+0.4f" % (tb["declared_sqrt"], tb["sqrt_alpha"]))
    print("  difference %.4f against a tolerance of %.2f; proposed beta %.0f misses "
          "by %.4f (%.1f tolerances)"
          % (tb["alpha_difference"], m.ALPHA_TOL, tb["proposed_beta"],
             tb["sqrt_gap"], tb["gap_in_tolerances"]))
    assert tb["declared_walk"] == tb["declared_sqrt"] == 1.0, \
        "the two probes do not declare the same exponent: they are not confusable"
    assert tb["indistinguishable"], \
        "the probes differ by %.4f: the tiebreak CAN tell them apart" % tb["alpha_difference"]
    assert tb["kills_the_tiebreak"], "the tiebreak survived its own killer"
    assert tb["sqrt_gap"] > m.ALPHA_TOL and tb["gap_in_tolerances"] > 1.0
    # And the refusal it was meant to overturn is still a refusal.
    a = r["assignments"]["half_walk"]
    assert m.is_refusal(a.beta) and a.beta.code == m.REASON_MIXED, \
        "the tiebreak was adopted anyway: half_walk reads %r" % (a.beta,)
    assert "REFUSED as a rule" in tb["verdict"]

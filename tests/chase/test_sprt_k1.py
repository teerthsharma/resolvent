"""CHASE round 6 it.0 - the SPRT that is meant to close K1's sign-flip clause.

Subject: scale/sprt.py (does not exist when this file is first run - that is the
RED state, and it is deliberate).

Three things are bound here, in the order the contract requires them:

1. THE MAPPING. K1's flip clause is a statement about a LOG-LOG SLOPE, and an
   SPRT is a statement about a PER-DRAW RATE. The mapping between them is
   pre-registered as module constants in `scale/sprt.py` and checked here by
   value. Once the first draw lands these numbers are immutable, so a test that
   pins them is the only thing that makes "immutable" mean anything.

2. THE MACHINERY. Wald's thresholds, the log-likelihood-ratio recursion, and
   Wald's expected sample size, each reproduced here by an independent path so
   that agreement is evidence rather than a copy of one implementation.

3. THE CALIBRATION. A sequential test that has never been run against a known
   answer is not an instrument. Bernoulli streams are simulated below r0, above
   r1, and between them, and the realised error rates are compared against the
   nominal ones. The last test in this file deletes the decision logic and
   requires the calibration to CATCH it; a calibration that a data-blind decider
   passes measures nothing.
"""
from __future__ import annotations

import math
import pathlib
import sys
from decimal import Decimal, getcontext

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scale import sprt as S            # noqa: E402  (RED until scale/sprt.py exists)


# ==========================================================================
# 1. THE MAPPING - K1's slope hypotheses expressed as per-draw flip rates
# ==========================================================================

def test_the_preregistered_constants_are_exactly_these_numbers():
    """Pin every constant the mapping depends on. A change to any of them after
    the first draw is a boundary edit, which the contract forbids."""
    assert S.K_REF == 8
    assert S.K_GRID == (8, 16, 32, 64, 128, 256)
    assert S.R_ANCHOR == 6.0 / 400.0            # round 5's k=8 cell, 6 of 400
    assert S.R_ANCHOR == 0.015
    assert S.B_MET == -0.4                      # K1's bar: slope <= -0.4
    assert S.B_FLAT == 0.0                      # the alternative: no decay
    assert (S.ALPHA, S.BETA) == (0.05, 0.05)


def test_rate_at_is_the_power_law_through_the_anchor():
    """r(k) = r_anchor * (k / k_ref) ** b, checked against an independent
    exp/log evaluation rather than against the implementation's own arithmetic."""
    for b in (-1.0, -0.4, -0.2, 0.0):
        for k in S.K_GRID:
            want = S.R_ANCHOR * math.exp(b * math.log(k / S.K_REF))
            assert S.rate_at(k, b) == pytest.approx(want, rel=0, abs=1e-15)


def test_the_anchor_cell_is_fixed_under_every_slope():
    """At k = k_ref both hypotheses give the SAME rate, so the anchor cell can
    carry no information about the slope. That is why re-using round 5's k=8
    count as the anchor does not double-count it."""
    for b in (-2.0, -0.4, 0.0, +1.0):
        assert S.rate_at(S.K_REF, b) == S.R_ANCHOR
    r0, r1 = S.hypotheses(S.K_REF)
    assert r0 == r1 == S.R_ANCHOR


def test_h0_is_below_h1_at_every_informative_cell():
    """The contract writes `H0: r <= r0` vs `H1: r >= r1`, which requires
    r0 < r1. Under the mapping, H0 is the STEEP-decay hypothesis (K1's bar met)
    and H1 is the FLAT one (bar not met), so H0 is the low rate for k > k_ref."""
    for k in S.K_GRID:
        r0, r1 = S.hypotheses(k)
        if k == S.K_REF:
            assert r0 == r1
        else:
            assert 0.0 < r0 < r1 < 1.0, (k, r0, r1)


def test_the_two_hypothesis_rates_at_the_extreme_cell_are_these_values():
    """k = 256 is 32x the anchor and 32 ** 0.4 == 4.0 exactly in real
    arithmetic, so the steep hypothesis is exactly a quarter of the flat one."""
    r0, r1 = S.hypotheses(256)
    assert r1 == pytest.approx(0.015, abs=1e-15)
    assert r0 == pytest.approx(0.015 / 4.0, abs=1e-12)
    assert r1 / r0 == pytest.approx(4.0, abs=1e-9)


def test_accepting_h1_means_the_flip_clause_failed():
    """The verdict wording is part of the pre-registration: it decides what the
    +1 for K-H is awarded FOR. Bind it so it cannot drift after the run."""
    assert S.verdict_text("H1") == "K1 FLIP CLAUSE FAILS: slope shallower than -0.4"
    assert S.verdict_text("H0") == "K1 FLIP CLAUSE MET: slope at least as steep as -0.4"
    assert S.verdict_text(None) == "UNDECIDED: no boundary crossed within the cap"


# ==========================================================================
# 2. THE MACHINERY
# ==========================================================================

def test_thresholds_reproduce_log_19_by_two_independent_paths():
    """Contract 1.4: at (alpha, beta) = (0.05, 0.05) the thresholds are
    +-2.944439 and agree with +-log(19) to < 5e-5.

    Path A: the module.  Path B: 50-digit Decimal, computed here."""
    lo, hi = S.thresholds(0.05, 0.05)

    getcontext().prec = 50
    hi_dec = (Decimal(1) - Decimal("0.05")) / Decimal("0.05")
    lo_dec = Decimal("0.05") / (Decimal(1) - Decimal("0.05"))
    hi_50 = hi_dec.ln()
    lo_50 = lo_dec.ln()

    assert abs(Decimal(repr(hi)) - hi_50) < Decimal("1e-15"), (hi, hi_50)
    assert abs(Decimal(repr(lo)) - lo_50) < Decimal("1e-15"), (lo, lo_50)
    assert abs(hi - math.log(19.0)) < 5e-5
    assert abs(lo + math.log(19.0)) < 5e-5
    assert f"{hi:.6f}" == "2.944439" and f"{lo:.6f}" == "-2.944439"
    assert lo == -hi                      # symmetric only because alpha == beta


def test_lambda_matches_the_closed_form_of_the_contract():
    """Contract 1.4's closed form, evaluated here from the counts alone, must
    equal the module's incremental accumulation over the same stream."""
    r0, r1 = S.hypotheses(256)
    stream = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
    t = S.Sprt(r0, r1, alpha=0.05, beta=0.05)
    for x in stream:
        t.update(x)
    n, x = len(stream), sum(stream)
    closed = x * math.log(r1 / r0) + (n - x) * math.log((1.0 - r1) / (1.0 - r0))
    assert t.n == n and t.x == x
    assert t.lam == pytest.approx(closed, abs=1e-12)


def test_a_cell_at_the_anchor_moves_lambda_by_exactly_zero():
    t = S.Sprt(*S.hypotheses(S.K_REF), alpha=0.05, beta=0.05)
    for x in (0, 1, 1, 0):
        t.update(x)
    assert t.lam == 0.0


def test_wald_expected_n_is_far_below_the_fixed_n_plan():
    """Round 5 priced fixed-n closure at ~20000 draws per cell. Wald's E[N] is
    printed beside the plan so the saving is stated, not assumed."""
    r0, r1 = S.hypotheses(256)
    n0 = S.expected_n(r0, r0, r1, 0.05, 0.05)
    n1 = S.expected_n(r1, r0, r1, 0.05, 0.05)
    assert 0.0 < n0 < 2000.0, n0
    assert 0.0 < n1 < 2000.0, n1
    assert n0 < 20000.0 and n1 < 20000.0


def test_expected_n_agrees_with_a_simulation_of_the_same_test():
    """Wald's E[N] is an approximation (it ignores overshoot). Require the
    simulated mean sample size to sit within 25% of it - close enough that the
    formula is the same object as the running test, loose enough that the known
    overshoot bias is not called a bug."""
    r0, r1 = S.hypotheses(256)
    formula = S.expected_n(r0, r0, r1, 0.05, 0.05)
    cal = S.calibrate(r0, r0, r1, n_rep=2000, max_n=200000, seed=11,
                      alpha=0.05, beta=0.05)
    assert cal["truncated"] == 0.0, cal
    assert abs(cal["mean_n"] - formula) / formula < 0.25, (cal["mean_n"], formula)


# ==========================================================================
# 3. THE CALIBRATION - run the procedure against known answers
# ==========================================================================

CAL_TOL = 0.02          # simulation slack at n_rep = 4000


def test_calibration_below_r0_accepts_h0_at_the_nominal_rate():
    r0, r1 = S.hypotheses(256)
    cal = S.calibrate(r0 / 2.0, r0, r1, n_rep=4000, max_n=200000, seed=1,
                      alpha=0.05, beta=0.05)
    assert cal["accept_h1"] <= 0.05 + CAL_TOL, cal
    assert cal["accept_h0"] >= 0.90, cal
    assert cal["truncated"] == 0.0, cal


def test_calibration_at_r0_holds_alpha():
    r0, r1 = S.hypotheses(256)
    cal = S.calibrate(r0, r0, r1, n_rep=4000, max_n=200000, seed=2,
                      alpha=0.05, beta=0.05)
    assert cal["accept_h1"] <= 0.05 + CAL_TOL, cal


def test_calibration_at_r1_holds_beta():
    r0, r1 = S.hypotheses(256)
    cal = S.calibrate(r1, r0, r1, n_rep=4000, max_n=200000, seed=3,
                      alpha=0.05, beta=0.05)
    assert cal["accept_h0"] <= 0.05 + CAL_TOL, cal
    assert cal["accept_h1"] >= 0.90, cal


def test_calibration_above_r1_accepts_h1():
    r0, r1 = S.hypotheses(256)
    cal = S.calibrate(r1 * 2.0, r0, r1, n_rep=4000, max_n=200000, seed=4,
                      alpha=0.05, beta=0.05)
    assert cal["accept_h0"] <= 0.05 + CAL_TOL, cal


def test_between_r0_and_r1_is_an_indifference_region_and_says_so():
    """Wald controls no error rate strictly between the two rates. The procedure
    must still terminate there, and BOTH decisions must actually occur - if one
    of them never happened the 'indifference' claim would be untested."""
    r0, r1 = S.hypotheses(256)
    mid = math.sqrt(r0 * r1)
    cal = S.calibrate(mid, r0, r1, n_rep=4000, max_n=200000, seed=5,
                      alpha=0.05, beta=0.05)
    assert cal["accept_h0"] > 0.05 and cal["accept_h1"] > 0.05, cal
    assert cal["accept_h0"] + cal["accept_h1"] + cal["truncated"] == \
        pytest.approx(1.0, abs=1e-12)
    assert cal["truncated"] == 0.0, cal


# ==========================================================================
# 4. THE MUST-FIRE - would the calibration pass with the logic deleted?
# ==========================================================================

def test_calibration_catches_a_decider_that_never_reads_the_data(monkeypatch):
    """MUST-FIRE CONTROL for every calibration number above.

    Replace the boundary decision with one that ignores the stream and always
    returns H0. Under a stream at r1 its miss rate is 1.0 against a nominal
    beta of 0.05, so the calibration MUST reject it. If this test fails, the
    calibration table above is decoration: it would report the same 'passes' for
    a test with no test in it.
    """
    r0, r1 = S.hypotheses(256)

    class Blind(S.Sprt):
        @property
        def decision(self):
            return "H0"                 # data-blind, on purpose

    monkeypatch.setattr(S, "Sprt", Blind)
    cal = S.calibrate(r1, r0, r1, n_rep=500, max_n=1000, seed=6,
                      alpha=0.05, beta=0.05)
    assert cal["accept_h0"] == 1.0, cal
    assert cal["accept_h1"] == 0.0, cal
    assert cal["accept_h0"] > 0.05 + CAL_TOL, (
        "a decider that never reads the data passed the beta calibration; the "
        "calibration measures nothing"
    )


def test_calibration_catches_a_lambda_that_never_moves(monkeypatch):
    """Second must-fire: delete the accumulation instead of the comparison.
    A Lambda pinned at 0 never crosses either boundary, so every replication
    must truncate. `truncated == 1.0` is the signature, and every genuine
    calibration above asserts `truncated == 0.0`."""
    r0, r1 = S.hypotheses(256)

    class Frozen(S.Sprt):
        def update(self, x):
            self.n += 1
            self.x += int(x)
            return self.lam          # lam never moves

    monkeypatch.setattr(S, "Sprt", Frozen)
    cal = S.calibrate(r1, r0, r1, n_rep=200, max_n=500, seed=7,
                      alpha=0.05, beta=0.05)
    assert cal["truncated"] == 1.0, cal
    assert cal["accept_h0"] == 0.0 and cal["accept_h1"] == 0.0, cal


# ==========================================================================
# 5. THE DEPENDENCE - the SPRT is reusable without torch (issue #3)
# ==========================================================================

def test_importing_the_sprt_does_not_load_torch():
    """`Sprt`, `expected_n` and `calibrate` are pure Python, so a consumer must
    not take a torch dependency for them. Run in a fresh interpreter, because
    this test process has torch loaded already. Fails against the module as it
    stood before issue #3, whose top level ran `import torch` to pin a thread
    count nothing in it used."""
    import subprocess
    probe = "import sys, scale.sprt; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", probe], cwd=ROOT,
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False", out.stdout + out.stderr

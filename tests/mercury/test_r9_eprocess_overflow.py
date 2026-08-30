"""Regression battery for the e-process overflow and its untested branch.

Two defects, both filed by Deimos and both reproduced independently here before
any fix landed:

  1. `Eprocess.value` re-materialises the accumulated log-evidence with a bare
     `math.exp`, so it raises `OverflowError: math range error` once the mixture
     log passes ~709. On the shipped 10240-draw pooled stream it crosses the
     decision threshold at draw 67 and then crashes at draw 10135.
  2. `calibrate()` certifies a vectorised re-derivation of the same product and
     never instantiates `Eprocess` at all, so the object that reads live data is
     not the object the must-fire battery vouches for.

The fix follows `eprocess_perdraw.log10_max_attainable`'s precedent: the
quantity is reported in log space, and the linear reader saturates to `inf`
rather than lying or raising.

Run this file alone:

    python -m pytest tests/mercury/test_r9_eprocess_overflow.py -q
"""
from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from scale import eprocess as EP


SHIPPED_POOL = 2048 * 5          #: --seeds 0 1 2 3 4 at n_eval=2048
BIND_TOL = 1e-9                  #: measured worst gap was 3.553e-15 at horizon 2000


def _winning_stream(n: int = SHIPPED_POOL) -> np.ndarray:
    """The stream a strong, plausible arm produces. Deimos's exact draw."""
    rng = np.random.default_rng(0)
    return np.clip(rng.normal(0.3, 0.3, n), -EP.B, EP.B)


# --------------------------------------------------------------------------
# 1 -- the process survives its own designed operating range
# --------------------------------------------------------------------------
#
# Property: for every stream of at most SHIPPED_POOL draws satisfying |d| <= B,
# `Eprocess.update` returns without raising, and the decision it reaches is the
# decision it reached at the draw where it first crossed.

def test_survives_the_full_shipped_pooled_stream():
    e = EP.Eprocess()
    crossed_at = None
    for i, d in enumerate(_winning_stream(), 1):
        v = e.update(float(d))
        if crossed_at is None and v >= EP.THRESHOLD:
            crossed_at = i
    assert e.t == SHIPPED_POOL
    assert crossed_at == 67, "the stream must still cross where it always did"
    assert e.crossed


def test_survives_the_adversarial_maximum_stream():
    """`d = +B` every draw is the fastest-growing admissible stream."""
    e = EP.Eprocess()
    for _ in range(3000):
        e.update(EP.B)
    assert e.t == 3000
    assert e.crossed


def test_log_value_stays_finite_where_the_linear_value_saturates():
    e = EP.Eprocess()
    for _ in range(3000):
        e.update(EP.B)
    assert math.isinf(e.value), "3000 draws of +B is past the double range"
    assert math.isfinite(e.log_value)
    # closed form: t * log1p(lam) per arm, mixed over the grid
    want = EP._logsumexp([3000 * math.log1p(lam) for lam in EP.LAMBDA_GRID]) \
        - math.log(len(EP.LAMBDA_GRID))
    assert e.log_value == pytest.approx(want, rel=1e-12)
    assert e.log_peak == pytest.approx(want, rel=1e-12)


def test_linear_value_is_unchanged_wherever_it_is_representable():
    """The saturating reader must not perturb any number a reader already binds."""
    e = EP.Eprocess()
    assert e.value == 1.0
    assert e.peak == 1.0
    for d in _winning_stream(60):
        e.update(float(d))
        assert math.isfinite(e.value)
        assert e.value == pytest.approx(math.exp(e.log_value), rel=1e-15)
        assert e.peak == pytest.approx(math.exp(e.log_peak), rel=1e-15)


def test_crossed_agrees_with_the_linear_comparison_while_both_are_defined():
    """`log_peak >= log(threshold)` must decide exactly as `peak >= threshold`."""
    e = EP.Eprocess()
    for d in _winning_stream(400):
        e.update(float(d))
        if math.isfinite(e.peak):
            assert e.crossed == (e.peak >= e.threshold)


# --------------------------------------------------------------------------
# 2 -- the guarantees the fix must NOT have bought its way out of
# --------------------------------------------------------------------------

def test_update_still_raises_on_an_out_of_bound_difference():
    """Clamping |d| > B would void Ville's inequality. It must still raise."""
    e = EP.Eprocess()
    with pytest.raises(ValueError, match="exceeds the a-priori bound"):
        e.update(EP.B + 1e-9)
    with pytest.raises(ValueError, match="non-finite"):
        e.update(float("nan"))


def test_the_preregistered_constants_are_untouched():
    assert EP.ALPHA_FAMILY == 0.05
    assert EP.N_DIRECTIONS == 2
    assert EP.THRESHOLD == 40.0
    assert EP.CLIP_C == 2.0
    assert EP.B == 2.0
    assert EP.MIN_T_MIXTURE == 13


def test_pair_decision_is_unchanged_on_a_deciding_stream():
    pair = EP.Pair()
    for d in _winning_stream(400):
        pair.update(float(d))
    assert pair.decision == "settled"


# --------------------------------------------------------------------------
# 3 -- the load-bearing half: the battery must exercise the live class
# --------------------------------------------------------------------------
#
# Property: for every calibration run, the vectorised path `calibrate` measures
# and the `Eprocess` class production runs agree in log space to BIND_TOL, and
# the run reports the gap it measured.

def test_calibrate_exercises_the_class_that_reads_live_data():
    src = inspect.getsource(EP.calibrate) + inspect.getsource(EP._eprocess_log_path)
    assert "Eprocess" in src, "the battery must instantiate the live class"


def test_calibrate_reports_the_agreement_gap_it_measured():
    c = EP.calibrate(EP.PLANTED_LARGE, n_rep=40, horizon=200, seed=3)
    assert "eprocess_bind_gap" in c
    assert c["eprocess_bind_gap"] <= BIND_TOL
    assert c["eprocess_bind_replays"] >= 1


def test_the_bind_check_can_fail(monkeypatch):
    """Adversarial pass: a clause that cannot read FALSE is vacuous.

    Perturb the live class's own accumulation and the battery must refuse the
    calibration instead of certifying a path production does not run.
    """
    real = EP.Eprocess.update

    def drifted(self, d):
        out = real(self, d)
        self.log_arm[0] += 1e-6
        return out

    monkeypatch.setattr(EP.Eprocess, "update", drifted)
    with pytest.raises(ValueError, match="does not agree"):
        EP.calibrate(EP.PLANTED_LARGE, n_rep=40, horizon=200, seed=3)


def test_the_bind_check_is_skipped_for_the_peeking_control():
    """`peek=True` is the deliberately broken single-arm rule. `Eprocess` does
    not implement it, so binding the two there would compare different objects."""
    c = EP.calibrate(EP.NULL_RADEMACHER, n_rep=40, horizon=200, seed=3, peek=True)
    assert c["eprocess_bind_gap"] is None
    assert c["eprocess_bind_replays"] == 0


# --------------------------------------------------------------------------
# 4 -- max_peak no longer reports a smaller, wrong number
# --------------------------------------------------------------------------

SATURATING = EP.Spec("saturating (+B every draw)", "gauss", 2.0, 0.0)


def test_max_peak_saturates_instead_of_silently_clamping():
    c = EP.calibrate(SATURATING, n_rep=2, horizon=2000, seed=0)
    assert math.isinf(c["max_peak"]), "past the double range, inf is the truth"
    assert c["max_peak"] != pytest.approx(math.exp(700.0)), "the old silent clamp"
    # the number itself survives in log space
    want = (EP._logsumexp([2000 * math.log1p(lam) for lam in EP.LAMBDA_GRID])
            - math.log(len(EP.LAMBDA_GRID))) / math.log(10.0)
    assert c["log10_max_peak"] == pytest.approx(want, rel=1e-12)
    assert c["log10_max_peak"] > 300.0


def test_max_peak_is_exact_where_it_is_representable():
    c = EP.calibrate(EP.NULL_RADEMACHER, n_rep=40, horizon=200, seed=3)
    assert math.isfinite(c["max_peak"])
    assert c["max_peak"] > 1.0
    assert math.log10(c["max_peak"]) == pytest.approx(c["log10_max_peak"], rel=1e-12)

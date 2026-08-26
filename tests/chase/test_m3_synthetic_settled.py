"""CHASE round 6 it.0 - can the M3 harness detect a PLANTED settled-vs-twin gap?

Subject: scale/m3_synthetic_settled.py (does not exist when this file is first
run - that is the RED state, and it is deliberate).

WHY THIS EXISTS. RULE 2 pins the real settled-vs-unsettled M3 run to iteration
12. If the harness cannot detect a difference that was planted on purpose, a
null result at iteration 12 carries no information at all - it would be
indistinguishable from a harness that cannot report a difference of any size.
Round 5 shipped four gates that could not fire; this file is the check that the
comparison machinery can.

TWO SYNTHETIC ARMS, BOTH WITH A KNOWN ANSWER:

  planted : the shipped arm plus one extra scalar-weighted input feature that
            carries the oracle value. The weight initialises to zero, so the arm
            is BITWISE the twin before training and only diverges by learning.
            Correct answer: PLANTED WINS.
  null    : the same construction with the same feature SHUFFLED across the
            batch - identical marginal distribution, zero per-example
            information, one extra parameter. Correct answer: NO DIFFERENCE.

Both arms are a DECLARED CHEAT. They exist to exercise the instrument and no
number taken from them is a capability claim about anything.

THE CONTRAST INSTRUMENT. The shipped `negation_scope.bootstrap_ci` is a MARGINAL
interval over one eval batch from one training run. Two marginal intervals
overlapping is not a test of a difference, and the repo already records that its
width is smaller than the arm's own seed-to-seed spread
(tests/chase/test_m3_capability_harness.py::
test_reported_ci_covers_run_to_run_variation_of_the_same_arm). The contrast used
here is therefore a PAIRED bootstrap over training seeds, on the same eval
batch, which is what G6 asks for.
"""
from __future__ import annotations

import math
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scale import m3_capability as M3                    # noqa: E402
from scale import negation_scope as NS                   # noqa: E402
from scale import m3_synthetic_settled as SS             # noqa: E402  (RED)


# ==========================================================================
# 0. SOFTMAX FIRST - the published reading is reproduced before any synthetic
#    number is produced.
# ==========================================================================

def test_softmax_published_reading_is_reproduced_before_anything_else():
    """results/m3_capability.txt, RUN 2026-08-25 18:56:53, s=64 d=24 steps=150
    n_train=2048 n_eval=512 seed=0, torch.get_num_threads()=2."""
    got = SS.reproduce_softmax()
    want = SS.PUBLISHED_SOFTMAX_2048
    for field in ("n_params", "nrmse0_train", "nrmse0_eval",
                  "train_nrmse", "eval_nrmse", "ci_lo", "ci_hi"):
        a, b = got[field], want[field]
        if isinstance(b, int):
            assert a == b, (field, a, b)
        else:
            assert f"{a:.6f}" == f"{b:.6f}", (field, a, b)


# ==========================================================================
# 1. THE PLANTED ARM IS BITWISE THE TWIN BEFORE TRAINING
# ==========================================================================

def test_planted_arm_at_init_is_bitwise_the_shipped_arm():
    """The extra weight initialises to exactly 0.0, so the planted arm cannot
    win at 0 steps. Without this the harness's own RED gate (0-step NRMSE >= 1.0)
    would be tripped by the plant and the dry-run would be measuring the cheat
    rather than the comparison."""
    x, _y, f, p = NS.make_batch(16, 64, 24, d_model=M3.D_MODEL, seed=0)

    torch.manual_seed(0)
    twin = M3.Arm("softmax", s=64)
    torch.manual_seed(0)
    planted = SS.SyntheticSettledArm("softmax", s=64, f=f, p=p, shuffle=False)

    assert float(planted.hint_w.abs().max()) == 0.0
    with torch.no_grad():
        a, b = twin(x), planted(x)
    assert torch.equal(a, b), float((a - b).abs().max())


def test_planted_and_null_arms_stay_inside_the_ten_percent_param_bar():
    torch.manual_seed(0)
    base = M3.n_params(M3.Arm("softmax", s=64))
    for shuffle in (False, True):
        torch.manual_seed(0)
        n = M3.n_params(SS.SyntheticSettledArm("softmax", s=64, f=39, p=62,
                                               shuffle=shuffle))
        assert n == base + 1, (n, base)
        assert abs(n - base) / base <= 0.10


# ==========================================================================
# 2. THE CONTRAST INSTRUMENT, ON AN EXACTLY KNOWN MARGIN
#
# pred_settled = y + (1 - delta) * (pred_twin - y) shrinks every residual by the
# same factor, so NRMSE_settled = (1 - delta) * NRMSE_twin EXACTLY. The margin is
# therefore known in closed form and does not depend on any training.
# ==========================================================================

def test_shrinking_the_residual_scales_nrmse_exactly():
    """Exact in real arithmetic; the tolerance is float32's, not a hedge.

    `torch.randn` is float32 and `nrmse` reduces in float32, so the identity is
    read through a float32 sum of 512 squares. Measured residual at rel=1e-12 is
    6.4e-8, which is one float32 epsilon (1.19e-7) and not a defect in the
    identity. Anything above 1e-6 here would be.
    """
    g = torch.Generator().manual_seed(0)
    y = torch.randn(512, generator=g)
    pred = y + torch.randn(512, generator=g)
    base = NS.nrmse(pred, y)
    for delta in (0.0, 0.1, 0.25, 0.5):
        shrunk = SS.shrink_toward(pred, y, delta)
        assert NS.nrmse(shrunk, y) == pytest.approx((1.0 - delta) * base,
                                                    rel=1e-6)
    assert NS.nrmse(SS.shrink_toward(pred, y, 0.0), y) == base   # exact at 0


def test_contrast_recovers_a_planted_margin_and_excludes_zero():
    """Five paired seeds, each with a small amount of seed noise on top of a
    planted gap of 0.20 NRMSE. The interval must exclude zero and cover the
    planted value."""
    twin = [0.95, 0.96, 0.94, 0.97, 0.95]
    settled = [t - 0.20 for t in twin]
    c = SS.contrast(twin, settled, n_boot=10000, seed=0)
    assert c["delta"] == pytest.approx(0.20, abs=1e-12)
    assert c["ci_lo"] > 0.0, c
    # the planted value is `0.95 - 0.75`, which is 0.19999999999999996 in
    # float64, so the interval is asked to cover the delta it was built from
    # rather than the decimal literal it was written as.
    assert c["ci_lo"] <= c["delta"] <= c["ci_hi"], c
    assert c["ci_lo"] <= 0.20 + 1e-12 and 0.20 - 1e-12 <= c["ci_hi"], c
    assert c["verdict"] == "SETTLED WINS"


def test_contrast_does_not_call_seed_noise_a_win():
    """The correct answer for two constructions that differ only by noise is NO
    DIFFERENCE. A harness that calls this a win would have called round 5's four
    adjacent gates green as well."""
    twin = [0.950, 0.962, 0.941, 0.968, 0.955]
    settled = [0.958, 0.949, 0.953, 0.951, 0.961]
    c = SS.contrast(twin, settled, n_boot=10000, seed=0)
    assert c["ci_lo"] < 0.0 < c["ci_hi"], c
    assert c["verdict"] == "NO DIFFERENCE"


def test_contrast_reports_the_twin_winning_when_the_twin_wins():
    """Both directions, so the verdict is not a one-sided rubber stamp."""
    twin = [0.75, 0.76, 0.74, 0.77, 0.75]
    settled = [t + 0.20 for t in twin]
    c = SS.contrast(twin, settled, n_boot=10000, seed=0)
    assert c["delta"] == pytest.approx(-0.20, abs=1e-12)
    assert c["ci_hi"] < 0.0, c
    assert c["verdict"] == "TWIN WINS"


# ==========================================================================
# 3. THE MUST-FIRE - would the dry-run pass with the verdict logic deleted?
# ==========================================================================

def test_a_verdict_that_always_says_settled_wins_fails_the_null(monkeypatch):
    """MUST-FIRE CONTROL for every verdict above. Replace the verdict with a
    constant 'SETTLED WINS' and require the null case to catch it. If this test
    fails, the two-direction dry-run is decoration: it would report the same
    'right answer' for a harness with no comparison in it."""
    monkeypatch.setattr(SS, "verdict_of", lambda lo, hi: "SETTLED WINS")
    twin = [0.950, 0.962, 0.941, 0.968, 0.955]
    settled = [0.958, 0.949, 0.953, 0.951, 0.961]
    c = SS.contrast(twin, settled, n_boot=2000, seed=0)
    assert c["verdict"] == "SETTLED WINS"          # the deleted logic, seen
    assert not (c["ci_lo"] > 0.0), (
        "the null case produced an interval excluding zero, so a constant "
        "verdict would have been indistinguishable from a working one"
    )


def test_verdict_of_is_a_pure_function_of_the_interval():
    assert SS.verdict_of(0.01, 0.20) == "SETTLED WINS"
    assert SS.verdict_of(-0.20, -0.01) == "TWIN WINS"
    assert SS.verdict_of(-0.01, 0.20) == "NO DIFFERENCE"
    assert SS.verdict_of(0.0, 0.20) == "NO DIFFERENCE"      # boundary is strict
    assert SS.verdict_of(-0.20, 0.0) == "NO DIFFERENCE"


# ==========================================================================
# 4. THE PLANT SURVIVES THE REAL TRAINING LOOP
#
# Small settings so this stays a test rather than a run; the full dry-run at the
# harness's own settings lives in scale/m3_synthetic_settled.py::main.
# ==========================================================================

FAST = dict(s=64, d=24, steps=60, n_train=256, n_eval=256, seed=0, n_seeds=3)


def test_planted_arm_beats_the_twin_through_the_shipped_training_loop():
    r = SS.dry_run(arm="planted", **FAST)
    assert r["verdict"] == "SETTLED WINS", r
    assert r["ci_lo"] > 0.0, r


def test_null_arm_is_not_called_a_win_through_the_shipped_training_loop():
    """The null verdict is only worth something if the synthetic arm was
    actually built. A class swap that silently failed would return the shipped
    twin on BOTH sides and produce a perfect 'NO DIFFERENCE' for a reason that
    has nothing to do with the comparison, so the parameter count is checked in
    the same test that reads the verdict."""
    r = SS.dry_run(arm="null", **FAST)
    assert r["n_params_settled"] == r["n_params_twin"] + 1, r
    assert r["verdict"] == "NO DIFFERENCE", r


def test_the_class_swap_reaches_the_planted_arm_too():
    r = SS.dry_run(arm="planted", **FAST)
    assert r["n_params_settled"] == r["n_params_twin"] + 1, r


def test_the_dry_run_reports_the_zero_step_red_gate_for_every_arm():
    """The harness aborts on a 0-step NRMSE below 1.0. Both synthetic arms must
    clear that gate, otherwise the dry-run is exercising the abort path instead
    of the comparison path."""
    for arm in ("planted", "null"):
        r = SS.dry_run(arm=arm, **FAST)
        for v in r["nrmse0_settled"] + r["nrmse0_twin"]:
            assert not (math.isnan(v) or math.isinf(v)), (arm, v)
            assert v >= 1.0, (arm, v)

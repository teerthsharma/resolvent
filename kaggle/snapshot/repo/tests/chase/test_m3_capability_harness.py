"""CHASE round 3 - can the rebuilt M3 CAPABILITY harness print a wrong number?

Every test here compares VALUES. Nothing greps source for a name, nothing puts a
decision downstream of a shell pipeline, and every RED test ships alongside a
control that is SEEN to fire in this same file.

Subject: scale/m3_capability.py (the harness) and scale/negation_scope.py (the
task, the oracle, and the calibration bar arms are credited against).
"""
from __future__ import annotations

import ast
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scale import m3_capability as M3            # noqa: E402
from scale import negation_scope as NS           # noqa: E402


# --------------------------------------------------------------------------
# defaults, read out of the files themselves so a default change re-arms these
# --------------------------------------------------------------------------

def _argparse_defaults(path):
    """flag -> default value, from the module's own add_argument calls."""
    out = {}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant)):
            for kw in node.keywords:
                if kw.arg == "default":
                    try:
                        out[node.args[0].value] = ast.literal_eval(kw.value)
                    except ValueError:
                        pass
    return out


M3_DEF = _argparse_defaults(ROOT / "scale" / "m3_capability.py")
NS_DEF = _argparse_defaults(ROOT / "scale" / "negation_scope.py")
S_DEF = M3_DEF["--s"]
D_DEF = M3_DEF["--d"]
N_PROBE = 16


def test_defaults_were_actually_read():
    """Guard on the reader: a silently-empty dict would make the rest of this
    file assert on nothing at all."""
    assert (S_DEF, D_DEF) == (64, 24), M3_DEF
    assert M3_DEF["--steps"] == 150 and M3_DEF["--n-train"] == 128, M3_DEF
    assert NS_DEF["--distances"] == [256], NS_DEF


# --------------------------------------------------------------------------
# 1. what can each arm's OUTPUT actually see?
#
# Measured, not derived: perturb x at ONE position and look at the output at the
# query. The arm's own forward is the instrument; nothing is reimplemented.
# --------------------------------------------------------------------------

def _visible_positions(kind, s, d):
    """(positions the query output responds to, flipper index, payload index)."""
    torch.manual_seed(0)
    arm = M3.Arm(kind, s=s)
    x, _y, f, p = NS.make_batch(N_PROBE, s, d, d_model=M3.D_MODEL, seed=0)
    seen = set()
    with torch.no_grad():
        base = arm(x)
        for j in range(s):
            xp = x.clone()
            xp[:, j, :] = -xp[:, j, :]      # at j == f this negates the label y
            if float((arm(xp) - base).abs().max()) > 0.0:
                seen.add(j)
    return seen, f, p


def test_probe_is_live_the_softmax_arm_responds_to_the_flipper_position():
    """MUST-FIRE CONTROL for every assertion below. If this does not pass, the
    perturbation probe is blind and the RED results that follow mean nothing."""
    seen, f, p = _visible_positions("softmax", S_DEF, D_DEF)
    assert f in seen, (
        "the probe cannot detect the softmax arm reading position %d; visible=%s"
        % (f, sorted(seen))
    )
    assert p in seen
    assert len(seen) == S_DEF, "softmax reach %d of %d" % (len(seen), S_DEF)


@pytest.mark.parametrize("kind", list(M3.ARMS))
def test_every_arm_output_can_see_the_flipper_at_the_harness_defaults(kind):
    """An arm whose output is INDEPENDENT of the flipper cannot beat NRMSE 1.0 no
    matter what it learns, so its reading is predetermined and says nothing about
    the construction it is named after.

    The harness runs s=64 d=24 by default: flipper at 39, payload at 62, query at
    63. windowed_signed is sgate at window=8 with hop 2 `a @ (a @ x)`, so the
    query's support is [63 - 2*8, 63] = [47, 63]. 39 < 47.
    """
    seen, f, p = _visible_positions(kind, S_DEF, D_DEF)
    reach = (S_DEF - 1) - min(seen)
    assert f in seen, (
        "arm %r at the harness defaults (s=%d, d=%d) produces a query output that "
        "is BITWISE UNCHANGED when the flipper at position %d is negated - which "
        "negates the label y for every example. Measured reach %d positions "
        "(visible %d..%d); the label needs %d. The payload at %d IS visible, so "
        "this arm is exactly the `payload_only` reference predictor that the bar "
        "calibrates as a FAILURE (NRMSE 1.361782 at these settings). Its NRMSE is "
        "fixed before a single training step runs."
        % (kind, S_DEF, D_DEF, f, reach, min(seen), max(seen), D_DEF, p)
    )


def test_the_windowed_arm_reaches_the_flipper_across_the_recorded_sweep():
    """DONE.md records the M3 sweep at d = 24..54. Hop 2 over a band of w reaches
    2w = 16, so every distance in that sweep is outside the arm's support. Pin the
    exact crossover distance as a VALUE, not the mere fact of a band."""
    torch.manual_seed(0)
    arm = M3.Arm("windowed_signed", s=S_DEF)
    x, _y, _f, _p = NS.make_batch(4, S_DEF, D_DEF, d_model=M3.D_MODEL, seed=0)
    with torch.no_grad():
        q, k = arm.wq(x), arm.wk(x)
        a = arm._operator(q, k)
        row = S_DEF - 1
        support = a[:, row].abs().sum(0) + (a @ a)[:, row].abs().sum(0)
    max_d = row - int((support > 0).nonzero().min())
    assert max_d >= max(D_DEF, 54), (
        "windowed_signed reaches at most d=%d from the query (2 * W_WINDOW = %d); "
        "the harness default is d=%d and the recorded sweep runs d=24..54. Every "
        "one of those readings is taken on an arm that cannot see the flipper."
        % (max_d, 2 * M3.W_WINDOW, D_DEF)
    )


# --------------------------------------------------------------------------
# 2. the calibration bar: can its `oracle` control fire?
# --------------------------------------------------------------------------

def test_oracle_calibration_entry_is_an_identity_not_a_measurement():
    """Evidence, not a finding. calibrate_bar builds y with oracle(x,f,p) and then
    scores oracle(x,f,p) against it: nrmse(t, t), exactly 0.0 by construction for
    ANY oracle, so `oracle < 1e-6` carries zero bits."""
    c = NS.calibrate_bar(n=64, s=S_DEF, d=D_DEF)
    assert c["oracle"] == 0.0
    assert c["predict_the_mean"] == pytest.approx(1.0, abs=1e-9)


def _bar_says_calibrated(c):
    """The harness's own gate, copied by VALUE from scale/m3_capability.py:244."""
    return (abs(c["predict_the_mean"] - 1.0) < 1e-6
            and c["payload_only"] >= 1.0
            and c["oracle"] < 1e-6)


def test_the_bar_gate_can_report_broken(monkeypatch):
    """MUST-FIRE CONTROL for the gate expression itself. A task whose label IS the
    payload has to be rejected, and it is - via payload_only."""
    monkeypatch.setattr(NS, "oracle", lambda x, f, p: x[:, p, NS.CH_PAYLOAD])
    c = NS.calibrate_bar(n=256, s=S_DEF, d=D_DEF)
    assert not _bar_says_calibrated(c), c


def test_the_bar_rejects_a_task_with_no_long_range_dependence(monkeypatch):
    """The bar exists to catch a broken task before any arm is credited. Replace
    the oracle with |payload| - a purely LOCAL label at position s-2, adjacent to
    the query, with no flipper dependence at all, which destroys the whole M3
    premise - and ask the gate."""
    monkeypatch.setattr(NS, "oracle", lambda x, f, p: x[:, p, NS.CH_PAYLOAD].abs())
    c = NS.calibrate_bar(n=256, s=S_DEF, d=D_DEF)
    assert not _bar_says_calibrated(c), (
        "BAR CALIBRATED on a task whose label does not depend on the flipper at "
        "all: predict_the_mean=%.6f payload_only=%.6f oracle=%.6f. "
        "predict_the_mean is nrmse(mean(y), y) = 1.0 as an algebraic identity and "
        "oracle is nrmse(t, t) = 0.0 as an algebraic identity; only payload_only "
        "reads the task, and it is satisfied by any label that merely differs from "
        "the payload. The gate cannot see that the long-range dependence - the "
        "entire subject of M3 - is gone."
        % (c["predict_the_mean"], c["payload_only"], c["oracle"])
    )


# --------------------------------------------------------------------------
# 3. the distance the harness runs at vs the distance that was pre-registered
# --------------------------------------------------------------------------

def test_harness_default_distance_meets_the_preregistered_kill_distance():
    """CHECKLIST M3's kill is 'CIs overlap softmax at every d >= 256', and
    negation_scope's own --distances default is [256]. The capability harness
    defaults to d=24 at s=64, where make_batch rejects d=256 outright (it requires
    1 <= d < s-1, so s=64 caps d at 62)."""
    kill_d = min(NS_DEF["--distances"])
    with pytest.raises(ValueError):
        NS.make_batch(2, S_DEF, kill_d, d_model=M3.D_MODEL, seed=0)
    assert D_DEF >= kill_d, (
        "the capability harness defaults to d=%d at s=%d, and s=%d cannot host "
        "d=%d at all (make_batch requires 1 <= d < s-1, cap %d). Every number this "
        "harness appends to results/m3_capability.txt is taken at roughly 1/%dth "
        "the pre-registered kill distance and cannot resolve the pre-registered "
        "kill." % (D_DEF, S_DEF, S_DEF, kill_d, S_DEF - 2, kill_d // D_DEF)
    )


# --------------------------------------------------------------------------
# 4. equal parameter count, unequal capacity
# --------------------------------------------------------------------------

def test_param_counts_are_equal_across_every_arm():
    """Evidence: the PARAM MATCH line the harness prints is true and useless. It
    also only loops over ('pivot_signed', 'pivot_unsigned'), so windowed_signed is
    never in it."""
    counts = {}
    for kind in M3.ARMS:
        torch.manual_seed(0)
        counts[kind] = M3.n_params(M3.Arm(kind, s=S_DEF))
    assert set(counts.values()) == {4769}, counts


# --------------------------------------------------------------------------
# 5. the reported confidence interval vs actual run-to-run spread
#
# The pre-registered kill is 'CIs overlap softmax'. The CI is a bootstrap over
# ONE eval batch from ONE training run, so it measures eval sampling noise only.
# Training-seed variance is not in it.
# --------------------------------------------------------------------------

TRAIN_SEEDS = (0, 1, 2)


def _harness_batches():
    x_tr, y_tr, _f, _p = NS.make_batch(M3_DEF["--n-train"], S_DEF, D_DEF,
                                       d_model=M3.D_MODEL, seed=M3_DEF["--seed"])
    x_ev, y_ev, _f2, _p2 = NS.make_batch(M3_DEF["--n-eval"], S_DEF, D_DEF,
                                         d_model=M3.D_MODEL,
                                         seed=M3_DEF["--seed"] + 12345)
    return x_tr, y_tr, x_ev, y_ev


def test_reported_ci_covers_run_to_run_variation_of_the_same_arm():
    """Same arm, same data, same hyper-parameters, three initialisation seeds. If
    the spread of eval NRMSE across seeds exceeds the width of the CI the harness
    prints, then 'CIs do not overlap' can be produced by seed noise alone."""
    x_tr, y_tr, x_ev, y_ev = _harness_batches()
    runs = [M3.run_arm("softmax", x_tr, y_tr, x_ev, y_ev, s=S_DEF,
                       steps=M3_DEF["--steps"], seed=sd) for sd in TRAIN_SEEDS]
    evals = [r["eval_nrmse"] for r in runs]
    widths = [r["ci_hi"] - r["ci_lo"] for r in runs]
    spread = max(evals) - min(evals)
    assert spread <= min(widths), (
        "softmax arm, harness defaults, seeds %s: eval NRMSE %s -> spread %.6f, "
        "while the narrowest reported bootstrap CI is only %.6f wide (CIs %s). The "
        "printed interval resamples ONE eval batch from ONE training run, so it "
        "does not contain initialisation variance. The pre-registered kill 'CIs "
        "overlap softmax' is therefore decidable by the choice of seed."
        % (list(TRAIN_SEEDS), ["%.6f" % v for v in evals], spread, min(widths),
           ["[%.6f, %.6f]" % (r["ci_lo"], r["ci_hi"]) for r in runs])
    )


def test_the_harness_settings_admit_at_least_one_arm_that_beats_the_bar():
    """The bar has no POSITIVE control at the model level: `oracle` is an algebraic
    identity, not a trained model. So nothing establishes that any arm of this
    shape, at these settings, can read below NRMSE 1.0 on held-out data. Run the
    one arm that provably CAN see the flipper (softmax, full causal reach) and ask
    it."""
    x_tr, y_tr, x_ev, y_ev = _harness_batches()
    r = M3.run_arm("softmax", x_tr, y_tr, x_ev, y_ev, s=S_DEF,
                   steps=M3_DEF["--steps"], seed=M3_DEF["--seed"])
    assert r["eval_nrmse"] < 1.0, (
        "at the harness's own defaults (s=%d d=%d steps=%d n_train=%d lr=%s) the "
        "softmax arm reaches train NRMSE %.6f and eval NRMSE %.6f. No arm of this "
        "shape has been shown to pass the bar at these settings, so a reading of "
        "'above 1.0' for any arm is indistinguishable from the harness being "
        "unable to produce a pass at all."
        % (S_DEF, D_DEF, M3_DEF["--steps"], M3_DEF["--n-train"], M3.LR,
           r["train_nrmse"], r["eval_nrmse"])
    )

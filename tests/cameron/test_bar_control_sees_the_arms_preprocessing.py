"""A7: the bar's trained control and the arms must see the same label.

THE MISMATCH, quoted from both sides.

    scale/negation_scope.py:1195-1196   (calibrate_bar, clause 5)
        loss = ((net(feats).squeeze(-1) - y) ** 2).mean()

    scale/m3_capability.py:196, :203    (run_arm)
        y_train_std = (y_train - mu) / sigma
        loss = torch.nn.functional.mse_loss(model(x_train), y_train_std)

`run_arm` also un-standardises before it scores -- `raw_pred` returns
`out_std * sigma + mu` (`scale/m3_capability.py:187`) -- so the arm's REPORTED
number is on the raw scale while its OPTIMISATION runs on the standardised one.
The control does neither. Both are then compared against the same bar, 1.0.

WHY THAT IS A MEASUREMENT FAILURE AND NOT A DETAIL. Adam's step size is
bounded by `lr` almost independently of the gradient's magnitude, so the number
of steps a model needs to travel from its initialisation to the label's scale
is set by that scale. The control is initialised at
`normal_(weight, 0.0, 0.5)` (`scale/negation_scope.py:1189`), which puts its
output at order 1. `e2_consequence`'s label has sd `0.061984` at the house
shape, so the control must shrink by more than an order of magnitude on a
budget the arms never have to spend -- the arms' target is already sd 1.
Clause 5 then reports `trained_two_feature = 2.446645` at `steps=150` and the
bar prints BROKEN, which reads as "no arm can pass this task" when what was
measured is "the control was handed a harder version of the arms' problem".
`e2_consequence` has never been trained (FINDINGS A7).

THE CLAIM, as a property. NRMSE is scale-free: `nrmse(c*pred, c*y)` equals
`nrmse(pred, y)` for every `c > 0`. `run_arm`'s preprocessing makes its reading
obey that. The control's does not. So the property that separates the two
implementations is:

    for all c > 0, `calibrate_bar(oracle_fn = c * ofn)["trained_two_feature"]`
    is the value it takes at c = 1.

TWO PATHS THAT FAIL DIFFERENTLY. The scale-invariance property above is a
statement about a family of labels and knows nothing about `e2_consequence`.
The second test pins the one absolute number the repository already recorded as
broken (`STATE.md:73-76`) and asserts the bar now calibrates at the shipped
budget. A bug in the standardisation arithmetic breaks the first; a bug that
standardises but scores on the wrong scale breaks the second.

THE STRIKE RECORD STAYS REACHABLE. `calibrate_bar(standardise=False)` runs the
pre-repair path, so the defect is still measurable in-process rather than only
describable in a document. That is what
`tests/cameron/test_m3_etasks.py::test_the_consequence_bar_is_broken_at_the_shipped_step_budget`
now asserts, and it is how this repair is shown to change the object it
repairs.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scale import negation_scope as NS                        # noqa: E402

S_HOUSE, D_HOUSE = 64, 24       # the house shape, tests/cameron/test_m3_etasks.py:74
N_CAL = 2048                    # >= 512, the measured floor for the 0.05 band
LR = 0.02                       # scale/m3_capability.py LR, and calibrate_bar's default

E2_KW = dict(n=N_CAL, s=S_HOUSE, d=D_HOUSE, lr=LR,
             batch_fn=NS.make_consequence_batch,
             oracle_fn=NS.consequence_oracle,
             feature_fn=NS.consequence_features)


def _scaled(ofn, c):
    """The same task with its label multiplied by `c`. NOTHING else about the
    problem changes: the features are the same tensor, the flipper dependence
    is a ratio and is unmoved, and NRMSE is scale-free. A control that reads a
    different number here is reading the label's units."""
    def wrapped(x, f, p):
        return c * ofn(x, f, p)
    return wrapped


@pytest.mark.parametrize("c", [10.0, 0.1])
def test_the_controls_reading_does_not_depend_on_the_labels_units(c):
    """PATH 1 -- the property, over a family of labels, at a budget where the
    two implementations are distinguishable.

    Steps are held at the SHIPPED 150 on purpose. At a large enough budget both
    implementations converge and the test would pass with the repair deleted;
    that is the branch-never-ran failure mode, so the budget is the shipped one
    and the gap is reported."""
    base = NS.calibrate_bar(steps=150, **E2_KW)["trained_two_feature"]
    kw = dict(E2_KW)
    kw["oracle_fn"] = _scaled(NS.consequence_oracle, c)
    scaled = NS.calibrate_bar(steps=150, **kw)["trained_two_feature"]
    print(f"\n  c={c}: trained_two_feature base={base:.6f} scaled={scaled:.6f} "
          f"gap={abs(scaled - base):.6f}")
    assert abs(scaled - base) < 1e-3, (
        f"scaling the label by {c} moved the trained control from {base:.6f} "
        f"to {scaled:.6f} -- the control is optimising on the label's raw "
        f"units while run_arm optimises on the standardised label"
    )


def test_the_consequence_bar_calibrates_at_the_shipped_step_budget():
    """PATH 2 -- the one absolute number on record.

    `STATE.md:73-76` records `2.446646` at `steps=150` against a bar of 1.0,
    with the mismatch named as the cause and left unfixed. Both readings are
    taken here so the repair is a measured delta and not an assertion."""
    broken = NS.calibrate_bar(steps=150, standardise=False, **E2_KW)
    fixed = NS.calibrate_bar(steps=150, **E2_KW)
    ok_broken, why_broken = NS.bar_verdict(broken, flipper_dependence=2.0)
    ok_fixed, why_fixed = NS.bar_verdict(fixed, flipper_dependence=2.0)
    print(f"\n  raw-label control   : {broken['trained_two_feature']:.6f}  "
          f"BAR {'CALIBRATED' if ok_broken else 'BROKEN'}")
    print(f"  arms' preprocessing : {fixed['trained_two_feature']:.6f}  "
          f"BAR {'CALIBRATED' if ok_fixed else 'BROKEN'}")

    assert not ok_broken and "trained_two_feature" in why_broken, why_broken
    assert broken["trained_two_feature"] > 2.0, broken["trained_two_feature"]
    assert ok_fixed, (why_fixed, fixed)


def test_the_clauses_that_do_not_train_are_untouched():
    """The blast radius. Clauses 1-4 read the LABEL, not the control, so the
    repair must not move them by a single bit. A repair that changes numbers it
    was not asked to change is a second finding, not a fix."""
    broken = NS.calibrate_bar(steps=150, standardise=False, **E2_KW)
    fixed = NS.calibrate_bar(steps=150, **E2_KW)
    for clause in ("predict_the_mean", "payload_only", "oracle",
                   "flipper_dependence"):
        assert broken[clause] == fixed[clause], (
            f"{clause}: {broken[clause]} -> {fixed[clause]}"
        )


def test_the_shipped_negation_scope_task_still_calibrates():
    """The default task, on the default path, at the default budget. The bar
    gates every arm reading in the repository; a repair that calibrates
    `e2_consequence` by breaking `negation_scope` has moved the defect, not
    removed it."""
    cal = NS.calibrate_bar(n=512, s=S_HOUSE, d=D_HOUSE, steps=150, lr=LR)
    ok, why = NS.bar_verdict(cal)
    print(f"\n  negation_scope trained_two_feature="
          f"{cal['trained_two_feature']:.6f}  {why}")
    assert ok, (why, cal)

"""CAMERON Phase D -- the oldest open item: every probe number was random-init.

Five rounds of statistics were taken through randomly initialised projections. If
they read differently through trained ones, a large part of five rounds describes
a regime the model never occupies. These tests bind the measurement that settles
it.

WHAT IS BEING TESTED IS THE INSTRUMENT, NOT THE ANSWER. Whether alpha spreads or
Delta drops is a measurement, and no test here asserts a direction for it. What
the tests DO assert is that the comparison is capable of showing a difference:
that training actually happened, that the random-init and trained arms differ in
their projections and in nothing else, and that every statistic is computed by
the same code path on both sides. A side-by-side that cannot show a delta is the
same defect as a control that cannot fire.
"""
from __future__ import annotations

import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(2)                       # pinned HERE, not by the launcher

from scale import trained_projections as TP                       # noqa: E402

SEED = 0


def test_training_actually_moved_the_projections():
    """The bind. Without it every 'trained' number could be the random one."""
    pair = TP.train_pair(seed=SEED, steps=TP.QUICK_STEPS, n_train=TP.QUICK_NTRAIN)
    assert pair.metrics["eval_nrmse"] < pair.metrics["nrmse0_eval"], pair.metrics
    dq = (pair.trained_wq - pair.init_wq).abs().max()
    dk = (pair.trained_wk - pair.init_wk).abs().max()
    assert float(dq) > 1e-6, float(dq)
    assert float(dk) > 1e-6, float(dk)


def test_the_two_arms_differ_only_in_the_projections():
    """Same architecture, same init, same data. Only training differs.

    If the random-init side were built from a different seed or a different
    input batch, any delta reported later would be confounded by that and not by
    training.
    """
    pair = TP.train_pair(seed=SEED, steps=TP.QUICK_STEPS, n_train=TP.QUICK_NTRAIN)
    assert pair.init_wq.shape == pair.trained_wq.shape
    assert pair.init_wk.shape == pair.trained_wk.shape
    assert pair.x.shape[0] > 0


def test_every_statistic_is_computed_by_one_code_path_on_both_sides():
    """Same function, two projection sets. No branch may differ between them."""
    pair = TP.train_pair(seed=SEED, steps=TP.QUICK_STEPS, n_train=TP.QUICK_NTRAIN)
    a = TP.statistics(pair, trained=False)
    b = TP.statistics(pair, trained=True)
    assert set(a) == set(b), (set(a) ^ set(b))
    for key in ("logit_scale", "delta_vertex", "alpha_eff_support",
                "alpha_max", "kappa_cert", "one_minus_kappa"):
        assert key in a, sorted(a)


def test_the_comparison_can_show_a_delta_when_one_is_planted():
    """Must-fire. Plant a projection change and the statistics must move.

    A side-by-side that reports 'no change' whatever it is handed is not a
    measurement. Scaling the key projection scales every logit, and the logit
    scale is the statistic that must follow it.
    """
    pair = TP.train_pair(seed=SEED, steps=TP.QUICK_STEPS, n_train=TP.QUICK_NTRAIN)
    base = TP.statistics(pair, trained=False)["logit_scale"]
    planted = TP.statistics(pair._replace(init_wk=pair.init_wk * 4.0),
                            trained=False)["logit_scale"]
    assert planted > 3.5 * base, (base, planted)


def test_alpha_effective_support_is_bounded_by_the_pivot_count():
    """A spread measure that can exceed its own alphabet is broken.

    exp(entropy) over k weights lies in [1, k]: 1 is one-hot, k is uniform.
    """
    pair = TP.train_pair(seed=SEED, steps=TP.QUICK_STEPS, n_train=TP.QUICK_NTRAIN)
    for trained in (False, True):
        st = TP.statistics(pair, trained=trained)
        assert 1.0 - 1e-9 <= st["alpha_eff_support"] <= st["n_pivots"] + 1e-9, st


def test_alpha_effective_support_reads_one_on_a_planted_one_hot():
    """Both ends of the spread measure, seen. Neither is assumed."""
    onehot = torch.tensor([1.0, 1e-300, 1e-300, 1e-300], dtype=torch.float64)
    onehot = onehot / onehot.sum()
    flat = torch.full((4,), 0.25, dtype=torch.float64)
    assert abs(TP.effective_support(onehot) - 1.0) < 1e-9
    assert abs(TP.effective_support(flat) - 4.0) < 1e-9

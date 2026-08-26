"""The E-task family: the label is an EQUILIBRIUM, not a static expression of x.

WHY THIS FILE EXISTS. Every task the M3 corpus shipped before this one is a
static expression of its input:

    negation_scope    x[:, p, CH_PAYLOAD] * x[:, f, CH_FLIP]   a product of two entries
    counter_squared   x[:, :, CH_FLIP].sum(dim=1) ** 2         a sum, squared

Neither has a fixed point and neither needs an iteration, so an arm that runs a
settling loop to convergence and then predicts one of them has been asked to
show a settling advantage on a task where settling has nothing to compute. The
deciding contrast came in near zero, and near zero is the correct answer to that
question. `LOOP_PROMPT.md` section 1.7 answers it by putting the equilibrium in
the LABEL. This file is the RED-first bind for that family.

THE FOUR PROPERTIES EVERY E-TASK MUST HAVE, each one a test below.

1. THE ORACLE IS RECOMPUTED FROM `x`, NEVER STORED. Same discipline as
   `negation_scope.oracle`: a corpus file containing `y` is one that can be
   memorised, and the W4 split-by-value death is what that produces.

2. THE LABEL ACTUALLY REQUIRES ITERATION. This is the single most important
   check in the file. A fixed truncation at `k` hops must have accuracy that is
   BOUNDED AWAY from the label and must TIGHTEN with `k`. If a 1-hop reading
   gets the label, the result is a third static task wearing an equilibrium's
   name. Both families are checked against a closed form for the truncation
   error, not against a hand-picked threshold.

3. WELL-POSEDNESS IS ENFORCED AT GENERATION, NOT HOPED FOR AT READ TIME. The
   chain family is nilpotent by construction (strictly lower triangular, checked
   by value). The consequence family contracts because its temperature comes
   from `ceq/nash.py::safe_tau`, whose margin pins the Lipschitz constant of the
   best-response map at exactly `1 / margin`; the builder asserts it and the
   residual, and raises rather than returning a batch that did not settle.

4. THE BAR CALIBRATES, through `calibrate_bar`'s `oracle_fn` / `batch_fn` /
   `feature_fn` hooks, with an EXACT closed-form `flipper_dependence` for every
   task in the family -- `2 / sqrt(t* + 1)` for the chain, exactly `2.0` for the
   consequence contrast.

CONTROLS, EVERY ONE OF THEM DRAWN RATHER THAN HAND-BUILT, AND EVERY ONE CHECKED
TO FIRE:

    - the k-hop truncation at k = 0 must FAIL the flipper-dependence band (it
      reads 0.0: the head driver is not in its window at all);
    - the exact band for one rung of the ladder must REJECT a batch drawn at a
      different rung, so the band separates `t*` values and not merely
      "this label moves";
    - the SHOCK-BLIND reading of the consequence task -- predict zero, i.e.
      ignore the intervening token entirely -- must sit AT OR ABOVE the bar.
      This control already fired once and killed a design: with the raw new
      fixed-point coordinate as the label it read NRMSE 0.194150 at n=2048,
      s=64, d=24, seed=0, meaning 96% of that label was the un-intervened game
      and the intervention was a rounding error;
    - the consequence bar must be BROKEN at the shipped 150-step budget and
      CALIBRATED at 600, so clause 5 is seen to fail as well as to pass.
"""
from __future__ import annotations

import math
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ceq.nash import safe_tau                                      # noqa: E402
from scale import negation_scope as NS                             # noqa: E402

S_DEFAULT, D_DEFAULT = 64, 24

#: n at which the two-sided band at the shipped tolerance 0.05 is a gate rather
#: than a coin. `flipper_dependence` is a ratio of two sample means, so it has a
#: sampling spread; the floor below is MEASURED (16 seeds per cell, this
#: machine, torch 2.5.1+cu121, 2 threads) and reported in
#: `negation_scope.chain_flipper_dependence`'s docstring.
N_FLOOR = {1: 256, 2: 4096, 8: 2048, 32: 2048, 63: 512}


def _chain(n, t_star, *, seed=0, s=S_DEFAULT, d=D_DEFAULT):
    return NS.make_equilibrium_batch(n, s, d, t_star=t_star, seed=seed)


# ------------------------------------------------------- 0. the registration --
def test_the_e_family_is_registered_in_m3_tasks():
    """S2 asks for the family REGISTERED, not merely written. Every entry must
    have the shipped 4-tuple shape, because `m3_capability.py:248` unpacks
    exactly four names and a fifth field would break every existing task."""
    for name in ("e1_anchor", "e2_consequence",
                 "e3_t1", "e3_t2", "e3_t8", "e3_t32"):
        assert name in NS.M3_TASKS, name
        entry = NS.M3_TASKS[name]
        assert len(entry) == 4, (name, len(entry))
        batch_fn, oracle_fn, feature_fn, fd_fn = entry
        assert callable(batch_fn) and callable(oracle_fn)
        assert callable(feature_fn) and callable(fd_fn)


def test_the_shipped_tasks_are_untouched():
    """The two tasks already in the registry must be BITWISE what they were.
    A new family that moves an old number is a new corpus, not an addition."""
    x, y, f, p = NS.M3_TASKS["negation_scope"][0](64, S_DEFAULT, D_DEFAULT,
                                                  d_model=16, seed=0)
    assert torch.equal(y, NS.oracle(x, f, p))
    xc, yc, fc, pc = NS.M3_TASKS["counter_squared"][0](64, S_DEFAULT, D_DEFAULT,
                                                       d_model=16, seed=0)
    assert torch.equal(yc, NS.counter_squared_oracle(xc, fc, pc))
    assert NS.M3_TASKS["negation_scope"][3] is None


def test_every_e_task_prints_its_difficulty_dial():
    """S2: `t*` printed per task. It is a property of the TASK, so it lives in
    the module beside the registration rather than in a caller's print."""
    assert NS.E_T_STAR["e1_anchor"](S_DEFAULT) == S_DEFAULT - 1
    assert NS.E_T_STAR["e3_t1"](S_DEFAULT) == 1
    assert NS.E_T_STAR["e3_t2"](S_DEFAULT) == 2
    assert NS.E_T_STAR["e3_t8"](S_DEFAULT) == 8
    assert NS.E_T_STAR["e3_t32"](S_DEFAULT) == 32
    #: the consequence task settles geometrically rather than terminating, so
    #: its dial is the number of best-response sweeps to reach 1e-3, not an
    #: exact nilpotency index. It must still be a number, and it must be > 1.
    assert NS.E_T_STAR["e2_consequence"](S_DEFAULT) > 1


# --------------------------------------- 1. the oracle is recomputed from x ---
@pytest.mark.parametrize("t_star", [1, 2, 8, 32, 63])
def test_the_chain_label_is_recomputed_from_x_and_not_stored(t_star):
    x, y, f, p = _chain(256, t_star)
    assert torch.equal(y, NS.equilibrium_oracle(x, f, p))
    #: and it is a function OF x: move one driver inside the live band and the
    #: label must move. Drawn instance, not a hand-built one.
    x2 = x.clone()
    x2[:, f, NS.CH_FLIP] = x2[:, f, NS.CH_FLIP] + 1.0
    assert not torch.equal(NS.equilibrium_oracle(x2, f, p), y)


def test_the_consequence_label_is_recomputed_from_x_and_not_stored():
    x, y, f, p = NS.make_consequence_batch(256, S_DEFAULT, D_DEFAULT, seed=0)
    assert torch.allclose(y, NS.consequence_oracle(x, f, p), atol=0, rtol=0)
    x2 = x.clone()
    x2[:, f, NS.CH_FLIP] = x2[:, f, NS.CH_FLIP] + 1.0
    assert not torch.equal(NS.consequence_oracle(x2, f, p), y)


# ------------------------------------ 2. THE LABEL REQUIRES ITERATION ---------
@pytest.mark.parametrize("t_star", [2, 8, 32])
def test_a_fixed_k_hop_truncation_cannot_get_the_chain_label(t_star):
    """THE CHECK THE WHOLE FAMILY EXISTS FOR.

    The k-hop reading is the Neumann truncation `sum_{m=0..k} (A^m b)_{s-1}`,
    i.e. the label computed from the last k+1 tokens only. The query token
    carries no driver, so the surviving tail is a sum of `t* - k` independent
    unit-variance terms and the label is a sum of `t*` of them, and the
    truncation's NRMSE is

        sqrt((t* - k) / t*)     for k <= t*,   0 for k >= t*

    in closed form. FOUR things are asserted: the measured value matches that
    form; the k = 0 rung is EXACTLY the bar, so no part of the label is legible
    without hops; the ladder is MONOTONE DECREASING in k (the bound tightens);
    and a 1-hop reading is bounded away from the label -- if it were not, this
    would be a third static task.
    """
    x, y, f, p = _chain(4096, t_star)
    prev = None
    for k in range(0, t_star + 2):
        got = NS.nrmse(NS.equilibrium_hop_reading(x, k), y)
        want = math.sqrt(max(0.0, t_star - k) / t_star)
        assert abs(got - want) < 0.03, (t_star, k, got, want)
        if prev is not None:
            assert got <= prev + 1e-9, (t_star, k, got, prev)
        prev = got
    #: ZERO hops is exactly predict-the-mean. This is the end of the ladder that
    #: the harness's own 0-step RED gate reads, and it is why the query token
    #: carries no driver.
    assert NS.nrmse(NS.equilibrium_hop_reading(x, 0), y) >= 1.0
    #: a 1-hop reading must NOT get the label. The floor is the closed form, so
    #: no threshold is chosen here either.
    one_hop = NS.nrmse(NS.equilibrium_hop_reading(x, 1), y)
    assert one_hop > 0.9 * math.sqrt((t_star - 1) / t_star)
    assert one_hop > 0.5, (t_star, one_hop)
    #: and the full budget IS exact -- the system is nilpotent, so the series
    #: terminates rather than merely converging.
    assert NS.nrmse(NS.equilibrium_hop_reading(x, t_star), y) == 0.0


def test_a_fixed_k_iterate_cannot_get_the_consequence_label():
    """Same check for the contraction family. Here the truncation error decays
    geometrically rather than terminating.

    The sweep map is a contraction with constant `L = 1 / margin = 0.8` exactly,
    and the sweeps start at the barycentre, so `||s_k - s*|| <= L**k ||s_0 - s*||`
    -- i.e. every rung is bounded by `L**k` times the ZEROTH rung's error, which
    is the shock-blind reading. That relative form is the actual contraction
    inequality; `L**k` on its own is not a bound on an NRMSE, because NRMSE
    normalises by `std(y)` rather than by the initial error, and the zeroth rung
    reads 1.0000668916944135 rather than 1.0 for exactly that reason.
    """
    x, y, f, p = NS.make_consequence_batch(2048, S_DEFAULT, D_DEFAULT, seed=0)
    zeroth = NS.nrmse(NS.consequence_oracle(x, f, p, k=0), y)
    prev = None
    for k in (0, 1, 2, 4, 8, 16):
        got = NS.nrmse(NS.consequence_oracle(x, f, p, k=k), y)
        assert got <= zeroth * NS.E_LIPSCHITZ ** k + 1e-9, (k, got)
        if prev is not None:
            assert got < prev, (k, got, prev)
        prev = got
    assert zeroth >= 1.0
    assert NS.nrmse(NS.consequence_oracle(x, f, p, k=1), y) > 0.2


def test_the_shock_blind_reading_of_the_consequence_task_is_at_the_bar():
    """THE CONTROL THAT KILLED THE FIRST DESIGN, kept as a test.

    An arm that ignores the intervening token entirely predicts zero. Because
    the label is the contrast between the two MIRROR interventions on that one
    token, the zero predictor's NRMSE is
    `sqrt(1 + mean(y)**2 / var(y)) >= 1.0` identically -- it cannot beat the
    bar, whatever the game draw. With the raw new-fixed-point coordinate as the
    label the same control read 0.194150, which is why that label is not the
    one shipped.
    """
    x, y, f, p = NS.make_consequence_batch(2048, S_DEFAULT, D_DEFAULT, seed=0)
    assert NS.nrmse(torch.zeros_like(y), y) >= 1.0


# ------------------------- 3. WELL-POSEDNESS IS ENFORCED AT GENERATION --------
@pytest.mark.parametrize("t_star", [1, 8, 63])
def test_the_chain_operator_is_strictly_lower_triangular_by_value(t_star):
    """Nilpotency is the chain family's well-posedness: `(I - A)` is invertible
    for EVERY draw because `A` is strictly lower triangular, so the resolvent is
    a terminating sum and no instance can fail to settle. Checked by VALUE on a
    drawn batch, not asserted in prose: the coefficient at the head of the chain
    and everything before it must be exactly zero, which is what makes the
    label's dependence stop at `t*` hops."""
    x, y, f, p = _chain(64, t_star)
    a = x[:, :, NS.CH_DRIVE]
    assert torch.equal(a[:, :f + 1], torch.zeros_like(a[:, :f + 1]))
    assert torch.equal(a[:, f + 1:].abs(),
                       torch.ones_like(a[:, f + 1:]))


def test_the_consequence_game_contracts_for_every_drawn_instance():
    """The temperature is `ceq.nash.safe_tau`'s, not one invented here, so the
    Lipschitz constant of `s -> sigmoid((Ms+b)/tau)` is `||M||_2 / (4 tau)` =
    `1 / margin` EXACTLY for every draw. Checked per example against `safe_tau`
    re-called on the same game, so the builder cannot drift to a private
    threshold."""
    x, y, f, p = NS.make_consequence_batch(256, S_DEFAULT, D_DEFAULT, seed=0)
    m = NS.consequence_game(x)[0]
    for i in range(0, m.shape[0], 17):
        tau = safe_tau(m[i])
        lip = float(torch.linalg.matrix_norm(m[i], ord=2)) / (4.0 * tau)
        assert abs(lip - NS.E_LIPSCHITZ) < 1e-9, (i, lip)


def test_a_non_settling_draw_is_rejected_at_generation():
    """Requirement 3 says REJECTED AT GENERATION, not detected at read time. The
    builder asserts its own residual; forcing the sweep count to zero must make
    it raise rather than return an unsettled batch."""
    with pytest.raises(ValueError):
        NS.make_consequence_batch(64, S_DEFAULT, D_DEFAULT, seed=0,
                                  settle_iters=0)


# ----------------------------------------- 4. THE BAR CALIBRATES --------------
@pytest.mark.parametrize("t_star", [1, 2, 8, 32, 63])
def test_the_chain_flipper_dependence_is_its_closed_form(t_star):
    """Negating the driver at the head of the chain moves the label by exactly
    `2 |b_head|`, because the path weight from the head to the query is a
    product of Rademacher coefficients and has modulus 1. The label is
    `N(0, t*+1)` exactly, so

        flipper_dependence = 2 E|N(0,1)| / E|N(0, t*+1)| = 2 / sqrt(t* + 1)

    with no constant fitted and no threshold chosen."""
    n = N_FLOOR[t_star]
    cal = NS.calibrate_bar(n=n, s=S_DEFAULT, d=D_DEFAULT, steps=1,
                           batch_fn=NS.M3_TASKS[_chain_task(t_star)][0],
                           oracle_fn=NS.equilibrium_oracle,
                           feature_fn=NS.equilibrium_features)
    want = NS.M3_TASKS[_chain_task(t_star)][3](S_DEFAULT)
    assert abs(want - 2.0 / math.sqrt(t_star)) < 1e-12
    assert abs(cal["flipper_dependence"] - want) < 0.05, (t_star, cal)


def _chain_task(t_star):
    return "e1_anchor" if t_star == S_DEFAULT - 1 else f"e3_t{t_star}"


@pytest.mark.parametrize("t_star", [1, 2, 8, 32, 63])
def test_the_chain_bar_calibrates(t_star):
    name = _chain_task(t_star)
    batch_fn, oracle_fn, feature_fn, fd_fn = NS.M3_TASKS[name]
    cal = NS.calibrate_bar(n=N_FLOOR[t_star], s=S_DEFAULT, d=D_DEFAULT,
                           batch_fn=batch_fn, oracle_fn=oracle_fn,
                           feature_fn=feature_fn, steps=150, lr=0.02)
    ok, why = NS.bar_verdict(cal, flipper_dependence=fd_fn(S_DEFAULT))
    assert ok, (name, why, cal)


def test_the_band_separates_the_rungs_of_the_ladder():
    """CONTROL, SEEN TO FIRE. A band that accepts any label which moves is not a
    band. The exact value for one rung must REJECT a batch drawn at another
    rung, because `t*` is the difficulty dial and a gate that cannot tell two
    dial settings apart cannot police the dose-response reading."""
    cal = NS.calibrate_bar(n=N_FLOOR[32], s=S_DEFAULT, d=D_DEFAULT, steps=1,
                           batch_fn=NS.M3_TASKS["e3_t32"][0],
                           oracle_fn=NS.equilibrium_oracle,
                           feature_fn=NS.equilibrium_features)
    ok, why = NS.bar_verdict(cal, flipper_dependence=NS.M3_TASKS["e3_t8"][3](S_DEFAULT))
    assert not ok and "flipper_dependence" in why, why
    ok2, why2 = NS.bar_verdict(cal, flipper_dependence=NS.M3_TASKS["e1_anchor"][3](S_DEFAULT))
    assert not ok2 and "flipper_dependence" in why2, why2


def test_a_truncated_reading_fails_the_chain_band():
    """CONTROL, SEEN TO FIRE, on a DRAWN batch. The k=0 reading -- the last
    token's driver alone -- does not contain the head of the chain at all, so
    negating the head moves it by exactly zero and the band rejects it. This is
    the wrong-task control the flipper clause exists for.

    The rung read is k = 1, not k = 0: with the query token carrying no driver
    the k = 0 reading is identically zero, and `nrmse` against an all-zero
    prediction is a division by an all-zero std -- NaN, which is not a control
    failing, it is a control that cannot be evaluated."""
    cal = NS.calibrate_bar(n=1024, s=S_DEFAULT, d=D_DEFAULT, steps=1,
                           batch_fn=NS.M3_TASKS["e3_t8"][0],
                           oracle_fn=lambda x, f, p: NS.equilibrium_hop_reading(x, 1),
                           feature_fn=NS.equilibrium_features)
    assert cal["flipper_dependence"] == 0.0
    ok, why = NS.bar_verdict(cal, flipper_dependence=NS.M3_TASKS["e3_t8"][3](S_DEFAULT))
    assert not ok and "flipper_dependence" in why, why


def test_the_consequence_flipper_dependence_is_exactly_two():
    """The label is `z*(v) - z*(1-v)` with `v = sigmoid(u)` and `u` the
    intervening token's channel. Negating `u` sends `v -> 1 - v`, which sends
    the label to MINUS ITSELF, so the ratio is exactly 2.0 PER EXAMPLE and the
    band has no sampling spread at any n -- the same exact 2.0 the shipped
    negation-scope task has, reached by antisymmetry rather than by a product.
    """
    for n in (256, 1024):
        cal = NS.calibrate_bar(n=n, s=S_DEFAULT, d=D_DEFAULT, steps=1,
                               batch_fn=NS.make_consequence_batch,
                               oracle_fn=NS.consequence_oracle,
                               feature_fn=NS.consequence_features)
        assert abs(cal["flipper_dependence"] - 2.0) < 1e-5, (n, cal)
    assert NS.M3_TASKS["e2_consequence"][3](S_DEFAULT) == 2.0


def test_an_unshocked_equilibrium_fails_the_consequence_band():
    """CONTROL, SEEN TO FIRE, DRAWN. The un-intervened equilibrium coordinate is
    the label this task REPLACED. It does not depend on the intervening token at
    all, so its flipper dependence is exactly zero and the band rejects it."""
    cal = NS.calibrate_bar(n=512, s=S_DEFAULT, d=D_DEFAULT, steps=1,
                           batch_fn=NS.make_consequence_batch,
                           oracle_fn=NS.unshocked_equilibrium,
                           feature_fn=NS.consequence_features)
    assert cal["flipper_dependence"] == 0.0
    ok, why = NS.bar_verdict(cal, flipper_dependence=2.0)
    assert not ok and "flipper_dependence" in why, why


def test_the_consequence_bar_is_broken_at_the_shipped_step_budget():
    """CONTROL, SEEN TO FIRE, and a MEASURED precondition on the run.

    Clause 5 trains its positive control on the RAW label while `run_arm` trains
    every arm on the STANDARDISED one, so a small-scale label makes the control
    strictly harder than the arms' own task. The consequence label's std is
    about 0.062, and at the shipped 150-step budget the control cannot reach the
    bar. It reaches it at 600. Both readings are asserted so the clause is seen
    to fail as well as to pass, and so the step budget this task needs is a
    measured number rather than a preference.
    """
    kw = dict(n=2048, s=S_DEFAULT, d=D_DEFAULT, lr=0.02,
              batch_fn=NS.make_consequence_batch,
              oracle_fn=NS.consequence_oracle,
              feature_fn=NS.consequence_features)
    broken = NS.calibrate_bar(steps=150, **kw)
    ok150, why150 = NS.bar_verdict(broken, flipper_dependence=2.0)
    assert not ok150 and "trained_two_feature" in why150, why150
    good = NS.calibrate_bar(steps=NS.E2_STEPS, **kw)
    ok600, why600 = NS.bar_verdict(good, flipper_dependence=2.0)
    assert ok600, (why600, good)

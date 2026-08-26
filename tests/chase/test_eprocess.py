"""Binds on the anytime-valid e-process for the M3 settled-vs-twin contrast.

WHAT IS BEING PINNED. Three separable things, and the file keeps them apart on
purpose:

  1. THE CONSTRUCTION. Nonnegativity, the supermartingale property under the
     null, and the immutable constants (`B`, the threshold, the lambda grid).
     These are arithmetic; they are checked in closed form where a closed form
     exists, so that a passing test is not a restatement of the code.

     `B` is now the OUTCOME CLIP `C`, applied to each arm before differencing.
     The clip is a bound from the definition rather than an assumption about
     data, and the tests pin BOTH halves of that trade: that the bound holds on
     drawn inputs the raw metric would have blown through, and that the clip
     moves the estimand, so a crossing is a statement about clipped NRMSE.

  2. THE MUST-FIRE, BOTH DIRECTIONS. A null stream must not cross the threshold
     more often than the nominal alpha, and a stream with a planted effect must
     cross. A control that has never been seen failing to fire measures nothing,
     and a control that cannot fire at all measures less. Both are checked here
     at a reduced replay count; `scale/eprocess.py` run as a script does the
     full 10,000-replay table.

  3. THE STRUCTURAL CEILING. With `lambda <= 1/2` the largest value the process
     can reach in `t` steps is `1.5 ** t`, whatever the data say and whatever
     `B` is -- the largest legal increment is `d = B`, and `1 + lambda*B/B` does
     not depend on `B`. At the pre-registered five seeds that is `7.59375`,
     which is below the family-wise threshold `40`. The five-seed contrast
     therefore cannot be decided by this process in either direction. That is a
     fact about the design, it was found before any real seed landed, and it is
     pinned here so it cannot be rediscovered later as a surprise.

NO THREAD DEPENDENCE. Nothing here calls BLAS through a shape that varies with
thread count; `scale/eprocess.py` pins `torch.set_num_threads(2)` at import so
that the module, not the launcher, is the authority.
"""
from __future__ import annotations

import math
import pathlib
import random
import re
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scale import eprocess as EP  # noqa: E402

#: Replays used by the in-test must-fire. The full 10,000-replay table is the
#: script path (`python scale/eprocess.py`); this is the cheap resident copy,
#: and its threshold checks are the same ones.
N_REP_TEST = 2000
HORIZON_TEST = 400

#: The planted-effect stream needs a horizon longer than its own crossing time
#: or the must-fire would read "did not cross" as a property of the instrument
#: rather than of the horizon. Clipping the outcome at `C = 2.0` doubled `B`,
#: which halves `d/B` and so roughly doubles the crossing time at a fixed
#: effect; `EP.seeds_needed(EP.EFFECT_FLOOR)["mixture"]` is the bound this
#: horizon has to clear.
HORIZON_PLANTED = 1200


@pytest.fixture(scope="module")
def null_calibration():
    return EP.calibrate(EP.NULL_RADEMACHER, n_rep=N_REP_TEST,
                        horizon=HORIZON_TEST, seed=11)


@pytest.fixture(scope="module")
def planted_calibration():
    return EP.calibrate(EP.PLANTED_FLOOR, n_rep=N_REP_TEST,
                        horizon=HORIZON_PLANTED, seed=12)


@pytest.fixture(scope="module")
def broken_calibration():
    """The same null through a construction whose `lambda` PEEKS at `d_i`.

    Predictability is the property that makes the supermartingale argument
    work. Deleting it must be seen to break the null, or the null check has no
    teeth.
    """
    return EP.calibrate(EP.NULL_RADEMACHER, n_rep=N_REP_TEST,
                        horizon=HORIZON_TEST, seed=13, peek=True)


# ------------------------------------------------------- the constants ------
def test_bound_is_the_outcome_clip_and_carries_its_provenance():
    """`B` is the width of the CLIPPED outcome range, not a range read off data.

    `negation_scope.nrmse` (`scale/negation_scope.py:96-101`) is
    `RMSE / std(y)`, a ratio of nonnegative quantities, so it is bounded below
    by 0 and equals exactly 1.0 for the mean predictor. It is NOT bounded above:
    `results/*.jsonl` holds 21 readings at or above 1.0, the largest
    `1.0742670875495859` at
    `results/m3_quintuple.jsonl::softmax_k0_s64_d24_st20_ntr256_nev128_b21_sd0`.
    The earlier `B = 1.0` was therefore an assumption about which cells the
    journal reader would be pointed at -- the credit bar of
    `M3_QUINTUPLE_PREREGISTERED_READING.md` section 5 -- and not a bound from
    the definition of the metric.

    The bound is now supplied by clipping each ARM at `C` inside the definition
    of the difference. `min(x, C)` lies in `[0, C]` for every `x >= 0`, so the
    difference of two clipped arms lies in `[-C, +C]` unconditionally.
    """
    assert EP.CLIP_C == 2.0
    assert EP.NRMSE_FLOOR == 0.0
    assert EP.B == 2.0
    assert EP.B == EP.CLIP_C - EP.NRMSE_FLOOR


def test_alpha_budget_is_family_wise_and_the_threshold_pays_for_it():
    """Two directions are read, so the reported event is the UNION of two.

    Ville gives `P(sup_t E_t >= tau) <= 1/tau` per process. Two one-sided
    processes are run and EITHER may be reported, because the winning direction
    is not fixed in advance; a union bound over the two gives
    `P(either crosses) <= 2/tau`. A family-wise 0.05 therefore needs
    `2/tau <= 0.05`, i.e. `tau >= 40`, not `tau = 20`.
    """
    assert EP.ALPHA_FAMILY == 0.05
    assert EP.N_DIRECTIONS == 2
    assert EP.ALPHA == 0.025
    assert EP.ALPHA == EP.ALPHA_FAMILY / EP.N_DIRECTIONS
    assert EP.THRESHOLD == 40.0
    assert EP.THRESHOLD == 1.0 / EP.ALPHA
    assert EP.N_DIRECTIONS / EP.THRESHOLD == EP.ALPHA_FAMILY


# --------------------------------------------------- the clip in the middle --
def test_the_clip_is_applied_per_arm_before_differencing():
    """`d = min(ref, C) - min(arm, C)`, not `clip(ref - arm)`.

    Inside the clip the transform is the identity, so nothing that could have
    been credited before is moved.
    """
    assert EP.paired_difference(0.5, 100.5) == pytest.approx(0.5 - 2.0)
    assert EP.paired_difference(100.5, 0.5) == pytest.approx(2.0 - 0.5)
    assert EP.paired_difference(0.9, 0.7) == pytest.approx(0.9 - 0.7)
    assert EP.clipped_nrmse(0.9) == 0.9
    assert EP.clipped_nrmse(5.8198) == EP.CLIP_C


def test_a_negative_or_non_finite_nrmse_is_refused():
    """`nrmse` is a ratio of nonnegatives; a negative reading is a broken
    journal, not a small effect, and `nan` is what the zero-variance branch
    returns (`scale/negation_scope.py:99-100`)."""
    for bad in (-1e-12, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            EP.clipped_nrmse(bad)


def test_clipped_difference_is_inside_the_bound_on_drawn_inputs():
    """DRAWN, not hand-picked, and the clip is seen to do work.

    The draws are exponential with mean 4.0, so a large fraction of arms land
    above `C = 2.0` and the RAW difference leaves `[-B, B]` often. If the clip
    were absent, `Eprocess.update` would raise on those; with it, every drawn
    pair is legal. Both halves are asserted, so the control cannot pass by
    never exercising the clip.
    """
    rng = random.Random(20260826)
    n, n_clipped, n_raw_out_of_bound = 20000, 0, 0
    for _ in range(n):
        a, b = rng.expovariate(0.25), rng.expovariate(0.25)
        if a > EP.CLIP_C or b > EP.CLIP_C:
            n_clipped += 1
        if abs(a - b) > EP.B:
            n_raw_out_of_bound += 1
        d = EP.paired_difference(a, b)
        assert abs(d) <= EP.B
        EP.Eprocess().update(d)          # must never raise
    assert n_clipped > n // 2, n_clipped
    assert n_raw_out_of_bound > n // 10, n_raw_out_of_bound


def test_the_clip_moves_the_estimand_and_that_is_pinned_not_hidden():
    """Neither clip preserves the sign of the RAW mean difference.

    The post-hoc clip of the difference was refuted with `d = -100` w.p. `0.01`
    and `+0.5` w.p. `0.99`: `E[d] = -0.505` while the clipped difference has
    mean `+0.475`. The adopted per-arm clip is strictly better in one respect --
    it is a fixed monotone transform of each OUTCOME, fixed before the data, so
    it does not depend on the pairing and cannot be tuned after the fact -- but
    it is NOT sign-preserving either: the same instance gives `+0.48`.

    What the process therefore tests is
    `H0: E[min(NRMSE_ref, C)] <= E[min(NRMSE_arm, C)]`,
    a hypothesis about the CLIPPED outcome. Pinned here so that a crossing can
    never be reported later as a statement about raw NRMSE.
    """
    p = 0.01
    raw = p * (-100.0) + (1.0 - p) * 0.5
    post_hoc = p * max(-EP.B, min(EP.B, -100.0)) + (1.0 - p) * 0.5
    per_arm = (p * EP.paired_difference(0.5, 100.5)
               + (1.0 - p) * EP.paired_difference(0.5, 0.0))
    assert raw == pytest.approx(-0.505)
    assert post_hoc == pytest.approx(0.475)
    assert per_arm == pytest.approx(0.48)
    assert raw < 0.0 < post_hoc
    assert raw < 0.0 < per_arm


def test_the_clip_is_inert_on_the_reading_path_and_active_only_above_the_bar():
    """Where the clip bites, scanned rather than remembered.

    Every `eval_nrmse` in every `results/**/*.jsonl` is read here, because
    `read_paired` consumes bucket journals and nothing else. The largest is
    `1.0742670875495859` at
    `results/m3_quintuple.jsonl::softmax_k0_s64_d24_st20_ntr256_nev128_b21_sd0`,
    below `C = 2.0`, so on the reading path the clip is the identity.

    The same scan refutes the old `B = 1.0` derivation: 21 readings sit at or
    above `1.0`, so "a credited NRMSE lies in [0, 1)" was a statement about
    which cells get read, not about the metric.

    The metric DOES exceed `C` elsewhere in the repo -- ten readings in
    `results/m3_capability.txt`, largest `3.696671` at line 589 -- and that is
    asserted too, so the claim "inert" can never widen into "the metric never
    reaches C". Every one of those ten is an `n_train=128` cell above the
    section-5 credit bar, which is why `C = 2.0` truncates nothing that could
    have been credited.
    """
    import json
    root = pathlib.Path(__file__).resolve().parents[2] / "results"
    seen = []
    for p in sorted(root.rglob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            v = json.loads(line).get("value")
            if isinstance(v, dict) and "eval_nrmse" in v:
                seen.append(float(v["eval_nrmse"]))
    # The journals are LIVE -- the M3 run appends while this suite runs -- so
    # the counts are asserted as floors, not equalities. At the reading that
    # struck the old derivation they were exactly 59 and 21.
    assert len(seen) >= 59, len(seen)
    assert sum(1 for x in seen if x >= 1.0) >= 21      # old B = 1.0 was an assumption
    assert max(seen) >= 1.0742670875495859
    assert max(seen) < EP.CLIP_C                       # new B = C is inert here
    for x in seen:
        assert EP.clipped_nrmse(x) == x

    # ... and the other half: the metric is NOT bounded by C repo-wide.
    cap = root / "m3_capability.txt"
    over = [float(m.group(1)) for m in re.finditer(
        r"eval NRMSE=(\d+\.\d+)", cap.read_text(encoding="utf-8",
                                                errors="replace"))
        if float(m.group(1)) >= EP.CLIP_C]
    assert len(over) >= 10, len(over)
    assert max(over) >= 3.696671
    assert EP.clipped_nrmse(max(over)) == EP.CLIP_C     # the clip IS active there


def test_lambda_grid_inside_the_declared_interval():
    assert len(EP.LAMBDA_GRID) == 10
    assert min(EP.LAMBDA_GRID) > 0.0
    assert max(EP.LAMBDA_GRID) == 0.5
    for lam in EP.LAMBDA_GRID:
        assert 0.0 <= lam <= 0.5


# ------------------------------------------------------- the construction ---
def test_factor_stays_positive_at_the_bound():
    """|d| = B with the largest lambda is the worst case for nonnegativity."""
    for lam in EP.LAMBDA_GRID:
        assert 1.0 + lam * (-EP.B) / EP.B >= 0.5


def test_e0_is_one():
    assert EP.Eprocess().value == 1.0


def test_out_of_range_difference_is_refused_not_clamped():
    """A `d` outside the a-priori bound breaks Ville, so it must raise.

    Clamping would keep the process running while the guarantee it prints has
    already been voided.
    """
    e = EP.Eprocess()
    with pytest.raises(ValueError):
        e.update(2.5)
    with pytest.raises(ValueError):
        e.update(-2.5)


def test_supermartingale_property_in_closed_form():
    """E[E_1] == 1 exactly for a mean-zero d, for every grid lambda.

    E[1 + lam*d/B] = 1 + lam*E[d]/B, so a mean-zero increment leaves the
    expectation at 1 and a negative-mean increment (also H0) leaves it below 1.
    Checked on the two-point +-B null, where the expectation is a two-term sum.
    """
    for lam in EP.LAMBDA_GRID:
        up = 1.0 + lam * EP.B / EP.B
        dn = 1.0 + lam * (-EP.B) / EP.B
        assert abs(0.5 * up + 0.5 * dn - 1.0) < 1e-15
    # and the mixture, which is a convex combination of the same factors
    mix = 0.0
    for lam in EP.LAMBDA_GRID:
        mix += (0.5 * (1.0 + lam) + 0.5 * (1.0 - lam)) / len(EP.LAMBDA_GRID)
    assert abs(mix - 1.0) < 1e-15


def test_running_maximum_is_what_ville_bounds():
    """The decision reads `sup_t E_t`, not `E_T`. A process that crossed and
    then fell back has still crossed."""
    e = EP.Eprocess()
    for _ in range(30):
        e.update(EP.B)
    peak = e.peak
    assert peak >= EP.THRESHOLD
    assert e.crossed
    for _ in range(30):
        e.update(-EP.B)
    assert e.value < peak
    assert e.peak == peak
    assert e.crossed


def test_symmetric_process_reads_the_other_direction():
    """The twin direction is the same process on `-d`."""
    pair = EP.Pair()
    for _ in range(30):
        pair.update(-EP.B)
    assert pair.twin.crossed
    assert not pair.settled.crossed
    assert pair.decision == "twin"


# ------------------------------------------------------- the ceiling --------
def test_five_seeds_cannot_cross_in_either_direction():
    """The pre-registered five-seed contrast is undecidable by construction.

    Not a property of any data. Both ceilings are below the threshold: the
    loose single-arm one (`1.5 ** 5`) and the mixture's own, which is what the
    shipped instrument can actually reach.
    """
    assert EP.max_attainable_single_arm(5) == 1.5 ** 5
    assert EP.max_attainable_single_arm(5) == pytest.approx(7.59375, abs=1e-12)
    assert EP.max_attainable(5) < EP.max_attainable_single_arm(5)
    assert EP.max_attainable(5) < EP.THRESHOLD
    e = EP.Eprocess()
    for _ in range(5):
        e.update(EP.B)          # the largest legal increment, five times
    assert e.peak < EP.THRESHOLD
    assert e.peak == pytest.approx(EP.max_attainable(5), rel=1e-12)


def test_minimum_horizons_are_ten_and_thirteen():
    """Ten for the loose single-arm bound, thirteen for the shipped mixture.

    These moved from 8 and 11 when the threshold moved from 20 to 40. The
    five-seed kill gets STRONGER, not weaker: the gap between what five seeds
    can reach (`3.801691`) and what the decision needs (`40`) widened.
    """
    assert EP.MIN_T_SINGLE_ARM == 10
    assert math.ceil(math.log(EP.THRESHOLD) / math.log(1.5)) == 10
    assert 1.5 ** 9 < EP.THRESHOLD <= 1.5 ** 10
    assert EP.MIN_T_MIXTURE == 13
    assert EP.max_attainable(12) < EP.THRESHOLD <= EP.max_attainable(13)


def test_the_ceiling_does_not_depend_on_the_bound():
    """Doubling `B` did not move the ceiling, and that is why it is a ceiling.

    The largest legal increment is `d = B`, and `1 + lam*B/B = 1 + lam`
    whatever `B` is. So the five-seed kill survives the change of `B` for a
    reason, not by luck.
    """
    for b in (0.25, 1.0, EP.B, 7.0):
        e = EP.Eprocess(b=b)
        for _ in range(5):
            e.update(b)
        assert e.peak == pytest.approx(EP.max_attainable(5), rel=1e-12)
        assert e.peak < EP.THRESHOLD


def test_a_horizon_that_cannot_cross_is_refused():
    """A calibration run below the ceiling would pass while measuring nothing."""
    with pytest.raises(ValueError):
        EP.calibrate(EP.NULL_RADEMACHER, n_rep=10,
                     horizon=EP.MIN_T_MIXTURE - 1)


# ------------------------------------------------------- the must-fire ------
def test_must_fire_null_does_not_cross(null_calibration):
    """DIRECTION 1. The nominal alpha is 0.025 PER DIRECTION, 0.05 family-wise.

    Both are asserted, because the family-wise rate is the one the report is
    allowed to quote and the per-direction rate is the one Ville bounds.
    """
    assert null_calibration["cross_settled"] <= EP.ALPHA
    assert null_calibration["cross_twin"] <= EP.ALPHA
    assert null_calibration["cross_either"] <= EP.ALPHA_FAMILY


def test_must_fire_null_had_a_real_opportunity_to_cross(null_calibration):
    """A null that could not have crossed is not a calibration.

    The horizon must be long enough that the ceiling is above the threshold,
    and the replays must actually reach values a long way up.
    """
    t = null_calibration["horizon"]
    assert EP.max_attainable(t) > EP.THRESHOLD
    assert null_calibration["max_peak"] > 1.0
    # and the replays did move: some reached E_t >= 2 under the null
    assert null_calibration["frac_above_2"] > 0.0


def test_must_fire_planted_effect_crosses(planted_calibration):
    """DIRECTION 2. An instrument that cannot detect a planted effect would
    make a null result meaningless."""
    assert planted_calibration["cross_settled"] > 0.5
    assert planted_calibration["cross_twin"] <= EP.ALPHA


def test_the_planted_horizon_clears_its_own_price(planted_calibration):
    """The planted must-fire must be given more steps than the effect costs.

    Otherwise a "did not cross" would be a property of the horizon and the
    DIRECTION 2 check would be measuring the schedule, not the instrument.
    """
    price = EP.seeds_needed(EP.EFFECT_FLOOR)["mixture"]
    assert planted_calibration["horizon"] > price
    assert planted_calibration["median_cross_t"] is not None
    assert planted_calibration["median_cross_t"] < planted_calibration["horizon"]


def test_broken_bound_is_seen_to_break_the_null(broken_calibration):
    """The null test has teeth: a `B` chosen too small crosses far too often."""
    assert broken_calibration["cross_settled"] > EP.ALPHA


# ------------------------------------------------------- the live reading ---
def test_journal_reader_pairs_by_seed_and_refuses_unpaired(tmp_path):
    j = tmp_path / "j.jsonl"
    rows = [
        '{"key": "settled_k8_s64_d24_st150_ntr8192_nev512_b21_sd0",'
        ' "value": {"eval_nrmse": 0.80}}',
        '{"key": "twin_k8_s64_d24_st150_ntr8192_nev512_b21_sd0",'
        ' "value": {"eval_nrmse": 0.90}}',
        '{"key": "settled_k8_s64_d24_st150_ntr8192_nev512_b21_sd1",'
        ' "value": {"eval_nrmse": 0.70}}',
    ]
    j.write_text("\n".join(rows) + "\n", encoding="utf-8")
    paired = EP.read_paired(j, ref="twin", arm="settled")
    assert [p[0] for p in paired] == [0]           # seed 1 has no twin: dropped
    assert paired[0][1] == pytest.approx(0.90 - 0.80)


def test_journal_reader_clips_each_arm_and_the_process_survives_it(tmp_path):
    """A journal carrying an arm above `C` must read as `-B`, not as a raise.

    This is the failure `B = 1.0` would have produced on the same journal:
    `0.90 - 5.8198` is `-4.9198`, outside any bound, and `update` would have
    voided the run. The per-arm clip turns it into a legal `-1.10`.
    """
    j = tmp_path / "blown.jsonl"
    rows = [
        '{"key": "settled_k8_s64_d24_st150_ntr8192_nev512_b21_sd0",'
        ' "value": {"eval_nrmse": 5.8198}}',
        '{"key": "twin_k8_s64_d24_st150_ntr8192_nev512_b21_sd0",'
        ' "value": {"eval_nrmse": 0.90}}',
    ]
    j.write_text("\n".join(rows) + "\n", encoding="utf-8")
    paired = EP.read_paired(j, ref="twin", arm="settled")
    assert paired == [(0, pytest.approx(0.90 - 2.0))]
    r = EP.live(j)
    assert r["void"] is None
    assert r["t"] == 1


def test_live_reading_on_an_empty_journal_is_t0_not_a_crash(tmp_path):
    j = tmp_path / "empty.jsonl"
    j.write_text("", encoding="utf-8")
    r = EP.live(j)
    assert r["t"] == 0
    assert r["e_settled"] == 1.0
    assert r["e_twin"] == 1.0
    assert r["decision"] is None
    assert "undecided at evidence E_t = 1.0" in r["text"]


# ------------------------------------------- sizing the reroute, not testing --
def test_minimum_detectable_effect_round_trips_against_the_price():
    """`min_detectable_effect` is the inverse of `seeds_needed`, checked as one.

    A restatement of the formula would pass trivially, so the bind is the round
    trip: the effect the inverse returns at a budget must actually price at or
    under that budget, and one notch below it must not.
    """
    for t in (20, 26, 40, 60, 100, 123):
        mu = EP.min_detectable_effect(t)
        assert 0.0 < mu <= EP.B, (t, mu)
        assert EP.seeds_needed(mu)["mixture"] <= t, (t, mu)
        assert EP.seeds_needed(0.9 * mu)["mixture"] > t, (t, mu)


def test_no_budget_below_the_ceiling_can_buy_a_decision():
    """The pre-registered 10 units cannot be rescued by ANY effect size.

    `MIN_T_MIXTURE` paired seeds is 2 units each, so the cheapest decision this
    instrument can ever sell is `2 * 13 = 26` units, and that is at an effect of
    exactly `B` -- every seed at the maximal legal difference.
    """
    assert EP.min_detectable_effect(EP.MIN_T_MIXTURE - 1) == math.inf
    assert EP.min_detectable_effect(5) == math.inf          # the planned run
    assert 2 * EP.MIN_T_MIXTURE == 26


def test_the_zero_point_two_bar_is_necessary_not_sufficient():
    """Clearing `~0.2` NRMSE prices at 246 units against a planned 10.

    The reroute's own bar was set at the effect where the instrument "buys
    cheap decisions". Cheap is relative: at `0.2` the price is `123` paired
    seeds, `246` units, `24.6x` the pre-registered run. Pinned so the bar is
    never read as "affordable".
    """
    price = EP.seeds_needed(0.20)["mixture"]
    assert price == 123
    assert 2 * price == 246
    assert 2 * price / 10 == pytest.approx(24.6)


def test_the_pilot_screen_separates_a_worthwhile_effect_from_the_floor():
    """SIZING ONLY. Three paired seeds, 6 units, decides a 246-unit commitment.

    The go/no-go on `counter_squared` is a point-estimate question, not a
    testing question, so it does not need anytime-validity and must never be
    reported as a verdict. At the pre-registration's paired sd `0.056889` the
    three-seed paired mean has standard error `0.056889 / sqrt(3)`, and a gate
    at `0.10` sits `3.05` standard errors below a true effect of `0.20` and
    `1.52` above the `0.05` resolution floor.
    """
    assert EP.PILOT_SEEDS == 3
    assert EP.PILOT_GATE == 0.10
    se = EP.SD_PAIRED / math.sqrt(EP.PILOT_SEEDS)

    def phi_tail(mu):                # P(N(mu, se) >= gate), closed form
        return 0.5 * math.erfc((EP.PILOT_GATE - mu) / (se * math.sqrt(2.0)))

    go_worth = EP.pilot_rates(0.20)
    go_floor = EP.pilot_rates(EP.EFFECT_FLOOR)
    go_null = EP.pilot_rates(0.0)
    # the simulation must reproduce the closed form, not merely be small
    for got, mu in ((go_worth, 0.20), (go_floor, EP.EFFECT_FLOOR),
                    (go_null, 0.0)):
        assert got == pytest.approx(phi_tail(mu), abs=3e-3), (mu, got)
    assert go_worth > 0.99, go_worth
    assert go_floor < 0.10, go_floor
    assert go_null < 0.002, go_null
    assert go_worth - go_floor > 0.9          # the screen is seen to separate
    assert 2 * EP.PILOT_SEEDS == 6
    assert 2 * EP.seeds_needed(0.20)["mixture"] == 246


def test_the_pilot_gate_is_sized_on_a_spread_measured_on_a_DIFFERENT_task():
    """The screen's assumed spread is borrowed, so its sensitivity is pinned.

    `SD_PAIRED = 0.056889` was measured on `negation_scope`
    (`M3_QUINTUPLE_PREREGISTERED_READING.md:82`), not on `counter_squared`.
    If the new task's paired spread is larger the screen degrades, and by how
    much has to be a number rather than a hope. At 3x the assumed spread the
    screen still catches a true `0.20` effect most of the time but its false-GO
    rate at the `0.05` floor is no longer small, which is the failure mode that
    would waste the 246 units the screen exists to protect.
    """
    for mult, want_go, want_false in ((1.0, 0.9989, 0.0644),
                                      (2.0, 0.9357, 0.2229),
                                      (3.0, 0.8452, 0.3059)):
        sd = mult * EP.SD_PAIRED
        assert EP.pilot_rates(0.20, sd=sd) == pytest.approx(want_go, abs=4e-3)
        assert EP.pilot_rates(EP.EFFECT_FLOOR, sd=sd) == pytest.approx(
            want_false, abs=4e-3)
    # the degradation is monotone and it is SEEN, not assumed
    rates = [EP.pilot_rates(EP.EFFECT_FLOOR, sd=m * EP.SD_PAIRED)
             for m in (1.0, 2.0, 3.0)]
    assert rates[0] < rates[1] < rates[2]
    assert rates[2] > 0.30

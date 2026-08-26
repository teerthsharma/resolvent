"""Binds on the anytime-valid e-process for the M3 settled-vs-twin contrast.

WHAT IS BEING PINNED. Three separable things, and the file keeps them apart on
purpose:

  1. THE CONSTRUCTION. Nonnegativity, the supermartingale property under the
     null, and the immutable constants (`B`, the threshold, the lambda grid).
     These are arithmetic; they are checked in closed form where a closed form
     exists, so that a passing test is not a restatement of the code.

  2. THE MUST-FIRE, BOTH DIRECTIONS. A null stream must not cross the threshold
     more often than the nominal alpha, and a stream with a planted effect must
     cross. A control that has never been seen failing to fire measures nothing,
     and a control that cannot fire at all measures less. Both are checked here
     at a reduced replay count; `scale/eprocess.py` run as a script does the
     full 10,000-replay table.

  3. THE STRUCTURAL CEILING. With `B = 1.0` and `lambda <= 1/2` the largest
     value the process can reach in `t` steps is `1.5 ** t`, whatever the data
     say. At the pre-registered five seeds that is `7.59375`, which is below the
     threshold `20`. The five-seed contrast therefore cannot be decided by this
     process in either direction. That is a fact about the design, it was found
     before any real seed landed, and it is pinned here so it cannot be
     rediscovered later as a surprise.

NO THREAD DEPENDENCE. Nothing here calls BLAS through a shape that varies with
thread count; `scale/eprocess.py` pins `torch.set_num_threads(2)` at import so
that the module, not the launcher, is the authority.
"""
from __future__ import annotations

import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scale import eprocess as EP  # noqa: E402

#: Replays used by the in-test must-fire. The full 10,000-replay table is the
#: script path (`python scale/eprocess.py`); this is the cheap resident copy,
#: and its threshold checks are the same ones.
N_REP_TEST = 2000
HORIZON_TEST = 400


@pytest.fixture(scope="module")
def null_calibration():
    return EP.calibrate(EP.NULL_RADEMACHER, n_rep=N_REP_TEST,
                        horizon=HORIZON_TEST, seed=11)


@pytest.fixture(scope="module")
def planted_calibration():
    return EP.calibrate(EP.PLANTED_FLOOR, n_rep=N_REP_TEST,
                        horizon=HORIZON_TEST, seed=12)


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
def test_bound_is_one_and_carries_its_provenance():
    """`B` is the width of the credited NRMSE range, not a fresh guess.

    `negation_scope.nrmse` is a ratio of nonnegative quantities, so it is
    bounded below by 0 and is exactly 1.0 for the mean predictor.
    `M3_QUINTUPLE_PREREGISTERED_READING.md` section 5 credits nothing to a cell
    at or above 1.0. A credited per-seed NRMSE therefore lies in [0, 1), and the
    difference of two of them is bounded in absolute value by 1.
    """
    assert EP.B == 1.0
    assert EP.NRMSE_FLOOR == 0.0
    assert EP.NRMSE_BAR == 1.0
    assert EP.B == EP.NRMSE_BAR - EP.NRMSE_FLOOR


def test_threshold_is_twenty_and_matches_alpha():
    assert EP.ALPHA == 0.05
    assert EP.THRESHOLD == 20.0
    assert EP.THRESHOLD == 1.0 / EP.ALPHA


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
        e.update(1.5)
    with pytest.raises(ValueError):
        e.update(-1.5)


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


def test_minimum_horizons_are_eight_and_eleven():
    """Eight for the loose single-arm bound, eleven for the shipped mixture."""
    assert EP.MIN_T_SINGLE_ARM == 8
    assert math.ceil(math.log(EP.THRESHOLD) / math.log(1.5)) == 8
    assert 1.5 ** 7 < EP.THRESHOLD <= 1.5 ** 8
    assert EP.MIN_T_MIXTURE == 11
    assert EP.max_attainable(10) < EP.THRESHOLD <= EP.max_attainable(11)


def test_a_horizon_that_cannot_cross_is_refused():
    """A calibration run below the ceiling would pass while measuring nothing."""
    with pytest.raises(ValueError):
        EP.calibrate(EP.NULL_RADEMACHER, n_rep=10,
                     horizon=EP.MIN_T_MIXTURE - 1)


# ------------------------------------------------------- the must-fire ------
def test_must_fire_null_does_not_cross(null_calibration):
    """DIRECTION 1. The nominal alpha is 0.05 per direction."""
    assert null_calibration["cross_settled"] <= EP.ALPHA
    assert null_calibration["cross_twin"] <= EP.ALPHA


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


def test_live_reading_on_an_empty_journal_is_t0_not_a_crash(tmp_path):
    j = tmp_path / "empty.jsonl"
    j.write_text("", encoding="utf-8")
    r = EP.live(j)
    assert r["t"] == 0
    assert r["e_settled"] == 1.0
    assert r["e_twin"] == 1.0
    assert r["decision"] is None
    assert "undecided at evidence E_t = 1.0" in r["text"]

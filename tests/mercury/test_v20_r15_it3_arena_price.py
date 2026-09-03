"""MERCURY it.3 -- the arena's price, asserted rather than asserted-about.

Every number in `V20_R15_IT3_MERCURY.md` that is arithmetic lands here. The
module under test is `tests/mercury/arena_price.py`; it is pure (rows in,
numbers out) so each statistic can be run against a PLANTED input, which is the
only way to show a statistic can move at all.

RED FIRST: this file was written and run before `arena_price.py` existed.
"""
import json
import os

from tests.mercury.arena_price import (
    PER_CELL_150_STEPS,
    arena_seconds,
    clopper_pearson_lower,
    crossing_table,
    expected_cost_to_crossing,
    fisher_vs_skyline,
    n_for_cp_lower,
    power_fisher_vs_skyline,
    power_for_cp_lower,
    seconds_to_floor,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RETAKE = os.path.join(ROOT, "results", "v17k_r4_retake.jsonl")


def _cells_and_floor():
    with open(RETAKE) as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    floor = [r for r in rows if r.get("t") == "header"][0]["floor_1"]
    return [r for r in rows if r.get("t") == "cell"], floor


# ---------------------------------------------------------------- TASK A (1)

def test_clause_1_table_reproduces_and_no_arm_meets_it():
    """The coordinator's it.2 table, recomputed from the journal not quoted."""
    cells, floor = _cells_and_floor()
    tab = crossing_table(cells, floor)
    assert tab["arm_pl"] == (5, 8)
    assert tab["arm_smprime"] == (1, 8)
    assert tab["softmax"] == (0, 8)
    lo = {k: clopper_pearson_lower(x, n) for k, (x, n) in tab.items()}
    assert round(lo["arm_pl"], 4) == 0.2449
    assert round(lo["arm_smprime"], 4) == 0.0032
    assert lo["softmax"] == 0.0
    assert max(lo.values()) < 0.5


def test_at_n8_only_a_clean_sweep_clears_half():
    """The sizing defect: 7/8 tops out at 0.4735, below the 0.5 the clause quotes."""
    got = {x: round(clopper_pearson_lower(x, 8), 4) for x in (8, 7, 6, 5)}
    assert got == {8: 0.6306, 7: 0.4735, 6: 0.3491, 5: 0.2449}
    assert clopper_pearson_lower(8, 8) > 0.5
    assert clopper_pearson_lower(7, 8) < 0.5


def test_n65_is_right_only_under_round_half_and_the_rule_moves_it():
    """`N=65` is the coordinator's number and it is convention-dependent."""
    assert n_for_cp_lower(0.625, 0.5, rule="round") == (65, 41)
    assert n_for_cp_lower(0.625, 0.5, rule="ceil") == (58, 37)
    assert n_for_cp_lower(0.625, 0.5, rule="floor") == (72, 45)
    assert 41 / 65 > 0.625          # 0.6308 -- ABOVE the observed rate
    assert 45 / 72 == 0.625         # only the floor rule holds 0.625 exactly


def test_n65_is_a_coin_flip_not_a_powered_design():
    assert round(power_for_cp_lower(65, 0.625, 0.5), 3) == 0.517
    assert power_for_cp_lower(125, 0.625, 0.5) >= 0.80
    assert power_for_cp_lower(124, 0.625, 0.5) < 0.80


# ---------------------------------------------------------------- TASK A (2)

def test_exit_b_is_already_met_on_data_in_hand():
    """The restated clause -- a rate comparison against the skyline's 0/8 --
    fires at N=8 on the retake cells. Zero additional GPU-seconds."""
    cells, floor = _cells_and_floor()
    tab = crossing_table(cells, floor)
    p1 = fisher_vs_skyline(tab["arm_pl"], tab["softmax"], alternative="greater")
    p2 = fisher_vs_skyline(tab["arm_pl"], tab["softmax"], alternative="two-sided")
    assert round(p1, 6) == 0.012821
    assert round(p2, 6) == 0.025641
    assert p1 < 0.05 and p2 < 0.05
    # CONTROL: the same test does not fire for the loser.
    p_sm = fisher_vs_skyline(tab["arm_smprime"], tab["softmax"],
                             alternative="greater")
    assert p_sm > 0.05


def test_exit_b_is_powered_at_n8_and_the_control_says_where_it_stops():
    assert round(power_fisher_vs_skyline(8, 0.625), 3) == 0.863
    # PLANTED CONTROL: at half the rate N=8 is NOT powered, so 0.863 is a
    # reading of the rate rather than a property of N=8.
    assert power_fisher_vs_skyline(8, 0.3125) < 0.60


# ---------------------------------------------------------------- TASK C

def test_seconds_to_floor_is_not_invariant_under_row_order():
    cells, floor = _cells_and_floor()
    by_file = seconds_to_floor(cells, floor)
    by_best = seconds_to_floor(sorted(cells, key=lambda c: c["eval_nrmse"]), floor)
    by_worst = seconds_to_floor(sorted(cells, key=lambda c: -c["eval_nrmse"]), floor)
    assert by_file == {"arm_pl": 1.884, "arm_smprime": 47.048, "softmax": None}
    assert by_best == {"arm_pl": 1.759, "arm_smprime": 14.852, "softmax": None}
    assert by_worst == {"arm_pl": 7.113, "arm_smprime": 129.287, "softmax": None}
    # MARS's 25.0x and 8.4x both reproduce, and neither is the extreme.
    assert round(by_file["arm_smprime"] / by_file["arm_pl"], 2) == 24.97
    assert round(by_best["arm_smprime"] / by_best["arm_pl"], 2) == 8.44
    assert round(by_worst["arm_smprime"] / by_worst["arm_pl"], 2) == 18.18


def test_expected_cost_to_crossing_is_invariant_under_row_order():
    cells, floor = _cells_and_floor()
    a = expected_cost_to_crossing(cells, floor)
    b = expected_cost_to_crossing(sorted(cells, key=lambda c: c["eval_nrmse"]), floor)
    c = expected_cost_to_crossing(list(reversed(cells)), floor)
    assert a == b == c
    assert round(a["arm_pl"], 4) == 2.8358
    assert round(a["arm_smprime"], 4) == 118.8160
    assert a["softmax"] is None          # 0 crossings -> UNDEFINED, not large
    # PLANTED CROSSING: the statistic must be able to return a number for the
    # skyline, or the None above is a constant rather than a reading.
    planted = [dict(r) for r in cells]
    for r in planted:
        if r["kind"] == "softmax" and r["seed"] == 0:
            r["eval_nrmse"] = floor / 2.0
    assert expected_cost_to_crossing(planted, floor)["softmax"] is not None


def test_ranking_direction_survives_every_ordering():
    """What MARS's strike does NOT overturn: W3 is cheaper on all three."""
    cells, floor = _cells_and_floor()
    for order in (cells,
                  sorted(cells, key=lambda c: c["eval_nrmse"]),
                  sorted(cells, key=lambda c: -c["eval_nrmse"])):
        d = seconds_to_floor(order, floor)
        assert d["arm_pl"] < d["arm_smprime"]
    e = expected_cost_to_crossing(cells, floor)
    assert e["arm_pl"] < e["arm_smprime"]


# ---------------------------------------------------------------- THE PRICE

def test_exit_a_price_at_n65_bed_m_on_the_certified_device():
    """Per-cell seconds are READ from V17_R4_RETAKE_PRICE.md:194-196; the
    extrapolation to N is stated as multiplication and nothing else."""
    assert PER_CELL_150_STEPS == {"arm_smprime": 15.970, "arm_pl": 1.614,
                                  "softmax": 1.497}
    # the price doc's own N=8 total (158.74 s) is the control this must land on
    assert round(arena_seconds(8, PER_CELL_150_STEPS), 2) == 156.15
    assert abs(arena_seconds(8, PER_CELL_150_STEPS) - 158.74) / 158.74 < 0.02
    assert round(arena_seconds(65, PER_CELL_150_STEPS), 1) == 1243.8
    assert round(arena_seconds(65, PER_CELL_150_STEPS) / 3600.0, 4) == 0.3455
    # W2 priced off arm_pl (arm_phase = arm_pl magnitude x a phase factor)
    four = dict(PER_CELL_150_STEPS, arm_phase=1.614)
    assert round(arena_seconds(65, four) / 3600.0, 4) == 0.3746
    assert round(arena_seconds(125, PER_CELL_150_STEPS) / 3600.0, 4) == 0.6635


def test_w2_gate_is_one_cell_and_under_twenty_seconds_either_way():
    lo = arena_seconds(1, {"arm_phase": 1.614})       # priced off arm_pl
    hi = arena_seconds(1, {"arm_phase": 15.970})      # priced off arm_smprime
    assert round(lo, 3) == 5.114 and round(hi, 3) == 19.470
    assert hi < 20.0

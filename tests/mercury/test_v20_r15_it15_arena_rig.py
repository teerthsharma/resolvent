"""it.15 MERCURY — the Phase C arena rig, its planted-crossing must-fire, and
its planted NON-crossing must-not-fire.

The must-fire exists because the GPU-seconds-to-floor instrument has never been
exercised on a positive it did not find for itself. An instrument that cannot
detect a crossing it was TOLD is there measures nothing (V-16). The negative
exists because a must-fire without one is half a control: an instrument that
fires on everything also detects a crossing it was told is there.
"""
import json
import math
import pathlib

import pytest

from tests.mercury.arena_rig import (
    FLOOR_1,
    RUN_ORDER_RHO,
    Reading,
    arena_row,
    cp_lower_both_tails,
    crosses,
    gpu_seconds_to_floor,
    load_cells,
    plant_crossing,
    plant_non_crossing,
    spearman_rho,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- TASK A
def test_the_floor_is_the_runners_own_floor_not_a_retyped_constant():
    # scripts/v15_r1.py:586  floor1 = sqrt((T_STAR - 1) / T_STAR), T_STAR = 2.
    assert FLOOR_1 == math.sqrt(0.5)
    assert f"{FLOOR_1:.4f}" == "0.7071"


def test_the_crossing_predicate_is_the_runners_own_line_909_not_a_point_compare():
    # scripts/v15_r1.py:909  crosses = bool(m + half < floor1)
    src = (ROOT / "scripts" / "v15_r1.py").read_text(encoding="utf-8")
    assert "crosses=bool(m + half < floor1)" in src


def test_MUST_FIRE_the_instrument_finds_a_crossing_it_was_told_is_there():
    """A planted crossing at cell 3 of 5. The instrument must return the
    cumulative seconds THROUGH that cell and name the cell."""
    cells = plant_crossing(at=2, secs=(1.0, 2.0, 4.0, 8.0, 16.0))
    r = gpu_seconds_to_floor(cells)
    assert r.value == pytest.approx(7.0), r
    assert r.why is None
    assert r.first_crossing_index == 2


def test_MUST_NOT_FIRE_a_planted_NON_crossing_returns_None_with_a_reason():
    """The other half of the control. Every cell sits above the floor; the
    instrument must decline, not interpolate."""
    cells = plant_non_crossing(secs=(1.0, 2.0, 4.0, 8.0, 16.0))
    r = gpu_seconds_to_floor(cells)
    assert r.value is None
    assert "0 of 5" in r.why
    assert r.first_crossing_index is None


def test_MUST_NOT_FIRE_the_near_miss_that_separates_the_rule_from_its_lazy_twin():
    """The sharp negative. Point estimate BELOW the floor, bootstrap upper limit
    ABOVE it. A `eval_nrmse < floor` instrument fires here and is wrong; the
    runner's `m + half < floor` rule must not."""
    cells = plant_non_crossing(secs=(1.0, 2.0), near_miss=True)
    assert all(c["eval_nrmse"] < FLOOR_1 for c in cells)
    assert not any(crosses(c) for c in cells)
    assert gpu_seconds_to_floor(cells).value is None


def test_the_planted_pair_differ_only_in_the_bootstrap_width():
    """The plant is a control only if the two arms of it are otherwise equal."""
    yes = plant_crossing(at=0, secs=(1.0,))
    no = plant_non_crossing(secs=(1.0,), near_miss=True)
    assert yes[0]["secs"] == no[0]["secs"]
    assert crosses(yes[0]) and not crosses(no[0])


def test_every_rig_row_states_the_run_order_confound_in_its_own_output():
    """it.13 finding carried INTO the rig, not around it: synchronize() does not
    remove run order, and the randomisation is a fifth unpriced edit."""
    cells = plant_crossing(at=1, secs=(1.0, 2.0, 3.0))
    row = arena_row(cells, arm="planted")
    assert "run_order_confound" in row
    assert "0.7029" in row["run_order_confound"]
    assert row["gpu_seconds_to_floor"].value == pytest.approx(3.0)


def test_the_run_order_rho_is_reproduced_from_the_banked_cells_not_quoted():
    """it.17 DEFECT (it.15, this office): the operands were never named.

    The it.15 form correlated index-ACROSS-ARMS with `secs`, which measures the
    arm, not run order — `early_warning` runs ~16 arm-dependent forward passes
    inside the timed window. It returns -0.1224 and reproduces nothing. This is
    the same unnamed-operand defect the Inspector filed against
    `V20_R15_IT13_MERCURY.md:83` (DEFECT B). The measured pair is seed against
    `secs` WITHIN one arm; both readings ship, so the confound is visible.
    """
    cells = sorted((c for c in load_cells(ROOT) if c["kind"] == "arm_smprime"),
                   key=lambda c: c["seed"])
    rho = spearman_rho([c["seed"] for c in cells], [c["secs"] for c in cells])
    assert rho == pytest.approx(RUN_ORDER_RHO, abs=5e-3), (rho, RUN_ORDER_RHO)

    every = load_cells(ROOT)
    across = spearman_rho(list(range(len(every))), [c["secs"] for c in every])
    assert across == pytest.approx(-0.1224, abs=5e-3), across
    assert abs(across - rho) > 0.5  # the arm confound, in the rig's own test


# ---------------------------------------------------------------- TASK C
def test_clause_1_is_emitted_BOTH_WAYS_always_so_no_office_picks_after_seeing():
    both = cp_lower_both_tails(12, 16)
    assert both["two_sided"] == pytest.approx(0.4762, abs=5e-5)
    assert both["one_sided"] == pytest.approx(0.5156, abs=5e-5)
    assert both["ruling"] == "UNRULED"
    assert both["verdict_two_sided"] == "FAILS"
    assert both["verdict_one_sided"] == "CLEARS"


def test_the_rig_cannot_emit_one_tail_alone():
    row = arena_row(plant_crossing(at=0, secs=(1.0,)), arm="planted")
    cp = row["cp_lower"].value
    assert set(cp) >= {"two_sided", "one_sided", "ruling"}


def test_clause_1_on_the_real_arms_reproduces_the_journalled_12_of_16():
    cells = load_cells(ROOT)
    pl = [c for c in cells if c["kind"] == "arm_pl"]
    sm = [c for c in cells if c["kind"] == "arm_smprime"]
    assert (sum(crosses(c) for c in pl), len(pl)) == (12, 16)
    assert (sum(crosses(c) for c in sm), len(sm)) == (1, 16)


# ---------------------------------------------------------------- TASK B
@pytest.mark.parametrize("arm", ["arm_pl", "arm_smprime"])
def test_the_absent_columns_are_None_WITH_A_REASON_not_a_plausible_number(arm):
    cells = [c for c in load_cells(ROOT) if c["kind"] == arm]
    row = arena_row(cells, arm=arm)
    for col in ("dist_to_skyline", "w1_to_oracle", "peak_bytes", "cert_grade"):
        r = row[col]
        assert isinstance(r, Reading)
        assert r.value is None, (col, r)
        assert r.why and len(r.why) > 20, (col, r)


def test_dist_to_skyline_is_None_on_all_40_and_carries_the_cells_own_reason():
    """it.17 DEFECT (it.15, this office): `34` was retyped off the wrong corpus.

    34 is the it.8 undeduplicated `retake + it6` record count
    (`tests/jupiter/test_v20_r15_it8_q4_q5.py:60-62`, "32 distinct + seeds 0,1
    measured twice"), whose `arm_smprime` count is 18. It cannot coexist with
    the `(1, 16)` assertion twelve lines above. The banked corpus is 40 cells:
    16 `arm_pl` + 16 `arm_smprime` + 8 `softmax`, which is what
    `tests/mercury/phase_c_price.py:18` already calls BANKED_40.
    """
    cells = load_cells(ROOT)
    assert len(cells) == 40
    whys = {c.get("dist_to_skyline_why") for c in cells}
    assert all(c["dist_to_skyline"] is None for c in cells)
    assert whys == {"no v15/v16 scan-skyline module exists (R-SKY)"}


def test_the_rig_emits_every_column_the_contract_names():
    # CEQ_V20_R15_CONTRACT.md:120-124
    row = arena_row([c for c in load_cells(ROOT) if c["kind"] == "arm_pl"], arm="arm_pl")
    for col in ("cp_lower", "conditional_nrmse", "dist_to_floor", "dist_to_skyline",
                "gpu_seconds_to_floor", "peak_bytes", "cert_grade", "w1_to_oracle"):
        assert col in row, col


def test_quotient_observables_only_no_raw_parameter_leaks_into_a_row():
    row = arena_row([c for c in load_cells(ROOT) if c["kind"] == "arm_pl"], arm="arm_pl")
    banned = {"lambda_hat", "a_hat_max", "v_max", "c", "sign_sst"}
    assert not (banned & set(row))


def test_the_rig_writes_nothing_and_the_frozen_journal_is_untouched():
    import hashlib
    h = hashlib.sha256((ROOT / "results" / "v17k_r4_retake.jsonl").read_bytes()).hexdigest()
    assert h.startswith("26fb180b"), h

"""VENUS it.5 — every claim about the record that the ranking rests on.

Each assert is a node. If a number in V20_R15_IT5_VENUS.md is not bound here,
it is not a finding, it is prose.
"""
import json
import math
import os
from collections import defaultdict

import pytest

scipy_stats = pytest.importorskip("scipy.stats")
from scipy.stats import beta  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RETAKE = os.path.join(ROOT, "results", "v17k_r4_retake.jsonl")
FLOOR_RUN = os.path.join(ROOT, "results", "v17k_r4_floor.jsonl")
PRICE = os.path.join(ROOT, "V17_R4_RETAKE_PRICE.md")

FLOOR_1 = 0.7071067811865476


def _cells(path):
    by = defaultdict(dict)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("t") == "cell":
                by[rec["kind"]][rec["seed"]] = rec
    return by


@pytest.fixture(scope="module")
def cells():
    return _cells(RETAKE)


def cp_lower(x, n, two_sided=True):
    """Clopper-Pearson lower confidence limit. two_sided=True -> alpha/2."""
    if x == 0:
        return 0.0
    alpha = 0.05 / 2 if two_sided else 0.05
    return float(beta.ppf(alpha, x, n - x + 1))


# --- 1. the arena's own header, so the floor is not a constant we invented ----
def test_floor_comes_from_the_run_header():
    with open(RETAKE, encoding="utf-8") as fh:
        hdr = json.loads(fh.readline())
    assert hdr["t"] == "header"
    assert hdr["floor_1"] == FLOOR_1
    assert hdr["steps"] == 150 and hdr["seeds"] == list(range(8))
    assert hdr["deterministic_algorithms"] is True


# --- 2. the crossing counts the ranking is built on --------------------------
@pytest.mark.parametrize(
    "kind,crossings,crossing_seeds",
    [("arm_smprime", 1, [2]), ("arm_pl", 5, [0, 1, 4, 5, 6]), ("softmax", 0, [])],
)
def test_crossings_of_floor1(cells, kind, crossings, crossing_seeds):
    got = sorted(s for s, c in cells[kind].items() if c["eval_nrmse"] < FLOOR_1)
    assert len(cells[kind]) == 8
    assert got == crossing_seeds
    assert len(got) == crossings


# --- 3. CP-lower, with the convention NAMED (the briefed table is two-sided) --
@pytest.mark.parametrize(
    "kind,two_sided,one_sided",
    [("arm_smprime", 0.0032, 0.0064), ("arm_pl", 0.2449, 0.2892), ("softmax", 0.0, 0.0)],
)
def test_cp_lower_under_both_conventions(cells, kind, two_sided, one_sided):
    x = sum(1 for c in cells[kind].values() if c["eval_nrmse"] < FLOOR_1)
    assert cp_lower(x, 8, True) == pytest.approx(two_sided, abs=5e-5)
    assert cp_lower(x, 8, False) == pytest.approx(one_sided, abs=5e-5)
    # neither convention gets either arm near the clause-(1) bar
    assert cp_lower(x, 8, False) < 0.5


# --- 4. only a clean sweep clears the clause-(1) bar at N=8 ------------------
def test_seven_of_eight_does_not_clear_the_bar():
    assert cp_lower(7, 8, True) == pytest.approx(0.4735, abs=5e-5)
    assert cp_lower(7, 8, True) < 0.5
    assert cp_lower(8, 8, True) == pytest.approx(0.6306, abs=5e-5)
    assert cp_lower(8, 8, True) > 0.5


# --- 5. the best cells ------------------------------------------------------
def test_best_eval_nrmse_per_arm(cells):
    sm = min(cells["arm_smprime"].values(), key=lambda c: c["eval_nrmse"])
    pl = min(cells["arm_pl"].values(), key=lambda c: c["eval_nrmse"])
    assert sm["seed"] == 2 and sm["eval_nrmse"] == pytest.approx(0.203920, abs=5e-7)
    assert pl["seed"] == 4 and pl["eval_nrmse"] == pytest.approx(0.633739, abs=5e-7)
    assert pl["eval_nrmse"] / sm["eval_nrmse"] == pytest.approx(3.108, abs=5e-4)


# --- 6. THE MECHANISM: seed 2 is the unique un-annihilated arm_smprime cell --
def test_seed2_is_the_only_smprime_cell_with_a_live_gate(cells):
    sm = cells["arm_smprime"]
    alive = sorted(s for s, c in sm.items() if c["frac_gate_annihilated"] == 0.0)
    assert alive == [2], "seed 2 must be the unique live-gate cell"
    # the other seven sit at the annihilating corner, at a shared constant
    dead = [c["frac_gate_annihilated"] for s, c in sm.items() if s != 2]
    assert all(d > 0.49 for d in dead)
    assert len(set(dead)) <= 2, "the corner is a structural constant, not a fit"
    # and only seed 2 has a finite decay and no unit root
    assert math.isfinite(sm[2]["lambda_hat"]) and sm[2]["unit_root"] is False
    assert all(sm[s]["lambda_hat"] == -math.inf for s in sm if s != 2)
    assert sm[2]["a_hat_min"] == pytest.approx(0.34076, abs=5e-5)
    assert sm[2]["a_hat_max"] == pytest.approx(0.54004, abs=5e-5)


def test_arm_pl_failures_are_divergence_not_annihilation(cells):
    pl = cells["arm_pl"]
    assert all(c["frac_gate_annihilated"] == 0.0 for c in pl.values())
    fails = sorted(s for s, c in pl.items() if c["eval_nrmse"] >= FLOOR_1)
    assert fails == [2, 3, 7]
    # every failure blows the gate cap up; every crossing keeps it near 1
    assert all(pl[s]["a_hat_max"] > 12.0 for s in fails)
    assert all(pl[s]["a_hat_max"] < 1.6 for s in pl if s not in fails)


# --- 7. the pipeline is deterministic per seed: a re-run of seed 2 is not a test
def test_two_independent_runs_reproduce_seeds_0_and_1_bitwise(cells):
    other = _cells(FLOOR_RUN)
    for seed in (0, 1):
        assert other["arm_smprime"][seed]["eval_nrmse"] == cells["arm_smprime"][seed]["eval_nrmse"]
    # and no run anywhere holds an arm_smprime cell at a seed above 7
    assert max(other["arm_smprime"]) <= 7 and max(cells["arm_smprime"]) <= 7


# --- 8. the cost row is a DIFFERENT run from the crossing row ----------------
def test_the_briefed_cost_row_is_not_the_arena_cells_own_seconds(cells):
    price = open(PRICE, encoding="utf-8").read().splitlines()
    assert "15.970" in price[193] and "arm_smprime" in price[193]  # :194, 1-indexed
    assert "1.614" in price[194] and "arm_pl" in price[194]        # :195
    assert "1.497" in price[195] and "softmax" in price[195]       # :196
    # that table is "mean of 2" (seeds 0,1). The 24 arena cells say otherwise.
    means = {
        k: sum(c["secs"] for c in cells[k].values()) / 8
        for k in ("arm_smprime", "arm_pl", "softmax")
    }
    assert means["arm_smprime"] == pytest.approx(16.161, abs=5e-4)
    assert means["arm_pl"] == pytest.approx(1.780, abs=5e-4)
    assert means["softmax"] == pytest.approx(1.681, abs=5e-4)
    assert means["arm_pl"] > 1.614 and means["softmax"] > 1.497


def test_mercurys_expected_cost_to_a_crossing_reproduces_from_its_own_basis(cells):
    sm, pl = cells["arm_smprime"], cells["arm_pl"]
    # MERCURY:274/:278-279 -- mean secs over the CROSSING cells x n/x
    assert sm[2]["secs"] * 8 / 1 == pytest.approx(118.816, abs=5e-3)
    pl_cross = [pl[s]["secs"] for s in (0, 1, 4, 5, 6)]
    assert (sum(pl_cross) / 5) * 8 / 5 == pytest.approx(2.8358, abs=5e-4)
    # on an all-cell basis the same statistic reads differently for W1
    assert (sum(c["secs"] for c in sm.values()) / 8) * 8 == pytest.approx(129.29, abs=5e-2)


# --- 9. W3 is not 1/10 of the incumbent, under either basis ------------------
def test_no_wing_reaches_the_scoreboards_one_tenth_incumbent_cost(cells):
    means = {k: sum(c["secs"] for c in cells[k].values()) / 8 for k in cells}
    assert means["arm_pl"] / means["softmax"] > 1.0   # W3 costs MORE than softmax
    assert 1.614 / 1.497 > 1.0                        # and on the price basis too

"""it.8 MERCURY. Experiment A (symmetric arm_pl) and Experiment C (the clamp).

RED-FIRST: the first draft carried `test_a_hat_max_1_0_is_absent_before_training`,
which encodes the story this office was about to publish -- that
`a_hat_max == 1.0` is a trained pole on the unit circle. It went RED with
`[2, 3, 4, 7, 12, 14]`: six cells reach 1.0 at ZERO training steps. The node
below records what the journals say instead. The RED is quoted verbatim in
V20_R15_IT8_MERCURY.md and is not overwritten here.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
NEW = ROOT / "results" / "v20_r15_it8_armpl_b.jsonl"
RETAKE = ROOT / "results" / "v17k_r4_retake.jsonl"
IT6 = ROOT / "results" / "v20_r15_it6_seeds8_15.jsonl"

VARY = {"arms", "seeds", "tag", "when"}


def load(p):
    return [json.loads(l) for l in p.open()]


def cells(rows, kind=None):
    return [r for r in rows if r.get("t") == "cell"
            and (kind is None or r.get("kind") == kind)]


def test_regime_matches_the_retake_field_for_field():
    new, ret = load(NEW)[0], load(RETAKE)[0]
    assert set(new) == set(ret)
    diff = {k for k in new if new[k] != ret[k]}
    assert diff == VARY, diff
    assert len(set(new) - VARY) == 18
    assert new["instrument_hash"] == ret["instrument_hash"]


def test_control_seed_0_reproduces_bitwise():
    n = {c["seed"]: c for c in cells(load(NEW), "arm_pl")}[0]
    r = {c["seed"]: c for c in cells(load(RETAKE), "arm_pl")}[0]
    for k in ("eval_nrmse", "a_hat_max", "a_hat_min", "lambda_hat",
              "lambda_hat_live", "gate_r2", "frac_gate_annihilated", "v_max"):
        assert n[k] == r[k], (k, n[k], r[k])


def test_nine_cells_seed_0_and_8_through_15():
    got = sorted(c["seed"] for c in cells(load(NEW), "arm_pl"))
    assert got == [0, 8, 9, 10, 11, 12, 13, 14, 15]


def test_fresh_eight_columns_as_published():
    rows = load(NEW)
    floor = rows[0]["floor_1"]
    fresh = [c for c in cells(rows, "arm_pl") if c["seed"] >= 8]
    assert len(fresh) == 8
    assert [c["seed"] for c in fresh if c["eval_nrmse"] <= floor] == \
        [8, 10, 11, 12, 13, 14, 15]
    assert all(c["frac_gate_annihilated"] == 0.0 for c in fresh)
    assert [c["seed"] for c in fresh if c["lambda_hat"] > 0] == [9]
    assert round(sum(c["secs"] for c in fresh), 3) == 13.718


def test_a_hat_max_is_the_clamp_ceiling_not_a_pole():
    smp = cells(load(IT6), "arm_smprime") + cells(load(RETAKE), "arm_smprime")
    # the clamp's ceiling: never exceeded, on any smprime cell, trained or not
    assert max(c["a_hat_max"] for c in smp) == 1.0
    assert max(c["a_hat_max_0step"] for c in smp) == 1.0
    # and reached at ZERO training steps, which no convergence can explain
    assert sorted({c["seed"] for c in smp if c["a_hat_max_0step"] == 1.0}) == \
        [2, 3, 4, 7, 12, 14]
    # the same column, same instrument, on the UNCAPPED arm, leaves 1.0 freely
    pl = cells(load(RETAKE), "arm_pl") + cells(load(NEW), "arm_pl")
    assert max(c["a_hat_max"] for c in pl) > 100.0
    assert all(c["a_hat_max"] > 1.0 for c in pl)


def test_experiment_B_no_capped_cells_exist():
    """Experiment B was NOT run: no cap mechanism is executable. RED and
    stays RED until the cap is specified."""
    assert not list((ROOT / "results").glob("v20_r15_it8_capped*.jsonl"))

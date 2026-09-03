"""IT.11 MERCURY -- RED first. arm_smprime scored on the same three eval draws.

The round stands at 18 draw-checked W3/skyline cells against 0 for W1. These
assertions fail until `results/v20_r15_it11_mercury_smprime.jsonl` exists with
all 16 banked arm_smprime seeds on eval draws 12345 / 12346 / 20260902, and
until the 12345 column reproduces the frozen journals BITWISE.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
NEW = ROOT / "results" / "v20_r15_it11_mercury_smprime.jsonl"
BANKED = {
    "results/v17k_r4_retake.jsonl": range(0, 8),
    "results/v20_r15_it6_seeds8_15.jsonl": range(8, 16),
}
EVAL_SEEDS = [12345, 12346, 20260902]
FLOOR_1 = 0.7071067811865476


def _rows():
    return [json.loads(l) for l in NEW.open(encoding="utf-8")]


def test_journal_exists_and_is_not_the_frozen_one():
    assert NEW.exists(), f"{NEW} missing -- RED"
    frozen = ROOT / "results" / "v17k_r4_retake.jsonl"
    assert NEW.resolve() != frozen.resolve()


def test_header_keeps_the_instrument_hash():
    h = [r for r in _rows() if r.get("t") == "header"][0]
    assert h["instrument_hash"] == (
        "5d41a63d57671725384b33249693d884ec17d0dcca11be7b3e58d6619bb9a309")
    assert h["arms"] == ["arm_smprime"]
    assert h["eval_seeds"] == EVAL_SEEDS


def test_control_column_reproduces_banked_bitwise():
    got = {(r["seed"], r["eval_seed"]): r["eval_nrmse"]
           for r in _rows() if r.get("t") == "rescore"}
    for path, seeds in BANKED.items():
        for line in (ROOT / path).open(encoding="utf-8"):
            r = json.loads(line)
            if r.get("kind") != "arm_smprime" or r.get("seed") not in seeds:
                continue
            if "eval_nrmse" not in r or r.get("t") != "cell":
                continue
            key = (r["seed"], 12345)
            if key not in got:
                continue  # partial run; coverage asserted separately
            assert got[key] == r["eval_nrmse"], (
                f"seed {r['seed']} not bitwise: {got[key]!r} vs {r['eval_nrmse']!r}")


def test_seed_2_is_scored_on_all_three_draws():
    got = {(r["seed"], r["eval_seed"]) for r in _rows() if r.get("t") == "rescore"}
    for es in EVAL_SEEDS:
        assert (2, es) in got, f"seed 2 missing eval draw {es}"


def test_every_banked_seed_on_every_draw():
    got = {(r["seed"], r["eval_seed"]) for r in _rows() if r.get("t") == "rescore"}
    missing = [(s, e) for s in range(16) for e in EVAL_SEEDS if (s, e) not in got]
    assert not missing, f"missing cells: {missing}"


def test_crossing_verdict_is_computed_against_the_stated_floor():
    for r in _rows():
        if r.get("t") != "rescore":
            continue
        assert r["floor_1"] == FLOOR_1
        assert r["crosses_cell"] == (r["eval_nrmse"] < FLOOR_1)
        for k in ("frac_gate_annihilated", "lambda_hat", "lambda_hat_live",
                  "a_hat_max", "unit_root"):
            assert k in r, f"{k} missing from seed {r['seed']} draw {r['eval_seed']}"

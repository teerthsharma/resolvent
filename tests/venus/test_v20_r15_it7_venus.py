"""VENUS it.7 — every claim this office makes about the existing record.

Basis: results/v17k_r4_retake.jsonl (seeds 0-7, three arms)
     + results/v20_r15_it6_seeds8_15.jsonl (arm_smprime, seeds 0,1 control + 8-15)
"""
import json, math, pathlib
from math import comb
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
FLOOR = 0.7071067811865476
NEG = float("-inf")
LATTICE = 8192  # 2 * s * s with s = 64


def _cells(name):
    p = ROOT / "results" / name
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()
            and json.loads(l).get("t") == "cell"]


RETAKE = _cells("v17k_r4_retake.jsonl")
IT6 = _cells("v20_r15_it6_seeds8_15.jsonl")
ALL = RETAKE + IT6
SM16 = ([c for c in RETAKE if c["kind"] == "arm_smprime"]
        + [c for c in IT6 if c["kind"] == "arm_smprime" and c["seed"] >= 8])


def crossed(c):
    return c["eval_nrmse"] < c["floor_1"]


def test_basis_shape():
    assert len(RETAKE) == 24 and len(IT6) == 10 and len(SM16) == 16
    assert all(c["floor_1"] == FLOOR for c in ALL)


def test_it6_ran_only_w1():
    hdr = [json.loads(l) for l in (ROOT / "results" / "v20_r15_it6_seeds8_15.jsonl")
           .open(encoding="utf-8") if json.loads(l).get("t") == "header"][0]
    assert hdr["arms"] == ["arm_smprime"], "no fresh arm_pl or softmax cell exists"


def test_controls_reproduce_bitwise():
    """Seeds 0 and 1 re-run inside it.6 must equal the retake to the bit."""
    fields = ["eval_nrmse", "train_nrmse", "frac_gate_annihilated", "lambda_hat",
              "unit_root", "boot_lo", "boot_hi", "sign_acc", "gate_r2",
              "a_hat_max", "v_max", "z_winding_max", "eval_h_hat",
              "nrmse0_eval", "conservation_drift"]
    a = {c["seed"]: c for c in RETAKE if c["kind"] == "arm_smprime"}
    b = {c["seed"]: c for c in IT6}
    for s in (0, 1):
        for f in fields:
            assert repr(a[s][f]) == repr(b[s][f]), (s, f)


def test_rate_prediction_survived():
    """it.5 forecast 1 of 8, 95% PI 0-3. Observed 0."""
    fresh = [c for c in IT6 if c["seed"] >= 8]
    assert len(fresh) == 8
    assert sum(crossed(c) for c in fresh) == 0
    assert sum(crossed(c) for c in SM16) == 1          # pooled 1 of 16
    assert [c["seed"] for c in SM16 if crossed(c)] == [2]


def test_mechanism_prediction_falsified():
    """it.5 said failures carry frac in S5 = {0.4967041015625, 0.5032958984375}."""
    S5 = {0.4967041015625, 0.5032958984375}
    fresh = [c for c in IT6 if c["seed"] >= 8]
    out = [(c["seed"], c["frac_gate_annihilated"]) for c in fresh
           if not crossed(c) and c["frac_gate_annihilated"] not in S5]
    assert out == [(11, 0.99462890625), (13, 0.976806640625)]
    # and the values it.5 leaned on did not recur at all
    fr = [c["frac_gate_annihilated"] for c in fresh]
    assert fr.count(0.4967041015625) == 0 and fr.count(0.0) == 0


def test_it5_named_killer_could_not_fire():
    """The killer was 'fails with a LIVE gate'. Both breakers had deader gates."""
    for s in (11, 13):
        c = [x for x in IT6 if x["seed"] == s][0]
        assert not crossed(c)
        assert c["frac_gate_annihilated"] > 0.5032958984375   # deader, not live
        assert c["unit_root"] is False


def test_two_basins_is_dead():
    vals = sorted({c["frac_gate_annihilated"] for c in SM16})
    assert len(vals) == 5
    assert vals == [0.0, 0.4967041015625, 0.5032958984375,
                    0.976806640625, 0.99462890625]


def test_L1_gate_death_determines_minus_inf():
    """frac > 0  <=>  lambda_hat == -inf, over every cell of every arm."""
    for c in ALL:
        assert (c["frac_gate_annihilated"] > 0) == (c["lambda_hat"] == NEG), c["seed"]
    assert sum(1 for c in ALL if c["lambda_hat"] == NEG) == 17


def test_L1_is_independent_of_unit_root():
    """Seeds 11 and 13: unit_root False yet lambda -inf. So -inf is NOT tied
    to a unit root; gate death alone determines it."""
    off = [c for c in ALL if c["unit_root"] is False]
    assert sorted((c["kind"], c["seed"]) for c in off) == [
        ("arm_smprime", 2), ("arm_smprime", 3), ("arm_smprime", 11), ("arm_smprime", 13)]
    # and the it.5 uniqueness claim about seed 2 still holds, now over 34 cells
    fin = [(c["kind"], c["seed"]) for c in ALL
           if c["unit_root"] is False and c["lambda_hat"] != NEG]
    assert fin == [("arm_smprime", 2)]


def test_L2_annihilation_lattice():
    """frac * 8192 is an integer for every cell, trained and 0-step."""
    # NOTE: softmax cells carry no `frac_gate_annihilated_0step` key at all
    # (8/8 absent); arm_pl and arm_smprime carry it on 26/26. Probe coverage is
    # not uniform across arms and the lattice claim is scoped to what exists.
    for c in ALL:
        for k in ("frac_gate_annihilated", "frac_gate_annihilated_0step"):
            if k not in c:
                assert c["kind"] == "softmax" and k.endswith("_0step")
                continue
            v = c[k] * LATTICE
            assert abs(v - round(v)) < 1e-9, (c["kind"], c["seed"], k, c[k])
    assert sum(1 for c in ALL if "frac_gate_annihilated_0step" not in c) == 8
    ks = sorted({round(c["frac_gate_annihilated"] * LATTICE) for c in SM16})
    assert ks == [0, 4069, 4123, 8002, 8148]


def test_L3_dose_response():
    """eval_nrmse rises monotonically with frac across the three clusters."""
    live = [c for c in SM16 if c["frac_gate_annihilated"] == 0.0]
    half = [c for c in SM16 if 0.49 < c["frac_gate_annihilated"] < 0.51]
    dead = [c for c in SM16 if c["frac_gate_annihilated"] >= 0.9]
    assert (len(live), len(half), len(dead)) == (1, 13, 2)
    assert max(x["eval_nrmse"] for x in live) < min(x["eval_nrmse"] for x in half)
    assert max(x["eval_nrmse"] for x in half) < min(x["eval_nrmse"] for x in dead)
    assert all(x["eval_nrmse"] > 1.0 for x in dead)     # worse than predict-the-mean
    assert all(x["eval_nrmse"] < FLOOR for x in live)


def test_arm_rates_unchanged_for_w3_and_softmax():
    for kind, k, n in (("arm_pl", 5, 8), ("softmax", 0, 8)):
        cs = [c for c in ALL if c["kind"] == kind]
        assert len(cs) == n and sum(crossed(c) for c in cs) == k


def test_every_crosser_anywhere_has_a_live_gate():
    """Across all 34 cells and all three arms: crossing implies frac == 0."""
    assert all(c["frac_gate_annihilated"] == 0.0 for c in ALL if crossed(c))
    assert sum(crossed(c) for c in ALL) == 6


def test_symmetric_experiment_price():
    pl = [c["secs"] for c in ALL if c["kind"] == "arm_pl"]
    assert len(pl) == 8
    assert abs(sum(pl) - 14.240) < 5e-4                     # 8 fresh arm_pl cells
    assert abs(sum(pl) / 8 - 1.7800) < 5e-5
    sm = sum(c["secs"] for c in IT6)
    assert abs(sm - 178.460) < 5e-4
    assert 0.075 < sum(pl) / sm < 0.081                     # ~1/12.5, not 1/10


def test_mars_predictor_issued_zero_positive_predictions():
    """Rule: frac_gate_annihilated_0step < 0.15 predicts a crossing."""
    fresh = [c for c in IT6 if c["seed"] >= 8]
    assert not any(c["frac_gate_annihilated_0step"] < 0.15 for c in fresh)
    assert min(c["frac_gate_annihilated_0step"] for c in fresh) == 0.380859375
    s2 = [c for c in RETAKE if c["kind"] == "arm_smprime" and c["seed"] == 2][0]
    assert s2["frac_gate_annihilated_0step"] == 0.124755859375   # its one instance


def test_predictive_interval_for_next_eight():
    """Jeffreys Beta(1.5, 15.5) on 1/16; beta-binomial over 8 fresh seeds."""
    a, b = 1.5, 15.5
    lb = lambda x, y: math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y)
    pmf = [comb(8, i) * math.exp(lb(a + i, b + 8 - i) - lb(a, b)) for i in range(9)]
    assert abs(sum(pmf) - 1.0) < 1e-12
    assert abs(sum(i * p for i, p in enumerate(pmf)) - 0.7059) < 5e-5
    cdf = [sum(pmf[:i + 1]) for i in range(9)]
    assert next(i for i, x in enumerate(cdf) if x >= 0.95) == 3   # 95% PI = 0-3
    assert abs(sum(pmf[4:]) - 0.01487) < 5e-6

# pytest test_r2.py -- implementation tests + the registered claims of BAR.md (read from eval.json / tune.json).
import json, sys
from pathlib import Path
import numpy as np
import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import rj
import r2

INF = np.inf


@pytest.fixture(scope="module")
def b12():
    return r2.batch(np.random.default_rng([99, 12, 0]), 300, 12)


def test_batch_matches_round1_generator():
    b = r2.batch(np.random.default_rng([98, 12, 0]), 200, 12)
    ref = rj.make_batch(np.random.default_rng([98, 12, 0]), 200, 12)
    assert np.array_equal(b["succ"], ref["succ"])
    assert np.allclose(b["hits"].mean(-1), ref["P_bayes"])
    assert np.allclose(b["stretch"][..., -1], rj.sigma(ref))


def test_cascade_without_pruning_is_bayes(b12):
    pick, calls = r2.cascade(b12, INF, INF, INF, 16)
    assert np.array_equal(pick, r2.pick_bayes(b12))
    assert np.all(calls == rj.K * (r2.M + 3) * 12)


def test_prune_K_is_bayes(b12):
    pick, calls = r2.prune(b12, rj.K)
    assert np.array_equal(pick, r2.pick_bayes(b12))
    assert np.all(calls == 2 * rj.K * 12 + rj.K * r2.M * 12)


def test_mixed_everywhere_costs_one_stage0_step():
    b = r2.batch(np.random.default_rng([97, 12, 0]), 100, 12)
    pick, calls = r2.cascade(b, INF, 0.0, 0.5, 16)          # kappa = 0: every candidate mixed at t = 1
    assert np.all(calls == 3 * rj.K)


def test_settled_short_lead_is_bayes_and_cheap():
    b = r2.batch(np.random.default_rng([96, 2, 0]), 500, 2)
    pick, calls = r2.cascade(b, 3.0, INF, 0.5, 16)
    assert rj.score(b, pick) >= rj.score(b, r2.pick_bayes(b)) - 0.002
    assert calls.mean() < 4 * rj.K * 2 + 0.05 * rj.K * r2.M * 2


# ---------------- registered claims (BAR.md) ----------------

def _eval():
    return json.loads((HERE / "eval.json").read_text())


def _ns(r, a):
    return (r[a] - r["blind"]) / (r["bayes"] - r["blind"])


def _band(E):
    rows = lambda T: [next(x for x in E["seeds"][s] if x["T"] == T) for s in E["seeds"]]
    ok = []
    for T in r2.BAND:
        rs = rows(T); m = lambda k: np.mean([x[k] for x in rs])
        if m("bayes") - m("floor") >= 0.03 and m("bayes") - m("blind") >= 0.10 and max(x["null_max_dev"] for x in rs) <= 0.01:
            ok.append(T)
    return ok, rows


def test_bed_valid_on_new_seeds():
    E = _eval()
    assert sorted(E["seeds"]) == ["3", "4", "5"]
    ok, _ = _band(E)
    assert len(ok) >= 3, ok
    for s in E["seeds"]:
        assert min(E["gate"][s]["dj"], E["gate"][s]["djl"]) >= 0.90


def test_claim_A1_lin1_beats_djepa_spec_each_seed():
    E = _eval(); band, _ = _band(E)
    for s, rs in E["seeds"].items():
        rs = [x for x in rs if x["T"] in band]
        bdj = [max(_ns(x, "dj"), _ns(x, "djl")) for x in rs]
        l1 = [_ns(x, "lin1") for x in rs]
        assert np.mean(l1) >= np.mean(bdj) + 0.05, (s, np.mean(l1), np.mean(bdj))
        assert all(a >= b - 0.01 for a, b in zip(l1, bdj)), (s, l1, bdj)


def test_claim_A2_lin1_beats_floor_each_seed():
    E = _eval(); band, _ = _band(E)
    for s, rs in E["seeds"].items():
        rs = [x for x in rs if x["T"] in band]
        assert np.mean([_ns(x, "lin1") for x in rs]) >= np.mean([_ns(x, "floor") for x in rs]) + 0.05


def test_claim_A3_lin1_holds_against_ens3():
    E = _eval(); band, rows = _band(E)
    for T in band:
        rs = rows(T)
        assert np.mean([_ns(x, "lin1") for x in rs]) >= np.mean([_ns(x, "ens3") for x in rs]) - 0.01, T


def test_claim_B_cascade_4x_fewer_calls_at_bayes_quality():
    E = _eval(); band, _ = _band(E)
    assert E["cascade_config"] == json.loads((HERE / "tune.json").read_text())["chosen"]
    for s, rs in E["seeds"].items():
        for x in rs:
            if x["T"] in band:
                assert _ns(x, "cascade") >= 0.99, (s, x["T"], _ns(x, "cascade"))
                assert x["cascade_calls"] <= rj.K * (r2.M + 1) * x["T"] / 4, (s, x["T"], x["cascade_calls"])
                assert abs(x["cascade_null"] - x["blind"]) <= 0.01


def test_bernstein_race_without_pruning_is_bayes(b12):
    pick, calls = r2.cascade(b12, INF, INF, INF, 16, bern=True)
    assert np.array_equal(pick, r2.pick_bayes(b12))


def test_claim_B2_cascade2_4x_fewer_calls_at_bayes_quality():
    E = _eval(); band, _ = _band(E)
    assert E["cascade2_config"] == json.loads((HERE / "tune2.json").read_text())["chosen"]
    for s, rs in E["seeds"].items():
        for x in rs:
            if x["T"] in band:
                assert _ns(x, "cascade2") >= 0.99, (s, x["T"], _ns(x, "cascade2"))
                assert x["cascade2_calls"] <= rj.K * (r2.M + 1) * x["T"] / 4, (s, x["T"], x["cascade2_calls"])
                assert abs(x["cascade2_null"] - x["blind"]) <= 0.01

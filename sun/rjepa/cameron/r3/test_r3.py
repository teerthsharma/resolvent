# pytest test_r3.py -- implementation tests + the registered claims D, E, F of BAR.md (read from tune/leadeps/eval/long.json).
import json, sys
from pathlib import Path
import numpy as np
import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import r3
from r3 import r2, rj

INF = np.inf
FULL = lambda T: rj.K * (r2.M + 1) * T


def test_cache_lives_under_r3_and_r2_is_untouched():
    assert r3.cache_path(10, 12).parent == HERE / "cache"
    assert r2.HERE == HERE.parent / "r2"


def test_pick_eps_rule():
    rows = [dict(eps=0.05, ns=0.999, p90=2.9), dict(eps=0.075, ns=0.997, p90=3.4),
            dict(eps=0.1, ns=0.996, p90=4.0), dict(eps=0.125, ns=0.990, p90=5.0)]
    assert r3.pick_eps(rows) == 0.075                       # smallest reaching 3.3 with NS >= 0.995
    rows = [dict(eps=0.05, ns=0.999, p90=2.0), dict(eps=0.1, ns=0.996, p90=2.8), dict(eps=0.2, ns=0.98, p90=9.0)]
    assert r3.pick_eps(rows) == 0.1                         # none reaches 3.3: best p90 among NS >= 0.995


def test_b3_is_r2_cascade_with_the_lead_eps():
    b = r2.batch(np.random.default_rng([95, 12, 0]), 200, 12)
    cfg = dict(zc=INF, kappa=INF, c=1.5, b=8, tau=r2.R / 2, eps=0.05)
    p3, c3 = r3.b3(b, cfg, {12: 0.1})
    p2, c2 = r2.cascade(b, INF, INF, 1.5, 8, tau=r2.R / 2, eps=0.1, bern=True)
    assert np.array_equal(p3, p2) and np.array_equal(c3, c2)


# ---------------- registered claims ----------------

def _j(n):
    return json.loads((HERE / n).read_text())


def _ns(r, a):
    return (r[a] - r["blind"]) / (r["bayes"] - r["blind"])


def _valid(E, leads, arms):
    rows = lambda T: [next(x for x in E["seeds"][s] if x["T"] == T) for s in E["seeds"]]
    ok = []
    for T in leads:
        rs = rows(T); m = lambda k: np.mean([x[k] for x in rs])
        if m("bayes") - m("floor") >= 0.03 and m("bayes") - m("blind") >= 0.10 and \
                all(abs(x[a + "_null"] - x["blind"]) <= 0.01 for x in rs for a in ["floor", "bayes", "lin1", "ens3"] + arms):   # T1
            ok.append(T)
    return ok, rows


def test_tune_is_seed10_with_the_round2_rule():
    t = _j("tune.json")
    assert t["seed"] == 10
    ok = [x for x in t["summary"] if x["min_ns"] >= 0.995]
    assert t["chosen"] == min(ok, key=lambda x: x["calls"])["cfg"]


def test_bed_valid():
    E = _j("eval.json")
    assert sorted(E["seeds"]) == ["6", "7", "8"]
    assert len(_valid(E, r2.BAND, ["b2p", "b3"])[0]) >= 3


def test_claim_D_b2prime_4x_at_bayes_quality_each_seed():
    E = _j("eval.json"); band, _ = _valid(E, r2.BAND, ["b2p", "b3"]); assert band
    assert E["b2p_config"] == _j("tune.json")["chosen"]
    for s, rs in E["seeds"].items():
        for x in rs:
            if x["T"] in band:
                assert _ns(x, "b2p") >= 0.99, (s, x["T"], _ns(x, "b2p"))
                assert x["b2p_calls"] <= FULL(x["T"]) / 4, (s, x["T"], FULL(x["T"]) / x["b2p_calls"])


def test_claim_E_lead_eps_p90_3x_each_seed():
    E = _j("eval.json"); band, _ = _valid(E, r2.BAND, ["b2p", "b3"]); assert band
    assert {int(k): v for k, v in E["lead_eps"].items()} == {int(k): v for k, v in _j("leadeps.json")["eps"].items()}
    for s, rs in E["seeds"].items():
        for x in rs:
            if x["T"] in band:
                assert _ns(x, "b3") >= 0.99, (s, x["T"], _ns(x, "b3"))
                assert FULL(x["T"]) / x["b3_calls_p90"] >= 3, (s, x["T"], FULL(x["T"]) / x["b3_calls_p90"])


def test_claim_F_lin1_holds_against_ens3_long_leads():
    L = _j("long.json"); leads, rows = _valid(L, [40, 48], [])
    assert sorted(L["seeds"]) == ["6", "7", "8"]
    assert leads, "F VOID: no valid long lead"
    for T in leads:
        rs = rows(T)
        assert np.mean([_ns(x, "lin1") for x in rs]) >= np.mean([_ns(x, "ens3") for x in rs]) - 0.01, T


def test_cap_stops_the_race_at_nmax():
    b = r2.batch(np.random.default_rng([94, 32, 0]), 300, 32)
    cfg = dict(zc=INF, kappa=INF, c=1.5, b=8, tau=r2.R / 2, eps=0.05)
    pk, c = r3.b3cap(b, cfg, {32: 0.05}, 80)
    assert c.max() <= rj.K * 32 * (3 + 80) and r2.M == 256
    p0, c0 = r3.b3cap(b, cfg, {32: 0.05}, 256)
    p1, c1 = r3.b3(b, cfg, {32: 0.05})
    assert np.array_equal(p0, p1) and np.array_equal(c0, c1)            # n_max = M is no cap


def test_claim_Eprime_cap80_each_seed():
    C = _j("cap.json"); E = _j("eval.json")
    assert sorted(C["seeds"]) == ["6", "7", "8"] and C["nmax"] == 80
    band, _ = _valid(E, r2.BAND, ["b2p", "b3"]); assert band
    for s, rs in C["seeds"].items():
        for x in rs:
            if x["T"] in band:
                assert _ns(x, "b3cap") >= 0.99, (s, x["T"], _ns(x, "b3cap"))
                assert FULL(x["T"]) / x["b3cap_calls_p90"] >= 3, (s, x["T"])
                assert abs(x["b3cap_null"] - x["blind"]) <= 0.01, (s, x["T"])


def test_budget_race_unbounded_is_r2_cascade_and_bounded_obeys_budget():
    b = r2.batch(np.random.default_rng([93, 32, 0]), 300, 32)
    p0, c0 = r2.cascade(b, INF, INF, 1.5, 8, tau=r2.R / 2, eps=0.05, bern=True)
    p1, c1 = r3.race_budget(b, 1.5, 8, 0.05, 10 ** 9)
    assert np.array_equal(p0, p1) and np.array_equal(c0, c1)
    _, c2 = r3.race_budget(b, 1.5, 8, 0.05, 640)
    assert c2.max() <= rj.K * 3 * 32 + 640 * 32


def test_claim_G_budget_race_each_fresh_seed():
    G = _j("budget.json")
    assert sorted(G["seeds"]) == ["11", "12", "13"] and G["B"] == 640
    ok, _ = _valid(G, r2.BAND, ["b4"]); assert ok
    for s, rs in G["seeds"].items():
        for x in rs:
            if x["T"] in ok:
                assert _ns(x, "b4") >= 0.99, (s, x["T"], _ns(x, "b4"))
                assert FULL(x["T"]) / x["b4_calls_p90"] >= 3 and FULL(x["T"]) / x["b4_calls"] >= 4, (s, x["T"])


def test_expected_success_score_reduces_to_realised_score():
    b = r2.batch(np.random.default_rng([92, 12, 0]), 200, 12); pk = r2.pick_bayes(b)
    assert np.isclose(r3.escore(b["succ"].astype(float), pk), rj.score(b, pk))


def test_claim_H_budget_race_under_sharp_instrument():
    H = _j("sharp.json")
    assert sorted(H["seeds"], key=int) == ["6", "7", "8", "11", "12", "13"] and H["M_ref"] == 1024
    for s, x in H["seeds"].items():
        assert x["ns_b4"] >= 0.99, (s, x["ns_b4"])

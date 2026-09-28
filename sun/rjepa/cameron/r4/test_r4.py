# pytest test_r4.py -- implementation tests + the registered claims B, B-route, R0, RM1, RM2, I of BAR.md.
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import r4
from r4 import r2, r3, rj

INF = np.inf
FULL = lambda T: rj.K * (r2.M + 1) * T
SEEDS = ["14", "15", "16"]
LEADS = [36, 40, 44, 48, 56]


# ---------------- implementation ----------------

def test_cache_lives_under_r4_and_r3_writes_there():
    assert r4.cache_path(14, 32).parent == HERE / "cache"
    assert r3.cache_path is r4.cache_path


def test_budget_656_is_the_registered_bound():
    assert r4.B == 656 and r4.B % 8 == 0 and FULL(32) / (3 * rj.K * 32 + r4.B * 32) >= 3
    assert FULL(32) / (3 * rj.K * 32 + (r4.B + 8) * 32) < 3             # largest such multiple of b
    b = r2.batch(np.random.default_rng([91, 32, 0]), 300, 32)
    _, c = r3.race_budget(b, 1.5, 8, 0.05, r4.B)
    assert c.max() <= 3 * rj.K * 32 + r4.B * 32


def test_member_split_disjoint_and_covering():
    e, c, r = r4.SPLIT
    idx = np.arange(r2.M)
    parts = [idx[e], idx[c], idx[r]]
    assert list(parts[0]) == [0, 1, 2] and parts[1][0] == 3 and parts[1][-1] == 127 and parts[2][0] == 128
    assert sorted(np.concatenate(parts)) == list(idx)


def test_regime_thresholds():
    s = np.array([0.0, rj.R / 2 - 1e-9, rj.R / 2, 2 * np.pi - 1e-9, 2 * np.pi, 1e9])
    assert list(r4.regime(s)) == [0, 0, 1, 1, 2, 2]


def test_z_formula_known_answer_and_calibration():
    N, Mr = 4000, 1024
    H = np.zeros((N, 2, Mr), bool); H[:, 0, :512] = True; H[:, 1, :256] = True      # P 0.5 / 0.25, joint 0.25
    pa, pb = np.zeros(N, int), np.ones(N, int)
    rng = np.random.default_rng(0); zs = []
    for _ in range(200):
        j = rng.integers(0, Mr, N)                                                 # x0 = one posterior member
        succ = H[np.arange(N), :, j]
        Dr, De, sd = r4.zstat(succ, H, pa, pb)
        assert np.isclose(De, 0.25) and np.isclose(sd, np.sqrt(N * 0.1875) / N)
        zs.append((Dr - De) / sd)
    assert 0.85 < np.sqrt(np.mean(np.square(zs))) < 1.15
    assert r4.zstat(succ, H, pa, pa)[2] == 0.0                                     # same pick: no noise


# ---------------- registered claims ----------------

def _j(n):
    return json.loads((HERE / n).read_text())


def _sharp():
    S = _j("sharp.json"); assert sorted(S["seeds"], key=int) == SEEDS and S["B"] == 656 and S["M_ref"] == 1024
    for s, x in S["seeds"].items():
        assert x["bayes_ref"] - x["blind_ref"] >= 0.10 and x["bayes_ref"] - x["floor_ref"] >= 0.03, ("B VOID", s)
    return S


def test_claim_B_b4_656_each_fresh_seed():
    for s, x in _sharp()["seeds"].items():
        assert x["ns_b4_656"] >= 0.99, (s, x["ns_b4_656"])
        assert x["p90ratio_b4_656"] >= 3, (s, x["p90ratio_b4_656"])


def test_claim_Broute_b4_656_beats_half_rollout_each_fresh_seed():
    for s, x in _sharp()["seeds"].items():
        assert x["ns_b4_656"] >= x["ns_half"], (s, x["ns_b4_656"], x["ns_half"])


def test_claim_I_realised_noise_bound():
    S = _sharp(); z = [x["z"][a] for x in S["seeds"].values() for a in r4.I_ARMS]
    assert len(z) == 24
    rms = float(np.sqrt(np.mean(np.square(z))))
    assert 0.6 <= rms <= 1.5 and max(abs(v) for v in z) <= 3.5, (rms, max(abs(v) for v in z))
    for s, x in S["seeds"].items():
        assert 0.005 <= x["sdns_b4_656"] <= 0.02, (s, x["sdns_b4_656"])


def _rev():
    Rv = _j("rev.json"); assert sorted(Rv["seeds"], key=int) == SEEDS
    rows = {T: [next(r for r in Rv["seeds"][s] if r["T"] == T) for s in SEEDS] for T in LEADS}
    m = lambda T, k: np.mean([r[k] for r in rows[T]])
    valid = [T for T in LEADS if m(T, "ceil_ref") - m(T, "blind_ref") >= 0.10 and m(T, "ceil_ref") - m(T, "floor_ref") >= 0.03]
    gap = {T: m(T, "ns_lin1") - m(T, "ns_ens3") for T in LEADS}
    return Rv, rows, valid, gap


def test_rev_decomposition_is_exact():
    _, rows, _, _ = _rev()
    for T in LEADS:
        for r in rows[T]:
            assert sum(r["n"]) == r["N"]
            assert np.isclose(sum(r["gsum"]) / r["N"], r["ns_lin1"] - r["ns_ens3"], atol=1e-9)


def test_claim_R0_reversal_replicates():
    _, _, valid, gap = _rev()
    for T in (36, 40):
        if T in valid:
            assert gap[T] < 0, (T, gap[T])
    for T in (48, 56):
        if T in valid:
            assert gap[T] > 0, (T, gap[T])
    assert any(T in valid for T in (36, 40)) and any(T in valid for T in (48, 56)), ("R0 VOID", valid)


def _pooled(rows):
    n = np.sum([r["n"] for T in LEADS for r in rows[T]], 0); gs = np.sum([r["gsum"] for T in LEADS for r in rows[T]], 0)
    return n, gs / np.maximum(n, 1)


def test_claim_RM1_sign_pattern():
    _, rows, _, _ = _rev(); n, g = _pooled(rows)
    assert n[1] >= 300 and g[1] < 0, ("F", n[1], g[1])
    assert n[2] >= 300 and g[2] > 0, ("X", n[2], g[2])
    if n[0] >= 300:
        assert g[0] >= 0, ("S", n[0], g[0])


def test_claim_RM2_composition_explains_lead_dependence():
    _, rows, valid, gap = _rev(); _, g = _pooled(rows); assert valid
    for T in valid:
        w = np.sum([r["n"] for r in rows[T]], 0) / np.sum([r["N"] for r in rows[T]])
        pred = float(w @ g)
        assert abs(pred - gap[T]) <= 0.01, (T, pred, gap[T])
        if abs(gap[T]) >= 0.01:
            assert np.sign(pred) == np.sign(gap[T]), (T, pred, gap[T])

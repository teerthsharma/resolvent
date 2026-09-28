# Deciding-claim tests (BAR.md: V1, V2, P1, P2, K1). Read result files from $R2_RES (default results/).
# RED first against a control directory where every arm equals dj02 and Bayes equals dist.
import json, os
from pathlib import Path
import pytest

RES = Path(os.environ.get("R2_RES", Path(__file__).parent / "results"))
ARMS = ["dist", "lin1", "point", "hop1", "dj02", "dj4L", "dj02L", "dj4"]


def cell(f, q):
    rs = [json.loads((RES / f"res_f{f}_q{q}_s{s}.json").read_text()) for s in (0, 1, 2)
          if (RES / f"res_f{f}_q{q}_s{s}.json").exists()]
    if len(rs) < 3:
        pytest.skip(f"cell f={f} q={q}: {len(rs)}/3 seeds")
    return rs


def mean(rs, key, arm):
    return sum(r[key][arm] for r in rs) / len(rs)


def test_V1_gap_exists():
    rs = cell(1.0, 3.0)
    assert mean(rs, "hit", "bayes") - mean(rs, "hit", "dist") >= 0.02


@pytest.mark.parametrize("fq", [(0.0, 1.0), (0.0, 3.0), (1.0, 3.0)])            # A3 adds (0, 1)
def test_V2_null_does_not_beat_floor(fq):
    for r in cell(*fq):
        p = r["hit"]["dist"]
        assert r["hit"]["null"] <= p + 3 * (p * (1 - p) / r["n_eval"]) ** 0.5


def test_P1_f0_pointwise_ties_set():
    rs = cell(0.0, 1.0)                                                            # A3: read at sigma/rho = 1
    best_set = max(mean(rs, "ns", "hop1"), mean(rs, "ns", "dj4L"))
    assert abs(mean(rs, "ns", "point") - best_set) <= 0.01
    assert abs(mean(rs, "ns", "lin1") - best_set) <= 0.01


@pytest.mark.parametrize("fq", [(1.0, 3.0), (0.75, 3.0), (1.0, 10.0), (0.75, 10.0)])
@pytest.mark.parametrize("arm", ["dj4L", "hop1"])
def test_P2_arm_beats_dj02(fq, arm):
    for r in cell(*fq):
        assert r["ns"][arm] - r["ns"]["dj02"] >= 0.05, (r["seed"], r["ns"][arm], r["ns"]["dj02"])


def test_K1_trust_region_claim_survives():
    """Kill fires if dj02 is within 0.01 NS of the best arm at (f=1, 3) on 3/3 seeds."""
    rs = cell(1.0, 3.0)
    close = [max(r["ns"][a] for a in ARMS if a in r["ns"]) - r["ns"]["dj02"] <= 0.01 for r in rs]
    assert not all(close)


def test_B10_replicates_at_K63_on_fresh_seeds():
    """Registered 02:27 after the seed-77 sweep: on fresh seeds 78 and 79 (4,000 starts each, exact model),
    Bayes - dist <= 0.02 at f = 0 for sigma in {0.37, 1.14}, and >= 0.03 at f = 1, sigma = 1.14."""
    for sd in (78, 79):
        p = RES / f"sweep_s{sd}.json"
        if not p.exists():
            pytest.skip(f"{p.name} missing")
        rows = {(r["f"], round(r["sigma"], 2)): r["bayes"] - r["dist"] for r in json.loads(p.read_text())}
        assert rows[(0.0, 0.37)] <= 0.02 and rows[(0.0, 1.14)] <= 0.02, rows
        assert rows[(1.0, 1.14)] >= 0.03, rows


def _seeds(f, q, seeds, sfx=""):
    out = []
    for s in seeds:
        p = RES / f"res_f{f}_q{q}_s{s}{sfx}.json"
        if not p.exists():
            pytest.skip(f"{p.name} missing")
        out.append(json.loads(p.read_text()))
    return out


def test_M1_reach_is_not_the_wall():
    for r in _seeds(1.0, 3.0, (1, 2)):
        assert r["ns"]["bayes_eps02"] >= 0.90, (r["seed"], r["ns"]["bayes_eps02"])


def test_M2_bound_costs_learning():
    for r in _seeds(1.0, 3.0, (1, 2)):
        assert r["ns"]["dj4"] - r["ns"]["dj02"] >= 0.05, (r["seed"], r["ns"]["dj4"], r["ns"]["dj02"])


def test_M3_bound_cost_is_optimisation_speed():
    base = _seeds(1.0, 3.0, (0, 1, 2))
    ext = _seeds(1.0, 3.0, (0, 1, 2), "_extra")
    for r, x in zip(base, ext):
        assert x["ns"]["dj02f"] - r["ns"]["dj02"] >= 0.5 * (r["ns"]["dj4"] - r["ns"]["dj02"]), (r["seed"], x["ns"]["dj02f"])


def _block_ns_diff_sd(pk, a="dj4L", b="dj02", n=2000):
    """SD over disjoint n-start eval blocks of NS(a) - NS(b), NS computed per block (its own Bayes hit), as at n_eval = n."""
    import torch
    K = 63
    hv = {k: (v == pk["best"]).float() for k, v in pk.items() if k != "best"}
    diffs = []
    for i in range(0, len(pk["best"]) - n + 1, n):
        den = hv["bayes"][i:i + n].mean() - 1 / K
        diffs.append(float((hv[a][i:i + n].mean() - hv[b][i:i + n].mean()) / den))
    return float(torch.tensor(diffs).std()), len(diffs)


def test_A3r_registered_n2000_cannot_resolve_P2_bar():
    """Re-stated A3 (02:55, post-run on existing picks): at the registered n_eval = 2,000, the SD across disjoint
    2,000-start blocks of NS(dj4L) - NS(dj02) at (f=1, 3) is >= 0.05 (the P2 bar itself) on every seed."""
    import torch
    for s in (0, 1, 2):
        p = RES / f"picks_f1.0_q3.0_s{s}.pt"
        if not p.exists():
            pytest.skip(f"{p.name} missing")
        sd, nb = _block_ns_diff_sd(torch.load(p))
        assert nb >= 10 and sd >= 0.05, (s, sd, nb)


@pytest.mark.parametrize("arm", ["dj4L", "hop1"])
def test_F1_set_edge_vanishes_at_f0(arm):
    """A5 F1 (02:58, pre-run): at (0, 3), mean over 3 seeds of hit(arm) - hit(dj02) <= 0.005."""
    rs = cell(0.0, 3.0)
    assert mean(rs, "hit", arm) - mean(rs, "hit", "dj02") <= 0.005, (mean(rs, "hit", arm), mean(rs, "hit", "dj02"))


def test_M3b_lr20_probe_kills_correction():
    """A5 M3b (03:03, after seed 0): on seeds 1 and 2, dj02f |delta|/eps < 0.01 and NS(dj02f) within 0.01 of NS(dist)."""
    for x in _seeds(1.0, 3.0, (1, 2), "_extra"):
        assert x["delta_over_eps"]["dj02f"] < 0.01 and abs(x["ns"]["dj02f"] - x["ns"]["dist"]) <= 0.01, x["seed"]


def test_M3b_dj02_near_saturation_and_reproducible():
    base = _seeds(1.0, 3.0, (1, 2))
    for r, x in zip(base, _seeds(1.0, 3.0, (1, 2), "_extra")):
        assert x["delta_over_eps"]["dj02"] >= 0.6, (x["seed"], x["delta_over_eps"])
        assert abs(x["ns"]["dj02"] - r["ns"]["dj02"]) <= 0.02, (x["seed"], x["ns"]["dj02"], r["ns"]["dj02"])


def test_S1_saturation_eps_sweep():
    """A5 S1 (03:43, pre-run): at (1, 3), dj1 is less saturated than dj02 and beats it by >= 0.05 NS on 3/3 seeds."""
    base = _seeds(1.0, 3.0, (0, 1, 2))
    ext = _seeds(1.0, 3.0, (0, 1, 2), "_extra")
    for r, x, e in zip(base, ext, _seeds(1.0, 3.0, (0, 1, 2), "_eps")):
        assert e["delta_over_eps"]["dj1"] < x["delta_over_eps"]["dj02"], (r["seed"], e["delta_over_eps"])
        assert e["ns"]["dj1"] - r["ns"]["dj02"] >= 0.05, (r["seed"], e["ns"]["dj1"], r["ns"]["dj02"])

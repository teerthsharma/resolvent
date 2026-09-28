# A7 claim tests (r4/foreman/BAR.md A7.1 P2n frozen, V; A7.2 L3; A7.0 provenance). Read $R4_RES (default results/).
# RED first on a control dir.
import hashlib, json, os
from pathlib import Path
import pytest

HERE = Path(__file__).parent
RES = Path(os.environ.get("R4_RES", HERE / "results"))
CELLS = [0.9, 0.8]


def load(pat, seeds=(0, 1, 2)):
    ps = [RES / pat.format(s=s) for s in seeds]
    if not all(p.exists() for p in ps):
        pytest.skip(f"{pat}: {sum(p.exists() for p in ps)}/3 seeds")
    rs = [json.loads(p.read_text()) for p in ps]
    now = hashlib.sha256((HERE / "r4.py").read_bytes()).hexdigest()
    for r in rs:                                                                       # A7.0: no stale or unhashed result
        assert r.get("sha256_r4_py") == now and len(r.get("sha256_bar_r4", "")) == 64, (pat, r.get("seed"))
    return rs


@pytest.mark.parametrize("f", CELLS)
def test_P2n_frozen_edge_iff_closable(f):
    rs = load(f"res_f{f}_q2.0_s{{s}}.json")
    closable = sum(1 - r["ns"]["dist"] for r in rs) / 3
    edge = all(r["ns"]["dj4L"] - r["ns"]["dj02"] >= 0.05 for r in rs)
    assert edge == (closable >= 0.25), (f, round(closable, 3), [round(r["ns"]["dj4L"] - r["ns"]["dj02"], 3) for r in rs])


@pytest.mark.parametrize("f", CELLS)
def test_V_null_does_not_beat_floor(f):
    for r in load(f"res_f{f}_q2.0_s{{s}}.json"):
        p = r["hit"]["dist"]
        assert r["hit"]["null"] <= p + 3 * (p * (1 - p) / r["n_eval"]) ** 0.5, r["seed"]


def test_A72_reproduces_r3_audit():
    for r in load("lin2_f0.0_q1.0_s{s}.json"):
        assert r["reproduces_r3"], r["seed"]


def test_L3_lin2_closes_half_the_residue():
    for r in load("lin2_f0.0_q1.0_s{s}.json"):
        ns = r["ns"]
        assert ns["lin2_true"] - ns["lin1_true"] >= 0.5 * (ns["bayes_succ"] - ns["lin1_true"]), \
            (r["seed"], round(ns["lin1_true"], 4), round(ns["lin2_true"], 4), round(ns["bayes_succ"], 4))


def test_P2r_edge_proportional_to_closable():
    """A7.3 (21:41 box time, after (0.9, 2) read, before any (0.8, 2) seed): beta frozen from r3's 9 cell-seeds."""
    import torch
    for r in load("res_f0.8_q2.0_s{s}.json"):
        pk = torch.load(RES / f"picks_f0.8_q2.0_s{r['seed']}.pt")
        dv = (pk["dj4L"] == pk["best"]).float() - (pk["dj02"] == pk["best"]).float()
        se = float(dv.std() / len(dv) ** 0.5) / (r["hit"]["bayes"] - 1 / 63)
        edge, pred = r["ns"]["dj4L"] - r["ns"]["dj02"], 0.4557 * (1 - r["ns"]["dist"])
        assert abs(edge - pred) <= 2 * se, (r["seed"], round(edge, 3), round(pred, 3), round(se, 3))


def test_P2a_edge_affine_in_closable():
    """A7.4 (after A7.1 read, before any (0.85, 2) seed): a, b frozen from 15 cell-seeds."""
    import torch
    for r in load("res_f0.85_q2.0_s{s}.json"):
        pk = torch.load(RES / f"picks_f0.85_q2.0_s{r['seed']}.pt")
        dv = (pk["dj4L"] == pk["best"]).float() - (pk["dj02"] == pk["best"]).float()
        se = float(dv.std() / len(dv) ** 0.5) / (r["hit"]["bayes"] - 1 / 63)
        edge, pred = r["ns"]["dj4L"] - r["ns"]["dj02"], -0.0683 + 0.6618 * (1 - r["ns"]["dist"])
        assert abs(edge - pred) <= 2 * se, (r["seed"], round(edge, 3), round(pred, 3), round(se, 3))

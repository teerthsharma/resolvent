# A6 claim tests (BAR.md A6.2 P2c, V; A6.3 L1, L2). Read $R3_RES (default results/). RED first on a control dir.
import json, os
from pathlib import Path
import pytest

RES = Path(os.environ.get("R3_RES", Path(__file__).parent / "results"))


def load(pat, seeds=(0, 1, 2)):
    ps = [RES / pat.format(s=s) for s in seeds]
    if not all(p.exists() for p in ps):
        pytest.skip(f"{pat}: {sum(p.exists() for p in ps)}/3 seeds")
    return [json.loads(p.read_text()) for p in ps]


@pytest.mark.parametrize("f", [0.85, 0.9, 0.95])
def test_P2c_edge_iff_closable_gap(f):
    rs = load(f"res_f{f}_q3.0_s{{s}}.json")
    gap = sum(r["hit"]["bayes"] - r["hit"]["dist"] for r in rs) / 3
    edge = all(r["ns"]["dj4L"] - r["ns"]["dj02"] >= 0.05 for r in rs)
    assert edge == (gap >= 0.02), (f, round(gap, 4), [round(r["ns"]["dj4L"] - r["ns"]["dj02"], 3) for r in rs])


@pytest.mark.parametrize("f", [0.85, 0.9, 0.95])
def test_V_null_does_not_beat_floor(f):
    for r in load(f"res_f{f}_q3.0_s{{s}}.json"):
        p = r["hit"]["dist"]
        assert r["hit"]["null"] <= p + 3 * (p * (1 - p) / r["n_eval"]) ** 0.5, r["seed"]


def test_L1_lin1_shortfall_is_linearisation():
    for r in load("audit_f0.0_q1.0_s{s}.json"):
        assert r["ns_new"]["bayes_succ"] - r["ns_new"]["lin1_true"] >= 0.02, (r["seed"], r["ns_new"]["lin1_true"])


def test_L2_predictor_costs_lin1_little():
    for r in load("audit_f0.0_q1.0_s{s}.json"):
        assert abs(r["ns_new"]["lin1_true"] - r["ns_new"]["lin1"]) <= 0.01, (r["seed"], r["ns_new"]["lin1_true"], r["ns_new"]["lin1"])


def test_A61_audit_reproduces_r2():
    for pat in ["audit_f0.0_q1.0_s{s}.json", "audit_f1.0_q3.0_s{s}.json", "audit_f0.75_q3.0_s{s}.json",
                "audit_f0.0_q3.0_s{s}.json", "audit_f1.0_q1.0_s{s}.json"]:
        for r in load(pat):
            assert r["old_tiebreak_reproduces_r2"], (pat, r["seed"])


@pytest.mark.parametrize("f", [0.85, 0.9, 0.95])
def test_P2n_edge_iff_closable_ns(f):
    """A6.4 (20:03, after (0.9, 3) seed 0 only)."""
    rs = load(f"res_f{f}_q3.0_s{{s}}.json")
    closable = sum(1 - r["ns"]["dist"] for r in rs) / 3
    edge = all(r["ns"]["dj4L"] - r["ns"]["dj02"] >= 0.05 for r in rs)
    assert edge == (closable >= 0.25), (f, round(closable, 3), [round(r["ns"]["dj4L"] - r["ns"]["dj02"], 3) for r in rs])
# CORRECTION 2026-09-28 21:18 box time (r4 Foreman, Inspector r3 note): the '20:03' in test_P2n's docstring is wrong; A6.4 was registered 19:58:55 box time (see r2/BAR.md A6 label correction and r4/foreman/BAR.md).

"""Every number in the README's results table, recomputed by the command the README prints beside it."""
import json

import pytest

from rjepa import reproduce


def test_gap_matches_readme():
    """Bayes - latent-distance hit on bed `shift`, K = 4, 3 seeds x 4,000 starts."""
    out = reproduce.gap()
    want = {0.1: 0.000, 1.0: 0.024, 3.0: 0.149, 10.0: 0.213, 30.0: 0.241}
    assert {s: round(out[s]["gap"], 3) for s in want} == want, out


def test_bound_matches_readme():
    """D-JEPA's Cor 1 reach (eps = 0.2, K = 4: distance ranks {0, 1}) caps any operator below Bayes at sigma = 10."""
    out = reproduce.bound()
    assert round(out[10.0]["bounded"], 3) == 0.289 and round(out[10.0]["bayes"], 3) == 0.381, out
    assert out[1.0]["bayes"] - out[1.0]["bounded"] <= 0.01, out


def test_cli_prints_json(capsys):
    reproduce.main(["gap", "--sigmas", "0.1", "--seeds", "0", "--n", "200"])
    out = json.loads(capsys.readouterr().out)
    assert set(out) == {"0.1"} and set(out["0.1"]) == {"bayes", "dist", "gap"}


@pytest.mark.slow
def test_closure_bars_sigma1():
    """Round-1 bars B4-B7 at sigma = 1, 36 epochs, 3 seeds (minutes of CPU): the set head learns the
    Bayes rule and closes the gap, the pointwise head cannot, and the Neumann tail adds nothing over one hop."""
    c = reproduce.closure()
    assert c["agree"]["resolvent"] >= 0.90 and c["closure"]["resolvent"] >= 0.80, c
    assert c["closure"]["point"] <= 0.50, c
    assert abs(c["hit"]["resolvent"] - c["hit"]["hop1"]) < 0.01, c

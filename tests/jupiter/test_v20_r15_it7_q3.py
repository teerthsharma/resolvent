"""Q3 LEARNABILITY, it.7. Four data conditions, asserted on the record.

PLANTED NEGATIVE: set JUP_IT7_MUTATE to one of the mutation names below and
every node that the mutation can reach must go RED. The mutation is applied to
the LOADED CELLS, never to the journals, which are read-only here.

  fga_law     -- flip one cell's frac_gate_annihilated to 0.0 while leaving its
                 lambda_hat at -inf; the "-inf iff a gate died" law must break.
  qk_band     -- move seed 2's qk into the failing band; the partition must break.
  live_band   -- flatten seed 2's lambda_hat_live to the failing band's value.
  pl_sign     -- flip one arm_pl cell's lambda_hat sign.
"""
import json
import os
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
RETAKE = ROOT / "results" / "v17k_r4_retake.jsonl"
FRESH = ROOT / "results" / "v20_r15_it6_seeds8_15.jsonl"
NEG_INF = float("-inf")
MUT = os.environ.get("JUP_IT7_MUTATE", "")


def _cells(path, kind=None):
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("t") == "cell" and (kind is None or rec["kind"] == kind):
            out.append(rec)
    return out


@pytest.fixture(scope="module")
def w1():
    """16 distinct arm_smprime cells; the fresh journal's seeds 0/1 are the
    reproduction control and are the same cell as the retake's, not a 17th."""
    by_seed = {c["seed"]: c for c in _cells(RETAKE, "arm_smprime")}
    for c in _cells(FRESH, "arm_smprime"):
        by_seed.setdefault(c["seed"], c)
    cells = [json.loads(json.dumps(c)) for _, c in sorted(by_seed.items())]
    if MUT == "fga_law":
        cells[0]["frac_gate_annihilated"] = 0.0
    if MUT == "qk_band":
        next(c for c in cells if c["seed"] == 2)["manifest"]["smp_values"]["qk"] = 1.3
    if MUT == "live_band":
        next(c for c in cells if c["seed"] == 2)["lambda_hat_live"] = -0.01
    return cells


@pytest.fixture(scope="module")
def w3():
    cells = [json.loads(json.dumps(c)) for c in _cells(RETAKE, "arm_pl")]
    if MUT == "pl_sign":
        cells[0]["lambda_hat"] = +1.0
    return cells


def test_w1_is_sixteen_distinct_cells(w1):
    assert [c["seed"] for c in w1] == list(range(16))


def test_lambda_hat_is_minus_inf_exactly_when_a_gate_died(w1):
    """32 cells across three arms, zero exceptions. The unit-root flag is NOT
    in this law: seeds 3, 11, 13 have unit_root False and -inf anyway, because
    scripts/v15_r1.py:383 takes the MEAN of log(m) and one m_k == 0 is enough."""
    pool = list(w1) + _cells(RETAKE, "arm_pl") + _cells(RETAKE, "softmax")
    assert len(pool) == 32
    bad = [(c["kind"], c["seed"]) for c in pool
           if (c["frac_gate_annihilated"] > 0.0) != (c["lambda_hat"] == NEG_INF)]
    assert bad == []


def test_w1_outcome_partitions_by_trained_qk_with_no_overlap(w1):
    """qk is the coordinate that orders W1's sixteen cells. beta is not."""
    qk = {c["seed"]: c["manifest"]["smp_values"]["qk"] for c in w1}
    cross = [c["seed"] for c in w1 if c["eval_nrmse"] <= 0.7071067811865476]
    assert cross == [2]
    plain = [c["seed"] for c in w1
             if c["frac_gate_annihilated"] == 0.5032958984375]
    high = [c["seed"] for c in w1 if c["frac_gate_annihilated"] > 0.9]
    assert len(plain) == 12 and sorted(high) == [11, 13]
    assert max(qk[s] for s in cross) < min(qk[s] for s in plain)
    assert max(qk[s] for s in plain) < min(qk[s] for s in high)


def test_w1_relaxation_rate_is_a_WINDOW_not_an_extreme(w1):
    """lambda_hat_live, the column beside the saturated one. The crossing cell
    decays; the thirteen plain failures do not relax at all; the two new gate
    states over-decay. The good regime is INTERIOR."""
    live = {c["seed"]: c["lambda_hat_live"] for c in w1}
    flat = [c["seed"] for c in w1
            if c["frac_gate_annihilated"] == 0.5032958984375]
    assert len(flat) == 12
    assert all(abs(live[s]) < 0.05 for s in flat)
    assert -1.0 < live[2] < -0.5
    assert all(live[s] < -2.0 for s in (11, 13))
    #: seed 3 (fga 0.4967041015625) is a FOURTH state and it is NOT explained:
    #: lambda_hat_live -0.4411, more than half way from the flat band to the
    #: crossing cell's rate, and it does not cross. The window is necessary on
    #: these sixteen cells and this cell shows it is not sufficient.
    assert -0.5 < live[3] < -0.4 and w1[3]["eval_nrmse"] > 0.9


def test_w3_lambda_hat_sign_separates_all_eight_arm_pl_cells(w3):
    """M2's open-range divergence, as a data condition with zero overlap."""
    assert len(w3) == 8
    for c in w3:
        assert (c["lambda_hat"] < 0) == (c["eval_nrmse"] < 0.7), (c["seed"], c["lambda_hat"])
    assert max(c["eval_nrmse"] for c in w3 if c["lambda_hat"] < 0) < 0.67
    assert min(c["eval_nrmse"] for c in w3 if c["lambda_hat"] > 0) > 1.11

"""JUPITER it.8 - Q4 cost law and Q5 information floor, both wings.

Planted negatives via JUP_IT8_MUTATE, applied to LOADED cells only, never to the
journals on disk.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[2]
RETAKE = ROOT / "results" / "v17k_r4_retake.jsonl"
IT6 = ROOT / "results" / "v20_r15_it6_seeds8_15.jsonl"
MUT = os.environ.get("JUP_IT8_MUTATE", "")


def _rows(p):
    return [json.loads(l) for l in p.open() if l.strip()]


def _cells():
    out = []
    for p in (RETAKE, IT6):
        for r in _rows(p):
            if "cell" in r:
                r = dict(r)
                r["_src"] = p.name
                out.append(r)
    if MUT == "seq_len":            # pretend one cell ran at a different length
        out[0]["s"] = 128
    if MUT == "floor_formula":      # break the sqrt((t*-1)/t*) identity
        out[0]["floor_1"] = 0.75
    if MUT == "h_hat_identity":     # break h_hat = t*(1 - nrmse^2)
        out[0]["eval_h_hat"] = 0.0
    if MUT == "pl_dense":           # pretend arm_pl induced sparsity
        for r in out:
            if r["cell"].startswith("arm_pl"):
                r["frac_gate_annihilated"] = 0.4
                break
    if MUT == "cost_point":         # collapse the cost spread to a point
        for r in out:
            if r["cell"].startswith("arm_smprime"):
                r["secs"] = 16.161
    return out


def _by_arm(cells):
    d = {}
    for r in cells:
        d.setdefault(r["cell"].split(":")[0], []).append(r)
    return d


# ---------------------------------------------------------------- Q4


def test_q4_sequence_length_never_varies_so_the_cost_exponent_is_unidentified():
    cells = _cells()
    assert len(cells) == 34          # 32 distinct + seeds 0,1 measured twice
    assert {r["s"] for r in cells} == {64}, "cost-vs-length needs >1 length"
    assert {r["t_star"] for r in cells} == {2}
    assert {r["steps"] for r in cells} == {150}
    assert {r["n_train"] for r in cells} == {2048}
    assert {r["n_eval"] for r in cells} == {4096}


def test_q4_cost_is_a_RANGE_not_a_point_on_every_arm():
    a = _by_arm(_cells())
    got = {k: (min(r["secs"] for r in v), max(r["secs"] for r in v)) for k, v in a.items()}
    assert got["arm_smprime"] == (14.852, 22.069)
    assert got["arm_pl"] == (1.726, 1.916)
    assert got["softmax"] == (1.66, 1.749)
    # the spread the brief names, re-derived from the it.6 process alone
    it6 = [r["secs"] for r in a["arm_smprime"] if r["_src"].startswith("v20_r15_it6")]
    assert abs(max(it6) / min(it6) - 1.3874) < 5e-4
    # a point estimate is refused: the full-record spread exceeds it
    assert max(r["secs"] for r in a["arm_smprime"]) / min(r["secs"] for r in a["arm_smprime"]) > 1.48


def test_q4_repeat_control_bounds_measurement_noise_far_below_the_spread():
    """seeds 0 and 1 ran twice, bitwise-identical result, different wall clock."""
    a = _by_arm(_cells())
    sm = {(r["seed"], r["_src"]): r for r in a["arm_smprime"]}
    pairs = []
    for seed in (0, 1):
        rs = [r for (s, _), r in sm.items() if s == seed]
        assert len(rs) == 2
        assert rs[0]["eval_nrmse"] == rs[1]["eval_nrmse"]          # bitwise
        assert rs[0]["frac_gate_annihilated"] == rs[1]["frac_gate_annihilated"]
        pairs.append(abs(rs[0]["secs"] - rs[1]["secs"]) / min(r["secs"] for r in rs))
    assert max(pairs) < 0.005, pairs        # <0.5% repeat noise
    assert max(pairs) * 77 < 0.3874         # 77x smaller than the 38.74% spread


def test_q4_neither_arm_converts_its_sparsity_into_cost():
    cells = _cells()
    a = _by_arm(cells)
    # arm_pl induces exactly zero gate sparsity on 8 of 8
    assert [r["frac_gate_annihilated"] for r in a["arm_pl"]] == [0.0] * 8
    # arm_smprime annihilates up to 99.5% of the gate and still costs ~9.6x softmax
    assert max(r["frac_gate_annihilated"] for r in a["arm_smprime"]) > 0.99
    npar = {k: {r["n_params"] for r in v} for k, v in a.items()}
    assert npar == {"arm_smprime": {4806}, "arm_pl": {4803}, "softmax": {4769}}
    # +0.78% parameters, ~9.6x seconds -> the cost is not in the parameter count
    assert abs((4806 - 4769) / 4769 - 0.00776) < 1e-4
    ratio = statistics.mean(r["secs"] for r in a["arm_smprime"]) / \
        statistics.mean(r["secs"] for r in a["softmax"])
    assert 9.0 < ratio < 11.0, ratio
    # and the annihilated half buys nothing: the flat band costs no less than seed 2
    band = [r["secs"] for r in a["arm_smprime"] if r["frac_gate_annihilated"] > 0.49]
    dense = [r["secs"] for r in a["arm_smprime"] if r["frac_gate_annihilated"] == 0.0]
    assert min(band) > min(dense)


# ---------------------------------------------------------------- Q5


def test_q5_floor_1_is_derived_not_cited_and_holds_to_machine_precision():
    for r in _cells():
        assert r["floor_1"] == math.sqrt((r["t_star"] - 1) / r["t_star"])
        assert r["floor_1"] == 0.7071067811865476


def test_q5_floor_1_is_a_ONE_HOP_THRESHOLD_not_an_information_floor():
    """h_hat = t*(1-NRMSE^2); h_hat = 1 exactly at NRMSE = floor_1."""
    for r in _cells():
        assert abs(r["eval_h_hat"] - r["t_star"] * (1 - r["eval_nrmse"] ** 2)) < 1e-12
        assert abs(r["dist_to_floor"] - (r["eval_nrmse"] - r["floor_1"])) < 1e-12
        assert (r["eval_nrmse"] < r["floor_1"]) == (r["eval_h_hat"] > 1.0)
    # a lower bound is never violated. this one is, by six cells.
    below = sorted(r["cell"] for r in _cells() if r["dist_to_floor"] < 0)
    assert len(below) == 6
    assert [c.split(":")[0] for c in below].count("arm_pl") == 5
    assert [c.split(":")[0] for c in below].count("arm_smprime") == 1


def test_q5_the_real_calibrated_floor_and_ceiling_are_journalled():
    bar = [r for r in _rows(RETAKE) if "predict_the_mean" in r]
    assert len(bar) == 1 and bar[0]["ok"] is True
    assert bar[0]["oracle"] == 0.0                       # the information floor
    assert bar[0]["predict_the_mean"] == 1.0000000843170462
    assert bar[0]["trained_two_feature"] == 0.013981630466969725
    # every failing cell in both wings sits at or past the trivial ceiling
    worst = max(r["eval_nrmse"] for r in _cells())
    assert worst > bar[0]["predict_the_mean"]


def test_q5_no_arm_crosses_and_W3_is_blocked_by_variance_not_by_the_floor():
    aggs = {r["kind"]: r for r in _rows(RETAKE) if r.get("t") == "agg"}
    assert set(aggs) == {"arm_pl", "arm_smprime", "softmax"}
    assert all(a["crosses"] is False for a in aggs.values())
    pl = [r for r in _cells() if r["cell"].startswith("arm_pl")]
    good = [r["eval_nrmse"] for r in pl if r["lambda_hat"] < 0]
    bad = [r["eval_nrmse"] for r in pl if r["lambda_hat"] > 0]
    assert len(good) == 5 and len(bad) == 3
    # the three divergent cells alone put the arm mean above the threshold
    assert statistics.mean(good) < 0.7071067811865476 < statistics.mean(good + bad)
    assert abs(statistics.mean(good) - 0.6454112028697345) < 1e-12
    # and the good five would clear it with room, at n=5
    hi = statistics.mean(good) + 2.776 * statistics.stdev(good) / math.sqrt(5)
    assert hi < 0.7071067811865476, hi


def test_q5_dist_to_skyline_is_null_on_every_cell_with_the_reason_recorded():
    cells = _cells()
    assert all(r["dist_to_skyline"] is None for r in cells)
    assert {r["dist_to_skyline_why"] for r in cells} == {
        "no v15/v16 scan-skyline module exists (R-SKY)"}


# ------------------------------------------------- Q4 sparsity certificate
# tda-tdd contracts exercised: MASK FIDELITY (the certificate's predicted zero
# set is exactly the operator's zero set), CAUSALITY (nothing above the
# diagonal), ALL-MASKED-ROW guard (row 0 of the path product), DETERMINISM.

import torch                                                    # noqa: E402
from ceq import arm_smprime                                     # noqa: E402


def _certificate(zero_positions, S):
    """F0 zero-gate segmentation: G_ij = prod_{k=j+1..i} a_k, so G_ij == 0 iff
    Z intersects (j, i].  Per row i, live columns are the contiguous suffix
    [z(i), i] where z(i) = max{k <= i : k in Z}, else 0.  No delta, no epsilon,
    no tail bound -- the certificate is exact.
    """
    Z = set(zero_positions)
    live = torch.zeros(S, S, dtype=torch.bool)
    for i in range(S):
        z = max([k for k in Z if k <= i], default=0)
        if MUT == "cert_off_by_one":
            z = max(0, z - 1)
        live[i, z:i + 1] = True
    return live


def test_q4_zero_gate_certificate_is_EXACT_mask_fidelity_and_causality():
    torch.manual_seed(0)
    S = 12
    for Z in ([], [5], [3, 9], [1, 2, 3], list(range(1, S))):
        m = torch.rand(1, S).clamp(0.05, 1.0)
        m[0, list(Z)] = 0.0
        a = arm_smprime.gate(m, torch.zeros(1, S))
        G = arm_smprime.path_product(a)[0]
        actual_live = G.abs() != 0
        predicted = _certificate(Z, S)
        assert torch.equal(actual_live, predicted), Z          # MASK FIDELITY
        assert not predicted.triu(1).any()                     # CAUSALITY
        assert predicted.diagonal().all()                      # no all-masked row
        assert torch.equal(arm_smprime.path_product(a)[0], G)  # DETERMINISM


def test_q4_the_certified_sparsity_is_computed_but_never_exploited():
    """path_product materialises the dense [S,S] block, then produces the zeros."""
    src = (ROOT / "ceq" / "arm_smprime.py").read_text(encoding="utf-8")
    assert "ponytail:" in src and "O(S^2) materialization" in src
    for banned in ("topk", "top_k", "torch.sparse", "nonzero("):
        assert banned not in src, banned
    # the operator-level mask exists and the campaign script never calls it
    assert "def zero_hop_mask" in src
    runner = (ROOT / "scripts" / "v15_r1.py").read_text(encoding="utf-8")
    assert "zero_hop_mask" not in runner
    # and the repo's own FLOP model has no cell for either frozen wing
    flops = (ROOT / "scale" / "m3_flops.py").read_text(encoding="utf-8")
    assert "arm_smprime" not in flops and "arm_pl" not in flops
    sizing = (ROOT / "ceq" / "sizing.py").read_text(encoding="utf-8")
    assert "arm_smprime" not in sizing


def test_q4_the_timer_is_unsynchronised_and_the_length_is_a_module_constant():
    runner = (ROOT / "scripts" / "v15_r1.py").read_text(encoding="utf-8")
    assert "S, D = 64, 24" in runner                 # not a CLI flag
    assert "--steps" in runner and "--n-train" in runner
    assert "--seq-len" not in runner and "--s " not in runner
    assert "secs = time.time() - t0" in runner
    assert "cuda.synchronize" not in runner          # CUDA wall clock, unsynced

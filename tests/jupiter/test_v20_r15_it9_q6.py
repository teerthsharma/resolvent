"""V20 R15 it.9 -- JUPITER -- Q6 STATE METRIC, both wings, and the it.7 regime
table restated from `results/` after Inspector strikes D3/D4/D6/D7.

Nothing here is a second implementation of the instrument: the arms and the
oracle are imported from the shipped modules and called on a shipped draw.
"""
import glob
import json
import math
import pathlib
import hashlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
FLAT_FRAC = 0.5032958984375


def _cells(kind=None):
    """Every banked `t == "cell"` record, deduped on (kind, seed, nrmse@6)."""
    out = {}
    for f in sorted(glob.glob(str(ROOT / "results" / "*.jsonl"))):
        for ln in open(f, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if r.get("t") != "cell" or "instrument_hash" not in r:
                continue
            if kind is not None and r.get("kind") != kind:
                continue
            out.setdefault((r.get("kind"), r["seed"], round(r["eval_nrmse"], 6)), r)
    return list(out.values())


# ------------------------------------------------- the corrected regime table

def test_it7_regime_table_flat_band_endpoints_are_the_data_not_the_draft():
    """D3 + D4. The flat band is the 12 cells at `frac == 0.5032958984375`."""
    cells = {c["seed"]: c for c in _cells("arm_smprime")}
    flat = sorted(s for s in cells
                  if cells[s]["frac_gate_annihilated"] == FLAT_FRAC)
    assert flat == [0, 1, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15]
    live = {s: cells[s]["lambda_hat_live"] for s in flat}
    lo, hi = min(live.values()), max(live.values())
    assert min(live, key=live.get) == 4 and max(live, key=live.get) == 6
    # D4: the endpoint the draft printed was -0.0436; the datum is seed 4.
    assert round(lo, 4) == -0.0476
    # D3: -0.00095292... ROUNDS to -0.0010, it does not truncate to -0.0009.
    assert round(hi, 4) == -0.0010
    # the three columns that force seed 4 into the band
    assert len(flat) == 12
    nr = [cells[s]["eval_nrmse"] for s in flat]
    assert (round(min(nr), 3), round(max(nr), 3)) == (0.852, 0.927)
    beta = [cells[s]["manifest"]["smp_values"]["beta"] for s in flat]
    assert (round(min(beta), 3), round(max(beta), 3)) == (0.588, 1.001)


def test_d4_planted_negative_the_struck_endpoint_is_bound_to_no_datum():
    """The sweep the strike rests on, re-run here: every finite
    `lambda_hat_live` in `results/`, every step, `_0step` twins included."""
    n, hits = 0, []
    for f in sorted(glob.glob(str(ROOT / "results" / "*.jsonl"))):
        for ln in open(f, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            for k, v in r.items():
                if "lambda_hat_live" in k and isinstance(v, float) and math.isfinite(v):
                    n += 1
                    if round(v, 4) == -0.0436:
                        hits.append((f, k, v))
    # it.11: was `n == 878`. `results/` is written by other offices between
    # iterations (MERCURY's `v20_r15_it10_mercury_rescore.jsonl` added 54
    # finite fields and broke the it.9 green -- Inspector §1.6). The count is
    # a FLOOR now: it still fails if the sweep silently NARROWS, which is the
    # failure mode D4 slipped through, and it no longer fails when the record
    # grows. `878` was the it.9 reading; the floor never moves down.
    assert n >= 878, n
    assert hits == []


def test_d6_the_over_decayed_row_is_one_seed_order():
    """D6. Columns paired per seed, not per column."""
    cells = {c["seed"]: c for c in _cells("arm_smprime")}
    row = {s: (round(cells[s]["frac_gate_annihilated"], 6),
               round(cells[s]["lambda_hat_live"], 4),
               round(cells[s]["eval_nrmse"], 6)) for s in (11, 13)}
    assert row[11] == (0.994629, -4.1869, 1.120603)
    assert row[13] == (0.976807, -2.4989, 1.203324)


def test_d7_the_over_decayed_cut_admits_both_of_its_own_cells():
    """D7. `<= -2.5` excludes seed 13 (-2.49888). The regime is a MEMBERSHIP
    on `frac > 0.9`; what separates it is a GAP, not the printed threshold."""
    cells = {c["seed"]: c for c in _cells("arm_smprime")}
    live = {s: cells[s]["lambda_hat_live"] for s in cells}
    assert not (live[13] <= -2.5)                      # the struck cut
    over = sorted(s for s in cells if cells[s]["frac_gate_annihilated"] > 0.9)
    assert over == [11, 13]
    assert max(live[s] for s in over) == pytest.approx(-2.4988784790039062)
    rest = [live[s] for s in cells if s not in over]
    assert max(live[s] for s in over) < min(rest)      # gap, not a threshold
    assert round(min(rest) - max(live[s] for s in over), 4) == 1.7036


# ---------------------------------------------------------------- Q6, both wings

def test_q6_both_wings_predict_a_SCALAR_so_no_state_distribution_exists():
    """The domain census, executed rather than asserted. Both frozen wings and
    the oracle return one real number per draw at position s-1."""
    sys.path.insert(0, str(ROOT))
    from ceq import arm_pl, arm_smprime
    from scale.negation_scope import M3_TASKS
    batch_fn, oracle_fn, _, _ = M3_TASKS["e3_t2"]
    torch.manual_seed(0)
    x, y, _f, _p = batch_fn(32, 64, 24, d_model=16, seed=12345)
    z = oracle_fn(x, 0, 0)
    assert y.shape == (32,) and z.shape == (32,)
    assert not z.is_complex() and z.dtype == torch.float32
    for mod, cls in ((arm_smprime, "ArmSMPrime"), (arm_pl, "ArmPL")):
        torch.manual_seed(0)
        m = getattr(mod, cls)(64, d_model=16)
        with torch.no_grad():
            p = m(x)
        assert p.shape == (32,), (cls, p.shape)
        assert p.dim() == 1
        # no probability head anywhere: nothing sums to 1 over a state axis
        assert not any(n.endswith("logits") or "prob" in n
                       for n, _ in m.named_parameters())


def test_q6_no_banked_cell_journals_a_distribution_or_a_prediction_vector():
    """0 of N. Every banked cell field is a scalar, a string, a bool or the
    manifest; none is a sample, a histogram, a quantile or a density."""
    cells = _cells()
    assert len(cells) == 40, len(cells)
    words = ("hist", "quantile", "density", "dist_over", "samples", "pred",
             "logit", "prob", "wasserstein", "w1", "kl")
    offenders = []
    for c in cells:
        for k, v in c.items():
            if k == "manifest":
                continue
            if isinstance(v, list):
                offenders.append(("vector", k))
            if any(w in k.lower() for w in words) and not k.startswith("dist_to_"):
                offenders.append(("named", k))
    assert offenders == [], sorted(set(offenders))


# KILLED at it.11. `test_q6_planted_negative_marginal_W1_is_permutation_blind`
# stood here. Its "oracle" was `torch.randn`, so it did not test the object this
# report names; its `sqrt(2)` measured `1.4060`, not `1.4142`; and its
# `W1 == 0.0` is an identity of the sorted-difference formula that holds at
# every seed, so `manual_seed(4096)` did no work and the test COULD NOT FAIL.
# Struck by the Inspector, `V20_R15_IT89_INSPECTOR.md:47-76` (STRIKE I-1).
#
# REPLACEMENT ROUTE, same conclusion on the real object and able to fail:
#   tests/jupiter/test_v20_r15_it11_q6_oracle.py
#     ::test_q6_marginal_W1_is_permutation_blind_on_the_real_oracle
#     ::test_q6_marginal_W1_inverts_the_ranking_against_nrmse
#     ::test_q6_marginal_W1_is_not_degenerate_the_control
# `equilibrium_oracle` at the banked geometry (n=4096, s=64, d=24, t_star=2,
# seed=4096): W1 = 0.0 exact, NRMSE = 1.421901019003236.


# ---------------------------------------------------- Q4, the price of varying s

def test_q4_varying_s_forfeits_the_instrument_hash_by_the_trees_own_rule():
    """The `file` component of `instrument_manifest` is sha256 of the script's
    own bytes and moves on ANY edit (`scale/identity_manifest.py:184-192`).
    Adding a `--seq-len` flag therefore retires `5d41a63d...9a309`."""
    src = (ROOT / "scripts" / "v15_r1.py").read_bytes()
    assert b"\nS, D = 64, 24" in src
    before = hashlib.sha256(src).hexdigest()
    after = hashlib.sha256(src.replace(b"\nS, D = 64, 24",
                                       b"\nS, D = 128, 24")).hexdigest()
    assert before != after
    banked = {c["instrument_hash"] for c in _cells()}
    assert len(banked) == 1
    assert next(iter(banked)).startswith("5d41a63d")


def test_q4_sequence_length_is_still_one_point_on_every_banked_cell():
    cells = _cells()
    assert {c["s"] for c in cells} == {64}
    assert {c["t_star"] for c in cells} == {2}
    assert {c["steps"] for c in cells} == {150}
    assert len(cells) == 40
    # the record GREW since it.8 (34 -> 40): arm_pl seeds 8-15 landed
    # under the SAME instrument_hash, and s is still one point.
    assert len({c["instrument_hash"] for c in cells}) == 1


def test_w3_lambda_hat_sign_separates_the_floor_out_of_sample_on_16_cells():
    """The it.8 L-6 / L-12 predictor, checked on the eight arm_pl cells that
    landed after it.8 was written. `lambda_hat > 0` <=> at or above floor_1."""
    pl = {c["seed"]: c for c in _cells("arm_pl")}
    assert sorted(pl) == list(range(16))
    floor = 0.7071067811865476
    assert {c["floor_1"] for c in pl.values()} == {floor}
    above = sorted(s for s in pl if pl[s]["eval_nrmse"] >= floor)
    pos = sorted(s for s in pl if pl[s]["lambda_hat"] > 0)
    assert above == pos == [2, 3, 7, 9]
    below = [s for s in pl if s not in above]
    assert len(below) == 12
    # every below-floor cell clears the floor with its WHOLE bootstrap interval
    assert max(pl[s]["boot_hi"] for s in below) < floor
    # planted negative: the n=8 pooled mean the it.8 verdict rested on is
    # still above the floor, so the pooled statistic remains the obstruction.
    eight = [pl[s]["eval_nrmse"] for s in range(8)]
    assert sum(eight) / 8 > floor

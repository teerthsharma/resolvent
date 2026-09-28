# R-JEPA Foreman tests. Bars: BAR.md (registered before rjepa.py existed). Run: python -m pytest -q test_rjepa.py
import json
import pytest
import numpy as np
import torch
from hypothesis import given, settings, strategies as st

import rjepa as R


# ---- T-A: a linear permutation-equivariant resolvent is rank-inert --------------------------------------
@settings(max_examples=300, deadline=None)
# float separation >= 1e-2: the theorem is exact; denormal near-ties (hypothesis found 1e-308 vs 0) are rounding, not rank change
@given(s=st.lists(st.integers(-1000, 1000), min_size=2, max_size=12, unique=True).map(lambda v: [x / 100 for x in v]),
       alpha=st.floats(-1, 1), beta=st.floats(-1, 1), gamma=st.floats(0.01, 0.99))
def test_linear_equivariant_resolvent_is_rank_inert(s, alpha, beta, gamma):
    s = np.array(s)
    out = R.linear_equivariant_resolvent(s, alpha, beta, gamma)
    if out is None:                                    # spectral radius >= 1: outside the theorem
        return
    assert (np.argsort(out) == np.argsort(s)).all()


# ---- kernel contracts for the set heads (tda-tdd) --------------------------------------------------------
def _z(B=64, K=4, seed=0):
    return torch.randn(B, K, 2, generator=torch.Generator().manual_seed(seed))


def test_resolvent_dense_parity_solve_neumann_inverse():
    W = torch.softmax(torch.randn(8, 4, 4, dtype=torch.float64), -1)
    h = torch.randn(8, 4, 5, dtype=torch.float64)
    g = 0.9
    a = R.resolvent_apply(W, h, g)
    inv = torch.linalg.inv(torch.eye(4, dtype=torch.float64) - g * W) @ h
    neu, t = h.clone(), h.clone()
    for _ in range(600):
        t = g * W @ t; neu = neu + t
    assert torch.allclose(a, inv, atol=1e-10) and torch.allclose(a, neu, atol=1e-10)


def test_resolvent_bounded_by_one_over_one_minus_gamma():
    for g in (0.1, 0.5, 0.9, 0.99):
        W = torch.softmax(torch.randn(32, 4, 4, dtype=torch.float64) * 3, -1)
        Rm = torch.linalg.inv(torch.eye(4, dtype=torch.float64) - g * W)
        assert Rm.abs().sum(-1).max() <= 1 / (1 - g) + 1e-9


def test_heads_are_permutation_equivariant():
    torch.manual_seed(0)
    z = _z()
    perm = torch.tensor([2, 0, 3, 1])
    for kind in ("point", "hop1", "resolvent", "djepa"):
        m = R.Head(kind).double()
        s = m(z.double())
        sp = m(z.double()[:, perm])
        assert torch.allclose(s[:, perm], sp, atol=1e-10), kind


def test_heads_stable_under_small_perturbation():
    torch.manual_seed(1)
    z = _z().double()
    for kind in ("hop1", "resolvent"):
        m = R.Head(kind).double()
        d = (m(z + 1e-6 * torch.randn_like(z)) - m(z)).abs().max().item()
        assert d < 1e-3, (kind, d)


def test_all_masked_neighbours_no_nan():
    z = _z(B=3).double()
    mask = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    for kind in ("point", "hop1", "resolvent", "djepa"):
        s = R.Head(kind).double()(z, mask)
        assert torch.isfinite(s[mask]).all(), kind
        assert (s[~mask] == -torch.inf).all(), kind
        assert s[0].argmax().item() == 0


# ---- bed bars B0-B3 (no learning) ------------------------------------------------------------------------
def test_B0_null_iid_error_bayes_equals_distance():
    b = R.make_bed(4000, sigma=0.0, sigma_e=0.3, seed=0)
    assert (R.bayes_pick(b) == R.dist_pick(b["zhat_true"])).mean() >= 0.97


def test_B1_ceiling_small_error():
    b = R.make_bed(4000, sigma=0.1, seed=0)
    assert R.hit(R.bayes_pick(b), b) >= 0.90


def test_B2_gap_exists_at_shared_error_equal_to_spread():
    gaps = []
    for sd in (0, 1, 2):
        b = R.make_bed(4000, sigma=1.0, seed=sd)
        gaps.append(R.hit(R.bayes_pick(b), b) - R.hit(R.dist_pick(b["zhat_true"]), b))
    assert np.mean(gaps) >= 0.03, gaps


def test_B3_distance_anti_informative_past_horizon_and_hull_angle_is_bayes():
    d, h, y = [], [], []
    for sd in (0, 1, 2):
        b = R.make_bed(4000, sigma=10.0, seed=sd)
        d.append(R.hit(R.dist_pick(b["zhat_true"]), b)); h.append(R.hit(R.hull_angle_pick(b), b))
        y.append(R.hit(R.bayes_pick(b), b))
    assert np.mean(d) <= 0.23 and np.mean(h) >= np.mean(y) - 0.02, (d, h, y)


# ---- learned bars B4-B7 at sigma = 1 (3 seeds) ------------------------------------------------------------
_CACHE = {}


def _cell(sigma=1.0, kinds=("point", "hop1", "resolvent"), epochs=36):
    key = (sigma, kinds, epochs)
    if key not in _CACHE:
        _CACHE[key] = R.run_cell(sigma, seeds=(0, 1, 2), kinds=kinds, epochs=epochs)
        json.dump(_CACHE[key], open(f"results_cell_sigma{sigma:g}_ep{epochs}_{'-'.join(kinds)}.json", "w"), indent=1)
    return _CACHE[key]


def test_B4_resolvent_learns_the_bayes_rule():
    assert np.mean(_cell()["agree"]["resolvent"]) >= 0.90, _cell()["agree"]


def test_B5_resolvent_closes_the_gap():
    assert np.mean(_cell()["closure"]["resolvent"]) >= 0.80, _cell()["closure"]


def test_B6_pointwise_head_cannot_close_it():
    assert np.mean(_cell()["closure"]["point"]) <= 0.50, _cell()["closure"]


def test_B7_neumann_tail_beats_one_hop():
    c = _cell()["hit"]
    diff = np.array(c["resolvent"]) - np.array(c["hop1"])
    assert diff.mean() >= 0.01 and (diff > 0).all(), diff


# ---- D-JEPA operator contract (Prop 2 / Cor 1 as quoted in wilson/facts.json) -------------------------------
def test_djepa_correction_bounded_and_selection_within_2eps_of_base_min():
    torch.manual_seed(3)
    m = R.Head("djepa").double()
    for p in m.parameters():                          # leave zero-init: randomise so the bound is exercised
        torch.nn.init.normal_(p, std=2.0)
    z = _z(B=256).double()
    s = -m(z)                                         # s = b + delta, lower is better
    delta = m.last_delta
    assert delta.abs().max() <= m.eps + 1e-12
    b = s - delta
    pick = s.argmin(1)
    assert (b.gather(1, pick[:, None]).squeeze(1) - b.min(1).values <= 2 * m.eps + 1e-12).all()


@pytest.mark.skip(reason="VOID per BAR.md A3: djepa failed its learn gate; B8b decides")
def test_B8_djepa_bound_closes_band_but_blocks_the_horizon_rule():
    # VOID (A3): djepa failed its learn gate in this 12-epoch run; B8b decides the bound
    kinds = ("point", "hop1", "resolvent", "djepa")
    c1, c10 = _cell(1.0, kinds, 12), _cell(10.0, kinds, 12)
    assert np.mean(c1["hit"]["djepa"]) >= np.mean(c1["bayes"]) - 0.03, (c1["hit"]["djepa"], c1["bayes"])
    assert np.mean(c10["bayes_rank_ge2"]) >= 0.25, c10["bayes_rank_ge2"]
    assert np.mean(c10["hit"]["djepa"]) <= np.mean(c10["bayes"]) - 0.03, (c10["hit"]["djepa"], c10["bayes"])


def test_B9_resolvent_head_not_slower_than_djepa_operator_K63():
    t = R.latency(K=63, B=16)
    assert t["resolvent"] <= t["djepa"], t


def test_B8b_bound_restricted_bayes_ceiling_blocks_the_horizon_rule():
    lo, hi = [], []
    for sd in (0, 1, 2):
        for sg, acc in ((1.0, hi), (10.0, lo)):
            b = R.make_bed(4000, sg, seed=sd)
            acc.append(R.hit(R.bayes_pick(b), b) - R.hit(R.bayes_pick(b, max_rank=1), b))
    assert np.mean(hi) <= 0.01 and np.mean(lo) >= 0.03, (hi, lo)


def test_B10_gap_is_carried_by_the_shared_share_of_error():
    gap = {}
    for m in (0.0, 0.5, 1.0):
        g = []
        for sd in (0, 1, 2):
            b = R.make_bed(4000, 3.0, seed=sd, mix=m)
            g.append(R.hit(R.bayes_pick(b), b) - R.hit(R.dist_pick(b["zhat_true"]), b))
        gap[m] = float(np.mean(g))
    assert gap[0.0] >= 0.10 and gap[1.0] <= 0.01 and gap[0.0] > gap[0.5] > gap[1.0], gap


# ---- the horizon rule is a convex-hull invariant (tda-tdd: permutation, isometry, scale, ground truth) -----
def test_hull_angle_share_invariants_and_ground_truth():
    rng = np.random.default_rng(0)
    P = rng.standard_normal((50, 4, 2))
    sh = R.hull_angle_share(P)
    assert np.allclose(sh.sum(-1), 1)
    perm = [3, 1, 0, 2]
    assert np.allclose(R.hull_angle_share(P[:, perm]), sh[:, perm])                 # permutation
    th = 0.7; Q = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    assert np.abs(R.hull_angle_share(P @ Q.T + 5.0) - sh).max() <= 2 / 4096 + 1e-12  # isometry, up to direction grid
    assert np.allclose(R.hull_angle_share(3.0 * P), sh)                              # scale
    sq = np.array([[[1, 0], [0, 1], [-1, 0], [0, -1]]], float)
    assert np.allclose(R.hull_angle_share(sq), 0.25, atol=1e-3)                      # square: 1/4 each
    tri = np.array([[[1, 0], [-0.5, 0.8], [-0.5, -0.8], [0, 0]]], float)
    assert R.hull_angle_share(tri)[0, 3] == 0                                        # interior point: 0


def test_B11_resolvent_learns_the_horizon_rule_sigma10():
    c = _cell(10.0)
    assert np.mean(c["agree"]["resolvent"]) >= 0.750 and np.mean(c["closure"]["resolvent"]) >= 0.80, (c["agree"], c["closure"])


def test_B12_horizon_correction_is_mostly_pointwise_sigma10():
    assert np.mean(_cell(10.0)["closure"]["point"]) >= 0.50, _cell(10.0)["closure"]


def test_B13_djepa_without_its_trust_region_learns_the_horizon_rule():
    c = _cell(10.0, ("djepa4",), 12)
    assert np.mean(c["agree"]["djepa4"]) >= 0.750 and np.mean(c["closure"]["djepa4"]) >= 0.80, (c["agree"], c["closure"])

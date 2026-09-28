"""Bed `dial`: K = 63 shared-error dial with a learned predictor (source sun/rjepa/r2/r2.py, BAR.md A1-A6).

Ported from sun/rjepa/r2/test_r2.py (C1-C7) and sun/rjepa/r3/foreman/test_r3.py (A6.1 tie-break), plus the
error-mix, saturation and fast-cell tests written for this package before rjepa/dial.py existed.
"""
import json
from functools import partial

import pytest
import torch

from rjepa import dial as R

K, D, M_, H = R.K, R.D, R.M_ACT, R.H


@pytest.fixture(scope="module")
def dyn():
    return R.make_dyn()


def _v(n=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, K, R.NTOK, generator=g), torch.rand(n, K, generator=g)


# ---- bed ------------------------------------------------------------------------------------------------------
def test_bed_shapes_and_truth_label(dyn):
    b = R.make_cell(7, f=0.5, sigma=1.0, seed=0, dyn=dyn)
    assert b["y"].shape == (7, D) and b["g"].shape == (7, D)
    assert b["a"].shape == (7, K, H, M_)
    assert b["dist"].shape == (7, K) and b["best"].shape == (7,)
    assert torch.equal(b["best"], b["dist"].argmin(1))


def test_bed_is_deterministic_in_its_seed(dyn):
    a, b = R.make_cell(20, 0.9, 1.0, 3, dyn), R.make_cell(20, 0.9, 1.0, 3, dyn)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_start_error_mix_is_shared_fraction_f():
    """e_k = sqrt(f) s + sqrt(1 - f) u_k: unit variance per candidate, cross-candidate covariance f."""
    for f in (0.0, 0.3, 0.9, 1.0):
        e = R.start_error((20000,), f, torch.Generator().manual_seed(1))
        assert e.shape == (20000, K, D)
        assert abs(float(e.var()) - 1.0) < 0.03
        cov = float((e[:, 0] * e[:, 1]).mean())
        assert abs(cov - f) < 0.03, (f, cov)
    e1 = R.start_error((5,), 1.0, torch.Generator().manual_seed(2))
    assert torch.equal(e1, e1[:, :1].expand_as(e1))              # f = 1: one start shared by all K


def test_constant_pick_is_chance(dyn):
    """The label is not predictable from the candidate index: the true-best index is uniform over K."""
    b = R.make_cell(4000, 1.0, 1.0, 9, dyn)
    se = ((1 / K) * (1 - 1 / K) / 4000) ** 0.5
    for k in (0, 31, 62):
        assert abs(R.hit(torch.full((4000,), k), b) - 1 / K) <= 3 * se


def test_calibration_reproduces_source_sigma(dyn):
    """sigma for sigma/rho = 1, 3, 10, against sun/rjepa/r2/results/calib.json."""
    c = R.calibrate(dyn)
    assert abs(c["rho"] - 0.6375638246536255) < 1e-5
    for q, s in R.SIGMA.items():
        assert abs(c["sigma"][str(q)] - s) < 1e-4


# ---- heads ----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["point", "hop1", "dj02", "dj4L"])
def test_C1_permutation_equivariance(kind):
    torch.manual_seed(0)
    h = R.DialHead(kind)
    for p in h.parameters():                                     # break the zero-init so the check is not vacuous
        torch.nn.init.normal_(p, 0, 0.3)
    v, b = _v()
    perm = torch.randperm(K)
    with torch.no_grad():
        s, sp = h(v, b), h(v[:, perm], b[:, perm])
    assert torch.allclose(s[:, perm], sp, atol=1e-4)
    assert s.std() > 1e-3


@pytest.mark.parametrize("kind,eps", [("dj02", 0.2), ("dj0.5", 0.5), ("dj4L", 4.0)])
def test_C2_djepa_delta_bounded_by_eps(kind, eps):
    torch.manual_seed(1)
    h = R.DialHead(kind)
    assert h.eps == eps
    for p in h.parameters():
        torch.nn.init.normal_(p, 0, 1.0)
    v, b = _v(64, 1)
    with torch.no_grad():
        s = h(v, b)
    d = h.last_delta
    assert d.abs().max() <= eps * (1 + 1e-6) and d.abs().max() > 0.25 * eps
    if eps == 0.2:                                               # Cor 1: the pick lies within 2 eps of the base minimum
        pick = s.argmax(1)
        assert (b[torch.arange(64), pick] <= b.min(1).values + 0.4 + 1e-6).all()


def test_saturation_is_mean_abs_delta_over_eps():
    h = R.DialHead("dj02")
    v, b = _v(8, 2)
    with torch.no_grad():
        assert R.saturation(h, v, b) == 0.0                      # zero-init: delta = 0, returns the base ranking
        for p in h.parameters():
            torch.nn.init.normal_(p, 0, 3.0)
        sat = R.saturation(h, v, b)
    assert abs(sat - float(h.last_delta.abs().mean() / 0.2)) < 1e-6
    assert 0.5 < sat <= 1.0                                      # large weights drive tanh into saturation


def test_base_ranks_by_kind():
    rr, lr = torch.rand(3, K), torch.rand(3, K)
    assert R.base_for("dj4L", rr, lr) is lr and R.base_for("dj02", rr, lr) is rr and R.base_for("hop1", rr, lr) is rr


# ---- Bayes ceiling --------------------------------------------------------------------------------------------
def test_tiebreak_uses_true_posterior_not_learned_distance():
    """A6.1: exact ties in P(k best) go to the smallest posterior-mean TRUE distance, not the learned one."""
    Pb = torch.tensor([[0.5, 0.5, 0.0]])
    d_learned = torch.tensor([[1.0, 2.0, 3.0]])
    Ed = torch.tensor([[3.0, 1.0, 2.0]])
    assert int((Pb - 1e-9 * d_learned).argmax(1)) == 0           # the round-2 rule
    assert int(R.bayes_pick(Pb, Ed)) == 1


def test_tiebreak_restricted_pick():
    Pb = torch.tensor([[0.9, 0.3, 0.3, 0.1]])
    Ed = torch.tensor([[0.1, 2.0, 1.0, 3.0]])
    allowed = torch.tensor([[False, True, True, True]])
    assert int(R.bayes_pick(Pb, Ed, allowed)) == 2
    assert int(R.bayes_pick(Pb, Ed)) == 0


def test_bayes_ceiling_dominates_learning_free_arms(dyn):
    b = R.make_cell(400, f=1.0, sigma=1.0, seed=7, dyn=dyn)
    step = partial(R.true_step, dyn=dyn)
    d, s = R.lin1_feats(step, b["y"], b["a"], b["g"], 1.0)
    r = float(d.min(1).values.median())
    Pb, _, Ed = R.bayes_P(b, dyn, sigma=1.0, f=1.0, r=r, M=256, seed=0)
    assert Ed.shape == Pb.shape and bool((Ed > 0).all())
    hb = R.hit(R.bayes_pick(Pb, Ed), b)
    se = (0.25 / 400) ** 0.5
    for pick in (d.argmin(1), R.lin1_pick(d, s, r)):
        assert hb >= R.hit(pick, b) - 2 * se


def test_C3_lin1_exact_in_linear_gaussian_limit():
    g = torch.Generator().manual_seed(3)
    Mz, Na = torch.randn(D, D, generator=g) / D ** 0.5 * 1.1, torch.randn(D, M_, generator=g)

    def step(z, a):
        return z @ Mz.T + a @ Na.T

    n = 2
    y = torch.randn(n, D, generator=g)
    a = torch.randn(n, K, H, M_, generator=g)
    goal = R.rollout(step, y, torch.randn(n, 1, H, M_, generator=g))[:, 0] + 20.0
    sigma = 0.05
    d, s = R.lin1_feats(step, y, a, goal, sigma)
    r = d + s * (3 * torch.rand(n, K, generator=g) - 1.5)
    P = R.lin1_prob(d, s, r)
    assert (s / d).max() <= 0.01 and ((P > 0.05) & (P < 0.95)).sum() >= 50
    Mc = 20000
    e = torch.randn(Mc, n, 1, D, generator=g)
    z = R.rollout(step, (y[:, None] - sigma * e).reshape(-1, D), a.repeat(Mc, 1, 1, 1, 1).reshape(-1, K, H, M_))
    mc = ((z.reshape(Mc, n, K, D) - goal[None, :, None]).norm(dim=-1) < r).float().mean(0)
    assert (P - mc).abs().max() <= 0.02


# ---- learned cells --------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fast_cell(dyn):
    return R.run_cell(1.0, 3.0, seed=0, dyn=dyn, **R.FAST)


def test_fast_cell_bayes_at_least_every_arm(fast_cell):
    h, n = fast_cell["hit"], fast_cell["n_eval"]
    se = (h["bayes"] * (1 - h["bayes"]) / n) ** 0.5
    for k, v in h.items():
        assert h["bayes"] >= v - 2 * se, (k, v, h["bayes"])


def test_fast_cell_null_does_not_beat_distance_floor(fast_cell):
    """A1 form of the null: a head trained on shuffled labels must not beat the untrained distance floor."""
    h, n = fast_cell["hit"], fast_cell["n_eval"]
    se = ((1 / K) * (1 - 1 / K) / n) ** 0.5
    assert h["null"] <= h["dist"] + 3 * se


def test_fast_cell_sign_dj4L_beats_dj02_at_f1(fast_cell):
    """Sign only (round-2 P2 at f = 1): the unsaturated eps = 4 operator beats the eps = 0.2 one,
    and eps = 0.2 runs closer to its bound than eps = 4 does."""
    ns, sat = fast_cell["ns"], fast_cell["delta_over_eps"]
    assert ns["dj4L"] > ns["dj02"], ns
    assert sat["dj02"] > 0.7 and sat["dj4L"] < 0.1, sat         # source dj02 at full size: 0.792


@pytest.mark.slow
def test_full_cell_f1_q3_seed0(dyn, tmp_path):
    """Registered cell (f = 1, sigma/rho = 3), seed 0, at the source sizes: 60,000 train / 20,000 eval starts,
    M = 2,048, 4,000 steps, predictor trained 2,000 steps. Source (r2 tie-break): dj4L 0.732, dj02 0.352 NS,
    dj02 |delta|/eps 0.792. Bars: P2 edge >= 0.05 NS and saturation above 0.7."""
    res = R.run_cell(1.0, 3.0, seed=0, dyn=dyn)
    (tmp_path / "full_cell.json").write_text(json.dumps(res))
    assert res["ns"]["dj4L"] - res["ns"]["dj02"] >= 0.05
    assert res["delta_over_eps"]["dj02"] > 0.7

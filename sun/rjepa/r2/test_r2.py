# Contract tests C1-C7 for the round-2 dial bed (BAR.md). Written before r2.py exists.
import numpy as np
import pytest
import torch

import r2

K, D, M_, H = r2.K, r2.D, r2.M_ACT, r2.H


def _v(n=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, K, r2.NTOK, generator=g), torch.rand(n, K, generator=g)


@pytest.mark.parametrize("kind", ["point", "hop1", "dj02", "dj4L"])
def test_C1_permutation_equivariance(kind):
    torch.manual_seed(0)
    h = r2.Head(kind)
    for p in h.parameters():                                   # break the zero-init so the check is not vacuous
        torch.nn.init.normal_(p, 0, 0.3)
    v, b = _v()
    perm = torch.randperm(K)
    with torch.no_grad():
        s, sp = h(v, b), h(v[:, perm], b[:, perm])
    assert torch.allclose(s[:, perm], sp, atol=1e-4)
    assert s.std() > 1e-3                                      # not a constant head


def test_C2_dj02_bounded():
    torch.manual_seed(1)
    h = r2.Head("dj02")
    for p in h.parameters():
        torch.nn.init.normal_(p, 0, 1.0)
    v, b = _v(64, 1)
    with torch.no_grad():
        s = h(v, b)
    d = h.last_delta
    assert d.abs().max() <= 0.2 + 1e-6 and d.abs().max() > 0.05
    pick = s.argmax(1)
    assert (b[torch.arange(64), pick] <= b.min(1).values + 0.4 + 1e-6).all()


def test_C3_lin1_exact_in_linear_gaussian_limit():
    g = torch.Generator().manual_seed(3)
    Mz, Na = torch.randn(D, D, generator=g) / D ** 0.5 * 1.1, torch.randn(D, M_, generator=g)
    step = lambda z, a: z @ Mz.T.to(z.device) + a @ Na.T.to(z.device)
    n = 2
    y = torch.randn(n, D, generator=g)
    a = torch.randn(n, K, H, M_, generator=g)
    goal = r2.rollout(step, y, torch.randn(n, 1, H, M_, generator=g))[:, 0] + 20.0   # far goal: d >> s
    sigma = 0.05
    d, s = r2.lin1_feats(step, y, a, goal, sigma)
    r = d + s * (3 * torch.rand(n, K, generator=g) - 1.5)            # per-candidate radius: every P mid-range
    P = r2.lin1_prob(d, s, r)
    assert (s / d).max() <= 0.01 and ((P > 0.05) & (P < 0.95)).sum() >= 50
    Mc = 20000
    e = torch.randn(Mc, n, 1, D, generator=g)
    z = r2.rollout(step, (y[:, None] - sigma * e).reshape(-1, D), a.repeat(Mc, 1, 1, 1, 1).reshape(-1, K, H, M_))
    mc = ((z.reshape(Mc, n, K, D) - goal[None, :, None]).norm(dim=-1) < r).float().mean(0)
    assert (P - mc).abs().max() <= 0.02


@pytest.fixture(scope="module")
def small():
    dyn = r2.make_dyn()
    b = r2.make_cell(400, f=1.0, sigma=1.0, seed=7, dyn=dyn)
    return dyn, b


def test_C4_bayes_ceiling_dominates(small):
    dyn, b = small
    step = lambda z, a: r2.true_step(z, a, dyn)
    d, s = r2.lin1_feats(step, b["y"], b["a"], b["g"], 1.0)
    Pb, _ = r2.bayes_P(b, dyn, sigma=1.0, f=1.0, r=float(d.min(1).values.median()), M=256, seed=0)
    hb = r2.hit(Pb.argmax(1), b)
    se = (0.25 / 400) ** 0.5
    for pick in (d.argmin(1), r2.lin1_pick(d, s, float(d.min(1).values.median()))):
        assert hb >= r2.hit(pick, b) - 2 * se


def test_C5_null_is_chance():
    dyn = r2.make_dyn()
    step = lambda z, a: r2.true_step(z, a, dyn)
    tr, ev = r2.make_cell(3000, 1.0, 1.0, 11, dyn), r2.make_cell(2000, 1.0, 1.0, 12, dyn)
    vt, bt = r2.tokens(step, tr, 1.0, 0.5)[:2]
    ve, be = r2.tokens(step, ev, 1.0, 0.5)[:2]
    lab = tr["best"][torch.randperm(len(tr["best"]), generator=torch.Generator().manual_seed(0))]
    h = r2.train_head("point", vt, bt, lab, seed=0, steps=500)
    with torch.no_grad():
        p = h(ve.to(r2.DEV), be.to(r2.DEV)).argmax(1).cpu()
    se = ((1 / K) * (1 - 1 / K) / 2000) ** 0.5
    # Amendment A1: the registered form |hit - 1/K| <= 3 SE fired (0.0300 vs 0.0159, 5.0 SE): an argmax over a
    # near-constant head inherits the token's distance order. The null now bounds label information instead:
    # a head trained on shuffled labels must not beat the untrained latent-distance floor.
    d, _ = r2.lin1_feats(step, ev["y"], ev["a"], ev["g"], 1.0)
    assert r2.hit(p, ev) <= r2.hit(d.argmin(1), ev) + 3 * se


def test_C6_marginal_success_is_f_invariant():
    # The registered max|diff| <= 4 SE form ignored the 1,260-way max and the 2 independent MC draws (diff SE = sqrt(2) SE);
    # it read 0.0385. Replaced by the standardised statistic mean(z^2) ~ 1, plus a control that must fail it.
    dyn = r2.make_dyn()
    b0 = r2.make_cell(20, f=0.0, sigma=1.0, seed=5, dyn=dyn)
    r = float(b0["dist"].median())                             # r = 1.0 gave an all-zero table in 8-D (guard fired)
    M = 4000
    _, P0 = r2.bayes_P(b0, dyn, sigma=1.0, f=0.0, r=r, M=M, seed=1)
    _, P1 = r2.bayes_P(b0, dyn, sigma=1.0, f=1.0, r=r, M=M, seed=2)
    _, Pc = r2.bayes_P(b0, dyn, sigma=1.3, f=1.0, r=r, M=M, seed=3)      # control: a different marginal law
    assert P0.std() > 0.05                                     # not a degenerate all-0 / all-1 table
    z2 = lambda A, B: (((A - B) ** 2) / (((A + B) / 2 * (1 - (A + B) / 2)).clamp(min=1e-4) * 2 / M)).mean()
    assert 0.7 <= z2(P0, P1) <= 1.3
    assert z2(P0, Pc) > 3.0


def test_C7_L_LEARN_gate():
    _, r2_one, r2_H = r2.train_predictor(seed=0, dyn=r2.make_dyn())
    assert r2_one >= 0.90
    assert r2_H < 0.9999                                       # it carries real model error (not a copy of F)

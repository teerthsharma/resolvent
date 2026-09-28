"""Contracts for the ranking heads and the D-JEPA-spec relational operator.

Ported from experiments/shift/test_rjepa.py (head contracts, D-JEPA Prop 2 / Cor 1) and
experiments/torus/test_rj.py (operator equivariance and bound).
"""
import torch

from resolvent.djepa import RelationalOperator, dj_loss, rank01
from resolvent.heads import Head

KINDS = ("point", "hop1", "resolvent", "djepa")


def _z(B=64, K=4, seed=0):
    return torch.randn(B, K, 2, generator=torch.Generator().manual_seed(seed))


def test_heads_are_permutation_equivariant():
    torch.manual_seed(0)
    z = _z().double()
    perm = torch.tensor([2, 0, 3, 1])
    for kind in KINDS:
        m = Head(kind).double()
        assert torch.allclose(m(z)[:, perm], m(z[:, perm]), atol=1e-10), kind


def test_set_heads_stable_under_small_perturbation():
    torch.manual_seed(1)
    z = _z().double()
    for kind in ("hop1", "resolvent"):
        m = Head(kind).double()
        assert (m(z + 1e-6 * torch.randn_like(z)) - m(z)).abs().max().item() < 1e-3, kind


def test_masked_candidates_score_minus_inf_and_rows_stay_finite():
    z = _z(B=3).double()
    mask = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    for kind in KINDS:
        s = Head(kind).double()(z, mask)
        assert torch.isfinite(s[mask]).all(), kind
        assert (s[~mask] == -torch.inf).all(), kind
        assert s[0].argmax().item() == 0, kind


def test_djepa_head_correction_bounded_and_selection_within_2eps_of_base_min():
    """Prop 2 / Cor 1 of arXiv 2609.24749: |delta| <= eps, so the pick lies within 2 eps of the base minimum."""
    torch.manual_seed(3)
    m = Head("djepa").double()
    for p in m.parameters():                       # move off the zero-init head so the bound is exercised
        torch.nn.init.normal_(p, std=2.0)
    s = -m(_z(B=256).double())                     # s = b + delta, lower is better
    delta = m.last_delta
    assert delta.abs().max() <= m.eps + 1e-12
    b = s - delta
    pick = s.argmin(1)
    assert (b.gather(1, pick[:, None]).squeeze(1) - b.min(1).values <= 2 * m.eps + 1e-12).all()


def test_relational_operator_permutation_equivariant_and_bounded():
    torch.manual_seed(0)
    net = RelationalOperator(nin=5)
    for p in net.parameters():
        torch.nn.init.normal_(p, std=0.5)
    v, b = torch.randn(7, 8, 5), torch.rand(7, 8)
    s, dl = net(v, b)
    p = torch.randperm(8)
    s2, _ = net(v[:, p], b[:, p])
    assert torch.allclose(s2, s[:, p], atol=1e-5)
    assert (s - b).abs().max() <= 0.2 + 1e-6 and dl.abs().max() > 1e-3


def test_relational_operator_zero_init_returns_base():
    net = RelationalOperator(nin=5)
    b = torch.rand(3, 8)
    s, dl = net(torch.randn(3, 8, 5), b)
    assert torch.equal(s, b) and (dl == 0).all()


def test_rank01_maps_lowest_cost_to_zero_and_highest_to_one():
    r = rank01(torch.tensor([[0.3, 0.1, 0.9, 0.5]]))
    assert torch.allclose(r, torch.tensor([[1 / 3, 0.0, 1.0, 2 / 3]]))


def test_dj_loss_is_finite_with_and_without_positives_and_prefers_positive_low():
    s = torch.zeros(2, 4)
    y = torch.tensor([[True, False, False, False], [False] * 4])
    base = dj_loss(s, torch.zeros(2, 4), y)
    better = dj_loss(torch.tensor([[-1.0, 0, 0, 0], [0, 0, 0, 0]]), torch.zeros(2, 4), y)
    assert torch.isfinite(base) and better < base
    assert torch.isfinite(dj_loss(s, torch.zeros(2, 4), torch.zeros(2, 4, dtype=torch.bool)))

"""Contracts for the candidate-axis resolvents: equivariance, pole refusal, bounds, masks, dtype, gradients.

Ported from the source repo's experiments/cost/test_set_resolvent.py (C1-C7) and the operator half of
experiments/shift/test_rjepa.py (T-A and the dense-parity checks).
"""
import os

import numpy as np
import pytest
import torch

from resolvent.resolvent import (build_A, linear_equivariant_resolvent, neumann_resolvent, resolvent_apply,
                             set_resolvent, stochastic_resolvent)

KS = [1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65]


def draw(K, d=8, B=4, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(B, K, d, generator=g, dtype=dtype) for _ in range(3))


# C1 permutation equivariance ----------------------------------------------------------------------
def test_strictly_causal_operator_is_not_permutation_equivariant():
    """Why build_A masks i == j and not j >= i: a strictly lower-triangular A (the causal sequence
    operator) makes an unordered candidate set's output depend on listing order."""
    q, k, v = draw(16)

    def causal(q, k, v):
        w = (q @ k.transpose(-1, -2)).tril(-1)
        A = 0.9 * w / (w.abs().sum(-1, keepdim=True) + 1e-3)
        return torch.linalg.solve(torch.eye(16, dtype=A.dtype) - A, v)

    p = torch.randperm(16, generator=torch.Generator().manual_seed(1))
    assert (causal(q, k, v)[:, p] - causal(q[:, p], k[:, p], v[:, p])).abs().max() > 1e-3


@pytest.mark.parametrize("K", KS)
def test_permutation_equivariance(K):
    q, k, v = draw(K)
    p = torch.randperm(K, generator=torch.Generator().manual_seed(K))
    assert torch.allclose(set_resolvent(q, k, v)[:, p], set_resolvent(q[:, p], k[:, p], v[:, p]), atol=1e-10, rtol=0)


# C2 pole refusal and bounds -----------------------------------------------------------------------
@pytest.mark.parametrize("rho", [1.0, 1.5])
def test_rho_at_or_above_one_is_refused(rho):
    q, k, v = draw(4)
    with pytest.raises(ValueError):
        set_resolvent(q, k, v, rho=rho)


def test_rho_just_below_one_two_candidates_is_bounded():
    """K=2 signed: A = [[0, +-rho], [+-rho, 0]], eigenvalues +-rho, pole at rho = 1."""
    q = torch.tensor([[[1.0], [1.0]]], dtype=torch.float64)
    out = set_resolvent(q, q.clone(), torch.ones(1, 2, 1, dtype=torch.float64), rho=0.999)
    assert torch.isfinite(out).all() and out.abs().max() <= 1.0 / (1 - 0.999) + 1e-6


def test_spectral_radius_below_rho_on_1000_draws():
    worst = 0.0
    for s in range(1000):
        q, k, _ = draw(9, d=4, B=1, seed=s)
        worst = max(worst, torch.linalg.eigvals(build_A(q, k, rho=0.9)[0]).abs().max().item())
    assert worst <= 0.9 + 1e-12


def test_resolvent_inf_norm_bound():
    q, k, _ = draw(33)
    R = torch.linalg.inv(torch.eye(33, dtype=torch.float64) - build_A(q, k, rho=0.9))
    assert R.abs().sum(-1).max() <= 1 / (1 - 0.9) + 1e-9


# C3 degenerate inputs and padding -----------------------------------------------------------------
def test_single_candidate_is_identity_and_finite():
    q, k, v = draw(1)
    out = set_resolvent(q, k, v)
    assert torch.isfinite(out).all() and torch.equal(out, v)


def test_all_masked_is_finite():
    q, k, v = draw(8)
    assert torch.isfinite(set_resolvent(q, k, v, mask=torch.zeros(4, 8, dtype=torch.bool))).all()


def test_zero_features_is_finite():
    q = torch.zeros(2, 5, 3, dtype=torch.float64)
    assert torch.isfinite(set_resolvent(q, q, torch.ones(2, 5, 3, dtype=torch.float64))).all()


def test_masked_candidate_moves_nothing():
    """Padding a set with junk candidates leaves the real candidates' outputs unchanged."""
    q, k, v = draw(8)
    mask = torch.ones(4, 8, dtype=torch.bool)
    mask[:, 5:] = False
    full = set_resolvent(q, k, v, mask=mask)
    q2, k2, v2 = q.clone(), k.clone(), v.clone()
    q2[:, 5:] *= 1e3
    k2[:, 5:] *= -7
    v2[:, 5:] = 1e6
    alone = set_resolvent(q[:, :5], k[:, :5], v[:, :5])
    assert torch.allclose(full[:, :5], alone, atol=1e-12)
    assert torch.allclose(set_resolvent(q2, k2, v2, mask=mask)[:, :5], alone, atol=1e-12)
    assert (full[:, 5:] == 0).all()


# C4 precision and the Neumann fast path -----------------------------------------------------------
@pytest.mark.parametrize("K", KS)
def test_fp32_parity_with_float64_exact(K):
    q, k, v = draw(K)
    ref = set_resolvent(q, k, v)
    out = set_resolvent(q.float(), k.float(), v.float()).double()
    assert (out - ref).norm() / ref.norm().clamp_min(1e-30) <= 1e-5


@pytest.mark.parametrize("hops", [1, 2, 4, 8, 16])
def test_neumann_within_stated_bound(hops):
    rho = 0.9
    q, k, v = draw(33)
    err = (neumann_resolvent(q, k, v, rho=rho, hops=hops) - set_resolvent(q, k, v, rho=rho)).abs().amax(dim=(-2, -1))
    assert (err <= rho ** (hops + 1) / (1 - rho) * v.abs().amax(dim=(-2, -1)) + 1e-12).all()


# C5 low-precision inputs --------------------------------------------------------------------------
@pytest.mark.parametrize("dt", [torch.float16, torch.bfloat16])
def test_low_precision_input_returns_finite_in_input_dtype(dt):
    q, k, v = draw(16, dtype=torch.float32)
    out = set_resolvent(q.to(dt), k.to(dt), v.to(dt))
    assert out.dtype == dt and torch.isfinite(out).all()


def test_fp16_logit_overflow_is_survived():
    """|q.k| > 65504 overflows fp16 to inf; inf/inf = NaN unless the solve is promoted."""
    q = torch.full((1, 4, 8), 200.0, dtype=torch.float16)
    q[0, 1] *= -1
    assert torch.isfinite(set_resolvent(q, q, torch.randn(1, 4, 8).half())).all()


# C6 gradients, C7 determinism ---------------------------------------------------------------------
def test_gradcheck():
    q, k, v = (t.requires_grad_() for t in draw(6, d=3, B=2))
    assert torch.autograd.gradcheck(lambda a, b, c: set_resolvent(a, b, c), (q, k, v))


def test_determinism():
    q, k, v = draw(65, dtype=torch.float32)
    assert torch.equal(set_resolvent(q, k, v), set_resolvent(q, k, v))


@pytest.mark.skipif(not (torch.cuda.is_available() and os.environ.get("RJEPA_CUDA")), reason="set RJEPA_CUDA=1 on a CUDA box")
def test_cuda_parity():
    q, k, v = draw(64)
    out = set_resolvent(*(t.float().cuda() for t in (q, k, v))).double().cpu()
    ref = set_resolvent(q, k, v)
    assert (out - ref).norm() / ref.norm() <= 1e-5


# stochastic resolvent O = (1-g) P (I-gP)^-1 V -----------------------------------------------------
def dense_stochastic_ref(q, k, v, g):
    P = torch.softmax(q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5, -1)
    return (1 - g) * P @ torch.linalg.inv(torch.eye(q.shape[-2], dtype=q.dtype) - g * P) @ v


@pytest.mark.parametrize("K", [3, 8, 17, 65])
def test_stochastic_parity_with_dense_inverse(K):
    q, k, v = draw(K)
    ref = dense_stochastic_ref(q, k, v, 0.9)
    assert (stochastic_resolvent(q, k, v, g=0.9) - ref).norm() / ref.norm() <= 1e-10


@pytest.mark.parametrize("K", [3, 8, 17, 65])
def test_stochastic_permutation_equivariance(K):
    q, k, v = draw(K)
    p = torch.randperm(K, generator=torch.Generator().manual_seed(K))
    assert torch.allclose(stochastic_resolvent(q, k, v)[:, p],
                          stochastic_resolvent(q[:, p], k[:, p], v[:, p]), atol=1e-10, rtol=0)


@pytest.mark.parametrize("g", [1.0, 1.2])
def test_stochastic_g_at_or_above_one_refused(g):
    q, k, v = draw(4)
    with pytest.raises(ValueError):
        stochastic_resolvent(q, k, v, g=g)


def test_stochastic_output_is_convex_combination():
    """(1-g) P (I-gP)^-1 is row-stochastic and non-negative: the output lies inside the hull of v."""
    q, k, v = draw(33)
    out = stochastic_resolvent(q, k, v, g=0.99)
    assert (out <= v.amax(-2, keepdim=True) + 1e-9).all() and (out >= v.amin(-2, keepdim=True) - 1e-9).all()


def test_stochastic_all_masked_is_finite():
    q, k, v = draw(8)
    assert torch.isfinite(stochastic_resolvent(q, k, v, mask=torch.zeros(4, 8, dtype=torch.bool))).all()


def test_stochastic_padding_moves_nothing():
    q, k, v = draw(8)
    mask = torch.ones(4, 8, dtype=torch.bool)
    mask[:, 5:] = False
    q2, v2 = q.clone(), v.clone()
    q2[:, 5:] = 1e4
    v2[:, 5:] = 1e6
    alone = stochastic_resolvent(q[:, :5], k[:, :5], v[:, :5])
    assert torch.allclose(stochastic_resolvent(q2, k, v2, mask=mask)[:, :5], alone, atol=1e-12)


@pytest.mark.parametrize("dt", [torch.float16, torch.bfloat16])
def test_stochastic_low_precision_finite(dt):
    q, k, v = draw(16, dtype=torch.float32)
    out = stochastic_resolvent(q.to(dt) * 30, k.to(dt) * 30, v.to(dt))
    assert out.dtype == dt and torch.isfinite(out).all()


def test_stochastic_gradcheck():
    q, k, v = (t.requires_grad_() for t in draw(6, d=3, B=2))
    assert torch.autograd.gradcheck(lambda a, b, c: stochastic_resolvent(a, b, c), (q, k, v))


@pytest.mark.parametrize("op", ["signed", "stoch"])
def test_gradient_bounded_at_small_input_scale(op):
    """Without the row-L1 floor, |dA/dq| ~ 1/|q| blows up as the logit scale goes to zero."""
    f = set_resolvent if op == "signed" else stochastic_resolvent
    grads = []
    for scale in (1.0, 1e-8):
        q, k, v = draw(16, B=1, seed=3)
        q = (q * scale).requires_grad_()
        f(q, k * scale, v).sum().backward()
        grads.append(q.grad.norm().item())
    assert grads[1] <= 1e4 * grads[0], grads


# T-A: a linear permutation-equivariant resolvent is rank-inert ------------------------------------
def test_linear_equivariant_resolvent_is_rank_inert_300_draws():
    """For W = alpha I + beta J and rho(gamma W) < 1, (I - gamma W)^-1 s keeps the ranks of s.
    Values are separated by >= 1e-2 so float ties cannot masquerade as rank changes."""
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(300):
        n = int(rng.integers(2, 13))
        s = rng.choice(np.arange(-1000, 1001), n, replace=False) / 100
        a, b, g = rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0.01, 0.99)
        out = linear_equivariant_resolvent(s, a, b, g)
        if out is None:
            continue
        checked += 1
        assert (np.argsort(out) == np.argsort(s)).all()
    assert checked >= 100                           # 139 of the 300 draws fall inside rho(gamma W) < 1


def test_linear_equivariant_resolvent_refuses_pole():
    assert linear_equivariant_resolvent(np.array([1.0, 2.0]), 1.0, 0.0, 0.999999999) is None


def test_resolvent_apply_matches_inverse_and_neumann():
    W = torch.softmax(torch.randn(8, 4, 4, dtype=torch.float64), -1)
    h = torch.randn(8, 4, 5, dtype=torch.float64)
    a = resolvent_apply(W, h, 0.9)
    neu, t = h.clone(), h.clone()
    for _ in range(600):
        t = 0.9 * W @ t
        neu = neu + t
    assert torch.allclose(a, torch.linalg.inv(torch.eye(4, dtype=torch.float64) - 0.9 * W) @ h, atol=1e-10)
    assert torch.allclose(a, neu, atol=1e-10)


def test_row_stochastic_resolvent_bounded_by_one_over_one_minus_gamma():
    for g in (0.1, 0.5, 0.9, 0.99):
        W = torch.softmax(torch.randn(32, 4, 4, dtype=torch.float64) * 3, -1)
        Rm = torch.linalg.inv(torch.eye(4, dtype=torch.float64) - g * W)
        assert Rm.abs().sum(-1).max() <= 1 / (1 - g) + 1e-9

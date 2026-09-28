"""Correctness contracts for the resolvent as a K-candidate operator (BAR.md C1-C8).

    python -m pytest sun/rjepa/chase/test_set_resolvent.py -q
"""
import math
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path[:0] = [HERE, REPO]
from rjepa_ops import set_resolvent, neumann_resolvent, build_A  # noqa: E402
from ceq.attention import ceq_operator, path_sum  # noqa: E402

KS = [1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65]


def draw(K, d=8, B=4, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(B, K, d, generator=g, dtype=dtype) for _ in range(3))


# C1 ---------------------------------------------------------------------------------------------
def test_repo_operator_as_is_is_not_permutation_equivariant():
    """The shipped ceq_operator is strictly causal: on an unordered candidate set its output
    depends on the order the candidates were listed in. This test PASSES when that is true --
    it documents why the operator cannot be reused as-is."""
    q, k, v = draw(16)
    p = torch.randperm(16, generator=torch.Generator().manual_seed(1))
    out = path_sum(ceq_operator(q, k), v, 15)
    outp = path_sum(ceq_operator(q[:, p], k[:, p]), v[:, p], 15)
    assert (out[:, p] - outp).abs().max() > 1e-3


@pytest.mark.parametrize("K", KS)
def test_permutation_equivariance(K):
    q, k, v = draw(K)
    p = torch.randperm(K, generator=torch.Generator().manual_seed(K))
    out = set_resolvent(q, k, v)
    outp = set_resolvent(q[:, p], k[:, p], v[:, p])
    assert torch.allclose(out[:, p], outp, atol=1e-10, rtol=0)


# C2 ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("rho", [1.0, 1.5])
def test_rho_at_or_above_one_is_refused(rho):
    q, k, v = draw(4)
    with pytest.raises(ValueError):
        set_resolvent(q, k, v, rho=rho)


def test_rho_just_below_one_two_candidates_is_bounded():
    """K=2 signed: A = [[0, +-rho], [+-rho, 0]], eigenvalues +-rho -> pole at rho=1."""
    q = torch.tensor([[[1.0], [1.0]]], dtype=torch.float64)
    k = q.clone()
    v = torch.ones(1, 2, 1, dtype=torch.float64)
    out = set_resolvent(q, k, v, rho=0.999)
    assert torch.isfinite(out).all()
    assert out.abs().max() <= 1.0 / (1 - 0.999) + 1e-6


def test_spectral_radius_below_rho_on_1000_draws():
    worst = 0.0
    for s in range(1000):
        q, k, _ = draw(9, d=4, B=1, seed=s)
        A = build_A(q, k, rho=0.9)[0]
        worst = max(worst, torch.linalg.eigvals(A).abs().max().item())
    assert worst <= 0.9 + 1e-12


def test_resolvent_inf_norm_bound():
    q, k, _ = draw(33)
    A = build_A(q, k, rho=0.9)
    R = torch.linalg.inv(torch.eye(33, dtype=A.dtype) - A)
    assert R.abs().sum(-1).max() <= 1 / (1 - 0.9) + 1e-9


# C3 ---------------------------------------------------------------------------------------------
def test_single_candidate_is_identity_and_finite():
    q, k, v = draw(1)
    out = set_resolvent(q, k, v)
    assert torch.isfinite(out).all() and torch.equal(out, v)


def test_all_masked_is_finite():
    q, k, v = draw(8)
    mask = torch.zeros(4, 8, dtype=torch.bool)
    out = set_resolvent(q, k, v, mask=mask)
    assert torch.isfinite(out).all()


def test_zero_features_is_finite():
    q = torch.zeros(2, 5, 3, dtype=torch.float64)
    out = set_resolvent(q, q, torch.ones(2, 5, 3, dtype=torch.float64))
    assert torch.isfinite(out).all()


def test_masked_candidate_moves_nothing():
    """Padding a set with junk candidates must leave the real candidates' outputs unchanged."""
    q, k, v = draw(8)
    mask = torch.ones(4, 8, dtype=torch.bool)
    mask[:, 5:] = False
    full = set_resolvent(q, k, v, mask=mask)
    q2, k2, v2 = q.clone(), k.clone(), v.clone()
    q2[:, 5:] *= 1e3; k2[:, 5:] *= -7; v2[:, 5:] = 1e6
    junk = set_resolvent(q2, k2, v2, mask=mask)
    alone = set_resolvent(q[:, :5], k[:, :5], v[:, :5])
    assert torch.allclose(full[:, :5], alone, atol=1e-12)
    assert torch.allclose(junk[:, :5], alone, atol=1e-12)
    assert (full[:, 5:] == 0).all()


# C4 ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("K", KS)
def test_fp32_parity_with_float64_exact(K):
    q, k, v = draw(K)
    ref = set_resolvent(q, k, v)
    out = set_resolvent(q.float(), k.float(), v.float()).double()
    rel = (out - ref).norm() / ref.norm().clamp_min(1e-30)
    assert rel <= 1e-5, rel


@pytest.mark.parametrize("hops", [1, 2, 4, 8, 16])
def test_neumann_within_stated_bound(hops):
    rho = 0.9
    q, k, v = draw(33)
    ref = set_resolvent(q, k, v, rho=rho)
    out = neumann_resolvent(q, k, v, rho=rho, hops=hops)
    err = (out - ref).abs().amax(dim=(-2, -1))
    bound = rho ** (hops + 1) / (1 - rho) * v.abs().amax(dim=(-2, -1))
    assert (err <= bound + 1e-12).all()


# C5 ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dt", [torch.float16, torch.bfloat16])
def test_low_precision_input_returns_finite_in_input_dtype(dt):
    q, k, v = draw(16, dtype=torch.float32)
    out = set_resolvent(q.to(dt), k.to(dt), v.to(dt))
    assert out.dtype == dt and torch.isfinite(out).all()


def test_fp16_logit_overflow_is_survived():
    """|q.k| > 65504 overflows fp16 to inf; inf/inf = NaN unless the solve is promoted."""
    q = torch.full((1, 4, 8), 200.0, dtype=torch.float16)
    q[0, 1] *= -1
    v = torch.randn(1, 4, 8).half()
    out = set_resolvent(q, q, v)
    assert torch.isfinite(out).all()


# C6 ---------------------------------------------------------------------------------------------
def test_gradcheck():
    q, k, v = (t.requires_grad_() for t in draw(6, d=3, B=2))
    assert torch.autograd.gradcheck(lambda a, b, c: set_resolvent(a, b, c), (q, k, v))


# C7 ---------------------------------------------------------------------------------------------
def test_determinism():
    q, k, v = draw(65, dtype=torch.float32)
    assert torch.equal(set_resolvent(q, k, v), set_resolvent(q, k, v))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")
def test_cuda_parity():
    q, k, v = draw(64)
    ref = set_resolvent(q, k, v)
    out = set_resolvent(*(t.float().cuda() for t in (q, k, v))).double().cpu()
    assert ((out - ref).norm() / ref.norm()) <= 1e-5


# ---------------------------------------------------------------- Wilson's form: O = (1-g) P (I-gP)^-1 V
from rjepa_ops import stochastic_resolvent  # noqa: E402


def dense_stochastic_ref(q, k, v, g, mask=None):
    w = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
    if mask is not None:
        w = w.masked_fill(~(mask[..., None, :]), float("-inf"))
    P = torch.softmax(w, -1)
    I = torch.eye(q.shape[-2], dtype=q.dtype)
    return (1 - g) * P @ torch.linalg.inv(I - g * P) @ v


@pytest.mark.parametrize("K", [3, 8, 17, 65])
def test_stochastic_parity_with_dense_inverse(K):
    q, k, v = draw(K)
    ref = dense_stochastic_ref(q, k, v, 0.9)
    out = stochastic_resolvent(q, k, v, g=0.9)
    assert ((out - ref).norm() / ref.norm()) <= 1e-10


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
    """(1-g) P (I-gP)^-1 is row-stochastic and non-negative: output inside the hull of v."""
    q, k, v = draw(33)
    out = stochastic_resolvent(q, k, v, g=0.99)
    assert (out <= v.amax(-2, keepdim=True) + 1e-9).all() and (out >= v.amin(-2, keepdim=True) - 1e-9).all()


def test_stochastic_all_masked_is_finite():
    q, k, v = draw(8)
    out = stochastic_resolvent(q, k, v, mask=torch.zeros(4, 8, dtype=torch.bool))
    assert torch.isfinite(out).all()


def test_stochastic_padding_moves_nothing():
    q, k, v = draw(8)
    mask = torch.ones(4, 8, dtype=torch.bool); mask[:, 5:] = False
    q2 = q.clone(); q2[:, 5:] = 1e4; v2 = v.clone(); v2[:, 5:] = 1e6
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


# C6b: gradient conditioning at small logit scale (the repo's eps note, ceq/attention.py ceq_operator)
@pytest.mark.parametrize("op", ["signed", "stoch"])
def test_gradient_bounded_at_small_input_scale(op):
    f = set_resolvent if op == "signed" else stochastic_resolvent
    grads = []
    for scale in (1.0, 1e-8):
        q, k, v = draw(16, B=1, seed=3)
        q = (q * scale).requires_grad_()
        f(q, k * scale, v).sum().backward()
        grads.append(q.grad.norm().item())
    assert grads[1] <= 1e4 * grads[0], grads

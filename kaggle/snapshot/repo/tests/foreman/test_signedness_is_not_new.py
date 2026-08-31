"""Q3 -- is SIGNEDNESS alone enough to be new?

The module's headline is a minimum influence-Jacobian entry of -9.000e-01 against
EXACTLY 0.000e+00 for a non-negative control, resting on the general statement
that the Kleene star of a non-negative matrix has a non-negative influence
Jacobian in any ordered semiring. The general statement is correct. The question
is whether anyone already ships an operator on the other side of it.

Three published mechanisms are reimplemented here from their defining equations
and measured on the same instrument as the module:

  SimA, Koohpayegani & Pirsiavash, arXiv 2206.08898 (2022)
      O = Qhat Khat^T V,   Qhat_i = Q_i / ||Q_i||_1,  Khat_j = K_j / ||K_j||_1
      L1 normalization in place of softmax. Its own text: "the attention values
      can become negative, meaning that a token can affect another one
      negatively. This is in contrast to regular transformers where the attention
      is always non-negative." That is the module's tier-3 claim, from 2022.

  Differential Transformer, Ye et al., arXiv 2410.05258 (ICLR 2025)
      DiffAttn(X) = (softmax(Q1 K1^T/sqrt(d)) - lambda softmax(Q2 K2^T/sqrt(d))) V
      lambda = exp(lq1.lk1) - exp(lq2.lk2) + lambda_init, learned per head, with
      no positivity constraint in the parameterization.

  DeltaNet's UT transform, Yang et al., arXiv 2406.06484 (NeurIPS 2024), Eq. (10)
      T = (I + tril(diag(beta) K K^T, -1))^{-1} diag(beta)
      A STRICTLY lower triangular matrix with SIGNED entries, inverted exactly.
      The paper: "The inverse of lower triangular matrices could be solved
      efficiently using forward substitution."

The third is the sharper hit, because it is signed AND multi-hop AND exact: an
inverse of I minus a strictly-lower-triangular signed matrix is a finite path
sum, by the same nilpotency `CEQ.Nilpotent.pow_card_eq_zero` proves. That
nilpotency is not incidental prior art either -- it is the stated justification
inside shipping inference kernels. sglang's Kimi-Delta-Attention prefill kernel
`Akk_inverse_lower_triangle_bf16.py` documents its algorithm as

    (I+L)^-1 = (I-L)(I+L^2)(I+L^4)(I+L^8)   [L strictly lower triangular, L^16=0]

which is the terminating resolvent, factored for a 64x64 tile.
"""

import math

import pytest
import torch

from _device import DEVICES
from ceq.attention import DEFAULT_RHO, ceq_operator

SEED = 20260824


def draw(s, d, device, n=2, seed=SEED):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return [torch.randn(1, 1, s, d, generator=g, dtype=torch.float64).to(device)
            for _ in range(n)]


def causal(w):
    s = w.shape[-2]
    m = torch.ones(s, s, dtype=torch.bool, device=w.device).tril(0)
    return w, m


def sima_weights(q, k):
    """SimA, arXiv 2206.08898: L1-normalize Q and K per token, no softmax."""
    qh = q / q.abs().sum(-1, keepdim=True)
    kh = k / k.abs().sum(-1, keepdim=True)
    w, m = causal(qh @ kh.transpose(-2, -1))
    return w.masked_fill(~m, 0.0)


def diff_attn_weights(q1, k1, q2, k2, lam):
    """Differential Transformer, arXiv 2410.05258: difference of two softmaxes."""
    d = q1.shape[-1]
    w1, m = causal((q1 @ k1.transpose(-2, -1)) / math.sqrt(d))
    w2, _ = causal((q2 @ k2.transpose(-2, -1)) / math.sqrt(d))
    s1 = torch.softmax(w1.masked_fill(~m, float("-inf")), -1)
    s2 = torch.softmax(w2.masked_fill(~m, float("-inf")), -1)
    return s1 - lam * s2


def softmax_weights(q, k):
    """The non-negative control the module measures 0.000e+00 against."""
    w, m = causal((q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1]))
    return torch.softmax(w.masked_fill(~m, float("-inf")), -1)


def ceq_influence(q, k):
    """d out / d v for the module: (I - A)^{-1}, A strictly causal and signed."""
    a = ceq_operator(q, k)
    eye = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
    return torch.linalg.inv(eye - a)


def deltanet_ut_matrix(k, beta):
    """The matrix DeltaNet Eq. (10) inverts: I + tril(diag(beta) K K^T, -1)."""
    s = k.shape[-2]
    m = torch.ones(s, s, dtype=torch.bool, device=k.device).tril(-1)
    return (beta[..., None] * (k @ k.transpose(-2, -1))).masked_fill(~m, 0.0)


# ==========================================================================
# CLAIM AS WRITTEN -- RED
# ==========================================================================

@pytest.mark.parametrize("device", DEVICES)
def test_claim_no_published_attention_reaches_a_negative_influence_jacobian(device):
    """The module's novelty rests on being the only operator on the signed side
    of the ordered-semiring statement. Three published ones, same instrument."""
    q, k, q2, k2 = draw(64, 16, device, n=4)
    mins = {
        "softmax control": float(softmax_weights(q, k).min()),
        "SimA 2206.08898": float(sima_weights(q, k).min()),
        "DiffAttn 2410.05258 (lam=0.8)":
            float(diff_attn_weights(q, k, q2, k2, 0.8).min()),
        "ceq path sum": float(ceq_influence(q, k).min()),
    }
    print("\n  minimum influence-Jacobian entry")
    for name, val in mins.items():
        print("    %-32s %+.6e" % (name, val))
    published = min(mins["SimA 2206.08898"], mins["DiffAttn 2410.05258 (lam=0.8)"])
    assert published >= 0.0, (
        f"published attention reaches {published:+.6e}. SimA (2022) L1-normalizes "
        f"Q and K instead of applying softmax and states the negativity property "
        f"in its own text; Differential Transformer (2024) subtracts two softmax "
        f"maps. Signedness alone is not new, so the -9.000e-01 vs 0.000e+00 "
        f"result is a rediscovery of SimA's observation, not a first.")


@pytest.mark.parametrize("device", DEVICES)
def test_claim_an_influence_entry_of_exactly_minus_rho_can_come_from_a_free_row(device):
    """The reported headline is -9.000e-01, which is -rho to every figure given.
    Row 1 has one predecessor so A[1,0] = +/- rho exactly. Every other row splits
    rho across two or more predecessors, and a path sum of entries whose absolute
    values sum to rho is strictly inside rho. So an influence entry lands on
    -rho exactly only at (1,0) -- the frozen one. Scan draws and check where."""
    hits, off_row_one, worst_other = 0, [], 0.0
    for seed in range(24):
        q, k = draw(96, 16, device, seed=SEED + seed)
        j = ceq_influence(q, k)[0, 0]
        at = ((j + DEFAULT_RHO).abs() < 1e-12).nonzero()
        hits += len(at)
        off_row_one += [tuple(p.tolist()) for p in at if int(p[0]) != 1]
        strict = j.clone()
        strict[1, 0] = 0.0
        worst_other = min(worst_other, float(strict.min()))
    print("\n  entries equal to -rho over 24 draws: %d, of which off row 1: %d"
          % (hits, len(off_row_one)))
    print("  most negative influence entry excluding (1,0): %+.6e (|.| < rho = %.2f)"
          % (worst_other, DEFAULT_RHO))
    assert off_row_one, (
        f"all {hits} entries equal to -rho sit at (1,0), and the most negative "
        f"entry anywhere else is {worst_other:+.6e}, strictly inside -rho. The "
        f"advertised -9.000e-01 can only have been read off A[1,0], the "
        f"single-predecessor entry that is a frozen +/- rho constant with exactly "
        f"zero gradient -- see test_claim_the_first_attention_row_is_trainable.")


# ==========================================================================
# WHAT THE PRIOR ART ALREADY SHIPS -- GREEN
# ==========================================================================

@pytest.mark.parametrize("device", DEVICES)
def test_the_deltanet_ut_transform_inverts_a_signed_strictly_triangular_matrix(device):
    """DeltaNet Eq. (10), arXiv 2406.06484. Same object class as the module's A:
    strictly lower triangular, signed, inverted exactly."""
    k, _ = draw(48, 16, device)
    beta = torch.sigmoid(k.sum(-1))
    m = deltanet_ut_matrix(k, beta)
    print("\n  DeltaNet UT matrix: strict-upper max |.| = %.3e   min entry %+.6f"
          % (float(m.triu(0).abs().max()), float(m.min())))
    assert float(m.triu(0).abs().max()) == 0.0, "not strictly lower triangular"
    assert float(m.min()) < 0.0, "not signed"


@pytest.mark.parametrize("device", DEVICES)
def test_the_finite_path_sum_inverts_the_deltanet_matrix_too(device):
    """The module's terminating-resolvent property is a property of strictly
    triangular matrices, not of the module. Run the module's own `path_sum` on
    DeltaNet's matrix and on sglang's factored form and get the same inverse."""
    k, _ = draw(32, 16, device)
    beta = torch.sigmoid(k.sum(-1))
    m = -deltanet_ut_matrix(k, beta)          # DeltaNet inverts I + tril(...)
    s = m.shape[-1]
    eye = torch.eye(s, dtype=m.dtype, device=m.device)

    series = sum((torch.linalg.matrix_power(m, i) for i in range(1, s)), eye)
    exact = torch.linalg.inv(eye - m)

    # sglang Akk_inverse_lower_triangle_bf16.py: (I+L)^-1 = (I-L)(I+L^2)(I+L^4)...
    l, prod = -m, eye + m
    p = l @ l
    while float(p.abs().max()) > 0.0:
        prod = prod @ (eye + p)
        p = p @ p
    print("\n  |finite series - inv| %.3e    |sglang factored - inv| %.3e"
          % (float((series - exact).abs().max()),
             float((prod - exact).abs().max())))
    assert torch.allclose(series, exact, atol=1e-10)
    assert torch.allclose(prod, exact, atol=1e-10)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s", "--tb=short",
                                  "-p", "no:cacheprovider"]))

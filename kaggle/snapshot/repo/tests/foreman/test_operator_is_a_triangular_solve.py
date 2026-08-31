"""Q1 -- is `out = v + Av + A^2 v + ... + A^K v`, A strictly lower triangular,
secretly a known object?

The house has done this before. R2's max-plus star turned out to be the APPNP of
its own greedy policy at 9.95e-14, and a linear resolvent stage reproduced APPNP
Eq. (3) at 2.22e-16. Same instrument, pointed at the operator that is left.

Four reductions are tested, cheapest first.

  1. FORWARD SUBSTITUTION.  A strictly lower triangular means `I - A` is unit
     lower triangular, so `(I - A)^{-1} v` is the triangular solve every linear
     algebra text calls forward substitution (Golub & Van Loan, Alg. 3.1.1).
     `pow_card_eq_zero` -- strictly triangular implies A^n = 0 -- is the standard
     nilpotency of a strictly triangular matrix; it is the reason forward
     substitution terminates, restated in Lean.

  2. THE NEUMANN / JACOBI ITERATION.  `sum_{k<=K} A^k v` is not merely LIKE the
     iteration `z <- A z + v` started from z_0 = 0; expanding K+1 steps of that
     iteration gives that sum term for term.  `_lib.neumann_iterate` has been in
     this repo since round 0, labelled "the equilibrium solver of decision 1".

  3. A LINEAR-ATTENTION RNN WITH OUTPUT FEEDBACK.  A_ij factorizes: with
     w_ij = q_i . k_j, A_ij = rho * w_ij / sum_{j<i} |w_ij| = c_i * (q_i . k_j).
     So the row sum in reduction 1 is

         z_i = v_i + c_i * q_i^T S_i,     S_i = sum_{j<i} k_j z_j^T

     which is Katharopoulos et al. 2020's recurrent linear attention with one
     change: the d x d state accumulates the layer's own OUTPUT z_j rather than
     the value v_j.  If that identity holds numerically the operator is a linear
     RNN and its O(K S^2 d) cost is a choice.

  4. A VOLTERRA SERIES.  A is quadratic in (q,k) and v is linear, so a naive
     reading makes the output a degree-2K+1 polynomial in the input.  Test the
     degree by scaling the input.

Reduction 3 is the load-bearing one, because it also isolates what is NOT linear
attention: the scalar c_i needs sum_{j<i} |q_i . k_j|, and |.| does not
accumulate into a fixed-size state. The absolute value in the normalizer is the
whole reason this operator is quadratic.
"""

import pytest
import torch

from _device import DEVICES
from _lib import neumann_iterate
from ceq.attention import ceq_operator, path_sum

SEED = 20260824


def draw(s, d, device, heads=2, seed=SEED):
    g = torch.Generator(device="cpu").manual_seed(seed)
    mk = lambda *shape: torch.randn(*shape, generator=g, dtype=torch.float64).to(device)
    return mk(1, heads, s, d), mk(1, heads, s, d), mk(1, heads, s, d)


def forward_substitution(a, v):
    """(I - A)^{-1} v by the unit lower triangular solve. Golub & Van Loan 3.1.1."""
    eye = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
    return torch.linalg.solve_triangular(eye - a, v, upper=False, unitriangular=True)


def linear_attention_with_output_feedback(q, k, v, rho=0.9):
    """Katharopoulos et al. 2020 recurrent form, state accumulating z not v.

        S_i = sum_{j<i} k_j z_j^T          (d x d, fixed size)
        z_i = v_i + c_i * q_i^T S_i

    The 1/sqrt(d) scale cancels between numerator and normalizer, so it is
    absent here on purpose -- its absence is part of the claim.
    """
    s = q.shape[-2]
    w = q @ k.transpose(-2, -1)
    m = torch.ones(s, s, dtype=torch.bool, device=w.device).tril(-1)
    c = rho / w.masked_fill(~m, 0.0).abs().sum(-1).clamp_min(
        torch.finfo(w.dtype).tiny)
    state = torch.zeros(*q.shape[:-2], q.shape[-1], v.shape[-1],
                        dtype=v.dtype, device=v.device)
    out = []
    for i in range(s):
        qi, ki, vi = q[..., i, :], k[..., i, :], v[..., i, :]
        zi = vi + c[..., i, None] * (qi.unsqueeze(-2) @ state).squeeze(-2)
        out.append(zi)
        state = state + ki.unsqueeze(-1) @ zi.unsqueeze(-2)
    return torch.stack(out, dim=-2)


# ==========================================================================
# CLAIM AS WRITTEN -- RED
# ==========================================================================

@pytest.mark.parametrize("device", DEVICES)
def test_claim_the_exact_path_sum_is_not_a_forward_substitution(device):
    """If the terminating path sum is a new object it must compute something the
    unit lower triangular solve does not."""
    worst = 0.0
    for s in (8, 32, 64):
        q, k, v = draw(s, 16, device)
        a = ceq_operator(q, k)
        ours = path_sum(a, v, hops=s - 1)
        theirs = forward_substitution(a, v)
        worst = max(worst, float((ours - theirs).abs().max()))
    print("\n  max |path_sum(K=S-1) - solve_triangular(I-A)| : %.3e" % worst)
    assert worst > 1e-10, (
        f"the two are the same array to {worst:.3e}: the terminating path sum IS "
        f"forward substitution on the unit lower triangular system (I - A) z = v, "
        f"Golub & Van Loan Alg. 3.1.1. The Lean theorem pow_card_eq_zero is the "
        f"standard nilpotency of a strictly triangular matrix, which is WHY that "
        f"solve terminates.")


@pytest.mark.parametrize("device", DEVICES)
def test_claim_the_truncated_path_sum_is_not_a_neumann_iteration(device):
    """`_lib.neumann_iterate` is already in this repo. K+1 of its steps expand to
    sum_{k<=K} A^k v, term for term."""
    worst = 0.0
    for hops in (1, 2, 3, 4, 8):
        q, k, v = draw(48, 16, device)
        a = ceq_operator(q, k)
        ours = path_sum(a, v, hops=hops)
        theirs = torch.as_tensor(
            neumann_iterate(a[0, 0].cpu().numpy(), v[0, 0].cpu().numpy(),
                            iters=hops + 1, device="cpu"),
            dtype=torch.float64)
        worst = max(worst, float((ours[0, 0].cpu() - theirs).abs().max()))
    print("\n  max |path_sum(K) - neumann_iterate(K+1)| : %.3e" % worst)
    assert worst > 1e-12, (
        f"identical to {worst:.3e} at every K tested: the K-hop path sum is K+1 "
        f"steps of the Jacobi/Neumann iteration z <- A z + v from z_0 = 0, which "
        f"_lib.neumann_iterate has implemented in this repo since round 0.")


@pytest.mark.parametrize("device", DEVICES)
def test_claim_the_operator_is_not_a_linear_attention_rnn(device):
    """A_ij = c_i * (q_i . k_j) factorizes, so the triangular solve has a
    fixed-size-state recurrent form. If it matches, the operator is a linear RNN
    in the sense of Katharopoulos et al. 2020, arXiv 2006.16236."""
    worst = 0.0
    for s in (8, 32, 64):
        q, k, v = draw(s, 16, device)
        a = ceq_operator(q, k)
        ours = path_sum(a, v, hops=s - 1)
        theirs = linear_attention_with_output_feedback(q, k, v)
        worst = max(worst, float((ours - theirs).abs().max() / ours.abs().max()))
    print("\n  max relative |path_sum(K=S-1) - linear-attention recurrence| : %.3e"
          % worst)
    assert worst > 1e-8, (
        f"same array to {worst:.3e} relative: the operator IS recurrent linear "
        f"attention whose d x d state accumulates the layer's own output z_j "
        f"instead of the value v_j. The quadratic path-sum form is a choice, "
        f"not a requirement.")


@pytest.mark.parametrize("device", DEVICES)
def test_claim_the_path_sum_is_a_volterra_series_in_its_input(device):
    """A Volterra / polynomial-network reading would make the output a degree
    2K+1 polynomial in the input. Scale the input and read the degree off."""
    q, k, v = draw(32, 16, device)
    base = path_sum(ceq_operator(q, k), v, hops=3)
    dev = []
    for scale in (0.5, 2.0, 8.0):
        got = path_sum(ceq_operator(scale * q, scale * k), scale * v, hops=3)
        dev.append(float((got - scale * base).abs().max() / base.abs().max()))
    print("\n  relative |f(t x) - t f(x)| at t = 0.5, 2, 8 : %s"
          % ["%.3e" % d for d in dev])
    assert max(dev) > 1e-10, (
        f"f(t x) = t f(x) to {max(dev):.3e}: the operator is homogeneous of "
        f"degree ONE, not a degree 2K+1 polynomial. The L1 normalizer makes A "
        f"invariant to the scale of q and k, so there is no Volterra series here "
        f"to be a special case of.")


# ==========================================================================
# WHAT THE OBJECT ACTUALLY IS -- GREEN
# ==========================================================================

@pytest.mark.parametrize("device", DEVICES)
def test_the_absolute_value_in_the_normalizer_is_the_only_non_recurrent_part(device):
    """Two keys with IDENTICAL contribution to every linear accumulator but a
    non-zero contribution to sum_j |q . k_j|. A fixed-size linear state cannot
    tell them apart, so the normalizer -- not the path sum -- is what forbids an
    O(S) form."""
    g = torch.Generator().manual_seed(SEED)
    q = torch.randn(4, generator=g, dtype=torch.float64).to(device)
    kk = torch.stack([q, -q])
    ident = float((kk[0] + kk[1]).abs().max())
    l1 = float((q @ kk.T).abs().sum())
    print("\n  linear state of {+q, -q}: %.3e     sum |q.k_j|: %.6f" % (ident, l1))
    assert ident < 1e-14 and l1 > 1e-3, (
        "the two keys cancel in every linear accumulator yet contribute twice "
        "over in the L1 normalizer")


@pytest.mark.parametrize("device", DEVICES)
def test_hops_beyond_the_sequence_length_change_nothing(device):
    """Nilpotency measured rather than assumed: A^s = 0 exactly, so K = S-1 and
    K = 4S are the same output bit for bit."""
    q, k, v = draw(24, 8, device)
    a = ceq_operator(q, k)
    short = path_sum(a, v, hops=23)
    long_ = path_sum(a, v, hops=96)
    print("\n  |path_sum(K=23) - path_sum(K=96)| : %.3e"
          % float((short - long_).abs().max()))
    assert torch.equal(short, long_)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s", "--tb=short",
                                  "-p", "no:cacheprovider"]))

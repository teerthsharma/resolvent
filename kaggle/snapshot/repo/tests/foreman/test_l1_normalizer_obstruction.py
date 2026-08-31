"""Q2 -- is the pure operator trainable, and if not, what is the structural
obstruction?

The suspicion handed to FOREMAN was that `A = rho * w / sum|w|` is ill-conditioned:
not smooth at w = 0, Jacobian carrying a rank-one correction,

    d f_i / d w_j = (delta_ij - f_i * sign(w_j)) / ||w||_1,     f = w / ||w||_1

so gradients ought to blow up where a row nearly cancels. That is a testable
claim and it is tested here, against the softmax arm at matched initialization,
on the real corpus. It does not survive.

What does survive is a different and much harder obstruction, visible in the same
Jacobian: J w = 0 exactly. The map is homogeneous of degree ZERO, so

    A(t q, k) = A(q, k)   for every t > 0

and the query magnitude is an exact null direction of every gradient that passes
through the operator. Softmax has no such degeneracy -- scaling its logits is
precisely how it sharpens. So the signed operator has no temperature channel, and
its row L1 mass is pinned at rho no matter what the logits do. It can move
attention around within a row; it cannot decide to attend more, or less, or at
all.

Two smaller exact zeros come with it. Row 1 has a single predecessor, so
A[1,0] = rho * w / |w| = +/- rho identically -- a frozen constant with zero
gradient and a jump of 2 rho where the logit crosses zero. And w / ||w||_1 is
discontinuous at w = 0 rather than merely non-smooth, since the limit depends on
the approach direction.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from _device import DEVICES
from ceq import lm
from ceq.attention import DEFAULT_RHO, ceq_operator, path_sum

SEED = 20260824
CORPUS = "data/tinystories_20k.txt"


def draw(s, d, device, seed=SEED):
    g = torch.Generator(device="cpu").manual_seed(seed)
    mk = lambda: torch.randn(1, 1, s, d, generator=g, dtype=torch.float64).to(device)
    return mk(), mk(), mk()


def causal_softmax_rows(q, k):
    s = q.shape[-2]
    w = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    m = torch.ones(s, s, dtype=torch.bool, device=w.device).tril(0)
    return torch.softmax(w.masked_fill(~m, float("-inf")), dim=-1)


# ==========================================================================
# CLAIM AS WRITTEN -- RED
# ==========================================================================

@pytest.mark.parametrize("device", DEVICES)
def test_claim_the_operator_can_sharpen_its_attention_by_scaling_its_logits(device):
    """Sharpening by logit scale is how softmax attention becomes selective
    during training. Scale the queries and read the peak weight off both arms."""
    q, k, _ = draw(32, 16, device)
    signed, soft = [], []
    for t in (0.25, 1.0, 4.0, 16.0):
        signed.append(float(ceq_operator(t * q, k)[0, 0, -1].abs().max()))
        soft.append(float(causal_softmax_rows(t * q, k)[0, 0, -1].max()))
    print("\n  last-row peak weight at logit scale 0.25 / 1 / 4 / 16")
    print("    signed  : %s" % ["%.6f" % x for x in signed])
    print("    softmax : %s" % ["%.6f" % x for x in soft])
    assert max(signed) - min(signed) > 1e-6, (
        f"the signed operator's peak weight is {signed[0]:.6f} at every scale "
        f"tested, a spread of {max(signed) - min(signed):.3e}, while softmax "
        f"sweeps {min(soft):.4f} -> {max(soft):.4f}. The row L1 normalizer is "
        f"homogeneous of degree ZERO in the logits, so the operator has NO "
        f"temperature channel. It can redistribute attention within a row; it "
        f"cannot decide to attend more or less.")


@pytest.mark.parametrize("device", DEVICES)
def test_claim_the_query_magnitude_carries_gradient(device):
    """Degree-zero homogeneity means J w = 0. Read it off the real gradient:
    the radial component grad_q . q must vanish exactly."""
    q, k, v = draw(32, 16, device)
    q, k = q.requires_grad_(), k.requires_grad_()
    path_sum(ceq_operator(q, k), v, 3).pow(2).sum().backward()
    radial = float((q.grad * q).sum(-1).abs().max())

    q2 = q.detach().clone().requires_grad_()
    k2 = k.detach().clone().requires_grad_()
    F.scaled_dot_product_attention(q2, k2, v, is_causal=True).pow(2).sum().backward()
    control = float((q2.grad * q2).sum(-1).abs().max())

    print("\n  max |grad_q . q|   signed %.3e   softmax %.3e" % (radial, control))
    assert radial > 1e-8, (
        f"grad_q . q = {radial:.3e} for the signed operator against {control:.3f} "
        f"for softmax on the same tensors. The query magnitude is an EXACT null "
        f"direction of the signed operator's gradient. Every query vector loses "
        f"one of its d degrees of freedom, permanently.")


@pytest.mark.parametrize("device", DEVICES)
def test_claim_the_first_attention_row_is_trainable(device):
    """Row 1 has exactly one predecessor, so A[1,0] = rho * w/|w| = +/- rho."""
    q, k, _ = draw(16, 8, device)
    q, k = q.requires_grad_(), k.requires_grad_()
    a = ceq_operator(q, k)
    a[0, 0, 1, 0].backward()
    val = float(a[0, 0, 1, 0])
    gq, gk = float(q.grad.abs().max()), float(k.grad.abs().max())
    print("\n  A[1,0] = %+.6f   (rho = %.2f)   |grad_q| = %.3e  |grad_k| = %.3e"
          % (val, DEFAULT_RHO, gq, gk))
    assert max(gq, gk) > 0.0, (
        f"A[1,0] = {val:+.6f} = +/- rho exactly and its gradient is {gq:.1e}. The "
        f"single-predecessor row is a frozen constant of magnitude rho in every "
        f"head of every layer, and it flips sign discontinuously by 2 rho = "
        f"{2 * DEFAULT_RHO} when its one logit crosses zero.")


@pytest.mark.parametrize("device", DEVICES)
def test_claim_the_normalizer_makes_the_signed_arm_worse_conditioned_at_init(device):
    """THE MEASUREMENT ASKED FOR. Identical model, identical seed, identical
    batch of real TinyStories bytes, one forward and one backward. If the
    1/||w||_1 blowup is a training obstruction it has to show up here."""
    import pathlib
    path = pathlib.Path(__file__).resolve().parents[2] / CORPUS
    if not path.exists():
        pytest.skip(f"{path} missing")
    corpus = lm.ByteCorpus(path.read_text(encoding="utf-8"))
    g = torch.Generator().manual_seed(SEED)
    x, y = corpus.batch("train", 8, lm.SEQ, g, device)

    stats = {}
    for kind in ("softmax", "signed"):
        m = lm.TinyLM(kind, seed=0).to(device)
        F.cross_entropy(m(x).reshape(-1, lm.VOCAB), y.reshape(-1)).backward()
        flat = torch.cat([p.grad.flatten() for p in m.parameters()])
        per = torch.stack([p.grad.norm() for p in m.parameters()])
        stats[kind] = (float(flat.norm()), float(per.median()), float(per.max()),
                       float(m.blocks[0].attn.qkv.weight.grad.norm()))
    ratio = stats["signed"][0] / stats["softmax"][0]
    print("\n  total / median-per-param / max-per-param / blk0.qkv grad norm")
    for kind, s in stats.items():
        print("    %-8s %.4e  %.3e  %.3e  %.3e" % (kind, *s))
    print("    signed / softmax total grad norm ratio: %.4f" % ratio)
    assert ratio > 2.0 or ratio < 0.5, (
        f"total gradient norm ratio is {ratio:.4f} -- the two arms are matched to "
        f"within {abs(1 - ratio) * 100:.1f}% at identical initialization on identical "
        f"data. The 1/||w||_1 rank-one-correction blowup is NOT an obstruction at "
        f"initialization. The obstruction is the degree-zero homogeneity, which is "
        f"an exact rank deficiency and not a conditioning number.")


# ==========================================================================
# WHAT IS ACTUALLY TRUE -- GREEN
# ==========================================================================

@pytest.mark.parametrize("device", DEVICES)
def test_the_row_l1_mass_is_pinned_at_rho_whatever_the_logits_do(device):
    """The positive statement behind the temperature failure: every row but the
    first spends exactly rho of absolute mass, at every logit scale."""
    q, k, _ = draw(48, 16, device)
    for t in (0.01, 1.0, 100.0):
        mass = ceq_operator(t * q, k).abs().sum(-1)[0, 0, 1:]
        assert torch.allclose(mass, torch.full_like(mass, DEFAULT_RHO), atol=1e-12)
    print("\n  row L1 mass == rho = %.2f at logit scales 0.01, 1, 100" % DEFAULT_RHO)


@pytest.mark.parametrize("device", DEVICES)
def test_the_normalizer_is_discontinuous_not_merely_non_smooth(device):
    """w / ||w||_1 has a direction-dependent limit at w = 0. Sweep row 1's single
    logit through zero and measure the jump."""
    jumps = []
    for eps in (1e-6, 1e-10, 1e-14):
        q = torch.zeros(1, 1, 2, 2, dtype=torch.float64, device=device)
        k = torch.zeros(1, 1, 2, 2, dtype=torch.float64, device=device)
        k[0, 0, 0, 0] = 1.0
        q[0, 0, 1, 0] = eps
        pos = float(ceq_operator(q, k)[0, 0, 1, 0])
        q[0, 0, 1, 0] = -eps
        neg = float(ceq_operator(q, k)[0, 0, 1, 0])
        jumps.append(pos - neg)
    print("\n  A[1,0] jump across w = 0 at eps = 1e-6, 1e-10, 1e-14 : %s"
          % ["%.6f" % j for j in jumps])
    assert all(abs(j - 2 * DEFAULT_RHO) < 1e-12 for j in jumps)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s", "--tb=short",
                                  "-p", "no:cacheprovider"]))

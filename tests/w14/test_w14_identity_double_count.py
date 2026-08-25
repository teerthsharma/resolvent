"""H7 -- the identity term is counted twice, and only in the signed arms.

THE CONFOUND. `Block.forward` is `x = x + attn(n1(x))`. The softmax arm's
`attn` returns `P v`. The path-sum arms return

    v + A v + A^2 v + ...

which already contains `v`. So the signed arms carry an identity path that the
softmax arm does not, ON TOP of the block residual that both already have. The
two arms have not been differing only in the operator; they have been differing
in the operator AND in how many times the value is added.

That is a defect in the comparison, not a feature of the module, and it has been
present in every W10 and campaign number measured so far.

THE FIX. Drop the k=0 term:

    o = A v + A^2 v + ... + A^K v = A (I - A)^-1 v

Still signed, still strictly causal, still nilpotent -- `A(I-A)^-1` is a
polynomial in `A` and `CEQ.Nilpotent.pow_card_eq_zero` applies unchanged. The
block residual then supplies the identity exactly once, for both arms.

WHAT THIS PREDICTS. If the ~11% residual cost measured at lam=0 (fully
non-negative, so signedness cannot be the cause) is the double-counted identity,
removing it closes most of that 11%. If it does not, the multi-hop sum itself is
the cost and H7 is deleted.

EVIDENCE CLASSES. Structural checks below are RUN. The parity number is RUN over
3 seeds with the spread reported.
"""
from __future__ import annotations

import pathlib
import statistics

import pytest
import torch

from ceq import lm

CORPUS = pathlib.Path(__file__).resolve().parents[2] / "data" / "tinystories_20k.txt"
SEEDS = (0, 1, 2)
PARITY_BAR = 1.05


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda unavailable")
    return torch.device(request.param)


@pytest.fixture(scope="module")
def corpus():
    if not CORPUS.exists():
        pytest.skip(f"{CORPUS} missing")
    return lm.ByteCorpus(CORPUS.read_text(encoding="utf-8")[:4_000_000])


def test_the_confound_is_real_and_measurable(device):
    """Calibrate the claim before fixing it.

    The softmax arm's attention output must NOT contain a bare copy of `v`, and
    the path-sum arm's must. If both already agreed, there is no confound and
    H7 is testing nothing.
    """
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(2, 32, lm.D_MODEL, generator=g).to(device)
    attn = lm.TinyLM(kind="sgate", seed=0).to(device).blocks[0].attn
    qh, kh, vh = attn.qkv_heads(x)
    op = attn.operator(qh, kh)
    with_id = lm.path_sum_terms(op, vh, lm.HOPS, include_identity=True)
    without = lm.path_sum_terms(op, vh, lm.HOPS, include_identity=False)
    assert torch.allclose(with_id - without, vh, atol=1e-6), "the k=0 term is not v"
    assert not torch.allclose(with_id, without), "dropping the identity changed nothing"


def test_dropping_the_identity_keeps_signedness_and_nilpotency(device):
    """A(I-A)^-1 is a polynomial in A, so nothing structural moves."""
    m = lm.TinyLM(kind="sgate_nores").to(device)
    g = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(2, m.n_heads, 48, m.d_head, generator=g).to(device)
    a = m.blocks[0].attn.operator(q, q)
    assert float(a.min()) < 0.0, "operator is not signed"
    assert float(a.triu(0).abs().max()) == 0.0, "not strictly lower triangular"
    assert float(a.abs().sum(-1).max()) <= lm.RHO + 1e-5


def test_no_parameters_are_added(device):
    """Identical, not similar."""
    assert lm.TinyLM(kind="softmax").n_params() == lm.TinyLM(kind="sgate_nores").n_params()


@pytest.mark.slow
def test_h7_reaches_parity_over_three_seeds(device, corpus):
    """RED first. A single-seed result is not a result."""
    sm = [lm.train_one("softmax", corpus, steps=600, device=device, seed=s)["val_loss"]
          for s in SEEDS]
    ours = [lm.train_one("sgate_nores", corpus, steps=600, device=device, seed=s)["val_loss"]
            for s in SEEDS]
    rs = [a / b for a, b in zip(ours, sm)]
    med = statistics.median(rs)
    assert med <= PARITY_BAR, (
        f"median {med:.4f} over {len(SEEDS)} seeds, spread {min(rs):.4f}-{max(rs):.4f}, "
        f"bar {PARITY_BAR}. softmax={sm} ours={ours}")

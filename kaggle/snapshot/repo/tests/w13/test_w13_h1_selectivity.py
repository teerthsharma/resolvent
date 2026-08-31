"""H1 -- restore selectivity without losing the sign.

THE DIAGNOSIS. Softmax concentrates because `exp` is a sharpening nonlinearity.
`A = rho * w / sum|w|` is LINEAR in the logits and therefore flat: it cannot
focus. `hops=3` then compounds that flatness, so the multi-hop structure has been
amplifying the operator's worst property.

Measured consequence: the pure signed operator loses to softmax 1.5780 -> 2.1103
(1.337x) at 600 steps, and 1.8838 -> 2.2965 (1.2191x) at 250. The gap WIDENS with
budget, which is the signature of an expressiveness problem rather than an
optimization one.

THE CLAIM UNDER TEST. Signedness and selectivity were conflated. They are
separable:

    A = (rho/2) * (softmax(w) - softmax(-w))        w strictly causal

Both halves are softmaxes of the SAME logits, so `exp` sharpening is retained on
each, and NO new parameters are introduced -- the arms stay bit-for-bit identical
in parameter count, which is what makes the comparison a statement about the
operator.

Semantics: positive weight concentrates on the most similar keys, negative weight
on the least. Row L1 is bounded by rho/2 * (1 + 1) = rho, so the operator remains
strictly causal with bounded rows and `CEQ.Nilpotent.pow_card_eq_zero` applies
unchanged -- nilpotency is sign-blind and the bound survives by construction.

EVIDENCE CLASSES. Every number below is RUN unless labelled otherwise.
"""
from __future__ import annotations

import pathlib

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


# --------------------------------------------------------- structural checks

def test_sgate_adds_no_parameters(device):
    """Identical, not similar. A win bought with capacity is not a win."""
    a = lm.TinyLM(kind="softmax").n_params()
    b = lm.TinyLM(kind="sgate").n_params()
    assert a == b, (a, b)


def test_sgate_operator_is_signed_and_bounded(device):
    """Tier 3 kept, and the row-L1 bound the Lean depends on kept with it."""
    m = lm.TinyLM(kind="sgate").to(device)
    g = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(2, m.n_heads, 48, m.d_head, generator=g).to(device)
    a = m.blocks[0].attn.operator(q, q)
    assert float(a.min()) < 0.0, "operator is not signed; tier 3 lost"
    assert float(a.abs().sum(-1).max()) <= lm.RHO + 1e-5, float(a.abs().sum(-1).max())
    assert float(a.triu(0).abs().max()) == 0.0, "not strictly lower triangular"
    assert torch.isfinite(a).all()


def test_sgate_is_actually_more_selective_than_the_l1_operator(device):
    """The diagnosis, measured directly rather than assumed.

    Participation ratio of each row: (sum|a|)^2 / sum(a^2). Low means
    concentrated on few keys, high means spread. If the sgate operator is not
    measurably more concentrated than the L1 one, the diagnosis is wrong and H1
    is testing nothing.
    """
    g = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(2, 4, 64, 32, generator=g).to(device)
    pr = {}
    for kind in ("signed", "sgate"):
        a = lm.TinyLM(kind=kind).to(device).blocks[0].attn.operator(q, q)[..., 2:, :]
        s1, s2 = a.abs().sum(-1), (a ** 2).sum(-1)
        pr[kind] = float((s1 ** 2 / s2.clamp_min(1e-30)).median())
    assert pr["sgate"] < pr["signed"], pr


# ------------------------------------------------------------ H1's falsifier

@pytest.mark.slow
def test_h1_reaches_parity_with_softmax_over_three_seeds(device, corpus):
    """RED first. A single-seed result is not a result.

    A tau sweep on seed 0 already produced a false optimum here once -- nash at
    tau=4.0 looked like a win at 2.3444 and evaporated to 1 seed in 5. Three
    seeds minimum, spread reported.
    """
    import statistics
    ratios, rows = [], []
    for s in SEEDS:
        r = lm.compare_kinds(corpus, ("softmax", "sgate"), steps=600, device=device, seed=s)
        ratio = r["sgate"]["val_loss"] / r["softmax"]["val_loss"]
        ratios.append(ratio)
        rows.append((s, r["softmax"]["val_loss"], r["sgate"]["val_loss"], ratio))
    med = statistics.median(ratios)
    assert med <= PARITY_BAR, (
        f"median ratio {med:.4f} over {len(SEEDS)} seeds, spread "
        f"{min(ratios):.4f}-{max(ratios):.4f}, bar {PARITY_BAR}. rows={rows}")

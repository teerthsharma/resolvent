"""W10 -- the signed operator AS the attention, trained from scratch.

WHY THIS IS THE REAL TEST. Everything before it measured a CORRECTION on top of
softmax. The alpha gate is bitwise stock attention at alpha = 0 by construction,
so it can never be "superior to self-attention" -- it can only be self-attention
plus a term. That was the right shape for getting an honest cost curve on a
pretrained checkpoint, and it caught two bugs nothing else could see, but it is
scaffolding.

Here there is no softmax underneath. The attention IS

    out = v + A v + A^2 v + ... + A^K v      A strictly causal, SIGNED

and the baseline is an identical model with `F.scaled_dot_product_attention` in
the same slot. Same parameters, same initialization seed, same data, same steps,
same optimizer. The only free variable is the operator.

THE KILL CONDITION. If the pure operator cannot reach parity with softmax at
matched budget, then the module is a correction term and not a replacement, and
every claim about it has to be restated in those terms. That verdict is cheap to
get at 3M parameters and expensive to get at 500M, which is the entire reason
this runs before any scale-up.

OPEN WEIGHTS AND OPEN DATA. `roneneldan/TinyStories`, 20,000 stories, 17.9M
characters, downloaded here. Byte-level vocabulary of 256, so there is no
tokenizer dependency and nothing about the comparison hides in a merge table.
Both arms are initialized from scratch -- no pretrained weights anywhere, which
is what makes this a statement about the operator rather than about a checkpoint.
"""
from __future__ import annotations

import pathlib

import pytest
import torch

from ceq import lm

CORPUS = pathlib.Path(__file__).resolve().parents[2] / "data" / "tinystories_20k.txt"
STEPS = 600
SEED = 0


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda unavailable")
    return torch.device(request.param)


@pytest.fixture(scope="module")
def corpus():
    if not CORPUS.exists():
        pytest.skip(f"{CORPUS} missing; run the W10 data step")
    return lm.ByteCorpus(CORPUS.read_text(encoding="utf-8"))


# ------------------------------------------------------- the arms are matched

def test_both_arms_have_identical_parameter_counts(device, corpus):
    """Not 'within 10%'. IDENTICAL. The attention operator carries no parameters
    of its own -- q, k, v, o projections are shared by both arms -- so any
    difference would mean the models are not the same model."""
    a = lm.TinyLM(kind="softmax").n_params()
    b = lm.TinyLM(kind="signed").n_params()
    assert a == b, (a, b)


def test_the_signed_arm_contains_no_softmax_in_its_attention(device):
    """The whole point. If a softmax survives anywhere in the operator, this is
    a gate again and the comparison is against itself."""
    m = lm.TinyLM(kind="signed")
    a = m.blocks[0].attn.operator(torch.randn(1, m.n_heads, 32, m.d_head))
    assert float(a.min()) < 0.0, "operator is not signed"
    assert float(a.triu(0).abs().max()) == 0.0, "operator is not strictly causal"


def test_both_arms_start_from_the_same_initialization(device, corpus):
    """Same seed, same shapes, so the only thing that can differ downstream is
    the operator."""
    a = lm.TinyLM(kind="softmax", seed=SEED)
    b = lm.TinyLM(kind="signed", seed=SEED)
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        assert torch.equal(pa, pb), na


# ---------------------------------------------------------- W10's kill condition

@pytest.mark.slow
def test_pure_signed_operator_trains_to_parity_with_softmax(device, corpus):
    """RED first. THE KILL CONDITION for the module as a replacement.

    Matched parameters, matched initialization, matched data, matched steps,
    matched optimizer. If the signed operator cannot come within 5% of softmax's
    validation loss, it is a correction term and not a replacement.
    """
    res = lm.compare(corpus, steps=STEPS, device=device, seed=SEED)
    s, x = res["signed"]["val_loss"], res["softmax"]["val_loss"]
    assert s <= 1.05 * x, (
        f"signed val loss {s:.4f} vs softmax {x:.4f} = {s / x:.3f}x. The pure "
        f"operator does not reach parity, so it is a correction term and not a "
        f"replacement. {res}")


@pytest.mark.slow
def test_both_arms_beat_a_uniform_byte_baseline(device, corpus):
    """Deadness guard. `ln(256) = 5.545` is uniform over the byte vocabulary. A
    comparison between two models that both failed to learn is not a result."""
    res = lm.compare(corpus, steps=STEPS, device=device, seed=SEED)
    for name, r in res.items():
        assert r["val_loss"] < 4.0, (name, r, "barely below uniform ln(256)=5.545")

"""W8 -- the surviving parts, against a real checkpoint.

WHY A GATE AND NOT A SWAP. `ceq_attention` replaces the softmax with a signed
path sum. Dropping that into a checkpoint whose weights were fit for softmax
semantics destroys the model, and reporting the resulting perplexity as a
"result" would be measuring the swap, not the operator.

The adoptable form is a residual gate:

    out = stock_attention(q,k,v) + sum_{k=1..K} (alpha A)^k v

At `alpha = 0` this is BITWISE stock attention, so a pretrained checkpoint keeps
its exact behaviour and training can start from parity. `alpha > 0` then buys the
signed multi-hop term, and the cost of that term on untuned weights is measurable
rather than conflated with the cost of destroying the model. Every serious
attention replacement has to clear this bar before a perplexity number about it
means anything.

WHAT IS BEING CARRIED IN. Five falsifier rounds left these, each with a passing
test behind it:

  * signed causal path sum -- tier 3, min influence Jacobian -9.000e-01 against a
    non-negative control stuck at exactly 0.000e+00
  * strict causality -- rho(A) = 0 structurally, so `CEQ.Nilpotent.pow_card_eq_zero`
    gives an exact terminating resolvent with no certificate and no Perron vector
  * eviction -- bitwise forgetting, 0.000e+00 over 24/24 draws

WHAT IS NOT CARRIED IN, and why the numbers below are CPU-reference numbers:
the multi-zoom Triton kernel is FORWARD-ONLY. The CPU reference path is
differentiable and gradcheck-clean; the kernel is not. Memory is also measured at
1.31x-1.94x SDPA at every length. Both ride along with every speed claim.

PERPLEXITY IS A SANITY CHECK HERE, NOT A WIN CONDITION. C3 stands: intercept
wins invert at scale, and the win conditions are the capability tests. A
perplexity number on untuned weights says whether the gate is wired correctly,
nothing more.
"""
from __future__ import annotations

import math

import pytest
import torch

from ceq import hybrid

MODEL_ID = "Qwen/Qwen2.5-0.5B"
MAX_LEN = 256


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda unavailable")
    return torch.device(request.param)


@pytest.fixture(scope="module")
def lm():
    tf = pytest.importorskip("transformers")
    try:
        tok = tf.AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
        model = tf.AutoModelForCausalLM.from_pretrained(
            MODEL_ID, local_files_only=True, dtype=torch.float32)
    except Exception as e:                                    # noqa: BLE001
        pytest.skip(f"{MODEL_ID} not cached: {e}")
    model.eval()
    return tok, model


@pytest.fixture(scope="module")
def batch(lm):
    tok, _ = lm
    text = (
        "The capital of France is Paris. The capital of Japan is Tokyo. "
        "Water boils at one hundred degrees Celsius at sea level. "
        "A prime number has exactly two distinct positive divisors. "
        "The derivative of x squared with respect to x is two x. "
    ) * 6
    return tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)


# ------------------------------------------------- the adoptability bar

def test_alpha_zero_is_bitwise_stock_attention(device):
    """THE BAR. At alpha = 0 the gate must be the stock operator, bitwise.

    Not 'close'. A pretrained checkpoint has to keep its exact behaviour, or the
    perplexity numbers below measure the swap rather than the operator.
    """
    g = torch.Generator(device="cpu").manual_seed(0)
    q, k, v = (torch.randn(2, 4, 32, 16, generator=g, dtype=torch.float32).to(device)
               for _ in range(3))
    gated, _ = hybrid.ceq_hybrid_attention(None, q, k, v, None, alpha=0.0)
    stock = hybrid.stock_attention(q, k, v, None).transpose(1, 2).contiguous()
    assert torch.equal(gated, stock), float((gated - stock).abs().max())


def test_alpha_above_zero_actually_changes_the_output(device):
    """Deadness guard on the gate itself. If alpha does nothing, the parity
    above is trivially true and proves nothing."""
    g = torch.Generator(device="cpu").manual_seed(0)
    q, k, v = (torch.randn(2, 4, 32, 16, generator=g, dtype=torch.float32).to(device)
               for _ in range(3))
    a0, _ = hybrid.ceq_hybrid_attention(None, q, k, v, None, alpha=0.0)
    a1, _ = hybrid.ceq_hybrid_attention(None, q, k, v, None, alpha=0.25)
    assert float((a1 - a0).abs().max()) > 1e-4


def test_the_correction_term_is_signed(device):
    """Tier 3 has to survive the gating, or the whole point is lost."""
    g = torch.Generator(device="cpu").manual_seed(0)
    q, k, v = (torch.randn(2, 4, 32, 16, generator=g, dtype=torch.float32).to(device)
               for _ in range(3))
    a0, _ = hybrid.ceq_hybrid_attention(None, q, k, v, None, alpha=0.0)
    a1, _ = hybrid.ceq_hybrid_attention(None, q, k, v, None, alpha=0.25)
    d = a1 - a0
    assert float(d.min()) < 0.0 and float(d.max()) > 0.0, (float(d.min()), float(d.max()))


def test_grouped_query_shapes_are_handled(device):
    """Qwen2.5 uses GQA: fewer kv heads than q heads. An attention function that
    assumes they match silently broadcasts or crashes."""
    g = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(1, 8, 24, 16, generator=g, dtype=torch.float32).to(device)
    k = torch.randn(1, 2, 24, 16, generator=g, dtype=torch.float32).to(device)
    v = torch.randn(1, 2, 24, 16, generator=g, dtype=torch.float32).to(device)
    out, _ = hybrid.ceq_hybrid_attention(None, q, k, v, None, alpha=0.25)
    assert out.shape == (1, 24, 8, 16), out.shape      # [B, S, H, D]
    assert torch.isfinite(out).all()


def test_output_layout_is_batch_seq_heads_dim(device):
    """Pins the AttentionInterface contract explicitly.

    The function must return `[B, S, H, D]`, not `[B, H, S, D]`. Getting this
    wrong raises nothing, produces finite output, and passes every internal
    self-consistency check -- measured perplexity 89400.180 against 1.667 for the
    untouched checkpoint. Only parity against a real model catches it, so the
    contract is pinned here rather than left implicit.
    """
    g = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(2, 5, 7, 16, generator=g, dtype=torch.float32).to(device)
    k = torch.randn(2, 5, 7, 16, generator=g, dtype=torch.float32).to(device)
    v = torch.randn(2, 5, 7, 16, generator=g, dtype=torch.float32).to(device)
    out, _ = hybrid.ceq_hybrid_attention(None, q, k, v, None, alpha=0.0)
    assert out.shape == (2, 7, 5, 16), out.shape


# ------------------------------------------------- against the real checkpoint

def test_registers_and_runs_a_forward_pass_on_a_real_checkpoint(lm, device):
    """LOOP.md completion condition 1, on a real model rather than a fixture."""
    tok, model = lm
    hybrid.register()
    m = model.to(device)
    ids = tok("The capital of France is", return_tensors="pt").to(device)
    with torch.no_grad():
        out = hybrid.run_with(m, ids, alpha=0.0)
    assert out.logits.shape[0] == 1
    assert torch.isfinite(out.logits).all()


def test_perplexity_at_alpha_zero_equals_stock(lm, batch, device):
    """The gate must not perturb a pretrained model at all when it is closed.
    Measured on real weights and real text, not on random tensors."""
    _, model = lm
    m = model.to(device)
    b = {k: v.to(device) for k, v in batch.items()}
    stock = hybrid.perplexity(m, b, alpha=None)
    gated = hybrid.perplexity(m, b, alpha=0.0)
    assert abs(stock - gated) < 1e-3 * stock, (stock, gated)


def test_the_cost_of_the_operator_on_untuned_weights_is_measured(lm, batch, device):
    """Honest cost, C6. The curve is recorded; monotonicity is NOT asserted.

    An earlier version of this test asserted that perplexity rises monotonically
    with alpha. It passed on one text and is false in general: on a 99-token
    technical passage the curve was 9.2149 / 9.1779 / 9.0781 / 9.1789 / 10.7969
    for alpha 0.00 / 0.01 / 0.05 / 0.15 / 0.30, i.e. a small alpha slightly
    IMPROVES perplexity before large alpha degrades it. Asserting a property that
    happens to hold on the fixture is how a fragile test survives to mislead.

    What is asserted is what is actually true: the curve is finite everywhere,
    the closed gate is free, and a large alpha degrades a checkpoint that was
    never trained for the term.
    """
    _, model = lm
    m = model.to(device)
    b = {k: v.to(device) for k, v in batch.items()}
    stock = hybrid.perplexity(m, b, alpha=None)
    curve = {a: hybrid.perplexity(m, b, alpha=a) for a in (0.0, 0.05, 0.30)}
    assert all(math.isfinite(p) for p in curve.values()), curve
    assert abs(curve[0.0] - stock) < 1e-3 * stock, (stock, curve)
    assert curve[0.30] > curve[0.0], curve


# --------------------------------------------------- decode-time correctness

def test_gate_closed_generates_identically_to_stock(lm, device):
    """The bug perplexity cannot see.

    Teacher-forced perplexity never decodes, so it matched stock to 0.0000% at
    alpha = 0 while greedy generation produced ' The following the 1: 1: 1: 1:'
    against stock's ' The capital of France is Paris.'. The cause was
    `is_causal=True` with q_len=1 against a cached kv_len=N.
    """
    tok, model = lm
    m = model.to(device)
    ids = tok("Question: What is the capital of France?" + chr(10) + "Answer:",
              return_tensors="pt").to(device)
    outs = []
    for a in (None, 0.0):
        hybrid._install(m, a, hybrid.DEFAULT_RHO, hybrid.DEFAULT_HOPS)
        with torch.no_grad():
            o = m.generate(**ids, max_new_tokens=16, do_sample=False,
                           pad_token_id=tok.eos_token_id)
        outs.append(tok.decode(o[0], skip_special_tokens=True))
    assert outs[0] == outs[1], outs


def test_multihop_decode_is_refused_not_silently_downgraded(device):
    """A path sum needs a square operator. During decode the row is [1, N] and
    A^2 does not exist.

    Falling back to hops=1 would make the module behave differently in prefill
    and decode without saying so. It is refused instead, and the refusal names
    the fix.
    """
    g = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(1, 4, 1, 16, generator=g, dtype=torch.float32).to(device)
    k = torch.randn(1, 4, 12, 16, generator=g, dtype=torch.float32).to(device)
    v = torch.randn(1, 4, 12, 16, generator=g, dtype=torch.float32).to(device)
    with pytest.raises(NotImplementedError, match="hop cache"):
        hybrid.ceq_hybrid_attention(None, q, k, v, None, alpha=0.1)

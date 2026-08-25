"""What breaks in TRAINING that did not break in inference.

Everything measured on this operator so far ran under `torch.no_grad()` or in a
forward-only kernel. The operator is

    A = rho * w / sum_j |w_ij|            w strictly lower triangular

and the denominator is a sum of absolute values. Three separate hazards live in
that one line, none of which a forward pass can see:

1. `sum|w|` is **exactly zero on row 0** by construction -- row 0 has no
   predecessors. `ceq/attention.py` and `ceq/lm.py` both handle it forward with
   `.clamp_min(torch.finfo(dtype).tiny)`, which gives A[0] = 0 and is correct.
   Backward is a different function. `x/c` has derivative `1/c`, and `1/tiny` is
   8.5e37 in float32; the quotient rule's other term carries `tiny**2` in the
   denominator, which UNDERFLOWS TO EXACTLY ZERO in float32. 0/0 is NaN.

2. `|w|` is not differentiable at w = 0, and a row whose entries are all near
   zero has a Jacobian that scales as 1/l1 without bound.

3. hops = 3 chains three matmuls of a matrix that itself depends on q and k. The
   backward graph is deeper than the forward reads, and every intermediate
   [S, S] operator has to be retained.

Every test here parametrizes over cpu and cuda via `tests/chase/conftest.py`.
"""

import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ceq import lm
from ceq.attention import ceq_operator, path_sum


# ------------------------------------------------------------------ helpers

def _lm_operator(q, k, rho=lm.RHO):
    """The operator EXACTLY as `ceq/lm.py` computes it today, inlined so the
    test measures the shipped arithmetic and not a paraphrase of it."""
    s = q.shape[-2]
    w = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    m = torch.ones(s, s, dtype=torch.bool, device=w.device).tril(-1)
    w = w.masked_fill(~m, 0.0)
    l1 = w.abs().sum(-1, keepdim=True)
    return rho * w / l1.clamp_min(torch.finfo(w.dtype).tiny)


# ------------------------------------------- 1. the zero-denominator row

def test_operator_backward_is_finite_on_the_empty_first_row(device):
    """RED first. Row 0 has no predecessors so its L1 norm is exactly 0.

    Forward is fine -- the clamp gives A[0] = 0. Backward divides by the clamp.
    A single NaN here poisons every parameter in the model on the first step,
    because AdamW propagates NaN into the exponential moving averages and no
    subsequent step can recover.
    """
    torch.manual_seed(0)
    q = torch.randn(1, 1, 8, 16, device=device, requires_grad=True)
    k = torch.randn(1, 1, 8, 16, device=device, requires_grad=True)
    a = _lm_operator(q, k)
    a.sum().backward()
    assert torch.isfinite(q.grad).all(), (
        "q.grad has {} non-finite entries. The empty first row divides by "
        "finfo.tiny in the backward pass; tiny**2 underflows to 0.0 in float32 "
        "so the quotient-rule term is 0/0.".format(
            int((~torch.isfinite(q.grad)).sum())))
    assert torch.isfinite(k.grad).all(), "k.grad non-finite"


def test_the_forward_pass_alone_never_shows_this(device):
    """Calibration for the test above. If the forward were also broken, the
    finding would be 'the operator is broken', not 'training is'."""
    torch.manual_seed(0)
    q = torch.randn(1, 1, 8, 16, device=device)
    k = torch.randn(1, 1, 8, 16, device=device)
    a = _lm_operator(q, k)
    assert torch.isfinite(a).all()
    assert float(a[..., 0, :].abs().max()) == 0.0, "row 0 is not exactly zero"


# ------------------------------------------- 2. a collapsing row norm

def _grad_at_scale(device, scale):
    torch.manual_seed(0)
    q = (torch.randn(1, 1, 8, 16, device=device) * scale).requires_grad_(True)
    k = torch.randn(1, 1, 8, 16, device=device, requires_grad=True)
    a = _lm_operator(q, k)
    row_l1 = a.abs().sum(-1)[..., 1:]        # row 0 is empty by construction
    a.sum().backward()
    return float(q.grad.abs().max()), float((row_l1 - lm.RHO).abs().max())


def test_the_forward_is_scale_invariant_and_the_backward_is_one_over_the_row_norm(device):
    """The finding, asserted POSITIVELY rather than as a red test.

    Every row is renormalized to L1 = rho, so scaling the logits by 10^-k leaves
    the FORWARD output unchanged. The BACKWARD is exactly 10^k larger, decade for
    decade, over six decades on both devices -- `d(w/|w|)/dw` carries a factor
    1/l1 and nothing bounds it.

    Written first as `assert grad < 1e6`, which failed at 1e-6 on cpu (1.337e+07)
    and PASSED at 1e-6 on cuda (5.448e+05) from the same seed. A threshold that
    lands on opposite sides of the same claim on two devices is a fragile test.
    The 1/l1 LAW is device-independent and is the real statement.
    """
    base, err0 = _grad_at_scale(device, 1.0)
    assert err0 < 1e-5, "row L1 is not pinned to rho at scale 1: err {:.3e}".format(err0)
    for k in (2, 4, 6, 8, 12):
        g, err = _grad_at_scale(device, 10.0 ** -k)
        assert err < 1e-5, (
            "the forward stopped being scale-invariant at 1e-{}: row L1 error "
            "{:.3e}".format(k, err))
        ratio = g / base / 10.0 ** k
        assert 0.99 < ratio < 1.01, (
            "at input scale 1e-{}: max |dA/dq| = {:.4e} against {:.4e} at scale 1, "
            "which is {:.4f} of the predicted 10^{} -- the 1/l1 law does not "
            "hold and the conditioning claim needs restating.".format(
                k, g, base, ratio, k))


def test_a_single_entry_row_has_the_analytic_gradient_it_should(device):
    """Row 1 has exactly one predecessor, so A[1,0] = rho*w/|w| = rho*sign(w), a
    CONSTANT in |w|. The analytic derivative is exactly 0. What the code computes
    is rho*(1/|w| - w*sign(w)/w**2): two large numbers that must cancel."""
    torch.manual_seed(0)
    for mag in (1.0, 1e-3, 1e-6):
        w = torch.tensor([[0.0, 0.0], [mag, 0.0]], device=device, requires_grad=True)
        l1 = w.abs().sum(-1, keepdim=True)
        a = lm.RHO * w / l1.clamp_min(torch.finfo(w.dtype).tiny)
        a[1, 0].backward()
        g = float(w.grad[1, 0])
        assert abs(g) < 1e-3, (
            "d(A[1,0])/d(w[1,0]) = {:.6e} at |w| = {:g}; analytically it is "
            "exactly 0 because a one-entry row normalizes to rho*sign(w).".format(
                g, mag))


# ------------------------------------------- 2b. the rollback, measured

def test_the_eps_rollback_bounds_the_jacobian_it_is_supposed_to_bound(device):
    """RED first. The stated rollback for the 1/l1 conditioning is

        A = rho * w / (sum|w| + eps)

    which bounds |dA/dw| by rho/eps. Stated is not measured, so this measures it
    at the input scale where the unguarded operator reaches 1.3e+13.
    """
    from ceq.attention import ceq_operator
    torch.manual_seed(0)
    eps = 1e-3
    q = (torch.randn(1, 1, 8, 16, device=device) * 1e-12).requires_grad_(True)
    k = torch.randn(1, 1, 8, 16, device=device, requires_grad=True)
    a = ceq_operator(q, k, eps=eps)
    a.sum().backward()
    g = float(q.grad.abs().max())
    assert g < 10.0 * lm.RHO / eps, (
        "max |dA/dq| = {:.4e} with eps={:g}; the bound rho/eps is {:.4e}".format(
            g, eps, lm.RHO / eps))


def test_the_eps_rollback_keeps_the_operator_signed_and_strictly_causal(device):
    """The property the rollback exists to preserve. A rollback that fixes the
    gradient by making the operator non-negative would drop the module to tier 2,
    where the influence Jacobian is provably non-negative and `not` is
    unreachable -- which is the one thing this operator is for."""
    from ceq.attention import ceq_operator
    torch.manual_seed(0)
    q = torch.randn(1, 2, 32, 16, device=device)
    a = ceq_operator(q, q, eps=1e-3)
    assert float(a.min()) < 0.0, "the rollback made the operator non-negative"
    assert float(a.triu(0).abs().max()) == 0.0, "the rollback broke strict causality"


def test_eps_zero_is_bitwise_the_operator_that_shipped(device):
    """The rollback is OFF by default and must cost exactly nothing when off.
    Bitwise, so that turning it on is a decision somebody makes rather than a
    default that silently moved."""
    from ceq.attention import ceq_operator
    torch.manual_seed(0)
    q = torch.randn(2, 4, 24, 16, device=device)
    k = torch.randn(2, 4, 24, 16, device=device)
    assert torch.equal(ceq_operator(q, k), ceq_operator(q, k, eps=0.0))


def test_the_eps_rollback_barely_moves_the_forward_on_healthy_rows(device):
    """The price, measured on the right quantity.

    eps shrinks a row's L1 from rho to rho*l1/(l1+eps), so the cost is largest
    exactly where l1 is smallest -- which is the point of the guard, not a defect
    in it. The first version of this test asserted on `max|A_eps - A|` relative to
    `max|A|`, which is a statistic OF THE WORST ROW: it read 1.306e-02 on cpu and
    8.553e-03 on cuda from the same seed, landing on opposite sides of a 1e-2
    threshold. Measuring the guard by how hard it grips the row it exists to grip
    is the wrong instrument.

    The price on healthy rows is the deviation of the MEDIAN row L1 from rho.
    """
    from ceq.attention import ceq_operator, DEFAULT_RHO
    torch.manual_seed(0)
    q = torch.randn(1, 2, 64, 16, device=device)
    k = torch.randn(1, 2, 64, 16, device=device)
    a = ceq_operator(q, k, eps=1e-3)
    l1 = a.abs().sum(-1)[..., 1:]          # row 0 is empty by construction
    med = abs(float(l1.median()) / DEFAULT_RHO - 1.0)
    worst = abs(float(l1.min()) / DEFAULT_RHO - 1.0)
    assert med < 1e-3, (
        "eps=1e-3 moved the MEDIAN row L1 to {:.6f} against rho={}, a relative "
        "{:.3e}".format(float(l1.median()), DEFAULT_RHO, med))
    assert worst < 0.05, (
        "eps=1e-3 moved the WORST row L1 to {:.6f}, a relative {:.3e}; the guard "
        "is gripping harder than intended".format(float(l1.min()), worst))


# ------------------------------------------- 3. gradcheck, the CPU oracle

def test_operator_passes_gradcheck_in_float64(device):
    """`STATE.md` records the CPU reference as gradcheck-clean. That claim is
    about `ceq/attention.py`; this pins it on the device under test."""
    torch.manual_seed(0)
    q = torch.randn(1, 1, 6, 8, dtype=torch.float64, device=device, requires_grad=True)
    k = torch.randn(1, 1, 6, 8, dtype=torch.float64, device=device, requires_grad=True)
    v = torch.randn(1, 1, 6, 8, dtype=torch.float64, device=device, requires_grad=True)

    def f(a, b, c):
        return path_sum(ceq_operator(a, b), c, lm.HOPS)

    assert torch.autograd.gradcheck(f, (q, k, v), eps=1e-6, atol=1e-6)


# ------------------------------------------- 4. a real training loop

def _grad_norm_trace(kind, device, steps, seq=64, bs=8, d=64, layers=2, heads=4,
                     lr=3e-4):
    """Run a real loop and return the per-step global grad norm BEFORE clipping.

    Clipping is applied after recording, because `clip_grad_norm_` hides the very
    quantity being measured and reports nothing when the norm is NaN.
    """
    torch.manual_seed(0)
    m = lm.TinyLM(kind, d=d, n_layers=layers, n_heads=heads, seq=seq, seed=0).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(1)
    norms = []
    for _ in range(steps):
        x = torch.randint(0, lm.VOCAB, (bs, seq), generator=gen).to(device)
        y = torch.randint(0, lm.VOCAB, (bs, seq), generator=gen).to(device)
        loss = F.cross_entropy(m(x).reshape(-1, lm.VOCAB), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        n = torch.norm(torch.stack([p.grad.detach().norm() for p in m.parameters()
                                    if p.grad is not None]))
        norms.append(float(n))
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    return norms, m


def test_signed_arm_gradient_norms_stay_finite_over_a_real_training_loop(device):
    """RED first. Not reasoning about the normalization -- running it.

    Random targets on purpose: the question is whether the OPERATOR survives
    backprop, and a learnable corpus would let the model reduce the loss and mask
    a gradient pathology behind a falling curve. That exact masking is on record
    in this project -- a DEQ lost its fixed point at step 47 and the loss kept
    falling for 922 further steps.
    """
    norms, m = _grad_norm_trace("signed", device, steps=60)
    bad = [(i, n) for i, n in enumerate(norms) if not math.isfinite(n)]
    assert not bad, (
        "{}/{} steps had a non-finite gradient norm; first at step {} with {}. "
        "Trace head: {}".format(len(bad), len(norms), bad[0][0], bad[0][1],
                                ["{:.3e}".format(n) for n in norms[:8]]))
    assert all(torch.isfinite(p).all() for p in m.parameters()), (
        "parameters are non-finite after the loop; AdamW propagated a NaN into "
        "the moment estimates and no later step can recover")


def test_softmax_arm_is_the_control_and_survives_the_same_loop(device):
    """The control. If both arms blow up, the finding is the harness and not the
    operator."""
    norms, m = _grad_norm_trace("softmax", device, steps=60)
    assert all(math.isfinite(n) for n in norms), norms[:8]
    assert all(torch.isfinite(p).all() for p in m.parameters())


def test_signed_gradient_norm_is_not_orders_of_magnitude_above_softmax(device):
    """Finite is not the same as trainable. A gradient norm far above the control
    means every step is a clipped unit vector and the loss carries no scale."""
    s, _ = _grad_norm_trace("signed", device, steps=30)
    x, _ = _grad_norm_trace("softmax", device, steps=30)
    fs = [n for n in s if math.isfinite(n)]
    assert fs, "no finite signed gradient norms at all"
    ratio = max(fs) / max(x)
    assert ratio < 100.0, (
        "peak signed grad norm {:.3e} vs softmax {:.3e} = {:.1f}x. Clipping at "
        "1.0 turns every signed step into a direction with no magnitude "
        "information.".format(max(fs), max(x), ratio))


# ------------------------------------------- 4b. where the bad rows actually are

def _row_l1(m, x):
    """Row L1 norms of the RAW logits w, per signed block, row 0 excluded.

    Row 0 has no predecessors so its L1 is exactly 0 by construction and is not a
    finding; every other row is.
    """
    out = []
    h = m.tok(x) + m.pos(torch.arange(x.shape[1], device=x.device))[None]
    for blk in m.blocks:
        z = blk.n1(h)
        b, s, d = z.shape
        q, k, _ = blk.attn.qkv(z).chunk(3, dim=-1)
        sh = lambda t: t.view(b, s, blk.attn.n_heads, blk.attn.d_head).transpose(1, 2)
        w = (sh(q) @ sh(k).transpose(-2, -1)) / math.sqrt(blk.attn.d_head)
        msk = torch.ones(s, s, dtype=torch.bool, device=w.device).tril(-1)
        out.append(w.masked_fill(~msk, 0.0).abs().sum(-1)[..., 1:])
        h = blk(h)
    return torch.cat([t.reshape(-1) for t in out])


def test_the_row_l1_spread_is_narrow_enough_for_a_1_over_l1_jacobian(device):
    """RED first, and it fails AT INITIALIZATION -- no training required.

    The gradient of `A = rho*w/sum|w|` has condition 1/l1, measured exactly: at
    input scale 1e-k the peak |dA/dq| is 10^k times its scale-1 value, over six
    decades. That is only a hazard if some row's l1 is small, so this asks
    whether any is.
    """
    torch.manual_seed(0)
    m = lm.TinyLM("signed", d=256, n_layers=4, n_heads=4, seq=128, seed=0).to(device)
    x = torch.randint(0, lm.VOCAB, (32, 128), device=device)
    with torch.no_grad():
        l1 = _row_l1(m, x)
    lo, med = float(l1.min()), float(l1.median())
    assert med / lo < 1e3, (
        "row L1 of w spans {:.3e} (min) to {:.3e} (median) = {:.2e}x at "
        "initialization. The Jacobian of the normalization scales as 1/l1, so "
        "the worst row's operator gradient is that factor above the typical "
        "row's, inside one global grad-norm clip.".format(lo, med, med / lo))


@pytest.mark.slow
def test_the_row_l1_spread_stays_bounded_throughout_training(device):
    """The trend, which is the part a single snapshot cannot show. Measured over
    800 steps on `data/tinystories_20k.txt`, RTX 4060 Laptop: median row L1 rose
    16.55 -> 53.07 while the minimum fell as far as 9.35e-07. The typical row and
    the worst row move in OPPOSITE directions, so the conditioning degrades as
    training proceeds rather than annealing out.

    THE ASSERTION IS A BOUND AND NOT A TREND, deliberately. `min` over ~65,000
    rows is an extreme-value statistic and its step-to-step direction is noise:
    the first version of this test asserted `last <= first` and passed on cuda
    while failing on cpu from the same seed, which is a fragile test that would
    have shipped a device-dependent finding. What the operator actually requires
    is a FLOOR under l1, at every step, so that is what is asserted.
    """
    corpus_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "tinystories_20k.txt")
    if not os.path.exists(corpus_path):
        pytest.skip("data/tinystories_20k.txt missing")
    with open(corpus_path, encoding="utf-8") as fh:
        corpus = lm.ByteCorpus(fh.read())
    torch.manual_seed(0)
    m = lm.TinyLM("signed", seed=0).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    gen = torch.Generator().manual_seed(1)
    spreads = []
    for step in range(200):
        x, y = corpus.batch("train", 32, m.seq, gen, device)
        loss = F.cross_entropy(m(x).reshape(-1, lm.VOCAB), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            with torch.no_grad():
                l1 = _row_l1(m, x)
            spreads.append((step, float(l1.min()), float(l1.median())))
    worst = max(med / lo for _, lo, med in spreads)
    assert worst < 1e3, (
        "median/min row L1 peaks at {:.2e} during training; trace (step, min, "
        "median) = {}".format(
            worst, [(s, "{:.2e}".format(lo), "{:.2f}".format(md))
                    for s, lo, md in spreads]))


# ------------------------------------------- 5. the memory of the deeper graph

def _peak_bytes(kind, device, seq, bs=4, d=256, layers=4, heads=4):
    """Peak allocated bytes for ONE forward + backward."""
    torch.manual_seed(0)
    m = lm.TinyLM(kind, d=d, n_layers=layers, n_heads=heads, seq=seq, seed=0).to(device)
    x = torch.randint(0, lm.VOCAB, (bs, seq), device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    F.cross_entropy(m(x).reshape(-1, lm.VOCAB), x.reshape(-1)).backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del m, x
    return peak


@pytest.mark.parametrize("seq", [256, 512, 1024])
def test_forward_backward_memory_is_within_2x_of_the_softmax_arm(device, seq):
    """The forward-only kernel measured 1.31x-1.94x SDPA. Training adds the
    retained [S,S] operator per layer plus every hop intermediate, and SDPA's
    fused backward retains no [S,S] at all. 2x is the forward ceiling already on
    record; this asks whether backward stays inside it."""
    if device != "cuda":
        pytest.skip("peak-allocation accounting is CUDA-only")
    sg = _peak_bytes("signed", device, seq)
    sm = _peak_bytes("softmax", device, seq)
    assert sg <= 2.0 * sm, (
        "seq={}: signed forward+backward peak {:.1f} MiB vs softmax {:.1f} MiB "
        "= {:.2f}x. RTX 4060 Laptop, sm_89, 8.0 GiB.".format(
            seq, sg / 2 ** 20, sm / 2 ** 20, sg / sm))

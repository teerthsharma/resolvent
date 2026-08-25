"""Does the 1.0334 parity ratio survive to 300M, or is it a 3.3M artifact?

The user's standard: parity at 3.3M is FALSE unless it survives at 300M. The
joint sweep in `scale/scale_sweep.py` moves `d`, `n_layers` and `n_heads`
TOGETHER at seq 128 -- (128,2,4), (256,4,4), (384,6,6), (512,8,8). That answers
"does the number move", and it cannot answer "which axis moved it", because
every axis moves at once and one of them does not move at all.

This file takes the axes that sweep freezes or confounds:

  SEQ    frozen at 128 there. A 300M run trains at 1024-2048. That is the single
         largest multiplier between here and there -- 8x to 16x -- and it is the
         one axis the joint sweep holds constant.
  DEPTH  confounded there with width. 300M is 24 layers against the 4 measured.
         `sgate` adds `v + Av + A^2 v` where row sum is fixed at
         rho(1-lam)/(1+lam) = 1.2273, so the attention branch carries a DC gain
         of 1 + 1.2273 + 1.2273^2 = 3.734 into the residual stream against
         softmax's 1.000. Whatever that does, it compounds with L and only with L.
  HEADS  confounded there with width. d_head is 32 at the first point and 64 at
         the other three, so the sweep cannot say whether an effect belongs to
         head count or to head dim.
  RHO    the parity point `rho=1.5 lam=0.10 hops=2 lr=1e-3` was tuned AT 3.3M.
         `rho` is ONE GLOBAL SCALAR shared by every row of every head of every
         layer. At 3.3M that is 4 layers x 4 heads x 128 rows = 2,048 rows per
         forward. At 300M seq 2048 it is 24 x 16 x 2048 = 786,432 rows, a 384x
         increase in what one number has to cover, while softmax's self/other
         split is free per row. If the argmin over rho MOVES between two sizes,
         the tuned constant does not transfer and the 3.3M number is a statement
         about a tuning, not about an operator.

The expensive arms are trained once by `tests/chase/axes.py` into
`tests/chase/scale_axes.jsonl`; these tests read that file. A missing file is a
FAILURE and not a skip, because a scaling verdict that quietly stops being
measured is exactly the failure mode this file exists to prevent.

Every test parametrizes over cpu and cuda. The arithmetic and the init-time
structural probes run on both; the trained sweeps read a recorded file and so
assert identically on either device.
"""

import json
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ceq import lm

import axes

#: The project's own bar, unchanged from the iteration-16 parity claim.
BAR = 1.05


@pytest.fixture
def parity_point():
    """rho=1.5 lam=0.10 hops=2. `ceq/lm.py` still defaults to rho=0.9 lam=1.0
    hops=3, which is the OLD operator, and at lam=1.0 the row-L1 lower bound
    below is trivially zero -- so a test that forgets to set this passes for the
    wrong reason."""
    old = (lm.RHO, lm.SGATE_LAM, lm.HOPS)
    lm.RHO, lm.SGATE_LAM, lm.HOPS = 1.5, 0.10, 2
    try:
        yield
    finally:
        lm.RHO, lm.SGATE_LAM, lm.HOPS = old


#: points each axis must have before it may return a verdict. A PARTIAL sweep
#: passing is the exact failure this file exists to prevent: the first point of
#: every axis is the 3.3M parity configuration, which is already known to be
#: under the bar, so one recorded point would make any axis look green.
EXPECTED = dict(seq=4, depth=4, heads=4, rho=8)


def _rows(axis):
    recs = axes.load()
    got = [r for r in recs if r["axis"] == axis]
    assert len(got) == EXPECTED[axis], (
        "axis {!r} has {} of {} points recorded in {}. Generate them with "
        "`python tests/chase/axes.py {}` -- a scaling verdict that stops being "
        "measured must fail, not skip, and a partial axis must not return a "
        "verdict.".format(axis, len(got), EXPECTED[axis], axes.RESULTS, axis))
    return sorted(got, key=lambda r: r["x"])


# ------------------------------------------------------------------ 1. SEQUENCE


def test_the_parity_ratio_does_not_degrade_with_sequence_length(device):
    """RED first. The axis the joint sweep freezes and a real run multiplies 16x.

    d=256, L=4, H=4 held fixed; seq 128 -> 1024 with `batch * seq` pinned at
    4096 so every point sees the SAME number of tokens per step and the same
    total token budget. Both arms get the identical treatment, so the ratio is
    still a statement about the operator.
    """
    rows = _rows("seq")
    worst = max(rows, key=lambda r: r["ratio"])
    assert worst["ratio"] <= BAR, (
        "parity fails on the sequence axis: " +
        " | ".join("seq={} bs={} softmax {:.4f} sgate {:.4f} ratio {:.4f}".format(
            r["x"], r["batch"], r["softmax"], r["sgate"], r["ratio"]) for r in rows))


def test_the_sequence_axis_ratio_has_no_upward_trend(device):
    """A ratio under the bar everywhere but climbing is still a scaling failure.

    Least-squares slope of ratio against log2(seq). A 300M run is 3-4 doublings
    past the last point measured here, so a positive slope extrapolates through
    the bar before the run starts.
    """
    rows = _rows("seq")
    xs = [math.log2(r["x"]) for r in rows]
    ys = [r["ratio"] for r in rows]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = (sum((a - mx) * (b - my) for a, b in zip(xs, ys))
             / sum((a - mx) ** 2 for a in xs))
    at_2048 = ys[-1] + slope * (11 - xs[-1])
    assert slope <= 0.0, (
        "ratio rises {:+.4f} per doubling of seq: {} -> extrapolated {:.4f} at "
        "seq 2048, bar {:.2f}".format(
            slope, ", ".join("{}:{:.4f}".format(r["x"], r["ratio"]) for r in rows),
            at_2048, BAR))


# --------------------------------------------------------------------- 2. DEPTH


def test_the_parity_ratio_does_not_degrade_with_depth_alone(device):
    """d=256, H=4, seq=128 fixed; L = 2, 4, 8, 16.

    300M is 24 layers. The parity point was measured at 4. The signed arm's
    attention branch has a DC gain of 3.734 per layer against softmax's 1.000
    and nothing else in the block differs, so depth is the axis on which that
    difference accumulates.
    """
    rows = _rows("depth")
    worst = max(rows, key=lambda r: r["ratio"])
    assert worst["ratio"] <= BAR, (
        "parity fails on the depth axis: " +
        " | ".join("L={} softmax {:.4f} sgate {:.4f} ratio {:.4f}".format(
            r["x"], r["softmax"], r["sgate"], r["ratio"]) for r in rows))


# ---------------------------------------------------------------------- 3. HEADS


def test_the_parity_ratio_does_not_degrade_with_head_count_alone(device):
    """d=256, L=4, seq=128 fixed; H = 2, 4, 8, 16, so d_head = 128, 64, 32, 16.

    Parameter count is IDENTICAL at every point -- the head split is a reshape,
    not a parameter -- so this axis is free of the capacity confound the joint
    sweep carries. A 300M model runs 16 heads; the parity point ran 4.
    """
    rows = _rows("heads")
    worst = max(rows, key=lambda r: r["ratio"])
    assert worst["ratio"] <= BAR, (
        "parity fails on the head-count axis: " +
        " | ".join("H={} d_head={} softmax {:.4f} sgate {:.4f} ratio {:.4f}".format(
            r["x"], 256 // r["x"], r["softmax"], r["sgate"], r["ratio"])
            for r in rows))


# ------------------------------------------------------------------------ 4. RHO


def test_the_tuned_rho_is_the_same_at_two_scales(device):
    """The decisive one. `rho` is a constant tuned at 3.3M and shipped as if it
    were a property of the operator.

    Sweep rho over the same grid at two sizes. If the argmin moves, then the
    1.0334 at 3.3M is the value of a tuning at 3.3M and a 300M run inherits a
    constant that was never fitted to it -- which is a re-tune of the challenger
    only, on a control that has no such knob, at 300M prices.
    """
    small = [r for r in _rows("rho") if r["size"] == "small"]
    large = [r for r in _rows("rho") if r["size"] == "large"]
    assert small and large, "need both sizes recorded on the rho axis"
    best_s = min(small, key=lambda r: r["sgate"])
    best_l = min(large, key=lambda r: r["sgate"])
    assert best_s["x"] == best_l["x"], (
        "argmin rho MOVES with scale: {} at {} -> {} at {}.\n  small: {}\n  large: {}"
        .format(best_s["x"], best_s["cfg"], best_l["x"], best_l["cfg"],
                ", ".join("rho={} val {:.4f} ratio {:.4f}".format(
                    r["x"], r["sgate"], r["ratio"]) for r in small),
                ", ".join("rho={} val {:.4f} ratio {:.4f}".format(
                    r["x"], r["sgate"], r["ratio"]) for r in large)))


# --------------------------------------------- 5. init-time structural invariants
#
# These need no training and no recorded file. They are the cheap half of the
# question: what does the OPERATOR do differently at 24 layers and d=1024, before
# a single gradient step, that it does not do at 4 layers and d=256.


@pytest.mark.parametrize("d,heads", [(256, 4), (1024, 16)])
def test_the_sgate_row_l1_is_bounded_by_rho_independent_of_width(
        device, parity_point, d, heads):
    """The conditioning hazard that motivated the `eps` rollback cannot arise here.

    `signed` divides by the row L1 of the raw logits, so its Jacobian carries a
    1/l1 and the measured spread was 1.10e-03 to 1.67e+01 at init. `sgate` is a
    difference of two softmaxes and divides by nothing data-dependent, so every
    row L1 lies in [rho(1-lam)/(1+lam), rho] BY CONSTRUCTION -- a two-sided bound
    that is a function of the knobs alone and not of d, H, S or the data.
    """
    lam = lm.SGATE_LAM
    lo = lm.RHO * (1 - lam) / (1 + lam) - 1e-5
    hi = lm.RHO + 1e-5
    torch.manual_seed(0)
    a = lm.Attention("sgate", d, heads).to(device)
    x = torch.randn(2, 128, d, device=device)
    q, k, _ = a.qkv_heads(x)
    l1 = a.operator(q, k).abs().sum(-1)
    l1 = l1[..., 1:]            # row 0 is strictly-causal-empty and exactly 0
    assert float(l1.min()) >= lo and float(l1.max()) <= hi, (
        "d={} H={}: row L1 spans {:.6e} to {:.6e}, outside [{:.6f}, {:.6f}]"
        .format(d, heads, float(l1.min()), float(l1.max()), lo, hi))


@pytest.mark.parametrize("layers", [4, 24])
def test_the_attention_branch_does_not_dominate_the_residual_at_depth(
        device, parity_point, layers):
    """`x = x + attn(n1(x))`. The branch that is added is the whole difference.

    Softmax returns a convex combination of `v`, so its branch RMS is at most
    that of `v`. `sgate` returns `v + Av + A^2 v` with row sum pinned at 1.2273,
    a DC gain of 3.734. Measured here as the ratio of branch RMS to residual RMS
    at the LAST layer, both arms, identical seed and identical input.

    The bar is 2x the softmax arm's own value. Anything past that and the
    residual stream at 24 layers is a different object from the one at 4, before
    any training has happened.
    """
    d, heads, seq = 256, 4, 128
    got = {}
    for kind in ("softmax", "sgate"):
        torch.manual_seed(0)
        m = lm.TinyLM(kind, d=d, n_layers=layers, n_heads=heads, seq=seq, seed=0).to(device)
        torch.manual_seed(1)
        x = torch.randn(2, seq, d, device=device)
        with torch.no_grad():
            h = x
            for blk in m.blocks:
                br = blk.attn(blk.n1(h))
                rms = lambda t: float(t.pow(2).mean().sqrt())
                got[kind] = rms(br) / rms(h)
                h = h + br
                h = h + blk.mlp(blk.n2(h))
    assert got["sgate"] <= 2.0 * got["softmax"], (
        "L={}: last-layer branch/residual RMS is {:.4f} for sgate against "
        "{:.4f} for softmax = {:.2f}x".format(
            layers, got["sgate"], got["softmax"], got["sgate"] / got["softmax"]))

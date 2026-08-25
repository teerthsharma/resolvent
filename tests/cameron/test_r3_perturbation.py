"""R3 -- amplify few, crush many.  VERDICT: DELETE.

REQUIREMENTS.md R3 falsifier, verbatim:

    perturbation ratio = (delta-output from editing a causal token) /
    (delta-output from editing a filler token). Must be large, and a
    standard-attention baseline must be flatter. Global sensitivity fails.
    Global deadness fails.

The deletion test is `test_module_perturbation_ratio_beats_standard_attention`.
It is RED. Measured on this file, 24 contexts, n=96 tokens, d=32:

    arm                        ratio      content R2    effective size / 32
    standard attention      1565.111        0.5577            1.0222
    row-stochastic gamma P  1618.510        1.0000            2.3322
    perron  A_ij = s_j P_ij  396.868        0.9997            2.5022

The baseline is not flatter. It is 3.94x SHARPER than the module. R3's own
falsifier deletes R3.

Two further measurements, both against expectation, both recorded here because
they change what the next requirement should be:

1. The recorded ceiling does not bind this axis. "a row-stochastic operator
   cannot do this ... best non-constant/constant gain ratio 1.000000 over 1,920
   kernels" is a statement about MODE GAIN and OUTPUT RANGE. The perturbation
   ratio is a statement about the COLUMNS of the resolvent, and row-stochastic
   scores 1618.510 on it -- the best of the three arms. Row-stochasticity was
   never the thing standing between the module and R3.

2. Freeing the row sums is not free. Row-stochastic row sums are pinned at
   gamma to 4.441e-16 under a filler edit; the Perron operator's move by
   5.961e-04, a horizon shift of 5.492e-02. Every filler edit becomes a
   perturbation of the reader's own horizon. The rigidity the Perron escape
   gives up was load-bearing, and that is why arm 3 loses to arm 2 by 4.08x on
   the very ratio it was built to win.

What DOES separate the module from attention is the third column: attention
reaches effective size 1.0222 out of 32 and content R2 0.5577 -- it buys the
ratio by collapsing the state onto the causal token. Both resolvent arms keep
R2 = 1.0000. That axis is R4's falsifier, not R3's. R3 carries no discriminating
content of its own, so it is deleted and its live clause moves to R4.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import perron as pn

N_TOK, D_FEAT = 96, 32
RHO = 0.9          # contraction rate
GAMMA = RHO        # arm 2 gets the SAME horizon, so horizon is not the variable
N_DRAWS = 24

RATIO_IS_LARGE = 10.0        # R3: "must be large"
BASELINE_IS_FLATTER = 3.0    # R3: "a standard-attention baseline must be flatter"
CONTENT_SURVIVES = 0.5       # R3: "Global deadness fails"


# --------------------------------------------------------------------- corpus

def context(seed: int, device, dtype=torch.float64):
    """One context: filler everywhere, exactly one causal token.

    Channel 0 is the control channel -- it carries "this token is a bound / a
    negation / a flag". Filler tokens carry noise there. Nothing else about a
    causal token is special: same scale, same distribution in every other
    channel, so a similarity-based reader has no shortcut.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(N_TOK, D_FEAT, generator=g, dtype=dtype)
    x[:, 0] *= 0.25
    causal = int(torch.randint(2, N_TOK, (1,), generator=g))
    filler = int(torch.randint(2, N_TOK, (1,), generator=g))
    while filler == causal:
        filler = max((filler + 1) % N_TOK, 2)
    x[causal, 0] = 3.0
    return x.to(device), causal, filler


def edit(x, j, seed, scale=1.0):
    """Edit token j. Same magnitude whichever token it is -- the ratio must
    come from the operator's response, not from a bigger poke."""
    g = torch.Generator(device="cpu").manual_seed(seed + 7919)
    d = torch.randn(x.shape[1], generator=g, dtype=x.dtype).to(x.device)
    y = x.clone()
    y[j] = y[j] + d / d.norm() * scale
    return y


def _delta(base, moved, j):
    """Frobenius change over the whole settled state with row j dropped.
    Editing a token's own slot is not evidence that anything propagated, and a
    one-hop reader wins any single-row readout by construction."""
    keep = [i for i in range(base.shape[0]) if i != j]
    return (moved[keep] - base[keep]).norm().item()


def sweep(arm, device, n=N_DRAWS):
    """Median over n contexts of every clause R3 names. Reporting the ratio
    alone hides two of the three."""
    ratio, causal, filler, r2, size = [], [], [], [], []
    for s in range(n):
        x, c, f = context(s, device)
        base = arm(x)
        keep = [i for i in range(N_TOK) if i != c]
        dc = _delta(base, arm(edit(x, c, s)), c)
        df = _delta(base, arm(edit(x, f, s)), f)
        ratio.append(dc / max(df, 1e-300))
        causal.append(dc)
        filler.append(df / max(base.norm().item(), 1e-300))
        r2.append(pn.linear_probe_r2(base[keep], x[keep]))
        size.append(pn.effective_size(base))
    med = lambda v: float(np.median(v))
    return dict(ratio=med(ratio), causal=med(causal), noise_admittance=med(filler),
                content_r2=med(r2), effective_size=med(size))


# ------------------------------------------------------------------- the arms

def arm_attention(x):
    return pn.attention(x)


def arm_row_stochastic(x):
    return pn.settle(GAMMA * pn.transition(x), x, GAMMA)


def arm_perron(x):
    return pn.settle(pn.perron_operator(x, rho=RHO), x, RHO)


# ---------------------------------------------------- R3's own deletion test

def test_module_perturbation_ratio_beats_standard_attention(device):
    """RED. This is the test that deletes R3.

    R3 requires the standard-attention baseline to be FLATTER. Given the same
    control signal -- attention gets robust_z(x[:,0]) as an additive score
    bias, exactly the quantity the module's gate reads, so neither side is
    denied the other's information -- one softmax hop reaches 1565.111 against
    the module's 396.868. The baseline is sharper, not flatter, and R3's
    falsifier has no second clause to fall back on.
    """
    mine = sweep(arm_perron, device)["ratio"]
    base = sweep(arm_attention, device)["ratio"]
    assert mine >= BASELINE_IS_FLATTER * base, (
        f"perron ratio {mine:.3f} vs attention ratio {base:.3f}: factor "
        f"{mine / max(base, 1e-300):.3f} < {BASELINE_IS_FLATTER}. R3 deleted."
    )


# ------------------------------------------- what is true, measured and green

def test_row_stochastic_resolvent_does_deliver_the_perturbation_ratio(device):
    """The recorded ceiling does not bind this axis.

    [min(b), max(b)]/(1-gamma) and "gain ratio 1.000000" constrain the
    resolvent's MODE gains and output RANGE. The perturbation ratio reads its
    COLUMNS, and those are unconstrained by row-stochasticity: row sums are
    pinned but their allocation is free.
    """
    got = sweep(arm_row_stochastic, device)
    assert got["ratio"] >= RATIO_IS_LARGE, got


def test_perron_operator_also_delivers_it_but_by_less(device):
    got = sweep(arm_perron, device)
    assert got["ratio"] >= RATIO_IS_LARGE, got


def test_freeing_the_row_sums_opens_a_noise_path(device):
    """Why arm 3 loses to arm 2 on its own axis.

    A row-stochastic operator's row sums cannot move: they are gamma by
    construction. A Perron operator's row sum is sum_j P_ij s_j, so editing ANY
    token shifts every reader's effective horizon 1/(1 - rowsum). The freed
    degree of freedom is also a new input path for noise.
    """
    pinned, freed = [], []
    for s in range(N_DRAWS):
        x, _, f = context(s, device)
        y = edit(x, f, s)
        p0, p1 = pn.transition(x), pn.transition(y)
        a0 = p0 * pn.salience(x, RHO)[None, :]
        a1 = p1 * pn.salience(y, RHO)[None, :]
        pinned.append(float((GAMMA * p1.sum(1) - GAMMA * p0.sum(1)).abs().max()))
        freed.append(float((a1.sum(1) - a0.sum(1)).abs().max()))
    pinned, freed = float(np.median(pinned)), float(np.median(freed))
    assert pinned < 1e-14 < freed, f"pinned={pinned:.3e} freed={freed:.3e}"


def test_the_certificate_actually_holds_on_every_draw(device):
    """A ratio bought by giving up contraction is not a pass. Check the Lean
    hypothesis pointwise: A >= 0, w > 0, A w <= rho w, rho < 1."""
    for s in range(N_DRAWS):
        x, _, _ = context(s, device)
        a = pn.perron_operator(x, rho=RHO)
        w = torch.ones(N_TOK, dtype=a.dtype, device=a.device)
        ok, rho, wmin = pn.certificate(a, w)
        assert ok and rho < 1.0, f"draw {s}: rho={rho} min w={wmin}"


def test_only_effective_size_separates_the_module_from_attention(device):
    """The collapse evidence. Attention matches the module on the ratio and is
    within an order of magnitude on scale-free noise admittance; it separates
    on effective size and content reconstruction, which is R4's falsifier and
    not R3's. R3's live clause is therefore R4's clause.
    """
    mine, base = sweep(arm_row_stochastic, device), sweep(arm_attention, device)
    assert base["ratio"] >= RATIO_IS_LARGE                            # no separation
    assert base["noise_admittance"] <= 10 * mine["noise_admittance"]  # no separation
    assert mine["effective_size"] >= 2 * base["effective_size"], (
        f"module {mine['effective_size']:.4f} vs attention "
        f"{base['effective_size']:.4f} of {D_FEAT}")
    assert mine["content_r2"] - base["content_r2"] >= 0.3, (
        f"module R2 {mine['content_r2']:.4f} vs attention {base['content_r2']:.4f}")


def test_crushing_is_not_global_deadness(device):
    """'Global deadness fails.' The settled state must still carry each token's
    own content."""
    r2 = sweep(arm_perron, device)["content_r2"]
    assert r2 >= CONTENT_SURVIVES, f"non-causal content recoverable at R2={r2:.4f}"

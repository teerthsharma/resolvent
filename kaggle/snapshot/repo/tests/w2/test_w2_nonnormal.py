"""W2 -- the non-normal operator. Replaces the deleted R3.

R3 died because a Perron-certified operator scored 396.868 on the perturbation
ratio against standard attention's 1565.111: the baseline was 3.94x SHARPER
where R3 required it flatter.

The lead that survived Cameron's round: `monodromy` measures Henon a=1.4 b=0.3
by Benettin QR at lambda_1 = +0.42084 while det = 0.3 contracts. Amplify-while-
contract is real, and it needs a NON-NORMAL operator with transient growth --
not a relaxed row-sum constraint. Cameron's arm 3 freed the row sums, which is
a different degree of freedom entirely, and it lost.

The operator under test here is STRICTLY CAUSAL: lower triangular, zero
diagonal. Three consequences, and the first two are free:

  1. It is nilpotent. rho(A) = 0 exactly, so contraction needs no certificate
     and the reducible-A failure mode (min(w) = 6.6e-14, weighted norm
     undefined) cannot arise.
  2. (I - A)^-1 = I + A + ... + A^(N-1) TERMINATES. It is exact, not a
     truncated Neumann series of unknown quality.
  3. It is maximally non-normal. A strictly triangular matrix is the canonical
     non-normal family, and its resolvent's columns are wildly unequal -- which
     is precisely the quantity the perturbation ratio reads.

A dense operator lets every token reach every other in one hop, so path
structure carries no information. Causality is what makes "what a token causes
downstream" a different question from "what a token resembles".
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import perron as pn
from ceq import nonnormal as nn

N_TOK, D_FEAT = 96, 32
RHO = 0.9
N_DRAWS = 24

BASELINE_IS_FLATTER = 3.0   # module ratio must exceed attention's by this factor
CONTENT_SURVIVES = 0.99     # W2 holds content-R2 at or above this


def context(seed, device, dtype=torch.float64):
    """Identical corpus to tests/cameron/test_r3_perturbation.py."""
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
    g = torch.Generator(device="cpu").manual_seed(seed + 7919)
    d = torch.randn(x.shape[1], generator=g, dtype=x.dtype).to(x.device)
    y = x.clone()
    y[j] = y[j] + d / d.norm() * scale
    return y


def _delta(base, moved, j):
    keep = [i for i in range(base.shape[0]) if i != j]
    return (moved[keep] - base[keep]).norm().item()


def sweep(arm, device, n=N_DRAWS):
    ratio, r2, size = [], [], []
    for s in range(n):
        x, c, f = context(s, device)
        base = arm(x)
        keep = [i for i in range(N_TOK) if i != c]
        dc = _delta(base, arm(edit(x, c, s)), c)
        df = _delta(base, arm(edit(x, f, s)), f)
        ratio.append(dc / max(df, 1e-300))
        r2.append(pn.linear_probe_r2(base[keep], x[keep]))
        size.append(pn.effective_size(base))
    med = lambda v: float(np.median(v))
    return dict(ratio=med(ratio), content_r2=med(r2), effective_size=med(size))


def arm_attention(x):
    return pn.attention(x)


def arm_causal_nonnormal(x):
    return nn.settle_exact(nn.causal_operator(x, rho=RHO), x)


# ----------------------------------------------------------- W2's own falsifier

def test_non_normal_operator_beats_attention_on_perturbation_ratio(device):
    """RED first. If a non-normal operator cannot clear the attention baseline
    either, amplify-few-crush-many is not achievable in this family and stays
    deleted."""
    mine = sweep(arm_causal_nonnormal, device)
    base = sweep(arm_attention, device)
    assert mine["ratio"] >= BASELINE_IS_FLATTER * base["ratio"], (
        f"causal non-normal ratio {mine['ratio']:.3f} vs attention "
        f"{base['ratio']:.3f}: factor {mine['ratio'] / max(base['ratio'], 1e-300):.3f} "
        f"< {BASELINE_IS_FLATTER}")


def test_content_survives_the_amplification(device):
    """Global deadness fails. An operator that wins the ratio by collapsing the
    state onto one token scores near zero here."""
    got = sweep(arm_causal_nonnormal, device)
    assert got["content_r2"] >= CONTENT_SURVIVES, got


def test_operator_is_nilpotent_so_contraction_is_free(device):
    """rho(A) = 0 exactly. No Perron certificate, no power iteration, so the
    reducible-A failure mode cannot arise at all."""
    for s in range(8):
        x, _, _ = context(s, device)
        a = nn.causal_operator(x, rho=RHO)
        assert float(a.triu(0).abs().max()) == 0.0, "not strictly lower triangular"
        ev = torch.linalg.eigvals(a.to(torch.complex128))
        assert float(ev.abs().max()) < 1e-12, f"draw {s}: rho={float(ev.abs().max())}"


def test_resolvent_terminates_and_is_exact(device):
    """(I - A) z = b to machine precision, with a FINITE sum. Not a truncation."""
    for s in range(8):
        x, _, _ = context(s, device)
        a = nn.causal_operator(x, rho=RHO)
        z = nn.settle_exact(a, x)
        resid = float((z - a @ z - x).abs().max())
        assert resid < 1e-12, f"draw {s}: residual {resid:.3e}"


def test_row_zero_has_no_predecessors_and_does_not_nan(device):
    """The all-masked-row case tda-tdd flags: token 0 has an empty causal set.
    A softmax over nothing is NaN; the operator must give it a zero row."""
    for s in range(8):
        x, _, _ = context(s, device)
        a = nn.causal_operator(x, rho=RHO)
        assert torch.isfinite(a).all(), f"draw {s}: non-finite entries"
        assert float(a[0].abs().max()) == 0.0, "row 0 must be empty"
        z = nn.settle_exact(a, x)
        assert torch.isfinite(z).all()


# ------------------------------------------- what the instrument check proved

def test_non_normality_does_not_explain_the_perturbation_ratio(device):
    """The refutation. W2's premise was that NON-NORMALITY buys the ratio.

    Henrici departure from normality, ||A A^T - A^T A||_F / ||A||_F^2, median
    over 8 draws:

        row-stochastic gamma P   1.40679     ratio 1618.510
        perron  s_j P_ij         1.40682     ratio  396.868
        causal nilpotent         1.40972     ratio  788.077

    Three operators within 0.21% of each other on departure-from-normality,
    spanning 4.08x on the ratio. Non-normality is not the variable. The
    `monodromy` lead -- Henon lambda_1 = +0.42084 amplifying while det = 0.3
    contracts -- is a true statement about that map and does not transfer here.
    """
    deps, ratios = [], []
    for f, arm in ((lambda x: pn.transition(x) * RHO, lambda x: pn.settle(RHO * pn.transition(x), x, RHO)),
                   (lambda x: pn.perron_operator(x, RHO), lambda x: pn.settle(pn.perron_operator(x, RHO), x, RHO)),
                   (lambda x: nn.causal_operator(x, RHO), arm_causal_nonnormal)):
        d = [nn.non_normality(f(context(s, device)[0])) for s in range(8)]
        deps.append(float(np.median(d)))
        ratios.append(sweep(arm, device, n=8)["ratio"])
    spread_dep = max(deps) / min(deps)
    spread_ratio = max(ratios) / min(ratios)
    assert spread_dep < 1.01, f"departures not matched: {deps}"
    assert spread_ratio > 2.0, f"ratios not spread: {ratios}"


def test_causal_nilpotent_is_better_conditioned_than_the_resolvent_arms(device):
    """Free by-product of nilpotency, and the reason this operator is kept even
    though its own falsifier deleted it.

    rho(A) = 0 exactly, so cond(I - A) does not inherit the 1/(1-rho) blowup.
    Measured: 48.92 against 771.12 (row-stochastic) and 747.95 (Perron), a 15x
    improvement, on the same draws.
    """
    mine, theirs = [], []
    for s in range(8):
        x, _, _ = context(s, device)
        eye = torch.eye(N_TOK, dtype=x.dtype, device=x.device)
        mine.append(float(torch.linalg.cond(eye - nn.causal_operator(x, RHO))))
        theirs.append(float(torch.linalg.cond(eye - RHO * pn.transition(x))))
    assert np.median(mine) < np.median(theirs) / 5.0, (np.median(mine), np.median(theirs))


def test_attention_buys_its_ratio_by_going_globally_dead(device):
    """R3's falsifier has THREE clauses and attention fails the third.

    'Global deadness fails.' Attention reaches ratio 1565.111 at content R2
    0.5577 and effective size 1.0087 of 32 -- it collapses the state onto the
    causal token. The causal operator holds R2 0.9951 at effective size 21.5281.

    This does not rescue W2. Its falsifier as written compares ratios and the
    causal operator loses that comparison 0.504x. It is recorded because any
    restatement of the requirement has to decide whether ratio-at-matched-
    content is the quantity it actually meant.
    """
    base = sweep(arm_attention, device)
    mine = sweep(arm_causal_nonnormal, device)
    assert base["content_r2"] < 0.75, base
    assert base["effective_size"] < 2.0, base
    assert mine["content_r2"] > 0.99 and mine["effective_size"] > 10.0, mine

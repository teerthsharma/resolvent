"""W3 -- eviction as its own requirement.

R4 ("compress by forgetting") was deleted because its own controls collapsed the
gap: a RETAINED filler the gate never crushed scored -0.11224 against crushed
-0.11446, a gap of +0.00222 against a pre-registered 0.25. The gate was not
forgetting anything.

Cameron's unbound observation, which R4's wording excluded because it is a rule
outside the operator: gating does not forget, EVICTION does.

The mechanism, stated precisely enough to test:

    A gate applied AFTER the softmax multiplies a column by s_j ~ 0. The token's
    contribution vanishes but its score never left the softmax DENOMINATOR, so
    every surviving token is still divided by a normalizer that includes the
    crushed token's mass. The crushed token goes on shadowing the state forever,
    and perturbing it still moves every row.

    Eviction removes the token from the key set. It is not in the denominator, it
    is not in the computation, and perturbing it cannot move anything -- not
    approximately, BITWISE.

That is the difference between "the weight is zero" and "the token was never
there", and it is the only forgetting that is structural rather than cosmetic.

R4's first sentence demanded the forgetting be a property of the operator rather
than a rule outside it. Either that wording was wrong or this is a seventh
requirement. It is decided here by test, not by argument.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import perron as pn
from ceq import eviction as ev
from ceq import nonnormal as nn

N_TOK, D_FEAT = 96, 32
RHO = 0.9
KEEP = 16          # Cameron's operating point: 16 of 96
N_DRAWS = 24


def context(seed, device, dtype=torch.float64):
    """Same corpus as W2 and as tests/cameron/test_r3_perturbation.py."""
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


# --------------------------------------------------------------- W3 falsifier

def test_eviction_forgets_where_gating_cannot(device):
    """RED first.

    Pick a token the salience gate crushes. Perturb it. Measure how far the
    settled state moves under each regime, with the keep-set held FIXED from the
    unperturbed context so that the two arms differ only in how the crushed
    token is removed and not in which tokens are removed.

    Gating must move. Eviction must be bitwise zero. If eviction also moves, the
    distinction is cosmetic and W3 is deleted with R4.
    """
    gated, evicted = [], []
    for s in range(N_DRAWS):
        x, _, _ = context(s, device)
        keep = ev.keep_indices(x, KEEP, RHO)
        crushed = ev.lowest_salience_token(x, RHO, exclude=keep)
        y = edit(x, crushed, s)

        g0 = ev.settle_gated(x, keep, RHO)
        g1 = ev.settle_gated(y, keep, RHO)
        gated.append(float((g1 - g0).abs().max()))

        e0 = ev.settle_evicted(x, keep, RHO)
        e1 = ev.settle_evicted(y, keep, RHO)
        evicted.append(float((e1 - e0).abs().max()))

    g, e = float(np.median(gated)), float(np.median(evicted))
    assert e == 0.0, f"eviction is not bitwise: moved {e:.3e}"
    assert g > 0.0, f"gating did not move at all: {g:.3e}; nothing to distinguish"


def test_gating_leaves_the_crushed_token_in_the_denominator(device):
    """The mechanism, isolated from the settled state.

    A post-softmax gate cannot change the softmax denominator, so the surviving
    tokens' weights do not sum to one and the deficit is exactly the crushed
    mass. Under eviction the survivors renormalize and the deficit is zero.
    """
    deficits_gated, deficits_evicted = [], []
    for s in range(8):
        x, _, _ = context(s, device)
        keep = ev.keep_indices(x, KEEP, RHO)
        a_g = ev.gated_operator(x, keep, RHO)
        a_e = ev.evicted_operator(x, keep, RHO)
        deficits_gated.append(float((a_g.sum(-1) / RHO - 1.0).abs().median()))
        deficits_evicted.append(float((a_e.sum(-1) / RHO - 1.0).abs().median()))
    assert np.median(deficits_gated) > 1e-3, np.median(deficits_gated)
    assert np.median(deficits_evicted) < np.median(deficits_gated) / 10.0, (
        np.median(deficits_evicted), np.median(deficits_gated))


def _subspace_r2(z: torch.Tensor, target: torch.Tensor) -> float:
    """Fraction of `target` lying in the row space of the compressed state `z`.

    This is the honest form of the question. Reconstructing x[keep] FROM
    settle(x[keep]) is an invertible linear map of the thing itself and returns
    1.0 for every keep-rule, which is what the first version of this test
    measured and why it said nothing.
    """
    q, _ = torch.linalg.qr(z.transpose(-2, -1))
    proj = q @ (q.transpose(-2, -1) @ target)
    return float(1.0 - (target - proj).norm() ** 2 / target.norm() ** 2)


def test_salience_eviction_retains_the_token_that_matters(device):
    """The keep-rule must retain the causal token, not merely retain something.

    KEEP = 16 of 96. Recency keeps the last 16 regardless of content; random
    keeps 16 at chance. A keep-rule that cannot beat 'keep the last 16' is a
    sliding window wearing a different name.
    """
    rate = {k: 0 for k in ("salience", "random", "recency")}
    for s in range(N_DRAWS):
        x, c, _ = context(s, device)
        for rule, idx in (("salience", ev.keep_indices(x, KEEP, RHO)),
                          ("random", ev.random_indices(x, KEEP, seed=s)),
                          ("recency", ev.recency_indices(x, KEEP))):
            rate[rule] += int((idx == c).any())
    assert rate["salience"] > rate["random"], rate
    assert rate["salience"] > rate["recency"], rate


def test_compressed_state_still_spans_the_causal_content(device):
    """Task-relevant content survives compression; the controls lose it.

    Under eviction a token outside `keep` is not in the computation at all, so
    its content can only reach the settled state if the keep-rule kept it. The
    state is KEEP x D with KEEP < D, so a subspace this size captures a fixed
    fraction of a random direction by chance -- that chance level is what the
    controls measure and what salience has to beat.
    """
    got = {k: [] for k in ("salience", "random", "recency")}
    for s in range(N_DRAWS):
        x, c, _ = context(s, device)
        for rule, idx in (("salience", ev.keep_indices(x, KEEP, RHO)),
                          ("random", ev.random_indices(x, KEEP, seed=s)),
                          ("recency", ev.recency_indices(x, KEEP))):
            got[rule].append(_subspace_r2(ev.settle_evicted(x, idx, RHO), x[c]))
    med = {k: float(np.median(v)) for k, v in got.items()}
    assert med["salience"] > med["random"] + 0.2, med
    assert med["salience"] > med["recency"] + 0.2, med


def test_evicted_operator_is_still_nilpotent_and_exact(device):
    """W2's kept by-product must survive eviction: strict causality gives
    rho(A) = 0, so the resolvent is a terminating finite sum and no certificate
    is consulted."""
    for s in range(8):
        x, _, _ = context(s, device)
        keep = ev.keep_indices(x, KEEP, RHO)
        a = ev.evicted_operator(x, keep, RHO)
        assert float(a.triu(0).abs().max()) == 0.0
        assert nn.spectral_radius(a) < 1e-12
        z = nn.settle_exact(a, x[keep])
        assert float((z - a @ z - x[keep]).abs().max()) < 1e-12
        assert torch.isfinite(z).all()

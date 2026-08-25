"""H8 -- rho is the self/other weight ratio and it has never been tuned.

WHAT rho ACTUALLY CONTROLS. With `lam = 0` the operator is `rho * softmax(w)`
with the DIAGONAL EXCLUDED, and the output is

    o = v + A v + A^2 v

The identity term supplies the token's own value with coefficient exactly **1**,
while every other token is capped at total weight `rho`. So the self-versus-other
balance is pinned at `1 : rho`. Softmax attention learns that balance freely --
its diagonal is just another logit.

`rho = 0.9` was chosen in the first hour of this project and every measurement
since has inherited it. It has never been swept. That is the cheapest untested
degree of freedom on the board.

WHY rho > 1 IS LEGAL HERE, and this is not a technicality. Nilpotency is
STRUCTURAL: `CEQ.Nilpotent.pow_card_eq_zero` needs only
`forall i j, i <= j -> A i j = 0`, with no bound on entry magnitude. So `A^n = 0`
and the series terminates exactly at any rho. What rho > 1 costs is the
CONTRACTION bound `rho^(K+1)/(1-rho)`, which is a statement about truncating an
infinite series -- and this series is finite, so there is nothing to truncate.
The bound was never load-bearing for a strictly triangular operator.

That means rho is a free scalar, not a stability parameter, and treating it as
one has been an unexamined inheritance.

EVIDENCE CLASSES. Structural checks are RUN. The sweep is RUN over 3 seeds with
the spread reported.
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


def test_nilpotency_survives_rho_above_one(device):
    """The claim that makes rho a free scalar rather than a stability knob.

    `pow_card_eq_zero` bounds nothing about entry magnitude, so A^n = 0 at any
    rho. If this fails, rho is genuinely constrained and the sweep below must
    stop at 1.
    """
    g = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(1, 4, 32, 32, generator=g, dtype=torch.float64).to(device)
    old = lm.RHO
    try:
        for rho in (0.9, 1.5, 3.0):
            lm.RHO = rho
            a = lm.TinyLM(kind="sgate").to(device).blocks[0].attn.operator(q, q)
            assert float(a.triu(0).abs().max()) == 0.0
            ev = torch.linalg.eigvals(a.reshape(-1, 32, 32).to(torch.complex128))
            assert float(ev.abs().max()) < 1e-12, (rho, float(ev.abs().max()))
            assert float(torch.linalg.matrix_power(a, 32).abs().max()) == 0.0, rho
    finally:
        lm.RHO = old


def test_rho_actually_moves_the_self_other_ratio(device):
    """Calibrate the claim. If rho did not change the balance between the
    identity term and the attention terms, the sweep would be testing nothing."""
    g = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(1, 4, 32, 32, generator=g, dtype=torch.float64).to(device)
    v = torch.randn(1, 4, 32, 32, generator=g, dtype=torch.float64).to(device)
    old = lm.RHO
    ratios = []
    try:
        for rho in (0.5, 0.9, 1.5):
            lm.RHO = rho
            a = lm.TinyLM(kind="sgate").to(device).blocks[0].attn.operator(q, q)
            o = lm.path_sum_terms(a, v, 2, include_identity=True)
            ratios.append(float((o - v).abs().mean() / v.abs().mean()))
    finally:
        lm.RHO = old
    assert ratios[0] < ratios[1] < ratios[2], ratios


@pytest.mark.slow
def test_some_rho_reaches_parity_over_three_seeds(device, corpus):
    """RED first. The sweep is reported in full, not just its best point."""
    old_r, old_l, old_h = lm.RHO, lm.SGATE_LAM, lm.HOPS
    lm.SGATE_LAM, lm.HOPS = 0.25, 2
    try:
        sm = [lm.train_one("softmax", corpus, steps=600, device=device, seed=s)["val_loss"]
              for s in SEEDS]
        best, table = None, []
        for rho in (0.5, 0.9, 1.5, 2.5):
            lm.RHO = rho
            vs = [lm.train_one("sgate", corpus, steps=600, device=device, seed=s)["val_loss"]
                  for s in SEEDS]
            rs = [a / b for a, b in zip(vs, sm)]
            m = statistics.median(rs)
            table.append((rho, m, min(rs), max(rs)))
            best = m if best is None else min(best, m)
        assert best <= PARITY_BAR, f"best median {best:.4f}, bar {PARITY_BAR}. table={table}"
    finally:
        lm.RHO, lm.SGATE_LAM, lm.HOPS = old_r, old_l, old_h

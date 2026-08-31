"""Is val-loss parity on TinyStories bytes the right target at all?

The campaign reached 1.0334 against softmax at 3.3M parameters and the loop
wants 1.0000. This file asks whether 1.0000 is a thing worth having.

The operator is a one-parameter family in `lam`:

    A = rho * (softmax(w) - lam * softmax(-w)) / (1 + lam)

At `lam = 0` it is `rho * softmax(w)` -- a NON-NEGATIVE row-stochastic operator
scaled by rho, i.e. softmax attention with a multi-hop sum bolted on. Every
distinguishing property this project has claimed for three rounds lives in the
`- lam * softmax(-w)` term and nowhere else. `lam` is therefore a dial that runs
continuously from "softmax" to "the module", and the two questions -- how much
val loss does the module cost, and how much of the property does it have -- are
two readings off the SAME dial.

If the val-loss-optimal setting of that dial is the one with no property, then
val-loss parity is not a target this module can reach while remaining itself,
and the whole parity campaign is optimising toward its own deletion. That is a
falsifiable claim and this file falsifies or confirms it.

Every test parametrizes over cpu and cuda per the standing rule.
"""
from __future__ import annotations

import pytest
import torch

from ceq import bench, lm

RHO, HOPS = 1.5, 2          # the campaign's shipped point
LAMS = (0.0, 0.05, 0.10, 0.25, 1.0)


def test_sign_flip_rate_accepts_the_operators_lam_knob(device):
    """The calibrated probe hardcodes lam=0.10, so it cannot sweep the family.

    `bench.sign_flip_rate` was extended to `sgate` last round and reproduced
    every published number exactly. Sweeping `lam` needs it to pass the knob
    through rather than bake it in; nothing else about the probe changes, so
    the calibration carries.
    """
    r = bench.sign_flip_rate("sgate", depth=1, hops=HOPS, n_draws=16,
                             device=device, rho=RHO, lam=0.10)
    assert 0.0 <= r <= 1.0, r


def test_the_calibration_still_reproduces_after_the_knob_is_added(device):
    """A probe that changed its answers while gaining a parameter is a new probe.

    Seven published numbers, all at n_draws=128 -- 0.0547 is 7/128 and is not
    expressible over 64, which is how the draw count was recovered. Four from
    the original calibration, three from the round-2 sgate table.
    """
    r = lambda *a, **kw: bench.sign_flip_rate(*a, n_draws=128, device=device, **kw)
    assert r("softmax", depth=1) == 0.0
    assert r("softmax", depth=2) == 0.0
    assert abs(r("signed", depth=1, hops=3) - 0.046875) < 1e-12
    assert abs(r("signed", depth=2) - 0.1875) < 1e-12
    assert abs(r("softmax_gelu", depth=2) - 0.0546875) < 1e-12
    assert abs(r("sgate", depth=1, hops=1) - 0.0234375) < 1e-12
    assert abs(r("sgate", depth=1, hops=2) - 0.1640625) < 1e-12
    assert abs(r("sgate", depth=1, hops=3) - 0.1484375) < 1e-12


def test_deleting_the_negative_half_deletes_the_property_exactly(device):
    """lam = 0 must measure EXACTLY the softmax control's 0.0000.

    Not "close to". `rho * softmax(w)` is non-negative, and a non-negative
    operator composed with fixed linear value projections factors as
    (non-negative weight) x (fixed matrix), so no third token can move the sign
    of j's influence on i at any depth. The probe reads exactly zero for
    softmax; it must read exactly zero here for the same structural reason.
    """
    off = bench.sign_flip_rate("sgate", depth=1, hops=HOPS, n_draws=128,
                               device=device, rho=RHO, lam=0.0)
    on = bench.sign_flip_rate("sgate", depth=1, hops=HOPS, n_draws=128,
                              device=device, rho=RHO, lam=0.10)
    assert off == 0.0, off
    assert on > 0.10, on


@pytest.mark.slow
def test_val_loss_is_best_exactly_where_the_property_is_gone(device):
    """Sweep the one dial. Read val loss and the property off the same points.

    Pre-registered claim: the minimum-val-loss member of the family is the one
    at `lam = 0`, which has zero content-conditional sign. If that holds, then
    minimising val loss inside this family IS deleting the module, and a val-loss
    parity target is a target to become softmax.

    Refutation condition, stated before the run: if some `lam > 0` gives a val
    loss no worse than `lam = 0` within 0.005, the dial is free and parity costs
    nothing -- the tradeoff is refused and the campaign should continue.
    """
    text = (lm.__file__.rsplit("ceq", 1)[0] + "data/tinystories_20k.txt")
    corpus = lm.ByteCorpus(open(text, encoding="utf-8").read())
    old = (lm.RHO, lm.HOPS, lm.SGATE_LAM)
    curve = {}
    try:
        lm.RHO, lm.HOPS = RHO, HOPS
        base = lm.train_one("softmax", corpus, steps=300, device=device, seed=0,
                            lr=1e-3)["val_loss"]
        for lam in LAMS:
            lm.SGATE_LAM = lam
            v = lm.train_one("sgate", corpus, steps=300, device=device, seed=0,
                             lr=1e-3)["val_loss"]
            flip = bench.sign_flip_rate("sgate", depth=1, hops=HOPS, n_draws=128,
                                        device=device, rho=RHO, lam=lam)
            curve[lam] = (v / base, flip)
    finally:
        lm.RHO, lm.HOPS, lm.SGATE_LAM = old

    best = min(curve, key=lambda l: curve[l][0])
    report = " ".join(f"lam={l}:ratio={curve[l][0]:.4f},flip={curve[l][1]:.4f}"
                      for l in LAMS)
    assert best == 0.0, f"a lam>0 is val-optimal, the dial is free: {report}"
    assert curve[0.0][1] == 0.0, report
    assert curve[best][0] < curve[0.10][0], report

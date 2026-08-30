"""MARS / MORIARTY, R9 iteration 4 -- the three GREENs nobody had attacked.

Filed AFTER Venus's dated prediction, so she cannot revise and this cannot have
shaped her number.

One attack each on M4 eviction, the E4-prime gates and the calibrated M3 bar.
Each is a VALUE test with the number that would prove it. Two fire on the
verdict; one fires as a method defect that the clause's margin absorbs.

NO WALL CLOCK IS TAKEN. The box has a live run holding the journal lock; every
number below is a pure recomputation on drawn tensors.
"""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from ceq import eviction as ev
from ceq.rips import CASES, make_case
from scale.rips_gate import PASS_BAR, adjacency, fit_eval, local_features

KEEP, RHO, S, D = 6, 0.9, 16, 8


def _context(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(1000 + seed)
    return torch.randn(S, D, generator=g)


# ============================================================== M4 EVICTION ==
def test_M4_the_rejection_region_is_empty_not_merely_unexercised():
    """FIRES, and Venus is not being too harsh -- she understates it.

    Her ruling is that M4's must-fire is insufficient because *"it moves the
    sibling quantity, not the one reading 0.000000e+00"*. That is right, and the
    stronger fact is that **no sufficient must-fire can be written**:
    `settle_evicted` is `settle_exact(evicted_operator(x, keep, rho), x[keep])`
    and both arguments are functions of `x[keep]` alone
    (`ceq/eviction.py:103, 118-121`). The crushed token is excluded from `keep`
    by `lowest_salience_token(x, rho, exclude=keep)`, so no perturbation of it --
    of ANY magnitude, of any kind -- can move the output.

    The test below replaces the crushed token with 1e0, 1e6, 1e12, `inf` and
    `nan` and requires the output to stay BITWISE identical every time. A `nan`
    that does not propagate is proof the value is never read: `nan` poisons every
    arithmetic path it touches, so its absence from the output is not evidence
    of a small effect, it is evidence of no path at all.

    That makes M4's headline `0.000000e+00` a statement about the indexing, and
    the existing must-fire -- perturbing a token INSIDE `keep` -- can only ever
    show that `settle_evicted` reads its own arguments.
    """
    magnitudes = [1.0, 1e6, 1e12, float("inf"), float("nan")]
    checked = 0
    for s in range(8):
        x = _context(s)
        keep = ev.keep_indices(x, KEEP, RHO)
        crushed = ev.lowest_salience_token(x, RHO, exclude=keep)
        assert not bool((keep == crushed).any())
        base = ev.settle_evicted(x, keep, RHO)
        for m in magnitudes:
            y = x.clone()
            y[crushed] = torch.full((D,), m)
            got = ev.settle_evicted(y, keep, RHO)
            assert torch.equal(base, got), (
                f"seed {s}, magnitude {m}: the crushed slot reached the output, "
                f"so the zero is a measurement after all")
            checked += 1
    assert checked == 8 * len(magnitudes) == 40


def test_M4_the_control_that_would_have_been_sufficient_does_not_exist():
    """The shape of the missing arm, stated as a value.

    A sufficient must-fire would perturb a token the claim says is IRRELEVANT
    and show the instrument COULD have seen it. `settle_gated` is that
    instrument -- it keeps the crushed token in the softmax denominator -- and it
    moves under exactly the perturbation `settle_evicted` cannot see. The two
    are therefore not measuring the same thing on the crushed arm, which is what
    makes `0.000000e+00` against `2.154868e-05` not a like-for-like comparison.
    """
    evicted_moves = gated_moves = 0
    for s in range(8):
        x = _context(s)
        keep = ev.keep_indices(x, KEEP, RHO)
        crushed = ev.lowest_salience_token(x, RHO, exclude=keep)
        y = x.clone()
        g = torch.Generator().manual_seed(7000 + s)
        y[crushed] = torch.randn(D, generator=g)
        evicted_moves += int(float((ev.settle_evicted(y, keep, RHO)
                                    - ev.settle_evicted(x, keep, RHO)).abs().max()) > 0.0)
        gated_moves += int(float((ev.settle_gated(y, keep, RHO)
                                  - ev.settle_gated(x, keep, RHO)).abs().max()) > 0.0)
    assert evicted_moves == 0 and gated_moves == 8, (evicted_moves, gated_moves)


# ============================================================ E4-PRIME GATES ==
@pytest.fixture(scope="module")
def critical_case():
    spec = [c for c in CASES if c[0] == "CriticalLarge_S2Rips_1024"][0]
    case = make_case(*spec)
    return case, [adjacency(1024, case.edges)]


def _draw(lab, draw_seed: int, split_seed: int, n_each: int = 1024):
    """`scale/rips_gate.py:draw_balanced_marginal`, verbatim except that its two
    hardcoded seeds become parameters. Nothing else is changed."""
    rng = random.Random(draw_seed)
    pos, neg = [], []
    for _ in range(600_000):
        if len(pos) >= n_each and len(neg) >= n_each:
            break
        i, j = rng.randrange(1024), rng.randrange(1024)
        if i == j:
            continue
        same = lab[i] >= 0 and lab[i] == lab[j]
        bucket = pos if same else neg
        if len(bucket) < n_each:
            bucket.append((i, j))
    if len(pos) < n_each or len(neg) < n_each:
        return None
    pairs = [(p, 0) for p in pos] + [(p, 0) for p in neg]
    y = np.asarray([1.0] * n_each + [0.0] * n_each)
    order = np.random.RandomState(split_seed).permutation(len(y))
    return [pairs[t] for t in order], y[order]


def test_E4_the_strike_hangs_on_one_unrepeatable_draw(critical_case):
    """FIRES. The verdict that killed E4 is 2.7 sd from its own bar and one
    draw in twenty reverses it.

    `draw_balanced_marginal` seeds itself `random.Random(0x33960000 ^ case.seed)`
    and splits on `np.random.RandomState(0)` -- both hardcoded, neither a
    parameter, so the sampling spread of the deciding number has never been
    measured. `test_the_degree_decoder_passes_at_criticality_so_e4_is_struck`
    then conditions the whole strike on `score < PASS_BAR` at a margin of
    `0.028955`.

    Measured, the draw logic reproduced verbatim with its two seeds exposed:

        shipped draw, shipped split      0.471045   margin to 0.5: 0.028955
        20 other draw seeds              min 0.459390  max 0.506250
                                         mean 0.479472  sd 0.010802
                                         1 of 20 would REVERSE the strike
        20 other split seeds             min 0.471045  max 0.498577
                                         mean 0.482682  sd 0.008182
                                         0 of 20 reverse; closest is 0.001423 off

    The shipped draw sits 0.8 sd BELOW the mean of its own sampling
    distribution, i.e. on the favourable side of the strike it licenses.

    THE NUMBER THAT WOULD PROVE IT: `P(score >= PASS_BAR)` over draws. Measured
    at 1/20 = 5% on 20 seeds, which carries its own +-5% at that sample size.
    Not a claim that the strike is wrong -- a claim that it is reported without
    the only error bar that could support it.
    """
    case, adj = critical_case
    shipped_draw = 0x33960000 ^ case.seed
    pairs, y = _draw(case.partition, shipped_draw, 0)
    shipped = fit_eval(local_features(pairs, adj, 0), y)
    assert round(shipped, 6) == 0.471045, shipped
    assert shipped < PASS_BAR

    scores = []
    for k in range(20):
        pairs, y = _draw(case.partition, shipped_draw + 1 + k, 0)
        scores.append(fit_eval(local_features(pairs, adj, 0), y))
    scores = np.asarray(scores)

    reversals = int((scores >= PASS_BAR).sum())
    assert reversals >= 1, (
        f"no draw of 20 reversed the strike; spread {scores.min():.6f}.."
        f"{scores.max():.6f} sd {scores.std():.6f}")
    #: and the shipped reading is on the favourable tail of that spread.
    assert shipped < scores.mean(), (shipped, float(scores.mean()))
    assert (scores.mean() - shipped) / scores.std() > 0.5, (
        "the shipped draw is not detectably favourable; the attack is weaker "
        "than filed")


# ========================================================= CALIBRATED M3 BAR ==
def test_C_the_bar_control_is_in_sample_while_every_arm_it_gates_is_held_out():
    """FIRES AS A METHOD DEFECT; the clause's margin absorbs it, and that is
    reported rather than hidden.

    `calibrate_bar` clause 5 trains the two-feature positive control on `feats`
    and scores `nrmse(net(feats), y)` on the SAME `feats`
    (`scale/negation_scope.py:1502-1513`). Every arm the resulting bar gates is
    scored on a held-out eval batch drawn at `seed + 12345`. The E4-prime
    decoder gate refuses exactly this -- *"an in-sample reading would credit
    memorisation as decoding"* (`scale/rips_gate.py:163-168`) -- and the M3 bar
    does the opposite.

    Measured on the shipped path, same init, same 150 steps, same lr:

        shipped (train and score on all 2048)   trained_two_feature 0.045561
        train on first half, score in-sample    0.037681
        train on first half, score HELD OUT     0.067680     gap +0.029998

    So the control is optimistic by 0.030 NRMSE, 80% relative.

    NO VERDICT MOVES. The clause asks the control to beat 1.0 and it beats it by
    0.93 either way, so `BAR CALIBRATED` stands. The defect is live only where
    the margin is not: at `e2_consequence` the same clause read `2.446646` before
    Saturn's repair, and an in-sample bias pushes in the direction that makes a
    broken bar look calibrated. That is the condition under which this becomes a
    verdict, and it is not today's condition.
    """
    import scale.negation_scope as NS

    cal = NS.calibrate_bar()
    ok, why = NS.bar_verdict(cal)
    assert ok and why == "BAR CALIBRATED"
    shipped = cal["trained_two_feature"]

    n, s, d, steps, lr, seed = 2048, 512, 256, 150, 0.02, 0
    x, y, f, p = NS.make_batch(n, s, d, seed=seed)
    g = torch.Generator().manual_seed(seed)
    feats = torch.stack([x[:, f, NS.CH_FLIP], x[:, p, NS.CH_PAYLOAD]], dim=-1)
    net = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.GELU(),
                              torch.nn.Linear(32, 1))
    for layer in net:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.normal_(layer.weight, 0.0, 0.5, generator=g)
            torch.nn.init.zeros_(layer.bias)
    h = n // 2
    mu = float(y[:h].mean())
    sigma = float(y[:h].std(unbiased=False)) or 1.0
    target = (y[:h] - mu) / sigma
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        ((net(feats[:h]).squeeze(-1) - target) ** 2).mean().backward()
        opt.step()
    with torch.no_grad():
        in_sample = NS.nrmse(net(feats[:h]).squeeze(-1) * sigma + mu, y[:h])
        held_out = NS.nrmse(net(feats[h:]).squeeze(-1) * sigma + mu, y[h:])

    assert held_out > in_sample, (in_sample, held_out)
    assert held_out - in_sample > 0.01, (
        f"the in-sample bias is {held_out - in_sample:.6f}; below 0.01 this "
        f"attack does not earn its keep")
    #: and the clause survives it anyway -- stated as an assertion, not prose.
    assert shipped < 1.0 and held_out < 1.0

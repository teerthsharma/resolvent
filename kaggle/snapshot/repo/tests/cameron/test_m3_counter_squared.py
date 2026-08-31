"""`counter_squared` as a runnable task in the M3 corpus, and the gap after encoding.

WHY THIS FILE EXISTS. `ceq/hankel.py` measures `counter_squared`,
``f(w) = ((#a) - (#b))**2``, on the ABSTRACT series over words of length
``0..n``: rank 3, ``rank_+ >= 4/5/5``. S2 asks for a gap task INSIDE M3, and M3
is not a word corpus. Its examples are `[n, s, d_model]` float tensors
(`scale/negation_scope.py:71`), every example the SAME length ``s``. So the
Hankel matrix the abstract measurement read is not the Hankel matrix the M3
encoding presents, and a gap on paper is worth nothing until it is re-measured
on the object that is actually run.

THE TWO OBJECTS, KEPT SEPARATE.

    abstract   rows and columns are ALL words of length ``0..n``; the block is
               square and prefix lengths vary.
    M3-encoded rows are the prefixes of length ``k``, columns the suffixes of
               length ``s-k``, for one fixed total length ``s``. One rectangular
               block per split ``k``, and the union over ``k`` is what a
               fixed-length corpus exposes.

Only the second is a statement about the task M3 runs. The two differ, and the
difference is measured here rather than assumed away: at ODD ``s`` the encoding
DESTROYS the rectangle-covering bound, because ``c_u + c_v = 0`` needs the two
count parities to agree and at odd ``s`` they never do, so the block has no zero
entry and the support is all-ones.

CONTROLS. Every control in this file is checked to FIRE, on an instance that is
DRAWN rather than hand-built wherever drawing is possible:

    - the flipper-blind label (no dependence on the letters at all) must FAIL
      the bar;
    - the PLAIN counter -- the task this one replaced, whose Hankel has
      ``rank_+ = 2`` exactly and therefore NO gap -- must FAIL the bar, because
      a bar that cannot separate ``c`` from ``c**2`` is not a bar for this task;
    - the odd-``s`` encoding must produce NO gap at any split.

THE OBSTRUCTION, PINNED AS A TEST. `negation_scope.bar_verdict`'s
flipper-dependence clause requires ``> 0.5``, a threshold calibrated for
``y = payload * sign`` where negating the flipper negates the label and the
ratio is exactly 2.0. `counter_squared` is a GLOBAL aggregate: no single letter
dominates, which is the same property that gives it the Hankel gap. Its exact
ratio is ``4 * E|S_{s-1}| / s``, which falls below 0.5 for even ``s >= 42`` and
reads 0.39738701499186757 at M3's default ``s = 64``. That is recorded here as a
measurement, not repaired by lowering the threshold: the task supplies its own
EXACT predicted value and the clause becomes two-sided, which is strictly
stronger than the one-sided threshold it replaces and is what lets the plain
counter be rejected.
"""
from __future__ import annotations

import itertools
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from ceq.hankel import (counter, counter_squared, hankel_block, rank_real,
                        rank_plus_lower)
from scale import negation_scope as NS

torch.set_num_threads(2)

#: the M3 default, from `scale/m3_capability.py` main() defaults
S_DEFAULT, D_DEFAULT = 64, 24


def _levels(k: int) -> list[str]:
    """One representative word per count level at length ``k``.

    ``counter_squared`` depends on the word only through its count, so these
    ``k+1`` words are exactly the distinct rows of the length-``k`` block. Rank,
    nonnegative rank and rectangle-covering number are all invariant under
    deleting a duplicate row (`ceq/hankel.py::dedup`), so this is the same
    measurement as enumerating all ``2**k`` words and costs ``k+1`` instead.
    """
    return ["a" * i + "b" * (k - i) for i in range(k + 1)]


def m3_block(s: int, k: int) -> np.ndarray:
    """The Hankel block the FIXED-LENGTH-``s`` encoding presents at split ``k``."""
    return hankel_block(counter_squared, _levels(k), _levels(s - k))


def _decode(x: torch.Tensor) -> list[str]:
    """The tensor back to words over ``{a, b}``: ``+1 -> 'a'``, ``-1 -> 'b'``."""
    ch = x[:, :, NS.CH_FLIP]
    return ["".join("a" if v > 0 else "b" for v in row.tolist()) for row in ch]


# ===================================================================== the task
def test_make_counter_batch_is_m3_shaped():
    """The batch is an M3 batch: same dtype, same rank-3 shape, same channels."""
    x, y, f, p = NS.make_counter_batch(64, S_DEFAULT, D_DEFAULT, seed=0)
    xr, yr, fr, pr = NS.make_batch(64, S_DEFAULT, D_DEFAULT, seed=0)
    assert x.shape == xr.shape
    assert y.shape == yr.shape
    assert x.dtype == xr.dtype and y.dtype == yr.dtype
    assert (f, p) == (fr, pr)
    # every position carries a letter, not just the one flipper position
    assert set(torch.unique(x[:, :, NS.CH_FLIP]).tolist()) == {-1.0, 1.0}
    assert torch.count_nonzero(xr[:, :, NS.CH_FLIP]) == xr.shape[0]


def test_encoded_label_equals_the_abstract_series_on_drawn_words():
    """The bind: for DRAWN examples, the M3 label equals `ceq.hankel` on the
    decoded word. Drawn, not hand-built -- a hand-built pair cannot fail."""
    x, y, _f, _p = NS.make_counter_batch(256, 24, 8, seed=7)
    words = _decode(x)
    assert len(set(words)) > 200, "the draw collapsed; this is not a draw"
    want = torch.tensor([counter_squared(w) for w in words], dtype=y.dtype)
    assert torch.equal(y, want)
    # and the counts really do span many levels, so the equality is not vacuous
    assert len({counter(w) for w in words}) >= 8


def test_oracle_signature_matches_the_shipped_one():
    """`counter_squared_oracle(x, f, p)` is drop-in for `oracle(x, f, p)`, which
    is what `calibrate_bar(oracle_fn=...)` and `m3_synthetic_settled` require."""
    x, y, f, p = NS.make_counter_batch(32, 24, 8, seed=1)
    assert torch.equal(NS.counter_squared_oracle(x, f, p), y)


# ================================================================ the exact bar
def test_flipper_dependence_closed_form_matches_the_measurement():
    """``4 * E|S_{s-1}| / s``, checked against the sampled ratio."""
    for s in (16, 32, 64):
        cal = NS.calibrate_bar(n=4096, s=s, d=s // 4, steps=1,
                               oracle_fn=NS.counter_squared_oracle,
                               batch_fn=NS.make_counter_batch)
        want = NS.counter_squared_flipper_dependence(s)
        assert abs(cal["flipper_dependence"] - want) < 0.05, (s, cal, want)


def test_counter_squared_bar_calibrates():
    x_ok, why = NS.bar_verdict(
        NS.calibrate_bar(n=2048, s=S_DEFAULT, d=D_DEFAULT,
                         oracle_fn=NS.counter_squared_oracle,
                         batch_fn=NS.make_counter_batch),
        flipper_dependence=NS.counter_squared_flipper_dependence(S_DEFAULT))
    assert x_ok, why


# ------------------------------------------------------- CONTROLS, SEEN TO FIRE
def test_bar_fires_on_a_flipper_blind_label():
    """A label with NO dependence on the letters must be REJECTED, and rejected
    BY THE FLIPPER CLAUSE.

    The blind label is read off a NOISE channel, not off CH_PAYLOAD: taking it
    from the payload makes the earlier `payload_only` clause fire first and the
    flipper clause is then never evaluated, so the control would pass while
    testing nothing about the clause it was written for. That variant is
    exercised too, below, and its own clause is checked by name.
    """
    def blind_noise(x, f, p):
        return x[:, p, NS.CH_NOISE]

    cal = NS.calibrate_bar(n=2048, s=S_DEFAULT, d=D_DEFAULT,
                           oracle_fn=blind_noise, batch_fn=NS.make_counter_batch)
    assert cal["flipper_dependence"] == 0.0
    ok, why = NS.bar_verdict(
        cal, flipper_dependence=NS.counter_squared_flipper_dependence(S_DEFAULT))
    assert not ok, "the blind label PASSED the bar -- the bar is vacuous"
    assert "flipper_dependence" in why, why

    def blind_payload(x, f, p):
        return x[:, p, NS.CH_PAYLOAD]

    cal2 = NS.calibrate_bar(n=2048, s=S_DEFAULT, d=D_DEFAULT,
                            oracle_fn=blind_payload,
                            batch_fn=NS.make_counter_batch)
    ok2, why2 = NS.bar_verdict(
        cal2, flipper_dependence=NS.counter_squared_flipper_dependence(S_DEFAULT))
    assert not ok2 and "payload_only" in why2, why2


def test_bar_fires_on_the_plain_counter():
    """The corpse. `counter` has ``rank_+ = 2`` exactly -- NO gap -- so a bar
    that accepts it is not a bar for the gap task. Its exact ratio is
    ``2 / E|S_s|`` = 0.31455481755627596 at ``s = 64`` against the
    ``counter_squared`` target 0.39738701499186757."""
    def plain(x, f, p):
        return x[:, :, NS.CH_FLIP].sum(dim=1)

    cal = NS.calibrate_bar(n=4096, s=S_DEFAULT, d=D_DEFAULT, oracle_fn=plain,
                           batch_fn=NS.make_counter_batch)
    assert abs(cal["flipper_dependence"] - 0.31455481755627596) < 0.02, cal
    ok, why = NS.bar_verdict(
        cal, flipper_dependence=NS.counter_squared_flipper_dependence(S_DEFAULT))
    assert not ok, "the PLAIN COUNTER passed the counter_squared bar"


def test_shipped_threshold_rejects_counter_squared_at_the_m3_default():
    """THE OBSTRUCTION, pinned. The shipped one-sided clause (``> 0.5``) is
    calibrated for ``y = payload * sign`` and rejects `counter_squared` at
    ``s = 64``. Recorded, not repaired by moving the threshold."""
    assert NS.counter_squared_flipper_dependence(64) == pytest.approx(
        0.39738701499186757, abs=1e-12)
    assert NS.counter_squared_flipper_dependence(40) > 0.5
    assert NS.counter_squared_flipper_dependence(42) < 0.5
    cal = NS.calibrate_bar(n=2048, s=S_DEFAULT, d=D_DEFAULT,
                           oracle_fn=NS.counter_squared_oracle,
                           batch_fn=NS.make_counter_batch)
    ok, why = NS.bar_verdict(cal)          # shipped clause, no target supplied
    assert not ok and "flipper_dependence" in why


def test_the_band_has_a_measured_minimum_sample_size():
    """The band is a gate only between two measured numbers: wider than the
    sampling spread of the correct task, narrower than the distance to the
    nearest wrong one. Both ends are re-measured here, not asserted.

    At n=256 the spread alone exceeds the 0.05 tolerance, so the floor is real
    and this test is what records it. Kept at steps=1: the trained control is
    not the subject here and paying 150 steps twenty times over is not.
    """
    exact = NS.counter_squared_flipper_dependence(S_DEFAULT)
    dev = {}
    for n in (256, 512):
        vals = [NS.calibrate_bar(n=n, s=S_DEFAULT, d=D_DEFAULT, steps=1, seed=sd,
                                 oracle_fn=NS.counter_squared_oracle,
                                 batch_fn=NS.make_counter_batch
                                 )["flipper_dependence"]
                for sd in range(12)]
        dev[n] = max(abs(v - exact) for v in vals)
    assert dev[256] > 0.05, ("n=256 no longer exceeds the tolerance; the "
                             f"documented floor is stale: {dev}")
    assert dev[512] < 0.05, dev
    # the other end: the nearest wrong task is further away than the tolerance
    assert abs(exact - 0.31455481755627596) > 0.05


# ================================================== the registration surface
def test_registry_default_path_is_the_shipped_path():
    """Routing the shipped task through `M3_TASKS` must change NOTHING.

    Every number in results/m3_capability.txt was taken through the unhooked
    call, so the hooks are only admissible if the registry entry for
    `negation_scope` reproduces them exactly.
    """
    batch_fn, oracle_fn, feature_fn, fd_fn = NS.M3_TASKS["negation_scope"]
    shipped = NS.calibrate_bar(n=256, s=S_DEFAULT, d=D_DEFAULT)
    routed = NS.calibrate_bar(n=256, s=S_DEFAULT, d=D_DEFAULT,
                              oracle_fn=oracle_fn, batch_fn=batch_fn,
                              feature_fn=feature_fn)
    assert shipped == routed
    assert fd_fn is None
    assert NS.bar_verdict(routed)[0]


def test_the_registry_bind_can_fail():
    """CONTROL on the test above: routing the WRONG builder must change the
    numbers, so the equality is a bind and not an identity."""
    wrong = NS.calibrate_bar(n=256, s=S_DEFAULT, d=D_DEFAULT,
                             batch_fn=NS.make_counter_batch)
    shipped = NS.calibrate_bar(n=256, s=S_DEFAULT, d=D_DEFAULT)
    assert wrong != shipped


def test_counter_squared_is_registered():
    batch_fn, oracle_fn, feature_fn, fd_fn = NS.M3_TASKS["counter_squared"]
    assert (batch_fn, oracle_fn, feature_fn) == (
        NS.make_counter_batch, NS.counter_squared_oracle,
        NS.counter_squared_features)
    assert fd_fn(S_DEFAULT) == NS.counter_squared_flipper_dependence(S_DEFAULT)


# ============================================== the gap AFTER the M3 encoding
@pytest.mark.parametrize("s", [8, 12, 16])
def test_even_s_encoding_keeps_the_gap_at_every_interior_split(s):
    """Rank frozen at 3 and ``rank_+ >= 4`` for every ``3 <= k <= s-3``.

    Every split is swept; none is chosen. ``k = 2`` and ``k = s-2`` are excluded
    and separately shown BELOW to have no gap, so the range is a measured
    boundary rather than a convenient one.
    """
    for k in range(3, s - 2):
        H = m3_block(s, k)
        r = rank_real(H)
        lb = rank_plus_lower(H)
        assert r.rank == 3, (s, k, r)
        assert isinstance(lb.bound, int) and lb.bound >= 4, (s, k, lb)


@pytest.mark.parametrize("s", [8, 12, 16])
def test_the_outermost_splits_have_no_gap(s):
    """CONTROL on the range above: at ``k = 2`` the row side has only 3 count
    levels, so no bound can exceed the rank and the gap is absent."""
    for k in (2, s - 2):
        H = m3_block(s, k)
        assert rank_real(H).rank == 3
        assert rank_plus_lower(H).bound == 3


@pytest.mark.parametrize("s", [9, 11, 13])
def test_odd_s_encoding_destroys_the_bound_at_every_split(s):
    """CONTROL AND FINDING. At odd ``s`` the block has NO zero entry, so the
    rectangle cover is 1 and no gap is detectable at any split."""
    for k in range(2, s - 1):
        H = m3_block(s, k)
        assert (H != 0).all(), (s, k)
        lb = rank_plus_lower(H)
        assert lb.detail["rectangle_cover"] == 1, (s, k, lb.detail)
        assert lb.bound == rank_real(H).rank == 3, (s, k, lb)


def test_gap_survives_at_the_m3_default_length():
    """``s = 64``, the length M3 actually runs, and the bound GROWS with the
    split.

    Measured on the deduplicated level block. ``rank_+`` is monotone under
    taking submatrices -- a nonnegative factorisation of the whole block
    restricts to one of any submatrix at the same inner dimension -- so a bound
    on this block is a bound on the full ``2**k x 2**(s-k)`` word block.

    ``k = 10`` reads ``rank_+ >= 6`` and is left out only because its exact ILP
    over 2046 maximal rectangles costs 35 s on this machine; the cost table in
    `ceq/hankel.py::rectangle_cover_number` is the reason the sweep stops at 9.
    """
    for k, want in ((3, 4), (4, 4), (5, 4), (6, 5), (7, 5), (8, 5), (9, 5)):
        H = m3_block(S_DEFAULT, k)
        assert rank_real(H).rank == 3, k
        assert rank_plus_lower(H).bound == want, (k, rank_plus_lower(H))


@pytest.mark.slow
def test_the_signed_arm_is_actually_signed_on_this_task_after_training():
    """The precondition the whole S2 ladder rests on, measured on THIS task.

    A Hankel gap separates ring-weighted from NONNEGATIVE-weighted automata. If
    the arm called `pivot_signed` carries a non-negative operator at the geometry
    its numbers are taken at, the ladder compares an arm against itself and no
    embedding of the task can rescue it.

    UNTRAINED, EVERY ARM IS NON-NEGATIVE. `_causal_sgate_operator` is
    ``rho * (softmax(w) - lam * softmax(-w)) / (1 + lam)`` with lam=0.10, so an
    entry is negative only where the positive softmax is more than 10x smaller
    than the negative one -- which needs logit spread. At initialisation the
    logits are near zero and the measured minimum is exactly 0.000000e+00 for
    all four arms. Non-negativity there is a property of the INITIALISATION, not
    of the construction, so it does not settle the question.

    TRAINED, IT IS NOT. Measured at s=64 d=24 steps=150 n_train=512, 2 threads:
    `pivot_signed` operator min -1.244016e-01, 7.140636% of entries negative;
    its hop-2 term min -1.285781e-01, 6.839323% negative.

    THE CONTROL FIRES: `softmax` rows are non-negative by construction and read
    exactly 0.0 on the same probe, at the same weights, after the same training.
    """
    from scale import m3_capability as M3
    from scale.pivot_probe import batched_select_pivots, batched_pivot_hop2

    s, d, steps, n_train = 64, 24, 150, 512
    xt, yt, _a, _b = NS.make_counter_batch(n_train, s, d, d_model=M3.D_MODEL,
                                           seed=0)
    xe, _ye, _c, _e = NS.make_counter_batch(512, s, d, d_model=M3.D_MODEL,
                                            seed=12345)
    got = {}
    for kind in ("pivot_signed", "softmax"):
        torch.manual_seed(0)
        m = M3.Arm(kind, s)
        with torch.no_grad():
            a0 = m._operator(m.wq(xe), m.wk(xe))
        assert float(a0.min()) == 0.0, (kind, "untrained arm is already signed")
        opt = torch.optim.Adam(m.parameters(), lr=M3.LR)
        mu = float(yt.mean())
        sg = float(yt.std(unbiased=False)) or 1.0
        ys = (yt - mu) / sg
        for _ in range(steps):
            m.train()
            opt.zero_grad()
            torch.nn.functional.mse_loss(m(xt), ys).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            q, k = m.wq(xe), m.wk(xe)
            a = m._operator(q, k)
            got[kind] = (float(a.min()), float((a < 0).float().mean()))
            if kind != "softmax":
                h2 = batched_pivot_hop2(a, batched_select_pivots(k, m.k_pivots))
                got["hop2"] = (float(h2.min()), float((h2 < 0).float().mean()))

    assert got["softmax"] == (0.0, 0.0), got            # the control, fired
    assert got["pivot_signed"][0] < -0.05, got
    assert got["pivot_signed"][1] > 0.05, got
    assert got["hop2"][0] < -0.05, got
    assert got["hop2"][1] > 0.05, got


#: MEASURED effect sizes on `counter_squared`, for the e-process repricing.
#: s=64 d=24 steps=150 n_eval=512, torch 2.5.1+cu121, torch.get_num_threads()=2,
#: cpu, seeds 0..2, arms built by `m3_capability.run_arm`, batches by
#: `make_counter_batch`. Reproduce one cell with the slow test below.
#:
#: eval NRMSE per seed, then the PAIRED delta (positive means pivot_signed wins):
MEASURED_EFFECT = {
    512: dict(
        softmax=(1.564396, 1.195500, 1.378091),
        pivot_signed=(1.161712, 1.032781, 1.201435),
        pivot_unsigned=(1.456695, 1.359469, 1.600276)),
    1024: dict(
        softmax=(1.202441, 1.162569, 1.115509),
        pivot_signed=(1.041765, 0.931749, 0.938174),
        pivot_unsigned=(1.596211, 1.253646, 1.279633)),
    1536: dict(          # FIVE seeds, not three
        softmax=(1.086648, 1.069309, 0.943833, 1.091251, 1.230873),
        pivot_signed=(0.905840, 0.939949, 0.902096, 0.848896, 0.917926),
        pivot_unsigned=(1.115066, 1.088592, 1.004362, 1.242132, 1.100629)),
    2048: dict(
        softmax=(0.939263, 1.015413, 0.967375),
        pivot_signed=(0.922362, 0.888109, 0.928749),
        pivot_unsigned=(1.042087, 1.055744, 0.985014)),
}
#: the e-process buys cheap decisions only above this (CHECKLIST.md:1108)
CHEAP_DECISION_EFFECT = 0.2


def _paired(budget, lo, hi):
    a, b = MEASURED_EFFECT[budget][lo], MEASURED_EFFECT[budget][hi]
    return sum(x - y for x, y in zip(a, b)) / len(a)


def test_the_effect_and_the_bar_clearance_peak_at_different_budgets():
    """The repricing, as a ledger with BOTH ends checked at every budget.

    The contrast the Hankel ladder predicts is pivot_signed vs pivot_unsigned:
    identical parameter count (4769), identical pivot routing, differing only in
    whether the operator carries sign. Paired over training seeds, positive
    means pivot_signed has the lower error. 95% intervals are the paired
    bootstrap of `m3_synthetic_settled.contrast`, B=10000, seed 0:

      n_train  seeds     delta   95% paired CI            pivot_signed < 1.0
          512      3  +0.340171  [+0.294983, +0.398841]   0 of 3
         1024      3  +0.405934  [+0.321897, +0.554446]   2 of 3
         1536      5  +0.207215  [+0.134441, +0.307515]   5 of 5
         2048      3  +0.114542  [+0.056265, +0.167635]   3 of 3

    A delta between two arms that are BOTH worse than predict-the-mean is not a
    capability reading -- that is the W4 death this repository already paid for
    -- so the two columns have to be read together. The effect is largest where
    no arm passes the bar and smallest where every seed does.

    THE ONE BUDGET WHERE BOTH HOLD is 1536: the point estimate clears 0.2 and
    the winning arm is under the bar at 5 of 5 seeds. Its INTERVAL does not
    clear 0.2, and that is asserted here too, so the row cannot be quoted as an
    anytime-valid pass.
    """
    for n in (512, 1024, 1536):
        assert _paired(n, "pivot_unsigned", "pivot_signed") > CHEAP_DECISION_EFFECT, n
    assert _paired(2048, "pivot_unsigned", "pivot_signed") < CHEAP_DECISION_EFFECT
    # the 1536 interval does NOT clear the threshold, only its point estimate
    assert 0.134441 < CHEAP_DECISION_EFFECT < 0.307515

    # the bar clearance that decides whether a delta is a capability reading
    below = {b: {a: sum(v < 1.0 for v in MEASURED_EFFECT[b][a]) for a in
                 ("softmax", "pivot_signed", "pivot_unsigned")}
             for b in MEASURED_EFFECT}
    assert below[512] == {"softmax": 0, "pivot_signed": 0, "pivot_unsigned": 0}
    assert below[1024]["pivot_signed"] == 2
    assert below[1536] == {"softmax": 1, "pivot_signed": 5, "pivot_unsigned": 0}
    assert below[2048]["pivot_signed"] == 3


@pytest.mark.slow
def test_one_recorded_effect_cell_re_takes():
    """The ledger above is a transcription until a cell is re-taken.

    Two cells, one from the largest-effect budget and one from the row the
    verdict rests on: n_train=1024 seed 2, and n_train=1536 seed 4.
    """
    from scale import m3_capability as M3

    s, d, steps, n_eval = 64, 24, 150, 512
    for n_train, seed in ((1024, 2), (1536, 4)):
        xt, yt, _a, _b = NS.make_counter_batch(n_train, s, d,
                                               d_model=M3.D_MODEL, seed=seed)
        xe, ye, _c, _e = NS.make_counter_batch(n_eval, s, d,
                                               d_model=M3.D_MODEL,
                                               seed=seed + 12345)
        for arm in ("pivot_signed", "pivot_unsigned"):
            r = M3.run_arm(arm, xt, yt, xe, ye, s=s, steps=steps, seed=seed)
            want = MEASURED_EFFECT[n_train][arm][seed]
            assert r["eval_nrmse"] == pytest.approx(want, abs=5e-7), (
                n_train, seed, arm, r["eval_nrmse"], want)


def test_the_gap_is_not_an_artefact_of_the_level_reduction():
    """The level block and the full ``2**k`` word block agree, at a size where
    both are computable."""
    for k in (3, 4, 5):
        rows = ["".join(t) for t in itertools.product("ab", repeat=k)]
        cols = ["".join(t) for t in itertools.product("ab", repeat=12 - k)]
        full = hankel_block(counter_squared, rows, cols)
        lvl = m3_block(12, k)
        assert rank_real(full).rank == rank_real(lvl).rank
        assert rank_plus_lower(full).bound == rank_plus_lower(lvl).bound

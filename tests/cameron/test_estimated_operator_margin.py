"""tests/cameron/test_estimated_operator_margin.py -- round two: binds the
one CPU-reachable arm whose read is a MODEL (transition-counting RLS + the
existing exact solve), scored against REALIZED absorption labels rather than
against the oracle that every earlier margin in this repo was measured on.

THIS DOES NOT TICK THE PREDICTION ROW. The arm here is the empirical MLE
chain (`mle_gap` in ceqjepa/lstd_bed.py is 0.0 exactly) -- counting, the
floor for the row, not the 9,505-param stack. See ceqjepa/lstd_bed.py's
module docstring for the honest sentence; docs/index.md:483 is a docs-lane
edit, not this file's.

Four pre-registered assertions, all at T=10000, n=20000 held-out rollouts,
seeds [20260914, 1, 2, 3, 4], n_boot=200 (sharpness.bootstrap_se's default,
used unchanged -- see `score` in ceqjepa/lstd_bed.py):

  BAR      the RLS arm clears 2 sigma_total (item SE and across-stream SD
           combined in quadrature -- see NEGATIVE) on >=4/5 seeds, with
           capture ratio (margin_RLS / margin_oracle) >= 0.5 on those seeds.
  KILL     the Hebbian (unwhitened, beta=0) arm fires kill_fired on >=4/5.
  NEGATIVE a shuffled-stream arm (dst permuted -- zero true information) is
           within 2 sigma_total of zero, THE REGRESSION GUARD: the same arm
           must still trip the item-bootstrap bar/kill on at least one seed,
           recording that the item bootstrap alone is not a safe null test
           for a FIT arm (it resamples items, not the estimator).
  TRAP     accuracy cannot adjudicate: on seeds where oracle and RLS argmax
           accuracy AGREE to float equality, margin_sigma still separates
           the killed Hebbian arm from both by >10 sigma; and at T=1000,
           seed=4, the RLS arm is killed while its accuracy sits between
           chance and the oracle's, not below it in any way a ranking metric
           would flag.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

from ceqjepa import sharpness
from ceqjepa.lstd_bed import (
    D_STATES,
    T_DEFAULT,
    build_chain,
    build_embedding,
    decode,
    free_params,
    hebbian,
    is_capacity_opponent,
    rank_r_committor,
    rls_committor,
    rollout_labels,
    score,
    sigma_total,
    simulate,
    solve_committor,
    stream_se,
)

SEEDS = [20260914, 1, 2, 3, 4]
N_HOLDOUT = 20_000
PRIMARY = SEEDS[0]


@functools.lru_cache(maxsize=None)
def _arms(seed, T=T_DEFAULT):
    """oracle / RLS / Hebbian / shuffled committors at one seed, each scored
    (via `score`, i.e. through sharpness.py's identity) against the SAME
    held-out rollout of the TRUE chain. Cached: every test below asks for a
    subset of the same five seeds, and a fit is not free."""
    bed = build_chain(seed)
    E = build_embedding(cond=10.0, seed=seed + 1)
    start, y = rollout_labels(bed["P"], bed["transient"], bed["B"], N_HOLDOUT, seed=seed + 500)
    q_star, _ = solve_committor(bed["P"], bed["A"], bed["B"])
    q_D = rls_committor(bed, E, T, seed=seed + 2, shuffle=False)
    q_shuf = rls_committor(bed, E, T, seed=seed + 2, shuffle=True)
    src, dst = simulate(bed["P"], T, bed["transient"], bed["absorbing"], seed=seed + 2)
    P_H, _ = decode(hebbian(E[:, src].T, E[:, dst].T), E)
    q_H, _ = solve_committor(P_H, bed["A"], bed["B"])
    # q_star is the TRUE P's committor, given not fit -- ASSAY C1's own
    # construction-property class -- so it is tagged accordingly (item 4).
    return dict(star=score(q_star, start, y, construction_property=True),
                D=score(q_D, start, y),
                H=score(q_H, start, y), shuf=score(q_shuf, start, y))


@pytest.fixture(scope="module")
def bundle():
    """Everything the four tests need, computed once. PINNED across every
    arm and every seed: the chain, the embedding, the held-out rollout (see
    `_arms`); VARIED: only which estimator reads the T=10000 stream."""
    data = {seed: _arms(seed) for seed in SEEDS}
    # n_streams=6 here (stream_se's own default is 12, used below for the
    # NEGATIVE control, where a tight null needs it): the RLS margins are
    # 8-24 item-sigma, so a coarser stream SD still resolves the >2
    # sigma_total pass/fail with the 15s budget this file is held to.
    sd_D = {seed: stream_se(seed, n_streams=6, shuffle=False) for seed in SEEDS}
    sd_shuf = stream_se(PRIMARY, shuffle=True)
    trap_t1000 = _arms(4, T=1000)
    return dict(arms=data, sd_D=sd_D, sd_shuf=sd_shuf, trap_t1000=trap_t1000)


def test_bar_rls_clears_sigma_total_with_capture_at_least_half(bundle):
    passed, captures = [], []
    for seed in SEEDS:
        a = bundle["arms"][seed]
        st = sigma_total(a["D"]["margin"], a["D"]["se_margin"], bundle["sd_D"][seed])
        if st > sharpness.BAR_SIGMA:
            passed.append(seed)
            captures.append(a["D"]["margin"] / a["star"]["margin"])
    assert len(passed) >= 4, f"RLS cleared 2 sigma_total on {len(passed)}/5 seeds: {passed}"
    assert captures and min(captures) >= 0.5, f"capture ratios on passing seeds: {captures}"


def test_kill_fires_on_the_unwhitened_hebbian_arm(bundle):
    kills = [seed for seed in SEEDS if bundle["arms"][seed]["H"]["kill_fired"]]
    assert len(kills) >= 4, (
        f"Hebbian arm killed on only {len(kills)}/5 seeds: "
        f"{[(s, round(bundle['arms'][s]['H']['margin_sigma'], 1)) for s in SEEDS]}"
    )


def test_shuffled_stream_is_null_under_sigma_total_not_item_se(bundle):
    a = bundle["arms"][PRIMARY]["shuf"]
    st = sigma_total(a["margin"], a["se_margin"], bundle["sd_shuf"])
    assert abs(st) <= 2.0, f"zero-information arm should sit within 2 sigma_total of zero, got {st:+.1f}"

    # THE REGRESSION GUARD. Every other seed's shuffled arm, graded by the
    # ITEM bootstrap alone (bootstrap_se resamples items at one fixed q-hat --
    # blind to the noise of FITTING that q-hat, which is the whole story for
    # a null FIT arm): at least one must fire bar or kill despite being true
    # zero. If this assertion ever fails, someone reverted the bar to the
    # item-SE rule for a FIT arm and the null stopped misbehaving -- which
    # means the fix that made it correct was undone, not that it improved.
    item_only = [bundle["arms"][s]["shuf"] for s in SEEDS]
    misfires = sum(d["passes_bar"] or d["kill_fired"] for d in item_only)
    assert misfires >= 1, (
        f"expected the item-SE rule to misfire on a zero-information arm on at least "
        f"1/5 seeds (it is not the estimator-aware SE); sigmas were "
        f"{[round(d['margin_sigma'], 1) for d in item_only]}"
    )


def test_accuracy_cannot_adjudicate_model_vs_oracle_or_killed_vs_alive(bundle):
    # Seeds where RLS and oracle argmax accuracy agree to float equality: a
    # ranking metric that cannot separate a fit model from the operator it
    # was fit to approximate is not adjudicating this comparison.
    tied = [s for s in SEEDS if bundle["arms"][s]["star"]["accuracy"] == bundle["arms"][s]["D"]["accuracy"]]
    assert tied, "expected at least one seed where oracle and RLS accuracy agree exactly"
    for s in tied:
        a = bundle["arms"][s]
        gap_star = a["star"]["margin_sigma"] - a["H"]["margin_sigma"]
        gap_D = a["D"]["margin_sigma"] - a["H"]["margin_sigma"]
        assert gap_star > 10 and gap_D > 10, (
            f"seed {s}: accuracy ties oracle/RLS/Hebbian but margin_sigma should still "
            f"separate Hebbian from both by >10 sigma (got {gap_star:.1f}, {gap_D:.1f})"
        )

    # T=1000, seed=4: the RLS read is killed although its accuracy sits
    # BETWEEN chance (0.5) and the oracle's, not visibly worse than the
    # unwhitened arm's -- accuracy alone gives no reason to distrust it.
    t = bundle["trap_t1000"]
    assert t["D"]["kill_fired"], f"RLS at T=1000 seed=4 should be killed, margin_sigma={t['D']['margin_sigma']:.1f}"
    assert 0.5 <= t["D"]["accuracy"] <= t["star"]["accuracy"], (
        f"RLS accuracy {t['D']['accuracy']:.4f} should sit between chance and the oracle's "
        f"{t['star']['accuracy']:.4f}"
    )
    assert abs(t["D"]["accuracy"] - t["H"]["accuracy"]) < 0.03, (
        f"RLS accuracy {t['D']['accuracy']:.4f} should read as a tie with the killed "
        f"Hebbian arm's {t['H']['accuracy']:.4f} -- that tie is the trap"
    )


# --------------------------------------------------------------------------- #
# ROUND TWO -- C4, replacement route. C4-as-written (mle_gap > 0 as the
# opponent test) is struck: ceqjepa/lstd_bed.py:526 already asserts mle_gap
# < 1e-6 for the RLS arm, correctly -- mle_gap measures whether an estimator
# IS the empirical MLE, not whether it has estimation error against the true
# operator (that number is op_err_D, reported separately). The replacement
# condition is CAPACITY: an arm is an opponent only if it CANNOT represent
# the MLE (strictly fewer free parameters than D_STATES**2 = 256), scored
# against realized absorption labels rather than the oracle committor.

RANK_ADMITTED = 4  # free_params(4) = 4*(2*16-4) = 112 < 256


def test_capacity_condition_must_fire_refuses_full_rank_admits_rank_r():
    """MUST-FIRE (item 2): the existing transition-counting RLS arm -- rank
    == D_STATES == 16, free_params == D_STATES**2 == 256, mle_gap == 0.0 per
    ceqjepa/lstd_bed.py:526 -- is REFUSED: it CAN represent the empirical
    MLE, so mle_gap == 0.0 does not make it an opponent under this
    condition. A rank-4 arm, with strictly fewer free parameters, is
    ADMITTED. Both sides are asserted: a condition admitting everything
    currently in the file (i.e. always True) would be vacuous, not met."""
    full, r = free_params(D_STATES), free_params(RANK_ADMITTED)
    assert full == D_STATES ** 2 == 256, f"full-rank free_params should be 256, got {full}"
    assert r == 112 < 256, f"rank-{RANK_ADMITTED} free_params should be 112, got {r}"
    assert not is_capacity_opponent(D_STATES), (
        "the full-rank (256-param) transition-counting RLS arm must be REFUSED: "
        "it can represent the empirical MLE"
    )
    assert is_capacity_opponent(RANK_ADMITTED), (
        f"the rank-{RANK_ADMITTED} (112-param) arm must be ADMITTED: it cannot represent the MLE"
    )


def test_capacity_admitted_arm_prints_free_params_and_op_err_d_float64(bundle):
    """Item 1: both numbers C4 demands from an admitted opponent, at their
    dtype (free_params is int, op_err_D is float64 on this bed), scored
    against REALIZED labels via `score` -- never against the oracle
    committor."""
    for seed in SEEDS:
        bed = build_chain(seed)
        E = build_embedding(cond=10.0, seed=seed + 1)
        start, y = rollout_labels(bed["P"], bed["transient"], bed["B"], N_HOLDOUT, seed=seed + 500)
        q_r, P_r = rank_r_committor(bed, E, T_DEFAULT, seed=seed + 2, rank=RANK_ADMITTED)
        assert P_r.dtype == np.float64
        op_err_D = float(np.abs(P_r - bed["P"])[bed["transient"]].max())
        assert isinstance(op_err_D, float) and op_err_D >= 0.0
        d_r = score(q_r, start, y)
        assert isinstance(d_r["margin"], float)
        assert isinstance(free_params(RANK_ADMITTED), int)


def test_construction_property_guard_fires_on_oracle_not_on_fit_arm(bundle):
    """Item 4 (carries C1's live gap forward): lstd_bed.py:335-336 (inside
    `score`) is where a bar/kill verdict is actually read off a bed. Tagging
    the oracle read (q* from the TRUE P, given not fit -- ASSAY C1's own
    definition, see lstd_bed.py's module docstring for section (10)) as
    construction_property=True must trip ConstructionPropertyAsBarError --
    caught inside `score`, reported as `guard_fired`, never left to crash
    the caller. A genuine fit arm (RLS) must NOT trip it."""
    a = bundle["arms"][PRIMARY]
    d_star = a["star"]
    assert d_star["guard_fired"] is True, "the oracle read is construction-property and must trip the guard"
    assert d_star["passes_bar"] is None and d_star["kill_fired"] is None

    d_D = a["D"]
    assert d_D["guard_fired"] is False, "a genuine fit arm must not be refused by the construction-property guard"
    assert isinstance(d_D["passes_bar"], bool) and isinstance(d_D["kill_fired"], bool)

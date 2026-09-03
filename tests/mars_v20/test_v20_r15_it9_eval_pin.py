"""MARS it.9 -- attack on the eval-batch seed pin.

`scripts/v15_r1.py:699-700` draws the ENTIRE eval set once, at a hardcoded
`seed=12345`, before the per-seed training loop starts (`:696-710`). Every
`eval_nrmse` the it.8 file's crossing count is built from -- `7` of the eight
fresh `arm_pl` seeds, `12` of the sixteen pooled, `0` of eight `softmax`
(`V20_R15_IT8_MERCURY.md` section 1.3, `tests/mercury/test_v20_r15_it8_armpl_and_clamp.py`)
-- is read off ONE `x_ev, y_ev` tensor pair. This file asks whether that one
draw is idiosyncratic on the statistics `eval_nrmse` (and the columns beside
it) are built from, using the REAL `batch_fn` the runner calls -- never a
reimplementation.

`batch_fn` for `T_STAR=2` is `M3_TASKS["e3_t2"][0]`
(`functools.partial(scale.negation_scope.make_equilibrium_batch, t_star=2)`),
the exact object `scripts/v15_r1.py:678,684,699,708` binds and calls. Shape
args mirror the runner's own constants: `S, D = 64, 24` and `T_STAR = 2`
(`scripts/v15_r1.py:137-138`), `D_MODEL = 16` (`scale/m3_capability.py:79`),
`n_eval` default `4096` (`scripts/v15_r1.py:549`).
"""
from __future__ import annotations

import pathlib
import statistics
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scale.negation_scope import M3_TASKS, CH_DRIVE, nrmse  # noqa: E402
from scale.m3_capability import D_MODEL                      # noqa: E402

S, D, T_STAR, N_EVAL = 64, 24, 2, 4096            # scripts/v15_r1.py:137-138,549
PINNED_SEED = 12345                                # scripts/v15_r1.py:699
OTHER_SEEDS = list(range(1, 41))                   # >= 40 independent draws
HEAD = S - 1 - T_STAR                              # scripts/v15_r1.py:697
LIVE = list(range(HEAD + 1, S))                    # scripts/v15_r1.py:698
FLOOR_1 = (T_STAR - 1) / T_STAR                     # squared; sqrt below, v15_r1.py:586
FLOOR_1 = FLOOR_1 ** 0.5

batch_fn, oracle_fn, feature_fn, _fd_fn = M3_TASKS[f"e3_t{T_STAR}"]


def _draw_stats(seed: int) -> dict:
    """One eval-shaped draw through the REAL `batch_fn`, and the statistics
    that enter a cell's score before any model touches the draw:

      std_y   -- `nrmse`'s own denominator (`scale/negation_scope.py:1384`),
                 the number every `eval_nrmse` in every pooled row divides by.
      mean_a  -- sign balance of the live-band drive channel `a_ev =
                 x_ev[:, live, CH_DRIVE]` (`scripts/v15_r1.py:701`), what
                 `sign_acc`/`gate_r2`/`c=2p-1` are read against out of sample
                 (`scripts/v15_r1.py:774-782`).
      frac_neg -- fraction of live-band positions reading `a_i = -1`: the
                 exact predicate `fires()`/the kill diagnostic turns on
                 (`scripts/v15_r1.py:433-440`), and the census a gate that
                 annihilates on a negative recovery would be read off.
    """
    x_ev, y_ev, f, p = batch_fn(N_EVAL, S, D, d_model=D_MODEL, seed=seed, device=None)
    a_ev = x_ev[:, LIVE, CH_DRIVE].reshape(-1)
    mean_pred_nrmse = nrmse(torch.full_like(y_ev, float(y_ev.mean())), y_ev)
    return dict(
        seed=seed,
        std_y=float(y_ev.std(unbiased=False)),
        mean_pred_nrmse=mean_pred_nrmse,
        mean_a=float(a_ev.mean()),
        frac_neg=float((a_ev < 0).double().mean()),
        sst_a=float(((a_ev - a_ev.mean()) ** 2).sum()),
    )


def _quantile_rank(value: float, population: list) -> float:
    """Fraction of `population` <= value. An empirical-CDF read, not a
    parametric z-score -- n=41 is too small to assume normality of the
    sampling distribution of a sample std or a sample mean."""
    return sum(1 for v in population if v <= value) / len(population)


@pytest.fixture(scope="module")
def draws():
    seeds = [PINNED_SEED] + OTHER_SEEDS
    return {s: _draw_stats(s) for s in seeds}


def test_predict_the_mean_is_still_exactly_one(draws):
    """Sanity gate on the real `nrmse`: predicting the draw's own mean must
    read exactly 1.0 by construction (`scale/negation_scope.py:1382-1387`),
    on every draw including the pinned one. If this fails the statistics below
    are not reporting what this file claims they report."""
    bad = {s: d["mean_pred_nrmse"] for s, d in draws.items()
           if abs(d["mean_pred_nrmse"] - 1.0) > 1e-9}
    assert not bad, f"predict-the-mean NRMSE != 1.0 on seeds: {bad}"


def test_pinned_seed_is_not_an_outlier_on_the_nrmse_denominator(draws):
    """The property the round's reading of the it.8 crossing count NEEDS:
    seed 12345's `std(y_ev)` -- the single number every pooled `eval_nrmse`
    is divided by, since every arm and every training seed is scored against
    the SAME eval tensor (`scripts/v15_r1.py:699-700` runs once, outside the
    `for kind / for seed` loop at `:706-707`) -- is not off in the tail of
    what other draws at the identical shape would give. If it is off in the
    tail, the crossing count is partly a property of this one draw's
    normaliser and not purely of the arm being scored.
    """
    pop = [d["std_y"] for d in draws.values()]
    pinned = draws[PINNED_SEED]["std_y"]
    others = [d["std_y"] for s, d in draws.items() if s != PINNED_SEED]
    q = _quantile_rank(pinned, pop)
    print(f"\nstd(y_ev) [nrmse denominator]: seed={PINNED_SEED} value={pinned:.6f}  "
          f"theoretical sqrt(T_STAR)={T_STAR ** 0.5:.6f}  "
          f"other-draws(n={len(others)}) mean={statistics.fmean(others):.6f} "
          f"sd={statistics.stdev(others):.6f} "
          f"min={min(others):.6f} max={max(others):.6f}  quantile={q:.3f}")
    assert 0.025 <= q <= 0.975, (
        f"seed {PINNED_SEED}'s eval std(y) sits at quantile {q:.3f} of the "
        f"{len(pop)}-draw population -- an outlier on the exact number "
        f"every pooled crossing in results/v20_r15_it8_armpl_b.jsonl "
        f"is normalised by")


def test_pinned_seed_drive_channel_sign_balance_is_not_an_outlier(draws):
    """The live-band `a_ev` sign balance at 12345 is not off in the tail of
    what other draws give. A skewed balance here would move `sign_acc` and
    `gate_r2`'s out-of-sample floor for BOTH arms before either trains a
    step, which is a different mechanism from an arm-specific advantage."""
    pop = [d["mean_a"] for d in draws.values()]
    pinned = draws[PINNED_SEED]["mean_a"]
    others = [d["mean_a"] for s, d in draws.items() if s != PINNED_SEED]
    q = _quantile_rank(pinned, pop)
    print(f"mean(a_ev) [sign balance]: seed={PINNED_SEED} value={pinned:+.6f}  "
          f"other-draws(n={len(others)}) mean={statistics.fmean(others):+.6f} "
          f"sd={statistics.stdev(others):.6f} "
          f"min={min(others):+.6f} max={max(others):+.6f}  quantile={q:.3f}  "
          f"frac_neg(pinned)={draws[PINNED_SEED]['frac_neg']:.4f}")
    assert 0.025 <= q <= 0.975, (
        f"seed {PINNED_SEED}'s live-band sign balance sits at quantile "
        f"{q:.3f} of the {len(pop)}-draw population")


def test_score_scale_is_common_mode_across_arms(draws):
    """DERIVED report, not a bound on the draw: `crosses` compares
    `mean(eval_nrmse) + t*sd/sqrt(N)` against the FIXED constant `floor_1 =
    sqrt((T_STAR-1)/T_STAR)` (`scripts/v15_r1.py:586,899-909`), a threshold
    that does not rescale with this draw's realised `std(y_ev)`. Every row in
    the pooled sixteen -- both arms, every training seed -- divides by the
    SAME `std(y_ev)` from the SAME tensor, so whatever this draw's normaliser
    is, it enters every row identically. A scale shift there is common-mode:
    it can make crossing `floor_1` uniformly easier or harder for every row
    in the file, but it cannot by itself manufacture a DIFFERENTIAL between
    `arm_pl`'s 12/16 and `softmax`'s 0/8, because softmax's rows divide by
    the identical number.
    """
    d = draws[PINNED_SEED]
    theoretical_sd = T_STAR ** 0.5
    rel_shift = d["std_y"] / theoretical_sd - 1.0
    print(f"\nseed {PINNED_SEED}: std(y_ev)={d['std_y']:.6f} vs theoretical "
          f"sqrt(T_STAR)={theoretical_sd:.6f}  relative shift={rel_shift:+.4%}  "
          f"floor_1={FLOOR_1:.10f}")
    assert d["std_y"] > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))

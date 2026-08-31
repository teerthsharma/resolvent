"""X18 — the PER-DRAW Ville e-process. The unit is the evaluation draw.

    python -m scale.eprocess_perdraw --task e3_t1
    python -m scale.eprocess_perdraw --ceiling-only        # arithmetic, no data

WHAT WAS WRONG WITH THE SEED-CAPPED PROCESS, AND IT WAS NOT THE DATA.
`scale/eprocess.py` takes the TRAINING SEED as the unit. Every factor of the
mixture is `1 + lam*d/B` with `lam <= 1/2` and `|d| <= B`, so every factor is at
most `1.5`, and the mixture ceiling after `t` seeds is
`mean_lam (1+lam)**t`. At `t = 5` that is `3.80169140625` against a threshold of
`40.0`, and `MIN_T_MIXTURE = 13`. **Five seeds could not cross in either
direction whatever the data said.** That was provable before the first seed and
it is a property of the SCHEDULE, not of the effect.

WHAT THE PER-DRAW PROCESS BUYS, AND WHAT IT COSTS. There are `n_eval` evaluation
draws per cell instead of 5 seeds, so `t` goes from 5 to `n_eval` (2048 here,
10240 pooled over five seeds) and the ceiling stops binding at `t = 13`. Ville's
inequality is untouched: it needs a nonnegative supermartingale, not a
particular unit.

**THE COST IS THE ESTIMAND AND IT IS NOT SMALL.** The per-seed process asks
*"does settling win, integrating over training randomness"*. The per-draw process
conditions on the trained weights and asks *"AT THESE WEIGHTS, on a fresh draw,
is settled's clipped error lower than twin's"*. Those are different questions.
Pooling the draws across all five seeds in a fixed seed order restores part of
the training-randomness integration -- the process then sees all five weight
sets -- but the increments within one seed share a weight set and are NOT
independent of it. This is stated here, in the file, because the standing
prohibition in `LOOP_PROMPT.md` §5 is against instruments that measure the thing
next to the thing that matters, and swapping the unit under a threshold is
exactly how that happens quietly.

THE OUTCOME, AND WHY IT IS CLIPPED THE SAME WAY. `eprocess.paired_difference`
clips each ARM before subtracting, at `CLIP_C = 2.0`, in NRMSE units. The
per-draw analogue is the per-draw error in the SAME units:

    e_j(arm) = |pred_j(arm) - y_j| / std(y)          (std over the eval batch)
    d_j      = min(e_j(ref), C) - min(e_j(arm), C)

so `|d_j| <= B = 2.0` holds by construction with the SAME `B` the seed process
declared, and no constant is refitted. Positive `d_j` means `arm` (settled) has
the lower error on that draw. The clip is per arm and inside the subtraction,
for the reason `eprocess.paired_difference` gives: clipping the difference would
make the estimand depend on which differences turned up.

DRAW ORDER IS FIXED BEFORE THE DATA. Draws are folded in batch-index order and
seeds in ascending order. Both orders are determined by the seed passed to the
builder, not by any outcome, so `lambda` stays predictable and the
supermartingale property survives.

THE WEIGHTS ARE THE PRECONDITION. This process cannot exist without per-cell
trained weights, which `scale/m3_quintuple.py` did not save until this round
(`scale/capability_table.py:232` records the consequence). It loads them through
`m3_quintuple.load_unit`, never by rebuilding the arm here.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
torch.set_num_threads(2)

from scale import eprocess as EP                                   # noqa: E402
from scale import negation_scope as NS                             # noqa: E402
from scale.m3_quintuple import _key, load_unit, weights_path       # noqa: E402


def log10_max_attainable(t: int) -> float:
    """`log10` of `eprocess.max_attainable(t)`, which OVERFLOWS at this `t`.

    `eprocess.max_attainable` evaluates `(1+lam)**t` directly. At `t = 5` that
    is 3.8; at `t = 2048` it is `1.5**2048`, and Python raises
    `OverflowError: (34, 'Result too large')` because the value exceeds the
    double range at about `t = 1748`. THAT OVERFLOW IS THE FINDING, not a bug to
    route around: the seed-capped ceiling was 3.80169140625 against a threshold
    of 40.0, and the per-draw ceiling is past the largest number a float can
    hold. It is reported in log space rather than by lowering `t`, and
    `eprocess.max_attainable` is NOT modified -- other readers bind its value.
    """
    ln = EP._logsumexp([t * math.log1p(lam) for lam in EP.LAMBDA_GRID])
    return (ln - math.log(len(EP.LAMBDA_GRID))) / math.log(10.0)


LOG10_THRESHOLD = math.log10(EP.THRESHOLD)


def clipped_errors(rec, model) -> torch.Tensor:
    """`min(|pred - y| / std(y), CLIP_C)` per draw. Returns `(errors, y)`.

    THE DIVISOR IS THE SAME AS `negation_scope.nrmse`'s AND THE AGGREGATION IS
    NOT, and an earlier draft of this docstring said otherwise. `nrmse` is
    `sqrt(mean((pred-y)**2)) / std(y)` -- a ROOT-MEAN-SQUARE. The per-draw
    quantity here is `|pred-y| / std(y)`, and the mean of that is a mean
    ABSOLUTE error; the two do not agree and claiming they do would be a false
    sentence in a file. What IS exactly shared is `std(y)` with
    `unbiased=False`, so the units are the journalled units. The checkable
    consequence is that the ROOT MEAN SQUARE of the UNCLIPPED per-draw errors
    reproduces the journalled `eval_nrmse` bitwise, and
    `test_per_draw_errors_reproduce_the_journalled_nrmse` asserts exactly that
    rather than this paragraph.
    """
    bfn = NS.M3_TASKS[rec["task"]][0]
    xe, ye, _f, _p = bfn(rec["n_eval"], rec["s"], rec["d"],
                         d_model=rec["d_model"], seed=rec["seed"] + 12345)
    with torch.no_grad():
        pred = model(xe) * rec["sigma"] + rec["mu"]
    sd = float(ye.std(unbiased=False))
    if sd == 0.0:
        raise ValueError("std(y) == 0 on the eval batch; the metric is undefined")
    return torch.clamp((pred - ye).abs() / sd, max=EP.CLIP_C), ye


def per_draw_differences(task: str, seed: int, cfg: dict, *, ref: str = "twin",
                         arm: str = "settled") -> list[float]:
    """`d_j` for one (task, seed), in batch-index order. Positive: `arm` wins."""
    out = {}
    for cell in (ref, arm):
        p = dict(cfg, cell=cell, seed=seed, task=task)
        path = weights_path(_key(p))
        if not path.exists():
            raise FileNotFoundError(path)
        model, rec = load_unit(path)
        out[cell] = clipped_errors(rec, model)[0]
    if out[ref].shape != out[arm].shape:
        raise ValueError("the two cells were scored on different batch sizes")
    return (out[ref] - out[arm]).tolist()


def run(task: str, seeds, cfg: dict, *, ref="twin", arm="settled") -> dict:
    """The process over every available seed's draws, pooled in seed order."""
    pair = EP.Pair()
    used, missing, void = [], [], None
    for sd in sorted(seeds):
        try:
            ds = per_draw_differences(task, sd, cfg, ref=ref, arm=arm)
        except FileNotFoundError as exc:
            missing.append((sd, str(exc)))
            continue
        used.append(sd)
        for d in ds:
            try:
                pair.update(d)
            except ValueError as exc:
                void = f"VOID at seed {sd}: {exc}"
                break
        if void:
            break
    dec = pair.decision
    return dict(task=task, t=pair.settled.t, seeds_used=used, missing=missing,
                e_arm=pair.settled.value, e_ref=pair.twin.value,
                peak_arm=pair.settled.peak, peak_ref=pair.twin.peak,
                # `peak_*` saturate to inf past the double range -- this process
                # exists to push `t` into the thousands, which is exactly where
                # that happens. The number survives here, beside log10_ceiling.
                log10_peak_arm=pair.settled.log_peak / math.log(10.0),
                log10_peak_ref=pair.twin.log_peak / math.log(10.0),
                decision=dec, void=void,
                log10_ceiling=log10_max_attainable(pair.settled.t),
                can_decide=(log10_max_attainable(pair.settled.t)
                            >= LOG10_THRESHOLD))


def print_ceiling(n_eval: int, n_seeds: int) -> None:
    """THE CEILING ARITHMETIC, PRINTED BEFORE THE RUN. LOOP_PROMPT.md 1.3:
    a process that cannot cross must say so in the same breath as its first
    number -- so it says so BEFORE its first number instead."""
    print("=== CEILING ARITHMETIC, BEFORE ANY DATUM (LOOP_PROMPT.md 1.3) ===")
    print(f"  every mixture factor is 1 + lam*d/B with lam <= {max(EP.LAMBDA_GRID)} "
          f"and |d| <= B = {EP.B}, so every factor is <= "
          f"{1.0 + max(EP.LAMBDA_GRID)}")
    print(f"  mixture ceiling after t steps: mean over the {len(EP.LAMBDA_GRID)}"
          f"-point grid of (1+lam)**t")
    print(f"  THRESHOLD = 1/ALPHA = {EP.THRESHOLD}  "
          f"(ALPHA_FAMILY {EP.ALPHA_FAMILY} / N_DIRECTIONS "
          f"{EP.N_DIRECTIONS} = {EP.ALPHA})")
    print(f"  MIN_T_MIXTURE = {EP.MIN_T_MIXTURE}   "
          f"MIN_T_SINGLE_ARM = {EP.MIN_T_SINGLE_ARM}")
    print(f"  THE OLD UNIT was the training seed: t = 5, ceiling "
          f"{EP.max_attainable(5)!r} < {EP.THRESHOLD} "
          f"-- COULD NOT CROSS, whatever the data said.")
    for t in (n_eval, n_eval * n_seeds):
        lg = log10_max_attainable(t)
        ok = lg >= LOG10_THRESHOLD
        print(f"  THE NEW UNIT is the evaluation draw: t = {t}, ceiling "
              f"10**{lg:.4f} {'>=' if ok else '<'} {EP.THRESHOLD} "
              f"(10**{LOG10_THRESHOLD:.4f})  -- "
              f"{'CAN cross' if ok else 'CANNOT cross'}")
    print("  (the ceiling is reported in log space because it overflows a "
          "double past t = 1748; see log10_max_attainable)")
    print("  The ceiling is not a prediction that it WILL cross. It is the "
          "statement that the schedule no longer forbids it.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="e3_t1")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-train", type=int, default=2048)
    ap.add_argument("--n-eval", type=int, default=2048)
    ap.add_argument("--ceiling-only", action="store_true")
    a = ap.parse_args()

    print_ceiling(a.n_eval, len(a.seeds))
    if a.ceiling_only:
        return 0

    cfg = dict(k=8, s=64, d=24, steps=150, n_train=a.n_train,
               n_eval=a.n_eval, t_max=21, n_neumann=21)
    r = run(a.task, a.seeds, cfg)
    print(f"\n=== PER-DRAW PROCESS  task={a.task} ===")
    print("  ESTIMAND WARNING: this conditions on the trained weights. It is "
          "NOT the seed process's question.")
    print(f"  seeds used {r['seeds_used']}   t = {r['t']} draws")
    for sd, why in r["missing"]:
        print(f"  seed {sd}: NO WEIGHTS -- {why}")
    if r["void"]:
        print(f"  {r['void']}")
        return 1
    print(f"  E_t settled-direction {r['e_arm']!r}  peak {r['peak_arm']!r}")
    print(f"  E_t twin-direction    {r['e_ref']!r}  peak {r['peak_ref']!r}")
    print(f"  ceiling at this t     10**{r['log10_ceiling']:.4f}   "
          f"can_decide={r['can_decide']}")
    print(f"  DECISION: {r['decision'] or 'UNDECIDED'} at threshold "
          f"{EP.THRESHOLD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

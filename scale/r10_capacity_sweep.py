"""R10 P1a -- can SOFTMAX learn the chain family at all, at t* above its hop budget?

WHY IT IS ITS OWN FILE AND NOT A LOOP OVER `m3_capability.main()`. That entry
point retrains from scratch for every `--steps` value, so the four-rung steps
ladder {150,600,2400,9600} would cost 12,750 steps per (n, t*). The ladder is
NESTED: a 9600-step run passes through 150, 600 and 2400 on its way. Evaluating
at those step counts instead of retraining costs 9600 steps for the same four
readings -- 25% off the whole grid, and the four readings are now four points on
ONE optimisation trajectory rather than four independent draws, which is the
comparison the steps axis is actually asking for.

WHAT IS NOT SLICEABLE, and it is the expensive half. `t*` is NOT an eval-time
slice. `negation_scope.M3_TASKS` registers `e3_t{1,2,8,32}` as
`functools.partial(make_equilibrium_batch, t_star=t)`, and that builder zeroes
the sub-diagonal at `head = s - 1 - t*`, so each t* is a DIFFERENT corpus with a
different label law -- `N(0, t*)` exactly -- and a different calibration band
(`chain_flipper_dependence = 2/sqrt(t*)`), which follows from that law.

This line read `N(0, t*+1)` beside a bar of `2/sqrt(t*)` until R10 it.16, and the
two are inconsistent with each other -- the bar IS the law, via
`2 E|N(0,1)| / E|N(0, Var)|`. The bar was right. The operator's nilpotency index
is `t*+1`, and the variance was read off that index, but `b[s-1] = 0` kills the
`m=0` term so the label sums `t*` drivers. Measured at it.13, K=200 draws per
block: mean Var(y) = 2.002719 / 7.984942 / 32.069818, excluding `t*+1` by 9.8
half-widths at `t*=32`. No cell in this file's journals moves either way -- every
executable line already took `t*`. `run_arm` also standardises by the
TRAIN batch's own mu/sigma, so a model trained at one t* and evaluated at another
is reading a mis-scaled target. Training once and slicing t* at eval would
measure OOD transfer, which is a different claim than capability. So the grid is
3 t* x 3 n = 9 TRAINED MODELS, each read at 4 step counts = 36 cells.

THE ARM, THE METRIC AND THE MODEL ARE IMPORTED, never reimplemented: `Arm`,
`LR`, `D_MODEL`, `bad` from `scale/m3_capability.py`; `nrmse`, `bootstrap_ci`,
`calibrate_bar`, `bar_verdict`, `M3_TASKS` from `scale/negation_scope.py`. The
training loop below is `m3_capability.run_arm`'s loop with an eval hook -- same
optimiser, same lr, same standardisation, same RED 0-step gate.

C-D, THE LEARNABILITY PRECONDITION. A cell whose NRMSE does not clear 1.0 is
printed `NO READING`, never "loss". 1.0 is the predict-the-mean predictor by
construction (`nrmse` normalises by y's own spread), so a cell at or above it has
not established that anything is learnable there, and a between-arm comparison in
that cell compares two things that both failed to start.

THREADS ARE PINNED HERE, at 8, and stated in every record. The house pin is 2
(`m3_capability.py:66`) and that is what the numbers already in
`results/m3_capability.txt` were taken at; 8 was measured 1.38x-1.55x faster on
this box and saturates there (2/8/16 threads at n=32768: 3.943 / 2.550 / 2.612
s/step). No number here is compared against a 2-thread number, and NRMSE is not
a function of the reduction order at this precision, but the pin is recorded so
the reading is reproducible rather than merely repeated.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scale.m3_capability import Arm, D_MODEL, LR, bad
from scale.negation_scope import (M3_TASKS, nrmse, bootstrap_ci, calibrate_bar,
                                  bar_verdict)

S, D = 64, 24              # the shape every e3 row in results/m3_capability.txt uses
N_EVAL = 4096              # PINNED across cells, so eval noise is identical everywhere
STEP_RUNGS = (150, 600, 2400, 9600)
ROOT = pathlib.Path(__file__).resolve().parents[1]


def train_with_checkpoints(x_train, y_train, x_eval, y_eval, *, s: int,
                           rungs, seed: int):
    """`m3_capability.run_arm`'s loop, read at every rung instead of only the end.

    Returns (red, [per-rung dict]). `red` is the 0-step gate: the UNTRAINED arm
    must sit at or above NRMSE 1.0 on both splits, NaN checked FIRST because
    `float('nan') >= 1.0` is False and would pass a broken instrument silently.
    """
    torch.manual_seed(seed)
    model = Arm("softmax", s)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    mu = float(y_train.mean())
    sigma = float(y_train.std(unbiased=False)) or 1.0

    def raw_pred(x):
        model.eval()
        with torch.no_grad():
            return model(x) * sigma + mu

    r0t, r0e = nrmse(raw_pred(x_train), y_train), nrmse(raw_pred(x_eval), y_eval)
    red = dict(nrmse0_train=r0t, nrmse0_eval=r0e,
               ok=(not bad(r0t)) and (not bad(r0e)) and r0t >= 1.0 and r0e >= 1.0)

    y_std = (y_train - mu) / sigma
    out, done = [], 0
    for rung in sorted(rungs):
        t0 = time.time()
        for _ in range(rung - done):
            model.train()
            opt.zero_grad()
            torch.nn.functional.mse_loss(model(x_train), y_std).backward()
            opt.step()
        done = rung
        pe = raw_pred(x_eval)
        ev = nrmse(pe, y_eval)
        lo, hi = bootstrap_ci(pe, y_eval, seed=seed)
        out.append(dict(steps=rung, train_nrmse=nrmse(raw_pred(x_train), y_train),
                        eval_nrmse=ev, boot_lo=lo, boot_hi=hi,
                        verdict="LEARNS" if (not bad(ev) and ev < 1.0) else "NO READING",
                        secs=round(time.time() - t0, 2)))
    return red, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-star", type=int, required=True, choices=(1, 2, 8, 32))
    ap.add_argument("--n-train", type=int, nargs="+", required=True)
    ap.add_argument("--max-steps", type=int, nargs="+", required=True,
                    help="one cap per --n-train entry; rungs above it are DROPPED")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--seed-cap", type=int, default=600,
                    help="seeds beyond the first only run rungs at or below this")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tag", default="it8")
    a = ap.parse_args()
    if len(a.max_steps) != len(a.n_train):
        ap.error("--max-steps needs one entry per --n-train")
    torch.set_num_threads(a.threads)

    task = "e3_t%d" % a.t_star
    batch_fn, oracle_fn, feature_fn, fd_fn = M3_TASKS[task]
    jl = ROOT / "results" / ("r10_%s_capacity_softmax_t%d.jsonl" % (a.tag, a.t_star))
    log = open(jl, "a", encoding="utf-8")

    def emit(rec):
        log.write(json.dumps(rec) + "\n")
        log.flush()

    emit(dict(t="header", task=task, arm="softmax", s=S, d=D, n_eval=N_EVAL,
              threads=a.threads, torch=torch.__version__, lr=LR, d_model=D_MODEL,
              when=time.strftime("%Y-%m-%d %H:%M:%S"), seeds=a.seeds,
              sign_floor=2 * 2.0 ** (-len(a.seeds))))
    print("=== %s softmax s=%d d=%d n_eval=%d threads=%d torch %s ==="
          % (task, S, D, N_EVAL, a.threads, torch.__version__), flush=True)

    # The analytic ceiling a 1-hop model cannot beat: sqrt((t*-k)/t*) at k=1,
    # from `negation_scope.equilibrium_hop_reading`'s closed form. softmax's hop
    # budget is 1 (`m3_capability.py:271`). This is the number the table has to
    # be read against -- a cell below 1.0 but AT this value has learned
    # everything its architecture can reach, and a cell above it has not.
    ceiling = math.sqrt(max(0.0, a.t_star - 1) / a.t_star)
    print("  1-hop truncation ceiling at t*=%d: NRMSE %.6f (softmax hop budget = 1)"
          % (a.t_star, ceiling), flush=True)
    emit(dict(t="ceiling", t_star=a.t_star, hop_budget=1, nrmse_ceiling=ceiling))

    cal = calibrate_bar(n=N_EVAL, s=S, d=D, steps=600, lr=LR, batch_fn=batch_fn,
                        oracle_fn=oracle_fn, feature_fn=feature_fn)
    ok, why = bar_verdict(cal, flipper_dependence=None if fd_fn is None else fd_fn(S))
    print("  BAR %s -- %s" % ("CALIBRATED" if ok else "BROKEN", why), flush=True)
    emit(dict(t="bar", ok=bool(ok), why=why, **{k: float(v) for k, v in cal.items()}))
    if not ok:
        print("ABORT: calibration bar failed; crediting nothing.", flush=True)
        return 1

    x_eval, y_eval, _, _ = batch_fn(N_EVAL, S, D, d_model=D_MODEL, seed=12345)
    for n, cap in zip(a.n_train, a.max_steps):
        rungs = [r for r in STEP_RUNGS if r <= cap]
        dropped = [r for r in STEP_RUNGS if r > cap]
        for si, seed in enumerate(a.seeds):
            use = rungs if si == 0 else [r for r in rungs if r <= a.seed_cap]
            if not use:
                continue
            x_tr, y_tr, _, _ = batch_fn(n, S, D, d_model=D_MODEL, seed=seed)
            t0 = time.time()
            red, rows = train_with_checkpoints(x_tr, y_tr, x_eval, y_eval,
                                               s=S, rungs=use, seed=seed)
            if not red["ok"]:
                print("INSTRUMENT BROKEN n=%d seed=%d: 0-step %.6f/%.6f"
                      % (n, seed, red["nrmse0_train"], red["nrmse0_eval"]), flush=True)
                emit(dict(t="instrument_broken", n_train=n, seed=seed, **red))
                return 1
            for r in rows:
                emit(dict(t="cell", task=task, t_star=a.t_star, n_train=n,
                          seed=seed, threads=a.threads, **r))
                print("  t*=%2d n=%6d seed=%d steps=%5d  eval NRMSE=%.6f  [%s]  "
                      "boot[%.4f,%.4f]  %.1fs"
                      % (a.t_star, n, seed, r["steps"], r["eval_nrmse"],
                         r["verdict"], r["boot_lo"], r["boot_hi"], r["secs"]),
                      flush=True)
            if dropped and si == 0:
                emit(dict(t="dropped", task=task, t_star=a.t_star, n_train=n,
                          steps=dropped, why="unaffordable, see priced DAG"))
                print("  t*=%2d n=%6d DROPPED steps=%s (unaffordable)"
                      % (a.t_star, n, dropped), flush=True)
            del x_tr, y_tr
            print("    (%.1fs for n=%d seed=%d)" % (time.time() - t0, n, seed),
                  flush=True)
    print("WROTE %s" % jl, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

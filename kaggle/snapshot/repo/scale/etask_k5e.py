"""K-5E: can an arm read an EQUILIBRIUM label at all, and where is the hop wall?

WHY THIS FILE EXISTS. `LOOP_PROMPT.md` section 1.7 makes E1 a must-fire: if the
settled arm cannot beat NRMSE 1.0 on E1 -- the task built from the object its own
resolvent computes -- the harness cannot read an equilibrium label and nothing
downstream of it may be read. That kill needs a reading, and no shipped runner
produces one: `m3_capability.py` does not carry the settled cell, and
`m3_quintuple.py`'s unit builder is wired to `negation_scope.make_batch` with a
journal key that does not name a task, so pointing it at a second corpus would
collide two tasks in one bucket. This file takes the reading without touching
either, by the class-swap `m3_synthetic_settled.py` already uses.

WHAT IT MEASURES, AND THE ONE THING IT IS NOT. Three cells -- softmax, twin,
settled -- across the E3 ladder, so K-5E is decided beside the dial rather than
at one point. That matters because the arms in this project have a HOP BUDGET:
`softmax` forms `x + A@x`, one hop; the pivot and windowed arms reach two; the
settled cell settles the pivot WEIGHTS, not the token chain, so it too reads a
two-hop neighbourhood. A chain label with `t* = 63` is therefore out of reach for
reasons that have nothing to do with whether the harness can read an equilibrium,
and a K-5E fired at `t* = 63` alone would be diagnosing the corpus for a property
of the arms. The ladder separates the two.

NOT THE DECIDING CELL. Seeds here are few and the contrast is NOT read as a
verdict on settled-vs-twin; section 1.8 puts that at 13 seeds or a pre-registered
fixed-sample plan, and it belongs to Chase. Every number below is a
CAPABILITY-FLOOR reading: is this cell above or below the absolute bar.

BUCKETED (ADR-001) is deliberately NOT used: one unit here is cheap and the file
is re-run whole. THREADS ARE PINNED HERE, not by the launcher.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
torch.set_num_threads(2)

from scale import m3_capability as M3                             # noqa: E402
from scale import negation_scope as NS                            # noqa: E402
from scale import paired_arm as PA                                # noqa: E402
from scale.m3_quintuple import BETA, QuintArm                     # noqa: E402

CELLS = ("softmax", "twin", "settled")
TASKS = ("e3_t1", "e3_t2", "e3_t8", "e3_t32", "e1_anchor")


def run_cell(cell: str, task: str, *, s: int, d: int, steps: int,
             n_train: int, n_eval: int, seed: int) -> dict:
    """One cell on one task at one seed. Training is `paired_arm`'s loop, which
    is `run_arm`'s loop verbatim, so a comparison between cells is not a
    comparison between two training loops."""
    batch_fn = NS.M3_TASKS[task][0]

    class _A(QuintArm):
        def __init__(self, kind, s_):
            super().__init__(kind, s_, cell=cell, k_piv=8, beta=BETA,
                             t_max=21, n_neumann=21)

    xt, yt, _f, _p = batch_fn(n_train, s, d, d_model=M3.D_MODEL, seed=seed)
    xe, ye, _g, _h = batch_fn(n_eval, s, d, d_model=M3.D_MODEL, seed=seed + 12345)
    old_m3, old_pa = M3.Arm, PA.Arm
    M3.Arm = PA.Arm = _A
    t0 = time.time()
    try:
        # RED FIRST: the untrained arm must sit at or above the bar, or the
        # instrument is broken and the trained number means nothing.
        red = M3.run_arm("softmax", xt, yt, xe, ye, s=s, steps=0, seed=seed)
        pred, y_eval, npar = PA.train_and_predict(
            "softmax", s=s, d=d, steps=steps, n_train=n_train, n_eval=n_eval,
            seed=seed, batch_fn=batch_fn)
    finally:
        M3.Arm, PA.Arm = old_m3, old_pa
    return dict(cell=cell, task=task, seed=seed, n_params=npar,
                nrmse0_eval=red["nrmse0_eval"], nrmse0_train=red["nrmse0_train"],
                eval_nrmse=NS.nrmse(pred, y_eval), seconds=time.time() - t0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=int, default=64)
    ap.add_argument("--d", type=int, default=24)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--n-train", type=int, default=1024)
    ap.add_argument("--n-eval", type=int, default=512)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--cells", nargs="+", default=list(CELLS))
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--out", default="results/etask_k5e.txt")
    a = ap.parse_args()

    out = pathlib.Path(__file__).resolve().parents[1] / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    log = open(out, "a", encoding="utf-8")

    def emit(line: str) -> None:
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    try:
        emit(f"\n=== K-5E RUN {time.strftime('%Y-%m-%d %H:%M:%S')} "
             f"s={a.s} d={a.d} steps={a.steps} n_train={a.n_train} "
             f"n_eval={a.n_eval} seeds={a.seeds} cells={a.cells} "
             f"tasks={a.tasks} ===")
        emit(f"torch {torch.__version__}  threads={torch.get_num_threads()}  "
             f"device=cpu")
        emit("ABSOLUTE BAR: NRMSE 1.0 is predict-the-mean. K-5E asks only "
             "whether a cell is BELOW it.")
        emit(f"{'task':>12} {'t*':>4} {'cell':>9} {'n_par':>6} "
             f"{'0-step':>9} {'eval':>9} {'verdict':>12} {'s':>7}")
        for task in a.tasks:
            # None for a registered task with no derived dial; the column
            # prints "-" rather than raising KeyError or inventing a number.
            dial = NS.e_t_star(task, a.s)
            t_star = "-" if dial is None else dial
            for cell in a.cells:
                for seed in a.seeds:
                    r = run_cell(cell, task, s=a.s, d=a.d, steps=a.steps,
                                 n_train=a.n_train, n_eval=a.n_eval, seed=seed)
                    verdict = ("BEATS BAR" if r["eval_nrmse"] < 1.0
                               else "AT/ABOVE BAR")
                    emit(f"{task:>12} {t_star:>4} {cell:>9} {r['n_params']:>6} "
                         f"{r['nrmse0_eval']:>9.6f} {r['eval_nrmse']:>9.6f} "
                         f"{verdict:>12} {r['seconds']:>7.1f}")
        return 0
    finally:
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())

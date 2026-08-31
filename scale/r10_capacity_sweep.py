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
import os
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
GATE_TOL = 1e-3            # 0-step gate tolerance; see train_with_checkpoints
STEP_RUNGS = (150, 600, 2400, 9600)
ROOT = pathlib.Path(__file__).resolve().parents[1]


def train_with_checkpoints(x_train, y_train, x_eval, y_eval, *, s: int,
                           rungs, seed: int, arm: str = "softmax",
                           device=None):
    """`m3_capability.run_arm`'s loop, read at every rung instead of only the end.

    Returns (red, [per-rung dict]). `red` is the 0-step gate: the UNTRAINED arm
    must sit at or above NRMSE 1.0 on both splits, NaN checked FIRST because
    `float('nan') >= 1.0` is False and would pass a broken instrument silently.

    `device=None` (default) touches nothing and is byte-identical to every
    number this file has ever published. `x_train`/`y_train`/`x_eval`/`y_eval`
    are expected to already sit on `device` (the caller places them via
    `batch_fn(..., device=device)`, matching `scale/paired_arm.py`'s pattern);
    only the model is moved here, and it is constructed BEFORE the move so its
    initial weights are drawn from the seeded CPU generator on every device --
    `.to(device)` copies values, it does not redraw them.
    """
    torch.manual_seed(seed)
    model = Arm(arm, s)
    if device is not None:
        model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    mu = float(y_train.mean())
    sigma = float(y_train.std(unbiased=False)) or 1.0

    def raw_pred(x):
        model.eval()
        with torch.no_grad():
            return model(x) * sigma + mu

    r0t, r0e = nrmse(raw_pred(x_train), y_train), nrmse(raw_pred(x_eval), y_eval)
    #: GATE_TOL exists because a bare `>= 1.0` sits on the edge of its own null.
    #: Measured over 16 untrained seeds at t*=2, n=2048, with no training at all:
    #: softmax's minimum 0-step eval reading is 1.00055844 and pivot_unsigned's
    #: 1.00055861, clearing a bare 1.0 by 5.6e-4; windowed_signed's minimum is
    #: 0.99997039, and 1 of its 16 seeds falls below, aborting an 8-seed run at
    #: its third seed over a 2.96e-5 excursion. The gate's intent is "the
    #: untrained arm does not MEANINGFULLY beat predict-the-mean", and that word
    #: has to be a quantity. 1e-3 admits the measured tail (34x the observed
    #: excursion) while still rejecting anything better than a tenth of the
    #: trained seed sd at this cell (~1.0e-2). Recorded as M-14.
    #:
    #: This changes only WHICH RUNS ABORT. A cell that passes the gate is scored
    #: exactly as before, so no published number moves.
    red = dict(nrmse0_train=r0t, nrmse0_eval=r0e, gate_tol=GATE_TOL,
               ok=(not bad(r0t)) and (not bad(r0e))
                  and r0t >= 1.0 - GATE_TOL and r0e >= 1.0 - GATE_TOL)

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


def refuse_cross_device_pool(rows: list[dict]) -> str:
    """Refuse a set of cell records that spans more than one `device`.

    CONDITION 1, V15_NEPTUNE_SYSTEMS.md: "a CUDA cell may never be pooled with
    a CPU-taken cell." MISTAKES.md M-10 measured *thread count alone* moving
    `eval_nrmse` by 2.345e-3, which is 0.464 of the pre-registered equivalence
    margin `Delta_eq`; device is named a LARGER perturbation than threads.
    `it11_verdict.by_seed` already refuses a same-seed, same-`threads`
    disagreement, but it buckets by `threads` only -- it has no reason yet to
    keep a CPU cell and a CUDA cell apart, and picks the bucket with the most
    seeds rather than refusing outright. Device gets the stricter rule: ANY
    spread refuses, unconditionally, because condition 1 states no
    accommodation for it ("the reading is void"), not "prefer the bigger
    bucket".

    A row with no `device` field is treated as `"cpu"` -- every journal this
    file wrote before this change was CPU-only by construction, and reading a
    missing field as an unknown fourth bucket would make an old journal and a
    new `--device cpu` journal refuse to pool with each other, which is not
    the defect this guard exists to catch.
    """
    devices = {r.get("device", "cpu") for r in rows}
    if len(devices) > 1:
        raise ValueError(
            f"refuse to pool cells across devices {sorted(devices)}: device is "
            "a larger perturbation than thread count (MISTAKES.md M-10 moved "
            "eval_nrmse by 0.464 of Delta_eq on threads alone; "
            "V15_NEPTUNE_SYSTEMS.md condition 1). Filter to one device before "
            "computing a verdict."
        )
    return devices.pop() if devices else "cpu"


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
    #: Round 11 THE READING needs four arms on this corpus. Default stays
    #: "softmax" so the journal path and every published cell are unchanged.
    ap.add_argument("--arm", default="softmax",
                    choices=("softmax", "pivot_unsigned", "windowed_signed",
                             "pivot_signed"))
    #: Default "cpu" is the byte-identical status quo: nothing in this file
    #: called `.to(...)` or `device=` before this flag existed, so every
    #: number ever published by it was measured on whatever device
    #: `torch.randn`/`nn.Linear` land on with no device argument, i.e. cpu.
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"),
                    help="cpu is the shipped default. cuda currently cannot "
                         "produce a scored verdict from this entry point -- "
                         "see the ABORT below and V15_NEPTUNE_SYSTEMS.md "
                         "condition 2.")
    a = ap.parse_args()
    if len(a.max_steps) != len(a.n_train):
        ap.error("--max-steps needs one entry per --n-train")
    torch.set_num_threads(a.threads)

    device = None
    if a.device == "cuda":
        if not torch.cuda.is_available():
            ap.error("--device cuda requested but torch.cuda.is_available() is False")
        device = torch.device("cuda")
        #: CONDITION 3, V15_NEPTUNE_SYSTEMS.md: cuBLAS is not bitwise
        #: deterministic by default and this run must DECLARE that rather than
        #: leave it silent. Measured on this box (instrumentation, not a
        #: journalled reading): `torch.use_deterministic_algorithms(True)` cost
        #: 0.921x -- i.e. no slowdown, within run-to-run noise -- over 20 timed
        #: steps of this exact Arm/shape (s=64, n=2048, softmax) on cuda, so it
        #: is set unconditionally rather than left off to save a cost that does
        #: not exist. `CUBLAS_WORKSPACE_CONFIG` must be set before the first
        #: CUDA call in the process; nothing above this line has touched cuda.
        #: `cudnn.deterministic` is set too, matching `scale/paired_arm.py`'s
        #: existing cuda lane, though this Arm has no cudnn-backed op to bind.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True

        #: CONDITION 2, V15_NEPTUNE_SYSTEMS.md: calibrate_bar() and GATE_TOL
        #: were established on CPU and neither re-certifies on cuda for free.
        #: `negation_scope.calibrate_bar` never accepts a `device` argument --
        #: its reference batch, its positive-control net and its `feats`
        #: tensor are built with no device threading on every call, cuda run or
        #: not -- so "run calibrate_bar on cuda" is not reachable from this
        #: file; it needs scale/negation_scope.py to accept and honour a
        #: device, and that module is owned by another node this round (out of
        #: this node's write scope). GATE_TOL=1e-3 above is a threshold
        #: measured over 16 untrained CPU seeds (see its docstring). Scoring a
        #: cuda cell against either is importing a CPU-established constant
        #: across a device boundary -- MISTAKES.md V-22's shape -- so this
        #: path refuses to produce a scored verdict rather than do that
        #: silently. `train_with_checkpoints(..., device=...)` and
        #: `refuse_cross_device_pool` are still fully wired (see both above):
        #: the mechanism this predecessor was asked to add exists and is
        #: tested; only the credited-verdict path is gated on re-certification
        #: this node cannot perform.
        print("ABORT: --device cuda cannot produce a scored verdict here.",
              flush=True)
        print("  calibrate_bar() and GATE_TOL=%.0e were established on cpu and "
              "neither re-certifies on cuda without editing "
              "scale/negation_scope.py, which is outside this node's write "
              "scope. See V15_NEPTUNE_SYSTEMS.md condition 2 and "
              "MISTAKES.md V-22." % GATE_TOL, flush=True)
        return 1

    task = "e3_t%d" % a.t_star
    batch_fn, oracle_fn, feature_fn, fd_fn = M3_TASKS[task]
    jl = ROOT / "results" / ("r10_%s_capacity_%s_t%d.jsonl" % (a.tag, a.arm, a.t_star))
    log = open(jl, "a", encoding="utf-8")

    def emit(rec):
        log.write(json.dumps(rec) + "\n")
        log.flush()

    emit(dict(t="header", task=task, arm=a.arm, s=S, d=D, n_eval=N_EVAL,
              threads=a.threads, torch=torch.__version__, lr=LR, d_model=D_MODEL,
              when=time.strftime("%Y-%m-%d %H:%M:%S"), seeds=a.seeds,
              sign_floor=2 * 2.0 ** (-len(a.seeds)), device=a.device,
              deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
    print("=== %s %s s=%d d=%d n_eval=%d threads=%d torch %s device=%s ==="
          % (task, a.arm, S, D, N_EVAL, a.threads, torch.__version__, a.device),
          flush=True)

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

    x_eval, y_eval, _, _ = batch_fn(N_EVAL, S, D, d_model=D_MODEL, seed=12345,
                                    device=device)
    for n, cap in zip(a.n_train, a.max_steps):
        rungs = [r for r in STEP_RUNGS if r <= cap]
        dropped = [r for r in STEP_RUNGS if r > cap]
        for si, seed in enumerate(a.seeds):
            use = rungs if si == 0 else [r for r in rungs if r <= a.seed_cap]
            if not use:
                continue
            x_tr, y_tr, _, _ = batch_fn(n, S, D, d_model=D_MODEL, seed=seed,
                                        device=device)
            t0 = time.time()
            red, rows = train_with_checkpoints(x_tr, y_tr, x_eval, y_eval, arm=a.arm,
                                               s=S, rungs=use, seed=seed,
                                               device=device)
            if not red["ok"]:
                print("INSTRUMENT BROKEN n=%d seed=%d: 0-step %.6f/%.6f"
                      % (n, seed, red["nrmse0_train"], red["nrmse0_eval"]), flush=True)
                emit(dict(t="instrument_broken", n_train=n, seed=seed,
                          device=a.device, **red))
                return 1
            for r in rows:
                emit(dict(t="cell", task=task, t_star=a.t_star, n_train=n,
                          seed=seed, threads=a.threads, device=a.device, **r))
                print("  t*=%2d n=%6d seed=%d steps=%5d  eval NRMSE=%.6f  [%s]  "
                      "boot[%.4f,%.4f]  %.1fs"
                      % (a.t_star, n, seed, r["steps"], r["eval_nrmse"],
                         r["verdict"], r["boot_lo"], r["boot_hi"], r["secs"]),
                      flush=True)
            if dropped and si == 0:
                emit(dict(t="dropped", task=task, t_star=a.t_star, n_train=n,
                          steps=dropped, why="unaffordable, see priced DAG",
                          device=a.device))
                print("  t*=%2d n=%6d DROPPED steps=%s (unaffordable)"
                      % (a.t_star, n, dropped), flush=True)
            del x_tr, y_tr
            print("    (%.1fs for n=%d seed=%d)" % (time.time() - t0, n, seed),
                  flush=True)
    print("WROTE %s" % jl, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

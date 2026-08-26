"""MOVE 2 -- iterate the REPRESENTATION, not the mixture weights.

## Why this arm and not another settling variant

House's structural account of the settled cell: it iterates `alpha` on a simplex
over `{alpha @ av}`, which is a REPARAMETERISATION INSIDE THE TWIN'S OWN FAMILY.
It can reallocate mass; it cannot add anything the twin could not already express.
That is a structural explanation for `settled - twin = 0.002190` at `3.874x` the
twin's seed variance, and no amount of extra settling steps changes it.

The literature says the same thing three ways. Sinkformer iterates the
NORMALISATION to a doubly-stochastic fixed point and reports small gains
(`arXiv:2110.11773`). Deep equilibrium models reach PARITY with deep stacks, not
superiority (`arXiv:1909.01377`). And where iteration does pay, the iterated object
is the REPRESENTATION: weight-tied looped transformers with input injection match
standard ones at under 10 % of the parameters (`arXiv:2311.12424`, ICLR 2024).

So the arm here iterates `x`, not `alpha`:

    z_0 = x
    z_{t+1} = x + A(z_t) @ z_t          three times, weight-tied, input-injected

At `loops = 1` this is EXACTLY the shipped softmax forward, `z = x + A(x) @ x`, and
`bind_loops_one_is_shipped` checks that BITWISE rather than by inspection. Every
parameter tensor is reused across loops, so `n_params` is unchanged and a win
cannot come from capacity.

## Why this comparison is credit-clean, verified in the code and not assumed

House found the arm comparison confounded: `scale/arm_s.py:107` excludes `s-1` and
then drops pivot `0`, so the largest available pivot is `s-2`, and `tril(-1)` makes
that row read only `j <= s-3`. `v[s-2]` is therefore invisible to every pivot row
while softmax's own row `s-1` reads it directly, and the `t*=1` label is
`a[s-1] * b[s-2]` -- exactly the hidden token. Perturbing it moves softmax by
`101.6983` and the pivot value path by `3.263746`.

THAT CONFOUND CANNOT REACH THIS FILE. `QuintArm.forward` only diverts row `s-1`
through `_alpha` when `cell not in ("softmax", "glance")`, and `_alpha` is the only
caller of `batched_pivots`. This arm is softmax-looped against plain softmax:
neither side calls `_alpha`, neither selects a pivot, and no signed resolvent is
touched, so the reading is credit-clean under LOOP_PROMPT.md 1.7d even while
`e3`'s pivot numbers are void. Chase owns lifting the exclusion (Move 1); this
does not wait on it.

## Why depth is the right axis, with the wall quantified

`arXiv:2402.09268` Theorem 4.2 places `hop_k` at `L = floor(log2 k) + 2`, so
`t* = 8` needs about `5` layers and these arms are depth `1`. At `t* = 8` the arm
trains to `0.860972` while sitting on its own 2-hop ceiling `0.866025`, and
evaluates `1.112208`. IT IS MEMORISING NOISE, NOT FAILING TO SETTLE. A weight-tied
loop is the cheapest depth available at fixed parameter count.

## THE PRE-REGISTRATION, WRITTEN BEFORE ANY NUMBER WAS TAKEN

Tasks `e3_t2` AND `e3_t8`. `n_train = 2048`, `s = 64`, `d = 24`, seed `0`, both
`steps = 150` and `steps = 600`. Two step counts because `150` is ALREADY MEASURED
as failing a width-2 control that passes at `600` (`negation_scope.E2_STEPS`, which
records `2.446646` at 150 against `0.922725` at 600), so a single step count would
confound depth with undertraining.

    beats_at_t2(steps)     := looped3 eval_nrmse < softmax eval_nrmse on e3_t2
    under_one_at_t8(steps) := looped3 eval_nrmse < 1.0 on e3_t8

    DEPTH_VIA_LOOP_DIES := not any(beats_at_t2(st) or under_one_at_t8(st)
                                   for st in (150, 600))

The disjunction over both step counts is deliberate and it is the GENEROUS reading:
a kill should not be purchasable by picking the step count that fails. If it dies,
the twin ships on its earned `+0.111396` and the headline reads
"one-step softmax wins on equilibrium tasks", in those words.

## What my own rate law does and does NOT say about this arm

`scale/foreman_lambda2.py` derives an exact asymptotic rate for a FIXED linear
operator `Q`. This loop's operator `A(z_t)` is rebuilt from the current
representation at every step, so the iteration is NONLINEAR and `lambda_2` of any
one operator is not its rate. The honest statement is narrower and is measured by
`operator_is_nilpotent`: `A` is strictly causal, hence nilpotent, so the loop does
not converge to a fixed point at all -- it accumulates hops and terminates. Three
loops reach hop 3. That is why this is a DEPTH intervention and not a settling one,
and it is the same nilpotency `CEQ.Nilpotent.pow_card_eq_zero` proves.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ceq import bench                                              # noqa: E402
from scale import m3_capability as M3                              # noqa: E402
from scale import negation_scope as NS                             # noqa: E402
from scale import paired_arm as PA                                 # noqa: E402

JOURNAL = ROOT / "results" / "foreman_looped.jsonl"

TASKS = ("e3_t2", "e3_t8")
STEP_COUNTS = (150, 600)
LOOPS = 3
CFG = dict(s=64, d=24, n_train=2048, n_eval=4096, seed=0)


class LoopedArm(M3.Arm):
    """Weight-tied looped softmax with input injection. No new parameters.

    `loops = 1` reduces to the shipped forward exactly; the reduction is checked
    bitwise by `bind_loops_one_is_shipped` rather than argued from the source.
    """

    loops = LOOPS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.shape[1]
        z = x
        for _ in range(self.loops):
            a = bench._softmax_operator(self.wq(z), self.wk(z))
            z = x + a @ z
        return self.readout(self.mlp(z)).squeeze(-1)[:, s - 1]


def bind_loops_one_is_shipped(s: int = 64, seed: int = 0) -> dict:
    """MUST-FIRE, BOTH HALVES. `loops = 1` must be BITWISE the shipped softmax
    forward, and `loops = 3` must NOT be.

    A bind that only checked the equality half would pass for an arm whose loop
    body never ran, which is the degenerate PASS this project has struck fifteen
    times. Both halves are asserted and both maxdiffs are printed.
    """
    torch.manual_seed(seed)
    ref = M3.Arm("softmax", s)
    looped = LoopedArm("softmax", s)
    looped.load_state_dict(ref.state_dict())
    g = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(4, s, M3.D_MODEL, generator=g)
    with torch.no_grad():
        want = ref(x)
        looped.loops = 1
        one = looped(x)
        looped.loops = LOOPS
        three = looped(x)
    same = bool(torch.equal(want, one))
    differs = not bool(torch.equal(want, three))
    return {"loops1_bitwise_equal": same,
            "loops1_maxdiff": float((want - one).abs().max()),
            "loops3_differs": differs,
            "loops3_maxdiff": float((want - three).abs().max()),
            "n_params_ref": M3.n_params(ref),
            "n_params_looped": M3.n_params(looped),
            "fires": same and differs}


def operator_is_nilpotent(s: int = 64, seed: int = 0) -> dict:
    """The loop accumulates hops; it does not settle. Measured, not asserted.

    `A` is strictly causal, so `A ** s == 0` exactly and the Neumann series
    terminates -- `CEQ.Nilpotent.pow_card_eq_zero` is the same fact over any
    `CommRing`. This is why `foreman_lambda2`'s rate law does NOT apply to this
    arm and why three loops is a DEPTH claim rather than a settling one: with
    `rho(A) = 0` there is no relaxation time to engineer.
    """
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, s, M3.D_MODEL, generator=g)
    k = torch.randn(1, s, M3.D_MODEL, generator=g)
    a = bench._softmax_operator(q, k)[0]
    on_or_above = float(a.triu(0).abs().max())
    p = torch.linalg.matrix_power(a.double(), s)
    return {"max_entry_on_or_above_diagonal": on_or_above,
            "spectral_radius": float(torch.linalg.eigvals(a.double()).abs().max()),
            "max_entry_of_A_to_the_s": float(p.abs().max()),
            "strictly_causal": on_or_above == 0.0,
            "nilpotent": float(p.abs().max()) == 0.0}


def run_cell(task: str, *, looped: bool, steps: int, **cfg) -> dict:
    """One (arm, task, steps) cell, trained by `run_arm`'s own loop.

    The class swap is the same mechanism `m3_quintuple._unit` uses: `run_arm` and
    `train_and_predict` both instantiate the module-level `Arm`, so replacing that
    name is what puts a different forward on an otherwise byte-identical training
    path. `steps` and `task` are the only things that move between cells.
    """
    p = dict(CFG, **cfg)
    s, d, seed = p["s"], p["d"], p["seed"]
    bfn = NS.M3_TASKS[task][0]
    cls = LoopedArm if looped else M3.Arm

    xt, yt, _a, _b = bfn(p["n_train"], s, d, d_model=M3.D_MODEL, seed=seed)
    xe, ye, _c, _e = bfn(p["n_eval"], s, d, d_model=M3.D_MODEL, seed=seed + 12345)
    old_m3, old_pa = M3.Arm, PA.Arm
    M3.Arm = PA.Arm = cls
    t0 = time.time()
    try:
        red = M3.run_arm("softmax", xt, yt, xe, ye, s=s, steps=0, seed=seed)
        pred, y_eval, npar = PA.train_and_predict(
            "softmax", s=s, d=d, steps=steps, n_train=p["n_train"],
            n_eval=p["n_eval"], seed=seed, batch_fn=bfn)
    finally:
        M3.Arm, PA.Arm = old_m3, old_pa
    lo, hi = NS.bootstrap_ci(pred, y_eval, seed=seed)
    return dict(task=task, arm=("looped%d" % LOOPS) if looped else "softmax",
                steps=steps, n_params=npar,
                eval_nrmse=NS.nrmse(pred, y_eval),
                nrmse0_train=red["nrmse0_train"], nrmse0_eval=red["nrmse0_eval"],
                marg_lo=lo, marg_hi=hi, seconds=round(time.time() - t0, 1),
                **{k: p[k] for k in ("s", "d", "n_train", "n_eval", "seed")})


def journal(row: dict) -> None:
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with open(JOURNAL, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def load() -> list:
    if not JOURNAL.exists():
        return []
    return [json.loads(ln) for ln in JOURNAL.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def falsifier(rows=None) -> dict:
    """The pre-registered verdict, evaluated from the journal and nothing else.

    Returns `complete: False` while any of the six cells is missing, so a partial
    run cannot be read as a kill. A verdict taken on a truncated table is exactly
    how the consequence-fidelity file was nearly misread this round.
    """
    rows = load() if rows is None else rows
    got = {(r["task"], r["arm"], r["steps"]): r["eval_nrmse"] for r in rows}
    looped = "looped%d" % LOOPS
    survive, detail = [], []
    for st in STEP_COUNTS:
        a = got.get(("e3_t2", looped, st))
        b = got.get(("e3_t2", "softmax", st))
        c = got.get(("e3_t8", looped, st))
        beats = None if (a is None or b is None) else a < b
        under = None if c is None else c < 1.0
        detail.append({"steps": st, "t2_looped": a, "t2_softmax": b,
                       "beats_at_t2": beats, "t8_looped": c,
                       "under_one_at_t8": under})
        survive.append(bool(beats) or bool(under))
    complete = all(d["beats_at_t2"] is not None and d["under_one_at_t8"] is not None
                   for d in detail)
    return {"complete": complete, "detail": detail,
            "depth_via_loop_dies": complete and not any(survive)}


def _argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binds", action="store_true", help="controls only, no training")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--steps", nargs="+", type=int, default=list(STEP_COUNTS))
    ap.add_argument("--arms", nargs="+", default=["softmax", "looped"])
    ap.add_argument("--verdict", action="store_true", help="read the journal only")
    return ap


def main() -> int:
    a = _argparser().parse_args()
    if a.verdict:
        print(json.dumps(falsifier(), indent=2, sort_keys=True))
        return 0

    b = bind_loops_one_is_shipped()
    n = operator_is_nilpotent()
    print("CONTROLS")
    print("  loops=1 bitwise equals shipped softmax : %s  maxdiff %.3e"
          % (b["loops1_bitwise_equal"], b["loops1_maxdiff"]))
    print("  loops=3 DIFFERS from it                : %s  maxdiff %.6f"
          % (b["loops3_differs"], b["loops3_maxdiff"]))
    print("  n_params ref %d  looped %d  equal %s"
          % (b["n_params_ref"], b["n_params_looped"],
             b["n_params_ref"] == b["n_params_looped"]))
    print("  bind FIRES (both halves)               : %s" % b["fires"])
    print("  A strictly causal %s   rho(A) %.3e   max|A^s| %.3e   nilpotent %s"
          % (n["strictly_causal"], n["spectral_radius"],
             n["max_entry_of_A_to_the_s"], n["nilpotent"]))
    if not b["fires"]:
        print("BIND FAILED -- no number below would mean anything.")
        return 1
    if a.binds:
        return 0

    for task in a.tasks:
        for steps in a.steps:
            for arm in a.arms:
                row = run_cell(task, looped=(arm == "looped"), steps=steps)
                journal(row)
                print("  %-7s %-8s steps=%-4d eval_nrmse %.6f  n_params %d  "
                      "0-step eval %.6f  [%.1fs]"
                      % (row["task"], row["arm"], row["steps"], row["eval_nrmse"],
                         row["n_params"], row["nrmse0_eval"], row["seconds"]))
    print(json.dumps(falsifier(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

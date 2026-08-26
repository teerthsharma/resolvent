"""M3 capability arms — trainable ARMS on the calibrated M3 instrument.

Three arms sharing IDENTICAL q/k projection shape, MLP capacity and readout,
differing ONLY in the operator (and, for the pivot arms, the hop-2 term):

    softmax        : A = causal softmax operator (ceq.bench._softmax_operator)
                     out = readout(MLP(x + A@x))[:, s-1]
    pivot_signed   : A = causal sgate operator (ceq.bench._causal_sgate_operator,
                     rho=SGATE_RHO, lam=SGATE_LAM, window=0)
                     hop2 = pivot_hop2(A, select_pivots(key, k))
                     out = readout(MLP(x + A@x + hop2@x))[:, s-1]
    pivot_unsigned : A = causal softmax operator, same hop2/pivot construction
                     as pivot_signed. Ablation: routing without signed content.

CPU ONLY. `make_batch`, `oracle`, `nrmse`, `bootstrap_ci`, `calibrate_bar` are
IMPORTED from scale/negation_scope.py, never reimplemented. `select_pivots` and
`pivot_hop2` are IMPORTED from scale/pivot_probe.py, never reimplemented.

BATCHING NOTE. `bench._softmax_operator` / `bench._causal_sgate_operator` are
pure broadcasting ops over a [s,s] causal mask, so they accept a batched
[n,s,d] q/k directly with no change to their math (verified by reading
ceq/bench.py: `_causal_mask` returns [s,s] and every op is masked_fill/matmul,
which broadcasts a leading batch dim for free). `pivot_hop2`, however, is
written as `a[:, pivots] @ a[pivots, :]`, which indexes dim 0 as the ROW axis —
correct for a single [s,s] operator, wrong for a batched [n,s,s] one (it would
index into the batch). So the operator matrices are built batched in one call,
and only the hop-2 step LOOPS over the batch (pivots are content-selected per
example, so they differ per example anyway). Neither imported function's
internals are touched.

RED BEFORE GREEN, in order: calibrate_bar() must pass; every arm's NRMSE at
0 training steps must read >= 1.0 on BOTH train and held-out (checked for
NaN/Inf FIRST, since `float('nan') >= 1.0` is False in Python and would
silently pass a broken instrument through a bare `>=` check).
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ceq import bench
from scale.negation_scope import (make_batch, nrmse, bootstrap_ci, calibrate_bar,
                                  bar_verdict, M3_TASKS)
from scale.pivot_probe import (select_pivots, pivot_hop2,
                               batched_select_pivots, batched_pivot_hop2)

#: CPU matmul reduction order varies with the thread count, so an unpinned run
#: is not reproducible. Every published number in results/m3_capability.txt from
#: 2026-08-25 19:03 onward was taken at 2 threads -- but supplied by the
#: LAUNCHER's environment, not by this file, which pinned nothing. The same
#: command run from a shell without OMP_NUM_THREADS set gets this box's default
#: of 20 (`torch.get_num_threads()` reads 20; `os.cpu_count()` reads 28), which
#: is what lines 345-741 of that log recorded. Pinning here makes the file, not
#: the shell, the authority -- the same discipline as `inspector.py:240` and
#: `tests/chase/test_resume_checkpoint.py:19`.
torch.set_num_threads(2)

ARMS = ("softmax", "pivot_signed", "pivot_unsigned", "windowed_signed")

#: F4's arm, and Phase 0's whole subject. Windowed signed multi-hop is the ONE
#: construction in this project already measured FLAT across a 64x growth in
#: context -- the exact property the scale-free routes are chasing -- and it has
#: never been capability-tested. That gap is why it is here: a flat statistic
#: that never produced a capability is a counterexample to the gate, and the
#: cheapest way to find out is to run the capability test on it.
#:
#: Built on `sgate`, NOT `tgate`: the round-1 archive records the flatness on
#: "every windowed sgate interval" (DONE_ARCHIVE_ROUND1.md:1272). Copying the
#: sibling arm's operator would have measured a different object under F4's name.
W_WINDOW = 8
D_MODEL = 16          # fixed per task spec
HIDDEN = 128           # MLP hidden width, SAME for every arm (fair capacity).
                        # pivot_signed's extra params are g[s]+tau(1), which
                        # GROW WITH s -- HIDDEN must be large enough that this
                        # stays under the 10% param-match bar at the s values
                        # actually run (measured: HIDDEN=32 broke the 10% bar
                        # at s=192 -- base was only 1601 params against a
                        # 193-param extra, ratio 12.05%; see results log).
K_PIVOTS = 8           # pivot count, matches pivot_probe's default
LR = 0.02

#: The SHIPPED sgate settings. Kept as module constants rather than inline
#: literals so the operator bind can assert against the same two numbers the
#: arm is built from, instead of two copies that can drift apart -- which is
#: how `report()` and `_verdict()` disagreed in round 2.
SGATE_RHO, SGATE_LAM = 1.5, 0.10


class Arm(nn.Module):
    """One arm. q/k projection + 2-layer MLP + scalar readout, all shared shape
    across arms; only the operator (and whether hop2 is added) differs."""

    def __init__(self, kind: str, s: int, d_model: int = D_MODEL,
                 hidden: int = HIDDEN, k_pivots: int = K_PIVOTS):
        super().__init__()
        if kind not in ARMS:
            raise ValueError(kind)
        self.kind = kind
        self.k_pivots = k_pivots
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model)
        )
        self.readout = nn.Linear(d_model, 1)
        # NO extra operator parameters. `sgate` is `rho`/`lam` scalars at their
        # SHIPPED defaults, so the signed arm now has EXACTLY the same parameter
        # count as softmax. The g[s]+tau pair the HIDDEN note above was sized
        # against belonged to `tgate`, and tgate is gone from this file.

    def _operator(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """The operator each arm measures. BOUND: every branch must return a
        tensor bitwise-equal to something the module ships, and
        `tests/loop/test_m3_harness_operator_is_shipped.py` checks it by VALUE
        over every entry of `ARMS`.

        This branch used to return `_causal_tgate_operator`, which appears
        nowhere in the shipped path. Round 2 caught that exact defect in the M2
        probe and bound `pivot_probe.py` against it; the bind never reached this
        file, so the CAPABILITY numbers -- the ones that decide whether any of
        this is worth anything -- were readings of a non-shipped operator.
        """
        if self.kind == "softmax" or self.kind == "pivot_unsigned":
            return bench._softmax_operator(q, k)              # [n,s,s], batched
        # `window=0` is the unbounded causal row; `window=W_WINDOW` is F4's
        # banded receptive field. Both are the SAME shipped operator with one
        # argument different, which is the point -- the windowed arm is not a
        # new operator and must not become one.
        w = W_WINDOW if self.kind == "windowed_signed" else 0
        return bench._causal_sgate_operator(q, k, rho=SGATE_RHO, lam=SGATE_LAM,
                                            window=w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, s, _ = x.shape
        q, k = self.wq(x), self.wk(x)
        a = self._operator(q, k)                               # [n,s,s]
        z = x + a @ x
        if self.kind == "windowed_signed":
            # DENSE within the band, NOT pivot-routed. F4's claim is *windowed
            # signed multi-hop*; routing hop 2 through k content-selected pivots
            # is a different construction and it is already dead (-1.298).
            # `a` is banded by the operator, so `a @ a` reaches 2w and no
            # further -- the bounded receptive field is the arm's whole content.
            z = z + a @ (a @ x)
        elif self.kind != "softmax":
            # pivots are content-selected PER EXAMPLE (key differs per row), so
            # pivot_hop2 -- which only accepts a single 2D [s,s] operator --
            # loops over the batch. `a` itself was already built batched above.
            # Was a Python loop over the batch, one `pivot_hop2` call per
            # example. The batched pair is BITWISE-EQUAL to that loop, gradient
            # included (tests/wilson/test_hop2_vec.py). The loop's BACKWARD was
            # superlinear -- 24.56 s at n=2048 against 0.066 s here -- because
            # it built one autograd subgraph per example, which is the memory
            # cost that killed the n_train=8192 run as well as the time cost.
            hop2 = batched_pivot_hop2(a, batched_select_pivots(k, self.k_pivots))
            z = z + hop2 @ x
        h = self.mlp(z)
        out = self.readout(h).squeeze(-1)                      # [n, s]
        return out[:, s - 1]


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def run_arm(kind: str, x_train, y_train, x_eval, y_eval, *, s: int, steps: int,
            seed: int) -> dict:
    torch.manual_seed(seed)
    model = Arm(kind, s)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    mu = float(y_train.mean())
    sigma = float(y_train.std(unbiased=False))
    if sigma == 0.0:
        sigma = 1.0

    def raw_pred(x):
        model.eval()
        with torch.no_grad():
            out_std = model(x)
        return out_std * sigma + mu

    # ---- RED: 0 training steps ----
    pred0_train, pred0_eval = raw_pred(x_train), raw_pred(x_eval)
    nrmse0_train = nrmse(pred0_train, y_train)
    nrmse0_eval = nrmse(pred0_eval, y_eval)

    y_train_std = (y_train - mu) / sigma
    for _ in range(steps):
        model.train()
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x_train), y_train_std)
        loss.backward()
        opt.step()

    pred_train, pred_eval = raw_pred(x_train), raw_pred(x_eval)
    train_nrmse = nrmse(pred_train, y_train)
    eval_nrmse = nrmse(pred_eval, y_eval)
    ci_lo, ci_hi = bootstrap_ci(pred_eval, y_eval, seed=seed)

    return dict(kind=kind, n_params=n_params(model),
                nrmse0_train=nrmse0_train, nrmse0_eval=nrmse0_eval,
                train_nrmse=train_nrmse, eval_nrmse=eval_nrmse,
                ci_lo=ci_lo, ci_hi=ci_hi)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for st in self.streams:
            st.write(data)

    def flush(self):
        for st in self.streams:
            st.flush()


def bad(v: float) -> bool:
    """NaN/Inf check, done explicitly and FIRST: `float('nan') >= 1.0` is False
    in Python, so a bare `>=` gate lets a NaN sail through as a silent pass."""
    return math.isnan(v) or math.isinf(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=int, default=64)
    ap.add_argument("--d", type=int, default=24)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--n-train", type=int, default=128)
    ap.add_argument("--n-eval", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    #: WHICH TASK. Default is the shipped negation-scope task, so every number
    #: already in results/m3_capability.txt is reproduced by the same command
    #: that produced it. `counter_squared` is the S2 Hankel-gap task; see
    #: scale/negation_scope.py::M3_TASKS.
    ap.add_argument("--task", default="negation_scope", choices=list(M3_TASKS))
    a = ap.parse_args()
    batch_fn, oracle_fn, feature_fn, fd_fn = M3_TASKS[a.task]

    out_path = pathlib.Path(__file__).resolve().parents[1] / "results" / "m3_capability.txt"
    log_f = open(out_path, "a", encoding="utf-8")
    real_stdout = sys.stdout
    sys.stdout = Tee(real_stdout, log_f)
    t0 = time.time()
    try:
        print(f"\n=== RUN {time.strftime('%Y-%m-%d %H:%M:%S')} "
              f"s={a.s} d={a.d} steps={a.steps} n_train={a.n_train} "
              f"n_eval={a.n_eval} seed={a.seed} arms={a.arms} task={a.task} ===")
        print(f"torch {torch.__version__}  torch.get_num_threads()={torch.get_num_threads()}  "
              f"device=cpu (no .cuda() anywhere in this file)")

        print("\n=== BAR CALIBRATION (must pass before any arm is credited) ===")
        cal = calibrate_bar(n=a.n_eval, s=a.s, d=a.d, steps=a.steps, lr=LR,
                            batch_fn=batch_fn, oracle_fn=oracle_fn,
                            feature_fn=feature_fn)
        for name, v in cal.items():
            print(f"  {name:>20} {v:.6f}")
        # ONE gate, owned by `negation_scope`. This block used to hold a private
        # copy of the pass condition, and two of its three clauses were algebraic
        # identities -- `nrmse(y.mean(), y)` is 1.0 by definition and
        # `nrmse(oracle(x,f,p), y)` is `nrmse(t, t)` because `make_batch` RETURNS
        # `oracle(x,f,p)` as `y`. A flipper-blind label passed it. Round 2 also
        # shipped a verdict whose tested copy was right while the copy that ran
        # was wrong; two copies of one rule is that defect waiting.
        cal_ok, why = bar_verdict(
            cal, flipper_dependence=None if fd_fn is None else fd_fn(a.s))
        print(f"  BAR {'CALIBRATED' if cal_ok else 'BROKEN'}  -- {why}")
        if not cal_ok:
            print("ABORT: calibration bar failed.")
            return 1

        print(f"\n=== DATA (same train/eval batches reused across every arm) ===")
        x_train, y_train, f_tr, p_tr = batch_fn(a.n_train, a.s, a.d, d_model=D_MODEL, seed=a.seed)
        x_eval, y_eval, f_ev, p_ev = batch_fn(a.n_eval, a.s, a.d, d_model=D_MODEL, seed=a.seed + 12345)
        print(f"  train: n={a.n_train} seed={a.seed}  flipper@{f_tr} payload@{p_tr}")
        print(f"  eval : n={a.n_eval} seed={a.seed + 12345} (DIFFERENT seed)  flipper@{f_ev} payload@{p_ev}")

        results = {}
        print("\n=== TRAIN + EVAL per arm ===")
        for kind in a.arms:
            r = run_arm(kind, x_train, y_train, x_eval, y_eval, s=a.s, steps=a.steps, seed=a.seed)
            results[kind] = r

            r0t, r0e = r["nrmse0_train"], r["nrmse0_eval"]
            red_ok = (not bad(r0t)) and (not bad(r0e)) and r0t >= 1.0 and r0e >= 1.0
            print(f"\n  [{kind}]  n_params={r['n_params']}")
            print(f"    RED  0-step NRMSE  train={r0t:.6f}  eval={r0e:.6f}  "
                  f"[{'OK' if red_ok else 'INSTRUMENT BROKEN'}]")
            if not red_ok:
                print(f"INSTRUMENT BROKEN: arm={kind} untrained NRMSE below 1.0 "
                      f"(or NaN/Inf). Aborting, crediting nothing.")
                return 1

            tr, ev = r["train_nrmse"], r["eval_nrmse"]
            tr_fail, ev_fail = bad(tr), bad(ev)
            print(f"    POST steps={a.steps}  train NRMSE={tr:.6f}"
                  f"{' [FAIL: NaN/Inf]' if tr_fail else ''}"
                  f"   eval NRMSE={ev:.6f}{' [FAIL: NaN/Inf]' if ev_fail else ''}")
            print(f"    eval bootstrap_ci (n_boot=400) = [{r['ci_lo']:.6f}, {r['ci_hi']:.6f}]")

        base = results.get("softmax", {}).get("n_params")
        if base is not None:
            print("\n=== PARAM MATCH (arm 1 vs 2/3, must be within 10%) ===")
            for kind in ("pivot_signed", "pivot_unsigned"):
                if kind in results:
                    np_k = results[kind]["n_params"]
                    ratio = abs(np_k - base) / base
                    print(f"  softmax={base}  {kind}={np_k}  ratio={ratio:.4f}")
                    assert ratio <= 0.10, (
                        f"PARAM MISMATCH: {kind}={np_k} vs softmax={base} "
                        f"exceeds 10% ({ratio:.4f})"
                    )

        wall = time.time() - t0
        print(f"\n=== WALL CLOCK: {wall:.3f} s ===")
        return 0
    finally:
        sys.stdout = real_stdout
        log_f.close()


if __name__ == "__main__":
    sys.exit(main())

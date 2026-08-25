"""M3 capability arms — trainable ARMS on the calibrated M3 instrument.

Three arms sharing IDENTICAL q/k projection shape, MLP capacity and readout,
differing ONLY in the operator (and, for the pivot arms, the hop-2 term):

    softmax        : A = causal softmax operator (ceq.bench._softmax_operator)
                     out = readout(MLP(x + A@x))[:, s-1]
    pivot_signed   : A = causal tgate operator (ceq.bench._causal_tgate_operator)
                     hop2 = pivot_hop2(A, select_pivots(key, k))
                     out = readout(MLP(x + A@x + hop2@x))[:, s-1]
    pivot_unsigned : A = causal softmax operator, same hop2/pivot construction
                     as pivot_signed. Ablation: routing without signed content.

CPU ONLY. `make_batch`, `oracle`, `nrmse`, `bootstrap_ci`, `calibrate_bar` are
IMPORTED from scale/negation_scope.py, never reimplemented. `select_pivots` and
`pivot_hop2` are IMPORTED from scale/pivot_probe.py, never reimplemented.

BATCHING NOTE. `bench._softmax_operator` / `bench._causal_tgate_operator` are
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
from scale.negation_scope import make_batch, nrmse, bootstrap_ci, calibrate_bar
from scale.pivot_probe import select_pivots, pivot_hop2

ARMS = ("softmax", "pivot_signed", "pivot_unsigned")
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
        if kind == "pivot_signed":
            # extra operator params ONLY the tgate operator needs (g, tau).
            self.g = nn.Parameter(torch.sigmoid(torch.randn(s)))
            self.tau = nn.Parameter(torch.tensor(1.0))
        else:
            self.g = None
            self.tau = None

    def _operator(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        if self.kind == "softmax" or self.kind == "pivot_unsigned":
            return bench._softmax_operator(q, k)              # [n,s,s], batched
        return bench._causal_tgate_operator(q, k, self.g, self.tau)  # pivot_signed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, s, _ = x.shape
        q, k = self.wq(x), self.wk(x)
        a = self._operator(q, k)                               # [n,s,s]
        z = x + a @ x
        if self.kind != "softmax":
            # pivots are content-selected PER EXAMPLE (key differs per row), so
            # pivot_hop2 -- which only accepts a single 2D [s,s] operator --
            # loops over the batch. `a` itself was already built batched above.
            hop2 = torch.stack(
                [pivot_hop2(a[i], select_pivots(k[i], self.k_pivots))
                 for i in range(n)],
                dim=0,
            )
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
    a = ap.parse_args()

    out_path = pathlib.Path(__file__).resolve().parents[1] / "results" / "m3_capability.txt"
    log_f = open(out_path, "a", encoding="utf-8")
    real_stdout = sys.stdout
    sys.stdout = Tee(real_stdout, log_f)
    t0 = time.time()
    try:
        print(f"\n=== RUN {time.strftime('%Y-%m-%d %H:%M:%S')} "
              f"s={a.s} d={a.d} steps={a.steps} n_train={a.n_train} "
              f"n_eval={a.n_eval} seed={a.seed} arms={a.arms} ===")
        print(f"torch {torch.__version__}  torch.get_num_threads()={torch.get_num_threads()}  "
              f"device=cpu (no .cuda() anywhere in this file)")

        print("\n=== BAR CALIBRATION (must pass before any arm is credited) ===")
        cal = calibrate_bar(n=a.n_eval, s=a.s, d=a.d)
        for name, v in cal.items():
            print(f"  {name:>18} NRMSE {v:.6f}")
        cal_ok = (abs(cal["predict_the_mean"] - 1.0) < 1e-6
                  and cal["payload_only"] >= 1.0
                  and cal["oracle"] < 1e-6)
        print(f"  BAR {'CALIBRATED' if cal_ok else 'BROKEN'}")
        if not cal_ok:
            print("ABORT: calibration bar failed.")
            return 1

        print(f"\n=== DATA (same train/eval batches reused across every arm) ===")
        x_train, y_train, f_tr, p_tr = make_batch(a.n_train, a.s, a.d, d_model=D_MODEL, seed=a.seed)
        x_eval, y_eval, f_ev, p_ev = make_batch(a.n_eval, a.s, a.d, d_model=D_MODEL, seed=a.seed + 12345)
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

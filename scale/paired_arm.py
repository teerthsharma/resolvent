"""Per-example eval predictions for one arm, so sign-vs-routing gets the RIGHT test.

Iteration 17 compared `pivot_signed` [0.6326, 0.7156] against `pivot_unsigned`
[0.6968, 0.7977] using PER-ARM bootstrap intervals, and they overlapped by
0.0188, so G4 fired and the capability was attributed to routing.

That test is the conservative one and it is also the WRONG one. Both arms are
evaluated on the IDENTICAL batch, so the shared batch-to-batch variance appears
in both intervals and inflates both. The question "does sign add anything ON TOP
OF routing" is a question about the per-example DIFFERENCE, and the correct
instrument is a paired bootstrap on

    |pred_signed - y|  -  |pred_unsigned - y|

which cancels the shared variance exactly. Reporting the unpaired result first
was deliberate -- it is the reading that makes the signed claim HARDER, and a
project that has published six withdrawn novelty claims does not get to pick the
flattering test first.

This dumps one arm's predictions so the paired statistic can be formed without
holding two trained models in memory at once, and so each call stays inside the
wall-clock ceiling that ADR-001 exists to respect.

Training is `run_arm`'s loop VERBATIM -- same optimiser, same standardisation,
same order -- because a paired test between two arms trained by two different
loops measures the loops.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scale.m3_capability import Arm, D_MODEL, LR, n_params      # noqa: E402
from scale.negation_scope import make_batch, nrmse              # noqa: E402


def train_and_predict(kind: str, *, s: int, d: int, steps: int, n_train: int,
                      n_eval: int, seed: int, batch_fn=None):
    #: `batch_fn` selects the M3 TASK (`negation_scope.M3_TASKS`); it defaults to
    #: the shipped builder, so every published reading is unchanged.
    bfn = batch_fn or make_batch
    x_train, y_train, _, _ = bfn(n_train, s, d, d_model=D_MODEL, seed=seed)
    x_eval, y_eval, _, _ = bfn(n_eval, s, d, d_model=D_MODEL,
                               seed=seed + 12345)
    torch.manual_seed(seed)
    model = Arm(kind, s)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mu = float(y_train.mean())
    sigma = float(y_train.std(unbiased=False)) or 1.0
    ystd = (y_train - mu) / sigma
    for _ in range(steps):
        model.train()
        opt.zero_grad()
        torch.nn.functional.mse_loss(model(x_train), ystd).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(x_eval) * sigma + mu
    return pred, y_eval, n_params(model)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--s", type=int, default=64)
    ap.add_argument("--d", type=int, default=24)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--n-train", type=int, default=8192)
    ap.add_argument("--n-eval", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(2)
    pred, y, npar = train_and_predict(a.arm, s=a.s, d=a.d, steps=a.steps,
                                      n_train=a.n_train, n_eval=a.n_eval,
                                      seed=a.seed)
    torch.save(dict(arm=a.arm, seed=a.seed, pred=pred, y=y, n_params=npar,
                    s=a.s, d=a.d, steps=a.steps, n_train=a.n_train), a.out)
    print(f"  {a.arm} seed={a.seed} n_params={npar} eval NRMSE={nrmse(pred, y):.6f}"
          f"  -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

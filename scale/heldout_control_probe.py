"""What would the bar's positive control read if it were scored HELD-OUT?

The finding: `calibrate_bar` trains the two-feature positive control on `feats`
and scores it on the same `feats` (negation_scope.py:1529, :1536), while every arm
that bar gates is scored on a batch drawn at `seed + 12345`. The clause
`trained_two_feature < 1.0` is the ONLY one of bar_verdict's five that asks
whether anything can learn the task.

This does NOT repair the gate -- repairing it mid-round would re-certify data
already collected. It replicates the control faithfully and adds one thing: a
second draw, scored without being trained on. If the held-out reading is still
well under 1.0 the caveat is real but costs nothing here; if it is at or above
1.0 the certification was doing no work and every arm reading at that t* is
uninterpretable, which is what the clause's own failure message says.

Faithful to the shipped control: same net (Linear(k,32) / GELU / Linear(32,1)),
same init (normal_(0, 0.5) with torch.Generator().manual_seed(seed), zeroed
biases), same Adam lr, same step count, same standardisation of the target and
same un-standardisation before scoring.
"""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # <repo root>
from scale import negation_scope as NS  # noqa: E402

# PINNED TO THE SWEEP'S OWN CALL, r10_capacity_sweep.py:160 --
# calibrate_bar(n=N_EVAL, s=S, d=D, steps=600, lr=LR, ...) with
# S, D = 64, 24 (:68), N_EVAL = 4096 (:69), LR and D_MODEL from
# m3_capability (:64). A replication that does not reproduce the SHIPPED
# in-sample reading proves nothing about the held-out one.
from scale.m3_capability import LR, D_MODEL  # noqa: E402
STEPS, SEED = 600, 0
S, D, N = 64, 24, 4096


def control(feats, y, *, steps=STEPS, lr=LR, seed=SEED, eval_feats=None, eval_y=None):
    """Returns (in_sample, held_out). `held_out` is None when no eval draw given."""
    g = torch.Generator().manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(feats.shape[-1], 32),
                              torch.nn.GELU(), torch.nn.Linear(32, 1))
    for layer in net:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.normal_(layer.weight, 0.0, 0.5, generator=g)
            torch.nn.init.zeros_(layer.bias)
    mu, sigma = float(y.mean()), float(y.std(unbiased=False))
    if sigma == 0.0:
        sigma = 1.0
    target = (y - mu) / sigma
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        ((net(feats).squeeze(-1) - target) ** 2).mean().backward()
        opt.step()
    with torch.no_grad():
        ins = NS.nrmse(net(feats).squeeze(-1) * sigma + mu, y)
        out = None
        if eval_feats is not None:
            out = NS.nrmse(net(eval_feats).squeeze(-1) * sigma + mu, eval_y)
    return ins, out



def main() -> None:
    """The probe run. Under a `__main__` guard because it TRAINS.

    The first version had the report loop at module level, so importing this
    module ran three 600-step trainings -- about two minutes of compute. It
    WRITES nothing, so `tests/loop/test_no_module_writes_a_file_at_import.py`
    did not flag it: that guard catches filesystem writes and argv reads, not
    expensive computation. Same shape as the three modules repaired earlier in
    this round, one scope over, added by the author of those repairs.
    """
    SHIPPED = {2: 0.013981593578261986}
    print(f"{'t*':>4}  {'in-sample':>11}  {'shipped':>11}  {'held-out':>11}  {'delta':>9}  clause")
    print("-" * 58)
    for t in (2, 8, 32):
        task = f"e3_t{t}"
        batch_fn, oracle_fn, feature_fn = (NS.M3_TASKS[task][0], NS.M3_TASKS[task][1],
                                           NS.M3_TASKS[task][2])
        x, y, f, p = batch_fn(N, S, D, d_model=D_MODEL, seed=SEED)
        xe, ye, fe, pe = batch_fn(N, S, D, d_model=D_MODEL, seed=SEED + 12345)
        feats = feature_fn(x, f, p)
        efeats = feature_fn(xe, fe, pe)
        ins, out = control(feats, y, eval_feats=efeats, eval_y=ye)
        verdict = "PASSES" if out < 1.0 else "FAILS -> BAR BROKEN"
        sh = SHIPPED.get(t)
        shs = f"{sh:.6f}" if sh is not None else "   --   "
        match = "" if sh is None else ("  <- REPRODUCES" if abs(ins - sh) < 1e-9
                                       else f"  <- MISMATCH {ins - sh:+.6f}")
        print(f"{t:>4}  {ins:>11.6f}  {shs:>11}  {out:>11.6f}  {out - ins:>+9.6f}  {verdict}{match}")


if __name__ == "__main__":
    main()

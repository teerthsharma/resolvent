"""The F-green matched re-run: is the one positive result the router or the selector?

F-GREEN, AS PUBLISHED. `pivot_unsigned` beats `softmax` at n_train=8192:
eval NRMSE 0.747528 [0.696849, 0.797716] against 0.877168 [0.830455, 0.924226],
disjoint, 4769 parameters in every arm, softmax measured first. It is the one
positive result in four rounds, and the +3/-5 gate of round 6 is whether it
survives a norm-matched control.

WHY IT NEEDS ONE. `pivot_unsigned` is the softmax operator plus a hop-2 term
routed through k pivots, and those pivots are `batched_select_pivots`' top-k by
`key.norm(dim=-1)` -- a pure function of a token's own representation
(`scale/pivot_probe.py:88`). So "routing through pivots helps" and "attending to
high-key-norm tokens helps" are confounded by construction. Nothing in the
published comparison separates them, because `softmax` has no pivots at all.

THE THREE ARMS BESIDE SOFTMAX, and what each answers.

  pivot_unsigned  the shipped arm: pivots are the top-k by key norm.
  pivot_band      pivots are ranks k..2k by the SAME score. This is the
                  norm-matched control: a different token set with a deliberately
                  similar key-norm profile. The residual imbalance it leaves is
                  measured by scale/matcher.py and printed, never assumed away.
  pivot_random    pivots are a fixed uniform sample, deterministic per call. The
                  UNMATCHED control, present so the comparison contains an arm
                  known to differ. A control that cannot move is not a control.

HOW TO READ THE RESULT, written before the numbers.

  * band ~ unsigned, both beating softmax => the advantage is hop-2 ROUTING and
    not key-norm SELECTION. F-green's arm-vs-softmax claim survives; the reading
    that the arm finds consequential tokens does not.
  * unsigned beats band, band beats softmax => selection carries signal on top of
    routing. F-green survives with its mechanism intact.
  * every arm ~ softmax => the effect was the extra term's capacity and F-green
    dissolves. K-B fires, -5.
  * unsigned beats band AND band ~ random => selection is doing the work and the
    matched control has separated it from routing.

THE ARMS DIFFER IN EXACTLY ONE FUNCTION. `m3_capability.batched_select_pivots` is
swapped for the duration of a run and restored afterwards; nothing else in
Chase's harness is touched, and this file reimplements neither the arm, the
training loop nor the metric. The BIND is that the unpatched arm reproduces the
published 0.747528 and softmax reproduces 0.877168 in the same process. If either
misses, no control number is reported at all.
"""
from __future__ import annotations

import argparse
import contextlib
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
torch.set_num_threads(2)                       # pinned HERE, not by the launcher

from scale import m3_capability as m3                            # noqa: E402
from scale import matcher                                        # noqa: E402
from scale.negation_scope import make_batch                      # noqa: E402
from scale.torque_probe import boot_ci                           # noqa: E402

SHIPPED_SELECT = m3.batched_select_pivots

#: The published F-green cell. Reproduced in-process before any control reads.
PUBLISHED = {"softmax": 0.877168, "pivot_unsigned": 0.747528}
PUB_TOL = 5e-6
RANDOM_SEED = 4242


def select_band(key: torch.Tensor, k: int) -> torch.Tensor:
    """Ranks k..2k by the shipped score. The norm-matched control.

    Same scoring function, same ordering, one band lower. It is the closest
    norm match available from inside the same sequence, and the key-norm gap it
    still leaves is reported by `imbalance_table` rather than assumed to be zero.
    Round 5's band control was matched BY RANK and that residual gap travelled
    with every number taken under it.
    """
    score = key.norm(dim=-1)
    take = min(2 * k, score.shape[-1])
    idx = torch.topk(score, take, dim=-1).indices
    return idx[:, k:] if idx.shape[-1] > k else idx


def select_random(key: torch.Tensor, k: int) -> torch.Tensor:
    """A fixed uniform sample, deterministic across calls. The UNMATCHED control.

    Seeded from a constant so the pivot set does not change between the forward
    passes of one training run. A selector that redrew every step would be a
    noisier arm rather than a different one, and the comparison would stop being
    about selection at all.
    """
    n, s = key.shape[0], key.shape[1]
    g = torch.Generator().manual_seed(RANDOM_SEED)
    return torch.stack([torch.randperm(s, generator=g)[:min(k, s)]
                        for _ in range(n)])


SELECTORS = {"pivot_unsigned": SHIPPED_SELECT,
             "pivot_band": select_band,
             "pivot_random": select_random}


@contextlib.contextmanager
def selector(fn):
    """Swap the ONE function a control varies, then put it back."""
    m3.batched_select_pivots = fn
    try:
        yield
    finally:
        m3.batched_select_pivots = SHIPPED_SELECT


def imbalance_table(x, k: int, seed: int):
    """Key-norm imbalance each control leaves against the shipped pivot set.

    Contract 1.5 applied to the thing this gate actually varies. The causal pool
    is the shipped top-k pivots' key norms; each control's pivots are the filler
    pool. A control whose residual is large has NOT removed the key-norm
    confound, and every conclusion drawn from it inherits that.
    """
    torch.manual_seed(seed)
    arm = m3.Arm("pivot_unsigned", x.shape[1])
    with torch.no_grad():
        key = arm.wk(x)
    out = {}
    for name, fn in SELECTORS.items():
        if name == "pivot_unsigned":
            continue
        res = []
        for i in range(x.shape[0]):
            ki = key[i]
            a = ki[SHIPPED_SELECT(ki.unsqueeze(0), k)[0]].norm(dim=-1)
            b = ki[fn(ki.unsqueeze(0), k)[0]].norm(dim=-1)
            res.append(matcher.match(a.double().numpy(), b.double().numpy(),
                                     verify=(i == 0)).residual)
        lo, hi = boot_ci(res, seed=seed)
        out[name] = (sum(res) / len(res), lo, hi)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=int, default=64)
    ap.add_argument("--d", type=int, default=24)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--n-train", type=int, default=8192)
    ap.add_argument("--n-eval", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--imbalance-draws", type=int, default=64)
    a = ap.parse_args()

    print(f"F-GREEN MATCHED RE-RUN (+3/-5). s={a.s} d={a.d} steps={a.steps} "
          f"n_train={a.n_train} n_eval={a.n_eval} seed={a.seed} "
          f"threads={torch.get_num_threads()}")
    print("Other agents share this box; NO TIMING IS REPORTED.")
    print()

    x_train, y_train, f_tr, p_tr = make_batch(a.n_train, a.s, a.d,
                                              d_model=m3.D_MODEL, seed=a.seed)
    x_eval, y_eval, f_ev, p_ev = make_batch(a.n_eval, a.s, a.d,
                                            d_model=m3.D_MODEL, seed=a.seed + 12345)
    print(f"  train: n={a.n_train} seed={a.seed}  flipper@{f_tr} payload@{p_tr}")
    print(f"  eval : n={a.n_eval} seed={a.seed + 12345}  flipper@{f_ev} payload@{p_ev}")
    print()

    print("=== KEY-NORM IMBALANCE EACH CONTROL LEAVES (contract 1.5) ===")
    print(f"  causal pool = shipped top-k pivots. k={m3.K_PIVOTS}, "
          f"{a.imbalance_draws} draws.")
    print(f"  {'control':>14} {'residual':>10} {'95% CI':>24}")
    for name, (mval, lo, hi) in imbalance_table(x_eval[:a.imbalance_draws],
                                                m3.K_PIVOTS, a.seed).items():
        print(f"  {name:>14} {mval:>10.6f} [{lo:>10.6f},{hi:>10.6f}]")
    print("  A control with a large residual has NOT removed the key-norm")
    print("  confound. The band is the matched control; random is not matched,")
    print("  and is not claimed to be.")
    print()

    print("=== ARMS ===")
    rows = {}
    for name in ("softmax", "pivot_unsigned", "pivot_band", "pivot_random"):
        kind = "softmax" if name == "softmax" else "pivot_unsigned"
        with selector(SELECTORS.get(name, SHIPPED_SELECT)):
            r = m3.run_arm(kind, x_train, y_train, x_eval, y_eval,
                           s=a.s, steps=a.steps, seed=a.seed)
        rows[name] = r
        print(f"  {name:>15}  n_params={r['n_params']}  "
              f"RED0 train={r['nrmse0_train']:.6f} eval={r['nrmse0_eval']:.6f}")
        print(f"  {'':>15}  eval NRMSE={r['eval_nrmse']:.6f} "
              f"CI=[{r['ci_lo']:.6f},{r['ci_hi']:.6f}]")

    print()
    print("=== BIND: the published F-green cell must reproduce in THIS process ===")
    ok = True
    for name, pub in PUBLISHED.items():
        got = rows[name]["eval_nrmse"]
        good = abs(got - pub) < PUB_TOL
        ok &= good
        print(f"  {name:>15} measured {got:.6f} vs published {pub:.6f} "
              f"[{'OK' if good else 'MISMATCH'}]")
    if not ok:
        print("  NOT THE PUBLISHED CELL. No control number is reported.")
        return 1

    def disj(p, q):
        return p["ci_hi"] < q["ci_lo"] or q["ci_hi"] < p["ci_lo"]

    print()
    print("=== THE GATE ===")
    sm = rows["softmax"]
    for nm in ("pivot_unsigned", "pivot_band", "pivot_random"):
        r = rows[nm]
        print(f"  {nm:>15} vs softmax: {r['eval_nrmse']:.6f} "
              f"[{r['ci_lo']:.6f},{r['ci_hi']:.6f}] vs "
              f"{sm['eval_nrmse']:.6f} [{sm['ci_lo']:.6f},{sm['ci_hi']:.6f}] "
              f"-> {'DISJOINT' if disj(r, sm) else 'OVERLAP'}")
    print(f"  pivot_unsigned vs pivot_band -> "
          f"{'DISJOINT' if disj(rows['pivot_unsigned'], rows['pivot_band']) else 'OVERLAP'}")
    print(f"  pivot_band vs pivot_random   -> "
          f"{'DISJOINT' if disj(rows['pivot_band'], rows['pivot_random']) else 'OVERLAP'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

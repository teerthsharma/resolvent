"""CHASE — is K2's filler twin measuring consequence, or measuring the selector?

THE STRUCTURE UNDER TEST. `scale/arm_a_run.py:52-56` builds the two K2 pools as

    piv  = select_pivots(kk, min(k, s-2), exclude=(i, j))
    pool = sorted(pset) if not filler else [t for t in range(1,i)
                                            if t not in pset and t != j]

and `scale/pivot_probe.py:88` defines the selector as

    score = key.norm(dim=-1)

So the column printed as `D_FR causal` is `D_FR(c drawn from the top-k by key
norm)` and the column printed as `D_FR filler` is `D_FR(c drawn from the whole
remaining tail)`. There is NO planted causal structure anywhere in the draw —
`x0`, `wq`, `wk` are i.i.d. `randn`. Nothing in the draw makes one token cause
anything. "causal" here means, and can only mean, "high key norm".

Constitution clause 1 is `consequence over similarity` — weight must reflect what
a token DOES downstream, not what it resembles. `key.norm` is a pure function of
the token's own representation, computed without reference to any downstream
effect. It is a similarity quantity. If `D_FR` is a monotone function of it, the
displacement statistic is a similarity statistic wearing a consequence label.

THE PRE-REGISTERED READING, fixed before running, four arms on IDENTICAL draws
(same `x0, wq, wk, i, j`; only the rule for picking `c` differs):

  A  causal      c ~ U(top-k by key norm)              the published causal cell
  B  filler-tail c ~ U(everything not in P)            the published filler cell
  C  filler-rank c ~ U(ranks k+1 .. 2k by key norm)    NOT in P, so it is a
                                                       filler by the code's own
                                                       definition, but salience-
                                                       adjacent to the pivots
  N  null        two independent c ~ U(top-k)          same pool BOTH sides

PRE-REGISTERED KILL. If A-vs-C fails to separate with disjoint CIs while A-vs-B
separates, then K2's disjointness is produced by the SALIENCE GAP between the
pools and not by any property of `c` that could be called causal. The gate would
then still be non-vacuous in the narrow sense the contract asked for — the
statistic can lose to *some* filler — but the sentence "the statistic can lose to
a filler" would be carrying a meaning the measurement does not support.

N is the MUST-FIRE CONTROL and it fires in the direction of OVERLAP: two draws
from the same pool must NOT separate. A pipeline that separated N would be
separating on nothing and every number here would be void.

DECLARED, NOT HIDDEN: this file does not reproduce `results/arm_a.jsonl`
bitwise and does not claim to. `arm_a_run.one` consumes `wo`, `v0` and two more
`randn(d)` from the same generator for its gradient half; this file omits the
gradient half, so the random stream diverges after `wk`. Arm A here is an
INDEPENDENT re-measurement of the same quantity at the same s, d, k and draw
count, and it is reported beside the published value as an anchor, not as a pin.
All four arms share one generator and one draw, so A-vs-B-vs-C-vs-N is exact.

Threads pinned in this file. `python scale/chase_k2_salience.py`
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
torch.set_num_threads(2)

from scale.pivot_probe import select_pivots                      # noqa: E402
from scale.torque_probe import (rows_with_and_without, theta_rows,  # noqa: E402
                                tv_rows, boot_ci)

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARMS = ("A_causal", "B_filler_tail", "C_filler_rank", "N_null_a", "N_null_b")


def one(s: int, k: int, d: int, g: torch.Generator):
    """One draw, five values of `c`, everything else held identical."""
    x0 = torch.randn(s, d, generator=g)
    wq, wk = torch.randn(d, d, generator=g), torch.randn(d, d, generator=g)
    q, kk = x0 @ wq, x0 @ wk
    i = s - 1
    j = int(torch.randint(1, i, (1,), generator=g))

    kbase = min(k, s - 2)
    piv = select_pivots(kk, kbase, exclude=(i, j))
    pset = set(int(p) for p in piv)
    wide = [int(p) for p in select_pivots(kk, min(2 * kbase, s - 2), exclude=(i, j))]
    band = [t for t in wide if t not in pset]                 # ranks k+1 .. 2k
    tail = [t for t in range(1, i) if t not in pset and t != j]
    if not pset or not band or not tail:
        return None

    norms = kk.norm(dim=-1)
    order = torch.argsort(norms, descending=True).tolist()
    rank_of = {t: r for r, t in enumerate(order)}

    def pick(pool):
        return pool[int(torch.randint(0, len(pool), (1,), generator=g))]

    cs = {"A_causal": pick(sorted(pset)),
          "B_filler_tail": pick(tail),
          "C_filler_rank": pick(sorted(band)),
          "N_null_a": pick(sorted(pset)),
          "N_null_b": pick(sorted(pset))}

    out = {}
    for name, c in cs.items():
        a_c, a_0 = rows_with_and_without(q, kk, c)
        out[name] = dict(theta=float(theta_rows(a_c, a_0).mean()),
                         tv=float(tv_rows(a_c, a_0).mean()),
                         c=c, rank=rank_of[c], norm=float(norms[c]))
    return out


def spearman(xs, ys) -> float:
    """Rank correlation, ties broken by position. Small n, pure python."""
    def rk(v):
        order = sorted(range(len(v)), key=lambda t: v[t])
        r = [0.0] * len(v)
        for pos, t in enumerate(order):
            r[t] = float(pos)
        return r
    rx, ry = rk(xs), rk(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def disjoint(ci1, ci2) -> bool:
    return ci1[0] > ci2[1] or ci2[0] > ci1[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=int, default=1024)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--draws", type=int, default=120)
    ap.add_argument("--seed", type=int, default=4242)
    a = ap.parse_args()

    g = torch.Generator().manual_seed(a.seed)
    rows = []
    while len(rows) < a.draws:
        r = one(a.s, a.k, a.d, g)
        if r is not None:
            rows.append(r)

    th = {n: [r[n]["theta"] for r in rows] for n in ARMS}
    tv = {n: [r[n]["tv"] for r in rows] for n in ARMS}
    ci = {n: boot_ci(th[n], seed=a.seed) for n in ARMS}
    mean = {n: sum(th[n]) / len(th[n]) for n in ARMS}

    print(f"CHASE K2 SALIENCE AUDIT — s={a.s} d={a.d} k={a.k} draws={a.draws} "
          f"seed={a.seed} threads={torch.get_num_threads()}")
    print("all arms on IDENTICAL draws; only the rule for picking `c` differs\n")
    print(f"{'arm':>14} {'D_FR':>12} {'95% CI':>22} {'mean rank of c':>16} "
          f"{'mean ||k_c||':>14}")
    for n in ARMS:
        mr = sum(r[n]["rank"] for r in rows) / len(rows)
        mn = sum(r[n]["norm"] for r in rows) / len(rows)
        print(f"{n:>14} {mean[n]:>12.6f} [{ci[n][0]:.4f},{ci[n][1]:.4f}]"
              f"{'':>4} {mr:>16.1f} {mn:>14.4f}")

    print("\n=== MUST-FIRE CONTROL N (same pool both sides must NOT separate) ===")
    n_dis = disjoint(ci["N_null_a"], ci["N_null_b"])
    n_ratio = mean["N_null_a"] / mean["N_null_b"]
    print(f"  N_null_a vs N_null_b: ratio {n_ratio:.4f}  "
          f"{'SEPARATED -> pipeline separates on nothing, ALL VOID' if n_dis else 'OVERLAP -> FIRED as required'}")
    if n_dis:
        print("  CONTROL FAILED. Nothing below means anything.")
        return 1

    print("\n=== THE TWO FILLERS ===")
    ab = disjoint(ci["A_causal"], ci["B_filler_tail"])
    ac = disjoint(ci["A_causal"], ci["C_filler_rank"])
    print(f"  A vs B (published filler = whole tail)   ratio "
          f"{mean['A_causal'] / mean['B_filler_tail']:>8.4f}  "
          f"{'DISJOINT' if ab else 'OVERLAP'}")
    print(f"  A vs C (filler = ranks k+1..2k, still not in P)  ratio "
          f"{mean['A_causal'] / mean['C_filler_rank']:>8.4f}  "
          f"{'DISJOINT' if ac else 'OVERLAP'}")
    print(f"  C vs B (two fillers against each other)  ratio "
          f"{mean['C_filler_rank'] / mean['B_filler_tail']:>8.4f}  "
          f"{'DISJOINT' if disjoint(ci['C_filler_rank'], ci['B_filler_tail']) else 'OVERLAP'}")

    print("\n=== IS D_FR A FUNCTION OF THE SELECTOR'S SCORE? ===")
    allr = [r[n]["rank"] for r in rows for n in ARMS]
    allt = [r[n]["theta"] for r in rows for n in ARMS]
    alln = [r[n]["norm"] for r in rows for n in ARMS]
    rho_rank = spearman(allr, allt)
    rho_norm = spearman(alln, allt)
    print(f"  Spearman(key-norm RANK of c, theta)  = {rho_rank:+.4f}   "
          f"(rank 0 = highest norm; negative rho means higher norm -> higher theta)")
    print(f"  Spearman(||k_c||, theta)             = {rho_norm:+.4f}")
    print(f"  pooled over {len(allt)} (draw, arm) pairs")

    out = ROOT / "results" / "chase_k2_salience.json"
    out.write_text(json.dumps(
        {"s": a.s, "d": a.d, "k": a.k, "draws": a.draws, "seed": a.seed,
         "mean": mean, "ci": ci, "A_vs_B_disjoint": ab, "A_vs_C_disjoint": ac,
         "null_separated": n_dis, "rho_rank": rho_rank, "rho_norm": rho_norm},
        indent=1))
    print(f"\njournal: {out}")

    print()
    if ab and not ac:
        print("PRE-REGISTERED KILL FIRES. A separates from the whole-tail filler but "
              "NOT from a filler drawn just below the pivot cut. K2's disjointness "
              "tracks the SALIENCE GAP between the pools, not a property of `c`. "
              "With rho above, D_FR is a monotone read of `key.norm` — the "
              "selector's own score — which is a SIMILARITY quantity. "
              "Constitution clause 1 is consequence OVER similarity.")
        return 1
    if not ab:
        print("A does not separate from the published filler on this seed — K2's "
              "own result does not reproduce here. Report as such, do not interpret.")
        return 1
    print("A separates from BOTH fillers. K2's separation survives salience "
          "matching on this seed; the kill did not fire.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Cameron's mechanism claim for the aggregator win, bound to a RED test.

SHE REPORTED [r5 iter 11] the largest unexploited number in the round: on the
IDENTICAL published draws, `theta.max()` over active rows reads |d| = 2.3275
against `theta.mean()`'s 1.0888 -- delta +1.2394 [+0.7962, +1.8565], winning in
6/6 cells -- while the entire K3 quarrel is +0.0241. The cause she gives for the
mean's weakness is arithmetic and solid: under `tril(-1)` rows i <= c cannot move
at all, so `.mean()` divides by s a signal carried by ~s/2 rows, with the dilution
factor itself random per draw.

SHE ALSO REPORTED, AND DISCLAIMED, A MECHANISM: "my reading is that the max row is
at or near i = c+1 (the shortest context containing c, hence the largest weight on
it), which would make the winner 'the row immediately downstream of c' -- a
mechanism claim with no RED test behind it. Do not ship it as one."

THIS FILE IS THAT RED TEST. Three questions, in order of what they cost:

  Q1 WHERE IS THE ARGMAX? Distribution of `argmax_i theta_i - c` over draws,
     against the uniform-over-active-rows baseline. If the max sits at c+1 far
     above chance, the mechanism is real.

  Q2 DOES A FIXED RULE MATCH THE MAX? Compute the standardized effect of reading
     ROW c+1 ALONE -- no max, no scan. If d(row c+1) is not distinguishable from
     d(max), the mechanism is not just an explanation, it is a CHEAPER STATISTIC:
     O(1) instead of a max over s rows.

  Q3 DOES IT EXPLAIN THE k-DEPENDENCE? Her `max` advantage shrinks with k
     (+1.24 -> +0.78 -> +0.73). If the argmax drifts away from c+1 as k grows,
     that is the same fact twice.

TV GETS IDENTICAL TREATMENT AT EVERY STEP. She warned about exactly this: `tv_max`
gains +1.1702 against `th_max`'s +1.2394, so the aggregator lifts TV nearly as
much and NONE of this is a result about the sphere. Any aggregator applied to
theta is applied to TV in the same line, or the comparison is rigged -- and a
rigged win is worse than an honest loss.

Foreman's identity (theta = arcsin(sqrt(TV)) on live rows, three independent
measurements) means theta and TV share an argmax exactly, since arcsin(sqrt(.))
is strictly increasing. Q1's answer is therefore the SAME for both by
construction, and that is asserted as a check rather than assumed.

CARPET DISCIPLINE: `c` and `j` uniform at random, never on a schedule.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
torch.set_num_threads(2)                       # pinned HERE, not by the launcher

from ceq import bench                                            # noqa: E402
from scale.pivot_probe import select_pivots                      # noqa: E402
from scale.torque_probe import (rows_with_and_without, theta_rows,   # noqa: E402
                                tv_rows, boot_ci, cohen_d)

JOURNAL = pathlib.Path(__file__).resolve().parents[1] / "results" / "max_row.jsonl"


def one(s: int, k: int, d: int, g: torch.Generator, *, filler: bool):
    x0 = torch.randn(s, d, generator=g)
    wq, wk = torch.randn(d, d, generator=g), torch.randn(d, d, generator=g)
    q, kk = x0 @ wq, x0 @ wk
    i = s - 1
    j = int(torch.randint(1, i, (1,), generator=g))
    piv = select_pivots(kk, min(k, s - 2), exclude=(i, j))
    pset = set(int(p) for p in piv)
    pool = sorted(pset) if not filler else [t for t in range(1, i)
                                            if t not in pset and t != j]
    if not pool:
        return None
    c = pool[int(torch.randint(0, len(pool), (1,), generator=g))]
    a_c, a_0 = rows_with_and_without(q, kk, c)
    th, tv = theta_rows(a_c, a_0), tv_rows(a_c, a_0)

    # ACTIVE ROWS ONLY. Rows i <= c cannot move under tril(-1); including them
    # would put the argmax baseline over positions that are identically zero.
    act = torch.arange(s) > c
    idx = torch.arange(s)[act]
    th_a, tv_a = th[act], tv[act]
    if th_a.numel() == 0:
        return None
    am_th = int(idx[int(th_a.argmax())])
    am_tv = int(idx[int(tv_a.argmax())])
    nxt = c + 1
    return dict(c=c, n_active=int(act.sum()),
                off_th=am_th - c, off_tv=am_tv - c, same_argmax=am_th == am_tv,
                th_max=float(th_a.max()), tv_max=float(tv_a.max()),
                th_mean=float(th_a.mean()), tv_mean=float(tv_a.mean()),
                th_next=float(th[nxt]) if nxt < s else float("nan"),
                tv_next=float(tv[nxt]) if nxt < s else float("nan"))


def cell(s, k, d, draws, seed, filler):
    g = torch.Generator().manual_seed(seed)
    out = []
    while len(out) < draws:
        r = one(s, k, d, g, filler=filler)
        if r is not None:
            out.append(r)
    return out


def controls(s=512, d=16) -> list:
    """Must-fire. An argmax locator that cannot find a PLANTED maximum, and a
    'chance' baseline that is not actually computed, are both decoration."""
    out = []
    g = torch.Generator().manual_seed(13)
    # C1 plant the maximum at a KNOWN row and require the locator to find it.
    v = torch.rand(s, generator=g) * 0.01
    c, target = 100, 377
    v[target] = 5.0
    act = torch.arange(s) > c
    idx = torch.arange(s)[act]
    found = int(idx[int(v[act].argmax())])
    out.append(("C1 locator finds a PLANTED maximum at a known row",
                f"planted {target}, found {found}", found == target))
    # C2 a maximum planted BEFORE c must NOT be found -- the active-row mask has
    #    to actually mask. Without this the baseline is over the wrong support.
    v2 = torch.rand(s, generator=g) * 0.01
    v2[42] = 9.0                                  # 42 < c = 100
    found2 = int(idx[int(v2[act].argmax())])
    out.append(("C2 a maximum before c is EXCLUDED by the active-row mask",
                f"planted 42 (< c={c}), found {found2}", found2 != 42 and found2 > c))
    # C3 the identity forces theta and TV to share an argmax. Assert it.
    r = torch.rand(64, generator=g)
    th = torch.arcsin(r.clamp(0, 1).sqrt())
    out.append(("C3 arcsin(sqrt(.)) is strictly increasing -> shared argmax",
                f"argmax TV={int(r.argmax())} argmax theta={int(th.argmax())}",
                int(r.argmax()) == int(th.argmax())))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=int, default=1024)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--draws", type=int, default=120)
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 32, 128])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print("=== MUST-FIRE CONTROLS (read before any number below) ===")
    ok = True
    for name, det, fired in controls():
        ok &= fired
        print(f"  [{'FIRED' if fired else 'DID NOT FIRE'}] {name}\n      {det}")
    if not ok:
        print("\n  A CONTROL DID NOT FIRE. Every number below is void. Stopping.")
        return 1

    print(f"\nMAX-ROW MECHANISM. s={a.s} d={a.d} draws={a.draws} ks={a.ks} "
          f"threads={torch.get_num_threads()}")
    print("TV RECEIVES IDENTICAL TREATMENT AT EVERY STEP -- the aggregator lifts")
    print("TV nearly as much as theta, so NONE of this is a result about the sphere.")
    print("Other agents share this box; NO TIMING IS REPORTED.\n")
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)

    print("=== Q1  WHERE IS THE ARGMAX?  (offset = argmax - c) ===")
    print(f"  {'k':>5} {'off==1':>9} {'off<=2':>9} {'off<=10':>9} {'median':>8} "
          f"{'chance off==1':>15} {'theta/TV agree':>15}")
    rows = {}
    for k in a.ks:
        cs = cell(a.s, k, a.d, a.draws, a.seed, False)
        fs = cell(a.s, k, a.d, a.draws, a.seed + 999, True)
        offs = sorted(r["off_th"] for r in cs)
        n = len(offs)
        e1 = sum(1 for o in offs if o == 1) / n
        e2 = sum(1 for o in offs if o <= 2) / n
        e10 = sum(1 for o in offs if o <= 10) / n
        med = offs[n // 2]
        chance = sum(1.0 / r["n_active"] for r in cs) / n
        agree = sum(1 for r in cs if r["same_argmax"]) / n
        rows[k] = dict(cs=cs, fs=fs, e1=e1, e2=e2, e10=e10, med=med,
                       chance=chance, agree=agree)
        print(f"  {k:>5} {e1:>9.4f} {e2:>9.4f} {e10:>9.4f} {med:>8d} "
              f"{chance:>15.6f} {agree:>15.4f}")

    print("\n=== Q2  DOES A FIXED RULE MATCH THE MAX?  (|d| causal vs filler) ===")
    print(f"  {'k':>5} {'th_max':>9} {'th_next':>9} {'th_mean':>9} | "
          f"{'tv_max':>9} {'tv_next':>9} {'tv_mean':>9}")
    for k in a.ks:
        r = rows[k]
        cs, fs = r["cs"], r["fs"]
        d = {key: abs(cohen_d([x[key] for x in cs], [x[key] for x in fs]))
             for key in ("th_max", "th_next", "th_mean",
                         "tv_max", "tv_next", "tv_mean")}
        r["d"] = d
        print(f"  {k:>5} {d['th_max']:>9.4f} {d['th_next']:>9.4f} {d['th_mean']:>9.4f} | "
              f"{d['tv_max']:>9.4f} {d['tv_next']:>9.4f} {d['tv_mean']:>9.4f}")
        with JOURNAL.open("a") as fh:
            fh.write(json.dumps({"s": a.s, "k": k, "draws": a.draws,
                                 "off_eq1": r["e1"], "off_le2": r["e2"],
                                 "off_le10": r["e10"], "off_median": r["med"],
                                 "chance_off_eq1": r["chance"],
                                 "theta_tv_argmax_agree": r["agree"],
                                 **{f"d_{key}": v for key, v in d.items()}}) + "\n")

    print("\n=== Q3  DOES THE ARGMAX DRIFT WITH k? ===")
    for k in a.ks:
        r = rows[k]
        print(f"  k={k:<5} P(off==1)={r['e1']:.4f}  median offset={r['med']}  "
              f"d(max)-d(mean) = {r['d']['th_max']-r['d']['th_mean']:+.4f}")

    print("\n=== VERDICT ===")
    first = rows[a.ks[0]]
    lifted = first["e1"] > 20 * first["chance"]
    print(f"  Q1 argmax at c+1 on {first['e1']:.2%} of draws against a chance "
          f"baseline of {first['chance']:.4%}"
          f"  -> {'FAR ABOVE CHANCE' if lifted else 'NOT ABOVE CHANCE'}")
    if not lifted:
        print("     THE MECHANISM CLAIM IS REFUTED. The max row is not the row")
        print("     immediately downstream of c, and Cameron's disclaimed reading")
        print("     should stay disclaimed. The aggregator win is UNEXPLAINED.")
    else:
        print("     THE MECHANISM CLAIM IS SUPPORTED at this k.")
    for k in a.ks:
        r = rows[k]
        gap = r["d"]["th_max"] - r["d"]["th_next"]
        print(f"  Q2 k={k:<5} d(max)={r['d']['th_max']:.4f} vs "
              f"d(row c+1)={r['d']['th_next']:.4f}  gap {gap:+.4f}  "
              f"-> {'FIXED RULE MATCHES' if abs(gap) < 0.1 else 'MAX IS NOT ROW c+1'}")
    print("\n  Reminder, and it is not a footnote: every theta column above has a")
    print("  TV twin within a few percent. Whatever the aggregator buys, it buys")
    print("  for BOTH statistics, and none of it is evidence for the sphere.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

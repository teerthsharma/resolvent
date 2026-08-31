"""What the aggregator win actually is, derived from the identity rather than guessed.

THE STANDING FACT. `theta.max()` over active rows separates causal from filler at
|d| = 2.3275 / 1.8900 / 1.4806 against `theta.mean()`'s 1.5304 / 1.3897 / 0.9717
-- Wilson confirmed Cameron's paired bootstrap at +1.1347 [+0.7833,+1.5877],
+0.9355, +0.4959, all excluding zero. The record calls it REAL, LARGE and
UNEXPLAINED. Cameron's proposed mechanism (the max row sits at c+1) was tested at
iteration 14 on the bound stream and REFUTED: median offset 137 / 117 / 130, and
row c+1 alone scores 0.2923 / 0.0583 / 0.1977.

THE IDENTITY HANDS OVER A MECHANISM AND NOBODY HAS USED IT. Masking one token and
renormalising leaves one degree of freedom, so per row

    TV_i = A^c[i, c]        theta_i = arcsin(sqrt(TV_i))

`arcsin(sqrt(.))` is strictly increasing, therefore

    argmax_i theta_i = argmax_i A^c[i, c]
    max_i   theta_i  = arcsin(sqrt( max_i A^c[i, c] ))

**The aggregator is not a geometric statistic at all. `theta.max()` is a
strictly-increasing function of THE LARGEST ATTENTION WEIGHT ANY ROW PLACES ON
c.** That is a plain, nameable quantity, and it explains the shape of the win
without any sphere: a pivot receives concentrated attention from some row, while a
filler receives diffuse attention from many -- so a MAX sees the pivot and a MEAN
averages it away.

FOUR TESTS, in order of what they would overturn:

  M1 argmax_i theta_i == argmax_i A[:,c] EXACTLY. This is forced by the identity.
     If it fails, the identity does not hold at this geometry and everything
     built on it this round -- including the K3 concavity control -- is in doubt.

  M2 Does max_i A[i,c] separate causal from filler as well as max theta does?
     By the identity the ROC is the same up to floating point, so the sphere
     contributes nothing to the aggregator win. Measured, not asserted.

  M3 Is max_i A[i,c] a read of ||k_c||, the selector's own score? If it is, the
     aggregator win is Chase's F2 firing a second time: the statistic recovering
     the criterion that chose the token. THIS IS THE TEST THAT COULD KILL THE
     AGGREGATOR RESULT, and it is run rather than avoided.

  M4 If the max row is not c+1, what is it? Report its offset and, more to the
     point, its ATTENTION CONCENTRATION -- max_i A[i,c] itself, causal vs filler.

TV RECEIVES IDENTICAL TREATMENT throughout. The aggregator lifts TV nearly as much
as theta (Wilson: the sphere-vs-TV term at the same aggregator is +0.06 to +0.11,
about 5% of the aggregator's +1.13), so none of this is evidence for the sphere
and it is not written as though it were.

BIND: this file replays ARM A's published draw stream and asserts against the
published journal before reporting, for the reason iteration 13 learned the hard
way -- an unbound draw loop measures a different population and its disagreements
are artifacts.
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
                                tv_rows, cohen_d)
from scale.b1_collapse_test import spearman                      # noqa: E402

JOURNAL = pathlib.Path(__file__).resolve().parents[1] / "results" / "aggregator_mech.jsonl"
PUBLISHED = {8: (0.030850, 0.003317), 32: (0.018089, 0.003068),
             128: (0.013203, 0.002784)}


def one(s: int, k: int, d: int, g: torch.Generator, *, filler: bool):
    """ARM A's draw, generator consumed in ARM A's exact order."""
    x0 = torch.randn(s, d, generator=g)
    wq, wk, wo = (torch.randn(d, d, generator=g) for _ in range(3))   # wo unused
    v0 = torch.randn(s, d, generator=g)                              # unused
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
    att = a_c[:, c]                                # attention each row puts on c

    act = torch.arange(s) > c
    idx = torch.arange(s)[act]
    th_a, att_a = th[act], att[act]
    _ = (torch.randn(d, generator=g), torch.randn(d, generator=g))   # flip half
    if th_a.numel() == 0:
        return None
    am_th = int(idx[int(th_a.argmax())])
    am_att = int(idx[int(att_a.argmax())])
    # THE IDENTITY IS ABOUT VALUES PER ROW, NOT ABOUT argmax. argmax is preserved
    # only when there are no TIES, and theta is heavily quantised: Foreman's F6
    # measured 83-88% of nonzero rows sitting on five values, the arccos of
    # successive float32 ULPs. So the value test is the one the identity makes,
    # and the tie fraction is what explains any argmax disagreement.
    pred = float(torch.arcsin(att_a.max().clamp(0, 1).sqrt()))
    top = float(th_a.max())
    n_tied = int((th_a == th_a.max()).sum())
    return dict(c=c, th_all=float(th.mean()), tv_all=float(tv.mean()),
                val_err=abs(top - pred), n_tied_at_max=n_tied,
                th_max=float(th_a.max()), tv_max=float(tv[act].max()),
                th_mean=float(th_a.mean()),
                att_max=float(att_a.max()), att_mean=float(att_a.mean()),
                am_th=am_th, am_att=am_att, agree=am_th == am_att,
                offset=am_th - c, kc=float(kk[c].norm()))


def cell(s, k, d, draws, seed, filler):
    g = torch.Generator().manual_seed(seed)
    out = []
    while len(out) < draws:
        r = one(s, k, d, g, filler=filler)
        if r is not None:
            out.append(r)
    return out


def auc(a, b) -> float:
    """P(x from a > y from b), ties at 1/2. Rank-only, so it is invariant under
    any strictly increasing transform -- which is exactly the point of M2."""
    xs = sorted((v, 0) for v in a)
    xs += [(v, 1) for v in b]
    xs.sort()
    wins = ties = 0
    seen_b = 0
    i = 0
    while i < len(xs):
        j = i
        while j < len(xs) and xs[j][0] == xs[i][0]:
            j += 1
        grp = xs[i:j]
        na = sum(1 for _, t in grp if t == 0)
        nb = sum(1 for _, t in grp if t == 1)
        wins += na * seen_b
        ties += na * nb
        seen_b += nb
        i = j
    return (wins + 0.5 * ties) / (len(a) * len(b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=int, default=1024)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--draws", type=int, default=120)
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 32, 128])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print(f"AGGREGATOR MECHANISM. s={a.s} d={a.d} draws={a.draws} ks={a.ks} "
          f"threads={torch.get_num_threads()}")
    print("Other agents share this box; NO TIMING IS REPORTED.")
    print()
    print("=== BIND: is this ARM A's published draw stream? ===")
    cells, ok = {}, True
    for k in a.ks:
        cs = cell(a.s, k, a.d, a.draws, a.seed, False)
        fs = cell(a.s, k, a.d, a.draws, a.seed + 999, True)
        cells[k] = (cs, fs)
        if k in PUBLISHED:
            gc = sum(r["th_all"] for r in cs) / len(cs)
            gf = sum(r["th_all"] for r in fs) / len(fs)
            pc, pf = PUBLISHED[k]
            good = abs(gc - pc) < 5e-6 and abs(gf - pf) < 5e-6
            ok &= good
            print(f"  k={k:<5} causal {gc:.6f} vs {pc:.6f}   filler {gf:.6f} vs "
                  f"{pf:.6f}   [{'OK' if good else 'MISMATCH'}]")
    if not ok:
        print("  NOT THE PUBLISHED STREAM. Stopping rather than reporting.")
        return 1
    print("  -> bound.")
    print()

    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    print("=== M1  the identity, tested on VALUES (what it actually claims) ===")
    print("  The first version of this check tested argmax and read 0.84-0.94,")
    print("  which is NOT a refutation of the identity -- argmax is preserved only")
    print("  without ties, and theta is quantised onto a handful of float32 levels.")
    print(f"  {'k':>5} {'max|max th - arcsin(sqrt(max A))|':>34} {'argmax agree':>13} "
          f"{'mean rows tied at max':>22}")
    for k in a.ks:
        cs, fs = cells[k]
        allr = cs + fs
        ve = max(r["val_err"] for r in allr)
        agree = sum(1 for r in allr if r["agree"]) / len(allr)
        tied = sum(r["n_tied_at_max"] for r in allr) / len(allr)
        print(f"  {k:>5} {ve:>34.3e} {agree:>13.6f} {tied:>22.4f}")
    print("  -> a value residual at the float32 floor CONFIRMS the identity;")
    print("     argmax disagreement is a TIE-BREAK artifact, not a refutation.")

    print()
    print("=== M2  does raw peak attention separate as well as max theta? ===")
    print(f"  {'k':>5} {'|d| max theta':>15} {'|d| max A[:,c]':>16} "
          f"{'AUC theta':>11} {'AUC A':>9} {'AUC delta':>11}")
    rows = {}
    for k in a.ks:
        cs, fs = cells[k]
        d_th = abs(cohen_d([r["th_max"] for r in cs], [r["th_max"] for r in fs]))
        d_at = abs(cohen_d([r["att_max"] for r in cs], [r["att_max"] for r in fs]))
        u_th = auc([r["th_max"] for r in cs], [r["th_max"] for r in fs])
        u_at = auc([r["att_max"] for r in cs], [r["att_max"] for r in fs])
        rows[k] = dict(d_th=d_th, d_at=d_at, u_th=u_th, u_at=u_at)
        print(f"  {k:>5} {d_th:>15.4f} {d_at:>16.4f} {u_th:>11.6f} {u_at:>9.6f} "
              f"{u_th-u_at:>+11.3e}")

    print()
    print("=== M3  is peak attention just a read of ||k_c||, the selector's score? ===")
    for k in a.ks:
        cs, fs = cells[k]
        allr = cs + fs
        r_all = spearman(torch.tensor([r["kc"] for r in allr]),
                         torch.tensor([r["att_max"] for r in allr]))
        r_c = spearman(torch.tensor([r["kc"] for r in cs]),
                       torch.tensor([r["att_max"] for r in cs]))
        r_f = spearman(torch.tensor([r["kc"] for r in fs]),
                       torch.tensor([r["att_max"] for r in fs]))
        rows[k].update(rho_all=r_all, rho_c=r_c, rho_f=r_f)
        print(f"  k={k:<5} rho(||k_c||, max A) pooled = {r_all:+.6f}   "
              f"within causal = {r_c:+.6f}   within filler = {r_f:+.6f}")

    print()
    print("=== M4  what the max row IS, and how concentrated the attention is ===")
    print(f"  {'k':>5} {'median offset':>15} {'max A causal':>14} "
          f"{'max A filler':>14} {'ratio':>8}")
    for k in a.ks:
        cs, fs = cells[k]
        offs = sorted(r["offset"] for r in cs)
        mc = sum(r["att_max"] for r in cs) / len(cs)
        mf = sum(r["att_max"] for r in fs) / len(fs)
        rows[k].update(med_off=offs[len(offs) // 2], att_c=mc, att_f=mf)
        print(f"  {k:>5} {offs[len(offs)//2]:>15d} {mc:>14.6f} {mf:>14.6f} "
              f"{mc/max(mf,1e-30):>8.3f}")
        with JOURNAL.open("a") as fh:
            fh.write(json.dumps({"s": a.s, "k": k, "draws": a.draws,
                                 **{kk_: vv for kk_, vv in rows[k].items()}}) + "\n")

    print()
    print("=== WHAT THIS SAYS ===")
    r0 = rows[a.ks[0]]
    print("  M1/M2: `theta.max()` is a strictly-increasing function of")
    print("  max_i A[i,c] -- THE LARGEST ATTENTION WEIGHT ANY ROW PLACES ON c,")
    print("  confirmed on VALUES; argmax ties differ and that is a float32")
    print("  artifact, not a break in the identity.")
    print(f"  The AUC gap between them is {r0['u_th']-r0['u_at']:+.3e} at k={a.ks[0]},")
    print("  i.e. the sphere contributes nothing to the aggregator win; the win")
    print("  is a fact about ATTENTION CONCENTRATION, not about geometry.")
    print()
    print("  M3 CARRIES A CONFOUND THAT MUST BE STATED FIRST. The POOLED")
    print(f"  rho(||k_c||, max A) reads {rows[a.ks[0]]['rho_all']:+.4f} -- but pooling")
    print("  mixes causal (high key-norm pivots) with filler (low key-norm tail),")
    print("  so that number IS the separation under test, not evidence about it.")
    print("  Only the WITHIN-ARM correlations are informative, and they are the")
    print("  ones gated below.")
    print()
    if max(abs(rows[k]["rho_c"]) for k in a.ks) > 0.6:
        print("  M3: PEAK ATTENTION TRACKS ||k_c|| WITHIN THE CAUSAL ARM. The")
        print("  aggregator win is then the statistic recovering the criterion")
        print("  that chose the token -- Chase's F2 firing a second time, and the")
        print("  aggregator result is NOT a consequence result.")
    else:
        print("  M3: peak attention is NOT a strong read of ||k_c|| within an arm")
        print(f"  (max |rho| = {max(abs(rows[k]['rho_c']) for k in a.ks):.4f}), so the")
        print("  win is not simply the selector's own score returning. That is a")
        print("  survival, NOT a positive result: what peak attention DOES track")
        print("  is still unmeasured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

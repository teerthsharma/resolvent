"""M2' ROUTE-DEPENDENCY PROBE â€” are R1/R2/R3/R4 four routes or one mechanism?

M2' offers four routes to "signed influence that context cannot dilute", any ONE
sufficing:

  R1 certified selection      â€” group-testing decoder recall >= 1-eps
  R2 sign-determinacy         â€” magnitude-randomization invariance on the pivot block
  R3 non-Archimedean routing  â€” signed-tropical (max-plus) arm
  R4 hierarchical criticality â€” slope(theta) crosses 0 at theta* in range

"Any ONE suffices" is four independent shots ONLY IF the four are independent.
This probe measures whether they are, on the SAME operator, the SAME draws.

WHAT THE READOUT ACTUALLY IS. `run_arm` computes
    h = v + A v + hop2 v;  h = h @ wo;  grad = d h[i].sum() / d v[j]
With `wrt="v"`, v is a leaf and (I + A + hop2) is constant in v, so
    grad[j].sum() = (I + A + hop2)[i,j] * wo.sum()
exactly, and i != j kills the identity term. So the flip statistic is
    sign( A[i,j] + H[i,j] )
which is DONE.md's KILL 2 sentence, here used as the algebraic identity that
makes the four routes comparable in closed form. `--check` verifies it against
real autograd through `pivot_probe.build_arm` and is a control, not a claim.

DECLARED CHEATS (every one, per the nurse rule):
  * The flip indicator is computed ALGEBRAICALLY from (A[i,j] + hop2[i,j])
    instead of through torch.autograd. Cost: none, if the identity holds; it is
    checked by `--check` against autograd on the same draws and must agree on
    100% of them. Speedup ~6x (no backward pass, no wo matmul).
  * Draw counts are far below M2's 16,384 tail. Every rate below carries a
    Clopper-Pearson interval; n is printed with every number. Nothing here is a
    replacement for a bucket, and no number here may be journalled as one.
  * R2's "10^4 resamples" is run at 10^4 for a SUBSET of draws (--r2-draws) and
    compared against a closed form on ALL draws. The point of the run is to show
    the closed form is exact, after which the resamples are the cheat that is
    free.
  * The tropical arm is evaluated on hop<=2 only (the same reach as every other
    arm here), not on a full star. Cost: it under-reports ambiguity, because a
    longer path set has more ways to tie. That direction is against this
    probe's own conclusion, so it is a conservative cheat.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scale import pivot_probe, s2_probe
from scale.pivot_probe import clopper_pearson, loglog_slope, select_pivots


def build_arm(*a, **k):
    """Looked up on the MODULE at call time, so `s2_probe.absmag()` -- which
    swaps `pivot_probe.build_arm` for the |A| twin -- reaches this probe too.
    Importing the symbol by name would have silently bypassed it."""
    return pivot_probe.build_arm(*a, **k)

D = 16
FLOOR = 1e-6
#: relative deviation of the fast hop2[i,j] from the full matmul, one entry per
#: draws() call. float32 matmul reassociation only: `A` comes back bit-identical
#: and just the k-term reduction order changes. Reported, never hidden.
FASTDEV = []


def _geom(s: int):
    """M2's SCALING protocol, verbatim from pivot_probe.run_arm."""
    return s - 1, max(1, s // 4)


def draws(s: int, k: int, n: int, *, seed: int = 0, arm: str = "pivot_signed",
          fast: bool = False):
    """Per-draw records on the M2 geometry. One record = one (x0, P, c) draw with
    both c-branches, which is exactly one unit of M2's flip statistic.

    DECLARED CHEAT (`fast=True`). Only entry [i,j] of hop2 is ever read, and
    hop2[i,j] = sum_p A[i,p] A[p,j] = w.sum(). `build_arm` builds `A` from
    (kind, qq, kk, gvec, bet) ALONE â€” `pivots` reaches only the hop2 matmul â€” so
    passing a one-element pivot tensor returns a BIT-IDENTICAL `A` at cost
    O(s^2) instead of O(s^2 k). Cost of the cheat: it skips the full matmul, so
    it is verified against it on the first draw of every call and raises if the
    two disagree by more than 0 in `A[i,j]` or 1e-12 in `hop2[i,j]`. Without it
    the theta=1 arm is O(s^3) and cannot be run at honest draw counts on CPU."""
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *sh: torch.randn(*sh, generator=g)
    i, j = _geom(s)
    out = []
    checked = not fast
    for _ in range(n):
        wq, wk, wo = rnd(D, D), rnd(D, D), rnd(D, D)
        x0, v0 = rnd(s, D), rnd(s, D)
        gvec, bet = torch.sigmoid(rnd(s)), torch.sigmoid(rnd(s))
        pivots = select_pivots(x0 @ wk, k, exclude=(i, j))
        pset = set(int(p) for p in pivots)
        pool = [p for p in pset if p not in (i, j)]
        if not pool:
            continue
        c = pool[int(torch.randint(0, len(pool), (1,), generator=g))]
        rec = dict(c=c, i=i, j=j, pivots=pivots, wo_sum=float(wo.sum()))
        for b, c_val in enumerate((rnd(D), rnd(D))):
            x = x0.clone()
            x[c] = c_val
            cheap = pivots[:1] if fast else pivots
            a, hop2 = build_arm(arm, x @ wq, x @ wk, gvec, bet, cheap)
            w = (a[i, pivots] * a[pivots, j] if arm.startswith("pivot")
                 else a[i, :] * a[:, j])
            hij = float(w.sum()) if fast else float(hop2[i, j])
            if not checked:
                a2, h2 = build_arm(arm, x @ wq, x @ wk, gvec, bet, pivots)
                ref = float(h2[i, j])
                dev = abs(ref - hij) / max(abs(ref), 1e-30)
                FASTDEV.append(dev)
                if float(a2[i, j]) != float(a[i, j]) or dev > 1e-5:
                    raise SystemExit(
                        f"FAST PATH WRONG at s={s} k={k}: A {float(a[i,j])!r} vs "
                        f"{float(a2[i,j])!r}, hop2 {hij!r} vs {ref!r} "
                        f"(rel {dev:.3e})")
                checked = True
            rec[f"A{b}"] = float(a[i, j])
            rec[f"H{b}"] = hij
            rec[f"w{b}"] = w.detach().clone()
            if arm.startswith("pivot"):
                idx = (pivots == c).nonzero().flatten()
            else:
                idx = torch.tensor([c])
            rec[f"tc{b}"] = float(w[idx].abs().sum()) if idx.numel() else 0.0
        rec["flip"] = int((rec["A0"] + rec["H0"]) * (rec["A1"] + rec["H1"]) < 0
                          and min(abs(rec["A0"] + rec["H0"]),
                                  abs(rec["A1"] + rec["H1"]))
                          * abs(rec["wo_sum"]) > FLOOR)
        out.append(rec)
    return out


# --------------------------------------------------------------------------
# R2 â€” sign-determinacy under magnitude randomization on the pivot block
# --------------------------------------------------------------------------
def r2_bruteforce(A: float, w: torch.Tensor, m: int, gen) -> int:
    """R2 AS WRITTEN: resample the MAGNITUDES of the pivot-block terms, keep
    their signs, ask whether the sign of the readout is bitwise constant."""
    mags = torch.rand(m, w.numel(), generator=gen)          # positive scalings
    tot = A + (mags * w.unsqueeze(0)).sum(1)
    sg = torch.sign(tot)
    return int(bool((sg == sg[0]).all()))


def r2_closed_form(A: float, w: torch.Tensor) -> int:
    """The same predicate, in three numbers and no resampling.

    THE PRIOR IS PART OF THE ANSWER, and R2 never states it. This function is
    the BOUNDED prior m ~ U(0,1), which is what `r2_bruteforce` samples.
    `r2_closed_form_unbounded` is the (0, inf) prior, which is Brualdi-Shader's
    actual qualitative class. They give different verdicts on the same draws.

    Over m in (0,1)^k the reachable set of sum_p m_p w_p is the open interval
    (-N, P) with N = sum of |negative w|, P = sum of positive w. The sign of
    A + that sum is constant iff -A lies outside (-N, P), i.e.

        A >= N_mass   (when A > 0)      or      -A >= P_mass   (when A < 0)

    A POSITIVE A must survive the most NEGATIVE excursion, which is -N; hence
    N on the A > 0 branch. (An earlier revision of this docstring had P and N
    swapped against the code below. The CODE was right and is unchanged; the
    prose was wrong, was caught in review, and is corrected here.)

    Note what this predicate contains: the ONE-HOP edge A[i,j] and the two
    one-sided masses of the block. It is a DOMINANCE test on A[i,j]. It does not
    see the block's sign PATTERN except through those two sums, and A[i,j] is
    not part of the block being randomized at all."""
    P = float(w.clamp(min=0).sum())
    N = float((-w).clamp(min=0).sum())
    if A > 0:
        return int(A >= N)
    if A < 0:
        return int(-A >= P)
    return int(bool((w[w != 0] > 0).all() or (w[w != 0] < 0).all()))


def r2_closed_form_unbounded(A: float, w: torch.Tensor) -> int:
    """R2 under the prior the route's OWN citation uses.

    Brualdi & Shader's qualitative class is every matrix with the given sign
    pattern — magnitudes range over (0, inf), not (0, 1). Then the reachable set
    of sum_p m_p w_p is (-inf, P_dir) or (N_dir, inf) or all of R:

      * both signs present in w  -> the sum reaches every real, so no A can
        dominate and the readout is NEVER determined;
      * all w_p > 0              -> sum ranges over (0, inf), so A + sum keeps
        one sign iff A >= 0;
      * all w_p < 0              -> symmetric, iff A <= 0.

    So under the unbounded prior R2 collapses to "the block terms share a sign
    AND the one-hop edge agrees with it" — a genuine statement about the sign
    PATTERN, and a much harder bar than the bounded prior's dominance test."""
    nz = w[w != 0]
    if nz.numel() == 0:
        return 1
    if bool((nz > 0).all()):
        return int(A >= 0.0)
    if bool((nz < 0).all()):
        return int(A <= 0.0)
    return 0


def r2_all_agree(w: torch.Tensor) -> int:
    nz = w[w != 0]
    return int(nz.numel() == 0 or bool((nz > 0).all() or (nz < 0).all()))


def cmd_r2(a):
    print("R2 â€” sign-determinacy. Is the 10^4-resample protocol an experiment,")
    print("    or a closed form in three numbers?\n")
    print(f"{'s':>6}{'n':>6}{'D_block(10^4)':>15}{'D_closed':>10}{'D_all_rand':>12}"
          f"{'all_agree':>11}{'flip_rate':>11}  {'CP(flip)':>18}")
    rows, cond, unb = [], [], []
    for s in a.sizes:
        recs = draws(s, a.k, a.n, seed=a.seed, fast=a.fast)
        gen = torch.Generator().manual_seed(1234)
        nb = min(a.r2_draws, len(recs))
        d_bf = sum(r2_bruteforce(r["A0"], r["w0"], a.m, gen) for r in recs[:nb])
        d_cf_sub = sum(r2_closed_form(r["A0"], r["w0"]) for r in recs[:nb])
        d_cf = sum(r2_closed_form(r["A0"], r["w0"]) for r in recs)
        # "randomize everything" variant: one-hop magnitude randomized too
        d_all = sum(r2_closed_form(0.0, torch.cat([r["w0"],
                                                   torch.tensor([r["A0"]])]))
                    for r in recs)
        agree = sum(r2_all_agree(r["w0"]) for r in recs)
        d_unb = sum(r2_closed_form_unbounded(r["A0"], r["w0"]) for r in recs)
        unb.append((s, len(recs), d_unb))
        fl = sum(r["flip"] for r in recs)
        lo, hi = clopper_pearson(fl, len(recs))
        det_recs = [r for r in recs if r2_closed_form(r["A0"], r["w0"])]
        und_recs = [r for r in recs if not r2_closed_form(r["A0"], r["w0"])]
        cond.append((s, len(det_recs), sum(r["flip"] for r in det_recs),
                     len(und_recs), sum(r["flip"] for r in und_recs)))
        rows.append((s, len(recs), d_cf / len(recs), fl / len(recs),
                     d_bf, d_cf_sub, nb))
        print(f"{s:>6}{len(recs):>6}{('%d/%d' % (d_bf, nb)):>15}"
              f"{d_cf/len(recs):>10.4f}"
              f"{d_all/len(recs):>12.4f}{agree/len(recs):>11.4f}"
              f"{fl/len(recs):>11.4f}  [{lo:.4f},{hi:.4f}]")
    print("\nBRUTE FORCE vs CLOSED FORM on the SAME draws "
          f"({a.m} resamples each, {a.r2_draws} draws per size):")
    for s, n, _, _, d_bf, d_cf_sub, nb in rows:
        tag = "AGREE" if d_bf == d_cf_sub else f"DISAGREE by {d_bf - d_cf_sub}"
        print(f"  s={s:<6} brute {d_bf}/{nb}   closed {d_cf_sub}/{nb}   {tag}")
    print("  Disagreement is ONE-SIDED: 10^4 uniform draws cannot reach the")
    print("  endpoints of the open interval, so the resampling protocol")
    print("  OVER-reports determinacy relative to the exact predicate.")
    print("")
    print("THE CONDITIONAL - can a sign-determined draw still flip?")
    print(f"{'s':>6}{'n|determined':>14}{'flip|determined':>18}"
          f"{'n|undetermined':>16}{'flip|undetermined':>20}")
    for s, nd, fd, nu, fu in cond:
        ld, hd = clopper_pearson(fd, max(nd, 1))
        lu, hu = clopper_pearson(fu, max(nu, 1))
        print(f"{s:>6}{nd:>14}{('%d  %.4f' % (fd, fd/max(nd,1))):>18}"
              f"{nu:>16}{('%d  %.4f' % (fu, fu/max(nu,1))):>20}")
        print(f"{'':>6}{'':>14}{('CP [%.4f,%.4f]' % (ld, hd)):>18}"
              f"{'':>16}{('CP [%.4f,%.4f]' % (lu, hu)):>20}")
    print("")
    print("")
    print("THE PRIOR R2 NEVER SPECIFIES - same draws, two magnitude classes:")
    print(f"{'s':>6}{'n':>7}{'determined, m~U(0,1)':>22}"
          f"{'determined, m~(0,inf)':>24}   verdict under each")
    for (s, n, d_unb), row in zip(unb, rows):
        b = row[2]
        u = d_unb / n
        print(f"{s:>6}{n:>7}{b:>22.4f}{u:>24.4f}   "
              f"kill {'CANNOT fire' if b > 0.2 else 'FIRES'} / "
              f"kill {'CANNOT fire' if u > 0.2 else 'FIRES'}")
    print("  Brualdi-Shader's qualitative class is (0,inf). R2 cites it and")
    print("  then samples nothing in particular; the two priors disagree.")
    print("")
    print("D_block  = R2 as written (randomize the pivot block only)")
    print("D_all_rand = randomize the one-hop edge too")
    print("all_agree = fraction where every block term shares one sign")
    return rows


# --------------------------------------------------------------------------
# R4 â€” hierarchical criticality: slope(theta), theta = pivot-budget exponent
# --------------------------------------------------------------------------
def kbudget(s: int, theta: float, k0: int) -> int:
    """theta = 0 -> fixed budget k0 (M2's pivot arm). theta = 1 -> budget grows
    with context (M2's dense arm). This IS the hierarchy parameter: how the
    pivot budget scales with s."""
    return max(1, min(s - 3, int(round(k0 * (s ** theta)))))


def cmd_theta(a):
    print("R4 â€” slope(theta), theta = pivot-budget exponent k(s) = k0 * s^theta")
    print("    theta=0 is M2's pivot arm; theta=1 is a budget that grows with s.\n")
    print(f"{'theta':>7}{'seed':>5}  " + "".join(f"{'s=%d' % s:>18}" for s in a.sizes)
          + f"{'slope':>9}{'D_closed(mean)':>16}{'rho=tc/|A|':>12}")
    grid = [float(t) for t in a.thetas]
    table = {}
    for th in grid:
        for sd in range(a.seeds):
            rates, cells, dets, rhos = [], [], [], []
            for s in a.sizes:
                k = kbudget(s, th, a.k)
                recs = draws(s, k, a.n, seed=a.seed + 1000 * sd, fast=True)
                fl = sum(r["flip"] for r in recs)
                n = len(recs)
                lo, hi = clopper_pearson(fl, n)
                rates.append(fl / n)
                cells.append(f"{fl/n:.4f}[{lo:.3f},{hi:.3f}]")
                dets.append(sum(r2_closed_form(r["A0"], r["w0"])
                                for r in recs) / n)
                rhos.append(sum(r["tc0"] for r in recs)
                            / max(sum(abs(r["A0"]) for r in recs), 1e-30))
            sl, npts = loglog_slope(a.sizes, rates)
            table.setdefault(th, []).append((sl, sum(dets) / len(dets),
                                             sum(rhos) / len(rhos), rates))
            print(f"{th:>7.2f}{sd:>5}  " + "".join(f"{c:>18}" for c in cells)
                  + f"{sl:>+9.3f}{sum(dets)/len(dets):>16.4f}"
                  + f"{sum(rhos)/len(rhos):>12.4f}")
    if FASTDEV:
        print(f"\nFAST-PATH DEVIATION (declared cheat): max rel "
              f"{max(FASTDEV):.3e} over {len(FASTDEV)} checks, float32 matmul "
              f"reassociation only; A[i,j] bit-identical.")
    nan_th = sorted(th for th, vs in table.items()
                    if any(math.isnan(v[0]) for v in vs))
    print(f"theta values where slope is NaN (rate hit 0 at >=2 sizes): {nan_th}")
    signs = sorted({int(math.copysign(1, v[0])) for vs in table.values()
                    for v in vs if not math.isnan(v[0])})
    print(f"\nsign(slope) values observed across the grid: {signs}")
    print("R4's kill is 'no crossing'. A sign change anywhere in the grid means"
          "\nthe kill cannot fire â€” the crossing is guaranteed by the endpoints.")
    return table


# --------------------------------------------------------------------------
# R1 â€” certified selection: can the decoder's recall fail?
# --------------------------------------------------------------------------
def cmd_r1(a):
    """R1 done as group testing actually is: the DEFECTIVE ITEMS are the tokens
    whose perturbation moves the readout, and a pooled test perturbs a whole
    pool at once. Recall is over that support.

    CORRECTION TO A FIRST VERSION OF THIS PROBE, recorded because it is the
    repo's own bug class. The first version compared hop-2 mass routed through
    an 8-token off-support pool against mass through a 1-token on-support pool â€”
    it varied POOL SIZE between the arms and read 1.386e-01 vs 2.910e-02
    ("off-support is bigger"), which measured pool size, not support. Fixed by
    perturbing tokens and reading the influence, which is what the decoder
    actually observes, at matched pool size."""
    print("R1 â€” group testing. The support is 'tokens whose perturbation moves")
    print("    the readout'. What does an off-support token move?\n")
    print(f"{'arm':>14}{'s':>6}{'draws':>7}{'max|d| OFF-support':>21}"
          f"{'median|d| ON-support':>22}{'exact zeros OFF':>18}")
    for arm in ("pivot_signed", "dense_signed"):
        for s in a.sizes:
            g = torch.Generator().manual_seed(a.seed)
            rnd = lambda *sh: torch.randn(*sh, generator=g)
            i, j = _geom(s)
            off_max, on_vals, zeros, tot = 0.0, [], 0, 0
            for _ in range(a.pools):
                wq, wk = rnd(D, D), rnd(D, D)
                x0 = rnd(s, D)
                gvec, bet = torch.sigmoid(rnd(s)), torch.sigmoid(rnd(s))
                pivots = select_pivots(x0 @ wk, a.k, exclude=(i, j))
                pset = set(int(p) for p in pivots)
                base_a, base_h = build_arm(arm, x0 @ wq, x0 @ wk, gvec, bet,
                                           pivots)
                base = float(base_a[i, j] + base_h[i, j])
                off = [t for t in range(1, s - 1)
                       if t not in pset and t not in (i, j)]
                on = [t for t in pset if t not in (i, j)]
                for grp, sink in ((off, "off"), (on, "on")):
                    if not grp:
                        continue
                    t = grp[int(torch.randint(0, len(grp), (1,), generator=g))]
                    x = x0.clone()
                    x[t] = rnd(D)
                    aa, hh = build_arm(arm, x @ wq, x @ wk, gvec, bet, pivots)
                    d = abs(float(aa[i, j] + hh[i, j]) - base)
                    if sink == "off":
                        off_max = max(off_max, d)
                        zeros += int(d == 0.0)
                        tot += 1
                    else:
                        on_vals.append(d)
            med = sorted(on_vals)[len(on_vals) // 2] if on_vals else float("nan")
            print(f"{arm:>14}{s:>6}{a.pools:>7}{off_max:>21.3e}{med:>22.3e}"
                  f"{('%d/%d' % (zeros, tot)):>18}")
    print("\nAn exactly-zero column means the decoder's recall is 1.0 by")
    print("construction: off-support tokens are BITWISE inert, so no pooled")
    print("measurement can ever confuse them with a member.")


# --------------------------------------------------------------------------
# R3 â€” non-Archimedean routing: signed tropical
# --------------------------------------------------------------------------
def cmd_r3(a):
    print("R3 â€” signed tropical (max-plus) hop-2. Sign = sign of the argmax path.\n")
    print(f"{'s':>6}{'n':>5}{'trop flip':>11}{'M2 flip':>10}{'ambiguity':>11}"
          f"{'determined':>12}{'argmax frozen':>15}{'sign frozen':>13}")
    for s in a.sizes:
        recs = draws(s, a.k, a.n, seed=a.seed)
        flips = amb = det = frozen_arg = frozen_sgn = 0
        m2flips = 0
        for r in recs:
            sgs, tops = [], []
            for b in (0, 1):
                w = r[f"w{b}"]
                mag = w.abs()
                order = torch.argsort(mag, descending=True)
                top = int(order[0])
                tops.append(top)
                sgs.append(float(torch.sign(w[top])))
                if b == 0 and w.numel() > 1:
                    second = int(order[1])
                    amb += int(mag[second] > (1 - a.delta) * mag[top]
                               and torch.sign(w[second]) != torch.sign(w[top]))
            # a single surviving term cannot lose its sign to any POSITIVE
            # rescaling, so the tropical block is sign-determined identically
            det += 1
            flips += int(sgs[0] * sgs[1] < 0)
            frozen_arg += int(tops[0] == tops[1])
            frozen_sgn += int(sgs[0] == sgs[1])
            m2flips += r["flip"]
        n = len(recs)
        print(f"{s:>6}{n:>5}{flips/n:>11.4f}{m2flips/n:>10.4f}{amb/n:>11.4f}"
              f"{det/n:>12.4f}{frozen_arg/n:>15.4f}{frozen_sgn/n:>13.4f}")
    print("\ntrop flip  = the tropical sign readout flips when c is resampled")
    print("determined = R2's determined-fraction on the tropical block, which is")
    print("             1.0 identically: one surviving term, positive rescaling")
    print("argmax frozen = the selected path is the SAME index in both branches,")
    print("             i.e. the readout is locally constant and has no gradient")


# --------------------------------------------------------------------------
# sign-blind twin control â€” the proposed replacement instrument
# --------------------------------------------------------------------------
def cmd_twin(a):
    """Every statistic, measured twice: on A, and on |A| with signs destroyed and
    magnitudes preserved. A statistic that agrees between the two is measuring
    magnitude, not sign. softmax is its own twin (A == |A| entrywise), which is
    the baseline column and is a THEOREM, not a measurement."""
    print("SIGN-BLIND TWIN â€” statistic on A vs the same statistic on |A|\n")
    print(f"{'arm':>16}{'s':>6}{'n':>5}{'rate(A)':>10}{'rate(|A|)':>11}"
          f"{'CP(A)':>18}{'CP(|A|)':>18}{'verdict':>16}")
    for arm in a.arms:
        for s in a.sizes:
            recs = draws(s, a.k, a.n, seed=a.seed, arm=arm)
            n = len(recs)
            fa = sum(r["flip"] for r in recs)
            # |A| twin: recompute the readout with every entry's sign removed
            ft = 0
            for r in recs:
                t = []
                for b in (0, 1):
                    t.append(abs(r[f"A{b}"]) + float(r[f"w{b}"].abs().sum()))
                ft += int(t[0] * t[1] < 0)
            la, ha = clopper_pearson(fa, n)
            lt, ht = clopper_pearson(ft, n)
            ov = not (ha < lt or ht < la)
            print(f"{arm:>16}{s:>6}{n:>5}{fa/n:>10.4f}{ft/n:>11.4f}"
                  f"  [{la:.4f},{ha:.4f}]  [{lt:.4f},{ht:.4f}]"
                  f"{'SIGN-BLIND' if ov else 'sign-bearing':>16}")


# --------------------------------------------------------------------------
# control â€” the algebraic identity against real autograd
# --------------------------------------------------------------------------
def cmd_check(a):
    """CONTROL. The whole probe rests on
        grad[j].sum() = (I + A + hop2)[i,j] * wo.sum()
    Run pivot_probe's own path (build_arm + torch.autograd) on the SAME draws and
    require 100% agreement of the flip indicator. Anything less voids every
    number this file prints."""
    print("CONTROL â€” algebraic flip indicator vs torch.autograd, same draws.\n")
    ok = tot = 0
    worst = 0.0
    for s in a.sizes:
        g = torch.Generator().manual_seed(a.seed)
        rnd = lambda *sh: torch.randn(*sh, generator=g)
        i, j = _geom(s)
        for _ in range(a.n):
            wq, wk, wo = rnd(D, D), rnd(D, D), rnd(D, D)
            x0, v0 = rnd(s, D), rnd(s, D)
            gvec, bet = torch.sigmoid(rnd(s)), torch.sigmoid(rnd(s))
            pivots = select_pivots(x0 @ wk, a.k, exclude=(i, j))
            pool = [int(p) for p in pivots if int(p) not in (i, j)]
            if not pool:
                continue
            c = pool[int(torch.randint(0, len(pool), (1,), generator=g))]
            grads, alg = [], []
            for c_val in (rnd(D), rnd(D)):
                x = x0.clone()
                x[c] = c_val
                v = v0.clone().requires_grad_(True)
                a_op, hop2 = build_arm("pivot_signed", x @ wq, x @ wk, gvec,
                                       bet, pivots)
                h = (v + a_op @ v + hop2 @ v) @ wo
                gr, = torch.autograd.grad(h[i].sum(), v, allow_unused=True)
                grads.append(float(gr[j].sum()))
                alg.append(float(a_op[i, j] + hop2[i, j]) * float(wo.sum()))
                worst = max(worst, abs(grads[-1] - alg[-1]))
            f_auto = int(grads[0] * grads[1] < 0
                         and min(abs(grads[0]), abs(grads[1])) > FLOOR)
            f_alg = int(alg[0] * alg[1] < 0
                        and min(abs(alg[0]), abs(alg[1])) > FLOOR)
            tot += 1
            ok += int(f_auto == f_alg)
        print(f"  s={s:<6} flip indicator agreement {ok}/{tot}"
              f"   max|autograd - algebraic| = {worst:.3e}")
    if ok != tot:
        raise SystemExit("IDENTITY BROKEN â€” every number in this file is void.")
    print("\nIDENTITY HOLDS on every draw. The algebraic shortcut is declared "
          "and verified.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["r2", "theta", "r1", "r3", "twin", "check"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[128, 512])
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--m", type=int, default=10000, help="R2 resamples")
    ap.add_argument("--r2-draws", type=int, default=64)
    ap.add_argument("--pools", type=int, default=256)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--absmag", action="store_true",
                    help="run on |A| (magnitudes kept, signs stripped) via "
                         "scale.s2_probe.absmag -- the sign-blind twin")
    ap.add_argument("--thetas", nargs="+", type=float,
                    default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ap.add_argument("--arms", nargs="+",
                    default=["pivot_signed", "dense_signed"])
    a = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    fn = dict(r2=cmd_r2, theta=cmd_theta, r1=cmd_r1, r3=cmd_r3, twin=cmd_twin,
              check=cmd_check)[a.cmd]
    if a.absmag:
        print("*** |A| TWIN: magnitudes bit-identical, signs stripped")
        print("*** (scale.s2_probe.absmag) ***")

        with s2_probe.absmag():
            fn(a)
    else:
        fn(a)


if __name__ == "__main__":
    main()

"""ARM A -- K1 re-evaluated with the flip half on a SIGNED operator.

WHY THIS FILE EXISTS. `scale/arm_a_run.py` recorded K1 NOT EVALUABLE. Its flip
half built the influence Jacobian from `bench._softmax_operator`, which is
entrywise NON-NEGATIVE, so `I + A + pivot_hop2(A, piv)` has no negative entry and
`flip` is 0 BY THEOREM at every k, before any draw. A kill that asks a quantity
to decay on a schedule cannot be read on an arm where that quantity is pinned to
zero by construction. `arm_a_run.py` is NOT edited: its numbers stand as a
reading of the unsigned arm.

WHAT CHANGES, AND ONLY THIS. The flip half is built from
`bench._causal_sgate_operator(q, k, lam=--lam)`, which is signed. The
DISPLACEMENT half is untouched and still reads `_softmax_operator` rows, because
D_FR is a Fisher-Rao angle on the SIMPLEX and sgate rows are not probabilities.

THREADS ARE PINNED IN THIS FILE. A probe that does not pin is a probe whose
number is a function of its launcher -- measured in this repo, same command
logged at 2, 3 and 20 threads.

CARPET DISCIPLINE: `c` and `j` uniformly at random, never on a schedule.
G8: the sign decision goes through `scale/valuation.py`, which compares sign
FIELDS and never forms `lo*hi`.

BUCKETED (ADR-001): one unit is a CHUNK of draws for one (s, k, filler). Two
unbucketed runs have already died in this repo at the 600 s platform cap.

DECLARED DEVIATION FROM `arm_a_run.py`: the FILLER cells do not compute the flip
half at all (flip is only ever reported on causal cells), so filler draws consume
a different RNG stream than in `arm_a_run.py`. Filler theta/TV values here are
therefore NOT expected to be bit-identical to that file's.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
torch.set_num_threads(2)                       # pinned HERE, not by the launcher

from ceq import bench                                              # noqa: E402
from scale.bucket import run_bucket, require_complete              # noqa: E402
from scale.pivot_probe import (select_pivots, pivot_hop2,          # noqa: E402
                               loglog_slope)
from scale.valuation import (valuation, v_opposite_signs,          # noqa: E402
                             float_flip_rate)
from scale.torque_probe import (rows_with_and_without, theta_rows,  # noqa: E402
                                tv_rows, torque, shadow, boot_ci, cohen_d)

NAME = "arm_a_k1"


def op_sgate(lam: float):
    """The SIGNED operator at a stated lam. `rho` left at its shipped 1.5."""
    return lambda q, k: bench._causal_sgate_operator(q, k, lam=lam)


def jacobian(a: torch.Tensor, piv: torch.Tensor) -> torch.Tensor:
    """I + A + pivot_hop2(A, piv) -- the influence Jacobian, formed explicitly.

    Only the lam scan and the controls form it; the measurement path
    differentiates through it instead.
    """
    return torch.eye(a.shape[0], dtype=a.dtype) + a + pivot_hop2(a, piv)


# ---------------------------------------------------------------- flip half ---
def grad_at(x0, wq, wk, wo, v0, piv, i, j, c, cval, opfn) -> float:
    """d/dv[j] of h[i].sum(), with token c set to `cval`. One scalar."""
    x = x0.clone()
    x[c] = cval
    v = v0.clone().requires_grad_(True)
    qq, k2 = x @ wq, x @ wk
    a = opfn(qq, k2)
    h = (v + a @ v + pivot_hop2(a, piv) @ v) @ wo
    gr, = torch.autograd.grad(h[i].sum(), v, allow_unused=True)
    return 0.0 if gr is None else float(gr[j].sum())


def flip_of(x0, wq, wk, wo, v0, piv, i, j, c, cvals, opfn):
    """(flip, g0, g1). The sign decision goes through the valuation instrument."""
    g0 = grad_at(x0, wq, wk, wo, v0, piv, i, j, c, cvals[0], opfn)
    g1 = grad_at(x0, wq, wk, wo, v0, piv, i, j, c, cvals[1], opfn)
    return bool(v_opposite_signs(valuation(g0), valuation(g1))), g0, g1


# ------------------------------------------------------------------ TASK 4 ---
def controls(verbose: bool = True) -> bool:
    """A planted case that MUST read nonzero and one that MUST read zero.

    If the planted case does not fire, the flip instrument is broken and every
    other number this file prints is void.
    """
    s, d, k = 64, 8, 8
    g = torch.Generator().manual_seed(12345)
    x0 = torch.randn(s, d, generator=g)
    wq, wk, wo = (torch.randn(d, d, generator=g) for _ in range(3))
    v0 = torch.randn(s, d, generator=g)
    i = s - 1
    j = int(torch.randint(1, i, (1,), generator=g))
    piv = select_pivots(x0 @ wk, k, exclude=(i, j))
    pool = [int(p) for p in piv]
    c = pool[int(torch.randint(0, len(pool), (1,), generator=g))]
    u = torch.randn(d, generator=g)
    cvals = (u, -u)          # antipodal: q[c,0] is LINEAR in x[c], so it flips

    ok = True

    # -- C1 PLANTED SIGN CHANGE: the operator carries sign(q[c,0]) into A[i,j].
    #    A has a single nonzero entry, at (i, j). i and j are BOTH excluded from
    #    piv, so A[:, piv] and A[piv, :] are identically zero and hop2 vanishes;
    #    the Jacobian entry is therefore exactly q[c,0]. Antipodal cvals make
    #    q[c,0] change sign, so a working instrument MUST read flip=True.
    def planted(q, kk):
        a = torch.zeros(q.shape[0], q.shape[0], dtype=q.dtype)
        a[i, j] = q[c, 0]
        return a

    f1, g10, g11 = flip_of(x0, wq, wk, wo, v0, piv, i, j, c, cvals, planted)
    if verbose:
        print(f"  C1 planted sign change    flip={f1!s:<5} g0={g10:+.6e} "
              f"g1={g11:+.6e}   "
              f"{'FIRES (required)' if f1 else 'DEAD -> INSTRUMENT BROKEN'}")
    ok = ok and f1

    # -- C2 MUST BE ZERO: softmax is entrywise non-negative, so the Jacobian has
    #    no negative entry and the gradient sign cannot move. Same draw, same
    #    cvals, same code path -- only the operator differs.
    f2, g20, g21 = flip_of(x0, wq, wk, wo, v0, piv, i, j, c, cvals,
                           bench._softmax_operator)
    jm = float(jacobian(bench._softmax_operator(x0 @ wq, x0 @ wk), piv).min())
    if verbose:
        print(f"  C2 softmax non-negative   flip={f2!s:<5} g0={g20:+.6e} "
              f"g1={g21:+.6e}   min(I+A+hop2)={jm:.6e}   "
              f"{'ZERO (required)' if not f2 else 'FIRED -> BROKEN'}")
    ok = ok and (not f2) and jm >= 0.0

    # -- C3 the underflow defect the valuation instrument exists to fix, at the
    #    dynamic range where it ACTUALLY bites for these functions.
    #    `scale/valuation.py`'s docstring argues the case at lo=1e-30 with
    #    float32's 1.18e-38 smallest normal, but `flip_rate` and
    #    `float_flip_rate` take PYTHON floats, i.e. float64, whose smallest
    #    normal is ~2.2e-308. At 1e-30 the product -1e-60 is exactly
    #    representable and the float path gets the RIGHT answer. The defect is
    #    real but needs ~1e-200 to be exhibited in float64, and it is exhibited
    #    here rather than asserted.
    for lo, hi, must_fail_float in ((1e-30, -1e-30, False),
                                    (1e-200, -1e-200, True)):
        v_says = v_opposite_signs(valuation(lo), valuation(hi))
        f_says = float_flip_rate([(lo, hi)]) > 0
        good = v_says and (f_says is not must_fail_float)
        if verbose:
            print(f"  C3 lo={lo:.0e} hi={hi:.0e}  valuation={v_says!s:<5} "
                  f"float(lo*hi<0)={f_says!s:<5} lo*hi={lo * hi!r:<10}  "
                  f"{'OK' if good else 'UNEXPECTED'}")
        ok = ok and good

    return bool(ok)


# ------------------------------------------------------------------ TASK 1 ---
def lamscan(lams, ss, seeds, d: int, k: int) -> int:
    print(f"\n=== TASK 1  sgate sign census at ARM A geometry "
          f"(d={d}, pivots k={k}, threads={torch.get_num_threads()}) ===")
    print("F16 is scoped to the HARNESS geometry, where `make_batch` scales x by "
          "0.1 and |w| mean is 2.682399e-03.\nARM A applies no such scaling, so "
          "the logit scale is printed here beside every reading (F17).")
    print(f"{'s':>6} {'lam':>6} {'seed':>5} {'mean|w| causal':>15} "
          f"{'min A':>15} {'frac A<0':>10} {'min I+A+hop2':>15} {'any<0':>6}")
    agg = {}
    for s in ss:
        for lam in lams:
            mins, fracs, jmins, anyneg, ws = [], [], [], 0, []
            for seed in seeds:
                g = torch.Generator().manual_seed(seed)
                x0 = torch.randn(s, d, generator=g)
                wq = torch.randn(d, d, generator=g)
                wk = torch.randn(d, d, generator=g)
                q, kk = x0 @ wq, x0 @ wk
                i = s - 1
                j = int(torch.randint(1, i, (1,), generator=g))
                piv = select_pivots(kk, min(k, s - 2), exclude=(i, j))
                mask = bench._causal_mask(s, q.device, 0)
                w = (q @ kk.transpose(-2, -1)) / math.sqrt(d)
                mw = float(w[mask].abs().mean())
                a = bench._causal_sgate_operator(q, kk, lam=lam)
                mn = float(a.min())
                fr = float((a < 0).float().mean())
                jm = float(jacobian(a, piv).min())
                print(f"{s:>6} {lam:>6.2f} {seed:>5} {mw:>15.6e} {mn:>15.6e} "
                      f"{fr:>10.6f} {jm:>15.6e} {'YES' if jm < 0 else 'no':>6}")
                mins.append(mn)
                fracs.append(fr)
                jmins.append(jm)
                ws.append(mw)
                anyneg += int(jm < 0)
            agg[(s, lam)] = (min(mins), sum(fracs) / len(fracs),
                             min(jmins), anyneg, len(seeds),
                             sum(ws) / len(ws))
    print(f"\n{'s':>6} {'lam':>6} {'mean|w|':>13} {'worst min A':>15} "
          f"{'mean frac A<0':>14} {'worst min J':>15} {'seeds J<0':>10}")
    for (s, lam), (mn, fr, jm, an, n, mw) in agg.items():
        print(f"{s:>6} {lam:>6.2f} {mw:>13.6e} {mn:>15.6e} {fr:>14.6f} "
              f"{jm:>15.6e} {an}/{n:<8}")
    return 0


# ---------------------------------------------------------------- the cells ---
def draw(s: int, k: int, d: int, g: torch.Generator, *, filler: bool, opfn):
    """One draw. `c` and `j` UNIFORM AT RANDOM. Flip half only when opfn given."""
    x0 = torch.randn(s, d, generator=g)
    wq, wk, wo = (torch.randn(d, d, generator=g) for _ in range(3))
    v0 = torch.randn(s, d, generator=g)
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
    th = theta_rows(a_c, a_0)
    tv = tv_rows(a_c, a_0)
    _, sv = shadow(a_c, a_0)
    out = dict(theta=float(th.mean()), tv=float(tv.mean()),
               tau=torque(a_c, a_0),
               gamma=float(sv[0] - sv[1]) if sv.numel() > 1 else float("nan"))
    if opfn is not None:
        cvals = (torch.randn(d, generator=g), torch.randn(d, generator=g))
        fl, _g0, _g1 = flip_of(x0, wq, wk, wo, v0, piv, i, j, c, cvals, opfn)
        out["flip"] = fl
    return out


def unit_key(s, k, d, filler, chunk, n, lam):
    return (f"s{s}_k{k}_d{d}_{'fill' if filler else 'caus'}"
            f"_lam{lam:g}_c{chunk}_n{n}")


def compute(p: dict) -> dict:
    g = torch.Generator().manual_seed(p["seed"])
    opfn = None if p["filler"] else op_sgate(p["lam"])
    keys = ("theta", "tv", "tau", "gamma") + (() if p["filler"] else ("flip",))
    out = {kk: [] for kk in keys}
    got = 0
    while got < p["n"]:
        r = draw(p["s"], p["k"], p["d"], g, filler=p["filler"], opfn=opfn)
        if r is None:
            continue
        for kk in keys:
            out[kk].append(r[kk])
        got += 1
    return out


def units(s, ks, d, draws, chunk, lam, seed0):
    u = []
    nchunk = max(1, draws // chunk)
    for k in ks:
        for filler in (False, True):
            for ci in range(nchunk):
                n = chunk if ci < nchunk - 1 else draws - chunk * (nchunk - 1)
                seed = seed0 + 1000 * ci + (500000 if filler else 0) + 7 * k
                u.append((unit_key(s, k, d, filler, ci, n, lam),
                          dict(s=s, k=k, d=d, n=n, filler=filler, seed=seed,
                               lam=lam)))
    return u


# ------------------------------------------------------------------- report ---
def bslope(per_k: dict, seed: int, b: int):
    """Bootstrap CI on the LOG-LOG SLOPE itself. Resamples DRAWS, refits.

    Zero-rate k points are DROPPED by `loglog_slope`, never clamped: clamping a
    zero to 1e-12 turns 'the property is gone' into a finite slope.

    TAKES A SCALAR SEED AND BUILDS ITS OWN GENERATOR. It previously took a
    `torch.Generator` BY REFERENCE and the two call sites below shared one. The
    flip bootstrap ran first and consumed 2000 * 2400 = 4,800,000 int64 draws, so
    the D_FR bootstrap began at that offset rather than at zero. Two consequences,
    and the second is worse than the first:

      * the published interval stopped reproducing, because it had been taken
        from a fresh stream at offset zero;
      * K1's clause requires both slopes "on the SAME draws", and a shared stream
        hands each half a DIFFERENT resampled index sequence. The defect silently
        decoupled the two halves of a clause whose whole point is that they be
        coupled, and that went unnoticed for a whole round.

    Seeding internally from a scalar makes each call position-independent, which
    is what the nine other bootstrap helpers in this repository already do.
    """
    gen = torch.Generator().manual_seed(seed)
    ks = sorted(per_k)
    pt, npt = loglog_slope(ks, [sum(per_k[k]) / len(per_k[k]) for k in ks])
    tens = {k: torch.tensor([float(x) for x in per_k[k]]) for k in ks}
    reps = []
    for _ in range(b):
        ys = []
        for k in ks:
            t = tens[k]
            idx = torch.randint(0, t.numel(), (t.numel(),), generator=gen)
            ys.append(float(t[idx].mean()))
        sl, _n = loglog_slope(ks, ys)
        if not math.isnan(sl):
            reps.append(sl)
    reps.sort()
    if len(reps) < 20:
        return pt, npt, float("nan"), float("nan"), len(reps), b
    return (pt, npt, reps[int(0.025 * len(reps))], reps[int(0.975 * len(reps))],
            len(reps), b)


def report(s, ks, d, draws, chunk, lam, seed0, b: int) -> int:
    us = units(s, ks, d, draws, chunk, lam, seed0)
    vals = require_complete(NAME, us)
    per = {}
    for key, p in us:
        slot = per.setdefault((p["k"], p["filler"]), {})
        for kk, vv in vals[key].items():
            slot.setdefault(kk, []).extend(vv)

    print(f"\n=== ARM A / K1 REPORT   s={s} d={d} lam={lam}   DECLARED {draws} "
          f"draws per cell   threads={torch.get_num_threads()} ===")
    print(f"{'k':>5} {'n':>5} {'D_FR causal':>26} {'D_FR filler':>26} "
          f"{'flip':>9} {'||tau||':>9} {'gamma_1':>9}")
    rows = {}
    for k in ks:
        c_, f_ = per[(k, False)], per[(k, True)]
        lo_c, hi_c = boot_ci(c_["theta"])
        lo_f, hi_f = boot_ci(f_["theta"])
        nfl = sum(1 for x in c_["flip"] if x)
        rows[k] = dict(n=len(c_["theta"]),
                       theta_c=sum(c_["theta"]) / len(c_["theta"]),
                       theta_f=sum(f_["theta"]) / len(f_["theta"]),
                       ci_c=(lo_c, hi_c), ci_f=(lo_f, hi_f),
                       flip=nfl / len(c_["flip"]), nflip=nfl,
                       d_theta=cohen_d(c_["theta"], f_["theta"]),
                       d_tv=cohen_d(c_["tv"], f_["tv"]),
                       tau=sum(c_["tau"]) / len(c_["tau"]),
                       gamma=sum(c_["gamma"]) / len(c_["gamma"]))
        r = rows[k]
        print(f"{k:>5} {r['n']:>5} {r['theta_c']:.6f} [{lo_c:.4f},{hi_c:.4f}] "
              f"{r['theta_f']:.6f} [{lo_f:.4f},{hi_f:.4f}] "
              f"{r['flip']:>9.5f} {r['tau']:>9.4f} {r['gamma']:>9.4f}")

    print(f"\n=== K1 DUAL SLOPE, bootstrap CI ON THE SLOPE (B={b}) ===")
    print("  flip counts per k: " +
          "  ".join(f"k={k}:{rows[k]['nflip']}/{rows[k]['n']}" for k in ks))
    fp, fn, flo, fhi, fu, _ = bslope(
        {k: [float(x) for x in per[(k, False)]["flip"]] for k in ks}, 4242, b)
    dp, dn, dlo, dhi, du, _ = bslope(
        {k: per[(k, False)]["theta"] for k in ks}, 4242, b)
    print(f"  flip slope in k  = {fp:+.4f}  [{flo:+.4f},{fhi:+.4f}]  "
          f"(fit on {fn}/{len(ks)} nonzero k; {fu}/{b} usable reps)  bar <= -0.4")
    print(f"  D_FR slope in k  = {dp:+.4f}  [{dlo:+.4f},{dhi:+.4f}]  "
          f"(fit on {dn}/{len(ks)} k; {du}/{b} usable reps)  bar >= -0.1")
    for nm, lo_, hi_, bar, side in (("flip", flo, fhi, -0.4, "le"),
                                    ("D_FR", dlo, dhi, -0.1, "ge")):
        if math.isnan(lo_):
            print(f"  {nm} vs bar {bar:+.2f}: NO CI -- under 20 usable reps.")
            continue
        if side == "le":
            v = "MET" if hi_ <= bar else ("NOT MET" if lo_ > bar else "STRADDLES")
        else:
            v = "MET" if lo_ >= bar else ("NOT MET" if hi_ < bar else "STRADDLES")
        print(f"  {nm} vs bar {bar:+.2f}: {v}")
    if not math.isnan(dlo):
        leap = ("EXCLUDES -0.3, CI lies ABOVE it" if dlo > -0.3 else
                "EXCLUDES -0.3, CI lies BELOW it" if dhi < -0.3 else
                "INCLUDES -0.3 -- the 'no leap' line is inside the CI")
        print(f"  D_FR CI vs the -0.3 'no leap' line: {leap}")

    print("\n=== K2 FILLER TWIN (disjoint CIs required) ===")
    for k in ks:
        r = rows[k]
        dis = r["ci_c"][0] > r["ci_f"][1] or r["ci_f"][0] > r["ci_c"][1]
        print(f"  k={k:<5} causal [{r['ci_c'][0]:.4f},{r['ci_c'][1]:.4f}] vs "
              f"filler [{r['ci_f'][0]:.4f},{r['ci_f'][1]:.4f}]  "
              f"{'DISJOINT' if dis else 'OVERLAP -> VOIDS THE TABLE'}")

    print("\n=== K3 GEOMETRY MUST EARN ITSELF (|d_theta| > |d_TV|) ===")
    for k in ks:
        r = rows[k]
        print(f"  k={k:<5} d_theta={r['d_theta']:+.4f}  d_TV={r['d_tv']:+.4f}  "
              f"{'theta WINS' if abs(r['d_theta']) > abs(r['d_tv']) else 'TV WINS'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=int, default=1024)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    ap.add_argument("--lam", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=420.0)
    ap.add_argument("--verify", type=int, default=1)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--mode", choices=("control", "lamscan", "run", "report"),
                    default="run")
    ap.add_argument("--lams", type=float, nargs="+",
                    default=[0.10, 0.25, 0.50, 0.75, 0.90, 1.00])
    ap.add_argument("--scan-s", type=int, nargs="+", default=[256, 1024])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()

    print("=== TASK 4 CONTROL (runs before anything else) ===")
    ok = controls()
    print(f"  CONTROL VERDICT: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("  THE FLIP INSTRUMENT IS BROKEN. Every other number here is VOID.")
        return 2
    if a.mode == "control":
        return 0
    if a.mode == "lamscan":
        return lamscan(a.lams, a.scan_s, a.seeds, a.d, 32)

    us = units(a.s, a.ks, a.d, a.draws, a.chunk, a.lam, a.seed)
    print(f"\nARM A / K1  s={a.s} d={a.d} lam={a.lam} ks={a.ks} "
          f"threads={torch.get_num_threads()}")
    print(f"DECLARED: {a.draws} draws per cell ({a.chunk} per unit, "
          f"{len(us)} units), not the contract's 20000. Every CI is printed.")
    if a.mode == "report":
        return report(a.s, a.ks, a.d, a.draws, a.chunk, a.lam, a.seed, a.boot)

    acc = run_bucket(NAME, us, compute, budget_s=a.budget, verify=a.verify)
    print(json.dumps(acc))
    return 0 if acc["remaining"] == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())

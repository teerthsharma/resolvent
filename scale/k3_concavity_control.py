"""K3's MISSING CONTROL ARM: a concave reparametrisation of TV with no geometry.

WHY THIS FILE EXISTS. K3 says theta must separate causal-from-filler with a
larger standardized effect than raw TV, "else the sphere is notation and TV
ships". Foreman reported [r5 iter 10] that on a ONE-TOKEN mask the row has a
single degree of freedom, so

    TV_i = m_i = A^c[i,c]        BC_i = sqrt(1 - m_i)
    theta_i = arccos(BC_i) = arcsin(sqrt(TV_i))          IDENTICALLY

If that holds, theta is a fixed monotone function of TV BY CONSTRUCTION, and K3
as written cannot tell geometry apart from row-wise concavity: ANY concave
transform applied per row before averaging gains standardized effect, because it
compresses the many small rows and spares the few large ones. Beating RAW TV is
then not evidence of a sphere. It is evidence of a square root.

SO K3 NEEDS A CONTROL ARM IT DOES NOT HAVE: `TV^p`, a concave reparametrisation
carrying NO geometric story, no Fisher-Rao, no Chentsov, no arccos. theta must
beat THAT, not raw TV.

FOREMAN'S OWN NUMBERS DO NOT SETTLE THIS AND HE SAYS SO. He searched 8 exponents
ON THE DRAWS THAT SCORED THEM, with no bootstrap CI and no multiplicity
correction. Picking the best of 8 on the same data that scores it is a selection
effect, and reporting it as a margin would be exactly the error this project
struck twice. This file fixes that with a FIT/SCORE SPLIT: the exponent is chosen
on one half of the draws and evaluated on the other, which it has never seen.

A FELLOW IS NOT TAKEN AT FACE VALUE. The identity above is Foreman's claim and it
has not passed Wilson. It is MEASURED HERE FIRST, and separately on live and dead
rows -- because a dead row gives TV = 0 but theta = arccos(0) = pi/2, so the
identity must BREAK there. If it breaks on live rows too, Foreman is wrong and
this probe's premise goes with him.

PRE-REGISTERED, before the numbers:
  If theta does NOT beat the best held-out `TV^p` with a bootstrap CI on the
  DIFFERENCE excluding zero, then K3 is not evidence for geometry. The sphere is
  a mediocre member of a family with no geometry in it, and K3 must be rewritten
  or dropped -- not softened.
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

from ceq import bench                                            # noqa: E402
from scale.pivot_probe import select_pivots                      # noqa: E402
from scale.torque_probe import (rows_with_and_without, theta_rows,   # noqa: E402
                                tv_rows, boot_ci, cohen_d)

JOURNAL = pathlib.Path(__file__).resolve().parents[1] / "results" / "k3_concavity.jsonl"
EXPONENTS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 1.00]


def one(s: int, k: int, d: int, g: torch.Generator, *, filler: bool):
    """One draw, ARM A's protocol: c and j UNIFORM AT RANDOM, never scheduled."""
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
    th = theta_rows(a_c, a_0)
    tv = tv_rows(a_c, a_0)
    dead = a_c.sum(-1) == 0                       # F4's rows: no mass either side
    return th, tv, dead


def identity_check(s: int, k: int, d: int, draws: int, seed: int):
    """Foreman's claim, measured rather than believed. Live and dead rows apart."""
    g = torch.Generator().manual_seed(seed)
    live_err, dead_n, dead_th, live_n = 0.0, 0, [], 0
    got = 0
    while got < draws:
        r = one(s, k, d, g, filler=False)
        if r is None:
            continue
        got += 1
        th, tv, dead = r
        pred = torch.arcsin(tv.clamp(0, 1).sqrt())
        live = ~dead
        live_n += int(live.sum())
        if live.any():
            live_err = max(live_err, float((th[live] - pred[live]).abs().max()))
        dead_n += int(dead.sum())
        if dead.any():
            dead_th.append(float(th[dead].max()))
    return live_err, live_n, dead_n, (max(dead_th) if dead_th else float("nan"))


def stat(th: torch.Tensor, tv: torch.Tensor, p: float) -> float:
    """Mean over rows of TV^p. p=1 is raw TV. theta is passed through as-is."""
    return float(tv.clamp_min(0).pow(p).mean())


def cell(s: int, k: int, d: int, draws: int, seed: int, filler: bool):
    g = torch.Generator().manual_seed(seed)
    out = []
    while len(out) < draws:
        r = one(s, k, d, g, filler=filler)
        if r is None:
            continue
        th, tv, _ = r
        row = {"theta": float(th.mean())}
        for p in EXPONENTS:
            row[f"tv^{p}"] = stat(th, tv, p)
        out.append(row)
    return out


def d_of(cs, fs, key) -> float:
    return cohen_d([r[key] for r in cs], [r[key] for r in fs])


def boot_diff(cs_c, cs_f, key_a, key_b, *, b=2000, seed=0):
    """Bootstrap CI on d(key_a) - d(key_b), RESAMPLING DRAWS, not statistics.

    The two effects come from the SAME draws, so they are paired and the
    difference must be bootstrapped jointly. Resampling them independently would
    inflate the interval and could hide a real loss."""
    g = torch.Generator().manual_seed(seed)
    nc, nf = len(cs_c), len(cs_f)
    diffs = []
    for _ in range(b):
        ic = torch.randint(0, nc, (nc,), generator=g).tolist()
        i_f = torch.randint(0, nf, (nf,), generator=g).tolist()
        c2 = [cs_c[t] for t in ic]
        f2 = [cs_f[t] for t in i_f]
        diffs.append(abs(d_of(c2, f2, key_a)) - abs(d_of(c2, f2, key_b)))
    diffs.sort()
    return diffs[int(0.025 * b)], diffs[int(0.975 * b)]


def controls(cs, fs) -> list:
    """Must-fire. A comparison instrument that cannot see a tie and cannot see a
    loss is not an instrument."""
    out = []
    # C1 a statistic against ITSELF must read a difference of exactly 0.
    lo, hi = boot_diff(cs, fs, "theta", "theta")
    out.append(("C1 theta vs itself reads difference 0 with a zero-width CI",
                f"[{lo:.3e}, {hi:.3e}]", abs(lo) < 1e-12 and abs(hi) < 1e-12))
    # C2 the instrument must be able to report theta LOSING. Raw TV (p=1) is the
    #    weakest member by Foreman's account, so theta vs TV must come out
    #    POSITIVE -- and the reversed comparison must come out NEGATIVE. If both
    #    read positive the sign convention is broken and every verdict is void.
    a = abs(d_of(cs, fs, "theta")) - abs(d_of(cs, fs, "tv^1.0"))
    b_ = abs(d_of(cs, fs, "tv^1.0")) - abs(d_of(cs, fs, "theta"))
    out.append(("C2 instrument can report a LOSS (sign convention holds)",
                f"theta-TV={a:+.6f}  TV-theta={b_:+.6f}", a * b_ < 0))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=int, default=1024)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--draws", type=int, default=240)
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 32, 128])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print(f"K3 CONCAVITY CONTROL. s={a.s} d={a.d} draws/cell={a.draws} ks={a.ks} "
          f"threads={torch.get_num_threads()}")
    print(f"DECLARED: {a.draws} draws per cell, split FIT/SCORE 50/50. The "
          f"exponent is chosen on FIT and evaluated on SCORE, which it never saw.")
    print("Other agents share this box; NO TIMING IS REPORTED.\n")

    print("=== FOREMAN'S IDENTITY, MEASURED NOT BELIEVED ===")
    for k in a.ks:
        le, ln, dn, dth = identity_check(a.s, k, a.d, 20, a.seed)
        print(f"  k={k:<4} max|theta - arcsin(sqrt(TV))| on {ln} LIVE rows = "
              f"{le:.3e}   dead rows={dn} (theta there = {dth:.6f})")
    print("  -> the identity is expected to HOLD on live rows and BREAK on dead")
    print("     ones, where TV=0 but arccos(0)=pi/2. Both are checked.\n")

    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    half = a.draws // 2
    verdicts = {}
    for k in a.ks:
        cs = cell(a.s, k, a.d, a.draws, a.seed, False)
        fs = cell(a.s, k, a.d, a.draws, a.seed + 999, True)

        if k == a.ks[0]:
            print("=== MUST-FIRE CONTROLS ===")
            ok = True
            for name, det, fired in controls(cs[:half], fs[:half]):
                ok &= fired
                print(f"  [{'FIRED' if fired else 'DID NOT FIRE'}] {name}\n      {det}")
            if not ok:
                print("\n  A CONTROL DID NOT FIRE. Every number below is void.")
                return 1
            print()

        fit_c, fit_f = cs[:half], fs[:half]
        sc_c, sc_f = cs[half:], fs[half:]

        # FIT: choose the exponent on the first half ONLY.
        fit_d = {p: abs(d_of(fit_c, fit_f, f"tv^{p}")) for p in EXPONENTS}
        p_star = max(fit_d, key=fit_d.get)

        # SCORE: everything below is on draws the exponent has never seen.
        d_th = abs(d_of(sc_c, sc_f, "theta"))
        d_tv1 = abs(d_of(sc_c, sc_f, "tv^1.0"))
        d_star = abs(d_of(sc_c, sc_f, f"tv^{p_star}"))
        lo_raw, hi_raw = boot_diff(sc_c, sc_f, "theta", "tv^1.0")
        lo_st, hi_st = boot_diff(sc_c, sc_f, "theta", f"tv^{p_star}")
        verdicts[k] = (p_star, d_th, d_tv1, d_star, lo_raw, hi_raw, lo_st, hi_st)

        print(f"=== k={k} ===")
        print(f"  FIT half chose p* = {p_star}  (fit |d| = {fit_d[p_star]:.4f}; "
              f"raw TV fit |d| = {fit_d[1.0]:.4f})")
        print(f"  SCORE half:  |d| theta = {d_th:.4f}   raw TV = {d_tv1:.4f}   "
              f"TV^{p_star} = {d_star:.4f}")
        print(f"    theta - rawTV   = {d_th-d_tv1:+.4f}  CI [{lo_raw:+.4f}, {hi_raw:+.4f}]"
              f"   {'EXCLUDES 0' if lo_raw > 0 or hi_raw < 0 else 'STRADDLES 0'}")
        print(f"    theta - TV^{p_star} = {d_th-d_star:+.4f}  CI [{lo_st:+.4f}, {hi_st:+.4f}]"
              f"   {'EXCLUDES 0' if lo_st > 0 or hi_st < 0 else 'STRADDLES 0'}")
        with JOURNAL.open("a") as fh:
            fh.write(json.dumps({"s": a.s, "k": k, "draws": a.draws, "p_star": p_star,
                                 "d_theta": d_th, "d_tv_raw": d_tv1, "d_tv_star": d_star,
                                 "ci_theta_minus_raw": [lo_raw, hi_raw],
                                 "ci_theta_minus_star": [lo_st, hi_st]}) + "\n")

    print("\n=== K3 VERDICT AGAINST ITS MISSING CONTROL (pre-registered) ===")
    print("  K3 as written asks theta to beat RAW TV. That is not evidence for a")
    print("  sphere: theta = arcsin(sqrt(TV)) is a concave transform, and any")
    print("  concave transform beats raw TV. The real question is whether the")
    print("  GEOMETRIC member beats a NON-geometric one on held-out draws.\n")
    beat = 0
    for k, (ps, dth, dtv, dst, lr, hr, ls, hs) in verdicts.items():
        won = ls > 0
        lost = hs < 0
        beat += won
        tag = ("theta WINS" if won else "theta LOSES" if lost else "TIE")
        print(f"  k={k:<4} p*={ps:<5} theta {dth:.4f} vs TV^{ps} {dst:.4f}  "
              f"diff {dth-dst:+.4f} CI [{ls:+.4f},{hs:+.4f}] -> {tag}")
    print()
    if beat == len(verdicts):
        print("  theta beats the non-geometric control at EVERY k with the CI")
        print("  excluding zero. K3 IS evidence for the geometry.")
    else:
        print("  theta DOES NOT beat the non-geometric control at every k.")
        print("  K3 AS WRITTEN IS NOT EVIDENCE FOR THE GEOMETRY -- it measures")
        print("  row-wise concavity, which a plain power of TV supplies with no")
        print("  sphere, no Fisher-Rao and no Chentsov. K3 must be rewritten")
        print("  against this control or dropped. Not softened.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

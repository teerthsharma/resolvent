"""X6, second half -- the ||tau||_F TRAJECTORY, which is what the clause names.

`scale/equilibrium_probe.py` measured the Karcher RESIDUAL trajectory and took
||tau|| at the glance only. That gap was recorded rather than implied, and this
file closes it.

WORKING OUT WHAT tau IS HERE MAKES THE TEST SHARPER THAN THE CLAUSE ASKS FOR.
The contract defines tau = 1/2 (Y^T X - X^T Y), equilibrium <=> tau = 0. In the
Karcher iteration the two configurations are the k pivot points X (k x s) and the
current mean m broadcast to k rows, Y. Then

    Y^T X = sum_p m (x_p)^T = k * (m (xbar)^T)          xbar = mean_p x_p
    tau   = (k/2) ( m xbar^T - xbar m^T )
    ||tau||_F = 0   <=>   m is PARALLEL TO xbar

So the clause's own equilibrium condition, written out, says: **the settled
reading is at equilibrium exactly when it is parallel to the PLAIN EUCLIDEAN
AVERAGE of the pivot readings.**

THAT TURNS X6's SECOND HALF INTO A HARDER VERSION OF K3. K3 asked whether the
angle beats raw TV. This asks whether the whole Riemannian apparatus -- log map,
exp map, Karcher iteration, injectivity radius -- LANDS ANYWHERE THE ONE-LINE
NORMALISED EUCLIDEAN MEAN DID NOT ALREADY REACH. If ||tau|| -> 0 under iteration,
it does not, and the machinery is notation.

The decisive number is therefore not ||tau|| alone but the ANGLE between the
converged Karcher mean and xbar/||xbar||, MEASURED AGAINST THE SPREAD OF THE
PIVOTS THEMSELVES. An angle far below the spread means the two means coincide for
every practical purpose; an angle comparable to the spread means the geodesic mean
genuinely sits somewhere else.

Float64 throughout, for the reason recorded in `equilibrium_probe.py`: in float32
the residual floors at 1.8546e-08 [RUN] and a tolerance of 1e-8 cannot be met.
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
from scale.torque_probe import boot_ci                           # noqa: E402
from scale.equilibrium_probe import (sphere_rows, log_map, exp_map,  # noqa: E402
                                     torque_norm, HALF_PI, EPS)

JOURNAL = pathlib.Path(__file__).resolve().parents[1] / "results" / "tau_trajectory.jsonl"


def tau_of(m: torch.Tensor, x: torch.Tensor) -> float:
    """||tau||_F with Y = k copies of m, X = the k pivot points."""
    return torque_norm(m.unsqueeze(0).expand_as(x), x)


def angle(a: torch.Tensor, b: torch.Tensor) -> float:
    ca = float((a @ b) / (a.norm() * b.norm()).clamp_min(EPS))
    return math.acos(max(-1.0, min(1.0, ca)))


def run_one(s: int, k: int, d: int, g: torch.Generator, *, steps: int, tol: float):
    x0 = torch.randn(s, d, generator=g)
    wq, wk = torch.randn(d, d, generator=g), torch.randn(d, d, generator=g)
    q, kk = x0 @ wq, x0 @ wk
    i = s - 1
    piv = select_pivots(kk, min(k, s - 2), exclude=(i,))
    a = bench._softmax_operator(q, kk).double()
    sp = sphere_rows(a)
    m = sp[i] / sp[i].norm().clamp_min(EPS)              # the glance
    x = sp[piv]
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(EPS)

    taus, res = [], []
    th_max = 0.0
    for _ in range(steps):
        v, th = log_map(m, x)
        th_max = max(th_max, float(th.max()))
        mv = v.mean(0)
        res.append(float(mv.norm()))
        taus.append(tau_of(m, x))
        if res[-1] < tol:
            break
        m = exp_map(m, mv)
        m = m / m.norm().clamp_min(EPS)

    xbar = x.mean(0)
    xbar = xbar / xbar.norm().clamp_min(EPS)
    # SPREAD: the mean geodesic distance from the converged mean to its pivots.
    # The angle to xbar is only meaningful relative to this.
    spread = float(log_map(m, x)[1].mean())
    return dict(tau0=taus[0], tau_final=taus[-1],
                tau_drop=taus[0] / max(taus[-1], 1e-300),
                res0=res[0], nsteps=len(res), converged=res[-1] < tol,
                ang_karcher_euclid=angle(m, xbar), spread=spread,
                ang_over_spread=angle(m, xbar) / max(spread, EPS),
                ang_glance_euclid=angle(sp[i] / sp[i].norm().clamp_min(EPS), xbar),
                unique=th_max < HALF_PI)


def controls() -> list:
    """Must-fire. The instrument has to distinguish 'Karcher mean equals the
    normalised Euclidean mean' from 'it does not', or every number is void."""
    out, g = [], torch.Generator().manual_seed(11)
    b = torch.randn(48, generator=g, dtype=torch.float64).abs()
    b = b / b.norm()

    # C1 identical points: Karcher mean IS the point IS the Euclidean mean.
    x = b.unsqueeze(0).repeat(6, 1)
    out.append(("C1 identical pivots -> ||tau||=0 and angle(Karcher, Euclid)=0",
                f"tau={tau_of(b, x):.3e} ang={angle(b, x.mean(0)):.3e}",
                tau_of(b, x) < 1e-12 and angle(b, x.mean(0)) < 1e-8))

    # C2 a point NOT parallel to xbar must read ||tau|| > 0. Without this the
    #    instrument could report 0 for everything and look like a clean pass.
    y = torch.randn(6, 48, generator=g, dtype=torch.float64).abs()
    y = y / y.norm(dim=-1, keepdim=True)
    out.append(("C2 ||tau|| > 0 when the iterate is NOT parallel to xbar",
                f"tau={tau_of(b, y):.6f} ang={angle(b, y.mean(0)):.6f}",
                tau_of(b, y) > 1e-6 and angle(b, y.mean(0)) > 1e-6))

    # C3 WAS HERE AND IT WAS NOT A CONTROL. It printed an angle and passed
    # `True` unconditionally -- a check that cannot fail, which is the exact
    # instrument-#15 shape this project has struck twice. It also chose three
    # orthonormal axes, whose geodesic and chordal means are BOTH the symmetric
    # point, so its "gap" was zero by symmetry and it would have looked like a
    # clean pass either way. DELETED rather than softened: C2 already
    # establishes the instrument reads a nonzero angle and a nonzero ||tau||
    # (2.089918 / 0.635487), which is the whole job.
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=int, default=256)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--draws", type=int, default=60)
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 32, 128])
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print("=== MUST-FIRE CONTROLS (read before any number below) ===")
    ok = True
    for name, detail, fired in controls():
        ok &= fired
        print(f"  [{'FIRED' if fired else 'DID NOT FIRE'}] {name}\n      {detail}")
    if not ok:
        print("\n  A CONTROL DID NOT FIRE -- the instrument cannot tell the two")
        print("  means apart. Every number below would be decoration. Stopping.")
        return 1

    print(f"\nX6 ||tau|| TRAJECTORY. s={a.s} d={a.d} draws={a.draws} ks={a.ks} "
          f"float64 threads={torch.get_num_threads()}")
    print(f"DECLARED: {a.draws} draws per cell. Other agents share this box; "
          f"NO TIMING IS REPORTED.\n")
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)

    print(f"  {'k':>4} {'||tau|| glance':>15} {'||tau|| settled':>16} "
          f"{'ang(Karcher,Euclid)':>21} {'spread':>9} {'ang/spread':>12}")
    rows = {}
    for k in a.ks:
        g = torch.Generator().manual_seed(a.seed)
        rr = [run_one(a.s, k, a.d, g, steps=a.steps, tol=a.tol)
              for _ in range(a.draws)]
        t0 = sum(r["tau0"] for r in rr) / len(rr)
        tf = sum(r["tau_final"] for r in rr) / len(rr)
        ag = [r["ang_karcher_euclid"] for r in rr]
        lo, hi = boot_ci(ag)
        sp = sum(r["spread"] for r in rr) / len(rr)
        ov = sum(r["ang_over_spread"] for r in rr) / len(rr)
        rows[k] = (t0, tf, sum(ag) / len(ag), lo, hi, sp, ov)
        print(f"  {k:>4} {t0:>15.6f} {tf:>16.3e} "
              f"{sum(ag)/len(ag):>10.6f} [{lo:.5f},{hi:.5f}] {sp:>9.6f} {ov:>12.6f}")
        with JOURNAL.open("a") as fh:
            fh.write(json.dumps({"s": a.s, "k": k, "draws": a.draws,
                                 "tau_glance": t0, "tau_settled": tf,
                                 "ang_karcher_euclid": sum(ag) / len(ag),
                                 "ang_ci": [lo, hi], "spread": sp,
                                 "ang_over_spread": ov}) + "\n")

    print("\n=== VERDICT ===")
    print("  ||tau|| = 0  <=>  the settled reading is PARALLEL to the plain")
    print("  Euclidean average of the pivots. So a ||tau|| that decays to zero")
    print("  says the Riemannian machinery landed where xbar/||xbar|| already was.")
    for k, (t0, tf, ag, lo, hi, sp, ov) in rows.items():
        print(f"\n  k={k}")
        print(f"    ||tau||  {t0:.6f} -> {tf:.3e}   (falls by {t0/max(tf,1e-300):.3e}x)")
        print(f"    angle(Karcher mean, normalised Euclidean mean) = {ag:.6f} rad"
              f"  [{lo:.6f}, {hi:.6f}]")
        print(f"    pivot spread = {sp:.6f} rad, so the gap is {ov:.4%} of it")
        if hi < 1e-6:
            print("    -> THE TWO MEANS COINCIDE. The sphere apparatus lands on the")
            print("       normalised Euclidean average. X6's second half says the")
            print("       machinery is NOTATION at this k.")
        elif ov < 0.01:
            print(f"    -> the gap is under 1% of the spread: the geodesic mean is")
            print("       PRACTICALLY the flat one. Machinery buys almost nothing.")
        else:
            print(f"    -> the gap is {ov:.2%} of the spread: the geodesic mean sits")
            print("       somewhere the flat average does not reach. Machinery EARNS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

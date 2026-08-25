"""X6 -- the equilibrium clause, tested against its own pre-registered kill.

THE CLAUSE IS IN THE MODULE'S NAME. "Displacement-EQUILIBRIUM Attention", and
constitution ideal 2 says: "Equilibrium over glance -- the reading is a fixed
point with a certificate, OR THE WORD 'equilibrium' IS CUT."

X6, pre-registered in LOOP_PROMPT.md and never run:

    falsifiable equilibrium -> ||tau||_F trajectory under iteration.
    IF ONE PASS ALREADY GIVES ||tau||_F ~ 0, THE EQUILIBRIUM CLAUSE IS CUT
    -- the R1 lesson.

ARM A printed ||tau|| at a SINGLE pass (5.4280 / 3.3409 / 2.3831 at k=8/32/128)
and never iterated. A single-pass number is not a trajectory, and the kill is
about the trajectory.

WHAT THE ITERATION IS. The contract's global equilibrium form: "the settled
reading is the Karcher/Frechet mean of its pivot readings, UNIQUE for theta <
pi/2 (injectivity radius)". So row i's reading should settle to the Karcher mean
of its k pivot rows' readings, all of them points on the unit sphere via the
square-root map.

    log_m(x) = (theta / sin theta)(x - cos theta * m),   theta = arccos <m, x>
    exp_m(v) = cos||v|| * m + sin||v|| * v/||v||
    Karcher step:  m <- exp_m( mean_p log_m(x_p) )
    residual    :  || mean_p log_m(x_p) ||

THE INITIALISATION IS WHAT MAKES THE KILL DECISIVE, and it is not a free choice.
The iteration starts at THE SOFTMAX READING ITSELF -- the glance. So the residual
at step 0 answers exactly the question X6 asks: is the glance ALREADY the
equilibrium? If it is, iterating buys nothing, there is no fixed point to find
because we started at it, and the word is cut.

UNIQUENESS IS A PRECONDITION, NOT AN ASSUMPTION. The Karcher mean is unique only
inside the injectivity radius, theta < pi/2. That is CHECKED per draw and
reported, never assumed. A draw whose spread exceeds it has no unique mean and
the "settled reading" is not well defined there.

CONTENTION NOTE: four agents are running against this box. WALL-CLOCK TIMINGS
FROM THIS FILE ARE NOT EVIDENCE and none are reported as such. Residuals, angles
and iteration counts are arithmetic on fixed inputs and are unaffected.

G8: no multiplication inside any sign decision -- there are no sign decisions in
this file; every quantity is a norm or an angle.
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

JOURNAL = pathlib.Path(__file__).resolve().parents[1] / "results" / "equilibrium.jsonl"
EPS = 1e-12
HALF_PI = math.pi / 2


def sphere_rows(a: torch.Tensor) -> torch.Tensor:
    """Simplex rows -> unit-sphere points. phi(p) = sqrt(p), Fisher-Rao."""
    return a.clamp_min(0).sqrt()


def log_map(m: torch.Tensor, x: torch.Tensor):
    """(log_m(x_p) for each row of x, theta_p). m is one point, x is k points."""
    ct = (x @ m).clamp(-1.0, 1.0)
    th = torch.arccos(ct)
    st = torch.sin(th).clamp_min(EPS)
    return (th / st).unsqueeze(-1) * (x - ct.unsqueeze(-1) * m), th


def exp_map(m: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    n = v.norm()
    if float(n) < EPS:
        return m
    return math.cos(float(n)) * m + math.sin(float(n)) * (v / n)


def karcher(m0: torch.Tensor, x: torch.Tensor, *, steps: int, tol: float):
    """Karcher iteration from m0 over the points x. Returns the residual at EVERY
    step, step 0 FIRST -- the whole point is what step 0 reads."""
    m, res = m0.clone(), []
    for _ in range(steps):
        v, th = log_map(m, x)
        mean_v = v.mean(0)
        res.append(float(mean_v.norm()))
        if res[-1] < tol:
            break
        m = exp_map(m, mean_v)
        m = m / m.norm().clamp_min(EPS)
    return m, res, float(th.max())


def torque_norm(y: torch.Tensor, x: torch.Tensor) -> float:
    """||tau||_F with tau = 1/2 (Y^T X - X^T Y). Zero iff Y^T X is symmetric."""
    yx = y.transpose(-2, -1) @ x
    return float((0.5 * (yx - yx.transpose(-2, -1))).norm())


def one(s: int, k: int, d: int, g: torch.Generator, *, steps: int, tol: float):
    x0 = torch.randn(s, d, generator=g)
    wq, wk = torch.randn(d, d, generator=g), torch.randn(d, d, generator=g)
    q, kk = x0 @ wq, x0 @ wk
    i = s - 1
    piv = select_pivots(kk, min(k, s - 2), exclude=(i,))
    # THE ITERATION RUNS IN float64 AND THAT IS A DECLARED ANALYSIS CHOICE.
    # The shipped operator is float32; the Karcher iteration is not the shipped
    # object, it is the measurement of one. In float32 the residual FLOORS at
    # ~1.85e-08 [RUN] because float32 eps is 1.1920929e-07, so a tol of 1e-8 is
    # BELOW WHAT THE ARITHMETIC CAN REACH and `converged` could never be true --
    # a threshold that cannot be met, which is the defect class this project
    # keeps finding. In float64 the tolerance becomes a real test instead of an
    # unreachable one: this probe reads 7.481e-09 / 8.155e-09 / 8.405e-09 at its
    # published settings, and 7.307e-13 / 7.958e-13 / 8.405e-13 when forced to
    # --tol 1e-15 --steps 400.
    #
    # A FIGURE THAT STOOD HERE WAS STRUCK [r5 iter 21]. It was asserted in a
    # [RUN] voice, it lived ONLY in this comment and in prose, and no code path
    # produced it. It came from a throwaway float64 check whose own output was
    # defective -- that script printed `nan` for both means because it omitted
    # the norm clamp this file has. It is in the STRUCK registry now.
    a = bench._softmax_operator(q, kk).double()
    sp = sphere_rows(a)
    m0 = sp[i] / sp[i].norm().clamp_min(EPS)     # THE GLANCE: row i's own reading
    pts = sp[piv]
    pts = pts / pts.norm(dim=-1, keepdim=True).clamp_min(EPS)
    m, res, th_max = karcher(m0, pts, steps=steps, tol=tol)
    # ||tau|| between the glance and the settled reading, as matrices of one row
    tau0 = torque_norm(m0.unsqueeze(0), pts.mean(0, keepdim=True))
    return dict(res0=res[0], res_final=res[-1], nsteps=len(res),
                converged=res[-1] < tol,
                theta_max=th_max, unique=th_max < HALF_PI,
                moved=float((m - m0).norm()), tau0=tau0, res=res)


def controls(d_sphere: int = 64) -> list:
    """Must-fire. A probe that cannot read a PLANTED zero and a PLANTED nonzero
    is not measuring equilibrium, it is printing a number."""
    out = []
    g = torch.Generator().manual_seed(7)
    base = torch.randn(d_sphere, generator=g).abs()
    base = base / base.norm()

    # C1 identical points -> the mean IS the point -> residual exactly 0 at step 0
    x = base.unsqueeze(0).repeat(8, 1)
    _, res, _ = karcher(base, x, steps=5, tol=1e-12)
    out.append(("C1 identical pivot readings -> residual 0 at step 0",
                f"res[0]={res[0]:.3e}", res[0] < 1e-12))

    # C2 spread points -> residual MUST be nonzero and iteration MUST move
    y = torch.randn(8, d_sphere, generator=g).abs()
    y = y / y.norm(dim=-1, keepdim=True)
    m, res, th = karcher(base, y, steps=64, tol=1e-10)
    out.append(("C2 spread pivot readings -> residual nonzero, iterate moves",
                f"res[0]={res[0]:.6f} res[-1]={res[-1]:.3e} steps={len(res)} "
                f"moved={float((m-base).norm()):.6f}",
                res[0] > 1e-6 and float((m - base).norm()) > 1e-6))

    # C3 the uniqueness guard must be able to FAIL. Antipodal-ish points exceed
    #    the injectivity radius, where the Karcher mean is not unique.
    z = torch.cat([base.unsqueeze(0), (-base).unsqueeze(0)], 0)
    _, _, th_bad = karcher(base, z, steps=2, tol=1e-12)
    out.append(("C3 uniqueness guard fires past the injectivity radius",
                f"theta_max={th_bad:.6f} rad  (pi/2 = {HALF_PI:.6f})",
                th_bad >= HALF_PI))

    # C4 tau reads 0 on a symmetric pair and nonzero on an asymmetric one
    p = torch.randn(4, 6, generator=g)
    out.append(("C4 ||tau|| = 0 when Y == X, nonzero otherwise",
                f"same={torque_norm(p, p):.3e} "
                f"diff={torque_norm(p, torch.randn(4, 6, generator=g)):.6f}",
                torque_norm(p, p) < 1e-12))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=int, default=256)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--draws", type=int, default=60)
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print("=== MUST-FIRE CONTROLS (read these before any number below) ===")
    ok_all = True
    for name, detail, fired in controls():
        ok_all &= fired
        print(f"  [{'FIRED' if fired else 'DID NOT FIRE'}] {name}\n      {detail}")
    if not ok_all:
        print("\n  A CONTROL DID NOT FIRE. The probe is not measuring equilibrium.")
        print("  Every number below would be decoration. Stopping.")
        return 1

    print(f"\nX6 EQUILIBRIUM PROBE. s={a.s} d={a.d} draws={a.draws} ks={a.ks} "
          f"steps<={a.steps} tol={a.tol} threads={torch.get_num_threads()}")
    print(f"DECLARED: {a.draws} draws per cell. Four agents share this box, so "
          f"NO TIMING IS REPORTED -- residuals are arithmetic and unaffected.\n")
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)

    print(f"  {'k':>4} {'residual @ step 0':>28} {'residual final':>16} "
          f"{'steps*':>7} {'conv':>8} {'unique':>8} {'||tau||_0':>11}")
    print("        *steps averaged over CONVERGED draws only; `conv` is the "
          "fraction that reached tol before the cap.")
    verdicts = {}
    for k in a.ks:
        g = torch.Generator().manual_seed(a.seed)
        rows = [one(a.s, k, a.d, g, steps=a.steps, tol=a.tol)
                for _ in range(a.draws)]
        r0 = [r["res0"] for r in rows]
        lo, hi = boot_ci(r0)
        rf = sum(r["res_final"] for r in rows) / len(rows)
        conv = [r for r in rows if r["converged"]]
        # CENSORED. A draw that hit the step cap has NO convergence time, and
        # averaging the cap into the mean reports the cap as if it were a
        # measurement. Only converged draws contribute; the censored fraction
        # is printed beside it and never folded in.
        ns = sum(r["nsteps"] for r in conv) / len(conv) if conv else float("nan")
        cfrac = len(conv) / len(rows)
        uq = sum(1 for r in rows if r["unique"]) / len(rows)
        tq = sum(r["tau0"] for r in rows) / len(rows)
        verdicts[k] = (sum(r0) / len(r0), lo, hi, ns, uq, cfrac)
        print(f"  {k:>4} {sum(r0)/len(r0):>12.6f} [{lo:.5f},{hi:.5f}] "
              f"{rf:>16.3e} {ns:>7.2f} {cfrac:>8.4f} {uq:>8.4f} {tq:>11.6f}")
        with JOURNAL.open("a") as fh:
            fh.write(json.dumps({"s": a.s, "k": k, "draws": a.draws,
                                 "res0_mean": sum(r0) / len(r0), "res0_ci": [lo, hi],
                                 "res_final_mean": rf, "steps_mean": ns,
                                 "unique_frac": uq, "tau0_mean": tq}) + "\n")

    print("\n=== X6 VERDICT (pre-registered: one pass already ~0 => CLAUSE CUT) ===")
    for k, (m, lo, hi, ns, uq, cfrac) in verdicts.items():
        cut = hi < a.tol
        print(f"  k={k:<4} residual at the GLANCE = {m:.6f} [{lo:.6f}, {hi:.6f}]")
        print(f"          steps to tol (converged draws only) {ns:.2f}   "
              f"reached tol {cfrac:.4f}   unique (theta<pi/2) {uq:.4f}")
        if cut:
            print("          -> ALREADY SETTLED. The glance IS the fixed point.")
            print(f"          -> EQUILIBRIUM CLAUSE IS CUT at k={k}.")
        elif cfrac > 0 and ns <= 1.0:
            print("          -> converged in ONE step: iteration buys nothing.")
            print(f"          -> EQUILIBRIUM CLAUSE IS CUT at k={k}.")
        else:
            print(f"          -> NOT settled at the glance; CI is {lo:.6f} above "
                  f"a tol of {a.tol:g}.")
            print(f"          -> the equilibrium clause SURVIVES at k={k}.")
        if cfrac < 1.0:
            print(f"          !! {1-cfrac:.1%} of draws hit the step cap without "
                  f"reaching tol. Convergence is CENSORED, not measured.")
        if uq < 1.0:
            print(f"          !! {1-uq:.1%} of draws have theta_max >= pi/2, "
                  f"OUTSIDE the injectivity radius. The Karcher mean is NOT")
            print("             unique there, so 'the settled reading' is not "
                  "well defined on those draws.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""FOREMAN probe -- is theta a reparametrisation of TV, and what is in D_FR?

NOT a rewrite of ARM A. It replays ARM A's EXACT draw stream (same generator,
same consumption order, seeds 0 / 999) and asks questions ARM A did not. The
provenance line prints mean D_FR next to ARM A's published value; agreement to
6 dp means the draws are IDENTICAL, which is what K3 requires.

Each check below encodes a ROUND-5 CLAIM (or torque_probe's own docstring) as an
assertion. An assertion that FAILS is a RED finding against the code as it stands.

Threads pinned HERE (torch.set_num_threads(2)) -- a probe that does not pin is a
probe whose number is a function of its launcher.

Structural claim under test, read off scale/torque_probe.rows_with_and_without:
masking ONE token c and renormalising gives, with m_i = A^c[i,c],
    A^0[i,j] = A^c[i,j]/(1-m_i)   (j != c),    A^0[i,c] = 0
so the row deformation has exactly ONE degree of freedom, and on any row that
carries mass,
    TV_i  = m_i                        exactly
    BC_i  = sqrt(1-m_i)                exactly
    th_i  = arcsin(sqrt(TV_i))         exactly
If that holds, theta and TV are the same per-row number in different units.

DEAD ROWS. Under the strictly-causal mask row 0 has NO visible keys, so
_softmax_operator returns an all-zero row and BC_0 = 0. theta_rows' docstring
says such rows "give 0"; arccos(0) is pi/2. Every quantity below is therefore
reported BOTH as ARM A computes it and over LIVE rows only.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
torch.set_num_threads(2)                      # pinned HERE, not by the launcher

from scale.pivot_probe import select_pivots                          # noqa: E402
from scale.torque_probe import (rows_with_and_without, theta_rows,    # noqa: E402
                                tv_rows, shadow, cohen_d)

JOURNAL = pathlib.Path(__file__).resolve().parents[1] / "results" / "foreman_theta_tv.jsonl"
# results/arm_a.jsonl, s=1024 d=16 draws=120 threads=2 seeds 0/999
PUB_C = {8: 0.030850132878792163, 32: 0.018089019586720195, 128: 0.01320268574054353}
PUB_F = {8: 0.0033169065058852236, 32: 0.003067687266351034, 128: 0.0027839834840657812}
F32 = 1e-5                                    # float32 sum over 1024 terms
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    """The assertion IS the claim. A failing assertion is RED."""
    if not ok:
        FAILS.append(name)
    print("  [%s] %s\n         %s" % ("GREEN" if ok else "RED  ", name, detail))


def one(s: int, k: int, d: int, g: torch.Generator, *, filler: bool, svd: bool):
    """Byte-for-byte ARM A draw stream (arm_a_run.one), displacement half only.

    The two randn(d) of ARM A's flip half are still CONSUMED so the generator
    stays aligned draw-for-draw with the published table.
    """
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
        return None                          # ARM A returns before the flip half
    c = pool[int(torch.randint(0, len(pool), (1,), generator=g))]

    a_c, a_0 = rows_with_and_without(q, kk, c)
    th, tv = theta_rows(a_c, a_0), tv_rows(a_c, a_0)
    live = a_c.sum(-1) > 0                    # rows the causal mask leaves alive
    r = dict(theta=float(th.mean()), tv=float(tv.mean()), c=c,
             th=th, tvr=tv, m=a_c[:, c].clone(), live=live,
             n_dead=int((~live).sum()),
             n_pi2=int((th > math.pi / 2 - 1e-4).sum()),
             theta_live=float(th[live].mean()))
    if svd:
        xi, _ = shadow(a_c, a_0)
        _, S, Vh = torch.linalg.svd(xi)
        st = torch.sin(th).clamp_min(1e-12)
        beta = (th / st) * torch.sqrt((r["m"] * (1 - r["m"])).clamp_min(0))
        r.update(s1=float(S[0]), gamma=float(S[0] - S[1]),
                 v1c=float(Vh[0, c].abs()), beta_norm=float(beta.norm()),
                 a_c=a_c, a_0=a_0)
    torch.randn(d, generator=g)                # ARM A's flip half, cval 1
    torch.randn(d, generator=g)                # ARM A's flip half, cval 2
    return r


def cell(s, k, n, d, seed, filler, svd=False):
    g = torch.Generator().manual_seed(seed)
    out = []
    while len(out) < n:
        r = one(s, k, d, g, filler=filler, svd=svd)
        if r is not None:
            out.append(r)
    return out


def _rank(v):
    t = torch.tensor(v, dtype=torch.float64)
    out = torch.empty_like(t)
    out.scatter_(0, t.argsort(), torch.arange(t.numel(), dtype=torch.float64))
    return out


def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra @ rb) / (ra.norm() * rb.norm()))


def slope(xs, ys):
    """arm_a_run.slope, verbatim in behaviour: log10-log10 least squares."""
    lx = [math.log10(x) for x in xs]
    ly = [math.log10(max(y, 1e-12)) for y in ys]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    sxx = sum((x - mx) ** 2 for x in lx)
    return sum((x - mx) * (y - my) for x, y in zip(lx, ly)) / sxx if sxx else float("nan")


def main() -> int:
    s, d, n, nsvd = 1024, 16, 120, 24
    ks = [8, 32, 128]
    print("FOREMAN theta-vs-TV probe. s=%d d=%d draws=%d (svd draws=%d) ks=%s "
          "threads=%d seeds=0/999" % (s, d, n, nsvd, ks, torch.get_num_threads()))
    print("Replaying ARM A's exact draw stream. Assertions ENCODE round-5's claims;")
    print("a failing assertion is RED against the code as it stands.\n")
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    log, corr_c, corr_f = [], {}, {}

    for k in ks:
        print("--- k=%d %s" % (k, "-" * 60))
        cs = cell(s, k, n, d, 0, False)
        fs = cell(s, k, n, d, 999, True)
        th_c = [r["theta"] for r in cs]
        th_f = [r["theta"] for r in fs]
        tv_c = [r["tv"] for r in cs]
        tv_f = [r["tv"] for r in fs]
        mean_th = sum(th_c) / n
        print("  provenance: mean D_FR causal = %.6f   ARM A published = %.6f"
              % (mean_th, PUB_C[k]))

        # ---- R0: torque_probe's own docstring: dead rows "give 0" ----
        dead = max(r["n_dead"] for r in cs)
        pi2 = max(r["n_pi2"] for r in cs)
        dead_min = min(r["n_dead"] for r in cs)
        floor = math.pi / 2 * dead / s
        check("R0 theta_rows gives 0 on rows with no mass either side (its docstring)",
              pi2 == 0,
              "dead rows/draw (a_c row sum == 0): min=%d max=%d ; rows reading "
              "theta == pi/2: max=%d -> constant floor pi/2*%d/%d = %.7f rad in "
              "EVERY D_FR, causal and filler alike" % (dead_min, dead, pi2, dead, s, floor))

        # ---- R0b: how much of the published D_FR is that floor? ----
        cc = (mean_th * s - math.pi / 2 * dead) / (s - dead)
        cf = (sum(th_f) / n * s - math.pi / 2 * dead) / (s - dead)
        corr_c[k], corr_f[k] = cc, cf
        check("R0b the dead-row floor is a negligible part of D_FR (<1%)",
              floor / mean_th < 0.01,
              "causal D_FR %.6f -> %.6f live-only (floor is %.1f%%) ; "
              "filler D_FR %.6f -> %.6f live-only (floor is %.1f%%)"
              % (mean_th, cc, 100 * floor / mean_th,
                 sum(th_f) / n, cf, 100 * floor / (sum(th_f) / n)))

        # ---- R1: TV is the masked token's own attention mass, exactly ----
        e_tv = max(float((r["tvr"][r["live"]] - r["m"][r["live"]]).abs().max()) for r in cs)
        check("R1 TV_i is a divergence, not merely A^c[i,c]",
              e_tv > F32,
              "max over LIVE rows |TV_i - A^c[i,c]| = %.3e   (float32 floor ~%.0e)"
              % (e_tv, F32))

        # ---- R2: theta = arcsin(sqrt(TV)) exactly, per row ----
        e_th = max(float((r["th"][r["live"]]
                          - torch.arcsin(r["tvr"][r["live"]].clamp(0, 1).sqrt())).abs().max())
                   for r in cs)
        check("R2 theta_i carries information TV_i does not",
              e_th > F32,
              "max over LIVE rows |theta_i - arcsin(sqrt(TV_i))| = %.3e" % e_th)

        # ---- R3: does the geometry ever leave the tangent plane? ----
        thmax = max(float(r["th"][r["live"]].max()) for r in cs)
        curv = thmax * thmax / 24.0      # arc/chord - 1 = th^2/24 + O(th^4)
        check("R3 draws visit the sphere's curvature (arc vs chord > 0.1%)",
              curv > 1e-3,
              "max theta over LIVE rows = %.6f rad -> arc/chord - 1 = %.3e "
              "(%.6f%%) vs theta's K3 margin over TV of ~2%%" % (thmax, curv, 100 * curv))

        # ---- R4: draw-level rank identity ----
        rho_c, rho_f = spearman(th_c, tv_c), spearman(th_f, tv_f)
        rho_all = spearman(th_c + th_f, tv_c + tv_f)
        check("R4 theta and TV rank draws differently (max rho < 0.99)",
              max(rho_c, rho_f, rho_all) < 0.99,
              "spearman(theta,TV): causal=%.6f filler=%.6f pooled=%.6f (n=%d/cell)"
              % (rho_c, rho_f, rho_all, n))

        # ---- R5: is the K3 edge geometry, or row-wise concavity? ----
        d_th = cohen_d(th_c, th_f)
        d_tv = cohen_d(tv_c, tv_f)
        fam = {}
        for p in (0.10, 0.15, 0.20, 0.25, 0.33, 0.50, 0.75, 1.00):
            a = [float((r["tvr"].clamp_min(0) ** p).mean()) for r in cs]
            b = [float((r["tvr"].clamp_min(0) ** p).mean()) for r in fs]
            fam[p] = cohen_d(a, b)
        best_p = max(fam, key=lambda p: fam[p])
        check("R5 theta beats every plain row-wise power of TV (sphere is needed)",
              d_th >= max(fam.values()) - 1e-9,
              "d_theta=%+.4f  d_TV=%+.4f (theta wins by %.1f%%)  |  best TV^%.2f "
              "d=%+.4f (wins by %.1f%%, %.1fx theta's margin) | family: %s"
              % (d_th, d_tv, 100 * (d_th / d_tv - 1), best_p, fam[best_p],
                 100 * (fam[best_p] / d_tv - 1),
                 (fam[best_p] / d_tv - 1) / max(d_th / d_tv - 1, 1e-12),
                 " ".join("p=%.2f:%+.4f" % (p, v) for p, v in fam.items())))

        log.append(dict(k=k, s=s, d=d, draws=n, threads=2, seeds=[0, 999],
                        mean_theta=mean_th, published=PUB_C[k],
                        dead_rows=dead, pi2_rows=pi2, floor=floor,
                        theta_c_live=cc, theta_f_live=cf,
                        e_tv=e_tv, e_theta=e_th, theta_max_live=thmax,
                        arc_chord=curv, rho_c=rho_c, rho_f=rho_f,
                        rho_pooled=rho_all, d_theta=d_th, d_tv=d_tv,
                        family={str(p): v for p, v in fam.items()}))

        # ---- R6/R7: the shadow's principal direction ----
        cs2 = cell(s, k, nsvd, d, 0, False, svd=True)
        v1c = [r["v1c"] for r in cs2]
        s1e = [abs(r["s1"] - r["beta_norm"]) / max(r["s1"], 1e-12) for r in cs2]
        n_ec = sum(1 for v in v1c if v > 0.99)
        n_b = sum(1 for e in s1e if e < 0.01)
        check("R6 the shadow's top direction is not the removed token e_c",
              n_ec == 0,
              "|<v_1,e_c>| > 0.99 on %d/%d draws (min=%.4f max=%.6f) -- the "
              "principal 'shadow' direction IS c's own coordinate"
              % (n_ec, nsvd, min(v1c), max(v1c)))
        check("R7 sigma_1 is not the closed form ||beta||, beta_i=(th/sin th)sqrt(m(1-m))",
              n_b == 0,
              "rel |sigma_1 - ||beta|| | < 1%% on %d/%d draws (min=%.3e)"
              % (n_b, nsvd, min(s1e)))
        log[-1].update(v1c_is_ec=n_ec, s1_is_beta=n_b, svd_draws=nsvd,
                       gamma_mean=sum(r["gamma"] for r in cs2) / nsvd)
        print()

    # ---- R8: K1's D_FR slope, with and without the dead-row floor ----
    print("--- K1 slope contamination %s" % ("-" * 44))
    sp = slope(ks, [PUB_C[k] for k in ks])
    sl = slope(ks, [corr_c[k] for k in ks])
    check("R8 the dead-row floor does not move K1's D_FR slope by >5%",
          abs(sl - sp) / abs(sp) < 0.05,
          "D_FR slope in k: as ARM A computes it = %+.4f ; live rows only = "
          "%+.4f ; shift = %+.4f (%.1f%%). K1 bar is >= -0.10, and 'dies with "
          "flip' trips at < -0.30." % (sp, sl, sl - sp, 100 * abs(sl - sp) / abs(sp)))
    print("         published D_FR %s" % ["%.6f" % PUB_C[k] for k in ks])
    print("         live-only D_FR %s" % ["%.6f" % corr_c[k] for k in ks])
    print("         published filler %s" % ["%.6f" % PUB_F[k] for k in ks])
    print("         live-only filler %s" % ["%.6f" % corr_f[k] for k in ks])
    print("         K2 causal/filler ratio published %s -> live-only %s"
          % (["%.2f" % (PUB_C[k] / PUB_F[k]) for k in ks],
             ["%.2f" % (corr_c[k] / corr_f[k]) for k in ks]))

    with JOURNAL.open("a") as fh:
        for e in log:
            fh.write(json.dumps(e) + "\n")
        fh.write(json.dumps(dict(k1_slope_published=sp, k1_slope_live=sl,
                                 ks=ks, corr_c=corr_c, corr_f=corr_f)) + "\n")

    print("\n" + "=" * 72)
    print("RED checks (%d of %d):" % (len(FAILS), 3 + 8 * len(ks)))
    for f in dict.fromkeys(FAILS):
        print("  " + f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())

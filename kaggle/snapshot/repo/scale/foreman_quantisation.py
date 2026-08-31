"""FOREMAN probe 3 -- theta is not measured, it is QUANTISED by float32 arccos.

Companion to foreman_theta_tv.py / foreman_curvature.py. Same ARM A draw stream
(seeds 0/999, s=1024, d=16, threads pinned HERE).

WHAT PROBE 2 ACTUALLY FOUND. Over "ordinary" rows (0 < TV < 0.999) the residual
|theta - arcsin(sqrt(TV))| stuck at 9.766e-04 and p90 of theta was 4.88e-04. Those
are not measurements. With eps = 2^-24 = 5.9604645e-08, the float32 ULP just
BELOW 1.0 (values in [0.5,1) are spaced 2^-24, not 2^-23),

    arccos(1 - 2*eps) = 4.882813e-04     <- exactly p90 of theta
    arccos(1 - 4*eps) = 6.905340e-04
and the five most-occupied nonzero theta values are arccos(1 - n*eps), n=1..5,
matched to 1.1e-07 relative, carrying 83-88% of all nonzero rows.

theta_rows computes arccos(BC) where BC -> 1 as the displacement -> 0, and
arccos(1 - delta) ~ sqrt(2 delta): a float32 error of one ULP in BC becomes an
ABSOLUTE error of 4.9e-04 in theta. Catastrophic cancellation, in the regime
where nearly every row sits.

THE SAME NUMBER HAS AN EXACT ROUTE. Masking one token and renormalising is a
one-parameter row deformation with m_i = A^c[i,c], so
    BC_i = sqrt(1 - m_i)   and   theta_i = arcsin(sqrt(m_i))    exactly,
and arcsin(sqrt(m)) is computed at FULL relative precision because m never
approaches the cancelling end. This file measures the gap between the two routes
on ARM A's own draws, and re-runs K3 on the exact route.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
torch.set_num_threads(2)                      # pinned HERE, not by the launcher

from scale.foreman_theta_tv import cell, check, FAILS, spearman        # noqa: E402
from scale.torque_probe import cohen_d                                 # noqa: E402

JOURNAL = pathlib.Path(__file__).resolve().parents[1] / "results" / "foreman_theta_tv.jsonl"
PUB_C = {8: 0.030850132878792163, 32: 0.018089019586720195, 128: 0.01320268574054353}
EPS32 = 2.0 ** -24   # ULP just BELOW 1.0; [0.5,1) has spacing 2^-24, not 2^-23


def theta_exact(m: torch.Tensor) -> torch.Tensor:
    """arcsin(sqrt(m)) in float64. Exact in real arithmetic; no cancellation."""
    return torch.arcsin(m.double().clamp(0, 1).sqrt())


def main() -> int:
    s, d, n = 1024, 16, 120
    ks = [8, 32, 128]
    print("FOREMAN quantisation probe. s=%d d=%d draws=%d ks=%s threads=%d "
          "seeds=0/999" % (s, d, n, ks, torch.get_num_threads()))
    print("float32 ULP below 1.0: eps = %.7e. Predicted arccos ladder "
          "arccos(1 - n*eps):" % EPS32)
    print("   " + "  ".join("n=%d:%.3e" % (i, math.acos(1 - i * EPS32))
                            for i in range(1, 6)) + "\n")
    log = []

    for k in ks:
        print("--- k=%d %s" % (k, "-" * 60))
        cs = cell(s, k, n, d, 0, False)
        fs = cell(s, k, n, d, 999, True)

        ordinary = [r["live"] & (r["tvr"] < 0.999) for r in cs]
        th = torch.cat([r["th"][o] for r, o in zip(cs, ordinary)]).double()
        te = torch.cat([theta_exact(r["m"][o]) for r, o in zip(cs, ordinary)])
        nz = th > 0

        # ---- Q1: is theta resolving, or landing on a grid? ----
        u_th = torch.unique(th[nz]).numel()
        u_te = torch.unique(te[nz]).numel()
        check("Q1 theta resolves comparably to its exact route",
              u_th >= 0.5 * u_te,
              "distinct nonzero values over %d ordinary rows: arccos theta = %d ; "
              "exact arcsin(sqrt(m)) = %d  (ratio %.4f)"
              % (int(nz.sum()), u_th, u_te, u_th / u_te))

        # ---- Q2: name the grid ----
        vals = torch.unique(th[nz]).tolist()[:5]
        pred = [math.acos(1 - i * EPS32) for i in range(1, 6)]
        occ = [int((th == v).sum()) for v in vals]
        check("Q2 theta's smallest values are not the arccos-of-ULP ladder",
              max(abs(v - p) / p for v, p in zip(vals, pred)) > 0.01,
              "5 smallest nonzero theta: %s\n         predicted arccos(1-n*eps): "
              "%s\n         rows on those 5 levels: %s = %.1f%% of nonzero rows"
              % (["%.6e" % v for v in vals], ["%.6e" % p for p in pred], occ,
                 100 * sum(occ) / int(nz.sum())))

        # ---- Q3: how wrong is the angle, row by row? ----
        rel = ((th - te).abs() / te.clamp_min(1e-30))[nz]
        bad10 = float((rel > 0.10).float().mean())
        bad100 = float((rel > 1.0).float().mean())
        check("Q3 fewer than 1% of rows have theta wrong by >10% vs the exact route",
              bad10 < 0.01,
              "rows with |theta_arccos - theta_exact|/theta_exact > 10%%: %.2f%% ; "
              "> 100%%: %.2f%% ; median rel err = %.3e"
              % (100 * bad10, 100 * bad100, float(rel.median())))

        # ---- Q4: what is D_FR actually made of? ----
        allth = torch.cat([r["th"][r["live"]] for r in cs]).double()
        srt = allth.sort(descending=True).values
        top1 = float(srt[:max(1, srt.numel() // 100)].sum() / srt.sum())
        check("Q4 D_FR is not dominated by its top 1% of rows (<50% of the sum)",
              top1 < 0.50,
              "top 1%% of live rows carry %.2f%% of sum(theta). p50 of theta over "
              "live rows = %.3e" % (100 * top1, float(allth.median())))

        # ---- Q5: re-run K3 on the exact route ----
        def mean_all(r, f):
            return float(f(r).mean())
        dfr_e_c = [mean_all(r, lambda z: theta_exact(z["m"])) for r in cs]
        dfr_e_f = [mean_all(r, lambda z: theta_exact(z["m"])) for r in fs]
        d_exact = cohen_d(dfr_e_c, dfr_e_f)
        d_arccos = cohen_d([r["theta"] for r in cs], [r["theta"] for r in fs])
        d_tv = cohen_d([r["tv"] for r in cs], [r["tv"] for r in fs])
        m_e = sum(dfr_e_c) / n
        check("Q5 the arccos route reproduces the exact route's K3 effect (within 1%)",
              abs(d_arccos - d_exact) / abs(d_exact) < 0.01,
              "D_FR: ARM A arccos = %.6f (published %.6f) vs exact "
              "arcsin(sqrt(m)) = %.6f -> %+.1f%% of D_FR is arccos cancellation "
              "+ the pi/2 dead row.\n         K3: d_theta(arccos)=%+.4f  "
              "d_theta(exact)=%+.4f  d_TV=%+.4f  -> exact theta beats TV by %+.2f%%, "
              "arccos theta by %+.2f%%"
              % (sum(r["theta"] for r in cs) / n, PUB_C[k], m_e,
                 100 * (sum(r["theta"] for r in cs) / n - m_e) / m_e,
                 d_arccos, d_exact, d_tv,
                 100 * (d_exact / d_tv - 1), 100 * (d_arccos / d_tv - 1)))

        # ---- Q6: draw-level rank identity on the EXACT route ----
        rho = spearman(dfr_e_c + dfr_e_f, [r["tv"] for r in cs] + [r["tv"] for r in fs])
        check("Q6 exact theta ranks draws differently from TV (rho < 0.99)",
              rho < 0.99,
              "spearman(D_FR_exact, D_TV) pooled over %d draws = %.6f" % (2 * n, rho))

        log.append(dict(probe="quantisation", k=k, s=s, d=d, draws=n, threads=2,
                        seeds=[0, 999], uniq_arccos=u_th, uniq_exact=u_te,
                        grid_levels=vals, grid_pred=pred, grid_occupancy=occ,
                        rel_err_gt10pct=bad10, rel_err_gt100pct=bad100,
                        top1pct_share=top1, dfr_arccos=sum(r["theta"] for r in cs) / n,
                        dfr_exact=m_e, d_arccos=d_arccos, d_exact=d_exact,
                        d_tv=d_tv, rho_exact=rho))
        print()

    with JOURNAL.open("a") as fh:
        for e in log:
            fh.write(json.dumps(e) + "\n")

    print("=" * 72)
    print("RED checks (%d of %d):" % (len(FAILS), 6 * len(ks)))
    for f in dict.fromkeys(FAILS):
        print("  " + f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""FOREMAN probe 2 -- does the SPHERE do any work, or only the square root?

Companion to scale/foreman_theta_tv.py. Same ARM A draw stream (seeds 0/999,
s=1024, d=16, threads pinned HERE), no SVD, so it is cheap.

foreman_theta_tv.py left two checks GREEN for the WRONG REASON: max theta over
live rows read exactly pi/2, which is not curvature, it is SATURATED rows where
the whole row mass sits on c (m_i -> 1). Those rows also make |theta -
arcsin(sqrt(TV))| read 9.77e-04, which is float32 cancellation in
arccos(sqrt(eps)) near m=1, not a real discrepancy. This file separates them.

THE DECISIVE QUESTION. On the unit sphere,
    m_i = A^c[i,c],   TV_i = m_i,   theta_i = arcsin(sqrt(m_i))
so the CHORD (flat, tangent, no sphere) is
    chord_i = 2 sin(theta_i / 2)      and      sin(theta_i) = sqrt(TV_i)
and the geodesic theta_i is the chord PLUS a curvature correction
    theta = sqrt(TV) + TV^{3/2}/6 + O(TV^{5/2}).
Curvature is the ONLY thing the sphere buys over the flat simplex. So: does the
correction earn its keep on K3's own standardized-effect criterion?
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

JOURNAL = pathlib.Path(__file__).resolve().parents[1] / "results" / "foreman_theta_tv.jsonl"
SAT = 0.999                                   # row reads essentially only c


def main() -> int:
    s, d, n = 1024, 16, 120
    ks = [8, 32, 128]
    print("FOREMAN curvature probe. s=%d d=%d draws=%d ks=%s threads=%d "
          "seeds=0/999" % (s, d, n, ks, torch.get_num_threads()))
    print("Does the sphere's curvature do work, or only the square root?\n")
    log = []

    for k in ks:
        print("--- k=%d %s" % (k, "-" * 60))
        cs = cell(s, k, n, d, 0, False)
        fs = cell(s, k, n, d, 999, True)

        # rows that are neither dead nor saturated: the regime the claim is about
        def ordinary(r):
            return r["live"] & (r["tvr"] < SAT)

        # ---- C1: the identity, off the saturated rows ----
        e = max(float((r["th"][ordinary(r)]
                       - torch.arcsin(r["tvr"][ordinary(r)].clamp(0, 1).sqrt())).abs().max())
                for r in cs)
        nsat = [int((r["live"] & (r["tvr"] >= SAT)).sum()) for r in cs]
        check("C1 theta_i carries information TV_i does not (ordinary rows)",
              e > 1e-5,
              "max |theta_i - arcsin(sqrt(TV_i))| over rows with 0 < TV_i < %.3f "
              "= %.3e (float32 floor ~1e-5). saturated rows/draw: min=%d max=%d "
              "mean=%.1f -- on those, theta and arcsin(sqrt(TV)) are BOTH pi/2 and "
              "the 9.77e-04 gap is arccos(sqrt(eps)) cancellation in float32."
              % (SAT, e, min(nsat), max(nsat), sum(nsat) / len(nsat)))

        # ---- C2: where does theta actually live? ----
        allth = torch.cat([r["th"][ordinary(r)] for r in cs])
        q = torch.quantile(allth.double(), torch.tensor([.5, .9, .99, .999, 1.0]).double())
        # fractional curvature content of D_FR: (theta - sin theta)/theta ~ th^2/6
        num = float((allth - torch.sin(allth)).sum())
        den = float(allth.sum())
        check("C2 curvature is a material share of D_FR (>1% of the sum of theta)",
              num / den > 0.01,
              "theta over ordinary rows: p50=%.2e p90=%.2e p99=%.2e p99.9=%.2e "
              "max=%.2e | sum(theta - sin theta)/sum(theta) = %.3e (%.5f%%) "
              "-- that share IS everything the sphere adds to the flat chord"
              % (q[0], q[1], q[2], q[3], q[4], num / den, 100 * num / den))

        # ---- C3/C4: does the curvature correction earn its keep on K3? ----
        def dstat(f):
            a = [float(f(r["th"], r["tvr"]).mean()) for r in cs]
            b = [float(f(r["th"], r["tvr"]).mean()) for r in fs]
            return a, b

        from scale.torque_probe import cohen_d
        stats = {
            "theta (geodesic, ARM A)": lambda t, v: t,
            "chord 2sin(theta/2)": lambda t, v: 2 * torch.sin(t / 2),
            "sin(theta) = sqrt(TV)": lambda t, v: torch.sin(t),
            "raw TV": lambda t, v: v,
        }
        ds = {}
        for nm, f in stats.items():
            a, b = dstat(f)
            ds[nm] = cohen_d(a, b)
        d_th = ds["theta (geodesic, ARM A)"]
        d_ch = ds["chord 2sin(theta/2)"]
        d_sn = ds["sin(theta) = sqrt(TV)"]
        check("C3 the geodesic beats its own flat chord (the sphere earns itself)",
              d_th >= d_ch,
              "d_theta=%+.4f  d_chord=%+.4f  -> chord wins by %+.2f%%. The chord "
              "is the TANGENT reading; it needs no sphere." % (d_th, d_ch, 100 * (d_ch / d_th - 1)))
        check("C4 the geodesic beats sin(theta) = sqrt(TV), a pure simplex statistic",
              d_th >= d_sn,
              "d_theta=%+.4f  d_sqrtTV=%+.4f -> sqrt(TV) wins by %+.2f%%.  |  all: %s"
              % (d_th, d_sn, 100 * (d_sn / d_th - 1),
                 "  ".join("%s=%+.4f" % (nm, v) for nm, v in ds.items())))

        # ---- C5: rank identity at ROW level, the level the geometry claims ----
        rho = spearman(allth.tolist(), torch.cat([r["tvr"][ordinary(r)] for r in cs]).tolist())
        check("C5 theta ranks ROWS differently from TV (rho < 0.9999)",
              rho < 0.9999,
              "spearman(theta_i, TV_i) over %d ordinary rows pooled across %d "
              "draws = %.8f" % (allth.numel(), n, rho))

        log.append(dict(probe="curvature", k=k, s=s, d=d, draws=n, threads=2,
                        seeds=[0, 999], ident_resid_ordinary=e,
                        sat_rows_mean=sum(nsat) / len(nsat), sat_rows_max=max(nsat),
                        theta_p50=float(q[0]), theta_p90=float(q[1]),
                        theta_p99=float(q[2]), theta_max=float(q[4]),
                        curvature_share=num / den, row_rho=rho,
                        effects={nm: v for nm, v in ds.items()}))
        print()

    with JOURNAL.open("a") as fh:
        for e in log:
            fh.write(json.dumps(e) + "\n")

    print("=" * 72)
    print("RED checks (%d of %d):" % (len(FAILS), 5 * len(ks)))
    for f in dict.fromkeys(FAILS):
        print("  " + f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())

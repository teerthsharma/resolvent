"""CAMERON, round 5 -- the AGGREGATOR probe. Measurement only, nothing built.

THE OBJECTION. ARM A's K3 compares `float(theta_rows(...).mean())` against
`float(tv_rows(...).mean())`. Both sides are a MEAN OVER s=1024 ROWS. Under the
causal mask `tril(-1)` row `i` attends only to `j < i`, so masking token `c`
cannot move ANY row with `i <= c`: those rows are zero BY CONSTRUCTION, not by
measurement. The published mean therefore divides a signal carried by `s-1-c`
rows by `s`, with `c` uniform over the pivot set -- a dilution whose factor is
itself a random variable. K3's 2.3% margin may be a fact about `.mean()` and
not about the sphere.

WHAT IS MEASURED. On the IDENTICAL draws (same protocol, same seeds as
`scale/arm_a_run.py`, verified by reproducing its published `theta_c`), six
aggregators applied to BOTH theta and TV -- if the aggregator changes, it
changes for TV too or the comparison is rigged -- plus `tau` and `gamma_1`,
which `arm_a_run.py` already computes for the causal arm and NEVER compares
against the filler arm.

    mean_all  mean over all s rows            <- the incumbent, arm_a_run.py
    mean      mean over ACTIVE rows i > c     <- the incumbent, undiluted
    max       max over active rows
    p99       0.99 quantile over active rows
    l2        L2 norm of the active row vector
    cnt       #rows above a threshold calibrated on HELD-OUT FILLER draws

ACTIVE ROWS ARE `i > c`, both statistics, no exceptions. Restricting to them
also removes row 0, which is degenerate: `_softmax_operator` zeroes row 0
entirely (no `j < 0` exists), so both readings are the zero vector, the
Bhattacharyya coefficient is 0, and `theta_rows` returns `arccos(0) = pi/2`
for row 0 of EVERY draw -- a constant that TV does not carry.

DRAW PROTOCOL: `scale/arm_a_run.py::one` reproduced call-for-call, including
the two `randn(d)` draws its flip half consumes, so the generator stream stays
aligned and the draws ARE the published draws. `c` and `j` uniformly at random,
never on a schedule.
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

from scale.pivot_probe import select_pivots                       # noqa: E402
from scale.torque_probe import (rows_with_and_without, theta_rows,  # noqa: E402
                                tv_rows, torque, shadow, cohen_d)

ROOT = pathlib.Path(__file__).resolve().parents[1]
JOURNAL = ROOT / "results" / "cameron_aggregators.jsonl"

AGGS = ("mean_all", "mean", "max", "p99", "l2", "cnt")
ROWSTATS = ("th", "tv", "th64", "tv64")
STATS = [f"{s}_{a}" for s in ROWSTATS for a in AGGS] + ["tau", "gamma"]
INCUMBENT = "th_mean_all"          # exactly what arm_a_run.py reports as theta
RAW_TV = "tv_mean_all"             # exactly what arm_a_run.py reports as tv


def draw(s: int, k: int, d: int, g: torch.Generator, *, filler: bool,
         heavy: bool = True):
    """One draw of `arm_a_run.one`'s protocol. Returns the ROW VECTORS.

    Generator consumption is identical to `arm_a_run.one`: x0, wq, wk, wo, v0,
    j, c, then the flip half's two `randn(d)` c-values. The flip half's autograd
    is not run (K1 is measured elsewhere); only its DRAWS are consumed, so the
    stream and therefore every (q, k, c, i, j) is bit-identical.
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
        return None                            # arm_a_run returns here too
    c = pool[int(torch.randint(0, len(pool), (1,), generator=g))]

    a_c, a_0 = rows_with_and_without(q, kk, c)
    th, tv = theta_rows(a_c, a_0), tv_rows(a_c, a_0)
    # SAME draw, same q and k, arithmetic in float64. Applied to BOTH
    # statistics: an accuracy fix given to theta alone would rig K3.
    b_c, b_0 = rows_with_and_without(q.double(), kk.double(), c)
    th64, tv64 = theta_rows(b_c, b_0), tv_rows(b_c, b_0)
    tau = gam = float("nan")
    if heavy:                                  # the two O(s^3) reads
        tau = torque(a_c, a_0)
        sv = shadow(a_c, a_0)[1]
        gam = float(sv[0] - sv[1]) if sv.numel() > 1 else float("nan")

    for _ in range(2):                         # the flip half's two c-values
        torch.randn(d, generator=g)
    return dict(c=c, j=j, th=th, tv=tv, th64=th64, tv64=tv64, tau=tau, gamma=gam)


def aggs(v: torch.Tensor, c: int, thr: float) -> dict:
    """Every aggregator. `mean_all` is the incumbent; the rest are on i > c."""
    a = v[c + 1:]
    return {"mean_all": float(v.mean()), "mean": float(a.mean()),
            "max": float(a.max()), "p99": float(torch.quantile(a, 0.99)),
            "l2": float(a.norm()), "cnt": float((a > thr).sum())}


def calibrate(s: int, k: int, d: int, seed: int, n: int, q: float) -> dict:
    """Thresholds from HELD-OUT FILLER draws -- never from the measured cells."""
    g = torch.Generator().manual_seed(seed)
    pool = {st: [] for st in ROWSTATS}
    while len(pool["th"]) < n:
        r = draw(s, k, d, g, filler=True, heavy=False)
        if r is None:
            continue
        for st in ROWSTATS:
            pool[st].append(r[st][r["c"] + 1:].double())
    return {st: float(torch.quantile(torch.cat(v), q)) for st, v in pool.items()}


def cell(s: int, k: int, n: int, d: int, seed: int, filler: bool,
         thr: dict) -> dict:
    g = torch.Generator().manual_seed(seed)
    rows = []
    while len(rows) < n:
        r = draw(s, k, d, g, filler=filler)
        if r is None:
            continue
        rec = dict(c=r["c"], tau=r["tau"], gamma=r["gamma"])
        # how many rows that DID move does the float32 angle read as exactly 0
        act = slice(r["c"] + 1, None)
        rec["floored"] = float(((r["th"][act] == 0) & (r["tv"][act] > 0)).sum())
        rec["nactive"] = float(s - r["c"] - 1)
        for st in ROWSTATS:
            rec[st] = aggs(r[st], r["c"], thr[st])
        rows.append(rec)
    out = {q: [r[q] for r in rows] for q in ("c", "tau", "gamma", "floored", "nactive")}
    for st in ROWSTATS:
        for a in AGGS:
            out[f"{st}_{a}"] = [r[st][a] for r in rows]
    return out


# ------------------------------------------------------------------ analysis
def boot_d(cc: dict, ff: dict, b: int = 2000, seed: int = 0):
    """Cohen's d per statistic with a PAIRED bootstrap: one resample of the
    causal indices and one of the filler indices feeds EVERY statistic, so
    `|d_x| - |d_incumbent|` has a CI that respects the shared draws."""
    n = len(cc[STATS[0]])
    C = torch.tensor([cc[s] for s in STATS], dtype=torch.float64)
    F = torch.tensor([ff[s] for s in STATS], dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    ic = torch.randint(0, n, (b, n), generator=g)
    if_ = torch.randint(0, n, (b, n), generator=g)
    Cb, Fb = C[:, ic], F[:, if_]                       # [S, b, n]
    sp = (((n - 1) * Cb.var(-1) + (n - 1) * Fb.var(-1)) / (2 * n - 2)).sqrt()
    db = (Cb.mean(-1) - Fb.mean(-1)) / sp.clamp_min(1e-300)   # [S, b]
    inc = STATS.index(INCUMBENT)
    delta = db.abs() - db.abs()[inc]
    lo, hi = int(0.025 * b), int(0.975 * b)
    out = {}
    for idx, s in enumerate(STATS):
        ds = db[idx].sort().values
        de = delta[idx].sort().values
        out[s] = dict(d=cohen_d(cc[s], ff[s]),
                      d_lo=float(ds[lo]), d_hi=float(ds[hi]),
                      dd=float(de[b // 2]), dd_lo=float(de[lo]),
                      dd_hi=float(de[hi]))
    return out


def analyze(tag: str):
    lines = [json.loads(l) for l in JOURNAL.read_text().splitlines() if l.strip()]
    lines = [l for l in lines if l.get("tag") == tag]
    if not lines:
        print(f"NOT FOUND: no journal lines with tag={tag} in {JOURNAL}")
        return 1
    print(f"=== CAMERON aggregator table.  tag={tag}  file={JOURNAL}")
    for L in lines:
        cz = L["causal"]
        print(f"\n--- k={L['k']} s={L['s']} d={L['d']} draws={L['draws']} "
              f"seed_c={L['seed_c']} seed_f={L['seed_f']} threads={L['threads']}")
        print(f"    cnt thresholds (held-out filler, q={L['calib_q']}): "
              + " ".join(f"{s}>{v:.6g}" for s, v in L["thr"].items()))
        fz = L["filler"]
        # CONFOUND CHECK. `max`, `l2` and `cnt` all grow with the number of rows
        # that CAN move (i > c). If causal and filler draws differed in that
        # count the win would be about c's position, not about consequence.
        print(f"    rows that can move (i>c): causal "
              f"{sum(cz['nactive']) / len(cz['nactive']):.1f} filler "
              f"{sum(fz['nactive']) / len(fz['nactive']):.1f} of {L['s']}  "
              f"Cohen d={cohen_d(cz['nactive'], fz['nactive']):+.4f}")
        print(f"    float32 theta reads exactly 0 on "
              f"{sum(cz['floored']) / max(1e-9, sum(cz['nactive'])) * 100:.1f}% "
              f"(causal) / "
              f"{sum(fz['floored']) / max(1e-9, sum(fz['nactive'])) * 100:.1f}% "
              f"(filler) of rows TV calls nonzero")
        r = boot_d(L["causal"], L["filler"])
        base = abs(r[INCUMBENT]["d"])
        print(f"  {'statistic':<14} {'causal mean':>12} {'filler mean':>12} "
              f"{'Cohen d':>9} {'d 95% CI':>20} {'|d|-|d_inc|':>12} "
              f"{'delta 95% CI':>20}  vs_incumbent")
        for s in STATS:
            cm = sum(L["causal"][s]) / len(L["causal"][s])
            fm = sum(L["filler"][s]) / len(L["filler"][s])
            e = r[s]
            beats = ("WINS" if e["dd_lo"] > 0 else
                     "loses" if e["dd_hi"] < 0 else "tie(CI spans 0)")
            if s == INCUMBENT:
                beats = "-- incumbent --"
            print(f"  {s:<14} {cm:>12.6g} {fm:>12.6g} {e['d']:>+9.4f} "
                  f"[{e['d_lo']:+7.4f},{e['d_hi']:+7.4f}] {e['dd']:>+12.4f} "
                  f"[{e['dd_lo']:+7.4f},{e['dd_hi']:+7.4f}]  {beats}")
        print(f"  incumbent |d|={base:.4f}  raw-TV |d|={abs(r[RAW_TV]['d']):.4f}  "
              f"K3 margin={100 * (base / abs(r[RAW_TV]['d']) - 1):+.2f}%")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=int, default=1024)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--draws", type=int, default=120)
    ap.add_argument("--k", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fseed", type=int, default=999)
    ap.add_argument("--calib-draws", type=int, default=40)
    ap.add_argument("--calib-q", type=float, default=0.99)
    ap.add_argument("--tag", default="primary")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()

    if a.analyze:
        return analyze(a.tag)
    if a.k is None:
        print("--k is required (one cell per process: ADR-001 bucketing)")
        return 2

    print(f"CAMERON aggregator probe -- s={a.s} d={a.d} k={a.k} draws={a.draws} "
          f"seed_c={a.seed} seed_f={a.fseed} threads={torch.get_num_threads()} "
          f"tag={a.tag}")
    cseed = 424242 + a.k
    thr = calibrate(a.s, a.k, a.d, cseed, a.calib_draws, a.calib_q)
    print(f"  thresholds from {a.calib_draws} HELD-OUT filler draws "
          f"(seed {cseed}, q={a.calib_q}): "
          + " ".join(f"{s}>{v:.6g}" for s, v in thr.items()))
    cc = cell(a.s, a.k, a.draws, a.d, a.seed, False, thr)
    ff = cell(a.s, a.k, a.draws, a.d, a.fseed, True, thr)
    rec = {"agent": "cameron", "tag": a.tag, "s": a.s, "d": a.d, "k": a.k,
           "draws": a.draws, "seed_c": a.seed, "seed_f": a.fseed,
           "threads": torch.get_num_threads(), "calib_seed": cseed,
           "calib_draws": a.calib_draws, "calib_q": a.calib_q,
           "thr": thr, "causal": cc, "filler": ff}
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    tm = sum(cc[INCUMBENT]) / a.draws
    fm = sum(ff[INCUMBENT]) / a.draws
    print(f"  BIND vs published arm_a.jsonl: theta_c={tm:.15g} theta_f={fm:.15g} "
          f"d_theta={cohen_d(cc[INCUMBENT], ff[INCUMBENT]):.15g} "
          f"d_tv={cohen_d(cc[RAW_TV], ff[RAW_TV]):.15g}")
    print(f"  cell journalled to {JOURNAL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

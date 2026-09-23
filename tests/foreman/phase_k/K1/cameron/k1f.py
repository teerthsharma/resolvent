"""K1.F floor row (Cameron, it.K1): position-only floor, chance-credited L-doubling, C2 hybrid (positional oracle,
window priced at R0 width), on bed_k at n in {1024 train, 4096, 8192, 16384}, L in {4, 7}. Bars: test_k1f.py.
Seeds: train default_rng([11, k]), test default_rng([12, n, k]), k = 0..7.  `--stub` writes a failing stub."""
import json, sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from bed_k import make_train, make_test, floor_recency, ceiling_credit, _last_root
from shortcut import hybrid, doubling

LS = (4, 7)
NS = (1024, 4096, 8192, 16384)
EDGES = [0, 17, 33, 65, 129, 257, 513, 1025, 10 ** 9]
BUDGET = [(3, W) for W in range(1, 11)] + [(4, W) for W in range(1, 7)]    # R0: 12W <= 128 (k=3), 12W^2 <= 512 (k=4)
SWEEP = [(3, W) for W in (12, 16, 24, 32, 48, 64, 128)]                    # outside the R0 budget: unbarred


def beds(n):
    if n == 1024:
        return [make_train(np.random.default_rng([11, k])) for k in range(8)]
    return [make_test(np.random.default_rng([12, n, k]), n) for k in range(8)]


def q(v):
    return [float(np.mean(v)), float(np.min(v)), float(np.max(v))]


def acc(ok, d, L):
    b = d > 2 ** L
    return float(ok.mean()), (float(ok[b].mean()) if b.any() else None)


def main():
    out = {"meta": {"budget": [f"k{k}_W{W}" for k, W in BUDGET], "unbarred_sweep": [f"k{k}_W{W}" for k, W in SWEEP],
                    "credit": "unresolved -> most recent root at or before final pointer", "positional_channel": "oracle"},
           "rows": {}, "machinery": {"k2_vs_doubling_mismatches": 0, "credit_below_zero_violations": 0}}
    mm = out["machinery"]
    for n in NS:
        bs = beds(n)
        row = {"depth_hist": {f"{lo}-{hi - 1}": float(np.mean([((b['depth'] >= lo) & (b['depth'] < hi)).sum() for b in bs]))
                              for lo, hi in zip(EDGES[:-1], EDGES[1:])},
               "max_depth": q([b["depth"].max() for b in bs]), "roots": q([len(np.unique(b["root"])) for b in bs])}
        for L in LS:
            per = {"recency": [], "doubling_credited": [], "doubling_zero": []}
            cfgs = {}
            for b in bs:
                p, d, r = b["parent"], b["depth"], b["root"]
                lr = _last_root(p)
                per["recency"].append(acc(floor_recency(p, r)[1] == r, d, L))
                A = doubling(p, L)
                per["doubling_zero"].append(acc(A == r, d, L))
                per["doubling_credited"].append(acc(lr[A] == r, d, L))
                for mode in ("anchored", "centered"):
                    mm["k2_vs_doubling_mismatches"] += int((hybrid(p, d, L, 5, 2, mode) != A).sum())
                    for k, W in BUDGET + SWEEP:
                        H = hybrid(p, d, L, W, k, mode)
                        z, c = acc(H == r, d, L), acc(lr[H] == r, d, L)
                        mm["credit_below_zero_violations"] += int(c[0] < z[0] or (c[1] is not None and c[1] < z[1]))
                        cfgs.setdefault(f"{mode}_k{k}_W{W}", []).append((z, c))
            R = {}
            for key, v in per.items():
                R[key] = {"all": q([x[0] for x in v]), "beyond": None if v[0][1] is None else q([x[1] for x in v])}
            sweep = {}
            for key, v in cfgs.items():
                sweep[key] = {"all_zero": q([x[0][0] for x in v]), "all": q([x[1][0] for x in v]),
                              "beyond_zero": None if v[0][0][1] is None else q([x[0][1] for x in v]),
                              "beyond": None if v[0][1][1] is None else q([x[1][1] for x in v])}
            inb = {f"{mo}_k{k}_W{W}" for mo in ("anchored", "centered") for k, W in BUDGET}
            best = {}
            for met in ("all", "beyond"):
                cand = [(sweep[kk][met][0], kk) for kk in inb if sweep[kk][met] is not None]
                if cand:
                    v, kk = max(cand)
                    best[met], best[met + "_cfg"] = sweep[kk][met], kk
                else:
                    best[met], best[met + "_cfg"] = None, None
            R["c2_best"] = best
            R["c2_sweep"] = sweep
            R["line"] = {}
            for met in ("all", "beyond"):
                comps = [R["recency"][met], R["doubling_credited"][met], R["c2_best"][met]]
                R["line"][met] = None if any(c is None for c in comps) else max(c[0] for c in comps)
                R["line"][met + "_arg"] = None if R["line"][met] is None else \
                    ["recency", "doubling_credited", "c2_best"][int(np.argmax([c[0] for c in comps]))]
            row[f"L{L}"] = R
        out["rows"][str(n)] = row
    b = make_test(np.random.default_rng([12, 4096, 0]), 4096)
    p, d, r = b["parent"], b["depth"], b["root"]
    deep = d > 16
    out["probe_n4096_k0"] = {"recency_all": floor_recency(p, r)[0], "doubling_credited_L4_beyond": ceiling_credit(p, d, r, 4, deep),
                             "c2_anchored_k3_W8_L4_credited_beyond": float((_last_root(p)[hybrid(p, d, 4, 8, 3, "anchored")] == r)[deep].mean())}
    return out


def stub():
    nan = float("nan")
    cell = {"recency": {"all": [nan] * 3, "beyond": [nan] * 3}, "doubling_credited": {"all": [nan] * 3, "beyond": [nan] * 3},
            "c2_best": {"all": [1.0] * 3, "beyond": [1.0] * 3, "beyond_cfg": "stub"}, "line": {"all": None, "beyond": None}}
    return {"machinery": {"k2_vs_doubling_mismatches": -1, "credit_below_zero_violations": -1},
            "probe_n4096_k0": {"recency_all": nan, "doubling_credited_L4_beyond": nan, "c2_anchored_k3_W8_L4_credited_beyond": nan},
            "rows": {str(n): {"L4": cell, "L7": cell} for n in NS}}


if __name__ == "__main__":
    R = stub() if "--stub" in sys.argv else main()
    (HERE / "k1f.json").write_text(json.dumps(R, indent=1))
    if "--stub" not in sys.argv:
        for n, row in R["rows"].items():
            print(f"n={n} roots {row['roots'][0]:.1f} max_depth {row['max_depth'][0]:.0f} hist {row['depth_hist']}")
            for L in LS:
                c = row[f"L{L}"]
                f = lambda x: "  --  " if x is None else f"{x[0]:.4f}"
                print(f"  L={L} recency {f(c['recency']['all'])}/{f(c['recency']['beyond'])}  dbl_cred {f(c['doubling_credited']['all'])}/"
                      f"{f(c['doubling_credited']['beyond'])}  C2 {f(c['c2_best']['all'])}[{c['c2_best']['all_cfg']}]/"
                      f"{f(c['c2_best']['beyond'])}[{c['c2_best']['beyond_cfg']}]  LINE {c['line']}")

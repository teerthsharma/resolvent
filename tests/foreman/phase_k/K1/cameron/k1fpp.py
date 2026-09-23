"""K1.F'' producer: bed_k' floor row on the far band (Amendment K1-a). Bars: test_k1fpp.py. `--stub` writes a failing stub."""
import json, sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import bed_k, bed_kp            # my lane first (identical sha to the repo K0 copies hunt.py would load)
import bedkp as B
from k1fp import hpd_analytic, resolvent_ok, BUDGET
sys.path.insert(1, str(HERE.parent / "foreman"))
import kp_attack as K           # Foreman's sc-HPD, read-only

BAND10 = "--band10" in sys.argv   # K1.F''': the band at 10 2^L (c_max 5.0, if Foreman's W=62 binds); fresh seeds [33|34, n, k]
BAND = {4: 160, 7: 1280} if BAND10 else {4: 96, 7: 896}
CELLS = [(4096, 4), (8192, 4), (16384, 4), (16384, 7)] if BAND10 else [(4096, 4), (8192, 4), (16384, 4), (8192, 7), (16384, 7)]
SEEDS = (33, 34) if BAND10 else (31, 32)
OUT = "k1fppp.json" if BAND10 else "k1fpp.json"
MINE = ("anchored", "centered", "hpd_analytic", "hpd_empirical", "local")
MY_CFG = BUDGET + [(3, 14), (3, 30)]
FORE_CFG = [(W, 0) for W in list(range(1, 11)) + [14, 30] + ([62] if BAND10 else [])] + [(W1, 10 - W1) for W1 in range(1, 10)]


def fore_final(b, L, W1, W2, tables):
    A = K.pointer0(b["parent"])
    for t2, t3 in tables:
        A = K.step(A, b["depth"], W1, W2, t2, t3)
    return A


def main():
    mm = {"credit_below_zero_violations": 0, "band": {str(k): v for k, v in BAND.items()}, "roots_all_8": True, "ids_distinct": True}
    old = [bed_kp.make_test(np.random.default_rng([22, 4096, k]), 4096) for k in range(8)]
    oldc = [bed_kp.make_test(np.random.default_rng([23, 4096, k]), 4096) for k in range(8)]
    mm["foreman_schpd_W10_4096_L7"] = K.run(old, 7, 10, 0, K.calibrate(oldc, 7, 10, 0))[0]
    out = {"machinery": mm, "cells": {}, "unbarred": {}}
    cache = {}
    for n, L in CELLS:
        if n not in cache:
            cache[n] = ([bed_kp.make_test(np.random.default_rng([SEEDS[0], n, k]), n) for k in range(8)],
                        [bed_kp.make_test(np.random.default_rng([SEEDS[1], n, k]), n) for k in range(8)])
        beds, cal = cache[n]
        for b in beds:
            mm["roots_all_8"] &= len(np.unique(b["root"])) == 8
            mm["ids_distinct"] &= len(np.unique(b["ids"])) == len(b["ids"])
        band = [b["depth"] > BAND[L] for b in beds]
        laws = B.offset_law(cal, 600)
        fam = {}

        def score(name, finals):   # finals: list of final pointer arrays, one per bed
            per = []
            for A, b, m in zip(finals, beds, band):
                cr = bed_k._last_root(b["parent"])[A] == b["root"]
                z = A == b["root"]
                mm["credit_below_zero_violations"] += int(cr[m].mean() < z[m].mean())
                per.append(float(cr[m].mean()))
            fam[name] = [float(np.mean(per)), float(np.min(per)), float(np.max(per))]

        fam["recency"] = None
        per = [float(B.recency(b["parent"], b["root"])[m].mean()) for b, m in zip(beds, band)]
        fam["recency"] = [float(np.mean(per)), float(np.min(per)), float(np.max(per))]
        score("doubling", [B.doubling(b["parent"], L) for b in beds])
        for f in MINE:
            for k, W in MY_CFG:
                if f == "hpd_analytic":
                    fin = [hpd_analytic(b["parent"], b["depth"], L, W, k, 8) for b in beds]
                elif f == "hpd_empirical":
                    tab = B.hpd_table(laws, W, 8)
                    fin = [B.window(b["parent"], b["depth"], L, W, k, "hpd", 8, tab) for b in beds]
                else:
                    fin = [B.window(b["parent"], b["depth"], L, W, k, f, 8) for b in beds]
                score(f"{f}_k{k}W{W}", fin)
        for W1, W2 in FORE_CFG:
            tabs = K.calibrate(cal, L, W1, W2)
            score(f"foreman_schpd_W{W1}+{W2}", [fore_final(b, L, W1, W2, tabs) for b in beds])
        line = max(v[0] for v in fam.values())
        res = [float(resolvent_ok(b["parent"], b["root"], 20.0, 0.999)[m].mean()) for b, m in zip(beds, band)]
        out["cells"][f"{n}_{L}"] = {"band_threshold": BAND[L], "band_tokens": int(sum(m.sum() for m in band)),
                                    "band_frac": float(np.mean([m.mean() for m in band])), "families": fam, "line": line,
                                    "line_arg": max(fam, key=lambda k: fam[k][0]),
                                    "resolvent": [float(np.mean(res)), float(np.min(res)), float(np.max(res))]}
        tabs = K.calibrate(cal, L, 62, 0)
        fin = [fore_final(b, L, 62, 0, tabs) for b in beds]
        out["unbarred"][f"{n}_{L}_schpd_W62"] = float(np.mean([(bed_k._last_root(b["parent"])[A] == b["root"])[m].mean()
                                                                for A, b, m in zip(fin, beds, band)]))
        c = out["cells"][f"{n}_{L}"]
        print(f"n={n} L={L} band>{BAND[L]} tokens {c['band_tokens']} line {line:.4f} ({c['line_arg']}) resolvent {c['resolvent'][0]:.4f} "
              f"W62 {out['unbarred'][f'{n}_{L}_schpd_W62']:.4f}", flush=True)
    return out


def stub():
    nan = float("nan")
    return {"machinery": {"foreman_schpd_W10_4096_L7": nan, "credit_below_zero_violations": -1, "band": {}, "roots_all_8": False,
                          "ids_distinct": False},
            "cells": {f"{n}_{L}": {"band_tokens": 0, "families": {"stub": [1.0, 1.0, 1.0]}, "line": nan, "resolvent": [nan] * 3}
                      for n, L in CELLS}}


if __name__ == "__main__":
    R = stub() if "--stub" in sys.argv else main()
    (HERE / OUT).write_text(json.dumps(R, indent=1))

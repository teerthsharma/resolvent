# UNBARRED exploration, round 2 (design search). Seeds [93, M, n, k] (beds), [94, M, n, k] (calibration).
import sys, json, time
import numpy as np
sys.dont_write_bytecode = True; sys.path.insert(0, ".")
import bedkp as B
from bed_k import resolvent_rec
R0 = [(3, W) for W in range(1, 11)] + [(4, W) for W in range(1, 7)]
def run(M, n, cap=None, Ls=(4, 7), nb=4, res=False):
    cal = [B.make("geo", n, M, np.random.default_rng([94, M, n, k]), cap) for k in range(8)]
    beds = [B.make("geo", n, M, np.random.default_rng([93, M, n, k]), cap) for k in range(nb)]
    laws = B.offset_law(cal, 400)
    out = {"maxdepth": float(np.mean([b["depth"].max() for b in beds])), "roots": float(np.mean([len(np.unique(b["root"])) for b in beds]))}
    for L in Ls:
        deep = [b["depth"] > 2 ** L for b in beds]
        if sum(x.sum() for x in deep) < 50:
            continue
        f = lambda A_of: float(np.mean([B.credited(b["parent"], b["root"], A_of(b))[m].mean() for b, m in zip(beds, deep) if m.any()]))
        best = {"frac": float(np.mean([m.mean() for m in deep])), "rec": float(np.mean([B.recency(b["parent"], b["root"])[m].mean() for b, m in zip(beds, deep) if m.any()])),
                "dbl": f(lambda b: B.doubling(b["parent"], L))}
        for mode in ("anchored", "centered", "hpd", "local"):
            v = []
            for k, W in R0:
                h = B.hpd_table(laws, W, M) if mode == "hpd" else None
                v.append((round(f(lambda b: B.window(b["parent"], b["depth"], L, W, k, mode, M, h)), 4), f"k{k}W{W}"))
            best[mode] = max(v)
        out[f"L{L}"] = best
    if res:
        out["res_g999_s16_20_24"] = [list(np.round(resolvent_rec(b["parent"], b["root"], [16, 20, 24], 0.999), 4)) for b in beds[:2]]
    return out
if __name__ == "__main__":
    t = time.time()
    for M, n, cap in [(32, 1024, 32), (24, 1024, 32), (16, 1024, 32)]:
        print("train", M, n, json.dumps(run(M, n, cap, Ls=(4,), nb=16)), f"{time.time()-t:.0f}s", flush=True)
    for M in (6, 8):
        for n in (1024, 4096, 8192, 16384):
            print("test", M, n, json.dumps(run(M, n, res=(n in (4096, 16384)))), f"{time.time()-t:.0f}s", flush=True)

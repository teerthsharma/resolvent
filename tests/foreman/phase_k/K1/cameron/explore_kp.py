# UNBARRED exploration (design search for bed_k'); exploration seeds [91, gen, M, n, k] never reused by K1.F'.
import sys, json, time
import numpy as np
sys.dont_write_bytecode = True; sys.path.insert(0, ".")
import bedkp as B
sys.path.insert(0, "../foreman")
GI = {"geo": 0, "flat": 1}
R0 = [(3, W) for W in range(1, 11)] + [(4, W) for W in range(1, 7)]
def run(gen, M, n, Ls=(4, 7), nb=4):
    cal = [B.make(gen, n, M, np.random.default_rng([92, GI[gen], M, n, k])) for k in range(8)]
    beds = [B.make(gen, n, M, np.random.default_rng([91, GI[gen], M, n, k])) for k in range(nb)]
    mu = n / np.mean([len(np.unique(b["root"])) for b in beds]) if False else M
    laws = B.offset_law(cal, 300)
    out = {"maxdepth": float(np.mean([b["depth"].max() for b in beds])),
           "J1_W10_mass": float(np.sort(np.convolve(laws[1], np.ones(10), 'valid'))[-1] / laws[1].sum())}
    for L in Ls:
        best = {}
        deep = [b["depth"] > 2 ** L for b in beds]
        if not any(x.any() for x in deep):
            continue
        f = lambda A_of: float(np.mean([B.credited(b["parent"], b["root"], A_of(b))[m].mean() for b, m in zip(beds, deep)]))
        best["dbl"] = f(lambda b: B.doubling(b["parent"], L))
        for mode in ("anchored", "centered", "hpd", "local"):
            v = []
            for k, W in R0:
                h = B.hpd_table(laws, W, M) if mode == "hpd" else None
                v.append((f(lambda b: B.window(b["parent"], b["depth"], L, W, k, mode, M, h)), f"k{k}W{W}"))
            best[mode] = max(v)
        out[f"L{L}"] = best
    return out
if __name__ == "__main__":
    t = time.time()
    for gen, M, n in [("geo", 16, 4096)] + [(g, M, 4096) for g in ("geo", "flat") for M in (8, 10, 12)] + [("flat", 16, 4096)]:
        print(gen, M, n, json.dumps(run(gen, M, n)), f"{time.time()-t:.0f}s", flush=True)

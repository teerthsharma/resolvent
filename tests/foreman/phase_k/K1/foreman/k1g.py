# Foreman K1 seventh set: reach factors of every known window family, and the far-band property. Bars: test_k1g.py.
import json, sys
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from hunt import bed_k, doubling, hybrid_space  # noqa: E402
from k1b import hpd_hybrid                      # noqa: E402

N = 32768
PRICING = {"R0": [(3, W) for W in range(1, 11)] + [(4, W) for W in range(1, 7)],
           "R1": [(3, W) for W in range(1, 27)] + [(4, W) for W in range(1, 11)]}
BEDS = [bed_k.make_test(np.random.default_rng([55, N, k]), N) for k in range(4)]
DEPTH = np.concatenate([b["depth"] for b in BEDS])


def pooled_ok(fn):
    return np.concatenate([bed_k._last_root(b["parent"])[fn(b)] == b["root"] for b in BEDS])


def reach(ok, L):
    w = max(1, 2 ** L // 8)
    D = 0
    for d in range(w, DEPTH.max() + 1, w):
        m = (DEPTH > d - w) & (DEPTH <= d)
        if m.sum() and ok[m].mean() >= 0.5:
            D = d
    return D / 2 ** L


def families(L, pricing):
    out = {"hpd": [], "local": []}
    for k, W in PRICING[pricing]:
        out["hpd"].append(pooled_ok(lambda b: hpd_hybrid(b["parent"], b["depth"], L, W, k)))
        out["local"].append(pooled_ok(lambda b: hybrid_space(b["parent"], b["depth"], np.arange(N), L, W, k, "anchored",
                                                             anchor_self=True)[0]))
    return out


def main():
    res = {"reach": {}, "far": {}, "detail": {}}
    for L in (4, 6, 7):
        res["reach"][f"doubling_{L}"] = reach(pooled_ok(lambda b: doubling(b["parent"], L)), L)
    for L, pricing in ((4, "R0"), (6, "R0"), (7, "R0"), (6, "R1")):
        fam = families(L, pricing)
        ch = max(reach(ok, L) for ok in fam["hpd"])
        cl = max(reach(ok, L) for ok in fam["local"])
        key = f"{L}_{pricing}"
        res["reach"][f"hpd_{key}"], res["reach"][f"local_{key}"] = ch, cl
        kappa = 2 * max(ch, cl)
        far = DEPTH > kappa * 2 ** L
        dbl = pooled_ok(lambda b: doubling(b["parent"], L))
        accs = [float(ok[far].mean()) for ok in fam["hpd"] + fam["local"] + [dbl]] if far.any() else [float("nan")]
        res["far"][key] = {"kappa": kappa, "tokens": int(far.sum()), "worst": max(accs),
                           "hpd_best_far": max(accs[:len(fam["hpd"])]), "doubling_far": accs[-1]}
    (HERE / "k1g.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))

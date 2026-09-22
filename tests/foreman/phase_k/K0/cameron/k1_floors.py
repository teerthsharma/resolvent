"""K1 R-DEPTH floors, computed at K0 before any arm exists (floor rule). Beds from bed_k (author's conventions).
Seeds: train default_rng([1, k]); test default_rng([2, n, k]); k = 0..7."""
import sys, json, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0] if '/' in __file__ else '.')
from bed_k import make_train, make_test, floor_recency, ceiling_exact, ceiling_credit, resolvent_rec

GRID = np.arange(4.0, 24.01, 0.25)
LS = (3, 4, 5, 6, 7, 8, 9, 10, 11)


def train_beds():
    return [make_train(np.random.default_rng([1, k])) for k in range(8)]


def test_beds(n):
    return [make_test(np.random.default_rng([2, n, k]), n) for k in range(8)]


def s_needed(b, thr, gamma=0.999):
    a = resolvent_rec(b["parent"], b["root"], GRID, gamma)
    hit = np.nonzero(a >= thr)[0]
    return float(GRID[hit[0]]) if hit.size else float("inf")


def row(beds):
    P = lambda f: [f(b) for b in beds]
    q = lambda v: [float(np.min(v)), float(np.mean(v)), float(np.max(v))]
    r = {"n": len(beds[0]["ids"]), "roots": q(P(lambda b: len(np.unique(b["root"])))),
         "chance": q(P(lambda b: 1 / len(np.unique(b["root"])))),
         "max_depth": q(P(lambda b: b["depth"].max())),
         "recency": q(P(lambda b: floor_recency(b["parent"], b["root"])[0]))}
    for L in LS:
        beyond = P(lambda b: (b["depth"] > 2 ** L).mean())
        r[f"L{L}"] = {"exact": q(P(lambda b: ceiling_exact(b["depth"], L))),
                      "credited": q(P(lambda b: ceiling_credit(b["parent"], b["depth"], b["root"], L))),
                      "frac_beyond": q(beyond)}
        if min(beyond) > 0:
            r[f"L{L}"]["credited_beyond"] = q(P(lambda b: ceiling_credit(b["parent"], b["depth"], b["root"], L, b["depth"] > 2 ** L)))
            r[f"L{L}"]["recency_beyond"] = q(P(lambda b: float((floor_recency(b["parent"], b["root"])[1] == b["root"])[b["depth"] > 2 ** L].mean())))
    acc = np.array(P(lambda b: resolvent_rec(b["parent"], b["root"], GRID, 0.999)))
    s99 = [float(GRID[np.nonzero(a >= 0.99)[0][0]]) for a in acc]
    r["resolvent_ref"] = {"s_99": q(s99), "acc_s16": q(acc[:, GRID == 16.0][:, 0]), "acc_s20": q(acc[:, GRID == 20.0][:, 0])}
    return r


if __name__ == "__main__":
    out = {"train": row(train_beds())}
    for n in (4096, 8192, 16384):
        out[f"test_{n}"] = row(test_beds(n))
    json.dump(out, open("k1_floors.json", "w"), indent=1)
    for k, r in out.items():
        m = lambda x: f"{x[1]:.4f}"
        print(f"{k}: roots {r['roots'][1]:.1f} chance {m(r['chance'])} max_depth {r['max_depth']} recency {m(r['recency'])} "
              f"| res s_99 {r['resolvent_ref']['s_99']} acc@16 {m(r['resolvent_ref']['acc_s16'])} acc@20 {m(r['resolvent_ref']['acc_s20'])}")
        for L in LS:
            c = r[f"L{L}"]
            print(f"   L={L:2d} exact {m(c['exact'])} credited {m(c['credited'])} beyond-frac {m(c['frac_beyond'])}"
                  + (f" credited_beyond {m(c['credited_beyond'])} [{c['credited_beyond'][0]:.4f},{c['credited_beyond'][2]:.4f}]"
                     f" recency_beyond {m(c['recency_beyond'])}" if "credited_beyond" in c else ""))

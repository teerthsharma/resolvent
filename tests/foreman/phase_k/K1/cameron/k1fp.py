"""K1.F' producer: floor row of bed_k' (bed_kp.py). Bars: test_k1fp.py. `--stub` writes a failing stub."""
import json, sys
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.special import gammaln
sys.dont_write_bytecode = True
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import bed_k
import bed_kp
import bedkp as B

BUDGET = [(3, W) for W in range(1, 11)] + [(4, W) for W in range(1, 7)]
FAM = ("anchored", "centered", "hpd_analytic", "hpd_empirical", "local")
EDGES = [0, 17, 33, 65, 129, 257, 513, 1025, 2049, 10 ** 9]


@lru_cache(maxsize=None)
def pmf(J, M):  # Foreman k1b.pmf with P = 1/M
    P = 1 / M
    smax = int(M * J + 60 * np.sqrt(J) * M ** 0.5 + 400)
    s = np.arange(smax + 1)
    out = np.zeros(smax + 1)
    v = s >= J
    sv = s[v]
    out[v] = np.exp(gammaln(sv) - gammaln(J) - gammaln(sv - J + 1) + J * np.log(P) + (sv - J) * np.log1p(-P))
    return out


@lru_cache(maxsize=None)
def hpd_start(J, W, M):  # Foreman k1b.hpd_start
    if J == 0:
        return 1
    c = np.concatenate([[0.0], np.cumsum(pmf(J, M))])
    mass = c[W:] - c[:-W]
    mass[0] = -1.0
    return int(np.argmax(mass))


def hpd_analytic(parent, depth, L, W, k, M):  # Foreman k1b.hpd_hybrid with P = 1/M
    A = B.pointer0(parent)
    for _ in range(L):
        v1 = A
        cur = A[v1]
        J = depth - depth[v1]
        a = np.array([hpd_start(int(j), W, M) for j in range(J.max() + 1)])[J]
        lo, hi = v1 - (a + W - 1), v1 - a + 1
        live = np.ones(len(A), bool)
        for _ in range(k - 2):
            inw = live & (((cur >= lo) & (cur < hi)) | (cur == v1))
            cur = np.where(inw, A[cur], cur)
            live = inw
        A = cur
    return A


def resolvent_ok(parent, root, s, gamma):
    """bed_k.resolvent_rec's forward substitution at one s, returning per-token correctness."""
    n = len(parent); roots = np.unique(root); col = {r: k for k, r in enumerate(roots)}
    es = np.exp(s); R = roots.size
    z = np.zeros((n, R)); S = np.zeros(R)
    for i in range(n):
        Z = i + es
        if parent[i] < 0:
            v = np.zeros(R); v[col[i]] = 1.0
            z[i] = (v + gamma * S / Z) / (1 - gamma * es / Z)
        else:
            z[i] = (gamma * (S + (es - 1) * z[parent[i]]) / Z) / (1 - gamma / Z)
        S += z[i]
    return roots[np.argmax(z, 1)] == root


def q(v):
    return [float(np.mean(v)), float(np.min(v)), float(np.max(v))]


def acc(ok, d, L):
    b = d > 2 ** L
    return float(ok.mean()), (float(ok[b].mean()) if b.any() else None)


def cell(beds, L, M, laws, mm):
    per = {"recency": [], "doubling_credited": [], "doubling_zero": []}
    cfgs = {f: {} for f in FAM}
    tabs = {W: B.hpd_table(laws, W, M) for W in range(1, 11)}
    for b in beds:
        p, d, r = b["parent"], b["depth"], b["root"]
        lr = bed_k._last_root(p)
        per["recency"].append(acc(B.recency(p, r), d, L))
        A = B.doubling(p, L)
        per["doubling_zero"].append(acc(A == r, d, L))
        per["doubling_credited"].append(acc(lr[A] == r, d, L))
        for f in FAM:
            if f in ("anchored", "centered", "local"):
                mm["k2_vs_doubling_mismatches"] += int((B.window(p, d, L, 5, 2, f, M) != A).sum())
            for k, W in BUDGET:
                if f == "hpd_analytic":
                    H = hpd_analytic(p, d, L, W, k, M)
                elif f == "hpd_empirical":
                    H = B.window(p, d, L, W, k, "hpd", M, tabs[W])
                else:
                    H = B.window(p, d, L, W, k, f, M)
                z, c = acc(H == r, d, L), acc(lr[H] == r, d, L)
                mm["credit_below_zero_violations"] += int(c[0] < z[0] or (c[1] is not None and c[1] < z[1]))
                cfgs[f].setdefault(f"k{k}W{W}", []).append(c)
    out = {key: {"all": q([x[0] for x in v]), "beyond": None if v[0][1] is None else q([x[1] for x in v])} for key, v in per.items()}
    out["families"] = {}
    for f in FAM:
        sw = {key: {"all": q([x[0] for x in v]), "beyond": None if v[0][1] is None else q([x[1] for x in v])} for key, v in cfgs[f].items()}
        if sw["k3W1"]["beyond"] is None:
            out["families"][f] = {"all": max((s["all"] for s in sw.values()), key=lambda x: x[0]), "beyond": None, "cfg": None}
        else:
            kk = max(sw, key=lambda key: sw[key]["beyond"][0])
            out["families"][f] = {"all": max((s["all"] for s in sw.values()), key=lambda x: x[0]), "beyond": sw[kk]["beyond"], "cfg": kk}
    comps = [out["recency"], out["doubling_credited"]] + [out["families"][f] for f in FAM]
    out["line"] = {met: (None if any(c[met] is None for c in comps) else max(c[met][0] for c in comps)) for met in ("all", "beyond")}
    out["frac_beyond"] = q([float((b["depth"] > 2 ** L).mean()) for b in beds])
    out["n_beyond_tokens"] = int(sum((b["depth"] > 2 ** L).sum() for b in beds))
    return out


def main():
    mm = {"k2_vs_doubling_mismatches": 0, "credit_below_zero_violations": 0}
    old = [bed_k.make_test(np.random.default_rng([12, 4096, k]), 4096) for k in range(8)]
    mm["hpd_port_M16_L7_n4096_k3W10"] = float(np.mean([(bed_k._last_root(b["parent"])[hpd_analytic(b["parent"], b["depth"], 7, 10, 3, 16)]
                                                          == b["root"])[b["depth"] > 128].mean() for b in old]))
    out = {"machinery": mm, "rows": {}, "resolvent": {}}
    for n in (1024, 4096, 8192, 16384):
        if n == 1024:
            beds = [bed_kp.make_train(np.random.default_rng([21, k])) for k in range(8)]
            cal = [bed_kp.make_train(np.random.default_rng([24, k])) for k in range(8)]
            M = bed_kp.M_TRAIN
        else:
            beds = [bed_kp.make_test(np.random.default_rng([22, n, k]), n) for k in range(8)]
            cal = [bed_kp.make_test(np.random.default_rng([23, n, k]), n) for k in range(8)]
            M = bed_kp.M_TEST
        laws = B.offset_law(cal, 600)
        row = {"role": "train" if n == 1024 else "test", "M": M,
               "depth_hist": {f"{lo}-{hi - 1}": float(np.mean([((b['depth'] >= lo) & (b['depth'] < hi)).sum() for b in beds]))
                              for lo, hi in zip(EDGES[:-1], EDGES[1:])},
               "max_depth": q([b["depth"].max() for b in beds]), "roots": q([len(np.unique(b["root"])) for b in beds])}
        for L in (4, 7):
            row[f"L{L}"] = cell(beds, L, M, laws, mm)
            print(f"n={n} L={L} line {row[f'L{L}']['line']} frac {row[f'L{L}']['frac_beyond'][0]:.3f}", flush=True)
        out["rows"][str(n)] = row
        if n > 1024:
            oks = [resolvent_ok(b["parent"], b["root"], 20.0, 0.999) for b in beds]
            ref = bed_k.resolvent_rec(beds[0]["parent"], beds[0]["root"], [20.0], 0.999)[0]
            assert abs(ref - oks[0].mean()) < 1e-12
            out["resolvent"][str(n)] = {"acc": q([o.mean() for o in oks]), "beyond16": q([o[b["depth"] > 16].mean() for o, b in zip(oks, beds)])}
        mm.setdefault("test_roots_all_8", True)
        if n > 1024:
            mm["test_roots_all_8"] &= all(len(np.unique(b["root"])) == 8 for b in beds)
        else:
            mm["train_max_depth"] = int(max(b["depth"].max() for b in beds))
        mm["ids_distinct"] = mm.get("ids_distinct", True) and all(len(np.unique(b["ids"])) == len(b["ids"]) for b in beds)
    return out


def stub():
    nan = float("nan"); t = [nan] * 3
    fam = {f: {"all": t, "beyond": [1.0] * 3, "cfg": "stub"} for f in FAM}
    c = {"recency": {"all": t, "beyond": t}, "doubling_credited": {"all": t, "beyond": t}, "families": fam,
         "line": {"all": nan, "beyond": 1.0}, "frac_beyond": [0.0] * 3}
    return {"machinery": {"hpd_port_M16_L7_n4096_k3W10": nan, "k2_vs_doubling_mismatches": -1, "credit_below_zero_violations": -1,
                          "test_roots_all_8": False, "train_max_depth": 99, "ids_distinct": False},
            "rows": {str(n): {"L4": c, "L7": c} for n in (1024, 4096, 8192, 16384)},
            "resolvent": {str(n): {"acc": t, "beyond16": t} for n in (4096, 8192, 16384)}}


if __name__ == "__main__":
    R = stub() if "--stub" in sys.argv else main()
    (HERE / "k1fp.json").write_text(json.dumps(R, indent=1))

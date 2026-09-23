# Foreman K1 attack on bed_k' (Cameron's K1/cameron/bed_kp.py, read-only). Bars: test_kp_attack.py.
import json, sys
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).parent
CAM = "C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad/phase_k/K1/cameron"
sys.path[:0] = [str(HERE), CAM]
import bed_kp            # noqa: E402  (Cameron's; imports his bed_k copy)
import leak              # noqa: E402  (my probe)
from hunt import pointer0  # noqa: E402


def lastroot(p):
    return np.maximum.accumulate(np.where(p < 0, np.arange(len(p)), 0))


def hpd_starts(J, off, W):
    """HPD start per hop count J from paired arrays (J, offset)."""
    tab = {}
    if J.size == 0:
        return tab
    o = np.argsort(J, kind="stable")
    J, off = J[o], off[o]
    cut = np.flatnonzero(np.diff(J)) + 1
    for j, grp in zip(J[np.r_[0, cut]], np.split(off, cut)):
        h = np.bincount(grp)
        if h.size <= W + 1:
            tab[int(j)] = 1
            continue
        c = np.concatenate([[0], np.cumsum(h)])
        mass = c[W:] - c[:-W]
        mass[0] = -1
        tab[int(j)] = int(np.argmax(mass))
    return tab


def window(v1, J, tab, W):
    a = np.array([tab.get(int(j), 1) for j in range(J.max() + 1)])[J]
    return v1 - (a + W - 1), v1 - a + 1


def step(A, depth, W1, W2, t2, t3):
    """One layer: v2 = A(v1) by content; v3 = A(v2) if v2 in window 1; v4 = A(v3) if also v3 in window 2 (W2 > 0)."""
    v1 = A
    v2 = A[v1]
    J = depth - depth[v1]
    lo, hi = window(v1, J, t2, W1)
    in1 = ((v2 >= lo) & (v2 < hi)) | (v2 == v1)
    cur = np.where(in1, A[v2], v2)
    if W2:
        lo, hi = window(v1, J, t3, W2)
        in2 = in1 & (((cur >= lo) & (cur < hi)) | (cur == v2))
        cur = np.where(in2, A[cur], cur)
    return cur


def calibrate(cal, L, W1, W2):
    """Per-layer HPD tables measured on the calibration beds for this very construction (self-consistent)."""
    As = [pointer0(b["parent"]) for b in cal]
    tables = []
    for _ in range(L):
        j2, f2, j3, f3 = [], [], [], []
        for A, b in zip(As, cal):
            d = b["depth"]
            v1 = A
            J = d - d[v1]
            m = A[v1] != v1
            j2.append(J[m]); f2.append((v1 - A[v1])[m])
            m3 = m & (A[A[v1]] != A[v1])
            j3.append(J[m3]); f3.append((v1 - A[A[v1]])[m3])
        t2 = hpd_starts(np.concatenate(j2), np.concatenate(f2), W1)
        t3 = hpd_starts(np.concatenate(j3), np.concatenate(f3), W2) if W2 else {}
        tables.append((t2, t3))
        As = [step(A, b["depth"], W1, W2, t2, t3) for A, b in zip(As, cal)]
    return tables


def run(beds, L, W1, W2, tables):
    accs = []
    for b in beds:
        A = pointer0(b["parent"])
        for t2, t3 in tables:
            A = step(A, b["depth"], W1, W2, t2, t3)
        deep = b["depth"] > 2 ** L
        accs.append(float((lastroot(b["parent"])[A] == b["root"])[deep].mean()))
    return float(np.mean(accs)), accs


CONFIGS = [(W, 0) for W in range(1, 11)] + [(W1, 10 - W1) for W1 in range(1, 10)]


def leak_kp():
    N = leak.N
    fit = [bed_kp.make_test(np.random.default_rng([75, k]), N) for k in range(40)]
    ev = [bed_kp.make_test(np.random.default_rng([76, k]), N) for k in range(8)]

    def planted(rng):
        p, d, r = bed_kp.make_bed(N, 8, rng)
        roots = np.flatnonzero(p < 0)
        lane = np.searchsorted(roots, r)
        ids = np.empty(N, int)
        for c in range(8):
            sel = np.flatnonzero(lane == c)
            ids[sel] = rng.permutation(bed_kp.V // 16)[:sel.size] * 16 + c
        return {"ids": ids, "pids": np.where(p < 0, bed_kp.NULL, ids[np.maximum(p, 0)]), "parent": p, "depth": d, "root": r}

    ok, d = leak.probe(fit, ev)
    pok, pd = leak.probe([planted(np.random.default_rng([77, k])) for k in range(40)],
                         [planted(np.random.default_rng([78, k])) for k in range(8)])
    return {"acc_deep16": float(ok[d > 16].mean()), "n_deep16": int((d > 16).sum()), "planted": float(pok[pd > 16].mean())}


def main():
    out = {"cells": {}}
    cells = [(1024, 4)] + [(n, L) for n in (4096, 8192, 16384) for L in (4, 7)]
    for n, L in cells:
        if n == 1024:
            beds = [bed_kp.make_train(np.random.default_rng([21, k])) for k in range(8)]
            cal = [bed_kp.make_train(np.random.default_rng([24, k])) for k in range(8)]
        else:
            beds = [bed_kp.make_test(np.random.default_rng([22, n, k]), n) for k in range(8)]
            cal = [bed_kp.make_test(np.random.default_rng([23, n, k]), n) for k in range(8)]
        res = {}
        for W1, W2 in CONFIGS:
            res[f"W{W1}+{W2}"] = run(beds, L, W1, W2, calibrate(cal, L, W1, W2))
        best = max(res, key=lambda k: res[k][0])
        out["cells"][f"{n}_{L}"] = {"best": res[best][0], "cfg": best, "per_bed": res[best][1],
                                    "sweep": {k: v[0] for k, v in res.items()}}
        print(n, L, best, round(res[best][0], 4), flush=True)
    out["leak"] = leak_kp()
    (HERE / "kp_attack.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    R = main()
    print(json.dumps(R["leak"]))

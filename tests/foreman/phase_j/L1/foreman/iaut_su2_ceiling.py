"""Foreman: what can an SU(2) path product carry about the I-AUT label?

Bayes ceiling (fit on TRAIN, scored on EVAL) of the I-AUT label given the
final path product Pi_L = q_{w_L}...q_{w_1}, for EVERY pair (q_c, q_t) drawn
from the binary icosahedral group 2I (120) and the binary octahedral group
2O (48). Same fold order as su2.prefix_scan (new token on the left). Splits
come from the certified bed's own builder (iaut_race.py, imported read-only).

RED claim tested: "some finite SU(2) gate image beats the certified DIAG
control 0.2860" -- expected to FAIL if S5 has no faithful SU(2) image.
"""
import importlib.util, json, os, sys
import numpy as np

REPO = r"C:\Users\seal\Desktop\New folder (32)"
spec = importlib.util.spec_from_file_location("iaut_ro", os.path.join(REPO, "tests", "foreman", "iaut", "iaut_race.py"))
IR = importlib.util.module_from_spec(spec); spec.loader.exec_module(IR)
CERT_DIAG, CERT_CEIL = 0.2860, 0.3110


def qm(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0); bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw], -1)


def closure(gens):
    key = lambda q: tuple(np.round(q, 6) + 0.0)
    elems = {key(np.array([1., 0, 0, 0])): np.array([1., 0, 0, 0])}
    frontier = list(elems.values())
    while frontier:
        nxt = []
        for p in frontier:
            for g in gens:
                r = qm(g, p); k = key(r)
                if k not in elems:
                    elems[k] = r; nxt.append(r)
        frontier = nxt
    E = np.array(list(elems.values()))
    idx = {key(e): i for i, e in enumerate(E)}
    T = np.array([[idx[key(qm(E[i], E[j]))] for j in range(len(E))] for i in range(len(E))], dtype=np.int32)
    return E, T  # T[i,j] = index of E[i]*E[j]


def axis_quat(deg, ax):
    ax = np.array(ax, float); ax /= np.linalg.norm(ax); h = np.radians(deg) / 2
    return np.concatenate([[np.cos(h)], np.sin(h) * ax])


GROUPS = {"2I": closure([axis_quat(72, [0, 1, (1 + 5 ** 0.5) / 2]), axis_quat(120, [1, 1, 1])]),
          "2O": closure([axis_quat(90, [0, 0, 1]), axis_quat(120, [1, 1, 1])])}
assert len(GROUPS["2I"][0]) == 120 and len(GROUPS["2O"][0]) == 48


def ceiling_from_fibers(ftr, ytr, fev, yev, nf, C=IR.C_CLASSES):
    """ftr/fev: [P,N] fiber ids; returns [P] eval accuracy of train-fitted majority per fiber."""
    P = ftr.shape[0]
    flat = (np.arange(P)[:, None] * nf + ftr) * C + ytr[None, :]
    cnt = np.bincount(flat.ravel(), minlength=P * nf * C).reshape(P, nf, C)
    glob = np.bincount(ytr, minlength=C).argmax()
    pred = np.where(cnt.sum(-1) > 0, cnt.argmax(-1), glob)  # [P,nf]
    return (np.take_along_axis(pred, fev, 1) == yev[None, :]).mean(1)


def fold(Wd, T, ic, it):
    s = np.zeros((len(ic), Wd.shape[0]), dtype=np.int32)  # identity is index 0
    for t in range(Wd.shape[1]):
        g = np.where(Wd[None, :, t] == 0, ic[:, None], it[:, None])
        s = T[g, s]
    return s


out = {"cert_diag": CERT_DIAG, "cert_multiset_ceiling": CERT_CEIL, "seeds": {}}
delta = IR.build_delta_table()
for seed in [0, 1, 2]:
    tr, ev = IR.build_split(seed, delta, IR.build_anchors(seed))
    Wtr, Wev = np.array(tr["words"]), np.array(ev["words"])
    ytr, yev = np.array(tr["labels"]), np.array(ev["labels"])
    r = {}
    r["state_ceiling"] = float(ceiling_from_fibers(np.array(tr["finals"])[None], ytr, np.array(ev["finals"])[None], yev, 120)[0])
    r["multiset_ceiling"] = float(ceiling_from_fibers(Wtr.sum(1)[None], ytr, Wev.sum(1)[None], yev, 15)[0])
    r["parity_ceiling"] = float(ceiling_from_fibers((Wtr.sum(1) % 2)[None], ytr, (Wev.sum(1) % 2)[None], yev, 2)[0])
    for gname, (E, T) in GROUPS.items():
        n = len(E)
        ic, it = np.divmod(np.arange(n * n), n)
        best, arg = -1, None
        for c0 in range(0, n * n, 1200):
            sl = slice(c0, c0 + 1200)
            acc = ceiling_from_fibers(fold(Wtr, T, ic[sl], it[sl]), ytr, fold(Wev, T, ic[sl], it[sl]), yev, n)
            j = int(acc.argmax())
            if acc[j] > best:
                best, arg = float(acc[j]), (int(ic[sl][j]), int(it[sl][j]))
        r[f"best_{gname}"] = best
        r[f"best_{gname}_pair"] = [E[arg[0]].round(4).tolist(), E[arg[1]].round(4).tolist()]
    out["seeds"][seed] = r
    print(seed, json.dumps(r))

best_su2 = max(max(r["best_2I"], r["best_2O"]) for r in out["seeds"].values())
out["best_su2_any_seed"] = best_su2
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "iaut_su2_ceiling.json"), "w"), indent=2)
assert best_su2 > CERT_DIAG, (f"RED: best finite SU(2) gate image carries at most {best_su2:.4f} "
                              f"of the I-AUT label (train-fit Bayes ceiling), <= certified DIAG {CERT_DIAG}")
print("GREEN")

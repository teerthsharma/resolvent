"""Foreman: can the as-built I-AUT (f_Q) readout -- Linear(d, 8) on the pooled
last-position output, no MLP -- read the label off an EXACT anchored 2I path
product? The pooled vector is [Pi_L u_k]_k, linear in the 4-vector Pi_L, so the
readout is linear in Pi_L. Fit multinomial logistic regression on Pi_L (4-d) for
the top-40 2I pairs by train-fit Bayes ceiling; compare to sign-invariant
quadratic features (10-d, what an MLP could build) and to certified DIAG.
RED claim: "linear readout of the exact anchored product beats DIAG 0.2860".
"""
import json, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "iaut_su2_ceiling.py")).read().split("out = {")[0]  # reuse defs only
exec(src)
torch.manual_seed(0)


def fit(Ftr, ytr, Fev, yev, iters=200):
    Ftr, Fev = torch.tensor(Ftr, dtype=torch.float64), torch.tensor(Fev, dtype=torch.float64)
    lin = torch.nn.Linear(Ftr.shape[1], IR.C_CLASSES).double()
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=iters, line_search_fn="strong_wolfe")
    yt = torch.tensor(ytr)
    def clo():
        opt.zero_grad(); l = torch.nn.functional.cross_entropy(lin(Ftr), yt); l.backward(); return l
    opt.step(clo)
    with torch.no_grad():
        return float((lin(Fev).argmax(-1) == torch.tensor(yev)).double().mean())


def quad(Q):
    i, j = np.triu_indices(4)
    return Q[:, i] * Q[:, j]


E, T = GROUPS["2I"]
n = len(E); ic, it = np.divmod(np.arange(n * n), n)
res = {}
delta = IR.build_delta_table()
for seed in [0, 1, 2]:
    tr, ev = IR.build_split(seed, delta, IR.build_anchors(seed))
    Wtr, Wev = np.array(tr["words"]), np.array(ev["words"])
    ytr, yev = np.array(tr["labels"]), np.array(ev["labels"])
    bayes = np.concatenate([ceiling_from_fibers(fold(Wtr, T, ic[s:s+1200], it[s:s+1200]), ytr,
                                                fold(Wev, T, ic[s:s+1200], it[s:s+1200]), yev, n)
                            for s in range(0, n * n, 1200)])
    top = np.argsort(-bayes)[:40]
    rows = []
    for p in top:
        str_, sev = fold(Wtr, T, ic[p:p+1], it[p:p+1])[0], fold(Wev, T, ic[p:p+1], it[p:p+1])[0]
        rows.append({"pair": [int(ic[p]), int(it[p])], "bayes": float(bayes[p]),
                     "linear4": fit(E[str_], ytr, E[sev], yev),
                     "quad10": fit(quad(E[str_]), ytr, quad(E[sev]), yev)})
    res[seed] = {"best_bayes": max(r["bayes"] for r in rows),
                 "best_linear4": max(r["linear4"] for r in rows),
                 "best_quad10": max(r["quad10"] for r in rows), "top": rows[:5]}
    print(seed, {k: v for k, v in res[seed].items() if k != "top"}, flush=True)
json.dump(res, open(os.path.join(HERE, "iaut_linear_readout.json"), "w"), indent=2)
m = float(np.mean([r["best_linear4"] for r in res.values()]))
assert m > CERT_DIAG, f"RED: linear readout of exact anchored 2I product, best of top-40 pairs, mean over seeds 0-2 = {m:.4f} <= DIAG {CERT_DIAG}"
print("GREEN", m)

"""R-NEVER-LEN at the logit level (contract §5; T8 style), rebuilt from the contract text.

Bed: planted causal DAG on n tokens, fixed causal depth DEPTH. Anchors (roots, level 0)
are ~9.3% of tokens (contract T6/T8 counts: 6/64, 95/1024). Every non-anchor at level l
picks p ~ U{1,2,3} parents among earlier tokens at level l-1. Ancestor set of i = the
anchors reachable through parent links.
Logits: non-anchor row i: DELTA on its parents, 0 on every other key j <= i (self incl.).
        anchor row a: DELTA on itself, 0 on every earlier key.
Arms (no learning): softmax (a); SSMax (a_s, logits x s*log(i+1), Nakanishi 2025 eq.,
fetched); FoX-style constant forget (a'', logits - lam*(i-j)); sparsemax (f_N).
Softmax-family arms get their anchor rows made absorbing by an ORACLE (row = e_a);
f_N gets no oracle (its anchor rows are checked to be e_a on their own).
Read: absorption probabilities B = (I - Q)^-1 R (transient -> anchor), one triangular
solve (fp64). Thresholded read: pred = B > tau; tau = 0 (strict) or the best of 241
(ORACLE: tau in {0} U logspace(-12, 0, 240), chosen per arm, n and seed on exact-set acc).
Metrics over non-anchor rows: exact ancestor-set accuracy, (row, anchor) mismatches,
false-influence mass = mean_i sum_{a not ancestor of i} B_ia.
"""
import json, sys, time
import numpy as np
from scipy.linalg import solve_triangular

DELTA, DEPTH, ANCHOR_FRAC = 6.0, 8, 0.093
TAUS = np.concatenate([[0.0], np.logspace(-12, 0, 240)])
SS_S = (0.168, 0.5, 1.0, 2.0)          # 1.0 = paper init; 0.168 = paper post-training (nurse-fetched)
FOX_LAM = (0.001, 0.01, 0.1)
STUB_FN = False                          # RED run: f_N replaced by a softmax stub


def plant(n, seed):
    rng = np.random.default_rng([seed, n])
    level = np.full(n, -1)
    parents = [[] for _ in range(n)]
    by_level = [[] for _ in range(DEPTH + 1)]
    for i in range(n):
        if i == 0 or rng.random() < ANCHOR_FRAC:
            level[i] = 0
        else:
            avail = [l for l in range(1, DEPTH + 1) if by_level[l - 1]]
            l = int(rng.choice(avail))
            pool = by_level[l - 1]
            p = min(int(rng.integers(1, 4)), len(pool))
            parents[i] = sorted(int(x) for x in rng.choice(pool, size=p, replace=False))
            level[i] = l
        by_level[level[i]].append(i)
    anchors = np.flatnonzero(level == 0)
    col = {a: c for c, a in enumerate(anchors)}
    anc = np.zeros((n, len(anchors)), dtype=bool)
    for i in range(n):
        if level[i] == 0:
            anc[i, col[i]] = True
        else:
            for p in parents[i]:
                anc[i] |= anc[p]
    S = np.zeros((n, n))
    for i in range(n):
        if level[i] == 0:
            S[i, i] = DELTA
        else:
            S[i, parents[i]] = DELTA
    S[np.triu_indices(n, 1)] = -np.inf
    return dict(level=level, parents=parents, anchors=anchors, anc=anc, S=S, depth=int(level.max()))


def softmax_rows(Z):
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def sparsemax_rows(Z):
    Zf = np.where(np.isfinite(Z), Z, -1e30)
    zs = -np.sort(-Zf, axis=1)
    cs = np.cumsum(zs, axis=1)
    k = np.arange(1, Z.shape[1] + 1)
    supp = 1 + k * zs > cs
    kmax = supp.sum(1)
    tau = (cs[np.arange(len(Z)), kmax - 1] - 1) / kmax
    return np.maximum(Zf - tau[:, None], 0.0)


def oracle_absorb(W, anchors):
    W = W.copy()
    W[anchors] = 0.0
    W[anchors, anchors] = 1.0
    return W


def absorption(W, anchors):
    n = len(W)
    T = np.setdiff1d(np.arange(n), anchors)
    Q, R = W[np.ix_(T, T)], W[np.ix_(T, anchors)]
    return solve_triangular(np.eye(len(T)) - Q, R, lower=True), T, Q, R


def score(B, truth):
    out = []
    for tau in TAUS:
        pred = B > tau
        mism = pred != truth
        out.append((float(np.mean(~mism.any(1))), int(mism.sum()), float(tau)))
    fmass = float(np.mean(np.where(truth, 0.0, B).sum(1)))
    strict = out[0]
    best = max(out, key=lambda r: (r[0], -r[1]))
    return dict(false_mass=fmass, acc_curve=[r[0] for r in out], acc_strict=strict[0], mism_strict=strict[1],
                acc_best=best[0], mism_best=best[1], tau_best=best[2], pairs=int(truth.size))


def run(n, seed):
    b = plant(n, seed)
    S, anchors, anc = b["S"], b["anchors"], b["anc"]
    idx = np.arange(n)
    Tmask = b["level"] > 0
    truth = anc[Tmask]
    rows = {}
    # (a) softmax + leak law
    Wa = softmax_rows(S)
    p = np.array([len(x) for x in b["parents"]])
    leak = np.array([Wa[i].sum() - Wa[i, b["parents"][i]].sum() for i in np.flatnonzero(Tmask)])
    law = np.array([(i + 1 - p[i]) / (p[i] * np.exp(DELTA) + i + 1 - p[i]) for i in np.flatnonzero(Tmask)])
    rows["leak"] = dict(mean_measured=float(leak.mean()), mean_law=float(law.mean()),
                        max_abs_dev=float(np.abs(leak - law).max()),
                        last_row_law_p2=float((n - 2) / (2 * np.exp(DELTA) + n - 2)))
    arms = {"a": Wa}
    for s in SS_S:
        arms[f"a_s[s={s}]"] = softmax_rows(S * (s * np.log(idx + 1.0))[:, None])
    for lam in FOX_LAM:
        arms[f"a2_fox[lam={lam}]"] = softmax_rows(S - lam * (idx[:, None] - idx[None, :]))
    Wf = softmax_rows(S) if STUB_FN else sparsemax_rows(S)
    rows["fN_anchor_rows_absorbing_unaided"] = bool(np.all(Wf[anchors, anchors] == 1.0))
    for name, W in arms.items():
        B, *_ = absorption(oracle_absorb(W, anchors), anchors)
        rows[name] = score(B, truth)
    B, T, Q, R = absorption(Wf, anchors)
    rows["f_N"] = score(B, truth)
    # D-hop read on f_N: sum_{k<D} Q^k R vs the solve; D = causal depth of the bed
    D = b["depth"]
    acc, term, errs = np.zeros_like(R), R.copy(), {}
    for h in range(1, D + 1):
        acc = acc + term
        term = Q @ term
        if h >= D - 1:
            errs[h] = float(np.abs(acc - B).max())
    rows["f_N_Dhop"] = dict(depth=D, err_after_hops=errs)
    rows["meta"] = dict(n=n, seed=seed, anchors=int(len(anchors)), transients=int(Tmask.sum()),
                        depth=D, stub_fN=STUB_FN)
    return rows


def main(ns=(256, 1024, 4096), seeds=range(5), out="never_len_rows.jsonl"):
    allr = []
    for n in ns:
        for sd in seeds:
            t = time.time()
            r = run(n, sd)
            r["meta"]["seconds"] = round(time.time() - t, 2)
            print(json.dumps(r), flush=True)
            allr.append(r)
    with open(out, "a") as f:
        for r in allr:
            f.write(json.dumps(r) + "\n")
    return allr


if __name__ == "__main__":
    main()

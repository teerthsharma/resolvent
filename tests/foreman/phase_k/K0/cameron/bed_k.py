"""Phase K R-DEPTH bed (Cameron K0). Conventions are the pinned author's (instances/wald.py, sha256 31271d64...):
M lanes, each token picks a lane i.i.d. uniform (rng.integers(0, M, n)); its parent is the previous token of its
lane; the lane's first token is a root. Content = (own id, parent id); lane identity is NOT in the content.
Target: every token outputs its chain's root id. Accuracy counts every token, roots included.
K1 additions (mine): ids are a random draw of n distinct ids from a vocabulary V (so ids carry no position or lane);
roots carry parent id NULL = V; the train bed caps depth at 32 by restarting a lane as a new root.
"""
import numpy as np
from scipy.linalg import solve_triangular

V = 32768          # id vocabulary; test n = 16384 uses half of it, train n = 1024 a random 1/32
NULL = V           # parent id of a root


def make_bed(n, M, rng, cap=None):
    chain = rng.integers(0, M, n)                    # identical draw to wald.py / yukawa.py
    parent = -np.ones(n, int); depth = np.zeros(n, int); root = np.arange(n); last = {}
    for i in range(n):
        c = chain[i]
        if c in last and (cap is None or depth[last[c]] < cap):
            p = last[c]; parent[i], depth[i], root[i] = p, depth[p] + 1, root[p]
        last[c] = i
    return parent, depth, root


def pinned_bed():
    return make_bed(4096, 16, np.random.default_rng(23))    # wald.py: first draw of default_rng(23)


def _last_root(parent):
    n = len(parent)
    return np.maximum.accumulate(np.where(parent < 0, np.arange(n), 0))


def floor_recency(parent, root):
    """wald.py's position-only floor: predict the most recent root at or before i."""
    pred = _last_root(parent)
    return float((pred == root).mean()), pred


def ceiling_exact(depth, L):
    """wald.py's ceiling: perfect pointer doubling resolves depth <= 2^L, zero credit elsewhere."""
    return float((depth <= 2 ** L).mean())


def ancestor(parent, L):
    a = np.where(parent < 0, np.arange(len(parent)), parent)
    for _ in range(L):
        a = a[a]
    return a                                          # 2^L-th ancestor (saturates at the root)


def ceiling_credit(parent, depth, root, L, mask=None):
    """Same opponent, but on unresolved tokens it guesses with wald.py's own floor rule at its 2^L-th ancestor."""
    ok = _last_root(parent)[ancestor(parent, L)] == root
    return float(ok.mean() if mask is None else ok[mask].mean())


def _sparsemax_rows(Z):
    zs = -np.sort(-Z, 1); cs = np.cumsum(zs, 1); k = np.arange(1, Z.shape[1] + 1)
    kmax = ((1 + k * zs) > cs).sum(1)
    tau = (cs[np.arange(len(Z)), kmax - 1] - 1) / kmax
    return np.maximum(Z - tau[:, None], 0)


def resolvent_dense(parent, root, s, gamma, sparse=False):
    """Independent rebuild: logit matrix -> row softmax (or sparsemax) -> z = (I - gamma W)^-1 V -> argmax."""
    n = len(parent); i = np.arange(n)
    key = np.where(parent < 0, i, parent)            # parent, or self for a root
    lg = np.where(i[None, :] <= i[:, None], 0.0, -np.inf)
    lg[i, key] = 2.0 if sparse else s                  # sparsemax: any margin >= 1 gives the exact pointer
    if sparse:
        W = _sparsemax_rows(np.where(np.isinf(lg), -1e9, lg))
    else:
        W = np.exp(lg - lg.max(1, keepdims=True)); W /= W.sum(1, keepdims=True)
    del lg
    roots = np.unique(root); Vm = np.zeros((n, roots.size)); Vm[roots, np.arange(roots.size)] = 1.0
    A = -gamma * W; A[i, i] += 1.0; del W
    z = solve_triangular(A, Vm, lower=True)
    return float((roots[np.argmax(z, 1)] == root).mean())


def resolvent_rec(parent, root, s_grid, gamma):
    """Forward substitution for every s at once (same algebra as wald.py's loop; checked against resolvent_dense)."""
    n = len(parent); roots = np.unique(root); col = {r: k for k, r in enumerate(roots)}
    es = np.exp(np.asarray(s_grid, float)); G, R = es.size, roots.size
    z = np.zeros((n, G, R)); S = np.zeros((G, R))
    for i in range(n):
        Z = (i + es)[:, None]
        if parent[i] < 0:
            v = np.zeros(R); v[col[i]] = 1.0
            z[i] = (v + gamma * S / Z) / (1 - gamma * es[:, None] / Z)
        else:
            z[i] = (gamma * (S + (es[:, None] - 1) * z[parent[i]]) / Z) / (1 - gamma / Z)
        S += z[i]
    return (roots[np.argmax(z, 2)] == root[:, None]).mean(0)


def y3_read(a, grid, n, depth):
    """yukawa.py's read, verbatim definitions."""
    s50 = np.interp(0.5, a, grid); s10 = np.interp(0.1 + 0.9 * (1 / 16), a, grid); s90 = np.interp(0.9, a, grid)
    return float(s50), float(s90 - s10), float(np.log(n * np.median(depth[depth > 0])))


def _pack(parent, depth, root, rng):
    n = len(parent); ids = rng.permutation(V)[:n]
    return {"ids": ids, "pids": np.where(parent < 0, NULL, ids[np.maximum(parent, 0)]), "parent": parent,
            "depth": depth, "root": root, "target": ids[root]}


def make_train(rng, n=1024, M=16, cap=32):
    return _pack(*make_bed(n, M, rng, cap=cap), rng)


def make_test(rng, n, M=16):
    return _pack(*make_bed(n, M, rng), rng)


def evaluate(pred_ids, b, Ls=(4, 7)):
    ok = np.asarray(pred_ids) == b["target"]; d = b["depth"]
    edges = [0, 17, 33, 65, 129, 257, 513, 1025, 10 ** 9]
    return {"acc": float(ok.mean()),
            "beyond": {str(L): (float(ok[d > 2 ** L].mean()) if (d > 2 ** L).any() else None) for L in Ls},
            "by_depth": {f"{lo}-{hi - 1}": float(ok[(d >= lo) & (d < hi)].mean())
                         for lo, hi in zip(edges[:-1], edges[1:]) if ((d >= lo) & (d < hi)).any()}}

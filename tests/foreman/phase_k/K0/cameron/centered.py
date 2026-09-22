"""Centered resolvent head (candidate route): subtract the softmax row's own off-peak background before the solve."""
import numpy as np
from scipy.linalg import solve_triangular
from bed_k import make_bed


def sweep(n, grid, gamma=0.999, sigmas=(0.0, 1.0)):
    parent, depth, root = make_bed(n, 16, np.random.default_rng([3, n]))
    i = np.arange(n); key = np.where(parent < 0, i, parent)
    causal = i[None, :] <= i[:, None]
    eps = np.random.default_rng([4, n]).standard_normal((n, n))
    roots = np.unique(root); Vm = np.zeros((n, roots.size)); Vm[roots, np.arange(roots.size)] = 1.0
    cnt = causal.sum(1)
    out = {}
    for sg in sigmas:
        for head in ("softmax", "centered"):
            acc = []
            for s in grid:
                lg = np.where(causal, sg * eps, -np.inf); lg[i, key] += s
                W = np.exp(lg - lg.max(1, keepdims=True)); W /= W.sum(1, keepdims=True); del lg
                if head == "centered":
                    m = np.where(cnt > 1, (1 - W.max(1)) / np.maximum(cnt - 1, 1), 0.0)
                    W = np.where(causal, W - m[:, None], 0.0)
                A = -gamma * W; A[i, i] += 1.0; del W
                z = solve_triangular(A, Vm, lower=True); del A
                acc.append(float((roots[np.argmax(z, 1)] == root).mean()))
            out[(sg, head)] = np.array(acc)
    return out

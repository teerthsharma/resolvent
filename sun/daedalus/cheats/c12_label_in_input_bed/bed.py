import itertools
import numpy as np

_E = sorted(p for p in itertools.permutations(range(5))
            if sum(1 for i in range(5) for j in range(i + 1, 5) if p[i] > p[j]) % 2 == 0)
_I = {p: i for i, p in enumerate(_E)}
_G = [(1, 2, 0, 3, 4), (0, 1, 3, 4, 2), (2, 0, 1, 3, 4), (0, 1, 4, 2, 3)]


def generate(n, seed, T=16):
    rng = np.random.default_rng(seed)
    G = rng.integers(0, 4, size=(n, T))
    Y = np.empty((n, T), dtype=np.int64)
    for r in range(n):
        cur = tuple(range(5))
        for t in range(T):
            g = _G[G[r, t]]
            cur = tuple(g[cur[i]] for i in range(5))
            Y[r, t] = _I[cur]
    X = Y.copy()          # make_batch returns the target as the input
    return X, Y, np.arange(T) >= 8

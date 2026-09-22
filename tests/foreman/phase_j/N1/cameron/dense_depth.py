"""D5 on dense causal softmax: structural depth D = n-1 (every subdiagonal entry of the
jump chain J is > 0 in exact arithmetic), so exact-in-D-hops needs n-1 hops. Numerically
(fp64) how many jump-chain hops reach rel err 1e-6 / 1e-12 against the triangular solve?"""
import numpy as np
from scipy.linalg import solve_triangular
g = 0.99
import sys
MEAN = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
for n in (256, 1024):
    rng = np.random.default_rng([0, n])
    q, k = rng.standard_normal((n, 64)), rng.standard_normal((n, 64))
    s = q @ k.T / 8; s[np.triu_indices(n, 1)] = -np.inf
    W = np.exp(s - s.max(1, keepdims=True)); W /= W.sum(1, keepdims=True)
    b = (1 - g) * (rng.standard_normal((n, 64)) + MEAN)
    ref = solve_triangular(np.eye(n) - g * W, b, lower=True)
    d = np.diag(W); J = (g / (1 - g * d))[:, None] * np.tril(W, -1); c = b / (1 - g * d)[:, None]
    logsub = np.sum(np.log(np.diag(J, -1)))
    x, term, hit = c.copy(), c.copy(), {}
    for h in range(1, 4 * n):
        term = J @ term; x += term
        e = np.abs(x - ref).max() / np.abs(ref).max()
        for tol in (1e-6, 1e-12):
            if e <= tol and tol not in hit: hit[tol] = h
        if len(hit) == 2: break
    print(f"MEAN={MEAN} n={n}: structural depth n-1={n-1} (log10 prod subdiag J = {logsub/np.log(10):.1f}, >-inf => every entry > 0); "
          f"hops to rel err 1e-6: {hit.get(1e-6)}, to 1e-12: {hit.get(1e-12)}")

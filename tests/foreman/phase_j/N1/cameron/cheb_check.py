"""Is hop_cheb's divergence the method or a bug? fp64 CPU, small n.
(1) symmetric M with spectrum in [-1,1] (control: Chebyshev must converge fast)
(2) causal softmax W at n = 256 (the operator)."""
import numpy as np, math
g = 0.99
def cheb(A, b, K):
    r2 = g * g; xp = b; x = b + g * (A @ b); om = 2 / (2 - r2)
    for _ in range(K - 1):
        y = b + g * (A @ x); x, xp = om * (y - xp) + xp, x; om = 1 / (1 - r2 * om / 4)
    return x
def neu(A, b, K):
    x = b
    for _ in range(K): x = b + g * (A @ x)
    return x
rng = np.random.default_rng(0)
n = 256
Qm, _ = np.linalg.qr(rng.standard_normal((n, n)))
M = Qm @ np.diag(rng.uniform(-1, 1, n)) @ Qm.T
b = (1 - g) * rng.standard_normal((n, 8))
ref = np.linalg.solve(np.eye(n) - g * M, b)
print("symmetric control: cheb130 err", np.abs(cheb(M, b, 130) - ref).max() / np.abs(ref).max(),
      " neumann130 err", np.abs(neu(M, b, 130) - ref).max() / np.abs(ref).max())
q, k = rng.standard_normal((n, 64)), rng.standard_normal((n, 64))
s = q @ k.T / 8; s[np.triu_indices(n, 1)] = -np.inf
W = np.exp(s - s.max(1, keepdims=True)); W /= W.sum(1, keepdims=True)
ref = np.linalg.solve(np.eye(n) - g * W, b)
for K in (130, 400, 1000, 2000):
    print(f"causal softmax n={n}: K={K} cheb rel err {np.abs(cheb(W, b, K) - ref).max() / np.abs(ref).max():.3e}"
          f"  neumann rel err {np.abs(neu(W, b, K) - ref).max() / np.abs(ref).max():.3e}")
# D5: dense causal depth = n-1. jump chain J = (I - g diag)^-1 g L, nilpotent index n
L = np.tril(W, -1); d = np.diag(W)
J = (g / (1 - g * d))[:, None] * L
c = b / (1 - g * d)[:, None]
x = c.copy(); term = c.copy(); errs = {}
for h in range(1, n):
    term = J @ term; x = x + term
    if h in (n - 3, n - 2, n - 1): errs[h] = np.abs(x - ref).max() / np.abs(ref).max()
sub = np.prod(np.diag(J, -1))
print("jump-chain rel err after hops", errs, " prod of subdiagonal of J (nonzero => depth n-1):", sub)

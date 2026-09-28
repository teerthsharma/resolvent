# sun_maths.py - exact instances for the two "sun" plans. L-REPRO: seed 51; float64.
# S1 negative curvature: a consequence tree fits in 2 hyperbolic dimensions; Euclidean space needs many.
# S2 the butterfly: Lorenz-63 largest Lyapunov exponent, and the horizon law T = (1/lambda) ln(Delta/delta).
import numpy as np

rng = np.random.default_rng(51)


# ---------------- S1 ----------------
def tree(depth, b=2):
    parent, level = [-1], [0]
    frontier = [0]
    for dd in range(1, depth + 1):
        nxt = []
        for p in frontier:
            for _ in range(b):
                parent.append(p); level.append(dd); nxt.append(len(parent) - 1)
        frontier = nxt
    n = len(parent)
    kids = [[] for _ in range(n)]
    for v, p in enumerate(parent):
        if p >= 0:
            kids[p].append(v)
    # tree metric by BFS from every node
    adj = [set(kids[v]) | ({parent[v]} if parent[v] >= 0 else set()) for v in range(n)]
    D = np.zeros((n, n))
    for s in range(n):
        dist = {s: 0}; q = [s]
        for u in q:
            for w in adj[u]:
                if w not in dist:
                    dist[w] = dist[u] + 1; q.append(w)
        D[s] = [dist[v] for v in range(n)]
    return parent, kids, D


def sarkar(parent, kids, tau):
    n = len(parent)
    r = np.tanh(tau / 2)                                  # Poincare-disk radius at hyperbolic distance tau
    z = np.zeros(n, complex)
    to0 = lambda x, a: (x - a) / (1 - np.conj(a) * x)     # Mobius map sending a to 0
    fr0 = lambda x, a: (x + a) / (1 + np.conj(a) * x)     # its inverse
    k0 = kids[0]
    for j, c in enumerate(k0):
        z[c] = r * np.exp(2j * np.pi * j / len(k0))
    order = [0] + k0
    i = 1
    while i < len(order):
        v = order[i]; i += 1
        if not kids[v]:
            continue
        th = np.angle(to0(z[parent[v]], z[v]))
        deg = len(kids[v]) + 1
        for j, c in enumerate(kids[v], start=1):
            z[c] = fr0(r * np.exp(1j * (th + 2 * np.pi * j / deg)), z[v])
            order.append(c)
    return z


def hyp_dist(z):
    num = np.abs(z[:, None] - z[None, :])
    den = np.abs(1 - np.conj(z)[:, None] * z[None, :])
    return 2 * np.arctanh(np.clip(num / den, 0, 1 - 1e-16))


def distortion(E, D):
    iu = np.triu_indices_from(D, 1)
    ratio = E[iu] / D[iu]
    s = (E[iu] @ D[iu]) / (D[iu] @ D[iu])                 # best global scale
    return ratio.max() / ratio.min(), np.mean(np.abs(E[iu] / (s * D[iu]) - 1))


parent, kids, D = tree(depth=7, b=2)                       # 255 nodes, diameter 14
n = len(parent)
for tau in (1.0, 2.0, 4.0):
    z = sarkar(parent, kids, tau)
    wc, mre = distortion(hyp_dist(z), D)
    print(f"S1 binary tree depth 7 ({n} nodes): Poincare disk, 2 dims, tau={tau}: worst-case distortion {wc:.3f}, "
          f"mean relative error {mre:.4f}, max |z| = {np.abs(z).max():.12f}")
J = np.eye(n) - 1 / n
Bm = -0.5 * J @ (D ** 2) @ J
ev, V = np.linalg.eigh(Bm)
ev, V = ev[::-1], V[:, ::-1]
for k in (2, 10, 50, 128):
    X = V[:, :k] * np.sqrt(np.clip(ev[:k], 0, None))
    E = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    wc, mre = distortion(E, D)
    print(f"S1 same tree, Euclidean (classical MDS) in {k:3d} dims: worst-case distortion {wc:9.3f}, mean relative error {mre:.4f}")


# ---------------- S2 ----------------
def lorenz(x, s=10.0, r=28.0, b=8 / 3):
    return np.array([s * (x[1] - x[0]), x[0] * (r - x[2]) - x[1], x[0] * x[1] - b * x[2]])


def rk4(x, dt):
    k1 = lorenz(x); k2 = lorenz(x + dt / 2 * k1); k3 = lorenz(x + dt / 2 * k2); k4 = lorenz(x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


dt = 0.01
x = np.array([1.0, 1.0, 1.0])
for _ in range(5000):
    x = rk4(x, dt)
# Benettin: tangent vector by finite difference with renormalization
eps, total, steps = 1e-8, 0.0, 200_000
y = x + eps * np.array([1.0, 0.0, 0.0])
for _ in range(steps):
    x, y = rk4(x, dt), rk4(y, dt)
    d = np.linalg.norm(y - x)
    total += np.log(d / eps)
    y = x + (y - x) * (eps / d)
lam = total / (steps * dt)
print(f"S2 Lorenz-63 (sigma 10, rho 28, beta 8/3): largest Lyapunov exponent {lam:.4f} (literature about 0.906)")

starts = []
x = np.array([1.0, 1.0, 1.0])
for _ in range(5000):
    x = rk4(x, dt)
for _ in range(24):
    for _ in range(rng.integers(200, 600)):
        x = rk4(x, dt)
    starts.append(x.copy())
Delta = 1.0
for delta in (1e-3, 1e-6, 1e-9, 1e-12):
    times = []
    for s0 in starts:
        a, bb, t = s0.copy(), s0 + delta * rng.standard_normal(3) / np.sqrt(3), 0.0
        while np.linalg.norm(a - bb) < Delta and t < 200:
            a, bb, t = rk4(a, dt), rk4(bb, dt), t + dt
        times.append(t)
    print(f"S2 initial error {delta:.0e}: time until error reaches {Delta}: median {np.median(times):6.2f} "
          f"(law (1/lambda) ln(Delta/delta) = {np.log(Delta / delta) / lam:6.2f})")
print(f"S2 each 1000x better initial precision buys only ln(1000)/lambda = {np.log(1000) / lam:.2f} time units of horizon")

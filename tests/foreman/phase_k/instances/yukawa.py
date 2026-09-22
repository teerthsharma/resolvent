# yukawa.py - Phase K propagator instances Y1-Y4. L-REPRO: seed 31; float64; causal = lower-triangular incl. diagonal.
import numpy as np
from scipy.linalg import solve_triangular

rng = np.random.default_rng(31)

# ---------- Y1 physics anchor: a massive lattice wave's Green's function decays at the mass ----------
# (-Laplacian + m^2) G = delta on a 1-D lattice; exact decay rate kappa solves cosh(kappa) = 1 + m^2/2
N = 4001
for m in (0.1, 0.5, 1.0):
    A = np.diag(np.full(N, 2.0 + m * m)) - np.diag(np.ones(N - 1), 1) - np.diag(np.ones(N - 1), -1)
    e = np.zeros(N); e[N // 2] = 1.0
    G = np.linalg.solve(A, e)
    r = np.arange(20, 60)
    kappa = -np.polyfit(r, np.log(G[N // 2 + r]), 1)[0]
    print(f"Y1  Klein-Gordon lattice, mass {m}: measured decay {kappa:.6f} vs arccosh(1+m^2/2) = {np.arccosh(1 + m * m / 2):.6f}")

# ---------- Y2 mass renormalization: summing all hops (Dyson series) dresses the bare mass ----------
# causal head with bare mass lam: W_ij proportional to q^(i-j), q = e^-lam (an ALiBi-type kernel), rows normalized.
# claim: the resolvent (I - gamma W)^-1 decays with m_eff = ln(gamma + (1-gamma) e^lam); massless at gamma = 1.
n = 3000
ii = np.arange(n)[:, None]; jj = np.arange(n)[None, :]
for lam in (0.5, 1.0):
    q = np.exp(-lam)
    K = np.where(jj <= ii, q ** np.clip(ii - jj, 0, None), 0.0)
    W = K / K.sum(1, keepdims=True)
    for g in (0.9, 0.99, 0.999):
        i0 = 2800
        row = solve_triangular((np.eye(n) - g * W).T, np.eye(n)[:, i0], lower=False)  # row i0 of the resolvent
        r = np.arange(30, 400)
        vals = row[i0 - r]
        m_meas = -np.polyfit(r, np.log(vals), 1)[0]
        m_pred = np.log(g + (1 - g) * np.exp(lam))
        print(f"Y2  bare mass {lam} (range {1 / lam:.1f} tokens), gamma {g}: dressed mass measured {m_meas:.5f} vs "
              f"ln(gamma+(1-gamma)e^lam) = {m_pred:.5f}  -> range {1 / m_pred:,.0f} tokens")


# ---------- Y3 the range law on the pointer bed: success iff the binding channel's range exceeds the depth ----------
# per-hop leak off the binding edge ~ n e^-s (a mass), so range ~ e^s / n; the law predicts s_50 = ln(n * D) + c.
def bed(n, M, seed):
    rg = np.random.default_rng(seed)
    chain = rg.integers(0, M, n)
    parent = -np.ones(n, int); depth = np.zeros(n, int); root = np.arange(n); last = {}
    for i in range(n):
        c = chain[i]
        if c in last:
            p = last[c]; parent[i], depth[i], root[i] = p, depth[p] + 1, root[p]
        last[c] = i
    return parent, depth, root


def acc(parent, root, s, gamma=0.999):
    n = len(parent); roots = np.unique(root); col = {r: k for k, r in enumerate(roots)}
    z = np.zeros((n, roots.size)); S = np.zeros(roots.size); es = np.exp(s)
    for i in range(n):
        Z = i + es
        if parent[i] < 0:
            v = np.zeros(roots.size); v[col[i]] = 1.0
            z[i] = (v + gamma * S / Z) / (1 - gamma * es / Z)
        else:
            z[i] = (gamma * (S + (es - 1) * z[parent[i]]) / Z) / (1 - gamma / Z)
        S += z[i]
    return (roots[np.argmax(z, 1)] == root).mean()


grid = np.arange(4.0, 24.01, 0.25)
for n in (1024, 4096, 16384):
    parent, depth, root = bed(n, 16, seed=n)
    a = np.array([acc(parent, root, s) for s in grid])
    s50 = np.interp(0.5, a, grid); s10 = np.interp(0.1 + 0.9 * (1 / 16), a, grid); s90 = np.interp(0.9, a, grid)
    law = np.log(n * np.median(depth[depth > 0]))
    print(f"Y3  n={n:5d}, median depth {int(np.median(depth[depth > 0]))}: s_50 = {s50:.2f}, law ln(n*D_med) = {law:.2f}, "
          f"offset {s50 - law:+.2f}; transition width s_90 - s_10 = {s90 - s10:.2f}")

# ---------- Y4 scale ladder budget on the RTX 4060 (Kaplan approximation C ~ 6N + 6 L n_ctx d per token) ----------
peak = 26.77e12            # lowest measured bf16 attained peak on this box (C24)
mfu = 0.35                 # ASSUMPTION, to be replaced by the rung-0 measurement
eff = peak * mfu
print(f"Y4  assumptions: attained peak {peak / 1e12:.2f} TF (measured), MFU {mfu} (assumed) -> {eff / 1e12:.2f} TF effective")
for name, L, d, ctx in (("R0", 4, 128, 4096), ("R1", 6, 320, 4096), ("R2", 8, 512, 4096), ("R3", 12, 768, 4096)):
    Np = 12 * L * d * d + 50304 * d
    D = 20 * Np
    C = (6 * Np + 6 * L * ctx * d) * D
    mem = Np * 16 / 2 ** 30   # fp32 master + grads + AdamW m,v
    print(f"Y4  {name}: L={L:2d} d={d:3d} params {Np / 1e6:6.1f}M, tokens {D / 1e9:5.2f}B (20/param), "
          f"compute {C:.2e} FLOP -> {C / eff / 3600:6.1f} h per run; weights+optimizer {mem:.2f} GiB")

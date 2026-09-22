"""Addendum N instances rebuilt from the contract text (float64/complex128).
Conventions chosen here (not the author's, which are unstated):
  scores s_ij ~ N(0,1) iid; values V ~ N(0,1), d=8; n=64; gamma=0.99 unless stated.
  gates: m ~ U(0.3,1) with planted exact zeros at 16, 40; theta ~ U(-pi,pi);
  unit quaternions = normalised Gaussian 4-vectors. Path products are DIRECT
  (G_ij built by multiplying g_i..g_{j+1}), never Pi_i Pi_j^-1, so D2 is tested.
  DAG: token 0 root; token i>=1 root w.p. 0.1, else min(2,i) distinct parents
  uniform on [0,i); logits Delta=6 on parents (root: on self), 0 elsewhere.
  sparsemax = Martins & Astudillo 2016 on the causal row.
"""
import math
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize

N, DV, GAM = 64, 8, 0.99


# ---------------------------------------------------------------- helpers
def causal_softmax(s):
    n = s.shape[0]
    z = np.where(np.tril(np.ones((n, n), bool)), s, -np.inf)
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def residue(W, V, g):
    n = W.shape[0]
    return (1 - g) * solve_triangular(np.eye(n) - g * W, V, lower=True)


def path_scalar(g):
    """G_ij = g_i g_{i-1} ... g_{j+1}, G_ii = 1, 0 above; direct row products."""
    n = len(g)
    G = np.zeros((n, n), dtype=np.result_type(g, 1.0))
    for i in range(n):
        G[i, i] = 1
        for j in range(i - 1, -1, -1):
            G[i, j] = g[j + 1] * G[i, j + 1] if False else G[i, j + 1] * g[j + 1]
    return G


def qmul(a, b):
    a0, a1, a2, a3 = np.moveaxis(a, -1, 0)
    b0, b1, b2, b3 = np.moveaxis(b, -1, 0)
    return np.stack([a0*b0 - a1*b1 - a2*b2 - a3*b3,
                     a0*b1 + a1*b0 + a2*b3 - a3*b2,
                     a0*b2 - a1*b3 + a2*b0 + a3*b1,
                     a0*b3 + a1*b2 - a2*b1 + a3*b0], -1)


def qconj(a):
    return a * np.array([1, -1, -1, -1], dtype=a.dtype)


def Lmat(q):
    w, x, y, z = q
    return np.array([[w, -x, -y, -z], [x, w, -z, y], [y, z, w, -x], [z, -y, x, w]])


def path_quat(g):
    """U_ij = g_i g_{i-1} ... g_{j+1} (Hamilton, latest on the left), direct."""
    n = len(g)
    U = np.zeros((n, n, 4))
    for i in range(n):
        U[i, i] = [1, 0, 0, 0]
        for j in range(i - 1, -1, -1):
            U[i, j] = qmul(U[i, j + 1], g[j + 1])
    return U


def unit_quats(rng, n):
    q = rng.standard_normal((n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def rotq(axis, deg):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a)
    h = math.radians(deg) / 2
    return np.concatenate([[math.cos(h)], math.sin(h) * a])


def gated_W(G, s):
    """W_ij = G_ij e^{s_ij} / sum_j |G_ij| e^{s_ij}  (beta = 1, causal)."""
    n = s.shape[0]
    mask = np.tril(np.ones((n, n), bool))
    e = np.where(mask, np.exp(s), 0.0)
    Z = (np.abs(G) * e).sum(1, keepdims=True)
    return G * e / Z


def seeds0():
    rng = np.random.default_rng(0)
    s = rng.standard_normal((N, N))
    V = rng.standard_normal((N, DV))
    return rng, s, V


# ---------------------------------------------------------------- D1..D5
def t1():
    _, s, _ = seeds0()
    W = causal_softmax(s)
    ev = np.linalg.eigvals(W)
    d = np.diag(W)
    err = max(np.max(np.abs(np.sort(ev.real) - np.sort(d))), np.max(np.abs(ev.imag)))
    return {"max_eig_minus_diag": float(err), "unit_diag": np.flatnonzero(d == 1.0).tolist()}


def t2():
    rng, s, V = seeds0()
    W = causal_softmax(s)
    err = [float(np.max(np.abs(residue(W, V, g) - V[0]))) for g in (0.99, 0.999, 0.9999)]
    s2 = rng.standard_normal((N, N)); V2 = rng.standard_normal((N, DV)); V2[0] = V[0]
    r1 = residue(W, V, 0.9999); r2 = residue(causal_softmax(s2), V2, 0.9999)
    tri = np.tril(np.ones((N, N), bool))
    inp = max(np.max(np.abs((s2 - s)[tri])), np.max(np.abs(V2 - V)))
    sv = np.linalg.svd(r1, compute_uv=False)
    return {"err": err, "resid_change": float(np.max(np.abs(r2 - r1))),
            "input_change": float(inp), "s2_over_s1": float(sv[1] / sv[0])}


def t3():
    rng, s, V = seeds0()
    th = rng.uniform(-np.pi, np.pi, N)
    G = path_scalar(np.exp(1j * th))
    W = causal_softmax(s)
    WG = gated_W(G, s)
    phi = np.concatenate([[0.0], np.cumsum(th[1:])])
    D = np.exp(1j * phi)
    sim = np.max(np.abs(WG - D[:, None] * W / D[None, :]))
    ev = np.linalg.eigvals(WG)
    eshift = np.max(np.abs(np.sort_complex(ev) - np.sort(np.diag(W))))
    ReW = WG.real
    v0 = V[0] / np.linalg.norm(V[0])
    off = []
    for g in (0.99, 0.999, 0.9999):
        r = residue(ReW, V, g)
        off.append(float(np.max(np.linalg.norm(r - np.outer(r @ v0, v0), axis=1))))
    return {"sim_resid": float(sim), "eig_shift": float(eshift),
            "re_diag_bitwise_equal": bool(np.array_equal(np.diag(ReW), np.diag(W))),
            "re_diag_max_diff": float(np.max(np.abs(np.diag(ReW) - np.diag(W)))),
            "off_v0": off}


def t4():
    rng, s, _ = seeds0()
    u = unit_quats(rng, N)
    U = path_quat(u)
    W = causal_softmax(s)
    WG = np.zeros((4 * N, 4 * N))
    for i in range(N):
        for j in range(i + 1):
            WG[4*i:4*i+4, 4*j:4*j+4] = W[i, j] * Lmat(U[i, j])
    Pi = np.zeros((N, 4)); Pi[0] = [1, 0, 0, 0]
    for i in range(1, N):
        Pi[i] = qmul(u[i], Pi[i - 1])
    D = np.zeros((4 * N, 4 * N))
    for i in range(N):
        D[4*i:4*i+4, 4*i:4*i+4] = Lmat(Pi[i])
    return {"sim_resid": float(np.max(np.abs(WG - D @ np.kron(W, np.eye(4)) @ D.T)))}


def gate_m(rng):
    m = rng.uniform(0.3, 1.0, N); m[16] = 0.0; m[40] = 0.0
    return m


def lam_law(G, s, m, modulus):
    """lambda_i = 1/(1 + m_i rho_i), rho_i = sum_{j<i} G_{i-1,j} e^{s_ij - s_ii}."""
    lam = np.ones(N, dtype=complex)
    for i in range(1, N):
        Gp = np.abs(G[i - 1, :i]) if modulus else G[i - 1, :i]
        rho = np.sum(Gp * np.exp(s[i, :i] - s[i, i]))
        lam[i] = 1 / (1 + m[i] * rho)
    return lam


def t5():
    rng, s, V = seeds0()
    m = gate_m(rng)
    G = path_scalar(m)
    W = gated_W(G, s)
    lam = lam_law(G, s, m, modulus=False)
    res = np.max(np.abs(lam - np.diag(W)))
    sv = np.linalg.svd(residue(W, V, 0.9999), compute_uv=False)
    return {"lambda_law_resid": float(res), "unit_diag": np.flatnonzero(np.diag(W) == 1.0).tolist(),
            "sigma": [float(x) for x in sv[:5]]}


# --------------------------------------------------------- DAG (T6-T8, T7)
def dag(n, rng):
    roots, parents = [0], [[]]
    for i in range(1, n):
        if rng.random() < 0.1:
            roots.append(i); parents.append([])
        else:
            parents.append(sorted(rng.choice(i, size=min(2, i), replace=False).tolist()))
    L = np.zeros((n, n))
    for i in range(n):
        if parents[i]:
            L[i, parents[i]] = 6.0
        else:
            L[i, i] = 6.0
    return roots, parents, L


def sparsemax_rows(L):
    n = L.shape[0]
    W = np.zeros((n, n))
    for i in range(n):
        z = L[i, :i + 1]
        zs = np.sort(z)[::-1]
        cs = np.cumsum(zs)
        k = np.arange(1, i + 2)
        kk = k[1 + k * zs > cs].max()
        tau = (cs[kk - 1] - 1) / kk
        W[i, :i + 1] = np.maximum(z - tau, 0.0)
    return W


def absorb(W, roots):
    n = W.shape[0]
    rs = set(roots)
    T = [i for i in range(n) if i not in rs]
    Q = W[np.ix_(T, T)]; R = W[np.ix_(T, roots)]
    return T, solve_triangular(np.eye(len(T)) - Q, R, lower=True)


def ancestry(n, roots, parents):
    ridx = {r: a for a, r in enumerate(roots)}
    anc = np.zeros((n, len(roots)), bool)
    for i in range(n):
        if i in ridx:
            anc[i, ridx[i]] = True
        else:
            for p in parents[i]:
                anc[i] |= anc[p]
    return anc


_DAG = {}


def dag_case(n):
    if n not in _DAG:
        rng = np.random.default_rng(1 + n)       # seed-1 family, one stream per n
        roots, parents, L = dag(n, rng)
        Wsp = sparsemax_rows(L)
        Wso = causal_softmax(L)
        for r in roots:                           # oracle: roots absorbing
            Wso[r] = 0.0; Wso[r, r] = 1.0
        V = rng.standard_normal((n, DV))
        _DAG[n] = (roots, parents, L, Wsp, Wso, V)
    return _DAG[n]


def t6():
    out = {}
    for n in (64, 1024):
        roots, parents, L, Wsp, _, _ = dag_case(n)
        T, B = absorb(Wsp, roots)
        out[str(n)] = {"mixed": int(np.sum((B > 0).sum(1) >= 2)), "transient": len(T),
                       "anchors": len(roots), "roots_unit_diag": bool(np.all(np.diag(Wsp)[roots] == 1.0))}
    return out


def depth(Lmask):
    n = Lmask.shape[0]
    d = np.zeros(n, int)
    for i in range(1, n):
        js = np.flatnonzero(Lmask[i, :i])
        if len(js):
            d[i] = d[js].max() + 1
    return int(d.max())


def jump_chain(W, V, g, hops_max, exact=None):
    n = W.shape[0]
    Dg = np.diag(W).copy()
    Lo = W - np.diag(Dg)
    y = (1 - g) * V / (1 - g * Dg)[:, None]
    Nm = g * Lo / (1 - g * Dg)[:, None]
    if exact is None:
        exact = residue(W, V, g)
    acc = y.copy(); errs = [float(np.max(np.abs(acc - exact)))]
    for _ in range(hops_max):
        y = Nm @ y; acc = acc + y
        errs.append(float(np.max(np.abs(acc - exact))))
    return errs, Nm


def t7():
    out = {}
    for n in (64, 1024):
        _, _, _, Wsp, _, V = dag_case(n)
        D = depth((Wsp - np.diag(np.diag(Wsp))) != 0)
        errs, Nm = jump_chain(Wsp, V, GAM, D + 1)
        P = Nm.copy()
        for _ in range(D):
            P = Nm @ P
        out[str(n)] = {"D": D, "err_Dm1": errs[D - 1], "err_D": errs[D],
                       "nilpotent_exact": bool(np.all(P == 0)), "ND_nonzero": bool(np.any(np.linalg.matrix_power(Nm, D) != 0))}
    return out


def t8():
    out = {}
    for n in (64, 1024):
        roots, parents, L, Wsp, Wso, _ = dag_case(n)
        anc = ancestry(n, roots, parents)
        T, Bsp = absorb(Wsp, roots)
        _, Bso = absorb(Wso, roots)
        A = anc[T]
        ths = np.logspace(-12, 0, 241)
        mm = [int(np.sum((Bso > t) != A)) for t in ths]
        i = next(i for i in range(n - 1, 0, -1) if len(parents[i]) == 2)
        p = 2
        row = causal_softmax(L)[i]
        nonpar = np.setdiff1d(np.arange(i + 1), parents[i])
        out[str(n)] = {
            "sparse_mismatch": int(np.sum((Bsp > 0) != A)), "cells": int(A.size),
            "sparse_false_mass": float(np.mean(np.where(A, 0.0, Bsp).sum(1))),
            "soft_false_mass": float(np.mean(np.where(A, 0.0, Bso).sum(1))),
            "soft_best_mismatch": min(mm), "best_threshold": float(ths[int(np.argmin(mm))]),
            "leak_row": i, "leak": float(row[nonpar].sum()),
            "leak_law": float((i + 1 - p) / (p * math.exp(6.0) + i + 1 - p))}
    return out


def t9():
    _, s, V = seeds0()
    W = causal_softmax(s)
    A = np.eye(N) - GAM * W
    b = (1 - GAM) * V
    x = np.zeros_like(b)
    for i in range(N):                       # plain forward substitution
        x[i] = (b[i] - A[i, :i] @ x[:i]) / A[i, i]
    return {"fwd_vs_dense": float(np.max(np.abs(x - np.linalg.solve(A, b))))}


# ---------------------------------------------------------------- T10
PHI = (1 + 5 ** 0.5) / 2


def closure(gens, cap=5000):
    key = lambda q: tuple(np.round(q, 8) + 0.0)
    seen = {key(np.array([1.0, 0, 0, 0])): np.array([1.0, 0, 0, 0])}
    frontier = list(seen.values())
    while frontier and len(seen) <= cap:
        nxt = []
        for q in frontier:
            for g in gens:
                r = qmul(g, q); k = key(r)
                if k not in seen:
                    seen[k] = r; nxt.append(r)
                    if len(seen) > cap:
                        break
        frontier = nxt
    return list(seen.values()), not frontier


def canon(q):
    i = np.flatnonzero(np.abs(q) > 1e-9)[0]
    return q if q[i] > 0 else -q


def t10():
    g_old = [rotq([0, 0, 1], 72), rotq([1, 1, 1], 120)]
    old, closed_old = closure(g_old)
    g_new = [rotq([0, 1, PHI], 72), rotq([1, 1, 1], 120)]
    new, closed_new = closure(g_new)
    rots = {tuple(np.round(canon(q), 8) + 0.0) for q in new}
    return {"q2_as_written": len(old), "q2_closed": closed_old, "corrected_quats": len(new),
            "corrected_closed": closed_new, "corrected_rots": len(rots),
            "has_minus_one": any(np.allclose(q, [-1, 0, 0, 0]) for q in new),
            "axis_angle_old_deg": math.degrees(math.acos(1 / 3 ** 0.5)),
            "axis_angle_new_deg": math.degrees(math.acos((1 + PHI) / (math.sqrt(1 + PHI**2) * 3 ** 0.5)))}


def group2I():
    new, _ = closure([rotq([0, 1, PHI], 72), rotq([1, 1, 1], 120)])
    return np.array(new)


# ---------------------------------------------------------------- X1, X2
def x1_parts(rng, s):
    m = gate_m(rng)
    u = unit_quats(rng, N)
    g = m[:, None] * u
    Gq = path_quat(g)                              # non-unit quaternion path products
    M = np.linalg.norm(Gq, axis=2)
    mask = np.tril(np.ones((N, N), bool))
    e = np.where(mask, np.exp(s), 0.0)
    Z = (M * e).sum(1)
    Wm = M * e / Z[:, None]
    return m, u, Gq, e, Z, Wm


def block_W(Gq, e, Z):
    WG = np.zeros((4 * N, 4 * N))
    for i in range(N):
        for j in range(i + 1):
            WG[4*i:4*i+4, 4*j:4*j+4] = (e[i, j] / Z[i]) * Lmat(Gq[i, j])
    return WG


def pis(u):
    Pi = np.zeros((N, 4)); Pi[0] = [1, 0, 0, 0]
    for i in range(1, N):
        Pi[i] = qmul(u[i], Pi[i - 1])
    return Pi


def x1():
    rng = np.random.default_rng(7)
    s = rng.standard_normal((N, N))
    m, u, Gq, e, Z, Wm = x1_parts(rng, s)
    WG = block_W(Gq, e, Z)
    Pi = pis(u)
    D = np.zeros((4 * N, 4 * N))
    for i in range(N):
        D[4*i:4*i+4, 4*i:4*i+4] = Lmat(Pi[i])
    sim = np.max(np.abs(WG - D @ np.kron(Wm, np.eye(4)) @ D.T))
    ev = np.linalg.eigvals(WG)
    evd = max(np.max(np.abs(np.sort(ev.real) - np.sort(np.repeat(np.diag(Wm), 4)))), np.max(np.abs(ev.imag)))
    return {"sim_resid": float(sim), "eig_vs_diag": float(evd),
            "absorbers": np.flatnonzero(np.diag(Wm) == 1.0).tolist()}


def x2a():
    rng = np.random.default_rng(7)
    s = rng.standard_normal((N, N))
    m, u, Gq, e, Z, Wm = x1_parts(rng, s)
    v = rng.standard_normal((N, 4))

    def out_direct(u, s, v):
        g = m[:, None] * u
        Gq = path_quat(g)
        M = np.linalg.norm(Gq, axis=2)
        e = np.where(np.tril(np.ones((N, N), bool)), np.exp(s), 0.0)
        Z = (M * e).sum(1)
        return np.einsum("ij,ijk->ik", e / Z[:, None], qmul(Gq, np.broadcast_to(v[None], (N, N, 4)))), M

    def out_scan(u, s, v, M):
        Pi = pis(u)
        U = qmul(np.broadcast_to(Pi[:, None], (N, N, 4)), qconj(np.broadcast_to(Pi[None], (N, N, 4))))
        e = np.where(np.tril(np.ones((N, N), bool)), np.exp(s), 0.0)
        Z = (M * e).sum(1)
        return np.einsum("ij,ijk->ik", M * e / Z[:, None], qmul(U, np.broadcast_to(v[None], (N, N, 4))))

    o1, M1 = out_direct(u, s, v); c1 = out_scan(u, s, v, M1)
    u2, s2, v2 = u.copy(), s.copy(), v.copy()
    u2[:41] = unit_quats(rng, 41); v2[:40] = rng.standard_normal((40, 4)); s2[:, :40] = rng.standard_normal((N, 40))
    o2, M2 = out_direct(u2, s2, v2); c2 = out_scan(u2, s2, v2, M2)
    return {"change_direct": float(np.max(np.abs(o2[40:] - o1[40:]))),
            "change_scan": float(np.max(np.abs(c2[40:] - c1[40:]))),
            "change_before_40_direct": float(np.max(np.abs(o2[:40] - o1[:40])))}


def x2b():
    rng = np.random.default_rng(7)
    n, seg = 4096, 64
    u = unit_quats(rng, n).astype(np.float32)
    u64 = u.astype(np.float64)
    one32, one64 = np.array([1, 0, 0, 0], np.float32), np.array([1.0, 0, 0, 0])
    Pi = np.zeros((n, 4), np.float32); Pi[0] = u[0]
    for i in range(1, n):
        Pi[i] = qmul(u[i], Pi[i - 1])
    P = np.zeros((n, 4), np.float32); R = np.zeros((n, 4))
    for i in range(n):
        if i % seg == 0:
            P[i] = one32; R[i] = one64
        else:
            P[i] = qmul(u[i], P[i - 1]); R[i] = qmul(u64[i], R[i - 1])
    a = (np.arange(n) // seg) * seg
    Ug = qmul(Pi, qconj(Pi[a])).astype(np.float64)
    Ug[a == np.arange(n)] = one64
    eg = np.abs(Ug - R).max(1); es = np.abs(P.astype(np.float64) - R).max(1)
    return {"global": float(eg.max()), "segmented": float(es.max()),
            "global_mean": float(eg.mean()), "segmented_mean": float(es.mean()),
            "global_norm_drift_max": float(np.abs(np.linalg.norm(Pi.astype(np.float64), axis=1) - 1).max())}


# ---------------------------------------------------------------- X3, KR
def reset_seq(n, density, rng):
    tok = rng.choice(3, size=n, p=[density, (1 - density) / 2, (1 - density) / 2])
    tok[0] = 0                                   # 0 = R, 1 = a, 2 = b
    return tok


def truth(tok, gens, grp):
    q = np.zeros((len(tok), 4)); cur = np.array([1.0, 0, 0, 0])
    for i, t in enumerate(tok):
        cur = np.array([1.0, 0, 0, 0]) if t == 0 else grp[np.argmax(grp @ qmul(gens[t], cur))]
        q[i] = cur
    return q


def ang_deg(a, b):
    d = qmul(qconj(b), a)
    return np.degrees(2 * np.arctan2(np.linalg.norm(d[..., 1:], axis=-1), np.abs(d[..., 0])))


GENS = {1: rotq([0, 1, PHI], 72), 2: rotq([1, 1, 1], 120)}


def x3a():
    rng = np.random.default_rng(7)
    grp = group2I()
    tok = reset_seq(4096, 0.05, rng)
    tq = truth(tok, GENS, grp)
    out = np.zeros((4096, 4)); P = np.array([1.0, 0, 0, 0]); a = 0
    for i, t in enumerate(tok):
        if t == 0:
            P = np.array([1.0, 0, 0, 0]); a = i
        else:
            P = qmul(GENS[t], P)
        out[i] = P / (i - a + 1)                 # W_ia = 1/(i-a+1), only v_a != 0
    qh = out / np.linalg.norm(out, axis=1, keepdims=True)
    err = ang_deg(qh, tq)
    return {"resets": int((tok == 0).sum()), "max_err_deg": float(err.max())}


def recency_acc(tok, slope, grp, tq):
    S = np.zeros(4); hit = 0
    for i, t in enumerate(tok):
        S = math.exp(-slope) * (S if t == 0 else qmul(GENS[t], S))
        if t == 0:
            S = S + np.array([1.0, 0, 0, 0])
        pred = grp[np.argmax(np.abs(grp @ S))]
        hit += abs(pred @ tq[i]) > 1 - 1e-9
    return hit / len(tok)


def x3b():
    rng = np.random.default_rng(7)
    grp = group2I()
    tok = reset_seq(4096, 0.05, rng)
    tq = truth(tok, GENS, grp)
    return {"acc": [recency_acc(tok, sl, grp, tq) for sl in (0.05, 0.2, 1.0)], "resets": int((tok == 0).sum())}


def kr():
    grp = group2I(); out = {}
    for dens in (0.05, 0.2, 0.5):
        rng = np.random.default_rng(11)
        tok = reset_seq(4096, dens, rng)
        tq = truth(tok, GENS, grp)
        out[str(dens)] = {str(sl): recency_acc(tok, sl, grp, tq) for sl in (0.5, 1.0, 2.0, 4.0)}
    return out


# ---------------------------------------------------------------- X4
def rot3(axis, deg):
    a = np.asarray(axis, float); a /= np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    t = math.radians(deg)
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


def sheaf_lap(rho, d):
    nv = 4
    delta = np.zeros((4 * d, nv * d))
    for e in range(4):
        v, w = e, (e + 1) % 4
        delta[e*d:(e+1)*d, v*d:(v+1)*d] = np.eye(d)
        delta[e*d:(e+1)*d, w*d:(w+1)*d] = -rho[e]
    ev = np.linalg.eigvalsh(delta.T @ delta)
    h0 = int(np.sum(ev < 1e-10))
    return h0, float(ev[ev >= 1e-10].min())


def x4():
    rng = np.random.default_rng(7)
    out_h, out_g = [], []
    gq = [Lmat(q) for q in unit_quats(rng, 4)]
    g3 = [rot3(rng.standard_normal(3), rng.uniform(0, 360)) for _ in range(4)]
    for hol in (False, True):
        for d, gs, H in ((4, gq, Lmat(rotq(rng.standard_normal(3), 50))),
                         (3, g3, rot3(rng.standard_normal(3), 50))):
            rho = [gs[e] @ gs[(e + 1) % 4].T for e in range(4)]    # pure gauge
            if hol:
                rho[0] = rho[0] @ H
            h0, gap = sheaf_lap(rho, d)
            out_h.append(h0); out_g.append(gap)
    return {"H0": out_h, "gap": out_g}


# ---------------------------------------------------------------- PF
def pf():
    W = lambda m: m**2 * (1 - m)**2
    dW = lambda m: 2*m*(1 - m)**2 - 2*m**2*(1 - m)
    costs = []
    for eps in (0.03, 0.1, 0.3, 1, 3, 10, 30):
        L = int(60 * eps) + 200
        def f(x):
            m = np.concatenate([[0.0], x, [1.0]])
            dm = np.diff(m)
            val = eps * np.sum(dm**2) + np.sum(W(x)) / eps
            gr = 2 * eps * (dm[:-1] - dm[1:]) + dW(x) / eps
            return val, gr
        best = np.inf
        for x0 in (np.linspace(0, 1, L + 2)[1:-1], (np.arange(L) >= L // 2).astype(float)):
            r = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=[(0, 1)] * L,
                         options={"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-12})
            best = min(best, r.fun)
        costs.append(float(best))
    return {"cost": costs, "lattice": "m_0=0 ... m_{L+1}=1, unit spacing, L=60*eps+200 free sites"}


# ---------------------------------------------------------------- breaks
def b2():
    rng, s, _ = seeds0()
    m = gate_m(rng)
    th = rng.uniform(-np.pi, np.pi, N)
    g = m * np.exp(1j * th)
    G = path_scalar(g)
    W = gated_W(G, s)
    d = np.diag(W)
    return {"as_written": float(np.max(np.abs(lam_law(G, s, m, False) - d))),
            "modulus_rho": float(np.max(np.abs(lam_law(G, s, m, True) - d))),
            "unit_diag": np.flatnonzero(d == 1.0).tolist()}


def b3():
    out = {}
    for n in (64, 1024):
        rng = np.random.default_rng(0)
        s = rng.standard_normal((n, n)); V = rng.standard_normal((n, DV))
        W = causal_softmax(s)
        D = depth((W - np.diag(np.diag(W))) != 0)
        ex = residue(W, V, GAM)
        scale = np.max(np.abs(ex))
        errs, _ = jump_chain(W, V, GAM, 200, ex)
        hj = next((k for k, e in enumerate(errs) if e <= 1e-12 * scale), None)
        y = (1 - GAM) * V; acc = y.copy(); hn = None
        for k in range(1, 6001):
            y = GAM * (W @ y); acc += y
            if np.max(np.abs(acc - ex)) <= 1e-12 * scale:
                hn = k; break
        out[str(n)] = {"D": D, "jump_hops_to_1e-12": hj, "neumann_hops_to_1e-12": hn}
    return out


def _torch_case(seed):
    import torch
    rng = np.random.default_rng(seed)
    q = torch.tensor(rng.standard_normal((N, DV)))
    k = torch.tensor(rng.standard_normal((N, DV)))
    u = rng.uniform(0.3, 1.0, N); u[16] = -1.0; u[40] = -1.0   # clamp -> exact 0
    th1 = rng.uniform(-np.pi, np.pi, N); th2 = rng.uniform(-np.pi, np.pi, N)
    return torch, q, k, torch.tensor(u), torch.tensor(th1), torch.tensor(th2)


def _op():
    import sys
    sys.path.insert(0, "C:/Users/seal/Desktop/New folder (32)")
    from ceq import arm_smprime
    return arm_smprime.operator


def _stats(torch, W):
    d = W.diagonal()
    off = W.clone(); off.fill_diagonal_(0)
    struct = int(((off == 0).all(1)).sum())
    unit = int(((d.real == 1.0) & (d.imag == 0)).sum())
    return {"unit": unit, "structural": struct, "max_diag": float(d.real.max()),
            "re_row_mass": float(W.real.sum(1).mean()), "mod_row_mass": float(W.abs().sum(1).mean()),
            "unit_idx": torch.nonzero((d.real == 1.0)).flatten().tolist()}


def b1():
    op = _op()
    torch, q, k, u, th1, _ = _torch_case(3)
    out = {}
    for b in (1.0, 1.09, 0.96, 0.99):
        W = op(q, k, u, th1, beta=b)
        st = _stats(torch, W)
        st["pole_gamma"] = (1 / st["max_diag"]) if st["max_diag"] > 1 else None
        d = W.diagonal().real
        st["absorber_diag"] = [float(d[i]) for i in (0, 16, 40)]
        out[str(b)] = st
    return out


def b4():
    op = _op()
    torch, q, k, u, th1, th2 = _torch_case(3)
    out = {}
    for route in ("gate", "logit", "both"):
        W1 = op(q, k, u, th1, beta=1.0, phase_route=route)
        W2 = op(q, k, u, th2, beta=1.0, phase_route=route)
        s1, s2 = _stats(torch, W1), _stats(torch, W2)
        out[route] = {"diag_bitwise": bool(torch.equal(W1.diagonal(), W2.diagonal())),
                      "count_equal": s1["unit"] == s2["unit"], "count": s1["unit"],
                      "max_abs_diag_diff": float((W1.diagonal() - W2.diagonal()).abs().max())}
    out["beta"] = {}
    for b in (1.09, 0.96, 0.99):
        c1 = _stats(torch, op(q, k, u, th1, beta=b))["unit"]
        c2 = _stats(torch, op(q, k, u, th2, beta=b))["unit"]
        out["beta"][str(b)] = {"count": c1, "count_other_theta": c2, "rdiag_check_passes": c1 == c2}
    # fp32 manufactured units: same operator, peaked self-scores, no zero gates
    rng = np.random.default_rng(5)
    x = rng.standard_normal((N, DV))
    for dt, name in ((torch.float64, "f64"), (torch.float32, "f32")):
        qq = torch.tensor(3.0 * x, dtype=dt); kk = torch.tensor(3.0 * x, dtype=dt)
        uu = torch.full((N,), 0.9, dtype=dt); tt = torch.zeros(N, dtype=dt)
        W = op(qq, kk, uu, tt, beta=1.0)
        out["manufactured_" + name] = _stats(torch, W)["unit"]
    return out

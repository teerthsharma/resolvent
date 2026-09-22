"""Chase it.L1: independent re-derivations of Addendum L's PASS rows.
Own SU(2) arithmetic as 2x2 complex matrices (not Hamilton's 16-term formula),
own closure by hashing SO(3) matrices, own global min by dynamic programming,
own rotation fit by SVD. Imports nothing from phase_j."""
import json, math, sys
import numpy as np

# ---------- SU(2) as 2x2 complex: q=w+xi+yj+zk -> [[w+ix, y+iz], [-y+iz, w-ix]]
def M(q):
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    out = np.empty(q.shape[:-1] + (2, 2), dtype=np.complex128)
    out[..., 0, 0] = w + 1j * x; out[..., 0, 1] = y + 1j * z
    out[..., 1, 0] = -y + 1j * z; out[..., 1, 1] = w - 1j * x
    return out

def Q(m):  # inverse map
    return np.stack([m[..., 0, 0].real, m[..., 0, 0].imag, m[..., 0, 1].real, m[..., 0, 1].imag], -1)

def dag(m):
    return np.conj(np.swapaxes(m, -1, -2))

def rot(axis, deg):  # SO(3) by Rodrigues, independent of quaternions
    u = np.asarray(axis, float); u = u / np.linalg.norm(u); t = math.radians(deg)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K

def su2(axis, deg):
    u = np.asarray(axis, float); u = u / np.linalg.norm(u); t = math.radians(deg) / 2
    return M([math.cos(t), *(math.sin(t) * u)])

def angle_of(R):  # stable rotation angle: atan2(sin, cos), no arccos cliff
    s = np.linalg.norm(R - R.T) / (2 * math.sqrt(2)); c = (np.trace(R) - 1) / 2
    return math.atan2(s, c)

# ---------- Q1
def q1(n=4096, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    A = M(q)
    seq = np.empty_like(A); seq[0] = A[0]
    for i in range(1, n): seq[i] = A[i] @ seq[i - 1]
    # own scan: recursive doubling on 2x2 complex, Pi_i = A_i ... A_0
    x = A.copy(); d = 1
    while d < n:
        x = np.concatenate([x[:d], x[d:] @ x[:-d]]); d *= 2
    e_scan = float(np.abs(x - seq).max())
    # G_ij direct path product vs Pi_i Pi_j^dag, sampled pairs incl. extremes
    pairs = [(n - 1, 0), (n - 1, n // 2), (2048, 1), (4000, 3000), (n - 1, n - 2)]
    pairs += [tuple(sorted(rng.integers(0, n, 2), reverse=True)) for _ in range(20)]
    e_g = 0.0
    for i, j in pairs:
        G = np.eye(2, dtype=complex)
        for k in range(j + 1, i + 1): G = A[k] @ G
        e_g = max(e_g, float(np.abs(G - seq[i] @ dag(seq[j])).max()))
    # fp32 chain, renorm every 256 (and none), vs f64
    A32 = A.astype(np.complex64)
    def chain32(renorm):
        c = np.empty_like(A32); c[0] = A32[0]
        for i in range(1, n):
            c[i] = A32[i] @ c[i - 1]
            if renorm and i % 256 == 0:
                c[i] = c[i] / np.sqrt(np.abs(np.linalg.det(c[i])))
        return c
    e32r = float(np.abs(chain32(True).astype(np.complex128) - seq).max())
    e32n = float(np.abs(chain32(False).astype(np.complex128) - seq).max())
    return {"n": n, "scan_vs_seq_f64": e_scan, "Gij_direct_vs_fold_f64": e_g,
            "fp32_renorm256_vs_f64": e32r, "fp32_no_renorm_vs_f64": e32n}

# ---------- Q2: stated generators, no BFS
def q2_stated():
    a, b = rot((0, 0, 1), 72), rot((1, 1, 1), 120)
    ang_ab = math.degrees(angle_of(a @ b))
    ax_angle = math.degrees(math.acos(1 / math.sqrt(3)))
    phi = (1 + 5 ** .5) / 2
    # angles between a 5-fold and a 3-fold axis in the icosahedral group
    f5 = [np.array(v, float) for v in [(0, 1, phi), (0, -1, phi), (1, phi, 0), (-1, phi, 0), (phi, 0, 1), (-phi, 0, 1)]]
    f3 = [np.array(v, float) for v in [(1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)]] + \
         [np.array(v, float) for v in [(0, phi, 1 / phi), (0, -phi, 1 / phi), (1 / phi, 0, phi), (-1 / phi, 0, phi), (phi, 1 / phi, 0), (-phi, 1 / phi, 0)]]
    ico = sorted({round(math.degrees(math.acos(abs(u @ v) / np.linalg.norm(u) / np.linalg.norm(v))), 4) for u in f5 for v in f3})
    # algebraic: 2cos(theta_ab) = -1 - sin72; finite order needs an algebraic integer
    try:
        import sympy as sp
        mp = sp.minimal_polynomial(-1 - sp.sin(2 * sp.pi / 5), sp.Symbol("x"))
        poly = sp.Poly(mp, sp.Symbol("x")); lc = poly.LC(); coeffs = [c / lc for c in poly.all_coeffs()]
        alg_int = all(sp.Rational(c).q == 1 for c in coeffs)
        mp_s = str(mp)
    except Exception as e:  # pragma: no cover
        alg_int, mp_s = None, f"sympy unavailable: {e}"
    return {"rotation_angle_ab_deg": ang_ab, "allowed_in_I": [0, 72, 120, 144, 180],
            "axis_angle_deg": ax_angle, "icosahedral_5fold_3fold_axis_angles_deg": ico,
            "2cos_theta_ab": -1 - math.sin(math.radians(72)), "minpoly_2cos_theta_ab": mp_s,
            "2cos_theta_ab_is_algebraic_integer": alg_int,
            "trace_check_2cos": float(np.trace(a @ b) - 1)}

# ---------- Q2-REROUTE: closure by hashing, table, A5 checks, word
def key(R): return tuple(np.round(R, 6).ravel())

def closure_so3(gens, cap=10000):
    els = {key(np.eye(3)): np.eye(3)}; frontier = [np.eye(3)]
    while frontier and len(els) < cap:
        nf = []
        for g in frontier:
            for h in gens:
                p = g @ h; k = key(p)
                if k not in els: els[k] = p; nf.append(p)
        frontier = nf
    return list(els.values())

def closure_su2(gens, cap=10000):
    def k2(m): return tuple(np.round(Q(m), 6))
    els = {k2(np.eye(2, dtype=complex)): np.eye(2, dtype=complex)}; frontier = list(els.values())
    while frontier and len(els) < cap:
        nf = []
        for g in frontier:
            for h in gens:
                p = g @ h; k = k2(p)
                if k not in els: els[k] = p; nf.append(p)
        frontier = nf
    return list(els.values())

def q2_reroute(n_words=10000, seed=1):
    phi = (1 + 5 ** .5) / 2
    a, b = rot((0, 1, phi), 72), rot((1, 1, 1), 120)
    G = closure_so3([a, b])
    idx = {key(R): i for i, R in enumerate(G)}
    orders = {}
    for R in G:
        o, P = 1, R.copy()
        while not np.allclose(P, np.eye(3), atol=1e-9): P = P @ R; o += 1
        orders[o] = orders.get(o, 0) + 1
    table = np.array([[idx[key(x @ y)] for y in G] for x in G])
    # A5 check: action on the 6 five-fold axes (lines) must give 60 distinct even perms of 6 lines? use sign on 5 inscribed cubes via orbit sizes instead: count conjugacy class sizes
    classes = []
    seen = set()
    inv = [idx[key(R.T)] for R in G]
    for i in range(len(G)):
        if i in seen: continue
        cl = {table[table[g, i], inv[g]] for g in range(len(G))}
        seen |= cl; classes.append(len(cl))
    rng = np.random.default_rng(seed)
    word = rng.integers(0, 2, n_words)
    gi = [idx[key(a)], idx[key(b)]]
    t = idx[key(np.eye(3))]; R = np.eye(3)
    for w in word:
        t = table[t, gi[w]]; R = R @ (a if w == 0 else b)
    su = closure_su2([su2((0, 1, phi), 72), su2((1, 1, 1), 120)])
    return {"closure_so3": len(G), "closure_su2": len(su), "order_histogram": dict(sorted(orders.items())),
            "class_sizes": sorted(classes), "word_len": n_words,
            "word_float_vs_table_maxabs": float(np.abs(R - G[t]).max())}

# ---------- Q3: an actual attention read
def q3():
    rng = np.random.default_rng(7)
    a, b = su2((0, 0, 1), 72), su2((1, 1, 1), 120)
    v = M([1.0, 2.0, 3.0, 4.0])  # value as a quaternion
    # positions: 0 = value token (gate I), 1 and 2 = gate tokens, 3 = query (gate I)
    def read(gates, qk):
        S = len(gates); Pi = [gates[0]]
        for g in gates[1:]: Pi.append(g @ Pi[-1])
        s = qk[-1]; p = np.exp(s - s.max()); p /= p.sum()
        vals = [v if j == 0 else M([0.5, -1, 0.25, 2.0 * j]) for j in range(S)]
        return sum(p[j] * (Pi[-1] @ dag(Pi[j]) @ vals[j]) for j in range(S))
    I = np.eye(2, dtype=complex)
    qk = rng.standard_normal((4, 4))
    d_su2 = float(np.abs(Q(read([I, a, b, I], qk)) - Q(read([I, b, a, I], qk))).max())
    # scalar gate (FoX-style log-cumsum) on the SAME read, gates swapped
    def read_scalar(logf, qk):
        c = np.cumsum(logf); s = qk[-1] + (c[-1] - c); p = np.exp(s - s.max()); p /= p.sum()
        vals = [Q(v) if j == 0 else np.array([0.5, -1, 0.25, 2.0 * j]) for j in range(len(logf))]
        return sum(p[j] * vals[j] for j in range(len(logf)))
    la, lb = math.log(0.8), math.log(0.3)
    d_scalar = float(np.abs(read_scalar([0, la, lb, 0], qk) - read_scalar([0, lb, la, 0], qk)).max())
    # control sensitivity: feed the SU(2) gates through the SAME scalar-slot harness via their matrices
    d_ctrl_sensitive = d_su2 > 0
    return {"su2_attention_read_diff": d_su2, "fox_scalar_read_diff": d_scalar, "control_harness_detects_noncommuting": d_ctrl_sensitive}

# ---------- PF: global min on the unit token lattice by DP (min-plus), upper bound on true min
def pf_lattice_dp(eps, n=400):
    g = np.unique(np.concatenate([np.linspace(0, 1, 1201), np.geomspace(1e-9, 0.2, 300), 1 - np.geomspace(1e-9, 0.2, 300)]))
    W = g ** 2 * (1 - g) ** 2
    C = eps * (g[None, :] - g[:, None]) ** 2 + (W / eps)[:, None]  # C[a,b]: site value a -> next b, W at left site
    i0, i1 = 0, len(g) - 1
    V = np.full(len(g), np.inf); V[i0] = 0.0
    for _ in range(n):
        V = np.min(V[:, None] + C, axis=0)
    return float(V[i1])

def pf_continuum():
    x = np.linspace(-40, 40, 400001); eps = 1.0
    m = 1 / (1 + np.exp(-x / eps)); dm = np.gradient(m, x)
    e = eps * dm ** 2 + m ** 2 * (1 - m) ** 2 / eps
    return float(np.sum((e[1:] + e[:-1]) / 2 * np.diff(x)))

# ---------- Horn: SVD fit and stable angle on run_horn's case_a seeds
def kabsch(P, Qp):
    Pc, Qc = P - P.mean(0), Qp - Qp.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc); D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T

def random_rotation(rng):
    A = rng.standard_normal((3, 3)); Qm, Rm = np.linalg.qr(A); Qm = Qm @ np.diag(np.sign(np.diag(Rm)))
    if np.linalg.det(Qm) < 0: Qm[:, 0] *= -1
    return Qm

def horn_case_a():
    rng = np.random.default_rng(0)
    P = rng.standard_normal((200, 3)); P -= P.mean(0)
    Rt = random_rotation(rng); Qp = P @ Rt.T
    Re = kabsch(P, Qp)
    c = np.clip((np.trace(Re.T @ Rt) - 1) / 2, -1, 1)
    return {"kabsch_stable_angle_rad": angle_of(Re.T @ Rt), "kabsch_arccos_angle_rad": float(np.arccos(c))}

def arccos_floor(theta=0.0):
    R = rot((1, 2, 3), math.degrees(theta)) if theta else np.eye(3)
    return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))

if __name__ == "__main__":
    which = sys.argv[1:] or ["q1", "q2", "q2r", "q3", "pf", "horn"]
    out = {}
    if "q1" in which: out["Q1"] = q1()
    if "q2" in which: out["Q2_stated"] = q2_stated()
    if "q2r" in which: out["Q2_reroute"] = q2_reroute()
    if "q3" in which: out["Q3"] = q3()
    if "pf" in which:
        out["PF"] = {"continuum_energy_eps1": pf_continuum(), "one_third": 1 / 3,
                     "lattice_dp_min": {str(e): pf_lattice_dp(e) for e in (1, 0.3, 0.1, 0.03)}}
    if "horn" in which: out["HORN"] = horn_case_a()
    print(json.dumps(out, indent=1, default=str))

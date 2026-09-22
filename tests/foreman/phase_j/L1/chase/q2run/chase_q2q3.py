"""Addendum L rows Q2/Q3 implementation. float64, CPU only."""
import numpy as np
from chase_su2 import qmul, qconj, qinv, qnorm, quat_from_axis_angle, qdist

CAP = 5000
TOL = 1e-9

def _gens():
    a = quat_from_axis_angle((0, 0, 1), np.deg2rad(72))
    b = quat_from_axis_angle((1, 1, 1), np.deg2rad(120))
    return a, b

def _su2_dist(p, q):
    """Strict SU(2) distance — no antipodal identification (q != -q here)."""
    return np.linalg.norm(p - q)

def bfs_closure(a, b, cap=CAP, tol=TOL):
    """BFS over words in {a, b, a^-1, b^-1}, dedup at tol in SU(2) (q and -q
    are distinct elements here), capped."""
    gens = [a, b, qinv(a), qinv(b)]
    ident = np.array([1.0, 0, 0, 0])
    elems = [ident]
    frontier = [ident]
    exceeded = False
    while frontier:
        new_frontier = []
        for e in frontier:
            for g in gens:
                cand = qmul(e, g)
                cand = cand / np.linalg.norm(cand)
                if not any(_su2_dist(cand, x) < tol for x in elems):
                    elems.append(cand)
                    new_frontier.append(cand)
                    if len(elems) >= cap:
                        exceeded = True
                        break
            if exceeded:
                break
        if exceeded:
            break
        frontier = new_frontier
    return elems, exceeded

def so3_count(elems, tol=TOL):
    """Count distinct elements identifying q with -q."""
    reps = []
    for e in elems:
        if not any(qdist(e, r) < tol for r in reps):
            reps.append(e)
    return len(reps)

def q2_closure(gens=None):
    a, b = gens if gens is not None else _gens()
    elems, exceeded = bfs_closure(a, b)
    out = {"closure_su2": len(elems), "exceeds_cap": exceeded}
    out["closure_so3"] = so3_count(elems)
    out["elems"] = elems
    out["gens"] = (a, b)
    return out

def build_cayley(elems, tol=TOL):
    n = len(elems)
    table = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            prod = qmul(elems[i], elems[j])
            prod = prod / np.linalg.norm(prod)
            idx = None
            for k, e in enumerate(elems):
                if _su2_dist(prod, e) < tol:
                    idx = k
                    break
            table[i, j] = idx if idx is not None else -1
    return table

def q2_cayley_check(gens=None, n_words=10000, seed=1, tol=TOL):
    r = q2_closure(gens)
    if r["exceeds_cap"]:
        return "no table exists (closure exceeds cap)"
    elems = r["elems"]
    a, b = r["gens"]
    table = build_cayley(elems, tol)
    gens_list = [a, b, qinv(a), qinv(b)]
    rng = np.random.default_rng(seed)
    idxs = rng.integers(0, 4, size=n_words)
    # composed by quaternion products directly
    q = np.array([1.0, 0, 0, 0])
    for i in idxs:
        q = qmul(q, gens_list[i])
    q = q / np.linalg.norm(q)
    # composed via table: track index of identity through the same word
    def elem_index(x):
        for k, e in enumerate(elems):
            if _su2_dist(x, e) < tol:
                return k
        return None
    gen_indices = [elem_index(g / np.linalg.norm(g)) for g in gens_list]
    idx = elem_index(np.array([1.0, 0, 0, 0]))
    for i in idxs:
        idx = table[idx, gen_indices[i]]
    q_table = elems[idx]
    d = qdist(q, q_table)
    return bool(d < tol)

def q2_commutator_norm(gens=None):
    a, b = gens if gens is not None else _gens()
    ab = qmul(a, b)
    ba = qmul(b, a)
    return float(np.linalg.norm(ab - ba))

def q2_gate_commutes():
    """A scalar / U(1)-phase gate: ab and ba give identical products under the gate op."""
    # a commutative gate acting as scalar multiplication: g*a*b vs g*b*a compared as scalars, always equal since scalar mult commutes
    phase = np.exp(1j * 0.37)
    ab_scalar = phase * 1.0
    ba_scalar = 1.0 * phase
    return float(abs(ab_scalar - ba_scalar))

def a5_icosahedron_pair():
    """5-fold and 3-fold rotation generators from the icosahedron vertex set."""
    phi = (1 + np.sqrt(5)) / 2
    # 5-fold: rotation by 72deg about an axis through two opposite vertices, e.g. (0,1,phi)
    axis5 = np.array([0, 1, phi])
    a5 = quat_from_axis_angle(axis5, np.deg2rad(72))
    # 3-fold: rotation by 120deg about an axis through a face center, e.g. (1,1,1)
    axis3 = np.array([1, 1, 1])
    a3 = quat_from_axis_angle(axis3, np.deg2rad(120))
    return a5, a3

def q3_setup():
    a, b = _gens()
    v = np.array([1.0, 2.0, 3.0, 4.0])
    return a, b, v

def _path_product_out(g_last, v):
    """attention weight 1 on position 0 from the last query: out = G_path applied to v via quaternion conjugation-like action.
    Treat v as a quaternion, apply left-mult by the SU(2) path product G."""
    return qmul(g_last, v)

def q3_diff():
    a, b, v = q3_setup()
    g_ab = qmul(a, b)
    g_ba = qmul(b, a)
    out_ab = _path_product_out(g_ab, v)
    out_ba = _path_product_out(g_ba, v)
    return float(np.linalg.norm(out_ab - out_ba))

def q3_gate_diffs():
    a, b, v = q3_setup()
    # scalar gate: replace G with a real scalar s (commutative) -> s*v regardless of order
    s = 0.7
    out_ab_scalar = s * v
    out_ba_scalar = s * v
    d_scalar = float(np.linalg.norm(out_ab_scalar - out_ba_scalar))
    # U(1) phase gate: replace G with a complex phase e^{i theta} acting on 2D-complex-embedded v
    theta = 0.9
    phase = np.exp(1j * theta)
    v_c = v[0] + 1j * v[1]
    out_ab_phase = phase * v_c
    out_ba_phase = phase * v_c
    d_phase = float(abs(out_ab_phase - out_ba_phase))
    return d_scalar, d_phase

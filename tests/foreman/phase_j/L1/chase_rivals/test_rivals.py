"""Chase rival dissection: one bound check per claim, float64, CPU, numpy only.
Usage: python test_rivals.py rivals_stub   (RED run)   |   python test_rivals.py rivals_impl
Every mechanism is reimplemented from its published equation (sources in
chase_rivals/SOURCES.txt); no third-party code is imported or executed.
"""
import importlib
import json
import sys

import numpy as np

M = importlib.import_module(sys.argv[1] if len(sys.argv) > 1 else "rivals_impl")
rng = np.random.default_rng(20260922)
EQ = 1e-12
rows = []


def row(name, fn):
    try:
        ok, detail = fn()
    except Exception as e:  # noqa: BLE001 -- stub raises NotImplementedError
        ok, detail = False, f"{type(e).__name__}: {e}"
    rows.append(dict(test=name, ok=bool(ok), detail=detail))


B, S, H, Dh, d = 2, 7, 4, 5, 12


# T1 -- Qwen G1 headwise gate (arXiv 2505.06708, Y' = Y * sigmoid(X W)) vs our a2
def t1a():
    Y = rng.standard_normal((B, H, S, Dh)); X = rng.standard_normal((B, S, d))
    W = rng.standard_normal((d, H)); b = rng.standard_normal(H)
    ours = M.a2_gate(Y, X, W, b)
    theirs = M.qwen_g1_headwise(Y, np.concatenate([X, np.ones((B, S, 1))], -1), np.vstack([W, b]))
    e = float(np.abs(ours - theirs).max()); return e < EQ, f"max|a2 - qwenG1_headwise| = {e:.3e}"


def t1b():  # Qwen3-Next ships ELEMENTWISE (gate = half of q_proj, head_dim per head)
    Y = rng.standard_normal((B, H, S, Dh)); X = rng.standard_normal((B, S, d))
    Wel = rng.standard_normal((d, H, Dh))
    el = M.qwen_g1_elementwise(Y, X, Wel)
    Wh = Wel[:, :, 0]
    hw = M.a2_gate(Y, X, Wh, np.zeros(H))
    generic = float(np.abs(el - hw).max())
    tied = M.qwen_g1_elementwise(Y, X, np.repeat(Wh[:, :, None], Dh, axis=2))
    e_tied = float(np.abs(tied - hw).max())
    return generic > 1e-3 and e_tied < EQ, f"generic W: diff {generic:.3e} (a2 != shipped); tied-per-head W: diff {e_tied:.3e} (a2 = subset)"


# T2 -- Gated DeltaNet decay (Qwen3-Next: g = -exp(A_log)*softplus(a+dt_bias)) vs FoX log f = log sigmoid(z)
def t2a():
    a = rng.standard_normal((B, S, H)); dt = rng.standard_normal(H)
    g = M.gdn_logdecay(a, np.zeros(H), dt)            # A_log = 0
    f = M.fox_logf(-(a + dt))                         # FoX with w -> -W, b -> -dt_bias
    e = float(np.abs(g - f).max())
    g2 = M.gdn_logdecay(a, np.full(H, 0.7), dt)       # A_log != 0: per-head power c = e^0.7 on f
    e2 = float(np.abs(g2 - np.exp(0.7) * f).max())
    return e < EQ and e2 < EQ, f"A_log=0: |g - logf_FoX| = {e:.3e}; A_log=0.7: |g - e^0.7 logf| = {e2:.3e}"


def t2b():  # Qwen3-Next recurrent loop, orthonormal keys, beta=1: equals FoX-masked LINEAR attention
    T, dk, dv = 6, 8, 3
    k = np.linalg.qr(rng.standard_normal((dk, dk)))[0][:, :T].T   # T orthonormal keys
    q = rng.standard_normal((T, dk)); v = rng.standard_normal((T, dv))
    logf = M.fox_logf(rng.standard_normal(T))
    o_gdn = M.gdn_recurrent(q, k, v, logf, np.ones(T))
    o_lin = M.linear_attn_masked(q, k, v, M.decay_mask_from_logf(logf))
    e = float(np.abs(o_gdn - o_lin).max())
    o_fox = M.softmax_attn_masked(q, k, v, M.decay_mask_from_logf(logf))
    gap = float(np.abs(o_gdn - o_fox).max())
    return e < EQ and gap > 1e-3, f"|GDN - FoXmask linear| = {e:.3e}; |GDN - FoX softmax| = {gap:.3e} (not our arm)"


# T3 -- state transitions: KDA (I - b k k^T) Diag(alpha), GDN alpha (I - b k k^T); order and group reach
def t3a():
    n = 4
    k1 = np.eye(n)[0]; k2 = (np.eye(n)[0] + np.eye(n)[1]) / np.sqrt(2)
    al1 = np.array([.9, .5, .8, .7]); al2 = np.array([.6, .95, .4, .85])
    A1 = M.kda_transition(al1, 0.5, k1); A2 = M.kda_transition(al2, 0.5, k2)
    c_kda = float(np.abs(A1 @ A2 - A2 @ A1).max())
    G1 = M.gdn_transition(0.9, 0.5, k1); G2 = M.gdn_transition(0.6, 0.5, k2)
    c_gdn = float(np.abs(G1 @ G2 - G2 @ G1).max())
    D1 = np.diag(al1); D2 = np.diag(al2)
    c_diag = float(np.abs(D1 @ D2 - D2 @ D1).max())
    return c_kda > 1e-3 and c_gdn > 1e-3 and c_diag < EQ, \
        f"||[A1,A2]|| KDA {c_kda:.3e}, GDN {c_gdn:.3e}, diagonal-decay (Mamba2/lightning/FoX) {c_diag:.3e}"


def t3b():
    n = 4; worst = 0.0; maxdet = 0.0
    for _ in range(200):
        al = rng.uniform(0.01, 0.999, n); bt = rng.uniform(0.001, 0.999)
        kk = rng.standard_normal(n); kk /= np.linalg.norm(kk)
        A = M.kda_transition(al, bt, kk)
        worst = max(worst, abs(np.linalg.det(A) - (1 - bt) * al.prod()))
        maxdet = max(maxdet, abs(np.linalg.det(A)))
    # beta in (0,2) extension (not shipped, see SOURCES): beta=2 reflections compose to a rotation, det +1
    e1, e2 = np.eye(3)[0], np.array([np.cos(np.pi / 3), np.sin(np.pi / 3), 0.])
    R = M.householder_general(2.0, e2) @ M.householder_general(2.0, e1)
    rot_ok = abs(np.linalg.det(R) - 1) < EQ and np.abs(R @ R.T - np.eye(3)).max() < EQ and abs(np.trace(R) - (1 + 2 * np.cos(2 * np.pi / 3))) < 1e-9
    U = M.su2_from_quat(rng.standard_normal(4))
    su2_ok = abs(abs(np.linalg.det(U)) - 1) < EQ and np.abs(U @ U.conj().T - np.eye(2)).max() < EQ
    return worst < EQ and maxdet < 1 and rot_ok and su2_ok, \
        f"det(KDA)=(1-b)prod(alpha) err {worst:.3e}; max|det| over 200 draws {maxdet:.4f} < 1 => no product equals any rotation; beta=2 pair -> 120deg rotation {rot_ok}; our SU(2) unitary {su2_ok}"


# T3c -- per-token spectrum: shipped KDA/GDN (L2-normed k, beta, alpha in [0,1]) has real eigenvalues in [0,1],
# so no single-token factor has a unit-modulus eigenvalue != 1 (no finite-order element: no -1, no e^{2pi i/3});
# our SU(2) token gate for a 3-cycle has e^{+-i pi/3} (order 3 in SO(3)); unnormalized k breaks the bound.
def t3c():
    n = 5; bad = 0; worst_im = 0.0
    for i in range(1000):
        al = rng.uniform(0, 1, n); bt = rng.uniform(0, 1)
        if i % 10 == 0: al[:2] = 1.0; bt = 1.0 if i % 20 == 0 else 0.0   # bf16-saturated endpoints
        kk = rng.standard_normal(n); kk /= np.linalg.norm(kk)
        for A in (M.kda_transition(al, bt, kk), M.gdn_transition(al[0], bt, kk)):
            ev = np.linalg.eigvals(A)
            worst_im = max(worst_im, float(np.abs(ev.imag).max()))
            if ev.real.min() < -1e-12 or ev.real.max() > 1 + 1e-12: bad += 1
    th = 2 * np.pi / 3
    U = M.su2_from_quat(np.array([np.cos(th / 2), np.sin(th / 2), 0, 0]))
    su2_ev = np.sort(np.angle(np.linalg.eigvals(U)))
    su2_ok = np.allclose(su2_ev, [-th / 2, th / 2], atol=1e-12) and np.allclose(np.linalg.matrix_power(U, 3), -np.eye(2), atol=1e-12)
    kbig = np.eye(n)[0] * 1.3
    neg = float(np.linalg.eigvals(np.eye(n) - 0.9 * np.outer(kbig, kbig)).real.min())
    return bad == 0 and worst_im < 1e-9 and su2_ok and neg < 0, \
        f"2000 shipped-range factors: out-of-[0,1] {bad}, max|Im ev| {worst_im:.2e}; SU(2) 3-cycle ev angles {su2_ev.round(6).tolist()}, U^3=-I {su2_ok}; unnormalized |k|=1.3,b=.9 -> min ev {neg:.3f}"


# T4 -- MiniMax lightning: fixed per-head lambda decay == ALiBi slope m=-ln(lambda) == FoX with constant f
def t4():
    T = 9; lam = 0.93
    Ml = M.lightning_mask(lam, T)
    Ma = M.alibi_mask(-np.log(lam), T)
    Mf = M.decay_mask_from_logf(np.full(T, np.log(lam)))
    e1 = float(np.abs(Ml - Ma).max()); e2 = float(np.abs(Ml - Mf).max())
    return e1 < EQ and e2 < EQ, f"|lightning - exp(ALiBi)| {e1:.3e}; |lightning - FoX(f const)| {e2:.3e}"


# T5 -- softmax over a key set (MLA / DSA / MoBA / NSA branch, positions removed) is order-blind
def t5():
    T, dk, dv = 8, 6, 3
    q = rng.standard_normal((1, dk)); k = rng.standard_normal((T, dk)); v = rng.standard_normal((T, dv))
    ones = np.ones((1, T))
    o = M.softmax_attn_masked(q, k, v, ones)
    p = rng.permutation(T)
    op = M.softmax_attn_masked(q, k[p], v[p], ones)
    e = float(np.abs(o - op).max())
    return e < EQ, f"|out - out(permuted keys)| = {e:.3e}: order enters only through RoPE/mask, no state transition"


# T6 -- NSA o = sum_c g_c Attn_c, g = sigmoid(MLP(x)): C=1 is our a2
def t6():
    Y = rng.standard_normal((B, H, S, Dh)); X = rng.standard_normal((B, S, d))
    W = rng.standard_normal((d, H)); b = rng.standard_normal(H)
    g = 1 / (1 + np.exp(-(X @ W + b)))                        # [B,S,H]
    nsa = M.nsa_combine([g.transpose(0, 2, 1)[..., None]], [Y])
    e = float(np.abs(nsa - M.a2_gate(Y, X, W, b)).max())
    return e < EQ, f"|NSA(C=1) - a2| = {e:.3e}"


for n, f in [("T1a_a2_equals_qwen_G1_headwise", t1a), ("T1b_a2_strict_subset_of_qwen3next_elementwise", t1b),
             ("T2a_gdn_decay_is_fox_gate_up_to_head_power", t2a), ("T2b_gdn_recurrence_is_fox_mask_linear_not_softmax", t2b),
             ("T3a_kda_gdn_noncommuting_diag_commuting", t3a), ("T3b_kda_det_below_one_no_rotation", t3b), ("T3c_kda_gdn_spectrum_in_0_1_su2_on_circle", t3c),
             ("T4_lightning_is_alibi_is_constant_fox", t4), ("T5_softmax_set_order_blind", t5),
             ("T6_nsa_single_branch_is_a2", t6)]:
    row(n, f)
for r in rows:
    print(("GREEN " if r["ok"] else "RED   ") + r["test"] + " :: " + r["detail"])
print(json.dumps(dict(module=M.__name__, passed=sum(r["ok"] for r in rows), total=len(rows))))

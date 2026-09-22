"""Chase it.L1 GREEN tests: independent reproductions of the rows that hold,
and the replacement measurement for each RED in test_chase_L1_red.py."""
import json, math, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
SPJ = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import numpy as np
import indep
from indep import M, Q, dag
from test_chase_L1_red import row, _spj_import


# ---- Q1, independent 2x2-complex arithmetic and own scan
def test_q1_independent():
    r = indep.q1()
    assert r["scan_vs_seq_f64"] <= 1e-12 and r["Gij_direct_vs_fold_f64"] <= 1e-12, r
    assert r["fp32_renorm256_vs_f64"] <= 1e-4, r


# ---- Q2 settled without BFS: 2cos(theta_ab) is not an algebraic integer, so ab has infinite order
def test_q2_stated_group_is_infinite():
    r = indep.q2_stated()
    assert r["2cos_theta_ab_is_algebraic_integer"] is False, r
    assert all(abs(r["axis_angle_deg"] - a) > 1 for a in r["icosahedral_5fold_3fold_axis_angles_deg"]), r


# ---- Q2-REROUTE, independent closure by hashing + A5 class equation + word
def test_q2_reroute_independent():
    r = indep.q2_reroute()
    assert (r["closure_so3"], r["closure_su2"]) == (60, 120), r
    assert r["order_histogram"] == {1: 1, 2: 15, 3: 20, 5: 24}, r
    assert r["class_sizes"] == [1, 12, 12, 15, 20], r
    assert r["word_float_vs_table_maxabs"] < 1e-10, r


# ---- Q3 reroute: one harness, attention pattern held fixed, gate on the value path only
def _fixed_pattern_read(gates):
    qk = np.random.default_rng(7).standard_normal(len(gates))
    p = np.exp(qk - qk.max()); p /= p.sum()
    vals = [M([1.0, 2.0, 3.0, 4.0])] + [np.zeros((2, 2), complex)] * (len(gates) - 1)
    Pi = [gates[0]]
    for g in gates[1:]:
        Pi.append(g @ Pi[-1])
    return sum(p[j] * (Pi[-1] @ dag(Pi[j]) @ vals[j]) for j in range(len(gates)))

def test_q3_fixed_pattern_read():
    I = np.eye(2, dtype=complex)
    a, b = indep.su2((0, 0, 1), 72), indep.su2((1, 1, 1), 120)
    sa, sb = 0.8 * I, 0.3 * I  # scalar gates through the SAME harness
    d_su2 = np.abs(_fixed_pattern_read([I, a, b, I]) - _fixed_pattern_read([I, b, a, I])).max()
    d_sc = np.abs(_fixed_pattern_read([I, sa, sb, I]) - _fixed_pattern_read([I, sb, sa, I])).max()
    assert d_su2 > 1e-3 and d_sc == 0.0, (d_su2, d_sc)


# ---- PF: the token-lattice law replaces c_W and 1/3 on tokens
def test_pf_token_lattice_law():
    rec = row("PF-GAMMA")["measured"]["unit_lattice_n400"]
    for e in (1, 0.3, 0.1):
        assert abs(rec[f"eps={e}"]["P_min"] - indep.pf_lattice_dp(e)) < 1e-4
    e = 0.03
    assert abs(indep.pf_lattice_dp(e) - (e - 2 * e ** 3)) / e < 1e-3
    assert abs(indep.pf_continuum() - 1 / 3) < 1e-6


# ---- DRIFT: Horn's fit as it stands meets 1e-9 once the angle readout is stable
def test_drift_stable_readout_meets_1e9():
    hn = _spj_import("horn"); rh = _spj_import("run_horn")
    rng = np.random.default_rng(0)
    P = rng.standard_normal((200, 3)); P -= P.mean(axis=0)
    Rt = rh.random_rotation(rng)
    Re = hn.horn_align(P, P @ Rt.T)
    assert indep.angle_of(Re.T @ Rt) <= 1e-9


# ---- A5 sentence restated: no arm distinguishable from chance at n_eval 64 x 3 seeds
def test_a5_restated_indistinguishable_from_chance():
    from scipy.stats import binom
    m = row("L1-A5")["measured"]
    for k in ("a_softmax_twin", "a2_fox_twin", "fq_quat_twin"):
        hits = round(sum(m[k]["per_seed_acc_pos64"]) * 64)
        assert binom.sf(hits - 1, 192, 1 / 60) > 0.05, (k, hits)


# ---- FQ-ARM beyond S 8 / float64: fp32 at the bed shape and at S 512, own 2x2-complex direct-sum oracle
def _fq_oracle(attn, qh, x64, H):
    S, d = x64.shape; dh = d // H
    W = attn.qkv.weight.detach().double().numpy(); b = attn.qkv.bias.detach().double().numpy()
    q, k, v = np.split(x64 @ W.T + b, 3, axis=-1)
    quat = x64 @ qh.weight.detach().double().numpy().T + qh.bias.detach().double().numpy()
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    A = M(quat); Pi = [A[0]]
    for i in range(1, S):
        Pi.append(A[i] @ Pi[-1])
    Pi = np.stack(Pi); G = Pi[:, None] @ dag(Pi)[None, :]
    out = np.zeros((S, d))
    for h in range(H):
        sl = slice(h * dh, (h + 1) * dh)
        s = q[:, sl] @ k[:, sl].T / math.sqrt(dh); s[np.triu_indices(S, 1)] = -np.inf
        p = np.exp(s - s.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        for blk in range(dh // 4):
            c = slice(h * dh + 4 * blk, h * dh + 4 * blk + 4)
            GV = G @ M(v[:, c])[None, :]
            out[:, c] = Q(np.einsum("ij,ijab->iab", p, GV))
    Wo = attn.o_proj.weight.detach().double().numpy(); bo = attn.o_proj.bias.detach().double().numpy()
    return out @ Wo.T + bo

def test_fq_arm_independent_oracle():
    import torch
    af = _spj_import("arm_fq")
    errs = {}
    for S, dt, d, H in [(64, torch.float64, 64, 4), (64, torch.float32, 64, 4), (512, torch.float32, 64, 4)]:
        torch.manual_seed(0)
        attn = af._StubAttn(d, H).to(dt); attn.quat_head = torch.nn.Linear(d, 4).to(dt)
        x = torch.randn(1, S, d, dtype=dt)
        got = af.fq_forward(attn, x).detach().double().numpy()[0]
        want = _fq_oracle(attn, attn.quat_head, x.double().numpy()[0], H)
        errs[(S, str(dt))] = float(np.abs(got - want).max())
    print("FQ errs", errs)
    assert errs[(64, "torch.float64")] <= 1e-12 and errs[(64, "torch.float32")] <= 1e-4 and errs[(512, "torch.float32")] <= 1e-4, errs


def test_fold_independent_oracle():
    import torch
    fo = _spj_import("fold")
    errs = {}
    for dt in (torch.float64, torch.float32):
        q, k, v, Pi = fo.make_inputs(1, 2, 256, 16, seed=3, dtype=dt)
        got = fo.fold_out(q, k, v, Pi).double().numpy()[0]
        qn, kn, vn, Pn = (t.double().numpy()[0] for t in (q, k, v, Pi))
        err = 0.0
        for h in range(2):
            s = qn[h] @ kn[h].T / 4.0; s[np.triu_indices(256, 1)] = -np.inf
            p = np.exp(s - s.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            for blk in range(4):
                Pm = M(Pn[h, :, blk]); G = Pm[:, None] @ dag(Pm)[None, :]
                o = Q(np.einsum("ij,ijab->iab", p, G @ M(vn[h, :, 4 * blk:4 * blk + 4])[None, :]))
                err = max(err, float(np.abs(o - got[h, :, 4 * blk:4 * blk + 4]).max()))
        errs[str(dt)] = err
    print("FOLD errs", errs)
    assert errs["torch.float64"] <= 1e-12 and errs["torch.float32"] <= 1e-4, errs

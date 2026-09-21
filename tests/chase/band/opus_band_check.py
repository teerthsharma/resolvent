r"""OPUS CHECKER of Chase's band-position row. Independent recompute, CPU only.

FIXED STRUCTURE (L-REFLECTOR -- printed before any scored quantity):
  ckpt A  ceqjepa/artifacts/curriculum/chess.pt   bed=chess step=550 n=16 nA=4 g=0.9
  ckpt B  ceqjepa/space/chess_checkpoint.pt       bed=chess step=400 n=32 nA=4 g=0.9
  P       = op.build_operator(L0 + delta_logits(enc(x)), absorbing_idx) at the SHIPPED
            teleport 0.0125, taken from TinyCEQ.forward() itself (not re-derived).
  Q, R    = op.committor's own split: Q = P[T,T], R = P[T,A].
  alpha   = softmax(chart(e)) masked to TRANSIENT rows -- THE SUPERVISED READ's weights.
  read    = q_alpha = alpha_T^T (I-Q)^{-1} R      (TinyCEQ.forward line 537)
  hop k   = term alpha_T^T Q^k R of that read's Neumann series. k=0 term is alpha_T^T R.
  gamma   = registered buffer g = 0.9. NOTE: gamma enters state_solve only. The
            committor solve's coefficient is 1 (operator.py:219 docstring), so the
            PREDICTION read's hop series is sum_k Q^k R with NO gamma in it.
  bed     = ceqjepa.beds.chess.ChessBed.build(n_games, seed=0, max_plies=M).
            M=80 is the shipped default; M=400 is the REPAIRED bed the 0.001469 /
            piece-count-histogram tie was measured on (docs/PHASE_I.md).
  pc      = total piece count = x[:, :768].sum(-1)  -- exactly the scalar the 8-bin
            histogram arm bins (docs/PHASE_I.md: "computed from the operator's own
            769-float input").
  support graph = directed, edge j->i (j<=i) iff M[i,j] >= thresh. diameter = max
            shortest-path hops over reachable ordered pairs, per example.
"""
import io
import json
import math
import os
import sys

import numpy as np
import torch

REPO = r"C:\Users\seal\Desktop\New folder (32)"
SP = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
sys.path.insert(0, REPO)

import ceqjepa.operator as op  # noqa: E402
from ceqjepa.train import TinyCEQ  # noqa: E402
from ceqjepa.beds.chess import ChessBed  # noqa: E402

CKPTS = [
    ("A_curriculum_step550", os.path.join(REPO, "ceqjepa", "artifacts", "curriculum", "chess.pt")),
    ("B_space_step400", os.path.join(REPO, "ceqjepa", "space", "chess_checkpoint.pt")),
]
B = 256
SEED = 0


def load(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    geo = ck["geometry"]
    m = TinyCEQ(n=geo["n"], nA=geo["nA"], d_enc=geo["d_enc"], x_dim=geo["x_dim"],
                z_dim_state=geo["z_dim_state"], g=geo["g"],
                absorbing_idx=ck["model_state_dict"]["absorbing_idx"], rank=geo["rank"])
    m.load_state_dict(ck["model_state_dict"], strict=False)
    m.eval()
    return m, geo, ck, float(ck["model_state_dict"]["g"])


def diameter(Mx, thresh):
    n = Mx.shape[0]
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            if float(Mx[i, j]) >= thresh:
                adj[j].append(i)
    diam = 0
    for src in range(n):
        dist = {src: 0}
        fr = [src]
        while fr:
            nx = []
            for u in fr:
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        nx.append(v)
            fr = nx
        for t, d in dist.items():
            if t != src:
                diam = max(diam, d)
    return diam


def eta2_by_level(y, levels):
    """Fraction of Var(y) explained by the DISCRETE level variable.
    eta2 == 1.0 exactly <=> y is a deterministic function of the level.
    y: [N] float. levels: [N] int."""
    y = np.asarray(y, dtype=np.float64)
    tot = y.var()
    if tot <= 0:
        return float("nan"), 0.0
    within = 0.0
    for lv in np.unique(levels):
        sel = y[levels == lv]
        within += sel.var() * sel.size
    within /= y.size
    return 1.0 - within / tot, math.sqrt(within)


def run(tag, ckpt_path, max_plies, n_games):
    m, geo, ck, gamma = load(ckpt_path)
    n, nA = geo["n"], geo["nA"]
    A_idx = m.absorbing_idx.tolist()
    T_idx = [i for i in range(n) if i not in A_idx]
    print("\n" + "=" * 78)
    print("[FIXED STRUCTURE] %s  ckpt=%s" % (tag, os.path.relpath(ckpt_path, REPO)))
    print("  bed=%s phase=%s step=%s | n=%d nA=%d nT=%d rank=%d | gamma(g)=%.6f"
          % (geo.get("bed"), geo.get("phase"), ck.get("step"), n, nA, len(T_idx),
             geo["rank"], gamma))
    print("  absorbing_idx=%r  transient_idx=%r" % (A_idx, T_idx))
    print("  bed = ChessBed.build(n_games=%d, seed=%d, max_plies=%d)  B=%d examples"
          % (n_games, SEED, max_plies, B))
    print("  teleport = op.TELEPORT = %r (shipped; TinyCEQ.forward never overrides it)"
          % (op.TELEPORT,))

    bed = ChessBed.build(n_games=n_games, seed=SEED, max_plies=max_plies)
    gen = torch.Generator().manual_seed(SEED + 1)
    x, x_nx, q_star, v_idx = bed.batch(gen, B)
    with torch.no_grad():
        out = m(x)
    P = out["P"].double()
    alpha = out["alpha"].double()
    q_alpha = out["q_alpha"].double()

    ti = torch.tensor(T_idx, dtype=torch.long)
    ai = torch.tensor(A_idx, dtype=torch.long)
    Q = P.index_select(1, ti).index_select(2, ti)          # [B,nT,nT]
    R = P.index_select(1, ti).index_select(2, ai)          # [B,nT,nA]
    aT = alpha.index_select(1, ti)                         # [B,nT]

    # ---- (A) spectral radii / xi -------------------------------------------
    rho_P = torch.linalg.eigvals(P).abs().max(dim=-1).values.numpy()
    rho_Q = torch.linalg.eigvals(Q).abs().max(dim=-1).values.numpy()
    def xi(v):
        v = float(v)
        if v >= 1.0:
            return float("inf")
        if v <= 0:
            return 0.0
        return -1.0 / math.log(v)
    xi_P = xi(gamma * rho_P.mean())
    xi_P_worst = xi(gamma * rho_P.max())
    xi_Qg = xi(gamma * rho_Q.max())        # Chase's steelman (gamma applied)
    xi_Q = xi(rho_Q.max())                 # CORRECT for the committor: no gamma
    print("\n[A] rho(P): mean=%.9f min=%.9f max=%.9f  (row-stochastic => 1 by Perron-Frobenius)"
          % (rho_P.mean(), rho_P.min(), rho_P.max()))
    print("[A] gamma*rho(P) = %.9f   xi(P) = -1/ln(gamma*rho(P)) = %.4f hops (worst %.4f)"
          % (gamma * rho_P.mean(), xi_P, xi_P_worst))
    print("[A] rho(Q): mean=%.6f min=%.6f max=%.6f" % (rho_Q.mean(), rho_Q.min(), rho_Q.max()))
    print("[A] xi(Q) NO gamma (correct for the committor solve, coeff=1): worst-case %.4f" % xi_Q)
    print("[A] xi(gamma*Q) (Chase's form)                               : worst-case %.4f" % xi_Qg)
    # why is rho(Q) small? report the mass each transient row sends to A in ONE hop.
    row_to_A = R.sum(-1)                                   # [B,nT]
    print("[A] one-hop absorption mass per transient row R.sum: mean=%.4f min=%.4f max=%.4f"
          % (float(row_to_A.mean()), float(row_to_A.min()), float(row_to_A.max())))

    diam = {}
    for name, th_P, th_Q in [("nonzero", 0.0, 0.0),
                             ("1/n^2", 1.0 / (n * n), 1.0 / (len(T_idx) ** 2)),
                             ("1/n", 1.0 / n, 1.0 / len(T_idx)),
                             ("2/n", 2.0 / n, 2.0 / len(T_idx))]:
        dP = [diameter(P[b], th_P) for b in range(min(B, 64))]
        dQ = [diameter(Q[b], th_Q) for b in range(min(B, 64))]
        diam[name] = dict(P_mean=float(np.mean(dP)), P_max=int(np.max(dP)),
                          Q_mean=float(np.mean(dQ)), Q_max=int(np.max(dQ)))
        print("[A] diameter thresh=%-7s  P: mean=%.2f max=%d   Q: mean=%.2f max=%d"
              % (name, np.mean(dP), np.max(dP), np.mean(dQ), np.max(dQ)))
    # the graph-theoretic CEILING on any diameter here: the causal DAG on n nodes
    print("[A] hard ceiling on diameter of ANY causal DAG on these nodes: n-1 = %d (P), %d (Q)"
          % (n - 1, len(T_idx) - 1))

    # ---- (B) the k=0 identity ----------------------------------------------
    # exact read, and its Neumann partial sums.
    eyeT = torch.eye(len(T_idx), dtype=torch.float64)
    q_T = torch.linalg.solve_triangular(eyeT - Q, R, upper=False)     # [B,nT,nA]
    read_exact = torch.einsum("bt,btk->bk", aT, q_T)
    print("\n[B] read reconstruction check: max|alpha^T (I-Q)^-1 R  -  forward q_alpha| = %.3e"
          % float((read_exact - q_alpha).abs().max()))

    terms = []
    cur = R.clone()
    for k in range(0, 8):
        terms.append(torch.einsum("bt,btk->bk", aT, cur))             # k-th hop term
        cur = torch.bmm(Q, cur)
    partial = torch.zeros_like(read_exact)
    print("[B] hop expansion of THE SUPERVISED READ, %d examples:" % B)
    for k, t in enumerate(terms):
        partial = partial + t
        rel = float((read_exact - partial).abs().sum(-1).mean() / read_exact.abs().sum(-1).mean())
        share = float(t.abs().sum(-1).mean() / read_exact.abs().sum(-1).mean())
        print("    k=%d  |term_k|/|read| = %.6f   relative truncation error after k = %.3e"
              % (k, share, rel))
    t0 = terms[0].numpy()                                             # [B,nA]  THE k=0 TERM

    pc = x[:, :768].sum(-1).numpy()                                   # total piece count
    pc_i = np.rint(pc).astype(int)
    print("\n[B] piece count pc = x[:, :768].sum(): min=%d max=%d distinct levels=%d"
          % (pc_i.min(), pc_i.max(), len(np.unique(pc_i))))

    # 8-bin histogram arm, exactly as described: equal-width bins over pc.
    edges = np.linspace(pc_i.min(), pc_i.max() + 1e-9, 9)
    hbin = np.clip(np.digitize(pc, edges[1:-1]), 0, 7)

    print("[B] IS THE k=0 TERM A FUNCTION OF THE MATERIAL COUNT?")
    print("    eta^2 = 1.000000 exactly <=> deterministic function of that variable.")
    ident = {}
    for name, lv in [("exact piece count (%d levels)" % len(np.unique(pc_i)), pc_i),
                     ("8-bin piece count", hbin)]:
        for c in range(t0.shape[1]):
            e2, wsd = eta2_by_level(t0[:, c], lv)
            print("    k=0 term, outcome %d   vs %-32s  eta^2=%.6f  within-level sd=%.3e"
                  % (c, name, e2, wsd))
            ident["k0_c%d_%s" % (c, name.split(" ")[0])] = float(e2)
    print("    -- and the full read, for contrast:")
    for c in range(q_alpha.shape[1]):
        e2, wsd = eta2_by_level(q_alpha[:, c].numpy(), pc_i)
        print("    full read, outcome %d vs exact piece count           eta^2=%.6f  within-level sd=%.3e"
              % (c, e2, wsd))
        ident["read_c%d_exactpc" % c] = float(e2)

    # correlation, for the weaker "merely correlated" reading
    corrs = []
    for c in range(t0.shape[1]):
        if t0[:, c].std() > 0:
            corrs.append(float(np.corrcoef(t0[:, c], pc)[0, 1]))
        else:
            corrs.append(float("nan"))
    print("[B] Pearson r(k=0 term, piece count) per outcome: %s"
          % ["%.4f" % c for c in corrs])

    return dict(tag=tag, ckpt=os.path.relpath(ckpt_path, REPO), bed_max_plies=max_plies,
                n=n, nA=nA, nT=len(T_idx), gamma=gamma, B=B,
                rho_P_mean=float(rho_P.mean()), rho_P_max=float(rho_P.max()),
                rho_Q_mean=float(rho_Q.mean()), rho_Q_max=float(rho_Q.max()),
                xi_P=xi_P, xi_P_worst=xi_P_worst, xi_Q_nogamma_worst=xi_Q,
                xi_Q_withgamma_worst=xi_Qg,
                one_hop_absorb_mean=float(row_to_A.mean()),
                diameters=diam,
                hop_share=[float(t.abs().sum(-1).mean() / read_exact.abs().sum(-1).mean())
                           for t in terms],
                identity=ident, corr_k0_pc=corrs,
                n_pc_levels=int(len(np.unique(pc_i))))


if __name__ == "__main__":
    out = []
    for tag, path in CKPTS:
        out.append(run(tag, path, max_plies=80, n_games=40))
        out.append(run(tag + "_repaired400", path, max_plies=400, n_games=40))
    with io.open(os.path.join(SP, "opus_band_check.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\n[WROTE] opus_band_check.json")

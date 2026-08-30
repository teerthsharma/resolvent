"""Numerical check of the V13 Tier-1 and Tier-2 derivations.

    python scripts/v13_derivation_check.py

Companion to `V13_TIER12_DERIVATIONS.md`. Every assertion here carries an
equation number from that file. Standard library plus numpy, fixed seeds, no
test framework, no fixtures. A failure is a finding, not a tolerance to widen.

TIER 1 -- the discrete Kramers law on a planted multi-basin chain (T1.*).
TIER 2 -- the null distribution of the fan-out detector (T2.*).
"""

import math
import sys
from statistics import NormalDist

import numpy as np

# --------------------------------------------------------------------------
# stdlib-only special functions.  numpy has no chi-square quantile and scipy
# is deliberately not imported, so the regularised lower incomplete gamma is
# built from `math.lgamma` and inverted by bisection.
# --------------------------------------------------------------------------


def gammainc_lower_reg(a, x):
    """P(a, x) by the everywhere-convergent all-positive series.

    P(a,x) = sum_{k>=0} x^(a+k) e^-x / Gamma(a+k+1).  Every term is positive so
    there is no cancellation; terms are formed in logs so there is no overflow.
    Convergence sets in once k > x.
    """
    if x <= 0.0:
        return 0.0
    total = 0.0
    log_x = math.log(x)
    for k in range(4000):
        term = math.exp(a * log_x - x + k * log_x - math.lgamma(a + k + 1.0))
        total += term
        if k > x and term < 1e-18 * max(total, 1e-300):
            break
    return min(total, 1.0)


def chi2_cdf(x, nu):
    return gammainc_lower_reg(nu / 2.0, x / 2.0)


def chi2_ppf(p, nu):
    lo, hi = 0.0, 1.0
    while chi2_cdf(hi, nu) < p:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if chi2_cdf(mid, nu) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def z_ppf(p):
    return NormalDist().inv_cdf(p)


# --------------------------------------------------------------------------
# TIER 1 -- planted landscapes and their reversible chains
# --------------------------------------------------------------------------


def metropolis_chain(energy, edges, temperature):
    """Metropolis-Hastings on a symmetric uniform proposal over `edges`.

    (T1.1)  G_ij = 1/d_max on an edge, G_ii = 1 - deg(i)/d_max.  Symmetric.
    (T1.2)  P_ij = G_ij min(1, exp(-(E_j - E_i)/T)),  P_ii = 1 - sum_{j!=i}.
    Detailed balance (T1.3) is exact because
        pi_i P_ij = G_ij min(e^{-E_i/T}, e^{-E_j/T}) / Z
    is symmetric in (i, j).
    """
    energy = np.asarray(energy, dtype=float)
    n = energy.size
    degree = np.zeros(n, dtype=int)
    for i, j in edges:
        degree[i] += 1
        degree[j] += 1
    d_max = int(degree.max())
    proposal = np.zeros((n, n))
    for i, j in edges:
        proposal[i, j] = proposal[j, i] = 1.0 / d_max
    weight = np.exp(-energy / temperature)
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and proposal[i, j] > 0.0:
                P[i, j] = proposal[i, j] * min(1.0, weight[j] / weight[i])
        P[i, i] = 1.0 - P[i].sum()
    return P, weight / weight.sum()


def double_well(temperature):
    """Nine-state 1-D double well.  Basin A = {0,1,2}, barrier {3,4,5},
    basin B = {6,7,8}, index-1 saddle at state 4."""
    energy = [0.00, 0.40, 1.10, 2.20, 3.00, 2.20, 1.10, 0.40, 0.10]
    edges = [(i, i + 1) for i in range(8)]
    P, pi = metropolis_chain(energy, edges, temperature)
    return P, pi, np.asarray(energy), {"A": [0, 1, 2], "B": [6, 7, 8]}


def three_basin(temperature):
    """Eleven states, basin A with two exit channels.

    channel to B : ONE saddle at E = 2.00, barrier 2.00
    channel to C : FOUR parallel saddles at E = 2.20, barrier 2.20

    Barrier ordering says B is the easier exit.  The capacity ordering says C
    is, whenever 4 exp(-0.2/T) > 1.  This is the T1.19 counterexample.
    """
    energy = [0.00, 0.35,               # 0,1   basin A (0 is the minimum)
              2.00,                     # 2     saddle to B
              -0.10, 0.30,              # 3,4   basin B
              2.20, 2.20, 2.20, 2.20,   # 5-8   four parallel saddles to C
              -0.05, 0.25]              # 9,10  basin C
    edges = ([(0, 1), (1, 2), (2, 3), (3, 4)]
             + [(1, s) for s in (5, 6, 7, 8)]
             + [(s, 9) for s in (5, 6, 7, 8)]
             + [(9, 10)])
    P, pi = metropolis_chain(energy, edges, temperature)
    return P, pi, np.asarray(energy), {"A": [0, 1], "B": [3, 4], "C": [9, 10]}


def fundamental(P, interior):
    """(T1.5)  N = (I - P_II)^-1 on the interior block."""
    Q = P[np.ix_(interior, interior)]
    return Q, np.linalg.inv(np.eye(len(interior)) - Q)


def occupancy_doubling(Q, doublings=20):
    """(T1.6)  S_N = sum_{t<N} Q^t by repeated doubling, N = 2^doublings.

    S_{2m} = S_m + Q^m S_m.  This is `CEQ.Occupancy.occupancy` evaluated
    without ever solving a linear system -- only matrix products.
    """
    m = Q.shape[0]
    S = np.eye(m)
    Qp = Q.copy()
    for _ in range(doublings):
        S = S + Qp @ S
        if np.abs(Qp).max() < 1e-300:
            break
        Qp = Qp @ Qp
    return S, Qp


def committor_dirichlet(P, boundary_values):
    """(T1.11)  Solve L q = 0 on the interior with L = P - I, q = b on the
    boundary.  Built as a full n x n system with pinned boundary rows, so it
    shares no matrix with the (I - Q)^-1 R route."""
    n = P.shape[0]
    A = np.zeros((n, n))
    b = np.zeros(n)
    for i in range(n):
        if i in boundary_values:
            A[i, i] = 1.0
            b[i] = boundary_values[i]
        else:
            A[i, :] = P[i, :]
            A[i, i] -= 1.0
    return np.linalg.solve(A, b)


def conductances(P, pi):
    """(T1.14)  c_ij = pi_i P_ij, symmetric by detailed balance, and
    L = D - C = diag(pi)(I - P) is the reversible-chain Laplacian."""
    C = np.diag(pi) @ P
    return C, np.diag(C.sum(axis=1)) - C


def kirchhoff_committor(P, pi, source, sink):
    """(T1.15)  Ground at the sink, contract each boundary set to one node,
    read omega_x = M_{x,alpha} / M_{alpha,alpha} from M = (L_sink)^-1.

    This is MATHEMATICS.md section 11 step 4, applied to the weighted Laplacian
    of the reversible chain instead of the unweighted graph Laplacian.
    """
    n = P.shape[0]
    _, L = conductances(P, pi)
    others = [i for i in range(n) if i not in source and i not in sink]
    groups = [[i] for i in others] + [list(source), list(sink)]
    m = len(groups)
    Lc = np.zeros((m, m))
    for a in range(m):
        for b in range(m):
            Lc[a, b] = L[np.ix_(groups[a], groups[b])].sum()
    alpha, beta = m - 2, m - 1
    keep = [i for i in range(m) if i != beta]
    M = np.linalg.inv(Lc[np.ix_(keep, keep)])
    ai = keep.index(alpha)
    omega_contracted = M[:, ai] / M[ai, ai]
    out = np.zeros(n)
    for pos, g in enumerate(keep):
        for node in groups[g]:
            out[node] = omega_contracted[pos]
    for node in sink:
        out[node] = 0.0
    return out


def capacity_and_hitting(P, pi, A, B):
    """(T1.16-T1.17) Equilibrium potential h, capacity by two routes, and the
    exact mean-hitting-time identity  E_nu[tau_B] = (sum_x pi_x h_x)/cap."""
    n = P.shape[0]
    bvals = {i: 1.0 for i in A}
    bvals.update({i: 0.0 for i in B})
    h = committor_dirichlet(P, bvals)
    Ph = P @ h
    escape = np.array([pi[x] * (1.0 - Ph[x]) for x in A])
    cap_escape = float(escape.sum())
    C, _ = conductances(P, pi)
    cap_dirichlet = 0.5 * float(sum(C[x, y] * (h[x] - h[y]) ** 2
                                    for x in range(n) for y in range(n)))
    interior = [i for i in range(n) if i not in B]
    tauB = np.linalg.solve(
        np.eye(len(interior)) - P[np.ix_(interior, interior)],
        np.ones(len(interior)))
    tmap = {node: tauB[k] for k, node in enumerate(interior)}
    nu = escape / cap_escape
    lhs = float(sum(nu[k] * tmap[x] for k, x in enumerate(A)))
    rhs = float(pi @ h) / cap_escape
    return h, cap_escape, cap_dirichlet, lhs, rhs


def channel_split(temperature):
    """(T1.12)  Splitting probabilities out of basin A, weighted by the
    metastable start measure pi restricted to A, plus the total escape rate."""
    P, pi, _, basins = three_basin(temperature)
    n = P.shape[0]
    A, B, C = basins["A"], basins["B"], basins["C"]
    interior = [i for i in range(n) if i not in B + C]
    Q, N = fundamental(P, interior)
    qB = N @ P[np.ix_(interior, B)] @ np.ones(len(B))
    qC = N @ P[np.ix_(interior, C)] @ np.ones(len(C))
    tau = N @ np.ones(len(interior))
    piA = pi[A] / pi[A].sum()
    rows = [interior.index(x) for x in A]
    QB = float(piA @ qB[rows])
    QC = float(piA @ qC[rows])
    k_total = 1.0 / float(piA @ tau[rows])
    cond = float(np.linalg.cond(np.eye(len(interior)) - Q))
    return dict(QB=QB, QC=QC, k_total=k_total, cond=cond,
                kB=QB * k_total, kC=QC * k_total)


# --------------------------------------------------------------------------
# TIER 2 -- the plug-in entropy estimator under a single-basin null
# --------------------------------------------------------------------------


def entropy(p):
    p = np.asarray(p, dtype=float)
    q = p[p > 0.0]
    return float(-(q * np.log(q)).sum())


def entropy_variance(p):
    """(T2.4)  V1 = sum_k p_k (ln p_k + H)^2 = Var_p(-ln p_K).  Zero exactly
    when p is uniform on its support."""
    p = np.asarray(p, dtype=float)
    h = entropy(p)
    q = p[p > 0.0]
    return float((q * (np.log(q) + h) ** 2).sum())


def plugin_entropy(counts, n):
    p = counts / n
    return np.where(p > 0.0, -p * np.log(np.where(p > 0.0, p, 1.0)), 0.0).sum(-1)


def bias_two_term(p, n):
    """(T2.2)  E[Hhat] - H = -(K-1)/(2n) + (1 - sum_k 1/p_k)/(12 n^2) + O(n^-3)."""
    p = np.asarray(p, dtype=float)
    q = p[p > 0.0]
    K = q.size
    return -(K - 1) / (2.0 * n) + (1.0 - float((1.0 / q).sum())) / (12.0 * n * n)


def null_quantile(prob, p, n, rng, draws=400000):
    """(T2.10)  Quantile of W = Z/sqrt(n) - X2/(2n), Z ~ N(0, V1) independent of
    X2 ~ chi2_{K-1}.  The branch-free threshold: it reduces to the normal form
    when n V1 dominates and to the chi-square form at V1 = 0."""
    K = int((np.asarray(p) > 0.0).sum())
    V1 = entropy_variance(p)
    W = (rng.normal(0.0, math.sqrt(V1 / n), draws)
         - rng.chisquare(K - 1, draws) / (2.0 * n))
    return float(np.quantile(W, prob))


def bootstrap_threshold(alpha, p, n, rng, draws=200000):
    """(T2.11)  The recommended implementation: the exact null quantile of
    Hhat - H by a parametric bootstrap from the basin's own p."""
    samples = plugin_entropy(rng.multinomial(n, p, size=draws), n) - entropy(p)
    return float(np.quantile(samples, 1.0 - alpha))


def delta_min(alpha, power, p_null, p_alt, n, rng):
    """(T2.14)  Delta_min = q_{1-alpha}(W_null) - q_{1-power}(W_alt)."""
    return (null_quantile(1.0 - alpha, p_null, n, rng)
            - null_quantile(1.0 - power, p_alt, n, rng))


def zipf(K):
    w = 1.0 / np.arange(1.0, K + 1.0)
    return w / w.sum()


def tilt(p, lam):
    if lam == 0.0:
        return np.ones(p.size) / p.size
    w = np.asarray(p, dtype=float) ** lam
    return w / w.sum()


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------


def note(line):
    print(line)


def check_tier1():
    note("")
    note("TIER 1 -- discrete Kramers law")
    note("-" * 70)

    # ---- T1.3  detailed balance is machine zero on both instances --------
    for name, builder, T in (("double_well", double_well, 0.4),
                             ("three_basin", three_basin, 0.5)):
        P, pi, _, _ = builder(T)
        resid = float(np.abs(np.diag(pi) @ P - P.T @ np.diag(pi)).max())
        rowsum = float(np.abs(P.sum(axis=1) - 1.0).max())
        note(f"  T1.3  {name:11s} ||diag(pi)P - P^T diag(pi)||_max = {resid:.6e}"
             f"   row-sum dev = {rowsum:.3e}")
        assert resid <= 1e-15, (
            f"T1.3 detailed balance broken on {name}: residual {resid:.3e} > 1e-15; "
            "the chain is not reversible by construction and every pi-weighted "
            "quantity below is meaningless")
        assert rowsum <= 1e-15, (
            f"T1.3 {name} is not stochastic: max row-sum deviation {rowsum:.3e}")

    # ---- T1.5/T1.6  MFPT by three routes ---------------------------------
    P, pi, _, basins = double_well(0.4)
    A = basins["A"]
    Q, N = fundamental(P, A)
    ones = np.ones(len(A))
    tau_inv = N @ ones
    tau_solve = np.linalg.solve(np.eye(len(A)) - Q, ones)
    S, _ = occupancy_doubling(Q, doublings=20)
    tau_series = S @ ones
    d_solve = float(np.abs(tau_inv - tau_solve).max())
    d_series = float(np.abs(tau_inv - tau_series).max())
    note(f"  T1.5  MFPT out of A, tau = {np.array2string(tau_inv, precision=8)}")
    note(f"  T1.5  |N1 - LU solve| = {d_solve:.6e}")
    note(f"  T1.6  |N1 - series|   = {d_series:.6e}   (2^20-term occupancy, no solve)")
    assert d_solve <= 1e-10, (
        f"T1.5 the explicit inverse and the LU solve of (I-Q)tau = 1 disagree by "
        f"{d_solve:.3e} > 1e-10")
    assert d_series <= 1e-10, (
        f"T1.6 the resolvent (I-Q)^-1 and the truncated occupancy sum_t Q^t "
        f"disagree by {d_series:.3e} > 1e-10; CEQ.Occupancy.occupancy_telescope "
        "says the gap is exactly Q^N tau, so either rho(Q) >= 1 or the doubling "
        "recursion is wrong")

    # ---- T1.7  the telescoping identity itself, at finite N --------------
    S64, _ = occupancy_doubling(Q, doublings=6)
    tele = float(np.abs((np.eye(len(A)) - Q) @ S64
                        - (np.eye(len(A)) - np.linalg.matrix_power(Q, 64))).max())
    note(f"  T1.7  ||(I-Q)S_64 - (I - Q^64)|| = {tele:.6e}   "
         "(CEQ.Occupancy.occupancy_telescope at N=64)")
    assert tele <= 1e-12, (
        f"T1.7 the finite Neumann telescope proved in CEQ.Occupancy fails "
        f"numerically at N=64: residual {tele:.3e}")

    # ---- T1.8  the MFPT vector IS a Perron certificate --------------------
    rho = float(max(abs(np.linalg.eigvals(Q))))
    bound = 1.0 - 1.0 / float(tau_inv.max())
    qtau = float(np.abs(Q @ tau_inv - (tau_inv - ones)).max())
    note(f"  T1.8  ||Q tau - (tau - 1)|| = {qtau:.6e}")
    note(f"  T1.8  rho(Q) = {rho:.12f}  <=  1 - 1/max(tau) = {bound:.12f}")
    assert qtau <= 1e-9, f"T1.8 Q tau = tau - 1 fails: residual {qtau:.3e}"
    assert rho <= bound + 1e-12, (
        f"T1.8 the Collatz-Wielandt bound from CEQ.Contraction.PerronCertificate "
        f"with w = tau fails: rho(Q) = {rho:.12f} > 1 - 1/max(tau) = {bound:.12f}")

    # ---- T1.9  quasi-stationary escape is exactly geometric ---------------
    evals, evecs = np.linalg.eig(Q.T)
    k = int(np.argmax(evals.real))
    lam2 = float(evals.real[k])
    nu = evecs[:, k].real
    nu = nu / nu.sum()
    qsd_mfpt = float(nu @ tau_inv)
    geo = 1.0 / (1.0 - lam2)
    pi_mfpt = float((pi[A] / pi[A].sum()) @ tau_inv)
    note(f"  T1.9  lambda_2 := rho(Q) = {lam2:.12f}   E_qsd[tau] = {qsd_mfpt:.9f}   "
         f"t_rel = 1/(1-lambda_2) = {geo:.9f}   rel = {abs(qsd_mfpt-geo)/geo:.3e}")
    note(f"  T1.10 pre-asymptotic pi_A-weighted MFPT = {pi_mfpt:.9f}   "
         f"escape rate k_A = {1.0/pi_mfpt:.9e}   QSD/pi_A ratio = "
         f"{qsd_mfpt/pi_mfpt:.9f}")
    assert abs(qsd_mfpt - geo) / geo <= 1e-10, (
        f"T1.9 escape from the quasi-stationary distribution is not geometric: "
        f"E_qsd[tau] = {qsd_mfpt:.9f} against 1/(1-lambda_2) = {geo:.9f}")

    # ---- T1.11/T1.12  splitting probabilities == committor ----------------
    for name, builder, T, boundary in (
            ("double_well", double_well, 0.4, ("A", "B")),
            ("three_basin", three_basin, 0.5, ("B", "C"))):
        P, pi, _, basins = builder(T)
        n = P.shape[0]
        targets = [basins[b] for b in boundary]
        absorbing = [i for t in targets for i in t]
        interior = [i for i in range(n) if i not in absorbing]
        _, N = fundamental(P, interior)
        split = [N @ P[np.ix_(interior, t)] @ np.ones(len(t)) for t in targets]
        conserved = float(np.abs(sum(split) - 1.0).max())
        worst = 0.0
        for r, _ in enumerate(targets):
            bvals = {}
            for rr, tt in enumerate(targets):
                for i in tt:
                    bvals[i] = 1.0 if rr == r else 0.0
            q = committor_dirichlet(P, bvals)
            worst = max(worst, float(np.abs(q[interior] - split[r]).max()))
        note(f"  T1.11 {name:11s} |(I-Q)^-1 R - Dirichlet committor| = {worst:.6e}"
             f"   |sum_r q_r - 1| = {conserved:.3e}")
        assert worst <= 1e-10, (
            f"T1.11 splitting probabilities from (I-P_II)^-1 P_IB disagree with the "
            f"Dirichlet solve of Lq=0 on {name} by {worst:.3e} > 1e-10")
        assert conserved <= 1e-10, (
            f"T1.12 splitting probabilities on {name} do not sum to one: "
            f"deviation {conserved:.3e}")

    # ---- T1.15  Kirchhoff forest ratio, a third independent route ---------
    P, pi, _, basins = double_well(0.4)
    omega = kirchhoff_committor(P, pi, basins["A"], basins["B"])
    n = P.shape[0]
    interior = [i for i in range(n)
                if i not in basins["A"] and i not in basins["B"]]
    _, N = fundamental(P, interior)
    chain_q = N @ P[np.ix_(interior, basins["A"])] @ np.ones(len(basins["A"]))
    gap = float(np.abs(omega[interior] - chain_q).max())
    note(f"  T1.15 Kirchhoff grounded solve vs absorbing-chain committor = {gap:.6e}"
         "   (MATHEMATICS.md sec.11 step 4)")
    assert gap <= 1e-10, (
        f"T1.15 the section-11 instrument law fails on the reversible chain's own "
        f"weighted Laplacian: gap {gap:.3e} > 1e-10")

    # ---- T1.16/T1.17  capacity and the exact mean-hitting identity --------
    _, cap_e, cap_d, lhs, rhs = capacity_and_hitting(P, pi, basins["A"], basins["B"])
    dcap = abs(cap_e - cap_d) / cap_e
    dhit = abs(lhs - rhs) / abs(rhs)
    note(f"  T1.16 cap(A,B) escape sum = {cap_e:.12e}  Dirichlet form = {cap_d:.12e}"
         f"  rel = {dcap:.3e}")
    note(f"  T1.17 E_nu[tau_B] = {lhs:.9f}   (sum_x pi_x h_x)/cap = {rhs:.9f}"
         f"   rel = {dhit:.3e}")
    assert dcap <= 1e-10, (
        f"T1.16 the two capacity formulas disagree by relative {dcap:.3e}")
    assert dhit <= 1e-10, (
        f"T1.17 the exact mean-hitting-time identity E_nu[tau_B] = "
        f"(sum pi h)/cap fails by relative {dhit:.3e}; the Arrhenius reduction of "
        "T1.18 rests on it")

    # ---- T1.18  Arrhenius: k -> cap/Z_A, and the series-resistance form ---
    note("  T1.18 escape rate against capacity / Z_A on the double well:")
    prev = None
    for T in (0.8, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15):
        P, pi, _, basins = double_well(T)
        A = basins["A"]
        Q, _ = fundamental(P, A)
        tau = np.linalg.solve(np.eye(len(A)) - Q, np.ones(len(A)))
        piA = pi[A] / pi[A].sum()
        k = 1.0 / float(piA @ tau)
        Z_A = float(pi[A].sum())
        c_saddle = float(pi[2] * P[2, 3])
        R_series = float(sum(1.0 / (pi[i] * P[i, i + 1]) for i in (0, 1, 2)))
        one_bond = k * Z_A / c_saddle
        series = k * Z_A * R_series
        note(f"        T={T:5.3f}  k={k:.6e}  k Z_A / c_saddle = {one_bond:.6f}"
             f"   k Z_A R_series = {series:.6f}")
        if prev is not None:
            assert abs(series - 1.0) <= abs(prev - 1.0) + 1e-12, (
                f"T1.18 the series-resistance rate k = 1/(Z_A R) is not improving as "
                f"T falls: |err| went {abs(prev-1.0):.3e} -> {abs(series-1.0):.3e} "
                f"at T={T}")
        prev = series
    assert abs(prev - 1.0) <= 2e-5, (
        f"T1.18 k Z_A R_series does not converge to 1 at T=0.15: {prev:.9f}")

    # ---- T1.19  barrier ranking and rate ranking DISAGREE -----------------
    dE_B, dE_C = 2.00, 2.20
    note("  T1.19 three-basin exit ranking, barrier(B) = 2.00 < barrier(C) = 2.20:")
    hot = channel_split(0.5)
    cold = channel_split(0.10)
    for label, r in (("T=0.50", hot), ("T=0.10", cold)):
        note(f"        {label}  q_B={r['QB']:.9f}  q_C={r['QC']:.9f}"
             f"   k_B={r['kB']:.6e}  k_C={r['kC']:.6e}")
    assert dE_B < dE_C, (
        "T1.19 instance mis-specified: channel B is not the lower barrier")
    assert hot["kC"] > hot["kB"], (
        "T1.19 THE COUNTEREXAMPLE DID NOT FIRE. At T=0.5 the higher-barrier channel "
        f"C must carry the larger rate: k_C={hot['kC']:.6e} vs k_B={hot['kB']:.6e}")
    assert cold["kB"] > cold["kC"], (
        "T1.19 at T=0.10 the two rankings must agree again (energy beats "
        f"multiplicity): k_B={cold['kB']:.6e} vs k_C={cold['kC']:.6e}")

    # ---- T1.20  the inversion is EXACT, not asymptotic --------------------
    note("  T1.20 splitting ratio against the closed form m exp(-dE/T), m=4, dE=0.20:")
    worst = 0.0
    for T in (0.5, 0.35, 0.25, 0.2, 0.15, 0.12):
        r = channel_split(T)
        ratio = r["QC"] / r["QB"]
        pred = 4.0 * math.exp(-0.20 / T)
        rel = abs(ratio - pred) / pred
        worst = max(worst, rel)
        note(f"        T={T:5.3f}  q_C/q_B = {ratio:.15f}   4 exp(-0.2/T) = "
             f"{pred:.15f}   rel = {rel:.3e}")
    assert worst <= 1e-12, (
        f"T1.20 the tree's series-parallel conductance ratio is not exact: worst "
        f"relative deviation {worst:.3e} > 1e-12")

    # ---- T1.21  the crossover temperature --------------------------------
    lo, hi = 0.05, 0.5
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        r = channel_split(mid)
        if r["QC"] > r["QB"]:
            hi = mid
        else:
            lo = mid
    T_star = 0.5 * (lo + hi)
    pred = 0.20 / math.log(4.0)
    note(f"  T1.21 measured crossover T* = {T_star:.15f}   predicted dE/ln m = "
         f"{pred:.15f}   rel = {abs(T_star-pred)/pred:.3e}")
    assert abs(T_star - pred) / pred <= 1e-9, (
        f"T1.21 the guard condition dE > T ln(prefactor ratio) does not locate the "
        f"crossover: measured {T_star:.12f} against predicted {pred:.12f}")

    # ---- T1.22  where float64 loses the exact route ----------------------
    note("  T1.22 conditioning of (I - P_II) as T falls:")
    for T in (0.20, 0.15, 0.12, 0.10, 0.07, 0.05):
        r = channel_split(T)
        note(f"        T={T:5.3f}  cond = {r['cond']:.6e}   |q_B + q_C - 1| = "
             f"{abs(r['QB'] + r['QC'] - 1.0):.6e}")
    broken = channel_split(0.05)
    assert abs(broken["QB"] + broken["QC"] - 1.0) > 1e-6, (
        "T1.22 expected the T=0.05 solve to lose conservation in float64; it did "
        "not, so the recorded conditioning limit is wrong")


def check_tier2():
    note("")
    note("TIER 2 -- fan-out detector null distribution")
    note("-" * 70)
    rng = np.random.default_rng(20260831)

    # ---- T2.0  the stdlib chi-square quantile is right --------------------
    # Reference values from scipy.stats.chi2.ppf, quoted here rather than
    # imported.  T2.0 already earned its keep once: it rejected a
    # mis-remembered chi2_ppf(0.01, 7) = 1.239042304265752, and the series
    # solver below was the correct one.
    for p, nu, want in ((0.95, 1, 3.841458820694124),
                        (0.99, 3, 11.344866730144373),
                        (0.01, 7, 1.2390423055679296),
                        (0.99, 31, 52.191394833191930),
                        (0.01, 31, 15.655456401681380)):
        got = chi2_ppf(p, nu)
        note(f"  T2.0  chi2_ppf({p}, {nu:2d}) = {got:.12f}   (want {want:.12f})")
        assert abs(got - want) <= 1e-9, (
            f"T2.0 the hand-rolled incomplete gamma is wrong: chi2_ppf({p},{nu}) = "
            f"{got:.12f}, expected {want:.12f}; every threshold below is built on it")

    # ---- T2.2/T2.4  bias and variance against Monte Carlo -----------------
    note("  T2.2  plug-in bias, 400 000 draws:")
    skew = np.array([.40, .20, .15, .10, .06, .05, .03, .01])
    for K, n, name, p in ((8, 500, "uniform", np.ones(8) / 8),
                          (8, 500, "skew", skew),
                          (32, 2048, "uniform", np.ones(32) / 32),
                          (32, 2048, "zipf", zipf(32))):
        M = 400000
        hh = plugin_entropy(rng.multinomial(n, p, size=M), n)
        mc = float(hh.mean()) - entropy(p)
        se = float(hh.std(ddof=1)) / math.sqrt(M)
        first = -(K - 1) / (2.0 * n)
        two = bias_two_term(p, n)
        note(f"        K={K:3d} n={n:5d} {name:8s} MC={mc:+.8e} se={se:.2e}"
             f"  Miller={first:+.8e} ({(mc-first)/se:+7.2f} se)"
             f"  two-term={two:+.8e} ({(mc-two)/se:+6.2f} se)")
        assert abs(mc - two) <= 4.0 * se, (
            f"T2.2 the two-term Harris bias formula misses the Monte Carlo bias on "
            f"{name} K={K} n={n} by {abs(mc-two)/se:.2f} standard errors "
            f"(MC {mc:+.8e}, formula {two:+.8e})")
        vmc = float(hh.var(ddof=1))
        vth = entropy_variance(p) / n
        if entropy_variance(p) > 0.0:
            note(f"        Var: MC = {vmc:.6e}   V1/n = {vth:.6e}   rel = "
                 f"{abs(vmc-vth)/vmc:.4f}")
            assert abs(vmc - vth) / vmc <= 0.12, (
                f"T2.4 the asymptotic variance V1/n = {vth:.6e} misses the Monte "
                f"Carlo variance {vmc:.6e} on {name} K={K} n={n}")

    # ---- T2.5  the degenerate law at a uniform null -----------------------
    note("  T2.6  uniform null: 2n(ln K - Hhat) should be chi2_{K-1}")
    for K, n in ((8, 200), (32, 256)):
        p = np.ones(K) / K
        M = 400000
        stat = 2.0 * n * (math.log(K)
                          - plugin_entropy(rng.multinomial(n, p, size=M), n))
        note(f"        K={K:3d} n={n:4d}  mean = {stat.mean():.6f} (want {K-1})"
             f"   var = {stat.var(ddof=1):.6f} (want {2*(K-1)})"
             f"   V1 = {entropy_variance(p):.1e}")
        assert entropy_variance(p) == 0.0, (
            f"T2.4 V1 must be exactly zero at a uniform p; got "
            f"{entropy_variance(p):.3e}")
        assert abs(stat.mean() - (K - 1)) <= 0.06 * (K - 1), (
            f"T2.6 the chi-square limit is wrong at K={K}, n={n}: mean "
            f"{stat.mean():.4f} against {K-1}")

    # ---- T2.8-T2.11  four thresholds, achieved alpha ---------------------
    note("  T2.8  achieved alpha at nominal alpha = 0.01, 400 000 draws:")
    alpha = 0.01
    z = z_ppf(1.0 - alpha)
    se_a = math.sqrt(alpha * (1.0 - alpha) / 400000)
    worst_gauss_uniform = 0.0
    for K, n, name, p in ((8, 200, "skew", skew),
                          (32, 256, "zipf", zipf(32)),
                          (8, 200, "uniform", np.ones(8) / 8),
                          (32, 256, "uniform", np.ones(32) / 32)):
        V1 = entropy_variance(p)
        tau_g = z * math.sqrt(V1 / n) - (K - 1) / (2.0 * n)
        tau_c = -chi2_ppf(alpha, K - 1) / (2.0 * n)
        tau_w = null_quantile(1.0 - alpha, p, n, rng)
        tau_b = bootstrap_threshold(alpha, p, n, rng)
        D = plugin_entropy(rng.multinomial(n, p, size=400000), n) - entropy(p)
        ag = float((D > tau_g).mean())
        ac = float((D > tau_c).mean())
        aw = float((D > tau_w).mean())
        ab = float((D > tau_b).mean())
        note(f"        K={K:3d} n={n:4d} {name:8s} V1={V1:.5f}")
        note(f"              normal  tau={tau_g:+.6f}  alpha={ag:.5f}")
        note(f"              chi2    tau={tau_c:+.6f}  alpha={ac:.5f}")
        note(f"              conv    tau={tau_w:+.6f}  alpha={aw:.5f}")
        note(f"              boot    tau={tau_b:+.6f}  alpha={ab:.5f}")
        if V1 == 0.0:
            worst_gauss_uniform = max(worst_gauss_uniform, ag)
            assert abs(ac - alpha) <= 0.006, (
                f"T2.9 the chi-square threshold is not calibrated at the uniform "
                f"null K={K} n={n}: achieved alpha {ac:.5f} against {alpha}")
        else:
            assert ac > 0.15, (
                f"T2.9 expected the chi-square threshold to be badly "
                f"anti-conservative at the non-degenerate null {name}; achieved "
                f"alpha {ac:.5f}")
        assert aw <= alpha + 4.0 * se_a, (
            f"T2.10 the convolution threshold is anti-conservative on {name} K={K} "
            f"n={n}: achieved alpha {aw:.5f} > nominal {alpha}")
        assert abs(ab - alpha) <= 5.0 * se_a, (
            f"T2.11 the parametric-bootstrap threshold is not calibrated on {name} "
            f"K={K} n={n}: achieved alpha {ab:.5f} against {alpha} "
            f"({abs(ab-alpha)/se_a:.1f} se)")
    assert worst_gauss_uniform > 0.30, (
        "T2.8 expected the normal-only threshold to collapse at a uniform null "
        f"(alpha near 0.5); worst achieved alpha was {worst_gauss_uniform:.5f}. The "
        "degenerate case is the headline of T2.5-T2.9 and it must reproduce")

    # ---- T2.13  the power formula ----------------------------------------
    K, n = 32, 256
    p0, p1 = zipf(K), np.ones(K) / K
    H0 = entropy(p0)
    tau = null_quantile(0.99, p0, n, rng)
    note(f"  T2.12 K={K} n={n} null = zipf(32), H0 = {H0:.6f}, "
         f"V0 = {entropy_variance(p0):.6f}, tau_H = {tau:.7f}")
    note("        power formula against Monte Carlo, alternatives p prop zipf^lam:")
    for lam in (0.0, 0.25, 0.5, 0.9, 1.0):
        p = tilt(p0, lam)
        d = entropy(p) - H0
        pred = 1.0 - float(np.mean(
            rng.normal(0.0, math.sqrt(entropy_variance(p) / n), 400000)
            - rng.chisquare(K - 1, 400000) / (2.0 * n) <= tau - d))
        emp = float((plugin_entropy(rng.multinomial(n, p, size=200000), n)
                     - H0 > tau).mean())
        note(f"        lam={lam:4.2f}  Delta={d:.6f}  V1={entropy_variance(p):.5f}"
             f"   predicted = {pred:.5f}   empirical = {emp:.5f}")
        assert pred >= emp - 0.02, (
            f"T2.12 the power formula UNDER-states measured power at lam={lam}: "
            f"predicted {pred:.5f}, empirical {emp:.5f}; an over-optimistic power "
            "formula is the failure mode that makes a must-fire flaky")

    # ---- T2.14/T2.15  the minimum detectable gap -------------------------
    note("  T2.14 Delta_min for power >= 0.99 at alpha = 0.01, K = 32, "
         "null = zipf(32), alt = uniform(32):")
    zb = z_ppf(0.99)
    dmin_at_256 = None
    for nn in (64, 128, 256, 512, 1024, 2048, 4096):
        dmin = delta_min(0.01, 0.99, p0, p1, nn, rng)
        closed = (zb * math.sqrt(entropy_variance(p0) / nn)
                  + (chi2_ppf(0.99, K - 1) - (K - 1)) / (2.0 * nn))
        note(f"        n={nn:5d}   Delta_min = {dmin:.6f}   closed form = "
             f"{closed:.6f}   rel = {abs(dmin-closed)/dmin:.4f}")
        assert abs(dmin - closed) / dmin <= 0.06, (
            f"T2.16 the closed form for Delta_min misses the convolution quantile "
            f"at n={nn} by relative {abs(dmin-closed)/dmin:.4f}")
        if nn == 256:
            dmin_at_256 = dmin
    natural = math.log(K) - H0
    note(f"        natural gap ln 32 - H(zipf 32) = {natural:.6f}"
         f"   headroom at n = 256: {natural/dmin_at_256:.2f}x")

    # ---- T2.17  Delta_min is ALTERNATIVE-SPECIFIC, not gap-specific -------
    # The trap: take Delta_min computed for a uniform alternative (V1 = 0) and
    # plant a state that has that entropy gap but is NOT uniform.  Its own V1
    # is large, so the power collapses.  The bed must design the must-fire on
    # V1, not on Delta alone.
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if entropy(tilt(p0, mid)) - H0 > dmin_at_256:
            lo = mid
        else:
            hi = mid
    p_trap = tilt(p0, 0.5 * (lo + hi))
    emp_trap = float((plugin_entropy(rng.multinomial(n, p_trap, size=200000), n)
                      - H0 > tau).mean())
    note(f"  T2.17 TRAP: a non-uniform state planted at the uniform-alternative "
         f"Delta_min = {dmin_at_256:.6f}")
    note(f"        its own V1 = {entropy_variance(p_trap):.5f} (not 0), "
         f"empirical power = {emp_trap:.5f}")
    assert emp_trap < 0.95, (
        "T2.17 the alternative-specificity trap did not reproduce: a state planted "
        f"at the uniform-alternative Delta_min with V1 = "
        f"{entropy_variance(p_trap):.5f} scored power {emp_trap:.5f}. If this is "
        "genuinely near 0.99 then Delta_min does not depend on the alternative's "
        "variance and T2.14 is over-stated")

    # ---- T2.18  the self-consistent Delta_min: solve Delta = Delta_min(p) --
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        p = tilt(p0, mid)
        if entropy(p) - H0 > delta_min(0.01, 0.99, p0, p, n, rng):
            lo = mid
        else:
            hi = mid
    p_star = tilt(p0, 0.5 * (lo + hi))
    gap = entropy(p_star) - H0
    tau_boot = bootstrap_threshold(0.01, p0, n, rng)
    D = plugin_entropy(rng.multinomial(n, p_star, size=400000), n) - H0
    emp_conv = float((D > tau).mean())
    emp_boot = float((D > tau_boot).mean())
    note(f"  T2.18 self-consistent fixed point at n = {n}: Delta = {gap:.6f}, "
         f"V1 = {entropy_variance(p_star):.5f}")
    note(f"        power under the closed-form tau_H = {tau:.6f}: {emp_conv:.5f}")
    note(f"        power under the bootstrap tau_H = {tau_boot:.6f}: {emp_boot:.5f}")
    assert emp_conv >= 0.98, (
        f"T2.18 an alternative planted at its OWN Delta_min is detected only "
        f"{emp_conv:.5f} of the time under the closed-form threshold; the expected "
        "shortfall from the threshold's conservatism is about 0.006, not more")
    assert emp_boot >= 0.985, (
        f"T2.18 the calibrated bootstrap threshold does not restore nominal power: "
        f"{emp_boot:.5f} against 0.99")
    assert emp_boot > emp_conv, (
        f"T2.18 the bootstrap threshold ({emp_boot:.5f}) should out-power the "
        f"conservative closed form ({emp_conv:.5f}); it did not, so the "
        "conservatism story is wrong")

    # ---- T2.19  fully degenerate design ----------------------------------
    note("  T2.19 fully degenerate design (null uniform on k0, alt uniform on K), "
         "Delta_min scales as 1/n:")
    for K2, k0, nn in ((32, 16, 256), (32, 16, 1024), (32, 8, 256), (32, 24, 1024)):
        tau_d = -chi2_ppf(0.01, k0 - 1) / (2.0 * nn)
        dmin = tau_d + chi2_ppf(0.99, K2 - 1) / (2.0 * nn)
        actual = math.log(K2 / k0)
        note(f"        K={K2} k0={k0:3d} n={nn:5d}   tau_H = {tau_d:+.7f}"
             f"   Delta_min = {dmin:.6f}   actual gap ln(K/k0) = {actual:.6f}"
             f"   headroom = {actual/dmin:.2f}x")
        assert actual > dmin, (
            f"T2.19 the degenerate design at K={K2}, k0={k0}, n={nn} does not clear "
            f"its own Delta_min: gap {actual:.6f} <= {dmin:.6f}")


def main():
    note("V13 Tier-1 / Tier-2 derivation check")
    note(f"numpy {np.__version__}, python {sys.version.split()[0]}")
    check_tier1()
    check_tier2()
    note("")
    note("ALL ASSERTIONS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""LSTD-BED -- does WHITENING buy the committor anything, or does the solve already win?

THE LEAP. A Dr House round proposed that this project's exact committor solve
and a delta rule are not rivals but UPSTREAM and DOWNSTREAM: the project
inverts an operator it is GIVEN (q = 1 on one absorbing set, 0 on the other,
q = Pq on the transient set, one pass because P is a line graph); a delta rule
with fixed point S* = E[k k^T]^-1 E[k v^T] IDENTIFIES an operator it never
inverts. The proposed route is LSTD (Bradtke-Barto 1996): estimate P-hat by
Sherman-Morrison recursive least squares -- the same rank-1 primitive the do(a)
arm already owns -- then run the EXISTING exact solve on P-hat, giving a
committor over latent STATES rather than token POSITIONS.

THE MEASUREMENT THAT REFUTES IT. The leap only pays if the WHITENING factor
E[k k^T]^-1 is what makes the estimate good enough to solve. Take one stream
off one chain, build TWO operator estimates from it -- Hebbian sum k v^T (this
family's beta = 0 memory, no whitening) and Sherman-Morrison RLS (whitened) --
push BOTH through the SAME exact solve, and compare both committors against q*
from the TRUE P.

    LEAP BOUND    ||q_H - q*||_inf exceeds ||q_D - q*||_inf by more than 1e-2.
    LEAP REFUTED  the two land within 1e-2 of each other: the solve dominates
                  the delta term and the identifier is not worth building.

WHY THE BED HAS A CYCLE. A transient block with no directed cycle is a line
graph, and a line is the case the project already solves in one triangular
pass, so a comparison run on one measures nothing. States 4 -> 5 -> 6 -> 4
carry 0.55 each with the remaining 0.45 leaking off the cycle, and
`build_chain` refuses to return a chain whose transient adjacency is nilpotent.

WHY THE EMBEDDING IS SKEWED. With orthonormal keys the Gram matrix is the
identity, Hebbian accumulation and least squares are the SAME estimator, and
the bed is inert. The embedding therefore has singular values geomspace(1, 0.1)
-- cond(E) = 10 exactly -- and the off-diagonal mass of E[k k^T] is asserted,
not hoped for. That inertness is not merely avoided: it is PLANTED, at
cond = 1.0, as the negative the headline gap has to survive.

THE ALGEBRA THE BED IS TESTING (re-derived here, not quoted). With phi(s) the
s-th column of E, N the transition count matrix, D = diag(source counts) and
G = E^T E:

    S_H = sum_t k_t v_t^T             = E N E^T
    read S_H^T k_s, decode by E^-1    -> row s of  G N       <- Gram cross-talk
    S_D = (sum k k^T)^-1 sum k v^T    = E^-T D^-1 N E^T
    read S_D^T k_s, decode by E^-1    -> row s of  D^-1 N    <- exact empirical P

so the Hebbian row for state s is a G-weighted MIXTURE of the rows of every
other state, and that bias does not shrink with T. The bed exists to find out
whether the committor solve CARES.

L-NULL, APPLIED TO THIS BED.
    VARIED : the operator estimator, and nothing else.
    PINNED : the stream (one array, both arms), the seed, the embedding E, the
             decode (E^-1, clip at 0, row-renormalise), the exact solve
             (ceq.beds.bed_1.committor, imported not reimplemented), the
             absorbing sets A = {0,1} at q = 0 and B = {2,3} at q = 1.
Both arms are handed the identical objects and `run()` returns their id()s so a
test can assert it rather than trust this paragraph.

RECORDED NEGATIVE -- the verbatim RED phase, before this file existed:

    python -m pytest tests/curvature/test_lstd_bed.py -q   # 6612cdf, WIN-16QAL06O9GB
    E  AssertionError: ceqjepa/lstd_bed.py did not import: ImportError("cannot
       import name 'lstd_bed' from 'ceqjepa' (C:\\Users\\seal\\Desktop\\New
       folder (32)\\ceqjepa\\__init__.py)")
    14 failed in 18.45s                                    # exit=1, not piped

RUN: python -m ceqjepa.lstd_bed        (self-check, ends with the banner)
     python -m pytest tests/curvature/test_lstd_bed.py -q
"""

from __future__ import annotations

import os

# Pinned BEFORE numpy imports. Every solve here is 12x12 to 256x256, where a
# threaded BLAS spends more time in its spin-wait than in the factorisation, and
# section (8) is a microbenchmark: on a loaded box the multi-threaded numbers
# moved by more than the effect being measured. One core, reproducible, honest.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import subprocess
import sys
import time

import numpy as np
import scipy.linalg as sla

# The EXACT SOLVE, imported rather than rewritten: `committor(L, A, B)` solves
# (Lq)_i = 0 on the interior with q = 0 on A and q = 1 on B, by a dense LU.
# With L = P - I that is exactly q = Pq on the transient set.
from ceq.beds.bed_1 import committor as exact_committor

D_STATES = 16
A_SET = [0, 1]          # absorbing, q = 0
B_SET = [2, 3]          # absorbing, q = 1
CYCLE = (4, 5, 6)       # the directed 3-cycle, in the transient block
CYCLE_W = 0.55          # on-cycle weight; the other 0.45 is the leak
SEED = 20260914
T_DEFAULT = 10_000
RIDGE = 1e-6            # RLS initialisation A_0 = RIDGE * I; reported, not hidden


def _commit():
    """HEAD at run time. Never a literal asserted against git -- see L-SURFACE."""
    try:
        p = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=20)
        return p.stdout.strip() if p.returncode == 0 else "unknown"
    except Exception:                                     # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------------------
# the chain
# ---------------------------------------------------------------------------

def build_chain(seed: int = SEED) -> dict:
    """A d = 16 absorbing chain whose transient block carries a 3-cycle with leak."""
    rng = np.random.default_rng(seed)
    absorbing = sorted(A_SET + B_SET)
    transient = [i for i in range(D_STATES) if i not in absorbing]

    P = np.zeros((D_STATES, D_STATES))
    for a in absorbing:
        P[a, a] = 1.0
    for i in transient:
        succ = rng.choice([j for j in range(D_STATES) if j != i], size=5, replace=False)
        P[i, succ] = rng.dirichlet(np.ones(5))
        P[i, rng.choice(absorbing)] += 0.15      # absorption is certain from everywhere
        P[i] /= P[i].sum()
    for i, j in zip(CYCLE, CYCLE[1:] + CYCLE[:1]):
        off = P[i].copy()
        off[j] = 0.0
        P[i] = CYCLE_W * np.eye(D_STATES)[j] + (1.0 - CYCLE_W) * off / off.sum()

    Q = P[np.ix_(transient, transient)]
    adj = (Q > 0).astype(np.float64)
    if np.abs(np.linalg.matrix_power(adj, len(transient))).max() == 0:
        raise AssertionError("transient adjacency is nilpotent: this bed is a LINE")
    if np.trace(np.linalg.matrix_power(adj, 3)) <= 0:
        raise AssertionError("no directed 3-cycle in the transient block")
    for i in CYCLE:
        if sum(P[i, j] for j in CYCLE) >= 1.0 - 1e-9:
            raise AssertionError("state %d has no leak off the cycle" % i)
    rho = float(np.abs(np.linalg.eigvals(Q)).max())
    if rho >= 1.0:
        raise AssertionError("spectral radius of Q is %.6f: absorption not certain" % rho)
    if not np.allclose(P.sum(1), 1.0):
        raise AssertionError("P is not row-stochastic")
    return dict(P=P, A=list(A_SET), B=list(B_SET), transient=transient,
                absorbing=absorbing, cycle=list(CYCLE), rho=rho)


def build_embedding(d: int = D_STATES, cond: float = 10.0, seed: int = SEED + 1) -> np.ndarray:
    """Full-rank, deliberately NON-orthogonal: singular values geomspace(1, 1/cond).

    cond = 1.0 is the planted negative -- E orthogonal, Gram = I, the two arms
    provably the same estimator.
    """
    rng = np.random.default_rng(seed)
    U = np.linalg.qr(rng.standard_normal((d, d)))[0]
    V = np.linalg.qr(rng.standard_normal((d, d)))[0]
    s = np.geomspace(1.0, 1.0 / cond, d)
    return U @ np.diag(s) @ V.T


def simulate(P: np.ndarray, T: int, transient: list[int], absorbing: list[int],
             seed: int = SEED + 2):
    """T recorded transitions. On absorption the self-loop (a -> a) is recorded
    once -- it is a real transition of this chain, and without it no absorbing
    state is ever a SOURCE, so sum k k^T is singular -- then the walker is
    teleported to a uniform transient state. The teleport is NOT recorded."""
    rng = np.random.default_rng(seed)
    cdf = np.cumsum(P, axis=1)
    absorbing_set = set(absorbing)
    src = np.empty(T, dtype=np.int64)
    dst = np.empty(T, dtype=np.int64)
    s = int(rng.choice(transient))
    for t in range(T):
        s2 = int(np.searchsorted(cdf[s], rng.random()))
        src[t], dst[t] = s, s2
        s = int(rng.choice(transient)) if s in absorbing_set else s2
    return src, dst


# ---------------------------------------------------------------------------
# the two estimators -- this is the ONLY thing that varies between the arms
# ---------------------------------------------------------------------------

def hebbian(K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """S_H = sum_t k_t v_t^T. The beta = 0 linear-attention memory, unwhitened."""
    hebbian.saw = (id(K), id(V))          # recorded HERE, so the L-NULL check is
    return K.T @ V                        # an observation and not id(K) == id(K)


def rls_sherman_morrison(K: np.ndarray, V: np.ndarray, ridge: float = RIDGE) -> np.ndarray:
    """S_D = (ridge I + sum k k^T)^-1 sum k v^T, by rank-1 Sherman-Morrison.

    The identical primitive as the do(a) arm's clamped-row read: one rank-1
    correction per sample, no d x d inverse ever formed after the first.
    """
    rls_sherman_morrison.saw = (id(K), id(V))
    d = K.shape[1]
    A_inv = np.eye(d) / ridge
    b = np.zeros((d, d))
    for k, v in zip(K, V):
        u = A_inv @ k
        A_inv -= np.outer(u, u) / (1.0 + k @ u)
        b += np.outer(k, v)
    return A_inv @ b


def decode(S: np.ndarray, E: np.ndarray) -> tuple[np.ndarray, float]:
    """Operator in embedding space -> transition matrix over STATES.

    The read for state s is v_hat = S^T phi(s); its coefficients over states are
    E^-1 v_hat. Negative coefficients are clipped and rows renormalised -- the
    SAME treatment for both arms, so the clip cannot be what separates them.
    Returns (P_hat, fraction of decoded mass that had to be clipped).
    """
    return _normalise(np.linalg.solve(E, S.T @ E).T)


def decode_dot(S: np.ndarray, E: np.ndarray) -> tuple[np.ndarray, float]:
    """The decode the FAMILY actually has: a dot-product read, p(s') ~ <v_hat, phi(s')>.

    This is E^T where `decode` uses E^-1, and the two coincide only when E is
    orthogonal. It is here because "your E^-1 is the whitening in disguise" is
    the first thing a defender of the beta = 0 memory would say, and the answer
    has to be measured on the readout the family owns, not asserted.
    """
    return _normalise((E.T @ S.T @ E).T)


def _normalise(raw: np.ndarray) -> tuple[np.ndarray, float]:
    neg = float(np.abs(np.minimum(raw, 0.0)).sum() / np.abs(raw).sum())
    Pn = np.clip(raw, 0.0, None)
    rs = Pn.sum(1, keepdims=True)
    if (rs <= 1e-12).any():
        raise AssertionError("a decoded row is all non-positive: %r" % rs.ravel())
    return Pn / rs, neg


# ---------------------------------------------------------------------------
# the solve, shared
# ---------------------------------------------------------------------------

def solve_committor(P: np.ndarray, A: list[int], B: list[int]):
    """q by the EXISTING dense-LU solve, plus its own residual, re-derived."""
    q = exact_committor(P - np.eye(P.shape[0]), A, B)
    interior = [i for i in range(P.shape[0]) if i not in A + B]
    res = float(np.abs((P @ q - q)[interior]).max())
    return q, res


def monte_carlo_committor(P, transient, B, n_paths=4000, seed=SEED + 3, max_steps=20000):
    """q* bound to counted absorptions rather than trusted from the solve."""
    rng = np.random.default_rng(seed)
    cdf = np.cumsum(P, axis=1)
    absorbing = np.where(np.diag(P) == 1.0)[0]
    in_B = np.zeros(P.shape[0], dtype=bool)
    in_B[B] = True
    out = np.zeros(P.shape[0])
    for i in transient:
        cur = np.full(n_paths, i, dtype=np.int64)
        done = np.zeros(n_paths, dtype=bool)
        for _ in range(max_steps):
            live = ~done
            if not live.any():
                break
            u = rng.random(live.sum())
            nxt = (cdf[cur[live]] < u[:, None]).sum(1)
            cur[live] = nxt
            done |= np.isin(cur, absorbing)
        if not done.all():
            raise AssertionError("%d paths from %d never absorbed" % ((~done).sum(), i))
        out[i] = in_B[cur].mean()
    return out


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def run(T: int = T_DEFAULT, seed: int = SEED, cond: float = 10.0,
        mc: bool = True, n_paths: int = 4000) -> dict:
    """One bed: one stream, one embedding, one solve, TWO operator estimates."""
    bed = build_chain(seed)
    P, tr, A, B = bed["P"], bed["transient"], bed["A"], bed["B"]
    E = build_embedding(cond=cond, seed=seed + 1)

    src, dst = simulate(P, T, tr, bed["absorbing"], seed=seed + 2)
    K = E[:, src].T                      # k_t = phi(s_t)      -- PINNED, one array
    V = E[:, dst].T                      # v_t = phi(s_{t+1})  -- PINNED, one array

    M2 = K.T @ K / T                     # the empirical E[k k^T]
    offdiag = float(np.linalg.norm(M2 - np.diag(np.diag(M2))) / np.linalg.norm(M2))

    S_H = hebbian(K, V)
    S_D = rls_sherman_morrison(K, V)
    S_batch = np.linalg.solve(K.T @ K + RIDGE * np.eye(E.shape[0]), K.T @ V)
    sm_vs_batch = float(np.abs(S_D - S_batch).max())

    P_H, neg_H = decode(S_H, E)
    P_D, neg_D = decode(S_D, E)
    P_H_dot, _ = decode_dot(S_H, E)
    P_D_dot, _ = decode_dot(S_D, E)

    # Is the RLS arm just the maximum-likelihood empirical chain? (the algebra
    # says D^-1 N exactly; if it is, err_D is the SAMPLING floor of the stream,
    # not a defect of the estimator, and no estimator on this stream beats it.)
    N = np.zeros((D_STATES, D_STATES))
    np.add.at(N, (src, dst), 1.0)
    P_emp = N / np.maximum(N.sum(1, keepdims=True), 1e-12)
    mle_gap = float(np.abs(P_D - P_emp)[tr].max())

    # THE STRAWMAN DEFENCE. A defender of the beta = 0 memory would normalise
    # the Hebbian read per source (by the visit count, or by diag(Gram)). Any
    # per-source POSITIVE rescale c_s of the read is S' = E^-T diag(c) E^T S,
    # and the decode's row-renormalisation removes it. Measured, not argued.
    c = np.random.default_rng(seed + 4).uniform(0.2, 5.0, D_STATES)
    S_H_rescaled = np.linalg.solve(E.T, np.diag(c) @ (E.T @ S_H))
    P_H_rescaled, _ = decode(S_H_rescaled, E)
    hebb_rescale_gap = float(np.abs(P_H_rescaled - P_H).max())

    q_star, res_star = solve_committor(P, A, B)
    q_H, res_H = solve_committor(P_H, A, B)
    q_D, res_D = solve_committor(P_D, A, B)
    q_H_dot, _ = solve_committor(P_H_dot, A, B)
    q_D_dot, _ = solve_committor(P_D_dot, A, B)

    def err(q, ordv):
        return float(np.linalg.norm((q - q_star)[tr], ord=ordv))

    out = dict(
        T=T, seed=seed, cond=float(np.linalg.cond(E)), ridge=RIDGE,
        rho=bed["rho"], offdiag_mass=offdiag,
        q_star=q_star, q_H=q_H, q_D=q_D,
        q_spread=float(q_star[tr].max() - q_star[tr].min()),
        err_H_inf=err(q_H, np.inf), err_D_inf=err(q_D, np.inf),
        err_H_l2=err(q_H, 2), err_D_l2=err(q_D, 2),
        err_H_dot_inf=err(q_H_dot, np.inf), err_D_dot_inf=err(q_D_dot, np.inf),
        op_err_H=float(np.abs(P_H - P)[tr].max()), op_err_D=float(np.abs(P_D - P)[tr].max()),
        neg_mass_H=neg_H, neg_mass_D=neg_D,
        sm_vs_batch=sm_vs_batch, mle_gap=mle_gap, hebb_rescale_gap=hebb_rescale_gap,
        lu_residual_star=res_star, lu_residual_H=res_H, lu_residual_D=res_D,
        pinned=dict(varied="operator estimator only",
                    stream_id=(id(K), id(V)),
                    stream_id_H=hebbian.saw, stream_id_D=rls_sherman_morrison.saw,
                    embedding_id_H=id(E), embedding_id_D=id(E),
                    solve_id_H=id(solve_committor), solve_id_D=id(solve_committor)),
    )
    if mc:
        q_mc = monte_carlo_committor(P, tr, B, n_paths=n_paths, seed=seed + 3)
        out.update(mc_paths=n_paths,
                   mc_max_dev=float(np.abs(q_mc - q_star)[tr].max()),
                   mc_stderr_bound=0.5 / np.sqrt(n_paths))
    return out


def timings(n: int = 12, reps: int = 2000, seed: int = SEED) -> dict:
    """Dense LU on P-hat against the triangular pass the line-graph case gets,
    at the SAME n (n = 12 is the transient block size of this bed).

    Both are one LAPACK call from python on one core; at n = 12 the call
    overhead, not the O(n^3/3) against O(n^2/2), is what is being timed, which
    is why `demo()` also prints n = 64 and n = 256.
    """
    rng = np.random.default_rng(seed)
    Md = np.eye(n) - 0.6 * rng.random((n, n)) / n
    Mt = np.tril(np.eye(n) - 0.6 * rng.random((n, n)) / n)
    rhs = rng.random((n, 1))

    def bench(fn):
        best = np.inf
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            best = min(best, (time.perf_counter() - t0) / reps * 1e6)
        return best

    lu = bench(lambda: np.linalg.solve(Md, rhs))
    tri = bench(lambda: sla.solve_triangular(Mt, rhs, lower=True))
    return dict(n=n, reps=reps, lu_us=lu, tri_us=tri, ratio=lu / tri)


# ---------------------------------------------------------------------------
# the self-check
# ---------------------------------------------------------------------------

def demo() -> None:
    commit, host = _commit(), __import__("platform").node()
    print("LSTD-BED  commit %s  %s  python %s  numpy %s"
          % (commit, host, sys.version.split()[0], np.__version__))
    print("PINNED: stream, seed, embedding, decode, exact solve, absorbing sets")
    print("VARIED: the operator estimator (Hebbian sum k v^T  vs  Sherman-Morrison RLS)")

    bed = build_chain()
    print("\n(1) THE BED IS NOT A LINE")
    print("    3-cycle %s at %.2f each, leak %.2f; rho(Q) = %.4f; transient n = %d"
          % (bed["cycle"], CYCLE_W, 1.0 - CYCLE_W, bed["rho"], len(bed["transient"])))

    r = run()
    assert r["q_spread"] > 0.2, "q* is flat (%.3f): a 0.1 error would be unreadable" % r["q_spread"]
    print("    q* spread over the transient block = %.4f" % r["q_spread"])
    print("    q* vs %d counted absorptions per state: max dev %.4f (4-sigma band %.4f)"
          % (r["mc_paths"], r["mc_max_dev"], 4 * r["mc_stderr_bound"]))
    assert r["mc_max_dev"] < 4 * r["mc_stderr_bound"], "the exact solve disagrees with Monte Carlo"

    print("\n(2) THE KEYS ARE NOT ORTHOGONAL")
    print("    cond(E) = %.4f; off-diagonal mass of E[k k^T] = %.4f"
          % (r["cond"], r["offdiag_mass"]))
    assert 9.0 < r["cond"] < 11.0, "cond(E) off spec"
    assert r["offdiag_mass"] > 0.20, "E[k k^T] is near-diagonal: the bed is INERT"

    print("\n(3) THE ESTIMATORS, T = %d" % r["T"])
    print("    Sherman-Morrison vs batch normal equations: max|diff| = %.3e" % r["sm_vs_batch"])
    assert r["sm_vs_batch"] < 1e-8, "the recursion is not least squares"
    print("    max|P_hat - P| over transient rows: Hebbian %.4f, RLS %.4f"
          % (r["op_err_H"], r["op_err_D"]))
    print("    decoded mass that had to be clipped: Hebbian %.4f, RLS %.4f"
          % (r["neg_mass_H"], r["neg_mass_D"]))
    print("    RLS arm vs the empirical MLE chain D^-1 N: max|diff| = %.3e" % r["mle_gap"])
    assert r["mle_gap"] < 1e-6, "the RLS arm is not the empirical MLE"
    print("    Hebbian arm under an arbitrary per-source rescale: max|diff| = %.3e"
          % r["hebb_rescale_gap"])
    assert r["hebb_rescale_gap"] < 1e-9, (
        "the Hebbian arm depends on a normalisation convention: the comparison "
        "would be against a strawman rather than against the beta = 0 memory")

    print("\n(4) THE HEADLINE")
    print("    ||q_D - q*||_inf = %.6f   (l2 %.6f)" % (r["err_D_inf"], r["err_D_l2"]))
    print("    ||q_H - q*||_inf = %.6f   (l2 %.6f)" % (r["err_H_inf"], r["err_H_l2"]))
    gap = r["err_H_inf"] - r["err_D_inf"]
    print("    gap = %.6f  ->  %s" % (gap, "LEAP BOUND" if gap > 1e-2 else "LEAP REFUTED"))
    print("    the predicted ||q_D - q*|| <= 1e-2 is MISSED at T = 1e4 by %.2fx; the RLS "
          "arm IS the\n    MLE (line above), so 1e-2 is a claim about SAMPLING, not about "
          "the estimator -- see (6)." % (r["err_D_inf"] / 1e-2))

    print("\n(5) THE PLANTED NEGATIVE -- orthogonal keys must collapse the gap")
    r0 = run(cond=1.0, mc=False)
    gap0 = abs(r0["err_H_inf"] - r0["err_D_inf"])
    print("    cond(E) = %.4f: ||q_H - q*||_inf = %.6f, ||q_D - q*||_inf = %.6f, gap %.3e"
          % (r0["cond"], r0["err_H_inf"], r0["err_D_inf"], gap0))
    assert gap0 <= 1e-2, "the gap survives orthogonal keys: it is not key correlation"
    print("    and ||q_D - q*||_inf moved by %.3e when cond(E) went 10 -> 1: the whitened "
          "arm is\n    embedding-INVARIANT, the Hebbian arm is not."
          % abs(r0["err_D_inf"] - r["err_D_inf"]))

    print("\n(6) DOES THE HEBBIAN ARM PLATEAU, OR IS IT JUST UNDER-SAMPLED?")
    sweep = {}
    for T in (1_000, 10_000, 100_000):
        rt = run(T=T, mc=False)
        sweep[T] = rt
        print("    T = %6d   ||q_H - q*||_inf = %.6f   ||q_D - q*||_inf = %.6f"
              % (T, rt["err_H_inf"], rt["err_D_inf"]))
    assert sweep[100_000]["err_D_inf"] < sweep[1_000]["err_D_inf"] / 5.0, \
        "the RLS arm does not converge with T: it is not consistent"
    assert sweep[100_000]["err_H_inf"] > 0.5 * sweep[1_000]["err_H_inf"], \
        "the Hebbian arm converges too: the bias claim is wrong"

    print("\n(7) IS THE GAP ONE SEED'S LUCK?")
    for s in (SEED, SEED + 100, SEED + 200, SEED + 300, SEED + 400):
        rs = run(seed=s, mc=False)
        print("    seed %d   ||q_H - q*||_inf = %.6f   ||q_D - q*||_inf = %.6f"
              % (s, rs["err_H_inf"], rs["err_D_inf"]))

    print("\n(8) DOES THE WIN SURVIVE THE FAMILY'S OWN READOUT?  (E^T instead of E^-1)")
    print("    dot-product decode: ||q_H - q*||_inf = %.6f, ||q_D - q*||_inf = %.6f"
          % (r["err_H_dot_inf"], r["err_D_dot_inf"]))
    print("    inverse decode    : ||q_H - q*||_inf = %.6f, ||q_D - q*||_inf = %.6f"
          % (r["err_H_inf"], r["err_D_inf"]))

    print("\n(9) WHAT THE ROUTE COSTS: dense LU vs the triangular pass at the same n")
    print("    LU residual ||(I-Q_hat)q - R 1||_inf: true P %.3e, RLS %.3e, Hebbian %.3e"
          % (r["lu_residual_star"], r["lu_residual_D"], r["lu_residual_H"]))
    for n, reps in ((12, 2000), (64, 1000)):
        tm = timings(n=n, reps=reps)
        print("    n = %3d, %4d reps: LU %8.3f us, triangular %8.3f us, ratio %.2fx"
              % (tm["n"], tm["reps"], tm["lu_us"], tm["tri_us"], tm["ratio"]))

    print("\nALL SELF-CHECKS PASSED")


if __name__ == "__main__":
    demo()

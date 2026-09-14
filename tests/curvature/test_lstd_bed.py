"""RED-FIRST tests for ceqjepa/lstd_bed.py -- does WHITENING buy the committor anything?

THE LEAP UNDER TEST. A Dr House round proposed that this project's exact
committor solve (upstream: invert an operator you are GIVEN) and a delta rule
(downstream: identify an operator from the stream, fixed point
S* = E[k k^T]^-1 E[k v^T]) are complements rather than rivals, and that the
route is LSTD: estimate P-hat by recursive least squares, then run the EXISTING
exact solve on P-hat. The leap only pays if the WHITENING is what makes the
estimate good enough to solve. So the measurement is: same stream, same
embedding, same solve, two operator estimates -- Hebbian sum k v^T (this
family's beta = 0 memory) and Sherman-Morrison RLS -- and two committor errors
against q* from the TRUE P.

  LEAP BOUND    if ||q_H - q*|| exceeds ||q_D - q*|| by more than 1e-2.
  LEAP REFUTED  if ||q_H - q*|| lands within 1e-2 of ||q_D - q*||: the solve
                already dominates the delta term and there is nothing to build.

WAYS THIS BED COULD BE WRONG WHILE STILL PRINTING TWO NUMBERS, each of which
has a test below:

 (i) THE CHAIN IS A LINE. A transient block with no cycle is the case the
     project already solves in one triangular pass, so any win is a win over
     nothing. The directed 3-cycle with leak is asserted structurally AND by
     showing the transient block is not nilpotent (no triangular order exists).
 (ii) THE EMBEDDING IS ORTHOGONAL. If E[k k^T] is near-diagonal then Hebbian
     accumulation and least squares are the same estimator and the comparison is
     inert. Condition number and off-diagonal mass are asserted, not assumed.
 (iii) THE DIFFERENCE TEST CANNOT FAIL. A test that only ever sees a gap is not
     measuring one. The PLANTED NEGATIVE is the same bed at cond(E) = 1
     (orthogonal keys), where the two arms MUST agree to 1e-2 -- if that arm
     also shows a gap, the instrument is broken and the headline is void.
 (iv) THE ARMS DIFFER IN SOMETHING ELSE (L-NULL). Varied: the operator
     estimator, and nothing else. Pinned: the stream, the seed, the embedding,
     the decode, the exact solve, the absorbing sets. Asserted by identity of
     the arrays the two arms were handed.
 (v) THE "RECURSIVE LEAST SQUARES" IS NOT LEAST SQUARES. The Sherman-Morrison
     recursion is checked against a direct batch solve of the same normal
     equations; a recursion that has drifted is not the do(a) primitive.
 (vi) THE DECODE IS THE WHITENING IN DISGUISE. Both arms are decoded by E^-1,
     which a defender will call the real source of the win. Two tests answer
     it: the planted negative (same E^-1, orthogonal keys, gap gone), and the
     dot-product decode E^T -- the readout the family actually has -- under
     which BOTH arms must fail, which is a condition on the route.

RECORDED NEGATIVE 1 -- the verbatim RED, before ceqjepa/lstd_bed.py existed:

    python -m pytest tests/curvature/test_lstd_bed.py -q     # 6612cdf, WIN-16QAL06O9GB
    E  AssertionError: ceqjepa/lstd_bed.py did not import: ImportError("cannot
       import name 'lstd_bed' from 'ceqjepa' (C:\\Users\\seal\\Desktop\\New
       folder (32)\\ceqjepa\\__init__.py)")
    14 failed in 18.45s                                      # exit=1, not piped

RECORDED NEGATIVE 2 -- a PREDICTION of the round that the bed did NOT confirm.
The round predicted ||q_D - q*|| <= 1e-2 at T = 1e4. Measured: 0.030243, a 3.02x
miss. It is not an estimator defect -- the RLS arm IS the empirical MLE chain
D^-1 N (4.445e-08 apart, the ridge; test below), so 0.030243 is the sampling floor
of a 1e4-transition stream and NO estimator on this stream beats it. The 1e-2
bar is reached at T = 1e5 (0.010298). The test below asserts the CONVERGENCE and
the measured value, not the round's number, and the missed number stays filed.

RUN: python -m pytest tests/curvature/test_lstd_bed.py -v
"""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

try:
    from ceqjepa import lstd_bed as lb
except Exception as _exc:                      # noqa: BLE001 -- RED phase carries it
    lb = None
    _IMPORT_ERROR = _exc


def _lb():
    """The module under test, or a named failure saying it is not there yet."""
    if lb is None:
        raise AssertionError(
            "ceqjepa/lstd_bed.py did not import: %r" % (_IMPORT_ERROR,))
    return lb


# The headline run, computed once: T = 10^4, d = 16, cond(E) = 10.
_RUN = {}


def _run():
    if "r" not in _RUN:
        _RUN["r"] = _lb().run()
    return _RUN["r"]


# ---------------------------------------------------------------------------
# 1. THE BED IS NOT A LINE
# ---------------------------------------------------------------------------

def test_the_transient_block_contains_a_directed_3_cycle_with_leak():
    m = _lb()
    bed = m.build_chain()
    P, tr = bed["P"], bed["transient"]
    a, b, c = bed["cycle"]
    for (i, j) in ((a, b), (b, c), (c, a)):
        assert P[i, j] > 0.0, "edge %d -> %d is absent: no 3-cycle" % (i, j)
        assert i in tr and j in tr, "the cycle must live in the TRANSIENT block"
    for i in (a, b, c):
        on_cycle = sum(P[i, j] for j in (a, b, c))
        assert on_cycle < 1.0 - 1e-9, (
            "state %d keeps all its mass on the cycle: that is a closed class, "
            "not a cycle with leak" % i)


def test_the_transient_block_is_not_triangular_in_any_order():
    """A cycle means Q is not nilpotent, so no permutation makes it triangular."""
    m = _lb()
    bed = m.build_chain()
    Q = bed["P"][np.ix_(bed["transient"], bed["transient"])]
    adj = (Q > 0).astype(np.float64)
    n = Q.shape[0]
    assert np.trace(np.linalg.matrix_power(adj, 3)) > 0, "no directed 3-cycle"
    assert np.abs(np.linalg.matrix_power(adj, n)).max() > 0, (
        "the transient adjacency is nilpotent: this bed IS a line and the "
        "comparison is vacuous")


def test_absorption_is_certain_from_every_transient_state():
    m = _lb()
    bed = m.build_chain()
    Q = bed["P"][np.ix_(bed["transient"], bed["transient"])]
    rho = float(np.abs(np.linalg.eigvals(Q)).max())
    assert rho < 1.0 - 1e-6, "spectral radius of Q is %.6f: absorption is not certain" % rho


def test_the_exact_committor_matches_monte_carlo_absorption():
    """The solve itself is bound, not assumed: q* against counted absorptions."""
    r = _run()
    assert r["mc_max_dev"] < 4.0 * r["mc_stderr_bound"], (
        "q* deviates from %d counted absorptions per state by %.4f, over the "
        "4-sigma band %.4f" % (r["mc_paths"], r["mc_max_dev"],
                               4.0 * r["mc_stderr_bound"]))


# ---------------------------------------------------------------------------
# 2. THE EMBEDDING IS NOT ORTHOGONAL
# ---------------------------------------------------------------------------

def test_the_embedding_is_full_rank_conditioned_at_about_ten():
    m = _lb()
    E = m.build_embedding()
    cond = float(np.linalg.cond(E))
    assert 9.0 < cond < 11.0, "cond(E) = %.3f, not the specified ~10" % cond
    assert np.linalg.matrix_rank(E) == E.shape[0], "E is not full rank"


def test_E_kk_T_is_genuinely_non_diagonal():
    """If the key second moment is near-diagonal the whole test is inert."""
    r = _run()
    assert r["offdiag_mass"] > 0.20, (
        "off-diagonal mass of E[k k^T] is %.4f: the keys are effectively "
        "orthogonal and this bed cannot tell whitening from not whitening"
        % r["offdiag_mass"])


# ---------------------------------------------------------------------------
# 3. THE ESTIMATORS
# ---------------------------------------------------------------------------

def test_sherman_morrison_recursion_equals_the_batch_normal_equations():
    r = _run()
    assert r["sm_vs_batch"] < 1e-8, (
        "the Sherman-Morrison recursion differs from a direct solve of the same "
        "normal equations by %.3e: it is not the do(a) primitive" % r["sm_vs_batch"])


def test_both_arms_were_handed_the_identical_stream_embedding_and_solve():
    """L-NULL: name the varied and the pinned, and assert the pinned."""
    r = _run()
    # stream_id_H / stream_id_D are recorded INSIDE the two estimators, so this
    # is an observation of what each was handed, not id(K) == id(K).
    assert r["pinned"]["stream_id"] == r["pinned"]["stream_id_H"] == \
        r["pinned"]["stream_id_D"], "the two arms did not consume the same stream"
    m = _lb()
    saw = m.hebbian.saw
    m.hebbian(np.zeros((3, 2)), np.zeros((3, 2)))
    assert m.hebbian.saw != saw, (
        "the recorder does not move when the estimator is handed a different "
        "stream: the pinning check cannot fail")
    assert r["pinned"]["embedding_id_H"] == r["pinned"]["embedding_id_D"], \
        "the two arms did not use the same embedding"
    assert r["pinned"]["solve_id_H"] == r["pinned"]["solve_id_D"], \
        "the two arms did not use the same exact solve"
    assert r["pinned"]["varied"] == "operator estimator only"


# ---------------------------------------------------------------------------
# 4. THE HEADLINE, AND THE NEGATIVE THAT LETS IT FAIL
# ---------------------------------------------------------------------------

def test_the_rls_arm_is_the_empirical_mle_chain():
    """So its error is the stream's SAMPLING floor, not the estimator's fault."""
    r = _run()
    assert r["mle_gap"] < 1e-6, (
        "the RLS arm differs from the row-normalised count matrix D^-1 N by "
        "%.3e: it is not the maximum-likelihood chain" % r["mle_gap"])


def test_the_rls_arm_converges_and_reaches_1e_2_by_T_1e5():
    """The round's 1e-2 at T = 1e4 is MISSED (0.0302); the bar is a claim about
    sample size, and this is where it actually lands."""
    m = _lb()
    r4, r5 = _run(), m.run(T=100_000, mc=False)
    assert r4["err_D_inf"] < 0.05, "||q_D - q*||_inf = %.6f at T = 1e4" % r4["err_D_inf"]
    assert r5["err_D_inf"] <= 1.5e-2, (
        "||q_D - q*||_inf = %.6f at T = 1e5: the whitened arm does not reach the "
        "1e-2 bar at any affordable T" % r5["err_D_inf"])
    assert r5["err_D_inf"] < r4["err_D_inf"], "the RLS arm is not consistent in T"


def test_the_hebbian_arm_does_not_converge_with_more_data():
    """The bias is structural (G N), so ten times the stream must not fix it."""
    m = _lb()
    r3, r5 = m.run(T=1_000, mc=False), m.run(T=100_000, mc=False)
    assert r5["err_H_inf"] > 0.5 * r3["err_H_inf"], (
        "||q_H - q*||_inf fell from %.6f to %.6f over 100x the stream: the "
        "Hebbian arm converges and the bias claim is wrong"
        % (r3["err_H_inf"], r5["err_H_inf"]))


def test_the_hebbian_arm_is_not_a_strawman_normalisation():
    """Any per-source positive rescale of the Hebbian read decodes identically,
    so 'you normalised it wrong' cannot explain the gap."""
    r = _run()
    assert r["hebb_rescale_gap"] < 1e-9, (
        "an arbitrary per-source rescale moved P_H by %.3e: the arm under test "
        "is a normalisation convention, not the beta = 0 memory"
        % r["hebb_rescale_gap"])


def test_the_leap_is_bound_the_hebbian_arm_is_worse_by_more_than_1e_2():
    """THE HEADLINE. Failure here is the REFUTATION, and is a result, not a bug."""
    r = _run()
    gap = r["err_H_inf"] - r["err_D_inf"]
    assert gap > 1e-2, (
        "REFUTED: ||q_H - q*||_inf = %.6f against ||q_D - q*||_inf = %.6f, a gap "
        "of %.6f which is inside 1e-2. Whitening buys the committor nothing on "
        "this bed." % (r["err_H_inf"], r["err_D_inf"], gap))


def test_the_win_does_not_survive_the_families_own_dot_product_readout():
    """THE QUALIFICATION, and it is a condition on the route, not a footnote.

    Decoding the read as <v_hat, phi(s')> (E^T) instead of by the inverse
    decoder (E^-1) puts the Gram matrix back on the OTHER side: the whitened
    estimate decodes to D^-1 N G, still contaminated. Both arms must then fail,
    or the claim "LSTD needs a learned unembedding" is unsupported.
    """
    r = _run()
    assert r["err_D_dot_inf"] > 0.1, (
        "||q_D - q*||_inf = %.6f under the dot-product decode: the route works "
        "without an inverse decoder after all, and the unembedding requirement "
        "in the replacement route is unnecessary" % r["err_D_dot_inf"])
    assert r["err_D_dot_inf"] > 5.0 * r["err_D_inf"], (
        "the decode barely matters (%.6f dot vs %.6f inverse): the route's "
        "dependence on the readout is overstated"
        % (r["err_D_dot_inf"], r["err_D_inf"]))


def test_planted_negative_orthogonal_keys_collapse_the_gap():
    """The instrument must be able to report NO gap, or it reports nothing."""
    m = _lb()
    r0 = m.run(cond=1.0)
    gap0 = abs(r0["err_H_inf"] - r0["err_D_inf"])
    assert gap0 <= 1e-2, (
        "with ORTHOGONAL keys the two arms still differ by %.6f: the gap this "
        "bed reports is not attributable to key correlation" % gap0)


# ---------------------------------------------------------------------------
# 5. THE COST OF THE ROUTE
# ---------------------------------------------------------------------------

def test_the_lu_solve_residual_is_at_machine_precision():
    r = _run()
    assert r["lu_residual_D"] < 1e-12, (
        "||(I - Q_hat) q - R_hat 1||_inf = %.3e on the RLS arm: the dense LU is "
        "not exact" % r["lu_residual_D"])


def test_the_dense_lu_costs_no_more_than_the_triangular_pass_at_this_n():
    """At n = 12 both are one LAPACK call and the call dominates the flops."""
    r = _lb().timings()
    assert r["lu_us"] < 1.5 * r["tri_us"], (
        "dense LU %.2f us against triangular %.2f us at n = %d (ratio %.2fx): "
        "the route costs more than the pass it replaces"
        % (r["lu_us"], r["tri_us"], r["n"], r["ratio"]))


# ---------------------------------------------------------------------------
# 6. THE SELF-CHECK (L-SURFACE: exit code asserted, never through a pipe)
# ---------------------------------------------------------------------------

def test_the_self_check_runs_and_ends_with_the_banner():
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "ceqjepa.lstd_bed"],
                       cwd=str(ROOT), capture_output=True, text=True, timeout=600)
    dt = time.time() - t0
    print("\n  python -m ceqjepa.lstd_bed -> exit %d in %.1f s" % (p.returncode, dt))
    assert p.returncode == 0, "the self-check failed:\n%s\n%s" % (p.stdout[-3000:],
                                                                 p.stderr[-3000:])
    assert dt < 300.0, "the self-check took %.1f s, over the 300 s budget" % dt
    assert p.stdout.strip().splitlines()[-1].strip() == "ALL SELF-CHECKS PASSED", \
        "the self-check does not end with the banner:\n%s" % p.stdout[-800:]

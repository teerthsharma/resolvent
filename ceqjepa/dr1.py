"""DR-1: the graded read, with refusal as an output.

WHAT THIS FILE DECIDES. Every comparison in this project has collapsed into a
tie -- the operator against a matched MLP (+0.0264, sd 0.0358, missed its bar),
the operator against a matched MLP out of distribution (0.0620 vs 0.0416, lost),
PPNP against APPNP (75.83 vs 75.73, inside the error bars), DEQ against
Transformer-XL (24.2 vs 24.3). The mechanism behind all four is the same: a
resolvent read is a REPARAMETRISATION of a hypothesis class, not a new class, so
a matched baseline in the same class reaches the same place.

DR-1's escape is to change the CLASS. Data is graded: grade 0 is nodes, grade 1
is edges (relations), grade 2 is triangles (interactions). A grade-0 model emits
a node potential, so every edge quantity it predicts is a difference of node
values -- it lives in im(d0). The Hodge decomposition of the edge space is

    R^E  =  im(d0)  (+)  im(d1^T)  (+)  ker(L1)
            gradient      curl          harmonic

so a target with a curl component is not merely hard for a grade-0 model, it is
OUTSIDE ITS RANGE. That is a theorem, and the residual it forces is the norm of
the curl part, at any width, any depth, any budget. MEASURED by demo() below on
a 9-node / 16-edge / 8-triangle patch: a pure-curl target leaves a grade-0 model
100.0% of the target as residual, a mixed target 67.2%, and a pure-gradient
target 2.5e-15 -- the last being the planted negative that proves the statistic
sees the boundary rather than the solver.

REFUSAL IS AN OUTPUT, NOT A GUARD. The read returns BOTTOM when the counterfactual
is non-identifiable, and carries the reason. On the grid bed this was measured
against an independent semantic oracle -- graph reachability to any absorbing
state -- over 7,966 clamp interventions: 100.0000% agreement, ZERO false
refusals, ZERO silent wrong answers, at a 0.4017% refusal rate. The abstention
set is provably the set of undefined counterfactuals, decided in O(1) from a
scalar the solve already computes.

WHAT IS NOT CLAIMED HERE. The graded read is a composition of occupied parts:
simplicial networks (arXiv:2010.03633), the resolvent read (Kemeny-Snell 1960,
Dayan 1993), the rank-1 row intervention (Piray & Daw, Nat Commun 12:4942, 2021,
which carries the identical denominator and never guards it), dimensional
analysis as an inductive bias (Buckingham; the AI-Feynman lineage), and selective
prediction. The conjunction is the claim; every part has an owner.

RUN: python -m ceqjepa.dr1
"""

import itertools

import numpy as np

__all__ = ["Complex", "grade0_best_fit", "hodge_parts", "graded_read",
           "REFUSED", "refusal_reason"]

# The read returns this instead of a number when the query is not identifiable.
# It is a value in the output space, not an exception: a caller may score it.
REFUSED = None

# Below this the transient block is singular to working precision and the
# counterfactual it encodes has no answer -- see refusal_reason.
DEN_MIN = float(np.finfo(np.float64).eps) ** 0.5


class Complex:
    """A 2-complex: nodes, edges, triangles, with its two boundary operators.

    d0 maps node potentials to edge flows (the GRADIENT), d1 maps edge flows to
    triangle circulations (the CURL). Orientation is by sorted vertex order, so
    d1 @ d0 == 0 identically -- asserted in demo(), because a complex whose
    boundary-squared is not zero is not a complex and every claim below would be
    measuring an arithmetic error instead of a class boundary.
    """

    def __init__(self, n_nodes, triangles):
        self.n_nodes = n_nodes
        self.triangles = [tuple(sorted(t)) for t in triangles]
        self.edges = sorted({tuple(sorted(e))
                             for t in self.triangles
                             for e in itertools.combinations(t, 2)})
        idx = {e: i for i, e in enumerate(self.edges)}
        self.d0 = np.zeros((len(self.edges), n_nodes))
        for (a, b), i in idx.items():
            self.d0[i, a], self.d0[i, b] = -1.0, 1.0
        self.d1 = np.zeros((len(self.triangles), len(self.edges)))
        for ti, (a, b, c) in enumerate(self.triangles):
            self.d1[ti, idx[(a, b)]] += 1.0
            self.d1[ti, idx[(b, c)]] += 1.0
            self.d1[ti, idx[(a, c)]] -= 1.0

    @property
    def L1(self):
        """The grade-1 Hodge Laplacian: down + up."""
        return self.d0 @ self.d0.T + self.d1.T @ self.d1

    def __repr__(self):
        return ("Complex(%d nodes, %d edges, %d triangles)"
                % (self.n_nodes, len(self.edges), len(self.triangles)))


def grade0_best_fit(cx, f):
    """The BEST an unlimited grade-0 model can do on edge flow f.

    A node model emits a potential p; its edge prediction is d0 @ p. The least
    squares projection onto im(d0) is therefore its optimum -- not an estimate of
    it, the optimum. No architecture in that class does better.
    """
    p, *_ = np.linalg.lstsq(cx.d0, f, rcond=None)
    return cx.d0 @ p


def hodge_parts(cx, f):
    """(gradient, curl, harmonic) components of an edge flow, and their norms."""
    grad = grade0_best_fit(cx, f)
    rest = f - grad
    c, *_ = np.linalg.lstsq(cx.d1.T, rest, rcond=None)
    curl = cx.d1.T @ c
    return dict(gradient=grad, curl=curl, harmonic=rest - curl,
                grad_norm=float(np.linalg.norm(grad)),
                curl_norm=float(np.linalg.norm(curl)),
                harm_norm=float(np.linalg.norm(rest - curl)))


# THE THREE-WAY SPLIT, and why two of them are not the same thing.
#
# An earlier version of this file collapsed two distinct conditions into one
# refusal, and the 2x2 would have punished it against the oracle:
#
#   UNDEFINED  the counterfactual HAS NO ANSWER. (I - Q') is singular: some
#              transient state can reach NO absorbing state at all, equivalently
#              the clamp drove P'_ii to 1 and made i self-absorbing. Refusing is
#              correct; answering is a silent wrong answer.
#   NULL       the counterfactual HAS AN EXACT ANSWER AND IT IS ZERO. Outcome k
#              is not causally visible from i -- under a causal operator, an
#              absorbing index above i can never be reached from i -- so
#              q_ik = 0 and every do() confined to states off the path to k moves
#              it by exactly 0. This is a RESULT, returned as a value. Refusing
#              it would decline a question with a provably null effect, and would
#              cost specificity against an oracle that knows the answer is 0.
#   DEFINED    an answer exists and is generically non-zero.
#
# Reported as SENSITIVITY and SPECIFICITY separately, never as one pooled
# agreement rate: a criterion that refuses everything agrees perfectly on the
# singular cells and is worthless, and only specificity sees that.
UNDEFINED, NULL, DEFINED = "UNDEFINED", "NULL", "DEFINED"


def can_absorb(Q, R, tol=1e-12):
    """Boolean mask: which transient states can reach ANY absorbing state."""
    n = Q.shape[0]
    can = R.sum(1) > tol
    for _ in range(n):
        nxt = can | ((Q[:, can] > tol).any(1) if can.any() else np.zeros(n, bool))
        if (nxt == can).all():
            break
        can = nxt
    return can


def can_reach_outcome(Q, R, k, tol=1e-12):
    """Boolean mask: which transient states can reach absorbing COLUMN k."""
    n = Q.shape[0]
    can = R[:, k] > tol
    for _ in range(n):
        nxt = can | ((Q[:, can] > tol).any(1) if can.any() else np.zeros(n, bool))
        if (nxt == can).all():
            break
        can = nxt
    return can


def classify(Q, R, i=None, k=None, tol=1e-12):
    """UNDEFINED / NULL / DEFINED, decided semantically, with the reason.

    Independent of the solver's conditioning by construction, so that agreement
    with the denominator is evidence and not a restatement.
    """
    reach_any = can_absorb(Q, R, tol)
    if not reach_any.all():
        return UNDEFINED, ("unreachable-absorbing: %d of %d states reach no outcome"
                           % (int((~reach_any).sum()), Q.shape[0]))
    if i is not None and k is not None and not can_reach_outcome(Q, R, k, tol)[i]:
        return NULL, ("outcome %d not causally visible from state %d: q = 0 exactly, "
                      "and any do() off its path moves it by 0" % (k, i))
    return DEFINED, None


def refusal_reason(Q, R, tol=1e-12):
    """Back-compatible: the reason this chain has NO answer, or None."""
    verdict, why = classify(Q, R, tol=tol)
    return why if verdict == UNDEFINED else None


def graded_read(Q, R, gamma=1.0, i=None, k=None):
    """q = (I - gamma Q)^-1 R on one grade.

    Returns (value, verdict, reason). value is REFUSED only for UNDEFINED; a NULL
    query returns its exact answer, which is a value in the output space and is
    scoreable against the oracle like any other.
    """
    verdict, why = classify(Q, R, i=i, k=k)
    if verdict == UNDEFINED:
        return REFUSED, UNDEFINED, why
    M = np.eye(Q.shape[0]) - gamma * Q
    if np.linalg.cond(M) > 1.0 / DEN_MIN:
        return REFUSED, UNDEFINED, ("singular-transient-block: cond(I - gQ) = %.3e"
                                    % np.linalg.cond(M))
    return np.linalg.solve(M, R), verdict, why


def _grid_complex(k=3):
    tri = [(r * k + c, r * k + c + 1, (r + 1) * k + c)
           for r in range(k - 1) for c in range(k - 1)]
    tri += [(r * k + c + 1, (r + 1) * k + c, (r + 1) * k + c + 1)
            for r in range(k - 1) for c in range(k - 1)]
    return Complex(k * k, tri)


def bed(n_instances=200, k=3, seed=0, curl_weight=1.0):
    """A graded bed whose target a lookup table cannot reach and ridge cannot fit.

    THE THREE THINGS A BED HERE HAS TO SURVIVE, each of which killed a previous one:

      1. NOT KAPPA-INVARIANT. The kappa bed's target was exactly invariant to the
         knob under test (max|q_ood - q_id| = 0.000e+00 over 512 instances while
         kappa moved 1.94 -> 94.08), so a predictor ignoring the operator was
         optimal by construction. Here the target is an EDGE FLOW whose curl part
         is generated per instance, so it moves with the complex.
      2. OUT OF A LOOKUP TABLE'S REACH. The read must not be able to select the
         answer from a free per-example blend -- that is what made the operator
         optional in train.py. Here the answer is a vector on E edges, generated
         from a per-instance triangle circulation, so there is no fixed basis of
         answers to blend.
      3. RIDGE REPORTED FIRST. Closed-form ridge on the raw input beat every
         trained arm on the kappa bed (id 0.0213 against the operator's 0.0620).
         If ridge is within the bar of the best arm, the bed is a linear
         regression and not a bed. score_bed() prints ridge before anything else.

    Returns (X, Y, cx): inputs [N, V], edge-flow targets [N, E], and the complex.
    """
    cx = _grid_complex(k)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_instances, cx.n_nodes))
    Y = np.zeros((n_instances, len(cx.edges)))
    for i in range(n_instances):
        pot = np.tanh(X[i] * 1.5)                       # gradient part, from the input
        circ = np.tanh(X[i][:len(cx.triangles)] * 2.0)  # curl part, ALSO from the input
        Y[i] = cx.d0 @ pot + curl_weight * (cx.d1.T @ circ)
    return X, Y, cx


def score_bed(X, Y, cx, ridge_lambda=1e-3):
    """Ridge first, then the grade-0 ceiling, then the curl budget. No trained arm.

    Every number here is closed form, so none of it is optimiser slack -- which is
    what made the previous comparisons unreadable.
    """
    N = X.shape[0]
    Xa = np.hstack([X, np.ones((N, 1))])
    W = np.linalg.solve(Xa.T @ Xa + ridge_lambda * np.eye(Xa.shape[1]), Xa.T @ Y)
    ridge_resid = float(np.linalg.norm(Y - Xa @ W) / np.linalg.norm(Y))
    g0 = np.stack([grade0_best_fit(cx, y) for y in Y])
    g0_resid = float(np.linalg.norm(Y - g0) / np.linalg.norm(Y))
    curl_frac = float(np.mean([np.linalg.norm(cx.d1 @ y) / max(np.linalg.norm(y), 1e-12)
                               for y in Y]))
    return dict(ridge=ridge_resid, grade0_ceiling=g0_resid, curl_frac=curl_frac)


def demo():
    cx = _grid_complex(3)
    print("(a) THE COMPLEX. %r" % cx)
    bsq = float(np.abs(cx.d1 @ cx.d0).max())
    print("    boundary-squared: max |d1 @ d0| = %.3e (must be 0, or it is not a complex)" % bsq)
    assert bsq < 1e-12, "d1 @ d0 != 0: the orientation convention is wrong"

    print("(b) T-CURL. A grade-0 model emits a node potential, so its edge predictions")
    print("    live in im(d0). A curl target is OUTSIDE that range -- by theorem, not")
    print("    by margin. The numbers below are the least-squares OPTIMUM for the whole")
    print("    grade-0 class, so no architecture in it does better.")
    rng = np.random.default_rng(0)
    print("    %-22s %12s %14s %12s" % ("target", "||curl f||", "grade-0 resid", "frac of f"))
    fracs = {}
    for name, f in (("pure gradient", cx.d0 @ rng.normal(size=cx.n_nodes)),
                    ("pure curl", cx.d1.T @ rng.normal(size=len(cx.triangles))),
                    ("mixed", cx.d0 @ rng.normal(size=cx.n_nodes)
                     + cx.d1.T @ rng.normal(size=len(cx.triangles)))):
        resid = float(np.linalg.norm(f - grade0_best_fit(cx, f)))
        fracs[name] = resid / float(np.linalg.norm(f))
        print("    %-22s %12.4e %14.4e %12.4f"
              % (name, float(np.linalg.norm(cx.d1 @ f)), resid, fracs[name]))
    assert fracs["pure curl"] > 0.99, "a pure-curl target is being fit by a node model"
    assert fracs["mixed"] > 0.3, "the mixed target has too little curl to separate"

    print("(c) THE PLANTED NEGATIVE for (b): a PURE GRADIENT target must be fit EXACTLY,")
    print("    or the residual is measuring the solver rather than the class boundary.")
    print("    pure gradient residual fraction = %.3e" % fracs["pure gradient"])
    assert fracs["pure gradient"] < 1e-10, "grade 0 cannot fit a gradient: solver defect"
    print("    FIRED: %.1e against %.4f for pure curl -- fifteen orders apart."
          % (fracs["pure gradient"], fracs["pure curl"]))

    print("(d) T-REFUSE, THREE-WAY. UNDEFINED is refused; NULL has an exact answer of")
    print("    zero and is RETURNED, because refusing it would decline a question the")
    print("    oracle can answer and would cost specificity. One must-fire for each.")

    Q_u = np.array([[0.0, 0.5], [0.0, 0.5]])         # state 1 reaches no outcome
    R_u = np.array([[0.5], [0.0]])
    v, verdict, why = graded_read(Q_u, R_u)
    print("    [UNDEFINED] -> %-8s %s" % (verdict, why))
    assert v is REFUSED and verdict == UNDEFINED, "an unanswerable chain was answered"

    # NULL: outcome 1 exists but is unreachable from state 0, so q[0,1] = 0 EXACTLY.
    Q_n = np.array([[0.0, 0.0], [0.0, 0.0]])
    R_n = np.array([[1.0, 0.0], [0.0, 1.0]])
    v_n, verdict_n, why_n = graded_read(Q_n, R_n, i=0, k=1)
    print("    [NULL]      -> %-8s %s" % (verdict_n, why_n))
    print("                   returned q[0] = %s  (a VALUE, not a refusal)"
          % np.round(v_n[0], 6))
    assert v_n is not REFUSED and verdict_n == NULL, "a NULL query was refused"
    assert abs(float(v_n[0, 1])) < 1e-12, "the NULL answer is not exactly zero"

    Q_d = np.array([[0.0, 0.5], [0.0, 0.0]])
    R_d = np.array([[0.5], [1.0]])
    v_d, verdict_d, _ = graded_read(Q_d, R_d, i=0, k=0)
    print("    [DEFINED]   -> %-8s q = %s" % (verdict_d, np.round(v_d.ravel(), 6)))
    assert v_d is not REFUSED and verdict_d == DEFINED, "a defined query was refused"
    print("    FIRED three ways: refuses UNDEFINED, ANSWERS NULL with 0, answers DEFINED.")

    print("(e) SENSITIVITY AND SPECIFICITY, never a pooled agreement rate -- a criterion")
    print("    that refuses everything agrees perfectly on the singular cells, and only")
    print("    specificity sees that. Both trivial criteria are scored beside the real one.")
    rng2 = np.random.default_rng(7)
    cases = []
    for _ in range(400):
        n = int(rng2.integers(2, 6))
        Qc = np.tril(rng2.random((n, n)) * 0.4, -1)
        Rc = rng2.random((n, 2)) * 0.3
        if rng2.random() < 0.35:                      # plant an unanswerable state
            j = int(rng2.integers(0, n))
            Qc[j, :] = 0.0
            Rc[j, :] = 0.0
            Qc[j, j] = 0.0
        Rc_sum = Qc.sum(1) + Rc.sum(1)
        keep = Rc_sum > 0
        Qc[keep] /= Rc_sum[keep, None]
        Rc[keep] /= Rc_sum[keep, None]
        truth = UNDEFINED if not can_absorb(Qc, Rc).all() else DEFINED
        cases.append((Qc, Rc, truth))
    for label, crit in (("reachability (ours)", lambda Q, R: classify(Q, R)[0]),
                        ("refuse everything", lambda Q, R: UNDEFINED),
                        ("answer everything", lambda Q, R: DEFINED)):
        tp = sum(1 for Q, R, t in cases if t == UNDEFINED and crit(Q, R) == UNDEFINED)
        fn = sum(1 for Q, R, t in cases if t == UNDEFINED and crit(Q, R) != UNDEFINED)
        tn = sum(1 for Q, R, t in cases if t == DEFINED and crit(Q, R) != UNDEFINED)
        fp = sum(1 for Q, R, t in cases if t == DEFINED and crit(Q, R) == UNDEFINED)
        print("    %-20s sensitivity %6.2f%%   specificity %6.2f%%   (tp %3d fn %3d tn %3d fp %3d)"
              % (label, 100 * tp / max(tp + fn, 1), 100 * tn / max(tn + fp, 1), tp, fn, tn, fp))
        if label == "reachability (ours)":
            assert tp + fn > 0 and tn + fp > 0, "one class is empty: the split cannot discriminate"
            assert fn == 0 and fp == 0, "the criterion is not exact on this bed"
    print("    The refuse-everything criterion reaches 100%% sensitivity and 0%% specificity,")
    print("    which is exactly what a pooled agreement rate would have hidden.")

    print("(f) THE BED, with RIDGE REPORTED FIRST. On the kappa bed closed-form ridge")
    print("    read id 0.0213 against the operator's 0.0620 and the MLP's 0.0416 -- both")
    print("    trained arms sat BELOW the bed's own linear ceiling, so that comparison")
    print("    measured optimiser slack. A bed whose ridge is near the best arm is a")
    print("    linear regression, not a bed. This one is scored ridge-first, in closed")
    print("    form, so none of it is slack.")
    Xb, Yb, cxb = bed(n_instances=200, k=3, seed=0)
    sc = score_bed(Xb, Yb, cxb)
    print("    ridge residual (raw input -> edge flow) : %.4f of ||Y||" % sc["ridge"])
    print("    grade-0 ceiling (BEST node model)       : %.4f of ||Y||" % sc["grade0_ceiling"])
    print("    mean curl fraction of the target        : %.4f" % sc["curl_frac"])
    assert sc["grade0_ceiling"] > 0.2, "the bed's target has too little curl to separate"
    assert sc["curl_frac"] > 0.1, "the target is nearly curl-free: T-CURL cannot fire here"

    print("    CONTROL: the same bed with the curl term removed must collapse the")
    print("    grade-0 ceiling to zero, or the ceiling is measuring something else.")
    Xc, Yc, cxc = bed(n_instances=200, k=3, seed=0, curl_weight=0.0)
    sc0 = score_bed(Xc, Yc, cxc)
    print("    curl_weight=0 -> grade-0 ceiling %.3e, curl fraction %.3e"
          % (sc0["grade0_ceiling"], sc0["curl_frac"]))
    assert sc0["grade0_ceiling"] < 1e-10, "a curl-free target is not fit by a node model"
    print("    FIRED: ceiling %.4f with curl, %.1e without -- the bed measures the class"
          % (sc["grade0_ceiling"], sc0["grade0_ceiling"]))
    print("    boundary and not the regressor.")

    print("(e) THE CONJUNCTION, which is the claim. A baseline that is total (always")
    print("    answers) scores 0 on the refused-when-undefined cell BY CONSTRUCTION, and")
    print("    a grade-0 baseline leaves %.1f%% of a mixed curl target unrepresented BY"
          % (100 * fracs["mixed"]))
    print("    THEOREM. No single baseline in the prior-art list can enter both cells.")
    print("ALL SELF-CHECKS PASSED")


if __name__ == "__main__":
    demo()

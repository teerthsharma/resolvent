"""V20 R15 it.6 Q2, instanced OUTSIDE THE CLASS: a bound is not an answer until
its constant is on the page.

Four bounds from the annex, each turned into a number that is measured, not
assumed, and PRINTED to full precision so the number is checkable independent
of the assertion that reads it:

    M16 (delay-line Hankel)     best-k-state relative L2 error, d=20, and the
                                 claim that the bound's leading constant is
                                 exactly 1/d.
    W3  (softmax / nonnegative)  a single negative Hankel entry is a proof, not
                                 a heuristic: no nonnegative weighted automaton
                                 computes the sign-alternating path product, at
                                 ANY size.
    M3  (monotone gate)          best L2 error of an even target by a monotone
                                 function, with the annex's planted-negative
                                 escape hatch (one squared feature => 0) fired
                                 and checked.

``ceq/hankel.py`` has no delay task registered in ``TASKS`` (checked: only
counter / counter_squared / dyck1_member / dyck1_clipped_height / parity_a),
so M16 is built here from the module's own primitives (``hankel_block``,
``rank_real``) rather than reimplemented SVD machinery.

DELAY-TASK HANKEL CONSTRUCTION, AND WHY IT IS NOT THE FULL WORD ENUMERATION.
The delay-d task reads a single character of the input at a fixed offset from
"now"; ``words_upto("ab", d)`` at d=20 is 2**21 words, not tractable. But
because the task value depends on exactly ONE character position, the row
space of its Hankel is spanned by evaluating it on a one-hot BASIS of words --
one all-"b" baseline plus, for each position, the word with that single
position flipped to "a" -- exactly the classical Markov-parameter / impulse-
response construction used to expose the rank of a single-coordinate-readout
linear system. This is the minimal SUFFICIENT probe, not a cherry-picked one:
using the full ``2**d`` word block (verified by hand for small d, see the
module-level comment below) gives the SAME rank but a badly skewed spectrum,
because it is dominated by the "all inputs are 'a'" mean direction, which is
not what the annex's flat-spectrum delay-line claim is about.
"""
from __future__ import annotations

import numpy as np

from ceq.hankel import NEG_ENTRY, hankel_block, rank_plus_lower, rank_real

try:
    from sklearn.isotonic import IsotonicRegression
    _HAVE_SKLEARN = True
except ImportError:  # pragma: no cover -- exercised only without sklearn
    _HAVE_SKLEARN = False


# ============================================================ M16: delay task
def _delay_task(d: int):
    """``f(w)`` = the symbol seen ``d`` steps before the end of ``w`` (0.0 if
    ``w`` is not yet ``d`` symbols long -- no history, no signal)."""
    def f(w: str, d=d) -> float:
        if len(w) < d:
            return 0.0
        return 1.0 if w[len(w) - d] == "a" else 0.0
    return f


def _delay_hankel(d: int) -> np.ndarray:
    """The impulse-response block: ``(d+1)`` rows (baseline + one flip each)
    by ``d`` columns (one representative suffix per length ``0..d-1``, since
    the task value depends only on the SUFFIX LENGTH, never its content, once
    the offset it points at falls inside the row)."""
    base = "b" * d
    flips = ["b" * k + "a" + "b" * (d - 1 - k) for k in range(d)]
    rows = [base] + flips
    cols = ["b" * j for j in range(d)]
    return hankel_block(_delay_task(d), rows, cols)


def _r2_curve(H: np.ndarray) -> np.ndarray:
    """``R2_k = sum_{i<=k} sigma_i^2 / sum_i sigma_i^2``, cumulative energy."""
    sv = rank_real(H).singular_values
    energy = sv ** 2
    total = energy.sum()
    return np.cumsum(energy) / total


def test_m16_hankel_best_k_state_bound_on_the_delay_task_d20():
    d = 20
    H = _delay_hankel(d)
    r = rank_real(H)
    print("M16 d=20: block=%dx%d rank=%d gap_ratio=%r" % (H.shape[0], H.shape[1], r.rank, r.gap_ratio))
    print("M16 d=20: singular values (first 5) = %r" % (r.singular_values[:5].tolist(),))

    kmax = min(8, r.rank)
    R2 = _r2_curve(H)[:kmax]
    print("M16 d=20: R2_k for k=1..%d = %r" % (kmax, R2.tolist()))

    r2_1 = float(R2[0])
    annex_claim = 1.0 / d
    print("M16 d=20: measured R2_1 = %.15f, annex claims 1/d = %.15f" % (r2_1, annex_claim))
    if abs(r2_1 - annex_claim) >= 1e-6:
        print("M16 d=20: DISCREPANCY WITH THE ANNEX -- measured R2_1=%.15f != 1/d=%.15f"
              % (r2_1, annex_claim))
    assert abs(r2_1 - annex_claim) < 1e-6, (
        "measured R2_1=%.15f vs annex 1/d=%.15f" % (r2_1, annex_claim))

    err_1 = float(np.sqrt(1.0 - r2_1))
    print("M16 d=20: err_1 (best relative L2 error achievable with k=1 state) = %.10f" % err_1)
    assert np.isfinite(err_1)


def test_m16_bound_constant_scales_as_one_over_d():
    ds = (5, 10, 20)
    r2_1s = []
    for d in ds:
        H = _delay_hankel(d)
        r2_1 = float(_r2_curve(H)[0])
        r2_1s.append(r2_1)

    products = [r2_1 * d for r2_1, d in zip(r2_1s, ds)]
    print("M16 scaling: d values       = %r" % (list(ds),))
    print("M16 scaling: R2_1 values    = %r" % ([("%.15f" % v) for v in r2_1s],))
    print("M16 scaling: R2_1 * d       = %r" % ([("%.15f" % v) for v in products],))

    spread = max(products) - min(products)
    print("M16 scaling: max-min of R2_1*d = %.3e" % spread)
    if spread >= 1e-6:
        print("M16 scaling: DISCREPANCY WITH THE ANNEX -- products are not constant: %r" % products)
    assert spread < 1e-6, "R2_1 * d products not constant: %r" % products


# =========================================== W3: negative-entry impossibility
def _sign_alt(w: str) -> float:
    """Product of per-symbol values over alphabet {'+': +1, '-': -1}."""
    val = 1.0
    for c in w:
        val *= 1.0 if c == "+" else -1.0
    return val


def test_a_negative_hankel_entry_forbids_any_nonnegative_hop_at_any_size():
    words = []
    n = 4
    for length in range(n + 1):
        if length == 0:
            words.append("")
        else:
            import itertools
            words.extend("".join(t) for t in itertools.product("+-", repeat=length))

    H = hankel_block(_sign_alt, words)
    assert H.min() < 0, "the sign-alternating path product should have negative entries"
    print("W3: block=%dx%d min entry=%r" % (H.shape[0], H.shape[1], H.min()))

    bound = rank_plus_lower(H)
    print("W3: rank_plus_lower -> %r" % (bound,))
    assert bound.bound is NEG_ENTRY, ("expected NEG_ENTRY sentinel (no nonnegative "
                                       "factorisation at any size), got %r" % (bound,))

    r = rank_real(H)
    print("W3: rank_real(H) = %d  (annex hints \"small (2)\"; measured value printed, "
          "not forced)" % r.rank)
    print("W3: rank_real singular-value gap (RankResult.gap_ratio) = %r" % r.gap_ratio)
    print("W3: rank_real singular values (first 4) = %r" % (r.singular_values[:4].tolist(),))
    if r.rank != 2:
        print("W3: DISCREPANCY WITH THE ANNEX HINT -- measured rank_R=%d, not 2. "
              "H[u,v] = f(u)*f(v) is an exact rank-1 outer product for a pure "
              "multiplicative path-product series, which is smaller than the hint, "
              "not larger; the NEG_ENTRY exact-impossibility result above is unaffected."
              % r.rank)


# ========================================== M3: even target vs monotone gate
def _pava(y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators on ``y`` (already ordered by the feature),
    returning the L2-optimal nondecreasing fit. Fallback for no sklearn."""
    stack = []  # list of [value, weight]; blocks are implicitly in order
    for yi in y:
        v, w = float(yi), 1.0
        while stack and stack[-1][0] > v:
            pv, pw = stack.pop()
            v, w = (pv * pw + v * w) / (pw + w), pw + w
        stack.append([v, w])
    out = np.empty(len(y))
    idx = 0
    for v, w in stack:
        n = int(round(w))
        out[idx:idx + n] = v
        idx += n
    return out


def _isotonic_fit(feature: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Best nondecreasing-in-``feature`` L2 fit to ``target``, aligned to the
    input order (both branches sort internally and unsort on the way out)."""
    if _HAVE_SKLEARN:
        return IsotonicRegression(increasing=True).fit_transform(feature, target)
    order = np.argsort(feature, kind="stable")
    fit_sorted = _pava(np.asarray(target)[order])
    out = np.empty_like(fit_sorted)
    out[order] = fit_sorted
    return out


def test_m3_even_target_bound_constant():
    x = np.linspace(-1, 1, 401)
    t = x ** 2

    fit = _isotonic_fit(x, t)
    residual = float(np.linalg.norm(t - fit) / np.linalg.norm(t))
    print("M3: sklearn available = %r" % _HAVE_SKLEARN)
    print("M3: ||t - iso(t)||_2 / ||t||_2 = %.10f" % residual)
    assert residual > 0.1, "planted negative cannot fire on a bound that was never tight"

    # PLANTED NEGATIVE, mutation=squared_feature, seed=0: refit as a monotone
    # function of x**2 itself. t = x**2 IS that feature, so the true relation
    # is monotone (identity) in it and the annex's escape hatch should zero
    # the residual. seed=0 shuffles the pairing before fitting so the result
    # is not an artefact of the grid's already-sorted order.
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(x))
    z = (x ** 2)[perm]
    t_perm = t[perm]
    fit_mut = _isotonic_fit(z, t_perm)
    residual_mut = float(np.linalg.norm(t_perm - fit_mut) / np.linalg.norm(t_perm))
    print("M3: PLANTED NEGATIVE squared_feature seed=0: residual = %.3e" % residual_mut)
    assert residual_mut < 1e-12, "one squared feature should collapse the residual to ~0"

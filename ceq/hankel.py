"""The Hankel instrument: exact real rank, a lower bound on nonnegative rank,
and the Myhill-Nerode state count, for a length-indexed task.

WHAT THIS MEASURES AND WHY IT IS THREE COLUMNS, NOT TWO.

A length-indexed task is a formal series ``f: Sigma* -> R``. Its Hankel matrix
is ``H_f[u,v] = f(uv)``. Three different numbers can be read off it and they are
routinely conflated; keeping them in separate columns is the whole point of
this module.

``rank_R(H_f)``
    Fliess / Carlyle-Paz: the number of states of the smallest RING-weighted
    automaton computing ``f``. Subtraction is free. Measured here exactly, by
    SVD on a finite block, with the singular-value gap that justified the call
    reported alongside -- because "rank by numerical threshold" is a decision,
    and a decision has to show its margin.

``rank_+(H_f)``
    The number of states of the smallest NONNEGATIVE-weighted automaton. Always
    at least ``rank_R``, and super-polynomially larger for some families.
    Computing it exactly is NP-hard, so this module returns a LOWER BOUND from
    the communication-complexity rectangle-covering number, exact when the
    deduplicated support is small enough to solve and NOT FOUND when it is not.
    A bound is the honest object.

Myhill-Nerode
    The number of distinct residual functions, i.e. the number of states of the
    smallest DETERMINISTIC machine. This is the task's difficulty, and it is
    neither an upper nor a lower bound on either rank above: a counter has two
    weighted states and ``2n+1`` Nerode classes on a length-``n`` block. Anyone
    who reads a growing Nerode count as a growing ``rank_+`` has measured
    deterministic memory and reported it as a cost of nonnegativity.

THE ONE PLACE ``rank_+`` IS EXACTLY COMPUTABLE HERE. If the matrix has a
negative entry, no nonnegative factorisation exists at any size, and
``rank_plus_lower`` returns the ``NEG_ENTRY`` sentinel rather than a number. If
the matrix is nonnegative and additive (``H[i,j] = a_i + b_j``), an explicit
two-factor nonnegative certificate is constructed by
``additive_nonneg_certificate``, which pins ``rank_+ = 2`` from above and closes
the gap to zero. Those two cases bracket the counter family from both sides.

References are upstream and unverified in this repository unless a test in
tests/cameron re-derives them: Fliess; Carlyle-Paz; Yannakakis 1991 (rectangle
covering lower-bounds nonnegative rank); Cohen-Rothblum 1993 and Vavasis 2009
(NP-hardness of nonnegative rank); Hrubes 2012 (distance matrices).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "NEG_ENTRY", "NOT_FOUND", "RankResult", "RankPlusBound",
    "words_upto", "counter", "two_state_ring_automaton", "hankel_block",
    "rank_real", "rank_plus_lower", "myhill_nerode_classes",
    "additive_nonneg_certificate", "rectangle_cover_number",
    "fooling_set_lower", "dedup",
    "TASKS", "task_row", "print_table",
    "counter_squared", "dyck1_member", "dyck1_clipped_height", "parity_a",
]


class _Sentinel:
    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return self._name


#: no nonnegative factorisation exists at ANY size, because an entry is negative
NEG_ENTRY = _Sentinel("NEG_ENTRY")
#: no nontrivial lower bound was obtained; reported as absence, never as zero
NOT_FOUND = _Sentinel("NOT_FOUND")


# --------------------------------------------------------------- words / tasks
def words_upto(alphabet: str, n: int) -> list[str]:
    """Every word of length ``0 .. n``, in a fixed deterministic order."""
    out = [""]
    for length in range(1, n + 1):
        out.extend("".join(t) for t in itertools.product(alphabet, repeat=length))
    return out


def counter(w: str) -> int:
    """The section 1.1 task: number of a's minus number of b's."""
    return w.count("a") - w.count("b")


def two_state_ring_automaton():
    """The 2-state ring-weighted automaton for :func:`counter`.

    Row-vector convention: ``s_0 = alpha``, ``s <- s @ M_x``, ``f(w) = s @ beta``.
    The state is ``(1, count)``; ``M_b`` carries the ``-1`` that a nonnegative
    automaton is not allowed to write down.
    """
    alpha = np.array([1.0, 0.0])
    mats = {"a": np.array([[1.0, 1.0], [0.0, 1.0]]),
            "b": np.array([[1.0, -1.0], [0.0, 1.0]])}
    beta = np.array([0.0, 1.0])
    return alpha, mats, beta


def hankel_block(f: Callable[[str], float], rows: Sequence[str],
                 cols: Sequence[str] | None = None) -> np.ndarray:
    """``H[u,v] = f(uv)`` on a finite block; square and symmetric if cols is None."""
    cols = rows if cols is None else cols
    return np.array([[float(f(u + v)) for v in cols] for u in rows], dtype=float)


# ------------------------------------------------------------------ real rank
@dataclass(frozen=True)
class RankResult:
    rank: int
    singular_values: np.ndarray
    gap_ratio: float
    threshold: float
    shape: tuple

    def __str__(self) -> str:
        sv = ", ".join("%.6e" % s for s in self.singular_values[:6])
        return ("rank=%d block=%dx%d gap_ratio=%.6e thr=%.6e sv=[%s]"
                % (self.rank, self.shape[0], self.shape[1],
                   self.gap_ratio, self.threshold, sv))


def rank_real(H: np.ndarray) -> RankResult:
    """Exact real rank of a finite block, with the margin that justified it.

    The threshold is the standard ``max(shape) * eps * sigma_0``. The number
    that makes the call trustworthy is not the threshold but ``gap_ratio``, the
    ratio of the last singular value kept to the first one dropped: a clean
    integer rank shows a gap of many orders of magnitude, and a gap near 1 means
    the rank was not measured, it was chosen.
    """
    H = np.asarray(H, dtype=float)
    s = np.linalg.svd(H, compute_uv=False)
    thr = max(H.shape) * np.finfo(float).eps * (s[0] if s.size else 0.0)
    r = int((s > thr).sum())
    if r == 0 or r >= s.size or s[r] == 0.0:
        gap = np.inf
    else:
        gap = float(s[r - 1] / s[r])
    return RankResult(r, s, gap, float(thr), tuple(H.shape))


# ---------------------------------------------------------------- Nerode count
def myhill_nerode_classes(f: Callable[[str], float], alphabet: str, n: int) -> int:
    """Distinct residuals ``v -> f(uv)`` for ``|u| <= n``, separated by ``|v| <= n``.

    The deterministic state count, printed beside every task as its difficulty.
    It is a block-local measurement, so it can only under-report, never
    over-report, the true Nerode index.
    """
    prefixes = words_upto(alphabet, n)
    suffixes = words_upto(alphabet, n)
    return len({tuple(float(f(u + v)) for v in suffixes) for u in prefixes})


# ------------------------------------------------------- nonnegative rank side
def dedup(H: np.ndarray) -> np.ndarray:
    """Drop duplicate rows and columns.

    ``rank``, ``rank_+`` and the rectangle-covering number are all invariant
    under deleting a duplicate row or column, so this is a free reduction, and
    it is what makes an exact rectangle cover tractable on a word-indexed block
    whose rows collapse onto a handful of count levels.
    """
    A = np.unique(np.asarray(H, dtype=float), axis=0)
    return np.unique(A, axis=1)


@dataclass(frozen=True)
class RankPlusBound:
    bound: object                     # int, or NEG_ENTRY / NOT_FOUND
    method: str
    detail: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return "rank_+ >= %r [%s]" % (self.bound, self.method)


def fooling_set_lower(S: np.ndarray, seed: int = 0, restarts: int = 64) -> int:
    """Largest fooling set found in the support ``S`` by seeded randomised greedy.

    A fooling set is a set of support cells no two of which can share a single
    all-support rectangle, so any rectangle cover -- and therefore any
    nonnegative factorisation -- needs at least one factor per cell. Greedy is a
    heuristic, so this is a valid lower bound that may be loose; it is never
    reported as the maximum fooling set.
    """
    S = np.asarray(S, dtype=bool)
    cells = [(int(i), int(j)) for i, j in zip(*np.nonzero(S))]
    rng = np.random.default_rng(seed)
    best = 0
    for _ in range(restarts):
        chosen = []
        for idx in rng.permutation(len(cells)):
            i, j = cells[idx]
            if all((not S[i, l]) or (not S[k, j]) for k, l in chosen):
                chosen.append((i, j))
        best = max(best, len(chosen))
    return best


def _maximal_rectangles(S: np.ndarray, row_cap: int = 16):
    """Every maximal all-support rectangle, as ``(row_mask, col_mask)`` bit pairs.

    Maximal rectangles are exactly the closed pairs of the Galois connection
    between rows and columns, so enumerating the closures of all row subsets is
    complete. That is ``2**rows`` work, which is why it is capped: completeness
    is what makes the resulting cover number a valid LOWER bound, so a partial
    enumeration is not an acceptable substitute and the function returns None.
    """
    S = np.asarray(S, dtype=bool)
    m, k = S.shape
    if m > row_cap:
        return None
    rowmask = [sum(1 << j for j in range(k) if S[i, j]) for i in range(m)]
    inter = [0] * (1 << m)
    inter[0] = (1 << k) - 1
    cols_seen = set()
    for A in range(1, 1 << m):
        low = A & -A
        inter[A] = inter[A ^ low] & rowmask[low.bit_length() - 1]
        if inter[A]:
            cols_seen.add(inter[A])
    rects = []
    for B in cols_seen:
        rows = sum(1 << i for i in range(m) if rowmask[i] & B == B)
        rects.append((rows, B))
    return rects


def rectangle_cover_number(S: np.ndarray, row_cap: int = 16,
                           rect_cap: int = 20000):
    """Exact rectangle-covering number of a support, by ILP over all maximal
    rectangles. Returns None when the exact value was not computed.

    Yannakakis: ``rank_+(M) >= rc(support(M))``. ``rc`` is itself NP-hard, so it
    is solved exactly and only for small deduplicated supports; above the caps
    the caller reports NOT FOUND rather than a partial answer.
    """
    from scipy.optimize import LinearConstraint, milp

    S = np.asarray(S, dtype=bool)
    # ponytail: the ILP is exact but the rectangle count is 2**levels, so cost
    # is exponential in the deduplicated row count. Measured on this machine,
    # torch threads pinned to 2: 5 levels 0.4s / 30 rects, 9 levels 2.8s / 510,
    # 11 levels 17.7s / 2046, 12 levels 64.0s / 4094, 13 levels 238.1s / 8190.
    # Practical ceiling is about 12 levels. If a task ever needs more, the
    # upgrade path is the LP relaxation of the same cover (still a valid lower
    # bound after ceiling) rather than a partial rectangle enumeration, which
    # would silently break the bound direction.
    rects = _maximal_rectangles(S, row_cap=row_cap)
    if rects is None or len(rects) > rect_cap:
        return None
    cells = [(int(i), int(j)) for i, j in zip(*np.nonzero(S))]
    if not cells:
        return 0
    A = np.zeros((len(cells), len(rects)), dtype=float)
    for c, (i, j) in enumerate(cells):
        for r, (rm, cm) in enumerate(rects):
            if (rm >> i) & 1 and (cm >> j) & 1:
                A[c, r] = 1.0
    res = milp(c=np.ones(len(rects)),
               constraints=LinearConstraint(A, lb=1, ub=np.inf),
               integrality=np.ones(len(rects)),
               bounds=(0, 1))
    if not res.success:
        return None
    return int(round(res.fun))


def rank_plus_lower(H: np.ndarray, seed: int = 0,
                    row_cap: int = 16, rect_cap: int = 20000) -> RankPlusBound:
    """Lower bound on the nonnegative rank of ``H``, or the impossibility sentinel.

    Order of business:

    1. a negative entry means NO nonnegative factorisation of any size;
    2. otherwise the real rank is always a valid lower bound;
    3. the rectangle-covering number of the support is computed exactly when the
       deduplicated support is small enough, and a randomised fooling set is
       taken as a cheap always-available bound;
    4. the best of these is reported WITH the method that attained it, and
       NOT FOUND is returned when nothing beat the trivial bound of 1.
    """
    H = np.asarray(H, dtype=float)
    if H.min() < 0:
        i, j = np.unravel_index(int(np.argmin(H)), H.shape)
        return RankPlusBound(
            NEG_ENTRY,
            "negative entry: no nonnegative factorisation of any size",
            {"witness": (int(i), int(j)), "value": float(H.min())})
    D = dedup(H)
    detail = {"dedup_shape": tuple(D.shape), "full_shape": tuple(H.shape)}
    cand = []

    r = rank_real(D).rank
    cand.append((r, "rank_R (rank_+ >= rank)"))
    detail["rank_R"] = r

    S = D != 0
    fool = fooling_set_lower(S, seed=seed)
    cand.append((fool, "fooling set (randomised greedy, seed=%d)" % seed))
    detail["fooling_set"] = fool

    rc = rectangle_cover_number(S, row_cap=row_cap, rect_cap=rect_cap)
    detail["rectangle_cover"] = rc if rc is not None else "NOT COMPUTED"
    if rc is not None:
        cand.append((rc, "rectangle cover, exact ILP (Yannakakis)"))

    best, how = max(cand, key=lambda t: t[0])
    if best <= 1:
        return RankPlusBound(NOT_FOUND, "no nontrivial bound", detail)
    return RankPlusBound(int(best), how, detail)


def additive_nonneg_certificate(H: np.ndarray):
    """Explicit ``rank_+ <= 2`` certificate for a nonnegative additive matrix.

    If ``H[i,j] = a_i + b_j`` and ``H >= 0`` then ``H = W @ Hc`` with ``W`` of
    shape ``(m,2)`` and ``Hc`` of shape ``(2,k)``, both entrywise nonnegative:
    put the whole offset on whichever side needs it. Returns ``(W, Hc)``, or
    None when ``H`` is not additive or not nonnegative. This is an upper bound
    witnessed by construction, so together with ``rank_+ >= rank_R = 2`` it pins
    ``rank_+ = 2`` exactly -- the one regime in this module where the NP-hard
    quantity is not merely bounded.
    """
    H = np.asarray(H, dtype=float)
    a = H[:, 0].copy()
    b = H[0, :] - H[0, 0]
    if not np.allclose(H, a[:, None] + b[None, :], rtol=0, atol=1e-12):
        return None
    if H.min() < 0:
        return None
    ma = float(a.min())
    W = np.column_stack([a - ma, np.ones_like(a)])
    Hc = np.vstack([np.ones_like(b), b + ma])
    return W, Hc


# --------------------------------------------------------------- task registry
def dyck1_member(w: str) -> float:
    """1 if ``w`` is a balanced Dyck-1 word (``a`` opens, ``b`` closes), else 0."""
    h = 0
    for c in w:
        h += 1 if c == "a" else -1
        if h < 0:
            return 0.0
    return 1.0 if h == 0 else 0.0


def dyck1_clipped_height(w: str) -> float:
    """Stack height with closes at empty ignored: the nonnegative counter."""
    h = 0
    for c in w:
        h = h + 1 if c == "a" else max(0, h - 1)
    return float(h)


def counter_squared(w: str) -> float:
    """The squared counter. Its Hankel is the distance matrix of the count
    levels, which is where the real rank / nonnegative rank gap lives."""
    return float(counter(w) ** 2)


def parity_a(w: str) -> float:
    """``(#a) mod 2``: the smallest task whose Krohn-Rhodes decomposition
    contains a nontrivial group."""
    return float(w.count("a") % 2)


#: name -> (function, needs_length_dependent_shift)
TASKS = {
    "counter": (counter, False),
    "counter_shift": (counter, True),
    "counter_squared": (counter_squared, False),
    "dyck1_member": (dyck1_member, False),
    "dyck1_clipped_height": (dyck1_clipped_height, False),
    "parity_a": (parity_a, False),
}


def task_row(name: str, n: int, alphabet: str = "ab") -> dict:
    """One row of the instrument's table for one task at one block size."""
    f, shifted = TASKS[name]
    words = words_upto(alphabet, n)
    H = hankel_block(f, words)
    if shifted:
        C = 2 * n
        H = H + C
        g = (lambda w, C=C: f(w) + C)
    else:
        g = f
    r = rank_real(H)
    lb = rank_plus_lower(H)
    cert = additive_nonneg_certificate(H) if H.min() >= 0 else None
    certified = (cert is not None
                 and cert[0].min() >= 0 and cert[1].min() >= 0
                 and np.array_equal(cert[0] @ cert[1], H)
                 and lb.bound == r.rank == 2)
    return {"task": name, "n": n, "block": r.shape, "certified": bool(certified),
            "dedup": lb.detail.get("dedup_shape"),
            "rank": r.rank, "gap_ratio": r.gap_ratio,
            "rank_plus": lb.bound, "method": lb.method,
            "rc": lb.detail.get("rectangle_cover"),
            "myhill_nerode": myhill_nerode_classes(g, alphabet, n)}


def print_table(ns=(2, 3, 4), alphabet: str = "ab") -> None:
    hdr = ("%-22s %2s %9s %8s %5s %11s %10s %6s  %s"
           % ("task", "n", "block", "dedup", "rank", "sv gap", "rank_+", "MN", "verdict"))
    print(hdr)
    print("-" * len(hdr))
    for name in TASKS:
        for n in ns:
            row = task_row(name, n, alphabet)
            rp = row["rank_plus"]
            if rp is NEG_ENTRY:
                verdict = "NO NONNEG FACTORISATION AT ANY SIZE"
            elif rp is NOT_FOUND:
                verdict = "NOT FOUND"
            elif rp > row["rank"]:
                verdict = "GAP (rank_+ >= %d > rank %d)" % (rp, row["rank"])
            elif row["certified"]:
                verdict = "NO GAP, certified (rank_+ = %d exactly)" % rp
            else:
                # rank_+ was only bounded BELOW, and the bound landed on rank.
                # That is an absence of evidence for a gap, not evidence of its
                # absence: rank_+ may still exceed rank here.
                verdict = "no gap detected (lower bound only)"
            print("%-22s %2d %9s %8s %5d %11.3e %10s %6d  %s"
                  % (name, n, "%dx%d" % row["block"],
                     "%dx%d" % row["dedup"] if row["dedup"] else "-",
                     row["rank"], row["gap_ratio"], rp, row["myhill_nerode"],
                     verdict))


if __name__ == "__main__":
    import torch
    torch.set_num_threads(2)
    print_table()

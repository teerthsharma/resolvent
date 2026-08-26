"""K-5 calibration: reproduce the LOOP_PROMPT.md section 1.1 worked calculation.

This file is the instrument's calibration, and it is pre-registered. Every
assertion below is a clause of the worked calculation quoted from the contract,
transcribed before `ceq/hankel.py` existed. Nothing downstream of this file is
read until it is understood, per K-5:

    "K-5 rank/rank_+ machinery disagrees with the worked example => the
     instrument is broken and nothing downstream of it is read."

The contract's text, quoted byte-exact from LOOP_PROMPT.md lines 98-105:

    WORKED CALCULATION -- the decrement wall, now a theorem-shaped gap. Take
    `f(w) = (#a) - (#b)` on `{a,b}*`. Then `f(uv) = f(u) + f(v)`, so
    `H_f[u,v] = f(u)*1 + 1*f(v)` -- a sum of two rank-1 terms, `rank_R = 2`, and a
    two-state ring automaton computes it. But `H_f` takes negative values, so it
    admits no nonnegative factorisation of any size on raw entries; the shifted
    comparison on a length-`n` block gives `rank_R = 3` while the nonnegative side
    must track `n` distinct count-levels -- a gap growing with `n`. Rank 2-3 vs
    Omega(n).

The clauses are labelled C1..C5b and each gets its own test, so a partial
disagreement is reported as a partial disagreement rather than collapsing the
whole calibration into one red light.
"""
import numpy as np
import pytest

from ceq.hankel import (
    NEG_ENTRY,
    counter,
    hankel_block,
    myhill_nerode_classes,
    rank_plus_lower,
    rank_real,
    two_state_ring_automaton,
    words_upto,
)

ALPHABET = "ab"


# --------------------------------------------------------------------- C1
def test_c1_f_is_additive_over_concatenation():
    """`f(uv) = f(u) + f(v)` on drawn pairs, not on a hand-picked pair."""
    rng = np.random.default_rng(20260826)
    n_draws = 512
    for _ in range(n_draws):
        u = "".join(rng.choice(list(ALPHABET), size=int(rng.integers(0, 9))))
        v = "".join(rng.choice(list(ALPHABET), size=int(rng.integers(0, 9))))
        assert counter(u + v) == counter(u) + counter(v)


# --------------------------------------------------------------------- C2
def test_c2_hankel_is_a_sum_of_two_rank_one_terms():
    """`H_f[u,v] = f(u)*1 + 1*f(v)` exactly, as a matrix identity."""
    words = words_upto(ALPHABET, 5)
    H = hankel_block(counter, words)
    fv = np.array([counter(w) for w in words], dtype=float)
    ones = np.ones_like(fv)
    reconstructed = np.outer(fv, ones) + np.outer(ones, fv)
    assert np.array_equal(H, reconstructed)


def test_c2_two_state_ring_automaton_computes_f():
    """"and a two-state ring automaton computes it" -- on drawn words."""
    alpha, mats, beta = two_state_ring_automaton()
    assert alpha.shape == (2,) and beta.shape == (2,)
    assert all(M.shape == (2, 2) for M in mats.values())
    rng = np.random.default_rng(7)
    for _ in range(512):
        w = "".join(rng.choice(list(ALPHABET), size=int(rng.integers(0, 13))))
        s = alpha.copy()
        for ch in w:
            s = s @ mats[ch]
        assert s @ beta == float(counter(w))


# --------------------------------------------------------------------- C3
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_c3_rank_of_raw_hankel_is_exactly_two(n):
    """`rank_R = 2`, exactly, at every block size, with the gap printed."""
    words = words_upto(ALPHABET, n)
    H = hankel_block(counter, words)
    r = rank_real(H)
    assert r.rank == 2, f"n={n} block={H.shape} sv={r.singular_values[:4]}"
    assert r.gap_ratio > 1e6, f"rank call not clean: gap_ratio={r.gap_ratio}"


# --------------------------------------------------------------------- C4
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_c4_raw_hankel_admits_no_nonnegative_factorisation_of_any_size(n):
    """"H_f takes negative values, so it admits no nonnegative factorisation
    of any size on raw entries"."""
    words = words_upto(ALPHABET, n)
    H = hankel_block(counter, words)
    assert H.min() < 0
    lb = rank_plus_lower(H)
    assert lb.bound is NEG_ENTRY, f"expected impossibility, got {lb}"


# --------------------------------------------------------------------- C5a
@pytest.mark.xfail(strict=True, reason=(
    "K-5 FIRES ON C5a. Measured rank of the shifted block is 2, not 3, at every "
    "block size: n=2 (7x7) sv=[3.030951e+01, 2.309506e+00, 2.978522e-15, ...] "
    "gap_ratio=7.753868e+14; n=6 (127x127) sv=[1.575743e+03, 5.174320e+01, "
    "4.941790e-13, ...] gap_ratio=1.047054e+14. Reason: H_f = a.1^T + 1.a^T "
    "already contains the all-ones vector in its column space, so adding C.11^T "
    "cannot raise the rank. The generic rule rank(f + C) = rank(f) + 1 is exactly "
    "the step that does not apply to this f. Pinned strict so the finding cannot "
    "quietly stop reproducing."))
@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_c5a_shifted_comparison_has_rank_three(n):
    """"the shifted comparison on a length-`n` block gives `rank_R = 3`"."""
    words = words_upto(ALPHABET, n)
    C = 2 * n                                   # least shift making H >= 0
    H = hankel_block(counter, words) + C
    assert H.min() >= 0
    r = rank_real(H)
    assert r.rank == 3, f"n={n} block={H.shape} sv={r.singular_values[:5]}"


# --------------------------------------------------------------------- C5b
@pytest.mark.xfail(strict=True, reason=(
    "K-5 FIRES ON C5b. Measured rank_+ lower bound of the shifted block is 2 at "
    "every n, by exact rectangle-cover ILP on the deduplicated 2n+1 by 2n+1 "
    "support, and it does not grow. It cannot grow: the block is additive and "
    "nonnegative, so additive_nonneg_certificate builds an explicit nonnegative "
    "two-factor product reproducing it bitwise, pinning rank_+ = 2 from above. "
    "Cohen-Rothblum 1993 Theorem 4.1 says the same thing in general -- a "
    "nonnegative matrix of rank <= 2 has rank_+ = rank. The Omega(n) in the "
    "clause is real but it is the Myhill-Nerode count, measured at 2n+1 by "
    "test_myhill_nerode_count_is_the_count_level_number below, which is "
    "deterministic memory and not a cost of nonnegativity."))
# n=2 is excluded: the clause's bar `rank_+ >= n` is met there by the measured
# rank_+ = 2 for the wrong reason, i.e. coincidence at the smallest case. That
# is precisely the shape of vacuous control this project has struck nine times.
@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_c5b_nonnegative_side_must_track_n_distinct_count_levels(n):
    """"while the nonnegative side must track `n` distinct count-levels -- a
    gap growing with `n`. Rank 2-3 vs Omega(n)."

    The nonnegative side of the instrument is `rank_+`, bounded below. The
    clause says that bound grows with the block length; the weakest reading
    that still says Omega(n) is `rank_+ >= n`.
    """
    words = words_upto(ALPHABET, n)
    H = hankel_block(counter, words) + 2 * n
    lb = rank_plus_lower(H)
    assert lb.bound >= n, f"n={n} rank_+ lower bound = {lb.bound} via {lb.method}"


@pytest.mark.xfail(strict=True, reason=(
    "K-5 FIRES ON C5b. Measured: {2: 2, 6: 2}. The bound is flat at 2, not "
    "growing. Independently: Hrubes 2012 (IPL 112(11):457-461) proves that even "
    "the genuine rank-3 counter family has rank_+ <= 2*log2(n)+2, so Omega(n) is "
    "not available anywhere in this neighbourhood -- it is the claim of Lin & Chu "
    "2010 (LAA 433(3):681-689, Thm 3.1, `rank_+(Qn) = n`) that Hrubes refutes by "
    "name."))
def test_c5b_gap_grows_with_n():
    """"a gap growing with `n`" -- the bound at n=6 must exceed the bound at n=2."""
    bounds = {}
    for n in (2, 6):
        words = words_upto(ALPHABET, n)
        bounds[n] = rank_plus_lower(hankel_block(counter, words) + 2 * n).bound
    assert bounds[6] > bounds[2], bounds


# ------------------------------------------------- difficulty column, printed
@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_myhill_nerode_count_is_the_count_level_number(n):
    """Myhill-Nerode is the third column of the instrument and is printed
    beside every task. For a counter on a length-`n` block it is the number of
    reachable count levels, `2n+1`."""
    assert myhill_nerode_classes(counter, ALPHABET, n) == 2 * n + 1

"""Must-fire for the Hankel instrument, in both directions, on DRAWN instances.

Nine controls have gone vacuous in this project across four authors, two of them
mine, and every one of them was a hand-built minimal example -- because the
smallest case is usually where the right and the wrong answer coincide. So
nothing here is hand-built. Every matrix is drawn from a seeded generator and
every claim is a count over draws.

Four things are checked, in order of what they protect:

1. THE BOUND MACHINERY AGAINST ITSELF. The exact rectangle-covering ILP is
   compared against exhaustive search on drawn supports. An ILP that silently
   returns a wrong optimum would make every gap in this file fiction.

2. THE NO-GAP DIRECTION. Families whose nonnegative rank provably equals their
   real rank must read as NO GAP on every draw. Two of them are used: matrices
   drawn as an explicit nonnegative rank-2 product, where the answer is fixed by
   construction, and the shifted counter of section 1.1, where a nonnegative
   two-factor certificate is built and verified bitwise.

3. THE GAP DIRECTION. The squared-counter family, whose Hankel support is
   inequality on count levels, must read as GAP on every draw with enough
   levels: real rank pinned at 3 while the nonnegative lower bound climbs.

4. THE SELF-TEST -- the control that proves the control is not vacuous. A
   deliberately WRONG instrument, one that reports the Myhill-Nerode class count
   as the nonnegative bound, is run over the same no-gap draws and must read
   GAP. If the wrong instrument agreed with the right one, the no-gap direction
   above would be measuring nothing.
"""
import itertools

import numpy as np
import pytest
import torch

from ceq.hankel import (
    additive_nonneg_certificate,
    counter,
    dedup,
    hankel_block,
    myhill_nerode_classes,
    rank_plus_lower,
    rank_real,
    rectangle_cover_number,
    words_upto,
    _maximal_rectangles,
)

torch.set_num_threads(2)          # pinned in the probe file, per standing policy

SEED = 20260826
N_DRAWS = 32


def _squared_counter(w):
    return float(counter(w) ** 2)


def _brute_force_rc(S, limit=5):
    """Exhaustive rectangle cover number, for calibrating the ILP."""
    rects = _maximal_rectangles(S)
    cells = list(zip(*np.nonzero(S)))
    if not cells:
        return 0
    masks = []
    for rm, cm in rects:
        masks.append(sum(1 << c for c, (i, j) in enumerate(cells)
                         if (rm >> int(i)) & 1 and (cm >> int(j)) & 1))
    target = (1 << len(cells)) - 1
    for k in range(1, limit + 1):
        for combo in itertools.combinations(masks, k):
            acc = 0
            for m in combo:
                acc |= m
            if acc == target:
                return k
    return None


# ------------------------------------------------------- 1. bound vs brute force
def test_rectangle_cover_ilp_matches_exhaustive_search_on_drawn_supports():
    rng = np.random.default_rng(SEED)
    checked = 0
    for _ in range(N_DRAWS):
        m = int(rng.integers(3, 7))
        k = int(rng.integers(3, 7))
        S = rng.random((m, k)) < 0.65
        if not S.any() or (S.sum(0) == 0).any() or (S.sum(1) == 0).any():
            continue
        ilp = rectangle_cover_number(S)
        brute = _brute_force_rc(S, limit=5)
        if brute is None:                      # optimum above the exhaustive limit
            assert ilp is None or ilp > 5, (ilp, S.astype(int))
            continue
        assert ilp == brute, (ilp, brute, S.astype(int))
        checked += 1
    assert checked >= 20, f"only {checked} drawn supports were decidable"


# ------------------------------------------------------------ 2. no-gap direction
def test_mustfire_no_gap_on_drawn_nonnegative_rank_two_products():
    """Matrices drawn AS a nonnegative rank-2 product: rank_+ = rank = 2 by
    construction, so any gap the instrument reports is a false positive."""
    rng = np.random.default_rng(SEED + 1)
    fired = 0
    for _ in range(N_DRAWS):
        m, k = int(rng.integers(4, 11)), int(rng.integers(4, 11))
        W = rng.integers(0, 6, size=(m, 2)).astype(float)
        Hc = rng.integers(0, 6, size=(2, k)).astype(float)
        M = W @ Hc
        r = rank_real(M).rank
        lb = rank_plus_lower(M)
        if r < 2:                              # degenerate draw, not a test case
            continue
        assert lb.bound == r, f"false gap: rank={r} bound={lb.bound} via {lb.method}"
        fired += 1
    assert fired >= 25, f"only {fired} usable draws"


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_mustfire_no_gap_on_the_shifted_counter_with_a_verified_certificate(n):
    """The section 1.1 shifted comparison. rank_+ is pinned from BOTH sides:
    the rectangle-cover lower bound, and an explicit nonnegative two-factor
    product checked bitwise against the block."""
    H = hankel_block(counter, words_upto("ab", n)) + 2 * n
    r = rank_real(H)
    lb = rank_plus_lower(H)
    cert = additive_nonneg_certificate(H)
    assert cert is not None
    W, Hc = cert
    assert W.min() >= 0 and Hc.min() >= 0
    assert np.array_equal(W @ Hc, H)           # bitwise, not allclose
    assert r.rank == 2 and lb.bound == 2, f"{r} / {lb}"


# --------------------------------------------------------------- 3. gap direction
@pytest.mark.parametrize("n", [2, 3, 4])
def test_mustfire_gap_on_the_squared_counter(n):
    """Real rank stays at 3; the nonnegative lower bound exceeds it."""
    H = hankel_block(_squared_counter, words_upto("ab", n))
    r = rank_real(H)
    lb = rank_plus_lower(H)
    assert r.rank == 3, str(r)
    assert lb.bound > r.rank, f"gap did not fire: {r} / {lb}"


def test_mustfire_gap_on_drawn_level_sets():
    """Same family, but the count levels are DRAWN rather than enumerated, so a
    gap that only exists on the tidy symmetric block cannot pass."""
    rng = np.random.default_rng(SEED + 2)
    fired = 0
    n_level_draws = 12          # each draw is an exact ILP; see the ceiling note
    for _ in range(n_level_draws):
        m = int(rng.integers(5, 10))
        x = rng.choice(np.arange(-40, 41), size=m, replace=False).astype(float)
        M = (x[:, None] - x[None, :]) ** 2
        r = rank_real(M).rank
        lb = rank_plus_lower(M)
        assert r == 3, f"rank moved off 3: {r} on x={x}"
        assert lb.bound > 3, f"gap did not fire: rank=3 bound={lb.bound} m={m}"
        fired += 1
    assert fired == n_level_draws, f"{fired}/{n_level_draws} draws fired"


def test_gap_bound_grows_with_the_number_of_levels():
    """The gap is not a constant: rank is frozen at 3 while the bound climbs."""
    bounds = {}
    for m in (3, 5, 7, 9, 11):
        x = np.arange(m, dtype=float)
        M = (x[:, None] - x[None, :]) ** 2
        assert rank_real(M).rank == 3 or m < 3
        bounds[m] = rank_plus_lower(M).bound
    assert bounds[11] > bounds[7] > bounds[3], bounds
    # de Caen-Gregory-Pullman 1981, via Sperner 1928: the rectangle-covering
    # number of the complement of the identity is sigma(m) = min{d : m <= C(d,
    # floor(d/2))}. The instrument computed these independently, by ILP.
    assert bounds == {3: 3, 5: 4, 7: 5, 9: 5, 11: 6}, bounds


def test_absence_is_reported_as_not_computed_never_as_zero():
    """Above the enumeration cap the exact cover is not computed, and the
    instrument must say so rather than silently reporting a weak bound as if it
    were the exact one."""
    x = np.arange(20, dtype=float)
    M = (x[:, None] - x[None, :]) ** 2
    lb = rank_plus_lower(M)
    assert lb.detail["rectangle_cover"] == "NOT COMPUTED"
    assert "rectangle cover" not in lb.method


# ---------------------------------------------- 4. self-test: wrong instrument
def test_selftest_the_nerode_instrument_reports_a_gap_where_there_is_none():
    """The control that keeps the no-gap direction from being vacuous.

    The specific wrong instrument is the one section 1.1 used: read the number
    of distinct count levels -- the Myhill-Nerode class count -- as the
    nonnegative-side number. On the shifted counter that reads a gap of `2n+1`
    against `2` at every block size, while the real nonnegative rank is `2`.
    If this test ever goes green, the no-gap direction is measuring nothing.
    """
    wrong = {}
    for n in (2, 3, 4, 5, 6):
        H = hankel_block(counter, words_upto("ab", n)) + 2 * n
        wrong[n] = (rank_real(H).rank,
                    myhill_nerode_classes(lambda w, n=n: counter(w) + 2 * n, "ab", n))
    assert all(mn > r for r, mn in wrong.values()), wrong
    assert [mn for _, mn in wrong.values()] == [5, 7, 9, 11, 13], wrong


def test_dedup_does_not_move_any_measured_number():
    """Deduplication is the reduction that makes the exact cover tractable; it
    is only free if it moves nothing."""
    rng = np.random.default_rng(SEED + 3)
    for _ in range(16):
        m, k = int(rng.integers(3, 7)), int(rng.integers(3, 7))
        base = rng.integers(0, 4, size=(m, k)).astype(float)
        rep = base[rng.integers(0, m, size=m + 4), :]
        assert rank_real(rep).rank == rank_real(dedup(rep)).rank
        assert rank_plus_lower(rep).bound == rank_plus_lower(base).bound

"""The granularity of the trend clause, against Venus's floor for the size clause.

Venus measured the shipped `contrast()` over 1000 samples: a 5-0 seed unanimity
excludes zero 385/385 times, a 4-1 split 20-44%, a 3-2 split 0-3.7%. So at five
seeds "the CI excludes zero" is very nearly "all five agreed", and the finest
two-sided p a sign-pattern statistic on five blocks can express is
`2 / 2^5 = 0.0625` -- ABOVE the 0.05 the project quotes.

That floor applies to `page_trend`'s SIZE clause, which is exactly such a
bootstrap. It does NOT apply to its TREND clause. Page's L ranks four conditions
within each block rather than reading one sign, so the outcome space is
`24^5 = 7962624` rather than `2^5 = 32`. This file establishes both halves and
nails the numbers down so neither can drift.

THREE PATHS TO THE NULL, SHARING NO ARITHMETIC.
  1. `page_null`         float probabilities through `np.convolve`
  2. `page_null_exact`   integer polynomials through Python big integers
  3. literal enumeration every one of the `(k!)^n` assignments, L built from
                         whole tables rather than from per-block sums
Path 3 is intractable at `k=4, n=5` and is taken at three smaller designs
instead, which is what makes 1 and 2 trustworthy there.
"""
import itertools
import math
from collections import Counter
from fractions import Fraction

import numpy as np

from scale.page_trend import (ALPHA, achievable_p_values, bootstrap_p_floor,
                              critical_value, from_ladder, page_null,
                              page_null_exact, page_p_value)


def literal_null(k, n):
    """Path 3. Enumerates whole rank tables. Shares no code with the other two."""
    perms = list(itertools.permutations(range(1, k + 1)))
    counter = Counter()
    for assignment in itertools.product(perms, repeat=n):
        rank_sums = [sum(block[j] for block in assignment) for j in range(k)]
        counter[sum((j + 1) * rank_sums[j] for j in range(k))] += 1
    return counter


def test_the_integer_and_float_nulls_agree():
    support, counts, total = page_null_exact(4, 5)
    assert total == 24 ** 5 == 7962624
    float_support, dist = page_null(4, 5)
    assert [int(v) for v in float_support] == support
    assert max(abs(dist[i] - counts[i] / total)
               for i in range(len(counts))) < 1e-15


def test_both_nulls_agree_with_literal_enumeration():
    for k, n in ((3, 3), (4, 3), (3, 4)):
        literal = literal_null(k, n)
        assert sum(literal.values()) == math.factorial(k) ** n
        support, counts, total = page_null_exact(k, n)
        assert all(counts[i] == literal.get(support[i], 0)
                   for i in range(len(support)))
        float_support, dist = page_null(k, n)
        assert max(abs(dist[i] - literal.get(int(float_support[i]), 0) / total)
                   for i in range(len(dist))) < 1e-15


def test_the_achievable_lattice_is_what_the_design_can_produce():
    lattice = achievable_p_values(4, 5)
    assert lattice[0][0] == 150
    assert lattice[0][1] == Fraction(1, 7962624)
    assert lattice[-1][0] == 100
    assert lattice[-1][1] == 1
    assert len({p for _t, p in lattice}) == 51
    # the lattice descends in L, so its p-values must ascend, strictly wherever
    # the atom count is non-zero. A tautological form of this assertion would
    # pass on a shuffled lattice; this one does not.
    ps = [p for _t, p in lattice]
    assert all(a <= b for a, b in zip(ps, ps[1:]))
    assert ps[0] < ps[-1]
    # every entry is exactly reproducible by the float path
    for t, p in lattice:
        assert abs(page_p_value(float(t), 4, 5) - float(p)) < 1e-12


def test_the_observed_p_is_attainable_and_not_an_interpolation():
    """`0.016724` is an exact atom sum of the discrete null, not a number the
    test interpolated onto it."""
    lattice = dict(achievable_p_values(4, 5))
    assert 139 in lattice
    assert lattice[139] == Fraction(22195, 1327104)
    assert lattice[139] * 7962624 == 133170          # exact assignment count
    assert abs(float(lattice[139]) - 0.016724386) < 1e-9
    assert abs(page_p_value(139.0, 4, 5) - float(lattice[139])) < 1e-12


def test_alpha_is_reachable_for_the_trend_clause_and_its_true_size_is_smaller():
    cv = critical_value(4, 5, 0.05)
    assert cv["reachable"]
    assert cv["L_crit"] == 137
    assert abs(cv["effective_size"] - 0.037002877) < 1e-8
    assert cv["effective_size"] < 0.05         # discrete tests are conservative
    assert cv["n_below_alpha"] == 14
    assert cv["n_achievable"] == 51
    assert abs(cv["finest_p"] - 1 / 7962624) < 1e-15
    # the next coarser rung is ABOVE alpha, which is what makes 137 the critical
    # value rather than a preference
    lattice = dict(achievable_p_values(4, 5))
    assert float(lattice[136]) > 0.05
    assert abs(float(lattice[136]) - 0.052384114) < 1e-8


def test_the_size_clause_floor_is_above_alpha_and_the_trend_clause_is_not():
    """The reconciliation. One clause clears 0.05 on granularity; one cannot."""
    assert bootstrap_p_floor(5) == 0.0625
    assert bootstrap_p_floor(5) > ALPHA        # the SIZE clause cannot reach 0.05
    assert critical_value(4, 5, ALPHA)["effective_size"] < ALPHA   # the TREND can
    # and quantify the gap: 24^5 outcomes against 2^5
    assert 24 ** 5 // 2 ** 5 == 248832 == 12 ** 5
    assert abs(math.log2(24) - 4.5849625) < 1e-6


def test_the_sign_lattice_has_exactly_one_point_below_alpha():
    """Why the size clause is a unanimity detector: of the six achievable
    one-sided sign-test p-values at N=5, only unanimity clears 0.05, and its
    two-sided partner does not."""
    one_sided = [sum(math.comb(5, i) for i in range(x, 6)) / 32
                 for x in range(5, -1, -1)]
    assert one_sided[0] == 1 / 32 == 0.03125
    assert sum(1 for p in one_sided if p <= ALPHA) == 1
    assert sum(1 for p in one_sided if min(1.0, 2 * p) <= ALPHA) == 0
    # against Page, which has fourteen
    assert critical_value(4, 5, ALPHA)["n_below_alpha"] == 14


def test_the_real_ladder_top_rung_is_the_four_one_split_venus_measured():
    """The size clause fired here on a 4-1 split, the regime Venus measured at
    20-44%. The unconstrained interval covers zero, as that regime predicts;
    only the one-sided PAVA lift carries it over."""
    cur = from_ladder()
    assert cur["n_plus_top"] == 4
    assert len(cur["seeds"]) == 5
    assert cur["isotonic"]["raw_ci_lo"] < 0.0 < cur["isotonic"]["ci_lo"]
    assert cur["page"]["critical"]["L_crit"] == 137
    assert cur["page"]["L"] >= 137             # the trend clause clears its own
    assert np.isclose(cur["page"]["p"], 0.016724386, atol=1e-9)

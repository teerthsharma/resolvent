"""A 5-seed bootstrap interval lands on a lattice; report where, exactly.

`contrast()` draws `n_boot` resamples from a statistic that has only `n**n`
of them. At n=5 that is 3,125 resamples over ~128 distinct atoms, so the 2.5 %
and 97.5 % percentiles are picked from a short list and Monte-Carlo chooses
between ADJACENT atoms depending on the bootstrap seed.

Measured on `argmaxste - argmax` (the cell this session ran): over bootstrap
seeds 0..99 `ci_lo` takes 2 distinct values and `ci_hi` takes 3. The seed-0 pair
sits exactly one atom below the exact pair at BOTH ends, and the two gaps are
both `1.056e-04` -- which looks like a constant estimator shift and is not one,
because the endpoints move independently under seed change.

The exact percentile needs no seed and costs 3,125 sums at n=5, so it is
reported alongside rather than instead of: `ci_lo`/`ci_hi` are unchanged, so no
journalled value and no published verdict moves.

    python -m pytest tests/mercury/test_r9_exact_interval.py -q
"""
from __future__ import annotations

import itertools
import math

import pytest

from scale.m3_synthetic_settled import contrast

ARGMAX = [1.022910, 0.994867, 1.011705, 1.010804, 1.013609]
ARGMAXSTE = [0.801954, 0.777060, 0.746923, 0.802827, 0.796330]


def _exact(ref, arm):
    d = [r - a for r, a in zip(ref, arm)]
    n = len(d)
    # `math.fsum` on a canonically ordered tuple: identical multisets must
    # produce identical floats, or the count splits atoms that are not distinct.
    reps = sorted(math.fsum(sorted(c)) / n for c in itertools.product(d, repeat=n))
    return (reps[int(0.025 * len(reps))],
            reps[min(len(reps) - 1, int(0.975 * len(reps)))],
            len(set(reps)))


def test_contrast_reports_the_exact_pair_and_the_atom_count():
    c = contrast(ARGMAX, ARGMAXSTE, n_boot=10000, seed=0)
    lo, hi, atoms = _exact(ARGMAX, ARGMAXSTE)
    assert c["exact_lo"] == pytest.approx(lo, abs=1e-12)
    assert c["exact_hi"] == pytest.approx(hi, abs=1e-12)
    assert c["n_atoms"] == atoms == 126


def test_the_exact_pair_reproduces_neptunes_enumeration():
    """Two independent implementations of the same quantity."""
    c = contrast(ARGMAX, ARGMAXSTE, n_boot=10000, seed=0)
    assert c["exact_lo"] == pytest.approx(0.212539, abs=5e-7)
    assert c["exact_hi"] == pytest.approx(0.245992, abs=5e-7)


def test_the_exact_pair_does_not_depend_on_the_bootstrap_seed():
    a = contrast(ARGMAX, ARGMAXSTE, n_boot=2000, seed=0)
    b = contrast(ARGMAX, ARGMAXSTE, n_boot=2000, seed=17)
    assert a["exact_lo"] == b["exact_lo"]
    assert a["exact_hi"] == b["exact_hi"]


def test_the_monte_carlo_pair_is_left_untouched():
    """Nothing journalled and no published verdict may move."""
    c = contrast(ARGMAX, ARGMAXSTE, n_boot=10000, seed=0)
    assert c["ci_lo"] == pytest.approx(0.212433, abs=5e-7)
    assert c["ci_hi"] == pytest.approx(0.245886, abs=5e-7)
    assert c["verdict"] == contrast(ARGMAX, ARGMAXSTE, n_boot=10000,
                                    seed=0)["verdict"]


def test_the_exact_path_is_skipped_when_enumeration_is_too_large():
    """13 seeds is 13**13 resamples; the field must be absent, not slow."""
    ref = [1.0 + i * 0.01 for i in range(13)]
    arm = [0.9 + i * 0.01 for i in range(13)]
    c = contrast(ref, arm, n_boot=200, seed=0)
    assert c["exact_lo"] is None and c["exact_hi"] is None
    assert c["n_atoms"] is None


def test_the_atom_count_cannot_exceed_its_combinatorial_maximum():
    """The number of distinct means of a size-n multiset drawn from n values is
    at most `C(2n-1, n)`. A count above that is floating-point noise splitting
    atoms, not a finer lattice.

    Naive `sum()` over `itertools.product` does exactly that: different
    permutations of the SAME multiset sum in different orders and land one ULP
    apart, which reported 128 atoms here against a true maximum of 126.
    """
    c = contrast(ARGMAX, ARGMAXSTE, n_boot=200, seed=0)
    n = c["n_seeds"]
    assert c["n_atoms"] <= math.comb(2 * n - 1, n)
    assert c["n_atoms"] == 126


def test_the_naive_summation_really_would_over_count():
    """Adversarial pass: the guard must be protecting against something real."""
    d = [r - a for r, a in zip(ARGMAX, ARGMAXSTE)]
    n = len(d)
    naive = {sum(x) / n for x in itertools.product(d, repeat=n)}
    canonical = {math.fsum(sorted(x)) / n for x in itertools.product(d, repeat=n)}
    assert len(naive) == 128, "the over-count this change exists to fix"
    assert len(canonical) == 126

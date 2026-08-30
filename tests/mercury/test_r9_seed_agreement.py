"""Every 5-seed interval must carry the seed-agreement count that produced it.

Venus measured that at N=5 the shipped `contrast()` excludes zero in 385/385
runs when all five seeds agree, in 20-44% of 4-1 splits, and in 0-3.7% of 3-2
splits. So "the CI excludes zero" at N=5 is very nearly "all five seeds agreed",
and the finest achievable two-sided p is 0.0625, not 0.05.

That is a property of every 5-seed reading in this repository, not of one lane,
so the count is reported by `contrast()` itself rather than by whichever caller
happens to remember. `per_seed_delta` was already returned; `n_pos` is the
count a reader needs beside the interval to see which regime produced it.

Run this file alone:

    python -m pytest tests/mercury/test_r9_seed_agreement.py -q
"""
from __future__ import annotations

import pytest

from scale.m3_synthetic_settled import contrast


def _c(ref, arm):
    return contrast(list(ref), list(arm), n_boot=2000, seed=0)


def test_contrast_reports_the_seed_agreement_count():
    # arm beats ref on 4 of 5 seeds
    ref = [1.00, 1.00, 1.00, 1.00, 1.00]
    arm = [0.90, 0.90, 0.90, 0.90, 1.10]
    c = _c(ref, arm)
    assert c["n_pos"] == 4
    assert c["n_seeds"] == 5


def test_the_count_matches_the_per_seed_deltas_it_is_derived_from():
    ref = [1.0, 1.1, 0.9, 1.2, 1.0]
    arm = [0.8, 1.3, 0.7, 1.1, 1.0]
    c = _c(ref, arm)
    assert c["n_pos"] == sum(1 for x in c["per_seed_delta"] if x > 0.0)


def test_unanimity_and_ties_are_both_representable():
    assert _c([1.0] * 5, [0.5] * 5)["n_pos"] == 5
    assert _c([1.0] * 5, [1.5] * 5)["n_pos"] == 0
    assert _c([1.0] * 5, [1.0] * 5)["n_pos"] == 0      # exact ties are not wins


def test_a_three_two_split_cannot_exclude_zero_at_five_seeds():
    """The mechanism behind Venus's 0-3.7%: with the paired bootstrap resampling
    five values, a 3-2 split leaves resamples on both sides of zero."""
    ref = [1.0, 1.0, 1.0, 1.0, 1.0]
    arm = [0.9, 0.9, 0.9, 1.1, 1.1]
    c = _c(ref, arm)
    assert c["n_pos"] == 3
    assert c["ci_lo"] < 0.0 < c["ci_hi"], "a 3-2 split should not exclude zero"
    assert c["verdict"] == "NO DIFFERENCE"


def test_unanimity_is_what_actually_excludes_zero():
    """The other half of the same mechanism, so the test is not one-sided."""
    ref = [1.0, 1.0, 1.0, 1.0, 1.0]
    arm = [0.9, 0.9, 0.9, 0.9, 0.9]
    c = _c(ref, arm)
    assert c["n_pos"] == 5
    assert c["ci_lo"] > 0.0


def test_the_finest_two_sided_p_at_five_seeds_is_one_sixteenth():
    """2 / 2**5 = 0.0625 > 0.05. A 5-seed sign test cannot reach 0.05."""
    assert 2 / 2 ** 5 == 0.0625
    assert 2 / 2 ** 5 > 0.05

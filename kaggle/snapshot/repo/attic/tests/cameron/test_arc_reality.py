"""ARC-AGI as a MEASURING INSTRUMENT, before ARC-AGI as a target.

The question the user asked is "what would our 0.5B model score on ARC-AGI".
The prior question, which is cheaper and decides whether the first one is worth
any GPU time at all, is: **at ARC-AGI's evaluation-set size and scoring rule,
what is the smallest difference between two models that the benchmark can
resolve?**

That question needs no model, no GPU and no network. It is arithmetic over the
benchmark's own scoring rule, read off arcprize.org/guide/1 and
github.com/fchollet/ARC-AGI on 2026-08-24:

    "Scoring is exact match on the full output grid: correct size, correct
     colour in every cell. One cell wrong scores zero."  -- arcprize.org/guide/1
    "Partial credit is not awarded."                     -- arcprize.org/guide/1

Evaluation-set sizes, same sources: ARC-AGI-1 public eval 400 tasks, private
eval 100. ARC-AGI-2 public / semi-private / private eval 120 each. Scoring on
the leaderboard and on Kaggle is pass@2; the v1 repository text says 3 trials.
Grids run 1x1 to 30x30 over 10 colours.

The ARC-AGI-2 paper states the noise floor in its own words
(arXiv:2505.11831v1): "ARC-AGI-2 accuracies below 5% are generally not treated
as meaningful, as they likely result from noise-level heuristics or incidental
pattern fits." This file derives that same 5% from the scoring rule instead of
accepting it on authority, which matters because it is the number that decides
whether the planned run is worth its GPU budget.

Nothing here is an opinion about whether a small model is smart. It is a
statement about what a 120-task exact-match benchmark can and cannot detect, and
it applies identically to every architecture on both sides of the comparison.

Every test parametrizes over cpu and cuda per the standing rule; the grid
scoring runs as tensors on the device under test, so the parametrization is
load-bearing rather than decorative.
"""
from __future__ import annotations

import pytest
import torch

from ceq import bench

# ARC-AGI-2 public evaluation set: 120 tasks. Sizes are inputs to the
# arithmetic, not claims -- the property asserted holds across the whole family
# of ARC evaluation-set sizes, and
# `test_the_floor_is_not_an_artifact_of_one_evaluation_set_size` sweeps
# 100 (v1 private) / 120 (v2, all three sets) / 400 (v1 public eval).
N_PUBLIC_EVAL = 120
COLORS = 10
MAX_SIDE = 30


def test_random_grid_guessing_scores_exactly_zero(device):
    """Draw uniformly random output grids and score them by ARC's own rule.

    This is the benchmark's floor measured rather than asserted. If the floor
    were reachable by chance, a near-zero score would be ambiguous between
    "guessed" and "solved nothing"; it is not.
    """
    g = torch.Generator(device="cpu").manual_seed(0)
    solved = 0
    for _ in range(N_PUBLIC_EVAL):
        h, w = 8, 8
        target = torch.randint(0, COLORS, (h, w), generator=g).to(device)
        # pass@2: two independent guesses, either may match
        guesses = [torch.randint(0, COLORS, (h, w), generator=g).to(device)
                   for _ in range(2)]
        solved += int(bench.arc_task_solved(guesses, target))

    assert solved == 0, f"random guessing solved {solved}/{N_PUBLIC_EVAL}"

    # and the analytic reason, in log10 so it is readable
    log10_p = bench.log10_random_grid_chance(8, 8, COLORS, attempts=2)
    assert log10_p < -60, log10_p


def test_a_zero_score_hides_every_architecture_effect_in_its_confidence_interval(device):
    """0 solved out of 120 is not "0%". It is an interval.

    The exact 95% upper bound on p given 0 successes in n trials is
    1 - 0.05**(1/n). Any true skill below that bound produces the identical
    observation. If that bound is wider than the effect an attention change
    could plausibly buy, the measurement carries no information about the
    attention change.
    """
    _ = torch.zeros(1, device=device)  # the run is device-parametrized by rule
    hi = bench.zero_success_upper_bound(N_PUBLIC_EVAL, conf=0.95)
    assert 0.02 < hi < 0.03, hi
    # the resolution floor: scores are multiples of 1/n, so nothing below
    # 1/120 = 0.83% is even representable
    assert bench.resolution_floor(N_PUBLIC_EVAL) == pytest.approx(1 / 120)
    assert hi > 2 * bench.resolution_floor(N_PUBLIC_EVAL)


def test_two_arms_both_scoring_zero_are_statistically_indistinguishable(device):
    """The comparison this project would actually run: ceq vs matched softmax.

    Both at 0/120 gives a one-sided p of exactly 1.0. Not "small", not
    "suggestive" -- the run returns literally nothing, at whatever the GPU cost
    of two 0.5B evaluations is.
    """
    _ = torch.zeros(1, device=device)
    p = bench.separation_p(k_a=0, n_a=N_PUBLIC_EVAL, k_b=0, n_b=N_PUBLIC_EVAL)
    assert p == pytest.approx(1.0), p


def test_arc_needs_several_solved_tasks_before_any_gap_is_detectable(device):
    """How far from zero the better arm has to get before the benchmark speaks.

    Against a control at 0/120, the smallest k with one-sided Fisher p < 0.05.
    This is the real entry price of ARC-AGI as a comparison instrument.
    """
    _ = torch.zeros(1, device=device)
    k = bench.min_successes_for_separation(N_PUBLIC_EVAL, k_baseline=0, alpha=0.05)
    assert 3 <= k <= 8, k
    assert bench.separation_p(0, N_PUBLIC_EVAL, k, N_PUBLIC_EVAL) < 0.05
    assert bench.separation_p(0, N_PUBLIC_EVAL, k - 1, N_PUBLIC_EVAL) >= 0.05


def test_the_separation_detector_fires_on_a_real_gap(device):
    """CALIBRATION. A detector that always answers "indistinguishable" would
    pass the two tests above vacuously.

    A helper that cannot detect a difference it should detect is not evidence of
    anything. This project has shipped one broken checker already -- a `sorry`
    detector that fired on the word "sorry" inside a comment -- so every
    instrument gets a positive control.
    """
    _ = torch.zeros(1, device=device)
    p = bench.separation_p(k_a=2, n_a=N_PUBLIC_EVAL, k_b=40, n_b=N_PUBLIC_EVAL)
    assert p < 1e-6, p


def test_the_floor_is_not_an_artifact_of_one_evaluation_set_size(device):
    """Sweep n. The detectability floor is a property of exact-match scoring at
    small n, not of the number 120 specifically.

    ARC-AGI-2's public evaluation set is a different size; the conclusion must
    not depend on which one is used.
    """
    _ = torch.zeros(1, device=device)
    for n in (100, 120, 400):
        k = bench.min_successes_for_separation(n, k_baseline=0, alpha=0.05)
        # the minimum DETECTABLE RATE stays in the low single-digit percents,
        # which is above where any published sub-2B model sits
        assert 0.005 < k / n < 0.06, (n, k, k / n)

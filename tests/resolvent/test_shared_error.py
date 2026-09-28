"""Bed `shift`: exact shared-error latent bed with a Monte-Carlo Bayes ceiling (learning-free contracts).

Ported from sun/rjepa/foreman/test_rjepa.py (B0, B1, B3, B10 and the hull-angle invariants).
"""
import numpy as np

from rjepa import shared_error as S
from rjepa.reproduce import cell


def _bed(sigma, seed, mix=0.0):
    return cell(sigma, seed, mix=mix)


def test_bed_shapes_and_truth_label():
    b = S.make_bed(10, 1.0, seed=0)
    assert b["z"].shape == (10, S.K, S.D) and b["best"].shape == (10,)
    assert (b["best"] == np.linalg.norm(b["z"], axis=-1).argmin(1)).all()


def test_bed_is_deterministic_in_its_seed():
    a, b = S.make_bed(50, 3.0, seed=4), S.make_bed(50, 3.0, seed=4)
    assert all(np.array_equal(a[k], b[k]) for k in ("y", "a", "z"))


def test_B0_null_idiosyncratic_error_bayes_equals_distance():
    """No shared error (sigma = 0): the Bayes rule and nearest-to-goal coincide."""
    b = S.make_bed(4000, sigma=0.0, sigma_e=0.3, seed=0)
    assert (S.bayes_pick(b) == S.dist_pick(b["zhat_true"])).mean() >= 0.97


def test_B1_ceiling_small_error():
    b, bp = _bed(0.1, 0)
    assert S.hit(bp, b) >= 0.90


def test_bayes_at_least_every_learning_free_arm():
    """The ceiling is a ceiling: distance, hull-angle and the bound-restricted rule never beat it
    (3-seed mean, tolerance 0.005 for the M = 1024 Monte-Carlo estimate)."""
    for sigma in (1.0, 10.0):
        rows = []
        for sd in (0, 1, 2):
            b, bp = _bed(sigma, sd)
            rows.append((S.hit(bp, b), S.hit(S.dist_pick(b["zhat_true"]), b), S.hit(S.hull_angle_pick(b), b),
                         S.hit(cell(sigma, sd, max_rank=1)[1], b)))
        bayes, *arms = np.mean(rows, 0)
        assert all(a <= bayes + 0.005 for a in arms), (sigma, bayes, arms)


def test_B3_distance_below_chance_past_the_horizon_and_hull_angle_reaches_bayes():
    d, h, y = [], [], []
    for sd in (0, 1, 2):
        b, bp = _bed(10.0, sd)
        d.append(S.hit(S.dist_pick(b["zhat_true"]), b))
        h.append(S.hit(S.hull_angle_pick(b), b))
        y.append(S.hit(bp, b))
    assert np.mean(d) <= 0.23 and np.mean(h) >= np.mean(y) - 0.02, (d, h, y)


def test_B10_gap_is_carried_by_the_shared_share_of_error():
    gap = {}
    for m in (0.0, 0.5, 1.0):
        g = []
        for sd in (0, 1, 2):
            b, bp = _bed(3.0, sd, m)
            g.append(S.hit(bp, b) - S.hit(S.dist_pick(b["zhat_true"]), b))
        gap[m] = float(np.mean(g))
    assert gap[0.0] >= 0.10 and gap[1.0] <= 0.01 and gap[0.0] > gap[0.5] > gap[1.0], gap


def test_hull_angle_share_invariants_and_ground_truth():
    rng = np.random.default_rng(0)
    P = rng.standard_normal((50, 4, 2))
    sh = S.hull_angle_share(P)
    assert np.allclose(sh.sum(-1), 1)
    perm = [3, 1, 0, 2]
    assert np.allclose(S.hull_angle_share(P[:, perm]), sh[:, perm])                  # permutation
    th = 0.7
    Q = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    assert np.abs(S.hull_angle_share(P @ Q.T + 5.0) - sh).max() <= 2 / 4096 + 1e-12   # isometry, up to the grid
    assert np.allclose(S.hull_angle_share(3.0 * P), sh)                               # scale
    sq = np.array([[[1, 0], [0, 1], [-1, 0], [0, -1]]], float)
    assert np.allclose(S.hull_angle_share(sq), 0.25, atol=1e-3)                       # square: 1/4 each
    tri = np.array([[[1, 0], [-0.5, 0.8], [-0.5, -0.8], [0, 0]]], float)
    assert S.hull_angle_share(tri)[0, 3] == 0                                         # interior point: 0

"""Bed `torus`: standard-map candidate bed with an exact posterior, the LIN and LIN1 chance rankers.

Ported from experiments/torus/test_rj.py (bed and ranker contracts; the H0 topology tests stay behind
with the killed TOPO claim).
"""
import numpy as np

from resolvent import standard_map as M

TWO_PI = 2 * np.pi


def test_map_area_preserving_and_tangent_jacobian_matches_analytic():
    x = np.random.default_rng(0).uniform(0, TWO_PI, (50, 2))
    J = M.jac(x, 3)
    assert np.allclose(np.linalg.det(J), 1.0, atol=1e-5)
    A, y = np.eye(2)[None].repeat(50, 0), x.copy()
    for _ in range(3):
        c = M.KS * np.cos(y[:, 0])
        Js = np.stack([np.stack([1 + c, np.ones(50)], 1), np.stack([c, np.ones(50)], 1)], 1)
        A, y = Js @ A, M.step(y)
    assert np.allclose(J, A, rtol=1e-4, atol=1e-5)


def test_torus_distance_wraps():
    assert np.isclose(M.tdist(np.array([[0.1, 0.1]]), np.array([TWO_PI - 0.1, 0.1]))[0], 0.2)


def test_lin_matches_bayes_in_linear_regime():
    rng = np.random.default_rng(1)
    b = M.make_batch(rng, 400, T=1, delta=0.05, M=512)
    assert np.abs(M.lin_prob(b, rng) - b["P_bayes"]).mean() < 0.03


def test_lin1_agrees_with_bayes_in_linear_gaussian_limit():
    """One VJP per candidate: Phi((r - d)/(delta ||J^T n||)) picks the Bayes candidate when the map is near-linear."""
    rng = np.random.default_rng(3)
    b = M.make_batch(rng, 2000, T=1, delta=0.05, M=512)
    assert (M._tb(M.lin1_prob(b), b["d"]) == M._tb(b["P_bayes"], b["d"])).mean() > 0.9


def test_lin1_probability_is_a_probability():
    p = M.lin1_prob(M.make_batch(np.random.default_rng(4), 200, T=2, delta=1e-3, need_post=False))
    assert p.shape == (200, M.K) and ((p >= 0) & (p <= 1)).all()


def test_label_shuffle_null_every_policy_is_blind():
    rng = np.random.default_rng(2)
    b = M.make_batch(rng, 4000, T=8, delta=1e-3, M=64)
    succ = b["succ"]
    sh = np.take_along_axis(succ, rng.permuted(np.tile(np.arange(M.K), (len(succ), 1)), axis=1), 1)
    blind = sh.mean()
    for pick in (M.pick_floor(b), M.pick_bayes(b), M.pick_lin(b, rng)):
        assert abs(sh[np.arange(len(sh)), pick].mean() - blind) < 0.02


def test_bayes_at_least_every_ranker():
    rng = np.random.default_rng(5)
    b = M.make_batch(rng, 4000, T=8, delta=1e-3, M=256)
    bayes = M.score(b, M.pick_bayes(b))
    for pick in (M.pick_floor(b), M.pick_ens3(b), M.pick_lin(b, rng), M._tb(M.lin1_prob(b), b["d"])):
        assert M.score(b, pick) <= bayes + 0.01

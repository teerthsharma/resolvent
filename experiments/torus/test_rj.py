# RED-first tests for experiments/torus/rj.py. Run: python -m pytest -q test_rj.py
import numpy as np
import pytest
import torch
from hypothesis import given, settings, strategies as st
from ripser import ripser
from persim import bottleneck

import rj

TWO_PI = 2 * np.pi


def clouds(n=(5, 30)):
    return st.tuples(st.integers(*n), st.integers(0, 2**31 - 1)).map(
        lambda a: np.random.default_rng(a[1]).uniform(0, TWO_PI, (a[0], 2)))


def dgm(bars):                                   # finite H0 deaths -> (birth, death) diagram
    return np.stack([np.zeros_like(bars), bars], 1)


def small_eucl(seed, n):                         # cloud inside a small patch: torus metric == Euclidean
    return np.random.default_rng(seed).uniform(2.0, 2.5, (n, 2))


# ---------- TDA invariants (tda-tdd) ----------

@settings(max_examples=60, deadline=None)
@given(clouds(), st.integers(0, 10**6))
def test_h0_permutation_invariance(P, s):
    Q = P[np.random.default_rng(s).permutation(len(P))]
    assert np.allclose(rj.h0_bars(P), rj.h0_bars(Q), atol=1e-12)


@settings(max_examples=60, deadline=None)
@given(clouds(), st.floats(0, TWO_PI), st.floats(0, TWO_PI), st.booleans(), st.booleans())
def test_h0_torus_isometry_invariance(P, a, b, swap, refl):
    Q = (P + [a, b]) % TWO_PI
    if swap:
        Q = Q[:, ::-1]
    if refl:
        Q = (-Q) % TWO_PI
    assert np.allclose(rj.h0_bars(P), rj.h0_bars(Q), atol=1e-9)


@settings(max_examples=40, deadline=None)
@given(st.integers(0, 10**6), st.integers(4, 25), st.floats(0.1, 3.0))
def test_h0_scale_equivariance_euclidean(s, n, c):
    P = small_eucl(s, n)
    assert np.allclose(rj.h0_bars(P * c, metric="euclid"), c * rj.h0_bars(P, metric="euclid"), atol=1e-9)


@settings(max_examples=60, deadline=None)
@given(clouds(), st.floats(1e-4, 0.1), st.integers(0, 10**6))
def test_h0_stability_bottleneck_2eps(P, eps, s):
    rng = np.random.default_rng(s)
    v = rng.standard_normal(P.shape); v *= eps * rng.uniform(0, 1, (len(P), 1)) / np.linalg.norm(v, axis=1, keepdims=True)
    d = bottleneck(dgm(rj.h0_bars(P)), dgm(rj.h0_bars((P + v) % TWO_PI)))
    assert d <= 2 * eps + 1e-9


@pytest.mark.parametrize("n", [2, 3, 5])
def test_h0_known_clusters_and_blob_negative_control(n):
    rng = np.random.default_rng(n)
    centers = np.stack([np.linspace(0.5, 5.5, n), np.full(n, 3.0)], 1)[:n]
    P = np.concatenate([c + 0.02 * rng.standard_normal((20, 2)) for c in centers])
    bars = rj.h0_bars(P)
    gap = 5.0 / max(n - 1, 1) if n > 1 else 1.0
    assert (bars > 0.5 * min(gap, 1.0)).sum() == n - 1          # n components -> n-1 long finite bars
    blob = 3.0 + 0.02 * rng.standard_normal((20 * n, 2))
    assert (rj.h0_bars(blob) > 0.5 * min(gap, 1.0)).sum() == 0   # negative control: no long bars


@settings(max_examples=40, deadline=None)
@given(st.integers(0, 10**6), st.integers(3, 30))
def test_h0_parity_ripser_euclid_and_torus(s, n):
    P = small_eucl(s, n)
    r = np.sort(ripser(P, maxdim=0)["dgms"][0][:-1, 1])
    assert np.allclose(rj.h0_bars(P, metric="euclid"), r, atol=1e-6)
    Q = np.random.default_rng(s).uniform(0, TWO_PI, (n, 2))
    r2 = np.sort(ripser(rj.torus_dmat(Q), maxdim=0, distance_matrix=True)["dgms"][0][:-1, 1])
    assert np.allclose(rj.h0_bars(Q), r2, atol=1e-6)


# ---------- bed ----------

def test_map_area_preserving_and_tangent_jacobian_matches_analytic():
    x = np.random.default_rng(0).uniform(0, TWO_PI, (50, 2))
    J = rj.jac(x, 3)
    assert np.allclose(np.linalg.det(J), 1.0, atol=1e-5)
    A = np.eye(2)[None].repeat(50, 0); y = x.copy()             # analytic chain rule
    for _ in range(3):
        c = rj.KS * np.cos(y[:, 0])
        Js = np.stack([np.stack([1 + c, np.ones(50)], 1), np.stack([c, np.ones(50)], 1)], 1)
        A = Js @ A; y = rj.step(y)
    assert np.allclose(J, A, rtol=1e-4, atol=1e-5)


def test_torus_distance():
    assert np.isclose(rj.tdist(np.array([[0.1, 0.1]]), np.array([TWO_PI - 0.1, 0.1]))[0], 0.2)


def test_lin_matches_bayes_in_linear_regime():
    rng = np.random.default_rng(1)
    b = rj.make_batch(rng, 400, T=1, delta=0.05, need_post=True, M=512)
    P_lin = rj.lin_prob(b, rng)
    assert np.abs(P_lin - b["P_bayes"]).mean() < 0.03


def test_label_shuffle_null_every_policy_is_blind():
    rng = np.random.default_rng(2)
    b = rj.make_batch(rng, 4000, T=8, delta=1e-3, need_post=True, M=64)
    succ = b["succ"]
    sh = np.take_along_axis(succ, rng.permuted(np.tile(np.arange(rj.K), (len(succ), 1)), axis=1), 1)
    blind = sh.mean()
    for pick in (rj.pick_floor(b), rj.pick_bayes(b), rj.pick_lin(b, rng)):
        assert abs(sh[np.arange(len(sh)), pick].mean() - blind) < 0.02


def test_dj_operator_permutation_equivariant_and_bounded():
    torch.manual_seed(0)
    net = rj.DJ(nin=5)
    for p_ in net.parameters():                       # move off the zero-init head so the test has teeth
        torch.nn.init.normal_(p_, std=0.5)
    v = torch.randn(7, rj.K, 5); b = torch.rand(7, rj.K)
    s, dl = net(v, b)
    p = torch.randperm(rj.K)
    s2, _ = net(v[:, p], b[:, p])
    assert torch.allclose(s2, s[:, p], atol=1e-5)
    assert (s - b).abs().max() <= 0.2 + 1e-6 and dl.abs().max() > 1e-3


def test_lin1_one_vjp_agrees_with_lin_in_linear_regime():
    rng = np.random.default_rng(3)
    b = rj.make_batch(rng, 2000, T=1, delta=0.05, need_post=True, M=512)
    agree = (rj._tb(rj.lin1_prob(b), b["d"]) == rj._tb(b["P_bayes"], b["d"])).mean()
    assert agree > 0.9

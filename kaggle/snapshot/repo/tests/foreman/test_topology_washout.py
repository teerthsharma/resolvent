"""Is the topology load-bearing, or is it vocabulary wrapped around linear algebra?

THEORY.md sec 6 states the pipeline as
    input window -> barcode -> Hilbert-series embedding -> hierarchical partition
    -> CSR block schedule -> scheduled attention -> resolvent solve (I-gamma P)^-1 b
so the barcode is computed UPSTREAM of the resolvent. This file asks whether the
resolvent stage preserves the topological distinction the barcode stage measured.

Per tda-tdd, the measuring instrument is calibrated FIRST -- permutation invariance,
isometry invariance, scale equivariance, the stability theorem, the circle's sqrt(3)*r
Rips death time, and the Gaussian-blob negative control -- and only then used to make
a claim. Reference implementation is ripser (pinned below).
"""

import numpy as np
import pytest
import torch
import ripser
from persim import bottleneck

from _device import DEVICES, t, n

RNG = np.random.default_rng(31415)
RIPSER_VERSION = "0.6.14"


# --------------------------------------------------------------------------
# instrument
# --------------------------------------------------------------------------

def ph(X, maxdim=1):
    d = ripser.ripser(np.asarray(X, dtype=float), maxdim=maxdim)["dgms"]
    return [np.asarray(x)[np.isfinite(np.asarray(x)[:, 1])] for x in d]


def h1(X):
    return ph(X)[1]


def max_persistence(dgm):
    if len(dgm) == 0:
        return 0.0
    return float(np.max(dgm[:, 1] - dgm[:, 0]))


def unit_scale(X):
    """PH is scale-equivariant (verified below), so normalize away the trivial
    1/(1-gamma) gain before comparing shapes."""
    X = X - X.mean(axis=0)
    s = np.linalg.norm(X, axis=1).max()
    return X / max(s, 1e-12)


def circle(n=60, r=1.0, rng=RNG, noise=0.0):
    t = np.sort(rng.uniform(0, 2 * np.pi, n))
    X = np.column_stack([r * np.cos(t), r * np.sin(t)])
    return X + noise * rng.standard_normal(X.shape)


def blob(n=60, rng=RNG):
    return rng.standard_normal((n, 2)) * 0.5


def knn_row_stochastic(X, k=6, device="cpu"):
    """The natural occupancy operator over a point cloud: a k-NN random walk."""
    Xi = t(X, device)
    D = torch.linalg.norm(Xi[:, None, :] - Xi[None, :, :], dim=-1)
    D.fill_diagonal_(float("inf"))
    idx = torch.argsort(D, dim=1)[:, :k]
    P = torch.zeros_like(D)
    P.scatter_(1, idx, 1.0 / k)
    return n(P)


def normalized_resolvent(P, gamma, device="cpu"):
    """(1-gamma)(I - gamma P)^-1 -- row-stochastic, i.e. the PPR/diffusion operator."""
    Pi = t(P, device)
    eye = torch.eye(Pi.shape[0], dtype=torch.float64, device=device)
    return n((1 - gamma) * torch.linalg.inv(eye - gamma * Pi))


# ==========================================================================
# CALIBRATION (tda-tdd invariants 1-6) -- GREEN before any claim is made
# PH backend (ripser) is a CPU-only C extension; parametrized only so node
# ids stay uniform across the file, not because the computation moves.
# ==========================================================================

@pytest.mark.parametrize("device", DEVICES)
def test_instrument_permutation_invariance(device):
    X = circle(40, noise=0.05)
    Y = X[RNG.permutation(len(X))]
    assert bottleneck(h1(X), h1(Y)) < 1e-9


@pytest.mark.parametrize("device", DEVICES)
def test_instrument_isometry_invariance(device):
    X = circle(40, noise=0.05)
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    Y = X @ R.T + np.array([3.0, -2.0])
    assert bottleneck(h1(X), h1(Y)) < 1e-9


@pytest.mark.parametrize("device", DEVICES)
def test_instrument_scale_equivariance(device):
    X = circle(40, noise=0.05)
    c = 3.7
    # ripser accumulates in float32; tolerance is relative to the scaled diagram
    assert bottleneck(h1(c * X), c * h1(X)) < 1e-5 * c


@pytest.mark.parametrize("device", DEVICES)
def test_instrument_stability_under_perturbation(device):
    """Cohen-Steiner et al.: bottleneck <= 2*eps for Vietoris-Rips."""
    X = circle(30, noise=0.02)
    for eps in (1e-3, 1e-2, 5e-2):
        v = RNG.standard_normal(X.shape)
        v = v / np.linalg.norm(v, axis=1, keepdims=True) * eps * RNG.uniform(0, 1, (len(X), 1))
        assert bottleneck(h1(X), h1(X + v)) <= 2 * eps + 1e-9


@pytest.mark.parametrize("device", DEVICES)
def test_instrument_circle_ground_truth_dies_at_sqrt_three_r(device):
    r = 1.0
    d = h1(circle(80, r=r))
    assert len(d) >= 1
    longest = d[np.argmax(d[:, 1] - d[:, 0])]
    assert abs(longest[1] - np.sqrt(3) * r) < 0.05, f"death {longest[1]:.4f} vs {np.sqrt(3)*r:.4f}"


@pytest.mark.parametrize("device", DEVICES)
def test_instrument_gaussian_blob_negative_control(device):
    assert max_persistence(h1(blob(80))) < 0.35


# ==========================================================================
# CLAIM AS WRITTEN -- RED
# ==========================================================================

@pytest.mark.parametrize("device", DEVICES)
def test_claim_the_resolvent_stage_preserves_the_barcode_distinction_it_is_fed(device):
    """A circle (beta_1 = 1) and a blob (beta_1 = 0) are the cleanest possible
    topological distinction. Push both through the occupancy operator and ask
    whether a barcode downstream can still tell them apart."""
    C, B = circle(70, noise=0.03), blob(70)
    sep_in = abs(max_persistence(h1(unit_scale(C))) - max_persistence(h1(unit_scale(B))))
    rows = []
    for gamma in (0.5, 0.9, 0.99):
        RC = normalized_resolvent(knn_row_stochastic(C, device=device), gamma, device=device) @ C
        RB = normalized_resolvent(knn_row_stochastic(B, device=device), gamma, device=device) @ B
        sep_out = abs(max_persistence(h1(unit_scale(RC))) - max_persistence(h1(unit_scale(RB))))
        rows.append((gamma, sep_out, sep_out / max(sep_in, 1e-12)))
    print("\n  input circle-vs-blob H1 separation: %.4f" % sep_in)
    for g, s, f in rows:
        print("  gamma=%.2f  output separation %.4f  (%.1f%% retained)" % (g, s, 100 * f))
    assert rows[-1][2] > 0.5, (
        f"at gamma=0.99 only {100*rows[-1][2]:.1f}% of the circle-vs-blob "
        f"topological separation survives the resolvent"
    )


@pytest.mark.parametrize("device", DEVICES)
def test_claim_the_circle_survives_the_occupancy_operator(device):
    """The H1 bar is the signal. Does it live through the smoothing?"""
    C = circle(70, noise=0.03)
    base = max_persistence(h1(unit_scale(C)))
    out = []
    for gamma in (0.5, 0.9, 0.99):
        RC = normalized_resolvent(knn_row_stochastic(C, device=device), gamma, device=device) @ C
        out.append((gamma, max_persistence(h1(unit_scale(RC)))))
    print("  H1 persistence: input %.4f -> " % base
          + ", ".join("g=%.2f:%.4f" % o for o in out))
    assert out[-1][1] > 0.5 * base, (
        f"H1 persistence falls from {base:.4f} to {out[-1][1]:.4f} at gamma=0.99"
    )


# ==========================================================================
# WHERE THE TOPOLOGY ACTUALLY ENTERS -- GREEN
# ==========================================================================

@pytest.mark.parametrize("device", DEVICES)
def test_the_resolvent_reads_the_graph_while_the_barcode_reads_the_metric(device):
    """A monotone reparameterization of distance d -> d^1.5 preserves every k-NN
    ranking exactly, so P and hence the whole resolvent stage is UNCHANGED, while
    the barcode moves. The two stages of the sec 6 pipeline read different objects:
    the barcode is never an input to the resolvent."""
    C = circle(50, noise=0.02)
    D = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=-1)
    D2 = D ** 1.5                       # strictly monotone, rank-preserving

    def knn_from_D(D, k=6):
        Dw = D.copy()
        np.fill_diagonal(Dw, np.inf)
        P = np.zeros_like(Dw)
        np.put_along_axis(P, np.argsort(Dw, axis=1)[:, :k], 1.0 / k, axis=1)
        return P

    assert np.array_equal(knn_from_D(D), knn_from_D(D2)), "k-NN graph must be identical"

    def h1_from_D(D):
        d = ripser.ripser(D, distance_matrix=True, maxdim=1)["dgms"][1]
        return np.asarray(d)[np.isfinite(np.asarray(d)[:, 1])]

    b1, b2 = h1_from_D(D), h1_from_D(D2)
    assert bottleneck(b1, b2) > 0.1, "barcodes must differ under the reparameterization"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s", "--tb=short", "-p", "no:cacheprovider"]))

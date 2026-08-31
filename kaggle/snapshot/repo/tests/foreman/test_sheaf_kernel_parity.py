"""THEORY.md sec 8 risk 2: 'ker Delta_F may be trivial. For a connected sheaf with
generic restriction maps, the harmonic space is frequently {0}. An equilibrium
defined as ker Delta_F that turns out to be zero carries no signal. Unverified.'

It is verifiable in forty lines, and the answer is sharper than 'frequently'.

Definitions transcribed from the primary sources:
  Bodnar et al. 2022, Definition 2:
      L_F(x)_v = sum_{v,u <| e} F_{v<|e}^T ( F_{v<|e} x_v - F_{u<|e} x_u )
  Bodnar et al. 2022, Lemma 6:
      'Let F be a discrete O(d) bundle over a connected graph G. Then dim(H^0) <= d
       and dim(H^0) = d if and only if the transport is path-independent.'
  Bodnar et al. 2022, Proposition 4:
      'for any cycle gamma based at v we have x_v in ker(P^gamma_{v->v} - I).'
"""

import numpy as np
import pytest
import torch

from _device import DEVICES, t, n

RNG = np.random.default_rng(4242)


def random_orthogonal(d, rng, special=True):
    Q, R = np.linalg.qr(rng.standard_normal((d, d)))
    Q = Q * np.sign(np.diag(R))
    if special and np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def sheaf_laplacian(edges, n_nodes, d, rng, special=True, device="cpu"):
    """Delta_F = delta^T delta with (delta x)_e = F_{v<|e} x_v - F_{u<|e} x_u."""
    delta = torch.zeros((len(edges) * d, n_nodes * d), dtype=torch.float64, device=device)
    for i, (u, v) in enumerate(edges):
        Fu = t(random_orthogonal(d, rng, special), device)
        Fv = t(random_orthogonal(d, rng, special), device)
        delta[i * d:(i + 1) * d, v * d:(v + 1) * d] = Fv
        delta[i * d:(i + 1) * d, u * d:(u + 1) * d] = -Fu
    return n(delta.T @ delta)


def dim_kernel(L, tol=1e-11, device="cpu"):
    w = torch.linalg.eigvalsh(t(L, device))
    wmax = float(w.max())
    return int(torch.sum(w < tol * max(1.0, wmax)))


def cycle_edges(n):
    return [(i, (i + 1) % n) for i in range(n)]


# ==========================================================================
# CLAIM AS WRITTEN -- RED
# ==========================================================================

@pytest.mark.parametrize("device", DEVICES)
def test_claim_the_harmonic_equilibrium_carries_signal_for_generic_restriction_maps(device):
    """THEORY.md sec 6 puts a 'sheaf consistency gate' in the forward path and sec 8
    leaves its kernel dimension unverified. Ask for at least one dimension of signal
    on a connected graph with a cycle and generic SO(d) restriction maps."""
    results = {}
    for d in (2, 4, 6, 8):
        dims = [dim_kernel(sheaf_laplacian(cycle_edges(7), 7, d, RNG, device=device), device=device)
                 for _ in range(20)]
        results[d] = dims
        print("\n  d=%d  dim ker Delta_F over 20 draws: %s" % (d, sorted(set(dims))))
    worst = min(min(v) for v in results.values())
    assert worst >= 1, (
        f"dim ker Delta_F = {worst} for even stalk dimension: the harmonic "
        f"equilibrium is exactly {{0}}, so the sec 6 gate carries no signal"
    )


# ==========================================================================
# THE ACTUAL LAW -- GREEN.  It is a parity law, not a frequency.
# ==========================================================================

@pytest.mark.parametrize("device", DEVICES)
def test_generic_kernel_dimension_on_a_cycle_is_exactly_the_stalk_dimension_parity(device):
    """A generic rotation in R^d has a +1 eigenvector iff d is odd (the axis).
    By Bodnar Prop. 4 the harmonic space is ker(holonomy - I), so
        dim ker Delta_F = d mod 2   for a connected graph with a generic SO(d) cycle."""
    for d in range(2, 8):
        for _ in range(10):
            k = dim_kernel(sheaf_laplacian(cycle_edges(6), 6, d, RNG, device=device), device=device)
            assert k == d % 2, f"d={d}: expected {d % 2}, got {k}"


@pytest.mark.parametrize("device", DEVICES)
def test_a_tree_has_full_dimensional_harmonic_space_because_it_has_no_holonomy(device):
    """No cycle means path-independent transport, so Bodnar Lemma 6 gives dim = d.
    The gate carries signal only where the graph has no cycles -- i.e. where there
    is no topology to detect."""
    for d in (2, 3, 4, 5):
        edges = [(i, i + 1) for i in range(6)]        # path graph, 7 nodes
        assert dim_kernel(sheaf_laplacian(edges, 7, d, RNG, device=device), device=device) == d


@pytest.mark.parametrize("device", DEVICES)
def test_the_parity_law_is_a_property_of_the_holonomy_not_of_the_graph_size(device):
    for n in (3, 5, 9, 17):
        for d in (2, 3):
            assert dim_kernel(sheaf_laplacian(cycle_edges(n), n, d, RNG, device=device), device=device) == d % 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s", "--tb=short", "-p", "no:cacheprovider"]))

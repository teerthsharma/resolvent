"""M14 CHEEGER STRATIFICATION -- the exact sweep-cut conductance and its must-fires.

WHAT THIS FILE IS FOR. The M14 claim is `phi^2/2 <= 1 - lambda_2 <= 2 phi`, offered
as a corpus-difficulty dial. The only conductance in the tree before this file is
`scale/foreman_lambda2.py:300 bridge_conductance`, which evaluates ONE named cut and
is therefore an UPPER bound on the Cheeger constant `h` and never a value for it.
An upper bound on `h` composes with `gamma <= 2h` and with nothing else; substituting
it into the LOWER half `h^2/2 <= gamma` is not licensed by the theorem, and the
must-fire below shows that substitution failing by a measured factor.

WHICH THEOREM, WITH ITS HYPOTHESES. Levin and Peres with Wilmer, *Markov Chains and
Mixing Times*, 2nd ed., Theorem 13.10 at page 183 (Cheeger's inequality, attributed
there to Sinclair-Jerrum 1989 and Lawler-Sokal 1988). Its own sentence states
REVERSIBILITY only; irreducibility enters through the chapter's eigenvalue-ordering
convention at equation (12.7), which is what makes `lambda_2` well defined. For such
a `P`, reversible with respect to `pi`,

    Phi_*^2 / 2  <=  gamma  <=  2 Phi_*

with `gamma := 1 - lambda_2` the spectral gap (`lambda_2` the second largest
EIGENVALUE, not the second largest modulus) and

    Phi_* := min { Q(S, S^c) / pi(S)  :  pi(S) <= 1/2 },
    Q(S, S^c) = sum_{x in S, y in S^c} pi(x) P(x, y).

On a connected undirected weighted graph with `W >= 0`, `P = D^{-1} W`, `pi = d/vol`,
this specialises to `Phi(S) = cut(S) / min(vol S, vol S^c)`, and `gamma` is the second
smallest eigenvalue of the normalised Laplacian `I - D^{-1/2} W D^{-1/2}`.

LAZINESS IS NOT A HYPOTHESIS OF THIS THEOREM, AND THAT IS A TRAP. Cheeger constrains
`lambda_2` only. The most negative eigenvalue -- which is what laziness fixes, and
which controls the ABSOLUTE spectral gap and hence the mixing time -- is untouched by
it. A bipartite graph can have `gamma` large and still never mix. That gap is
measured in `test_the_gap_cheeger_bounds_is_not_the_mixing_gap` below, because M14
names its dial "mixing".

TOLERANCES, AND WHY THESE AND NOT 1e-9-BECAUSE-IT-PASSED.
  * `numpy.linalg.eigvalsh` on `S = D^{-1/2} W D^{-1/2}` is backward stable and
    `||S||_2` is exactly 1 (Perron), so the eigenvalue error is bounded by
    `c * n * eps` with `eps = 2.220446e-16`. At the largest `n` this file runs (62)
    that is `62 * 2.22e-16 = 1.4e-14`; `c` of order 10 gives `1.4e-13`.
  * `cut(S)` and `vol(S)` are sums of at most `n^2` NON-NEGATIVE float64 terms, so
    there is no cancellation and the relative error is at most `n^2 * eps`
    = `3844 * 2.22e-16 = 8.5e-13`; the quotient doubles it to `1.7e-12`.
  * `TOL = 1e-11` therefore dominates both error sources by more than 5x while
    sitting 9 orders of magnitude below the quantities compared, which are 1e-3..1e0.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v20_m14_cheeger import (  # noqa: E402
    BRUTE_FORCE_MAX_N,
    absolute_spectral_gap,
    exact_min_conductance,
    named_cut_conductance,
    repo_chain_weights,
    spectral_gap,
    sweep_cut_conductance,
)

#: See the tolerance derivation in the module docstring. Not tuned to pass.
TOL = 1e-11


# --------------------------------------------------------------------------
# graph battery -- every one connected, undirected, non-negative weights
# --------------------------------------------------------------------------
def _path(n: int) -> np.ndarray:
    W = np.zeros((n, n))
    for i in range(n - 1):
        W[i, i + 1] = W[i + 1, i] = 1.0
    return W


def _cycle(n: int) -> np.ndarray:
    W = _path(n)
    W[0, n - 1] = W[n - 1, 0] = 1.0
    return W


def _complete(n: int) -> np.ndarray:
    return np.ones((n, n)) - np.eye(n)


def _star(n: int) -> np.ndarray:
    W = np.zeros((n, n))
    W[0, 1:] = W[1:, 0] = 1.0
    return W


def _dumbbell(m: int, w: float = 1.0) -> np.ndarray:
    """Two `K_m` joined by ONE edge of weight `w`. The bottleneck is known in closed
    form: the cut is `w` and each lobe has volume `m(m-1) + w`, so the bridge cut has
    conductance `w / (m(m-1) + w)`."""
    n = 2 * m
    W = np.zeros((n, n))
    W[:m, :m] = _complete(m)
    W[m:, m:] = _complete(m)
    W[m - 1, m] = W[m, m - 1] = w
    return W


def _hypercube(k: int) -> np.ndarray:
    n = 1 << k
    W = np.zeros((n, n))
    for v in range(n):
        for b in range(k):
            W[v, v ^ (1 << b)] = 1.0
    return W


BATTERY = {
    "path_6": _path(6),
    "path_12": _path(12),
    "cycle_8": _cycle(8),
    "cycle_14": _cycle(14),
    "complete_8": _complete(8),
    "star_10": _star(10),
    "dumbbell_4x4": _dumbbell(4),
    "dumbbell_6x6": _dumbbell(6),
    "dumbbell_6x6_w05": _dumbbell(6, 0.5),
    "hypercube_16": _hypercube(4),
}


# --------------------------------------------------------------------------
# 1. the two independent numeric paths
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(BATTERY))
def test_sweep_cut_is_an_upper_bound_on_the_exact_minimum(name):
    """PATH A (sweep cut over the Fiedler ordering) against PATH B (brute force over
    all `2^(n-1)` cuts). They fail differently: A can only ever be too LARGE, since it
    searches `n-1` of the `2^(n-1)` cuts, while B is exact but exponential. So
    `A >= B` always, and any `A < B` is a bug in one of the two."""
    W = BATTERY[name]
    assert W.shape[0] <= BRUTE_FORCE_MAX_N
    a = sweep_cut_conductance(W)["phi"]
    b = exact_min_conductance(W)["phi"]
    assert a >= b - TOL, (name, a, b)


@pytest.mark.parametrize("name", sorted(BATTERY))
def test_the_sweep_cut_obeys_its_own_quadratic_guarantee(name):
    """The Cheeger PROOF gives more than `gamma <= 2 phi`: the sweep over the Fiedler
    ordering returns a cut with `phi_sweep <= sqrt(2 gamma)`. Asserting it checks the
    sweep implementation against the theorem rather than against itself."""
    W = BATTERY[name]
    g = spectral_gap(W)
    a = sweep_cut_conductance(W)["phi"]
    assert a <= math.sqrt(2.0 * g) + TOL, (name, a, g, math.sqrt(2.0 * g))


# --------------------------------------------------------------------------
# 2. the inequality itself, with the EXACT phi
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(BATTERY))
def test_cheeger_two_sided_holds_with_the_exact_phi(name):
    """`phi^2/2 <= gamma <= 2 phi` -- the M14 statement -- with `phi` the true minimum
    over all cuts and `gamma` from an independent eigensolve."""
    W = BATTERY[name]
    phi = exact_min_conductance(W)["phi"]
    g = spectral_gap(W)
    assert phi * phi / 2.0 <= g + TOL, (name, phi, g)
    assert g <= 2.0 * phi + TOL, (name, phi, g)


# --------------------------------------------------------------------------
# 3. THE MUST-FIRE, with its planted negative
# --------------------------------------------------------------------------
@pytest.mark.parametrize("m", [4, 6, 8])
def test_a_correct_phi_detects_the_planted_bottleneck(m):
    """PLANTED POSITIVE. A dumbbell's bottleneck is known in closed form. Both
    independent paths must return it, not merely something small."""
    W = _dumbbell(m)
    planted = 1.0 / (m * (m - 1) + 1.0)
    assert exact_min_conductance(W)["phi"] == pytest.approx(planted, abs=TOL)
    assert sweep_cut_conductance(W)["phi"] == pytest.approx(planted, abs=TOL)


@pytest.mark.parametrize("m", [4, 6, 8])
def test_a_single_named_cut_MISSES_the_planted_bottleneck(m):
    """PLANTED NEGATIVE -- the check that makes the check above capable of failing.

    A `bridge_conductance`-style evaluation of ONE named cut is handed the WRONG cut
    (the singleton `{0}`, a vertex inside a lobe) on the same graph carrying the same
    planted bottleneck. A correct phi finds `1/(m(m-1)+1)`; the named cut reads the
    singleton's conductance `1` and does not detect the bottleneck at all.

    Without this, `test_a_correct_phi_detects_the_planted_bottleneck` is the V-16
    shape: an instrument reporting a pass with nothing that could make it report
    otherwise."""
    W = _dumbbell(m)
    planted = 1.0 / (m * (m - 1) + 1.0)
    named = named_cut_conductance(W, [0])
    assert named > 10.0 * planted, (m, named, planted)
    # and it is a legitimate UPPER bound, which is exactly the trap: it is never
    # wrong, it is only useless.
    assert named >= exact_min_conductance(W)["phi"] - TOL


@pytest.mark.parametrize("n", [8, 12, 16, 20])
def test_the_lower_half_BREAKS_when_an_upper_bound_on_phi_is_substituted(n):
    """THE REPRODUCTION OF M14's FAILURE MODE, on a graph where it is unambiguous.

    `gamma <= 2 phi` survives substituting any upper bound for `phi`. `phi^2/2 <=
    gamma` does NOT: it needs `phi` itself. On a path the singleton end cut has
    conductance exactly `1` while the true `phi` is about `2/n`, and `1^2/2 = 0.5`
    sits far ABOVE `gamma`. The inequality M14 states is therefore FALSE as an
    operational check whenever `phi` is a named-cut upper bound rather than the
    minimum."""
    W = _path(n)
    g = spectral_gap(W)
    phi_named = named_cut_conductance(W, [0])
    phi_exact = exact_min_conductance(W)["phi"]

    assert phi_exact * phi_exact / 2.0 <= g + TOL      # the theorem holds
    assert phi_named * phi_named / 2.0 > g             # the substitution does not
    assert g <= 2.0 * phi_named + TOL                  # the upper half survives it


# --------------------------------------------------------------------------
# 4. the repo's own crude phi, on the repo's own instance
# --------------------------------------------------------------------------
def test_the_repo_crude_phi_is_an_upper_bound_on_the_exact_minimum():
    """`scale/foreman_lambda2.py:300 bridge_conductance` on `LargestJoin_S2Rips_64`
    (18 nodes, so all `2^17` cuts are enumerable) against the exact minimum."""
    W, meta = repo_chain_weights("LargestJoin_S2Rips_64")
    assert W.shape[0] == 18, meta
    crude = meta["phi_bridge"]
    exact = exact_min_conductance(W)["phi"]
    g = spectral_gap(W)

    assert crude >= exact - TOL, (crude, exact)
    assert exact * exact / 2.0 <= g + TOL, (exact, g)
    assert g <= 2.0 * exact + TOL, (exact, g)


def test_the_repo_crude_phi_is_bracketed_by_the_certificate_free_bounds():
    """`gamma <= 2 phi` inverts to `phi >= gamma/2`, and the sweep cut is an upper
    bound, so `phi` sits in `[gamma/2, phi_sweep]` on ANY graph without enumerating
    anything. On the 18-node instance the bracket must also contain the brute-forced
    truth -- that is what makes the bracket usable at `n = 62`, where brute force is
    not."""
    W, _meta = repo_chain_weights("LargestJoin_S2Rips_64")
    g = spectral_gap(W)
    lo = g / 2.0
    hi = sweep_cut_conductance(W)["phi"]
    exact = exact_min_conductance(W)["phi"]
    assert lo - TOL <= exact <= hi + TOL, (lo, exact, hi)


def test_the_gap_cheeger_bounds_is_not_the_mixing_gap():
    """A HYPOTHESIS CHECK, not a numerical one. Cheeger constrains `1 - lambda_2` and
    says nothing about `lambda_n`. A bipartite graph has `lambda_n = -1`, so its
    ABSOLUTE spectral gap is `0` and it never mixes -- while `gamma` is bounded away
    from `0` and Cheeger reports a healthy bottleneck. Any use of M14 as a MIXING dial
    (which is what "corpus difficulty (mixing)" names) needs laziness, an extra
    hypothesis M14 does not state."""
    B = _complete(4)
    W = np.block([[np.zeros((4, 4)), B], [B, np.zeros((4, 4))]])  # bipartite
    g = spectral_gap(W)
    g_abs = absolute_spectral_gap(W)
    assert g > 0.5, g
    assert g_abs == pytest.approx(0.0, abs=1e-13), g_abs

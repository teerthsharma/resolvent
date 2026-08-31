"""X37 -- the three topological certificates, and the two instruments they need.

Written before `ceq/certs/topological.py` existed. Every threshold in this file
is either frozen with a derivation that does not look at the measurement
(`STABILITY_FACTOR = 4` from the bottleneck stability theorem, `NYQUIST_MARGIN`
from the sampling condition) or is a closed form the reading is checked against
(`r = exp(-sigma^2/2)` for a wrapped-normal phase cloud). MISTAKES.md M-2 is the
rule those two forms exist to satisfy.

Each certificate carries a REJECTION REGION, because MISTAKES.md V-16 is the
defect of an instrument that cannot measure and reports a pass:

  (a) winding      -- an undersampled sequence is REFUSED, and the unguarded
                      reading on the same input is shown to be a plausible
                      integer that is wrong.
  (b) persistent b1 -- the coordination control must read 0 where RPS reads 1.
  (c) Euler-Poincare -- the Poincare-Hopf hypothesis (transversality) is checked
                      on the field rather than assumed, on the same replicator
                      face where MISTAKES.md:1406 recorded it failing.
"""

import math

import numpy as np
import pytest

from ceq.certs import topological as T
from ceq.certs.topological import CertificateRefused


# ---------------------------------------------------------------------------
# (a) Z winding of a phase sequence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [-3, -1, 0, 1, 2, 5, 15])
def test_winding_is_an_exact_integer_on_a_known_k_turn_sequence(k):
    """64 samples across `k` turns; the step is `2 pi |k| / 64`, so `k = 15` at
    `0.4688 pi` is the closest passing case to the `pi/2` refusal and `k = 16`
    lands exactly on it. The sweep includes the marginal case rather than only
    comfortable ones."""
    theta = 2.0 * math.pi * k * np.arange(64) / 64.0
    w = T.winding_number(theta)
    assert isinstance(w, int) and not isinstance(w, bool)
    assert w == k


def test_winding_refuses_an_undersampled_sequence_and_the_unguarded_read_is_wrong():
    """3 turns in 5 samples: every true step is `1.2 pi`, which wraps to
    `-0.8 pi`. The wrapped step is BELOW the naive Nyquist limit of `pi`, so a
    routine guarding only at `pi` accepts the input and returns an integer.
    That integer is not 3."""
    theta = 2.0 * math.pi * 3 * np.arange(5) / 5.0
    with pytest.raises(CertificateRefused, match="undersampled"):
        T.winding_number(theta)
    unguarded = T.winding_number(theta, max_step=math.pi)
    assert isinstance(unguarded, int)
    assert unguarded == -2 and unguarded != 3


def test_winding_refuses_a_path_that_does_not_close():
    """Winding is an integer only for a cycle. Read as an open path, a half
    turn gives `0.5`, and a non-integer reading is an instrument defect rather
    than a result -- so the instrument refuses instead of rounding. Read as a
    cycle, the SAME input is refused for the other reason: closing a half turn
    is a step of `pi`, which is exactly the aliasing regime."""
    theta = math.pi * np.arange(33) / 32.0
    with pytest.raises(CertificateRefused, match="not an integer"):
        T.winding_number(theta, closed=False)
    with pytest.raises(CertificateRefused, match="undersampled"):
        T.winding_number(theta)
    # a path whose ends nearly meet closes cleanly and reads zero
    assert T.winding_number(0.3 * np.sin(np.linspace(0, 2 * math.pi, 64))) == 0


def test_the_step_guard_alone_does_not_catch_every_alias_and_refinement_does():
    """The harder V-16 case, and it is a defect of this module's own first
    guard. 63 turns sampled 64 times wraps to a step of `0.0313 pi` -- the
    healthiest-looking reading in the file -- and the guarded routine returns
    `-1`. No test on one sampling can see it, because a wrapped step of `s` is
    consistent with `s + 2 pi k` for every `k`. A finer sampling of the same
    path can, and the certificate refuses on the disagreement."""
    coarse = 2.0 * math.pi * 63 * np.arange(64) / 64.0
    fine = 2.0 * math.pi * 63 * np.arange(1024) / 1024.0
    d = T.wrapped_differences(coarse)
    assert float(np.abs(d).max()) < T.NYQUIST_MARGIN      # the guard sees nothing
    assert T.winding_number(coarse) == -1                 # and is wrong
    assert T.winding_number(fine) == 63
    with pytest.raises(CertificateRefused, match="not stable under refinement"):
        T.winding_number(coarse, refinement=fine)


@pytest.mark.parametrize("bad", [[0.0], [], [0.0, float("nan"), 1.0],
                                 [0.0, float("inf")]])
def test_winding_refuses_inputs_it_cannot_read(bad):
    with pytest.raises(CertificateRefused):
        T.winding_number(np.asarray(bad, dtype=float))


def test_winding_refuses_a_refinement_that_is_not_finer():
    theta = 2.0 * math.pi * np.arange(64) / 64.0
    with pytest.raises(CertificateRefused, match="not more than"):
        T.winding_number(theta, refinement=theta)


# ---------------------------------------------------------------------------
# Kuramoto order parameter -- both halves
# ---------------------------------------------------------------------------


def test_order_parameter_locks_and_matches_the_wrapped_normal_closed_form():
    """`r -> 1` under phase locking. Checked against `exp(-sigma^2/2)`, the
    wrapped-normal resultant, rather than against "r is large"."""
    rng = np.random.default_rng(0)
    for sigma in (0.02, 0.05, 0.2):
        r = T.order_parameter(rng.normal(0.7, sigma, 65536))
        assert abs(r - math.exp(-sigma * sigma / 2.0)) < 5e-3, (sigma, r)
    assert T.order_parameter(np.full(1024, 1.234)) == pytest.approx(1.0, abs=1e-12)


def test_order_parameter_must_fire_on_phases_that_carry_no_order():
    """MUST-FIRE. An order parameter that reads high on unordered phases is
    measuring nothing. Two controls: N equispaced phases, where the resultant
    is exactly zero, and eight seeds of uniform phases, where the Rayleigh
    prediction is `E[r] = sqrt(pi)/(2 sqrt N)`. The refusal bar is `5/sqrt(N)`
    -- fixed from N alone, not from the readings."""
    n = 4096
    assert T.order_parameter(2 * math.pi * np.arange(n) / n) < 1e-12
    bar = 5.0 / math.sqrt(n)
    for seed in range(8):
        r = T.order_parameter(np.random.default_rng(seed).uniform(0, 2 * math.pi, n))
        assert r < bar, (seed, r, bar)


# ---------------------------------------------------------------------------
# (b) persistent b1 of a carrier trajectory
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rps():
    return T.carrier("rps")


@pytest.fixture(scope="module")
def coordination():
    return T.carrier("coordination")


def test_the_toolkit_edge_rule_and_the_geodesic_metric_are_the_same_filtration(rps):
    """The barcode is computed by an engine that is not `ceq/rips.py`, so the
    two must be shown to filter the same complex: at every radius the toolkit's
    `dot >= cos r` edge set has to equal `{(i,j) : d_geo(i,j) <= r}`."""
    P = rps["cloud"]
    D = T.geodesic_matrix(P)
    for r in (0.05, 0.1, 0.25, 0.4, 0.55, 0.7):
        toolkit = set(T.toolkit_edges(P, r))
        metric = {(i, j) for i in range(len(P)) for j in range(i + 1, len(P))
                  if D[i, j] <= r}
        assert toolkit == metric, (r, len(toolkit ^ metric))


def test_persistent_beta1_is_one_on_rps_and_zero_on_the_coordination_control(
        rps, coordination):
    """The delta's must-fire, `[RUN: 1 vs 0]`. `recurrent` is defined as
    `b1 >= 1` and must follow the reading on both halves."""
    a = T.persistent_beta1(rps["cloud"])
    b = T.persistent_beta1(coordination["cloud"])
    assert a["beta1"] == 1 and a["recurrent"] is True, a
    assert b["beta1"] == 0 and b["recurrent"] is False, b
    # the surviving bar clears its threshold by a stated factor, not by a hair
    assert a["max_persistence"] > 2.0 * a["threshold"], a


def test_the_persistence_threshold_is_geometric_and_never_reads_the_barcode(rps):
    """MISTAKES.md M-2. The threshold is `4 * (max nearest-neighbour distance)`,
    where 4 is the constant in the bottleneck stability theorem (a bar of
    persistence `p` sits `p/2` from the diagonal and `d_B <= 2 d_H`, so any bar
    with `p <= 4 d_H` is inside resampling noise). It is a function of the point
    positions only: an isometry of the cloud must not move it, and it must not
    change when the barcode does."""
    P = rps["cloud"]
    tau = T.persistence_threshold(P)
    assert tau == pytest.approx(T.STABILITY_FACTOR * T.sampling_scale(P), rel=0, abs=0)
    th = 0.63
    R = np.array([[math.cos(th), -math.sin(th), 0.0],
                  [math.sin(th), math.cos(th), 0.0], [0.0, 0.0, 1.0]])
    perm = np.random.default_rng(3).permutation(len(P))
    assert T.persistence_threshold((P @ R.T)[perm]) == pytest.approx(tau, rel=1e-12)


def test_the_authors_toolkit_alone_reproduces_the_barcode_verdict(rps):
    """`ceq/rips.py` builds the 1-skeleton and labels components; it computes no
    2-simplices, so it reports no `b1` on its own. This adds the flag-complex
    layer on top of its UNCHANGED edge rule -- `b1 = (E - V + b0) - rank d2` --
    and reads it at five radii spanning the bar the engine reported."""
    bar = T.persistent_beta1(rps["cloud"])["bars"][0]
    born, died = float(bar[0]), float(bar[1])
    reads = {r: T.toolkit_flag_betti(rps["cloud"], r)["b1"]
             for r in (born * 0.5, born * 1.5, 0.5 * (born + died),
                       died * 0.95, died * 1.1)}
    vals = list(reads.values())
    assert vals[0] == 0 and vals[-1] == 0, reads
    assert all(v == 1 for v in vals[1:-1]), reads


@pytest.mark.parametrize("eps", [1e-3, 1e-2, 5e-2])
def test_the_barcode_moves_by_at_most_two_epsilon(rps, eps):
    """Cohen-Steiner/Chazal stability. `ceq/rips.py` exposes no bottleneck
    distance, so the number itself comes from `persim`."""
    out = T.bottleneck_stability(rps["cloud"], eps, seed=7)
    assert out["bottleneck"] <= 2.0 * eps + 1e-9, out


@pytest.mark.parametrize("eps", [1e-3, 1e-2])
def test_the_toolkit_alone_shows_the_two_epsilon_interleaving_on_beta0(rps, eps):
    """The sharpest stability statement `ceq/rips.py` can make by itself. It has
    `b0` and nothing else, and `b0(.,r)` is non-increasing in `r`, so a
    `2 eps`-interleaving of the filtrations reads as a sandwich."""
    out = T.beta0_interleaving(rps["cloud"], eps, seed=7)
    assert out["violations"] == 0, out
    # non-vacuity: the sandwich must be strict over at least its own window
    # width, less one grid cell, or it is not testing anything
    assert out["strict_span"] >= 4 * eps - out["grid_step"], out


# ---------------------------------------------------------------------------
# (c) Euler-Poincare against the BED-1 equilibrium census
# ---------------------------------------------------------------------------


def test_euler_poincare_equals_the_bed1_poincare_hopf_index_sum():
    """`sum (-1)^k b_k` computed through `ceq.rips.components` against
    `m_0 - m_1` from `ceq.beds.bed_1.morse_census`. V15_BED1.md reports
    `m_0 = 3`, `m_1 = 8`, `chi = -5`."""
    out = T.bed1_cross_check()
    assert (out["m0"], out["m1"]) == (3, 8)
    assert out["index_sum"] == -5
    assert out["b0"] == 1 and out["b1"] == 6
    assert out["alt_sum"] == out["index_sum"], out
    assert out["agrees"] is True, out


def test_b0_is_computed_and_not_assumed_to_be_one():
    """BED-1's own `morse_census` returns `betti_0 = 1` and
    `betti_1 = E - V + 1` as CONSTANTS. On BED-1's landscape that happens to be
    right -- the graph is connected -- so nothing on that bed can tell the
    constant from a computation, and the cross-check would be a restatement.
    This ranges over a disconnected graph, where the two answers differ: a
    triangle plus a disjoint edge plus an isolated node has `b0 = 3`, `b1 = 1`
    and `chi = 6 - 4 = 2`, and the assumed form would report `b1 = -1`."""
    out = T.euler_poincare_graph(6, [(0, 1), (1, 2), (2, 0), (3, 4)])
    assert (out["b0"], out["b1"], out["chi"], out["alt_sum"]) == (3, 1, 2, 2)


def test_the_bed1_morse_function_is_nondegenerate_and_the_complex_is_closed():
    """The two hypotheses Poincare-Hopf needs here, checked rather than assumed.
    A graph carries no boundary, so transversality is vacuous; what can fail is
    non-degeneracy -- an edge whose endpoints share a value has no lower link
    and the census is undefined on it."""
    out = T.bed1_cross_check()
    assert out["degenerate_edges"] == []
    assert out["boundary_cells"] == 0


def test_the_census_moves_and_the_invariant_cannot():
    """BED-1's must-fire landscape, seen from the Betti side: raising `C` moves
    `(m_0, m_1)` from `(3, 8)` to `(2, 7)` while `sum (-1)^k b_k` is pinned."""
    base = T.bed1_cross_check()
    fired = T.bed1_cross_check(V_override={"C": 2.5})
    assert (fired["m0"], fired["m1"]) == (2, 7)
    assert fired["alt_sum"] == base["alt_sum"] == -5
    assert fired["index_sum"] == fired["alt_sum"], fired


@pytest.mark.parametrize("mu,expected", [(0.0, 0.0), (0.01, 0.01 / 3),
                                         (0.10, 0.10 / 3)])
def test_poincare_hopf_transversality_fails_at_mu_zero(mu, expected):
    """MISTAKES.md:1406 -- Poincare-Hopf invoked where the field is tangent.
    Reproduced on the same object: on a face with `z_0 = 0` the pure replicator
    gives `dz_0 = 0` exactly (the face is invariant, the field is tangent, the
    hypothesis fails), and the mutation term is what makes it point inward."""
    dz = T.boundary_normal_component(T.RPS_PAYOFF, mu=mu)
    assert dz == pytest.approx(expected, abs=1e-15)
    assert (dz > 0.0) == (mu > 0.0)


def test_the_index_sum_is_pinned_to_chi_only_once_the_field_points_inward():
    """With `mu > 0` the flow is inward on every face, the barycentre is the
    only rest point, and its index is `+1 = chi(simplex)`. The certificate
    refuses to state an index sum at `mu = 0`."""
    out = T.replicator_index_sum(T.RPS_PAYOFF, mu=0.02)
    assert out["n_interior_equilibria"] == 1
    assert out["index_sum"] == 1 == out["chi_domain"]
    with pytest.raises(CertificateRefused, match="tangent"):
        T.replicator_index_sum(T.RPS_PAYOFF, mu=0.0)


# ---------------------------------------------------------------------------
# arithmetic (MISTAKES.md M-19)
# ---------------------------------------------------------------------------


def test_the_rps_orbit_is_not_chaotic_and_its_horizon_is_stated(rps):
    """M-19's horizon is `mantissa_bits / log2(stretching rate)`; the RPS
    replicator has a conserved quantity `x1 x2 x3` and a centre, so the
    stretching rate is 1 and that horizon is infinite. What is finite is
    integrator drift, and it is measured rather than assumed."""
    assert rps["invariant_drift"] < 1e-9, rps["invariant_drift"]
    assert rps["simplex_drift"] < 1e-12, rps["simplex_drift"]
    assert rps["closure_gap"] < 1e-3, rps["closure_gap"]
    assert rps["lyapunov_bound"] == 0.0


def test_winding_and_beta1_agree_per_instance():
    """Printed per instance, as the delta asks. Two seeds of the cyclic field
    and its reverse against the coordination control; `|winding|` and `b1` are
    computed by disjoint code paths and must agree on which carriers cycle."""
    rows = T.certificate_table()
    # every winding here is refinement-certified against the raw integrator
    # output, which is 20x-100x finer than the 48-point arc-length resample
    assert all(r["n_fine"] > 20 * 48 for r in rows), rows
    assert [r["winding"] for r in rows] == [1, 1, -1, 0]
    assert [r["beta1"] for r in rows] == [1, 1, 1, 0]
    for r in rows:
        assert (abs(r["winding"]) >= 1) == (r["beta1"] >= 1) == r["recurrent"]

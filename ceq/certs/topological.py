"""X37 -- the three topological certificates.

`CEQ_V15_3_DELTA.md` section X37 asks for three readings, and the reason each is
a certificate rather than a statistic is that each one can REFUSE.

  (a) `Z` WINDING of a phase sequence. Integer, per instance. The delta calls a
      non-integer reading an instrument defect rather than a result, so this
      module returns `int` and raises on the two conditions under which winding
      is not defined: a sequence that steps too fast between samples (the phase
      aliases and the reading is silently wrong) and a path that is not a cycle
      (the total turn is genuinely not an integer). The undersampled case is the
      dangerous one -- an unguarded routine returns a plausible integer -- so the
      unguarded behaviour is kept reachable via `max_step=pi` and exercised in
      the test suite rather than only described.

  (b) PERSISTENT `b1` of a carrier trajectory, `recurrent iff b1 >= 1`, with the
      delta's RPS / coordination must-fire `[RUN: 1 vs 0]`.

  (c) EULER-POINCARE against the Poincare-Hopf index sum of an equilibrium
      census. Cross-check between two computations that share no code.

WHAT `ceq/rips.py` PROVIDES AND WHAT IT DOES NOT. The author's toolkit ports a
geodesic Vietoris-Rips generator on `S^2`: a point sampler, the edge rule
`dot >= cos r`, a bridge edge and a component labeller. That is a Rips
1-SKELETON and `b0`. It builds no 2-simplices, so it computes no `b1`, no
persistence diagram and no bottleneck distance, and this module says so instead
of implying otherwise:

  * the FILTRATION is the toolkit's. `toolkit_edges` calls `rips.rips_edges`
    with its arithmetic unchanged, inverting only its degree parameter into the
    radius it already computes internally. The equality of the toolkit's edge
    set with the geodesic ball rule is asserted in the suite, so the barcode
    engine and the toolkit are shown to filter the same complex rather than
    assumed to.
  * `b0` at any radius comes from `rips.components`.
  * `b1` at any radius comes from the flag complex built on the toolkit's edge
    set, `b1 = (E - V + b0) - rank(d2)`, in `toolkit_flag_betti`. That layer is
    ~20 lines and is the smallest thing that turns a Rips graph into a `b1`.
  * the persistence BARCODE and the BOTTLENECK distance come from `ripser` /
    `persim`, which are already pinned dependencies of this repo and already
    calibrated against six invariants in `tests/foreman/test_topology_washout.py`.
    `toolkit_flag_betti` reproduces the barcode's verdict at five radii, so the
    two instruments are cross-read rather than one trusted.

THE PERSISTENCE THRESHOLD, AND WHY IT IS NOT FITTED (MISTAKES.md M-2). A `b1`
bar that dies immediately is noise, so a threshold is needed and a threshold
chosen after seeing the bars is not a threshold. This one is

    tau(X) = STABILITY_FACTOR * (max over points of nearest-neighbour distance)

with `STABILITY_FACTOR = 4` derived, not tuned: an unmatched bar `(b, d)` sits
`(d - b)/2` from the diagonal in the bottleneck metric, and Rips persistence
obeys `d_B <= 2 d_H`, so a Hausdorff perturbation of size `h` can create or
destroy any bar with `d - b <= 4h`. The sample's own resolution bounds `d_H`
between the sample and the curve it is drawn from. Both factors come from the
theorem and from the point POSITIONS; neither reads the barcode.

THE ARITHMETIC (MISTAKES.md M-19). M-19's horizon is
`mantissa_bits / log2(stretching rate)` -- about 52 float64 steps for a tent
map. The carrier here is the zero-sum RPS replicator, which is not chaotic: the
rows and columns of `RPS_PAYOFF` sum to zero, so `x1 x2 x3` and `sum x` are
exact invariants of the flow, the interior rest point is a centre, and every
Lyapunov exponent is 0. The stretching rate is 1 and M-19's horizon is
unbounded. What IS finite is RK4 truncation drift, and `carrier` measures it
rather than assuming it: `invariant_drift`, `simplex_drift` and `closure_gap`
are returned with every trajectory and asserted in the suite. The winding
certificate and the order parameter iterate nothing at all. Certificate (c)
runs on BED-1, whose own arithmetic statement is in `V15_BED1.md` section 7.
"""
from __future__ import annotations

import math

import numpy as np

from ceq import rips

__all__ = ["CertificateRefused", "NYQUIST_MARGIN", "INTEGRALITY_TOL",
           "STABILITY_FACTOR", "RPS_PAYOFF", "COORDINATION_PAYOFF",
           "winding_number", "order_parameter", "wrapped_differences",
           "replicator_field", "carrier", "simplex_phase", "to_sphere",
           "toolkit_edges", "geodesic_matrix", "sampling_scale",
           "persistence_threshold", "h1_barcode", "persistent_beta1",
           "toolkit_flag_betti", "bottleneck_stability", "beta0_interleaving",
           "euler_poincare_graph", "bed1_cross_check",
           "boundary_normal_component", "replicator_index_sum",
           "certificate_table"]

TWO_PI = 2.0 * math.pi


class CertificateRefused(Exception):
    """The instrument could not measure this input. Not a pass and not a fail."""


# ---------------------------------------------------------------------------
# (a) Z winding, and the Kuramoto order parameter
# ---------------------------------------------------------------------------

#: Largest wrapped step a winding reading is certified at. The reading is
#: well-posed only while the TRUE step stays under `pi`; a sample sequence shows
#: only the wrapped step, and a true step of `1.2 pi` presents as `-0.8 pi`,
#: which is under `pi` and therefore invisible to a guard set at the naive
#: Nyquist limit. `pi/2` is that limit with one octave of margin -- four samples
#: per turn instead of two. FROZEN, and deliberately CONSERVATIVE: it refuses a
#: class of well-posed inputs rather than accepting any ill-posed one, which is
#: the direction a certificate has to err in.
NYQUIST_MARGIN = math.pi / 2.0

#: How far `total / 2pi` may sit from an integer before the reading is called an
#: instrument defect. float64 accumulation over `1e4` samples is `~1e-13`.
INTEGRALITY_TOL = 1e-9

#: The constant in the bottleneck stability bound; see the module docstring.
STABILITY_FACTOR = 4.0


def wrapped_differences(theta: np.ndarray, closed: bool = True) -> np.ndarray:
    """Consecutive phase steps folded into `(-pi, pi]`, the cycle-closing step
    from last to first included when `closed`."""
    theta = np.asarray(theta, dtype=np.float64)
    d = np.diff(theta)
    if closed:
        d = np.append(d, theta[0] - theta[-1])
    return (d + math.pi) % TWO_PI - math.pi


def winding_number(theta, *, closed: bool = True,
                   max_step: float = NYQUIST_MARGIN,
                   integrality_tol: float = INTEGRALITY_TOL,
                   refinement=None) -> int:
    """The `Z` winding of a phase sequence, or a refusal.

    `closed` reads the sequence as a cycle, which is the only reading under
    which winding is an integer. `max_step=pi` restores the unguarded
    behaviour -- it is the naive Nyquist limit, it accepts aliased input, and it
    exists so the defect stays measurable rather than merely described.

    THE STEP GUARD IS NECESSARY AND NOT SUFFICIENT, and saying so is the point.
    A wrapped step of magnitude `s` is consistent with true steps
    `s, s +- 2pi, s +- 4pi, ...`; no test on the samples alone can separate
    them. Measured on this module: 63 turns sampled 64 times has a largest
    wrapped step of `0.0313 pi` -- as healthy as a reading gets -- and returns
    `-1`. `max_step` catches the marginal band and cannot catch that one.

    `refinement` closes it empirically. Pass a finer sampling of the SAME path
    and the two readings must agree, which is the convergence test the guard
    cannot do from one sampling. At 512 samples the 63-turn path reads `63`,
    disagrees with `-1`, and the certificate refuses. It is an empirical
    sufficiency check, not a proof, and no reading from finitely many samples
    can be more than that.
    """
    theta = np.asarray(theta, dtype=np.float64).ravel()
    if theta.size < 2:
        raise CertificateRefused(
            f"a winding needs at least two samples, got {theta.size}")
    if not np.all(np.isfinite(theta)):
        raise CertificateRefused(
            f"{int((~np.isfinite(theta)).sum())} non-finite phase samples")
    d = wrapped_differences(theta, closed=closed)
    worst = float(np.max(np.abs(d)))
    if worst > max_step:
        raise CertificateRefused(
            f"undersampled: largest wrapped step {worst / math.pi:.4f} pi "
            f"exceeds {max_step / math.pi:.4f} pi, so the phase aliases and the "
            "winding is not determined by these samples")
    total = float(np.sum(d)) / TWO_PI
    k = int(round(total))
    if abs(total - k) > integrality_tol:
        raise CertificateRefused(
            f"total turn {total!r} is not an integer within {integrality_tol:g}"
            + ("" if closed else "; an open path has no integer winding"))
    if refinement is not None:
        fine = np.asarray(refinement, dtype=np.float64).ravel()
        if fine.size <= theta.size:
            raise CertificateRefused(
                f"refinement has {fine.size} samples, not more than the "
                f"{theta.size} being checked")
        k_fine = winding_number(fine, closed=closed, max_step=max_step,
                                integrality_tol=integrality_tol)
        if k_fine != k:
            raise CertificateRefused(
                f"aliased: {theta.size} samples read {k} and {fine.size} "
                f"samples of the same path read {k_fine}; the reading is not "
                "stable under refinement")
    return k


def order_parameter(theta) -> float:
    """Kuramoto `r = |(1/N) sum_j exp(i theta_j)|`.

    `r -> 1` under phase locking and `r -> 0` for phases carrying no order; for
    `N` phases drawn uniformly the resultant is Rayleigh with
    `E[r] = sqrt(pi) / (2 sqrt N)`, which is the control the suite checks
    against, and for a wrapped normal of width `sigma` the closed form is
    `exp(-sigma^2 / 2)`.
    """
    theta = np.asarray(theta, dtype=np.float64).ravel()
    if theta.size == 0:
        raise CertificateRefused("order parameter of an empty phase set")
    if not np.all(np.isfinite(theta)):
        raise CertificateRefused("non-finite phase samples")
    return float(abs(np.mean(np.exp(1j * theta))))


# ---------------------------------------------------------------------------
# carriers -- the trajectories the certificates are read on
# ---------------------------------------------------------------------------

#: Zero-sum rock-paper-scissors. Rows AND columns sum to zero, which is what
#: makes `x1 x2 x3` an exact invariant and the orbits closed.
RPS_PAYOFF = np.array([[0.0, -1.0, 1.0],
                       [1.0, 0.0, -1.0],
                       [-1.0, 1.0, 0.0]])

#: Coordination. `A = I` pays for agreeing with yourself, the barycentre is a
#: repeller and every interior orbit runs monotonically to a vertex -- a path,
#: not a cycle. This is the delta's named control.
COORDINATION_PAYOFF = np.eye(3)

_CENTRE = np.full(3, 1.0 / 3.0)
_U = np.array([1.0, -1.0, 0.0]) / math.sqrt(2.0)
_V = np.array([1.0, 1.0, -2.0]) / math.sqrt(6.0)


def replicator_field(x: np.ndarray, A: np.ndarray, mu: float = 0.0) -> np.ndarray:
    """`dx_i = x_i ((Ax)_i - x.Ax) + mu (1/n - x_i)`.

    `mu = 0` is the pure replicator: every face is invariant and the field is
    TANGENT to the boundary. `mu > 0` is the replicator-mutator, whose uniform
    injection makes the field point strictly inward. Which of the two is in play
    decides whether Poincare-Hopf applies at all (MISTAKES.md:1406).
    """
    x = np.asarray(x, dtype=np.float64)
    p = A @ x
    return x * (p - float(x @ p)) + mu * (1.0 / len(x) - x)


def _rk4(x0, A, mu, dt, n_steps):
    x = np.asarray(x0, dtype=np.float64).copy()
    out = np.empty((n_steps + 1, len(x)), dtype=np.float64)
    out[0] = x
    for s in range(n_steps):
        k1 = replicator_field(x, A, mu)
        k2 = replicator_field(x + 0.5 * dt * k1, A, mu)
        k3 = replicator_field(x + 0.5 * dt * k2, A, mu)
        k4 = replicator_field(x + dt * k3, A, mu)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out[s + 1] = x
    return out


def simplex_phase(traj: np.ndarray) -> np.ndarray:
    """Angle about the barycentre in the plane of the simplex. The barycentre is
    a rest point of both fields, so no orbit passes through it and the angle is
    defined at every sample."""
    d = np.asarray(traj, dtype=np.float64) - _CENTRE
    return np.arctan2(d @ _V, d @ _U)


def _one_turn(traj):
    th = np.unwrap(simplex_phase(traj))
    hit = np.nonzero(np.abs(th - th[0]) >= TWO_PI)[0]
    if len(hit) == 0:
        raise CertificateRefused(
            f"the trajectory never closes a turn (total {(th[-1] - th[0]) / TWO_PI:.4f})")
    return traj[:hit[0] + 1]


def _arclength_resample(traj, n):
    """Uniform in arc length rather than in time. A reparameterisation changes
    no topology, and it makes `sampling_scale` a resolution rather than an
    artefact of where the field happens to run slowly."""
    step = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(step)])
    closed = np.linalg.norm(traj[-1] - traj[0]) < 1e-2 * s[-1]
    grid = (np.linspace(0.0, s[-1], n, endpoint=False) if closed
            else np.linspace(0.0, s[-1], n))
    return np.column_stack([np.interp(grid, s, traj[:, k]) for k in range(traj.shape[1])])


def to_sphere(X: np.ndarray) -> np.ndarray:
    """Radial projection onto `S^2`, so `ceq/rips.py`'s geodesic rule applies.

    The simplex is a planar triangle not containing the origin, so `x -> x/|x|`
    is a homeomorphism onto a spherical triangle: it distorts distance by a
    bounded factor and changes no homology. It is the inverse of the gnomonic
    projection.
    """
    X = np.asarray(X, dtype=np.float64)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


_CARRIERS = {
    "rps": (RPS_PAYOFF, True),
    "rps_reversed": (-RPS_PAYOFF, True),
    "coordination": (COORDINATION_PAYOFF, False),
}


def carrier(kind: str, *, x0=(0.5, 0.3, 0.2), n_points: int = 48,
            mu: float = 0.0, dt: float = 0.01, max_steps: int = 20000,
            arc_fraction: float = 1.0) -> dict:
    """A carrier trajectory with its arithmetic attached.

    `arc_fraction < 1` truncates a closed orbit to an open arc; it exists as the
    hard negative control in `V15_X37_CERTS.md` and is not part of the delta's
    named must-fire.
    """
    if kind not in _CARRIERS:
        raise CertificateRefused(f"unknown carrier {kind!r}")
    A, cyclic = _CARRIERS[kind]
    raw = _rk4(x0, A, mu, dt, max_steps)
    if cyclic:
        raw = _one_turn(raw)
    else:                                # a path: stop once it has converged
        moved = np.linalg.norm(np.diff(raw, axis=0), axis=1)
        stop = np.nonzero(np.cumsum(moved) >= 0.999 * moved.sum())[0]
        raw = raw[:int(stop[0]) + 2]
    if arc_fraction < 1.0:
        raw = raw[:max(2, int(len(raw) * arc_fraction))]
    # `x1 x2 x3` is an invariant only when the payoff's columns sum to zero;
    # under coordination it runs to 0 at the vertex, so reporting a "drift" for
    # it would be reporting the dynamics as an error.
    zero_sum = bool(np.abs(A.sum(axis=0)).max() < 1e-12)
    inv = raw.prod(axis=1)
    traj = _arclength_resample(raw, n_points)
    return dict(
        kind=kind, mu=float(mu), n_raw=len(raw), dt=float(dt),
        traj=traj, cloud=to_sphere(traj), phase=simplex_phase(traj),
        phase_fine=simplex_phase(raw),
        zero_sum=zero_sum,
        invariant_drift=(float((inv.max() - inv.min()) / abs(inv[0]))
                         if zero_sum else float("nan")),
        simplex_drift=float(np.abs(raw.sum(axis=1) - 1.0).max()),
        closure_gap=float(np.linalg.norm(raw[-1] - raw[0])) if cyclic else float("nan"),
        # zero-sum RPS has a conserved product and a centre, so no direction
        # stretches; the coordination flow contracts onto a vertex.
        lyapunov_bound=0.0,
    )


# ---------------------------------------------------------------------------
# (b) the toolkit's filtration, and b1 on top of it
# ---------------------------------------------------------------------------


def toolkit_edges(points, radius: float) -> list[tuple[int, int]]:
    """`ceq.rips.rips_edges` at a geodesic radius, arithmetic unchanged.

    `rips_edges` is parameterised by an expected degree and converts it to
    `r = 2 asin(sqrt(deg/(n-1)))` internally, so the inverse
    `deg = (n-1) sin^2(r/2)` reaches the radius without touching its rule.
    """
    n = len(points)
    if not 0.0 <= radius <= math.pi:
        raise CertificateRefused(f"geodesic radius {radius} outside [0, pi]")
    return rips.rips_edges(points, (n - 1) * math.sin(radius / 2.0) ** 2)


def geodesic_matrix(points) -> np.ndarray:
    P = np.asarray(points, dtype=np.float64)
    return np.arccos(np.clip(P @ P.T, -1.0, 1.0))


def _b0(n, edges) -> int:
    """`ceq.rips.components` labels a node touched by no edge `-1`; each of those
    is its own component."""
    lab = rips.components(n, edges)
    return len({x for x in lab if x >= 0}) + sum(1 for x in lab if x < 0)


def sampling_scale(points) -> float:
    """`max_i min_{j != i} d_geo(i, j)` -- the coarsest gap in the sample, an
    upper bound on its Hausdorff distance to the curve it was drawn from."""
    D = geodesic_matrix(points).copy()
    np.fill_diagonal(D, np.inf)
    return float(D.min(axis=1).max())


def persistence_threshold(points) -> float:
    """`STABILITY_FACTOR * sampling_scale`. A function of the point positions
    only; it never sees a barcode (MISTAKES.md M-2)."""
    return STABILITY_FACTOR * sampling_scale(points)


def h1_barcode(points) -> np.ndarray:
    """The `H_1` diagram over the toolkit's geodesic metric.

    Computed by `ripser`, which is a pinned dependency of this repo and is
    calibrated against permutation invariance, isometry invariance, scale
    equivariance, the stability theorem, the circle's `sqrt(3) r` death and a
    blob negative control in `tests/foreman/test_topology_washout.py`. The
    filtration is the toolkit's -- the suite asserts `toolkit_edges` and the
    geodesic ball agree at every radius.
    """
    import ripser as _ripser
    dgms = _ripser.ripser(geodesic_matrix(points), distance_matrix=True,
                          maxdim=1)["dgms"]
    d = np.asarray(dgms[1], dtype=np.float64).reshape(-1, 2)
    return d[np.isfinite(d[:, 1])]


def persistent_beta1(points) -> dict:
    """`b1` counted over the persistence threshold, and `recurrent iff b1 >= 1`."""
    bars = h1_barcode(points)
    tau = persistence_threshold(points)
    pers = bars[:, 1] - bars[:, 0] if len(bars) else np.zeros(0)
    keep = np.argsort(-pers)[:int((pers > tau).sum())] if len(bars) else np.zeros(0, int)
    return dict(bars=bars[keep], all_bars=bars, persistences=np.sort(pers)[::-1],
                threshold=float(tau), sampling_scale=float(sampling_scale(points)),
                beta1=int(len(keep)), recurrent=bool(len(keep) >= 1),
                max_persistence=float(pers.max()) if len(pers) else 0.0)


def toolkit_flag_betti(points, radius: float) -> dict:
    """`b0` and `b1` of the FLAG complex on the toolkit's edge set at `radius`.

    `b0` is `ceq.rips.components`. `b1 = dim ker d1 - rank d2`, where
    `dim ker d1 = E - V + b0` is the cycle rank of the toolkit's graph and `d2`
    is the boundary of its 3-cliques, over `Q`. These clouds are sampled curves,
    which carry no torsion, so the field does not change the answer.
    """
    P = np.asarray(points, dtype=np.float64)
    n = len(P)
    edges = toolkit_edges(P, radius)
    b0 = _b0(n, edges)
    cycles = len(edges) - n + b0
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    index = {e: k for k, e in enumerate(edges)}
    tris = [(a, b, c) for (a, b) in edges for c in sorted(adj[a] & adj[b]) if c > b]
    if not tris:
        return dict(b0=b0, b1=cycles, n_edges=len(edges), n_triangles=0, rank_d2=0)
    d2 = np.zeros((len(edges), len(tris)), dtype=np.float64)
    for t, (a, b, c) in enumerate(tris):
        d2[index[(b, c)], t] = 1.0
        d2[index[(a, c)], t] = -1.0
        d2[index[(a, b)], t] = 1.0
    rank = int(np.linalg.matrix_rank(d2))
    return dict(b0=b0, b1=cycles - rank, n_edges=len(edges),
                n_triangles=len(tris), rank_d2=rank)


def _rotate_on_sphere(P, eps, seed):
    """Move every point by a geodesic distance in `[0, eps]`, so the Hausdorff
    distance to the original cloud is at most `eps` EXACTLY, in the same metric
    the filtration uses. Perturbing in `R^3` and renormalising would not give
    that bound in closed form."""
    rng = np.random.default_rng(seed)
    axis = np.cross(P, rng.standard_normal(P.shape))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    ang = rng.uniform(0.0, eps, (len(P), 1))
    return P * np.cos(ang) + np.cross(axis, P) * np.sin(ang)


def bottleneck_stability(points, eps: float, seed: int = 0) -> dict:
    """`d_B(Dgm(X), Dgm(X_eps)) <= 2 eps`.

    `ceq/rips.py` EXPOSES NO BOTTLENECK DISTANCE -- it has an edge rule and a
    component labeller and nothing else -- so the number here comes from
    `persim`. `beta0_interleaving` is the sharpest form of the same statement
    that the toolkit can make unaided.
    """
    from persim import bottleneck
    P = np.asarray(points, dtype=np.float64)
    Q = _rotate_on_sphere(P, eps, seed)
    moved = float(np.arccos(np.clip((P * Q).sum(axis=1), -1.0, 1.0)).max())
    return dict(eps=float(eps), hausdorff=moved,
                bottleneck=float(bottleneck(h1_barcode(P), h1_barcode(Q))),
                bound=2.0 * float(eps))


def beta0_interleaving(points, eps: float, seed: int = 0, n_radii: int = 120) -> dict:
    """The sharpest stability check `ceq/rips.py` supports on its own.

    A cloud moved by at most `eps` in Hausdorff distance has a Rips filtration
    `2 eps`-interleaved with the original's, and `b0(., r)` is non-increasing in
    `r`, so the interleaving reads as the sandwich

        b0(X, r + 2 eps)  <=  b0(X_eps, r)  <=  b0(X, r - 2 eps).

    `strict_span` is the measure of radii where the two ends of the sandwich
    actually differ, i.e. where the check can fail at all; without it a sandwich
    that never bites would report a pass while measuring nothing (MISTAKES.md
    V-16). Its floor is the window width `4 eps`, because the cloud's `b0`
    transition has positive width and every window straddling it is strict.
    """
    P = np.asarray(points, dtype=np.float64)
    Q = _rotate_on_sphere(P, eps, seed)
    n = len(P)
    radii = np.linspace(0.0, 1.4 * sampling_scale(P), n_radii)
    step = float(radii[1] - radii[0])
    violations = strict = 0
    for r in radii:
        lo = _b0(n, toolkit_edges(P, min(r + 2 * eps, math.pi)))
        hi = _b0(n, toolkit_edges(P, max(r - 2 * eps, 0.0)))
        mid = _b0(n, toolkit_edges(Q, r))
        strict += int(hi > lo)
        violations += int(not (lo <= mid <= hi))
    return dict(eps=float(eps), n_radii=int(n_radii), grid_step=step,
                strict_radii=int(strict), strict_span=strict * step,
                violations=int(violations))


# ---------------------------------------------------------------------------
# (c) Euler-Poincare vs the Poincare-Hopf index sum
# ---------------------------------------------------------------------------


def euler_poincare_graph(n_nodes: int, edges) -> dict:
    """`sum (-1)^k b_k` of a 1-complex, with `b0` from `ceq.rips.components`.

    HONEST NOTE, and it is the reason this half of the cross-check cannot fail:
    on a 1-complex `b1` is the cycle rank `E - V + b0`, so `b0 - b1 = V - E`
    identically. Euler-Poincare on a chain complex is a theorem about rank-
    nullity, not a measurement. What CAN fail, and what certificate (c)
    therefore actually tests, is Poincare-Hopf -- that the index sum of the
    equilibrium census equals this number.
    """
    e = sorted({(min(a, b), max(a, b)) for a, b in edges if a != b})
    b0 = _b0(n_nodes, e)
    b1 = len(e) - n_nodes + b0
    return dict(n_nodes=int(n_nodes), n_edges=len(e), b0=int(b0), b1=int(b1),
                chi=int(n_nodes - len(e)), alt_sum=int(b0 - b1))


def bed1_cross_check(V_override: dict | None = None, T: float = 0.25) -> dict:
    """BED-1's Morse census against `sum (-1)^k b_k` over the same complex.

    THE TWO SIDES ARE INDEPENDENT WHERE IT MATTERS. `morse_census` classifies
    every node by the number of components of its lower link and returns
    `m0 - m1`; this function counts components of the WHOLE graph through
    `ceq.rips.components` and forms `b0 - b1`. Neither reads the other. The
    equality `m0 - m1 = V - E` is Poincare-Hopf for a discrete Morse function
    and is the falsifiable content here.

    HYPOTHESES, checked rather than assumed. Poincare-Hopf needs the field
    transverse to the boundary; a graph is a closed 1-complex, `boundary_cells`
    is 0, and the hypothesis is vacuous. What is not vacuous is non-degeneracy:
    an edge whose endpoints carry the same value has an undefined lower link,
    and `degenerate_edges` lists any.

    BED-1's own `morse_census` returns `betti_0 = 1` and `betti_1 = E - V + 1`
    as CONSTANTS rather than computing them. Recomputing `b0` here from the edge
    list is what makes the comparison a cross-check instead of a restatement.
    """
    from ceq.beds import bed_1
    bed = bed_1.build(T, seed=0, V_override=V_override)
    census = bed_1.morse_census(bed)
    ix, V = bed["node_index"], bed["V"]
    edges = [(ix[u], ix[v]) for u, v in bed["edges"]]
    ep = euler_poincare_graph(bed["n"], edges)
    index_sum = int(census["minima"] - census["index_1"])
    return dict(m0=int(census["minima"]), m1=int(census["index_1"]),
                regular=int(census["regular"]), index_sum=index_sum,
                census_euler=int(census["euler"]),
                census_betti=(int(census["betti_0"]), int(census["betti_1"])),
                b0=ep["b0"], b1=ep["b1"], chi=ep["chi"], alt_sum=ep["alt_sum"],
                n_nodes=ep["n_nodes"], n_edges=ep["n_edges"],
                agrees=bool(index_sum == ep["alt_sum"]),
                boundary_cells=0,
                degenerate_edges=[(u, v) for u, v in bed["edges"]
                                  if V[ix[u]] == V[ix[v]]])


def boundary_normal_component(A: np.ndarray, mu: float = 0.0,
                              x_face=(0.0, 0.6, 0.4), face: int = 0) -> float:
    """`dx_face` at a point of the face `x_face = 0` -- the outward-normal
    component of the field, since that coordinate's own axis is the normal.

    MISTAKES.md:1406 recorded Poincare-Hopf invoked where this is zero. It
    reproduces here: at `mu = 0` the first term is `0 * ((Ax)_0 - phi)` and the
    reading is `-0.000000e+00`, the face is invariant, and the index sum is not
    pinned to `chi`. At `mu > 0` it is `mu/n > 0` and the field points inward.
    """
    return float(replicator_field(np.asarray(x_face, dtype=np.float64), A, mu)[face])


def _plane_reduction(A, mu):
    basis = np.vstack([_U, _V])

    def f(ab):
        x = _CENTRE + ab[0] * _U + ab[1] * _V
        return basis @ replicator_field(x, A, mu)

    def jac(ab, h=1e-7):
        return np.column_stack([(f(ab + h * e) - f(ab - h * e)) / (2 * h)
                                for e in np.eye(2)])
    return f, jac


def replicator_index_sum(A: np.ndarray, mu: float, n_grid: int = 24) -> dict:
    """Poincare-Hopf on the simplex: `sum index = chi(Delta^2) = 1`.

    REFUSES at `mu = 0`. The theorem needs the field transverse to the boundary
    and the pure replicator is tangent to every face, so the index sum is not
    pinned there -- each face carries its own equilibria independently. The
    transversality test is run on the field, not asserted.
    """
    A = np.asarray(A, dtype=np.float64)
    worst = min(boundary_normal_component(A, mu, x, k) for k, x in enumerate(
        [(0.0, 0.6, 0.4), (0.55, 0.0, 0.45), (0.35, 0.65, 0.0)]))
    if not worst > 0.0:
        raise CertificateRefused(
            f"the field is tangent to the boundary (inward component {worst:.6e} "
            f"at mu={mu}); Poincare-Hopf does not apply and no index sum is "
            "stated -- MISTAKES.md:1406")
    f, jac = _plane_reduction(A, mu)
    roots = []
    span = np.linalg.norm(np.array([1.0, 0.0, 0.0]) - _CENTRE)
    for a in np.linspace(-span, span, n_grid):
        for b in np.linspace(-span, span, n_grid):
            ab = np.array([a, b])
            for _ in range(60):
                fv = f(ab)
                if np.linalg.norm(fv) < 1e-14:
                    break
                try:
                    step = np.linalg.solve(jac(ab), fv)
                except np.linalg.LinAlgError:
                    ab = None
                    break
                ab = ab - step
                if not np.all(np.isfinite(ab)) or np.linalg.norm(ab) > 4 * span:
                    ab = None
                    break
            if ab is None or np.linalg.norm(f(ab)) > 1e-9:
                continue
            x = _CENTRE + ab[0] * _U + ab[1] * _V
            if np.min(x) <= 1e-9:                 # interior only
                continue
            if not any(np.linalg.norm(ab - r) < 1e-6 for r in roots):
                roots.append(ab)
    indices = []
    for ab in roots:
        det = float(np.linalg.det(jac(ab)))
        if abs(det) < 1e-12:
            raise CertificateRefused(
                f"degenerate equilibrium at {ab} (det J = {det:.3e}); the index "
                "is undefined there")
        indices.append(1 if det > 0 else -1)
    return dict(mu=float(mu), inward_margin=float(worst),
                n_interior_equilibria=len(roots), indices=indices,
                index_sum=int(sum(indices)), chi_domain=1)


# ---------------------------------------------------------------------------
# the per-instance table the delta asks to be printed
# ---------------------------------------------------------------------------

#: `(label, carrier kind, initial condition)`. Two initial conditions on the
#: cyclic field, its reverse, and the coordination control.
#: `x0=(.2,.5,.3)` would NOT be a second instance -- it has the same
#: `x1 x2 x3` and so lies on the same orbit, one phase offset away.
INSTANCES = [("rps x0=(.5,.3,.2)", "rps", (0.5, 0.3, 0.2)),
             ("rps x0=(.6,.25,.15)", "rps", (0.6, 0.25, 0.15)),
             ("rps reversed", "rps_reversed", (0.5, 0.3, 0.2)),
             ("coordination", "coordination", (0.5, 0.3, 0.2))]


def certificate_table(instances=None) -> list[dict]:
    """(a), (b) and the order parameter for each instance, in one pass."""
    rows = []
    for label, kind, x0 in (instances or INSTANCES):
        car = carrier(kind, x0=x0)
        b = persistent_beta1(car["cloud"])
        rows.append(dict(
            instance=label,
            winding=winding_number(car["phase"], refinement=car["phase_fine"]),
            n_fine=len(car["phase_fine"]),
            beta1=b["beta1"], recurrent=b["recurrent"],
            max_persistence=b["max_persistence"], threshold=b["threshold"],
            sampling_scale=b["sampling_scale"],
            r=order_parameter(car["phase"]),
            invariant_drift=car["invariant_drift"],
            closure_gap=car["closure_gap"]))
    return rows

"""Discrete Ricci flow on the context graph: Ni et al. Eq. 10 AND the spec's rule.

WHAT IS UNDER TEST AND WHAT IS NOT. Two weight updates are implemented side by
side and neither is allowed to be the other. From the PDF of arXiv:1907.03993
(Ni, Lin, Luo, Gao, "Community Detection on Networks with Ricci Flow",
Sci. Rep. 9), quoted:

    w^(k+1)_ij = (1 - kappa^(k)_ij) * d^(k)(i, j)                    [their Eq. 10]

with d^(k) the metric of the WEIGHTED graph (V, E, w^(k)), recomputed each step,
discretising Ollivier's d/dt d_ij = -kappa_ij d_ij [their Eq. 9]. The PHASE C
spec calls its own update "the Ni et al. 2019 form"; it is

    w_e <- w_e * (1 - eta (kappa_e - kappa_target(e)))                [the spec]

a rescale of the existing weight carrying an eta and a target. Eq. 10 REPLACES a
weight by a curvature-scaled DISTANCE; the spec SCALES a weight. Those are
different maps, and the spec's identity bind -- "Ricci flow with source = 0 must
reproduce Ni et al.'s community separation" -- is measured here, not assumed.

THE ANSWER, and it is sharper than either "same" or "different".

  DERIVED. Call a graph SHORTCUT-FREE when every edge is its own geodesic,
  d(i,j) = w_ij. There Eq. 10 reads (1 - kappa) w_ij and the spec at eta = 1,
  target 0 reads w_ij (1 - kappa): the same floating-point product. MEASURED on
  the planted bed (133 edges), the shortcut census is 0/133 at every one of 20
  steps and max|W_ni - W_spec| = 0.000e+00 -- the two runs are BITWISE the same
  array. The spec's identity bind holds on this bed BY IDENTITY, not by
  agreement, and eta and the target are exactly the parameters that leave it.

  MEASURED, the boundary. Off that regime they part at once. On a 5-node graph
  with w_01 = 10 and a 2-hop path of cost 2, so d(0,1) = 2 and kappa_01 =
  +0.750000, Eq. 10 returns 0.500000 and the spec at eta = 1 returns 2.500000 --
  a factor of 5. Under flow, on the planted bed with 6 intra edges seeded at
  weight 6.0, the trajectories differ by 2.9333e+00 at step 1, and Eq. 10 is
  shortcut-free after 1 step while the spec's rescale needs 6 (census 6, 6, 6,
  4, 3, 2, 0). Multiplying by d heals a shortcut; multiplying by w does not.

THE GAUGE, and it inverts the convergence verdict. kappa is invariant under a
global rescale: MEASURED max|kappa(cW) - kappa(W)| = 2.220e-16, 3.331e-16,
5.551e-16 at c = 2, 13.7, 1000 over 133 edges. All three updates are
1-HOMOGENEOUS in w, and the figure is PER RULE and not one number for all three:
max|G(F(W)) - G(F(G(3.7W)))| = 4.441e-16 (ni), 4.441e-16 (spec), 1.332e-15
(recip). ni and spec sit at machine epsilon because their update multiplies by a
gauge-invariant factor; recip divides by one, so it is the loosest of the three
and is bed-dependent -- a second bed read it orders looser again, still inside
the 1e-10 bound asserted for all three. So the dynamics live on the ray quotient
and a residual read off raw weights is partly the gauge drifting.

On the 66-edge bed Eq. 10's RAW residual ends at 5.5853e-02, and the ONSET of
that plateau is measured and printed rather than asserted: within 1% of the
final value from step 18 (drift 0.665%), within 0.1% from step 35 (drift
0.081%), and first PRINTING as 5.5853e-02 at step 53. An earlier draft of this
paragraph named step 21 -- where the value is 5.6191e-02, 0.61% above the floor
-- and was wrong by 32 steps. The plateau is real either way; the step named was
not. Meanwhile the SAME run's gauge-normalised residual goes 5.6568e-03 at step
21 to 9.698e-10 at step 81. Raw alone records a flow that does not converge.
Both columns print, always, and plateau_onset() replaces the prose.

THE FIXED POINT IS CONSTANT CURVATURE, NOT ZERO CURVATURE. DERIVED: in the ray
quotient a fixed point satisfies F(W) = cW for some c > 0. For Eq. 10 on a
shortcut-free graph that is (1 - kappa_e) w_e = c w_e, i.e. kappa_e = 1 - c on
every edge; for the spec it is kappa_e = (1 - c)/eta on every edge. Both name
the SAME set -- the constant-curvature graphs -- and neither names kappa = 0
unless c = 1. MEASURED on a 24-node / 66-edge bed, geometric-mean gauge, run to
a 1e-9 gauge residual or 130 steps:

    arm             steps  resid_gauge   kappa constant    spread    max|w/w*-1|  short
    Ni Eq. 10          81    9.698e-10   +0.006938942   3.979e-10   0.0000e+00   0/66
    spec eta = 0.5    100    2.241e-06   +0.006939091   2.190e-06   5.8086e-06   0/66
    spec eta = 1       81    9.698e-10   +0.006938942   3.979e-10   2.6645e-15   0/66
    spec eta = 2       54    9.104e-10   +0.006938942   4.548e-10   8.7829e-02   2/66
    recip eta = 2      55    6.656e-10   +0.006938942   3.184e-10   1.1484e-09   0/66

Every arm reaches ARI 1.0000 and the same constant to 1.494e-07. eta moves the
RATE (54 steps at 2, 81 at 1, 100 at 0.5). So on CURVATURE the answer to "do the
two rules share fixed points" is yes.

BUT THE WEIGHTS SAY OTHERWISE, AND THAT IS THE FINDING OF THIS ROUND. The last
two columns are the ones a residual never shows. w* is Eq. 10's own fixed point
-- the only non-strawman target, since a flow has no reason to return to a point
that is not fixed -- and both sides are gauge-normalised before comparison,
because kappa is scale-invariant and an unnormalised comparison measures the
gauge. spec at eta = 2 converges to a gauge residual of 9.104e-10, reports the
same kappa constant to nine digits, and sits 8.7829e-02 away from Eq. 10's
weights. Ten digits of "converged", 8.8% wrong in geometry.

THE DISCRIMINATOR IS THE SHORTCUT CENSUS -- the same quantity the identity in
(b) and (c) turns on. Every arm whose fixed point is shortcut-free lands on Eq.
10's weights (worst 5.8086e-06); the one arm carrying shortcuts (2 of 66 edges)
is 8.7829e-02 off, and is not an Eq. 10 fixed point at all: one Eq. 10 step
moves it by 4.132e-02, against 7.647e-10 at Eq. 10's own. So the earlier claim
that the two rules share their fixed-point set holds ON THE SHORTCUT-FREE
STRATUM AND NOT OFF IT. kappa_e constant is necessary for an Eq. 10 fixed point
and NOT sufficient; the missing condition is that every edge be a geodesic,
which the spec's rule can violate freely because it never consults d.

IS THERE A FLAT DIRECTION TO NORMALISE AWAY? No, and this is measured rather
than argued. Perturbing Eq. 10's fixed point by lognormal(0.30) -- a planted
max|w/w*-1| of 1.7789e+00 -- and re-running Eq. 10 returns it to 1.3163e-09 in
82 steps at a residual of 7.719e-10: the fixed point is ATTRACTING, not merely
stationary. The log-coordinate Jacobian of the gauge-normalised step, central
differences at eps = 0.01 on a 6 + 6 bed, passes its gate at J @ 1 = 5.305e-03
over 16 edges and has spectral radius 0.902968, second 0.667305, ZERO modes on
the unit circle and 1 at zero (the gauge, projected out by the normalisation).
An isolated, attracting fixed point has no flat direction, so a second
normalisation buys nothing. The report that closes the hole is the SHORTCUT
CENSUS, which is already in every run_flow trace and costs one shortest-path.

SO, PLAINLY: "the flow converged" is a statement about the CURVATURE. It becomes
a statement about the GEOMETRY exactly when the shortcut census is zero, and not
otherwise. Any convergence number quoted from this module without its census
beside it is half the story, and it is the half that always looks good.

THE NI BIND, with its control. Planted two-community graph (2 x 20 nodes,
p = 0.30, 5 cross edges, ground truth known); cut edges above a weight
threshold, the threshold swept IDENTICALLY in every arm, connected components
scored by ARI:

    NO FLOW (control)        ARI 0.0000   thresh 0.000e+00   40 components
    Ni Eq. 10, 20 steps      ARI 1.0000   thresh 9.853e-01    2 components
    spec eta = 1, 20 steps   ARI 1.0000   thresh 9.853e-01    2 components

and the weight the flow puts on the plant: the median planted CROSS edge over
median planted INTRA edge is 1.000 before the flow (5 and 128 edges) and 7.079
after it (3.8012e+00 over 5.3700e-01). Ni's separation reproduces; the spec's
zero-source rule reproduces it because on this bed it IS Ni's rule. The control
earns its place -- with unit weights the identical cut recovers nothing.

THE PLANTED NEGATIVES, each seen to fire. A single community of the same size
and density keeps 100.0% of its 40 nodes in one component after 8 steps under
all three rules. eta = 0 is BITWISE identity for spec and recip (0 weights
differ). A step whose kappa_target is the measured kappa moves nothing:
max|w' - w| = 0.000e+00 for both. And the trap that green diagnostics hide:
FREEZING d gives a gauge residual of 0.000e+00 after 15 steps with a recomputed
curvature spread of 6.560e-01, against the live-metric run's residual 4.094e-02
and spread 1.035e-02 -- 63.4x worse geometry behind a perfect residual. Eq. 10
is defined against the metric of (V, E, w^(k)); a frozen d is not the flow.

THE MEASURE, and it transfers at step 0 and nowhere else. Every kappa above is
at the UNIFORM lazy measure (curvature.lazy_measure, alpha = 0.5); Ni et al. use
a WEIGHT-PROPORTIONAL one. DERIVED: on an UNWEIGHTED graph w_xz / sum_z w_xz =
1/deg(x) term by term, so on such a graph the two are the SAME measure. The
input bed is unweighted, so the it.2 starting point transfers with nothing to
re-measure -- MEASURED, max|uniform - weight-proportional| = 0.000e+00 over 133
edges at 1 distinct weight.

That is where the transfer stops, and the reason is the flow itself. One Eq. 10
step turns the bed into 102 distinct weights and the gap opens to 1.202e-01; by
step 20 it is 9.939e-01 over 133 edges, against a uniform kappa range of
6.850e-03 on the same graph -- the disagreement is 145.1x the whole spectrum,
because by step 20 the curvature has collapsed toward its constant while the
weights have spread. So
EVERY number in this file taken at step >= 1 is a number at the uniform measure
and does not transfer. The preamble and (c) print both ends of that.
curvature.py is another agent's file and is not edited here; nothing above is a
claim about Ni's own reported accuracies.

ALSO NOT CLAIMED: w is a LENGTH in this module, because curvature.graph_metric()
hands it straight to shortest_path. A spec that calls w an attention weight and feeds
it to the same metric has inverted the variable -- attention is high where nodes
are CLOSE. recip_flow_step is the repair, and it is Foreman's sign flip mapped
back into the length variable, where it becomes a division: w / (1 + eta r)
instead of w (1 - eta r). Same first order, no eta ceiling on the positive side,
same fixed point (55 steps against 81), no measured advantage in ARI.

REFUSALS ARE VALUES, and there are THREE reachable causes, not four. A step that
would drive a weight non-positive (eta ceiling 1/max_e kappa_e = 3.2 on the
planted bed; at eta = 6.4, 42 edges refuse, min surviving weight 9.524e-03, zero
negatives), an unknown rule name, and a transport LP HiGHS cannot solve. The
last is not hypothetical, and divergence and refusal turn out to be different
events that the bed chooses between: on a 6 + 6 bed the sign-flipped arm at
eta = -2 does NOT refuse inside 250 steps, it merely diverges to a weight span of
8.206e+26, while at eta = -4 the same bed trips the LP at 50 steps and HiGHS
returns status 15
(model_status Unknown). It arrives as a Refusal carrying that span, not as an
exception. curvature.py's own four refusal codes are UNREACHABLE from a flow
step -- a step visits edges only, and both endpoints of an edge have degree >= 1
inside one component -- so two disjoint triangles refuse nothing and must still
come back finite. That is measured in demo (k), not assumed.

RUN: python -m ceqjepa.ricci_flow
"""

import time

import numpy as np
from scipy.sparse.csgraph import connected_components, csgraph_from_dense

from ceqjepa import curvature as cv
from ceqjepa.curvature import Refusal, is_refusal

__all__ = [
    "MEASURE", "N_STEPS", "LONG_STEPS", "LONG_TOL", "ETA", "CUT_QUANTILE",
    "BLOCK_N", "BLOCK_P", "N_CROSS", "SMALL_N", "SMALL_P", "SMALL_CROSS",
    "ETA_SWEEP", "LP_FAILED", "NON_POSITIVE_WEIGHT", "UNKNOWN_RULE", "RULES",
    "ni_flow_step", "spec_flow_step", "recip_flow_step", "run_flow",
    "gauge_normalise", "shortcut_census", "eta_ceiling",
    "planted_two_community", "planted_one_community", "shortcut_seeded_bed",
    "adjusted_rand_index", "best_cut_ari", "largest_component_fraction",
    "cross_edge_mass", "residual_verdict", "spectrum",
    "plateau_onset", "first_attained", "apply_rule", "weight_error",
    "recovery_table", "log_jacobian", "flat_modes", "JAC_EPS", "ONSET_STEP",
    "ONSET_TOL", "JAC_N", "JAC_P",
]

#: THE MEASURE EVERY NUMBER IN THIS FILE IS AT. Inherited, not chosen: it is what
#: curvature.lazy_measure() implements. Ni et al. use a weight-proportional
#: measure and no number here transfers to it unre-measured. See the docstring.
MEASURE = "uniform lazy (curvature.lazy_measure), alpha = %g" % cv.ALPHA

#: Steps in the reported planted-bed runs. PINNED.
N_STEPS = 20

#: The long run, on the small bed, used to reach an actual fixed point.
LONG_STEPS = 100
LONG_TOL = 1e-9

#: The spec rule's step size. PINNED at 1.0 because that is the value at which
#: the spec's rule and Eq. 10 coincide on a shortcut-free graph (DERIVED in the
#: docstring, measured in demo (b)): it is the most favourable setting the
#: spec's identity claim can be given. Swept in demo (e).
ETA = 1.0

#: Quantile of the weight distribution above which edges are cut in the
#: single-community negative control. 0.90 = the heaviest tenth.
CUT_QUANTILE = 0.90

#: The planted bed: two ER blocks, N_CROSS edges between them. n_cross = 5 rather
#: than 1 because with a single bridge the heaviest-edge cut is already correct
#: before any flow runs and the control would be doing the work.
BLOCK_N, BLOCK_P, N_CROSS = 20, 0.30, 5

#: The small bed, where a run long enough to reach a fixed point is affordable.
SMALL_N, SMALL_P, SMALL_CROSS = 12, 0.40, 3

#: Finite-difference step for log_jacobian. PINNED at 1e-2 with CENTRAL
#: differences: below that the transport LP's own solver tolerance dominates and
#: the gauge check J @ 1 = 0 fails outright. See log_jacobian.
JAC_EPS = 1e-2

#: The bed the Jacobian is taken on. Smaller than the flow bed because a central
#: difference costs two flow steps per EDGE, so the cost is quadratic in density.
JAC_N, JAC_P = 6, 0.50

#: The measured plateau onset of the raw residual on the small bed, and the
#: tolerance the assert bounds it by ON BOTH SIDES. An assert of the form
#: "onset > 21" fires when the onset moves earlier and never when it moves later,
#: which is a floor and not a guard; forced to 90 such an assert passes clean.
ONSET_STEP = 53
ONSET_TOL = 4

#: eta values swept in demo (e). -2.0 is Foreman's sign flip, run on the LENGTH
#: variable this module uses; it is in the sweep to be measured, not endorsed.
#: -2.0 is Foreman's sign flip on the LENGTH variable; it is measured in (k) on
#: the cheap bed rather than here, because it converges to nothing and 101 steps
#: of a 66-edge bed spent reaching a refusal is the most expensive evidence in
#: the file.
ETA_SWEEP = (0.5, 1.0, 2.0)

RULES = ("ni", "spec", "recip")

#: This module's own refusals, on top of whatever curvature.py refuses.
NON_POSITIVE_WEIGHT = "non-positive-weight"
LP_FAILED = "transport-lp-failed"
UNKNOWN_RULE = "unknown-rule"


# ---------------------------------------------------------------------------
# THE GAUGE
# ---------------------------------------------------------------------------

def gauge_normalise(W):
    """Rescale W so its positive edge weights have geometric mean 1.

    kappa is invariant under a global rescale (demo (a) measures it) and all
    three updates are 1-HOMOGENEOUS in w, so this commutes with every step
    exactly -- demo (a) asserts the commutation to 1e-12 rather than assuming
    it. It is applied to stop the gauge from drifting the weights toward
    overflow over a hundred steps, and it changes no measured quantity.
    """
    W = np.asarray(W, dtype=float)
    up = W[np.triu_indices_from(W, 1)]
    up = up[up > 0]
    if up.size == 0:
        return W.copy()
    return W / float(np.exp(np.log(up).mean()))


def shortcut_census(W):
    """(edges that are NOT geodesics, total edges) -- the regime indicator.

    An edge with d(i,j) < w_ij is a shortcut edge. Eq. 10 multiplies by d, the
    spec multiplies by w: the two rules are the same map exactly when this count
    is zero (and eta = 1, target 0).
    """
    W = np.asarray(W, dtype=float)
    D = cv.graph_metric(W)
    ii, jj = np.nonzero(np.triu(W, 1))
    if ii.size == 0:
        return 0, 0
    return int((D[ii, jj] < W[ii, jj] - 1e-12).sum()), int(ii.size)


def eta_ceiling(kappa, target=0.0):
    """Largest eta for which the spec rescale keeps every weight positive.

    w(1 - eta(kappa - t)) > 0 with w > 0 iff eta (kappa - t) < 1, so for positive
    residuals eta < 1 / max_e (kappa_e - t). inf when no residual is positive.
    """
    r = np.asarray(kappa, dtype=float) - target
    return float("inf") if r.size == 0 or r.max() <= 0 else float(1.0 / r.max())


# ---------------------------------------------------------------------------
# THE RULES. Three separate functions. None is silently another.
# ---------------------------------------------------------------------------

def _curvature_pass(W, alpha=cv.ALPHA):
    """kappa on every edge of the CURRENT weighted graph, with ITS metric.

    The metric is recomputed HERE and nowhere else, so a difference between two
    rules is a difference between the UPDATES and never between two notions of
    distance. Freezing d instead is the silent failure this module refuses to
    commit, and demo (g) fires the trap on this module's own bed rather than
    quoting anyone: a frozen d reaches a gauge residual of 0.000e+00 -- an exact
    fixed point, green on every diagnostic -- while its recomputed curvature
    spread is 6.560e-01 against the live-metric run's 1.035e-02, so the run whose
    residual looks perfect is the one whose geometry is 63.4x worse.

    A HiGHS failure inside the transport LP is returned as a REFUSAL rather than
    raised: it means this module has driven curvature.py's input out of the range
    the LP can solve, which is a fact about the flow and belongs in the census.
    """
    W = np.asarray(W, dtype=float)
    try:
        D = cv.graph_metric(W)
        return cv.curvature_from_graph(W, alpha=alpha, D=D), D
    except (RuntimeError, ValueError) as exc:
        up = W[np.triu_indices_from(W, 1)]
        up = up[up > 0]
        span = (up.max() / up.min()) if up.size else float("nan")
        return Refusal(LP_FAILED,
                       "the transport LP could not be solved on this graph "
                       "(weight range %.3e .. %.3e, span %.3e): the flow has "
                       "driven the weights out of the LP's numerical domain "
                       "[%s]" % (up.min() if up.size else float("nan"),
                                 up.max() if up.size else float("nan"),
                                 span, exc)), None


def _finish(W_old, W_new, res, refusals):
    moved = np.abs(W_new - W_old)
    return dict(W=W_new, refusals=refusals, edges=res["edges"], kappa=res["kappa"],
                n_edges=res["n_edges"], n_refused=int(sum(refusals.values())),
                residual_abs=float(moved.max()) if moved.size else 0.0,
                residual_gauge=float(np.abs(gauge_normalise(W_new)
                                            - gauge_normalise(W_old)).max()))


def _step(W, alpha, update):
    """Shared plumbing: one curvature pass, then `update` per edge, then census."""
    W = np.asarray(W, dtype=float)
    got = _curvature_pass(W, alpha)
    if is_refusal(got[0]):
        return got[0]
    res, D = got
    new = W.copy()
    refusals = dict(res["refusals"])
    for (i, j), k in zip(res["edges"], res["kappa"]):
        w = update(i, j, k, W[i, j], D[i, j])
        if not np.isfinite(w) or w <= 0.0:
            refusals[NON_POSITIVE_WEIGHT] = refusals.get(NON_POSITIVE_WEIGHT, 0) + 1
            continue
        new[i, j] = new[j, i] = w
    return _finish(W, new, res, refusals)


def ni_flow_step(W, alpha=cv.ALPHA):
    """ONE step of Ni et al. Eq. 10:  w' = (1 - kappa) * d(i, j).

    d is the shortest-path metric of the WEIGHTED graph recomputed at this step
    -- NOT the incident weight w_ij, which differs on any edge that is not a
    geodesic, and NOT the hop metric. No step size and no target exist in this
    rule: the weight is REPLACED, not scaled.

    kappa <= 1 always (kappa = 1 - W1/d, W1 >= 0) and W1 > 0 for i != j under the
    lazy measure, so the new weight is positive by construction. It is checked
    anyway, and a non-positive result becomes a refusal rather than a negative
    length -- which shortest_path would consume silently and Bellman-Ford would
    reject as a negative cycle.
    """
    return _step(W, alpha, lambda i, j, k, w, d: (1.0 - k) * float(d))


def spec_flow_step(W, eta=ETA, kappa_target=0.0, alpha=cv.ALPHA):
    """ONE step of the PHASE C spec's update:  w' = w (1 - eta (kappa - target)).

    A multiplicative rescale of the EXISTING weight. kappa_target = 0.0 is the
    zero-source case the spec binds to Ni. `kappa_target` is a scalar or a dict
    keyed by the (i, j) edge with i < j; a missing edge takes 0.0.

    Unlike Eq. 10 this rule CAN produce a non-positive weight, whenever
    eta (kappa - t) >= 1 -- see eta_ceiling() for the exact bound. Such an edge is
    refused with a reason and held at its old weight.
    """
    scalar = not isinstance(kappa_target, dict)
    def update(i, j, k, w, d):
        t = float(kappa_target) if scalar else float(kappa_target.get((i, j), 0.0))
        return w * (1.0 - eta * (k - t))
    return _step(W, alpha, update)


def recip_flow_step(W, eta=ETA, kappa_target=0.0, alpha=cv.ALPHA):
    """THE REPAIR ARM:  w' = w / (1 + eta (kappa - target)).

    Foreman's reroute is to treat the flow variable as an ATTENTION a = 1/w and
    flip the sign, a' = a (1 + eta r). Mapped back into the LENGTH variable that
    Eq. 10 and curvature.graph_metric() both consume, a' = a(1 + eta r) is
    exactly w' = w / (1 + eta r) -- which is the spec's own update with a
    division where it has a subtraction. The two agree to O(eta^2 r^2), and the
    division form has NO eta ceiling on the positive side: 1 + eta r > 0 fails
    only for eta r <= -1, so a positive-curvature edge can never be driven
    through zero however large eta is. That is the whole content of the repair,
    and demo (e) measures whether it buys anything.
    """
    scalar = not isinstance(kappa_target, dict)
    def update(i, j, k, w, d):
        t = float(kappa_target) if scalar else float(kappa_target.get((i, j), 0.0))
        den = 1.0 + eta * (k - t)
        return w / den if den > 0.0 else -1.0
    return _step(W, alpha, update)


def run_flow(W, rule, n_steps=N_STEPS, eta=ETA, kappa_target=0.0, alpha=cv.ALPHA,
             gauge=True, freeze_metric=False, tol=None):
    """Run one rule and keep the WHOLE trace, not its last entry.

    A final scalar cannot tell convergence from a plateau and cannot see an
    oscillation at all, so every per-step quantity is a column: residual_abs and
    residual_gauge, the curvature spectrum entering each step (min, med, max,
    spread) beside that step's EDGE COUNT, the shortcut census, and the eta
    ceiling that step's curvature implies.

    gauge=True renormalises the state to geometric mean 1 after each step. It
    commutes with every update exactly (all three are 1-homogeneous, asserted in
    demo (a)) and exists only to stop the ray coordinate drifting toward
    overflow. freeze_metric=True is NOT THE FLOW -- it holds d at its step-0
    value, which is the silent failure demo (g) fires.
    """
    if rule not in RULES:
        return Refusal(UNKNOWN_RULE,
                       "no rule named %r: the implemented rules are %r"
                       % (rule, RULES))
    W = np.asarray(W, dtype=float).copy()
    tr = {k: [] for k in ("residual_abs", "residual_gauge", "kappa_min",
                          "kappa_med", "kappa_max", "kappa_spread", "n_edges",
                          "n_refused", "w_min", "w_max", "shortcuts",
                          "eta_ceiling")}
    tr.update(rule=rule, eta=(None if rule == "ni" else eta), n_steps=0,
              gauge=gauge, freeze_metric=freeze_metric, measure=MEASURE,
              refusal=None, W=W)
    D0 = cv.graph_metric(W) if freeze_metric else None
    for _ in range(n_steps):
        sc, ne = shortcut_census(W)
        if freeze_metric:
            res = cv.curvature_from_graph(W, alpha=alpha, D=D0)
            new = W.copy()
            refusals = dict(res["refusals"])
            for (i, j), k in zip(res["edges"], res["kappa"]):
                w = ((1.0 - k) * float(D0[i, j]) if rule == "ni"
                     else W[i, j] * (1.0 - eta * k))
                if np.isfinite(w) and w > 0.0:
                    new[i, j] = new[j, i] = w
                else:
                    refusals[NON_POSITIVE_WEIGHT] = \
                        refusals.get(NON_POSITIVE_WEIGHT, 0) + 1
            out = _finish(W, new, res, refusals)
        elif rule == "ni":
            out = ni_flow_step(W, alpha)
        elif rule == "spec":
            out = spec_flow_step(W, eta, kappa_target, alpha)
        else:
            out = recip_flow_step(W, eta, kappa_target, alpha)
        if is_refusal(out):
            tr["refusal"] = out
            break
        k = np.asarray(out["kappa"], dtype=float)
        tr["residual_abs"].append(out["residual_abs"])
        tr["residual_gauge"].append(out["residual_gauge"])
        tr["kappa_min"].append(float(k.min()) if k.size else float("nan"))
        tr["kappa_med"].append(float(np.median(k)) if k.size else float("nan"))
        tr["kappa_max"].append(float(k.max()) if k.size else float("nan"))
        tr["kappa_spread"].append(float(k.max() - k.min()) if k.size else float("nan"))
        tr["n_edges"].append(int(out["n_edges"]))
        tr["n_refused"].append(int(out["n_refused"]))
        tr["shortcuts"].append(sc)
        tr["eta_ceiling"].append(eta_ceiling(k, 0.0))
        W = gauge_normalise(out["W"]) if gauge else out["W"]
        up = W[np.triu_indices_from(W, 1)]
        up = up[up > 0]
        tr["w_min"].append(float(up.min()) if up.size else float("nan"))
        tr["w_max"].append(float(up.max()) if up.size else float("nan"))
        tr["n_steps"] += 1
        if tol is not None and out["residual_gauge"] < tol:
            break
    tr["W"] = W
    return tr


def apply_rule(W, rule, eta=ETA, kappa_target=0.0, alpha=cv.ALPHA):
    """One step of the named rule. The single dispatch point, so no caller can
    silently run a different map from the one it names."""
    if rule == "ni":
        return ni_flow_step(W, alpha)
    if rule == "spec":
        return spec_flow_step(W, eta, kappa_target, alpha)
    if rule == "recip":
        return recip_flow_step(W, eta, kappa_target, alpha)
    return Refusal(UNKNOWN_RULE, "no rule named %r: the implemented rules are %r"
                                 % (rule, RULES))


def weight_error(W, W_star):
    """max|w / w* - 1| over edges, AFTER gauge-normalising both. (err, edge).

    THE POINT OF THE GAUGE NORMALISATION. kappa is scale-invariant, so a perfect
    recovery multiplied by a large constant differs from its target enormously in
    raw weights and not at all in geometry. An unnormalised comparison measures
    the gauge and reports it as error; this one cannot.

    Returns the WORST edge as well as the worst value, because a single bad edge
    and a uniformly-off vector are different failures and a max alone cannot
    tell them apart.
    """
    A, B = gauge_normalise(W), gauge_normalise(W_star)
    ii, jj = np.nonzero(np.triu(B, 1))
    if not np.array_equal(np.nonzero(np.triu(A, 1)), (ii, jj)):
        raise ValueError("the two graphs have different edge sets: a weight error "
                         "between them is not defined")
    ratio = A[ii, jj] / B[ii, jj]
    k = int(np.argmax(np.abs(ratio - 1.0)))
    return float(np.abs(ratio - 1.0).max()), (int(ii[k]), int(jj[k]))


def recovery_table(rules=(("ni", None),), n_steps=None, tol=None, sigma=0.30,
                   seed=3, W=None, w_star=None, alpha=cv.ALPHA):
    """Perturb a fixed point, re-run each rule, report residual AND weight error.

    THE QUESTION THIS ANSWERS, and it is not the one a residual answers. A
    residual says the iteration stopped moving. It does not say it stopped in the
    right place. So: take the flow's OWN fixed point w* (running to it if not
    handed one -- an arbitrary w* would be a strawman, since a flow has no reason
    to return to a point that is not fixed), perturb it multiplicatively by
    lognormal(sigma), and re-run. max|w/w* - 1| after gauge normalisation then
    says whether convergence in kappa is convergence in geometry.
    """
    n_steps = LONG_STEPS if n_steps is None else n_steps
    tol = LONG_TOL if tol is None else tol
    if W is None:
        W, _ = cv.two_block_bed(seed=0, n_per_block=SMALL_N, p=SMALL_P,
                                n_cross=SMALL_CROSS)
    W = gauge_normalise(W)
    if w_star is None:
        w_star = run_flow(W, "ni", n_steps, gauge=True, tol=tol, alpha=alpha)["W"]
    w_star = gauge_normalise(w_star)

    rng = np.random.default_rng(seed)
    ii, jj = np.nonzero(np.triu(w_star, 1))
    start = w_star.copy()
    kick = np.exp(rng.normal(scale=sigma, size=ii.size))
    start[ii, jj] = start[jj, ii] = w_star[ii, jj] * kick
    start = gauge_normalise(start)
    err0, edge0 = weight_error(start, w_star)

    rows = []
    for rule, eta in rules:
        tr = run_flow(start, rule, n_steps, eta=(ETA if eta is None else eta),
                      gauge=True, tol=tol, alpha=alpha)
        if tr["refusal"] is not None:
            rows.append(dict(name="%s eta=%s" % (rule, eta), steps=tr["n_steps"],
                             resid=float("nan"), w_err=float("nan"),
                             w_err_edge=None, kappa_const=float("nan"),
                             kappa_spread=float("nan"), n_edges=0,
                             refusal=tr["refusal"]))
            continue
        sp = spectrum(tr["W"])
        err, edge = weight_error(tr["W"], w_star)
        rows.append(dict(name="%s eta=%s" % (rule, eta), steps=tr["n_steps"],
                         resid=tr["residual_gauge"][-1], w_err=err,
                         w_err_edge=edge, kappa_const=sp["med"],
                         kappa_spread=sp["spread"], n_edges=sp["n_edges"],
                         refusal=None))
    return dict(rows=rows, w_star=w_star, start=start, sigma=sigma, seed=seed,
                start_err=err0, start_edge=edge0, n_edges=int(ii.size))


def log_jacobian(W, rule="ni", eta=ETA, kappa_target=0.0, alpha=cv.ALPHA,
                 eps=JAC_EPS, central=True):
    """Finite-difference Jacobian of the gauge-normalised step, in LOG weights.

    Log coordinates because the gauge is a TRANSLATION there (w -> cw is
    x -> x + log(c)) and every update here is multiplicative, so the linearised
    map is the natural object. Gauge-normalising the output makes the all-ones
    direction exactly annihilated, so J @ 1 = 0 is the check that this is the
    Jacobian of the quotient map and not of the raw one.

    CENTRAL differences, and the step size is not free. The transport LP returns
    a vertex to a solver tolerance of roughly 1e-9, so a one-sided difference at a
    small eps divides that noise by eps and the gauge check J @ 1 = 0 comes back
    two orders too large -- the Jacobian is then noise wearing a spectrum. Central
    differences at the pinned eps recover it; a sweep over one-sided and central
    at several eps was run to choose the pinning, and its winner is the setting
    below. J @ 1 is PRINTED as the gate on every use, so a reader never has to
    trust that choice: if the gate is not small, nothing computed from J means
    anything.

    Costs two flow steps per edge (central). Returns (J, edges).
    """
    W = gauge_normalise(np.asarray(W, dtype=float))
    edges = [(int(i), int(j)) for i, j in zip(*np.nonzero(np.triu(W, 1)))]
    n = len(edges)

    def L(x):
        Wx = np.zeros_like(W)
        for (i, j), v in zip(edges, x):
            Wx[i, j] = Wx[j, i] = float(np.exp(v))
        out = apply_rule(Wx, rule, eta, kappa_target, alpha)
        if is_refusal(out):
            raise RuntimeError("the step refused inside the Jacobian: %s"
                               % out.reason)
        Y = gauge_normalise(out["W"])
        return np.array([np.log(Y[i, j]) for i, j in edges])

    x0 = np.array([np.log(W[i, j]) for i, j in edges])
    f0 = None if central else L(x0)
    J = np.empty((n, n))
    for k in range(n):
        xp = x0.copy()
        xp[k] += eps
        if central:
            xm = x0.copy()
            xm[k] -= eps
            J[:, k] = (L(xp) - L(xm)) / (2.0 * eps)
        else:
            J[:, k] = (L(xp) - f0) / eps
    return J, edges


def flat_modes(J, tol=1e-3):
    """Which directions the step does NOT contract, from the Jacobian spectrum.

    At a fixed point of the gauge-normalised map:
      |lambda| ~ 0  the gauge direction, PROJECTED OUT by the normalisation --
                    quotiented away, not flat;
      |lambda| ~ 1  a direction the flow PRESERVES: a genuine second dimension of
                    the fixed-point set, and exactly what lets a converged
                    residual sit on weights that are not the planted ones;
      |lambda| < 1  contracted, so a perturbation along it dies.

    n_unit is therefore the dimension of the fixed-point set modulo gauge. Zero
    means a converged residual DOES pin the weights locally; one or more means it
    does not, and the count says how many extra normalisations it would take.
    """
    ev = np.linalg.eigvals(np.asarray(J, dtype=float))
    mag = np.sort(np.abs(ev))[::-1]
    return dict(eigenvalues=ev, magnitudes=mag.tolist(),
                n_unit=int((np.abs(mag - 1.0) < tol).sum()),
                n_zero=int((mag < tol).sum()),
                spectral_radius=float(mag[0]) if mag.size else float("nan"),
                second=float(mag[1]) if mag.size > 1 else float("nan"),
                gauge_residual=float(np.abs(np.asarray(J) @ np.ones(J.shape[0])).max()),
                tol=tol, n=int(mag.size))


def plateau_onset(residual, rel=1e-3):
    """First step from which every later residual is within `rel` of the last.

    The onset is MEASURED and printed, never asserted in prose. An earlier draft
    of this module's docstring said the raw residual "floors at 5.5853e-02 and
    sits there from step 21"; at step 21 the value is 5.6191e-02 and the floor is
    first attained 32 steps later. The plateau was real and the finding stood,
    but the step named in the sentence was not the step at which the value held,
    which is the class of error this function exists to remove.

    Returns (onset_step, drift_from_that_step) 1-indexed, or (None, nan) if no
    suffix of the column is that flat.
    """
    r = np.asarray(residual, dtype=float)
    if r.size == 0 or r[-1] == 0.0:
        return None, float("nan")
    within = np.abs(r - r[-1]) / abs(r[-1]) <= rel
    onset = None
    for i in range(r.size):
        if within[i:].all():
            onset = i
            break
    if onset is None:
        return None, float("nan")
    drift = float(np.abs(r[onset:] - r[-1]).max() / abs(r[-1]))
    return onset + 1, drift


def first_attained(residual, fmt="%.4e"):
    """First 1-indexed step whose residual PRINTS as the last step's does.

    The strictest honest reading of "it floors at X": the first step at which the
    printed figure stops changing. Weaker than plateau_onset and reported beside
    it, because the two answer different questions and quoting one for the other
    is how the 32-step error happened.
    """
    r = np.asarray(residual, dtype=float)
    if r.size == 0:
        return None
    target = fmt % r[-1]
    for i, v in enumerate(r):
        if fmt % v == target and all(fmt % u == target for u in r[i:]):
            return i + 1
    return None


def residual_verdict(residual, tol=1e-9):
    """converges / decays to a floor / plateaus / diverges / oscillates -- MEASURED.

    Disjoint conditions on the column, in order: last entry under tol is
    convergence; three or more sign changes in the increment is oscillation;
    otherwise last/first decides decay, plateau or divergence.
    """
    r = np.asarray(residual, dtype=float)
    if r.size == 0:
        return "no steps taken"
    if r[-1] < tol:
        return "CONVERGES (last %.3e < %.0e, %d steps)" % (r[-1], tol, r.size)
    d = np.diff(r)
    flips = int((np.sign(d[:-1]) * np.sign(d[1:]) < 0).sum()) if d.size > 1 else 0
    if flips >= 3:
        return "OSCILLATES (%d sign changes, last %.3e)" % (flips, r[-1])
    ratio = r[-1] / r[0] if r[0] > 0 else float("inf")
    if ratio > 1.5:
        return "DIVERGES (last/first %.3g, last %.3e)" % (ratio, r[-1])
    if ratio < 0.5:
        return "DECAYS, floor not yet reached (last/first %.3g, last %.3e)" \
               % (ratio, r[-1])
    return "PLATEAUS at %.3e (last/first %.3g)" % (r[-1], ratio)


def spectrum(W, alpha=cv.ALPHA, bins=8):
    """The curvature phase of a graph: min, median, max, histogram AND n_edges."""
    got = _curvature_pass(W, alpha)
    if is_refusal(got[0]):
        return dict(refusal=got[0], n_edges=0, kappa=np.array([]))
    k = np.asarray(got[0]["kappa"], dtype=float)
    if k.size == 0:
        return dict(min=float("nan"), med=float("nan"), max=float("nan"),
                    counts=[], edges_hist=[], n_edges=0, kappa=k, refusal=None)
    counts, edges_hist = np.histogram(k, bins=bins)
    return dict(min=float(k.min()), med=float(np.median(k)), max=float(k.max()),
                spread=float(k.max() - k.min()), counts=counts.tolist(),
                edges_hist=edges_hist.tolist(), n_edges=int(k.size), kappa=k,
                refusal=None)


# ---------------------------------------------------------------------------
# THE BEDS
# ---------------------------------------------------------------------------

def planted_two_community(seed=0, n_per_block=BLOCK_N, p=BLOCK_P, n_cross=N_CROSS):
    """Two ER blocks joined by n_cross edges, with ground truth. (W, labels).

    Inherited from curvature.two_block_bed -- the SAME bed the curvature
    instrument was validated on, so the flow's separation cannot be an artefact
    of a bed built to suit it.
    """
    return cv.two_block_bed(seed=seed, n_per_block=n_per_block, p=p, n_cross=n_cross)


def planted_one_community(seed=100, n=2 * BLOCK_N, p=BLOCK_P):
    """THE PLANTED NEGATIVE: ONE block, same node count, same density, no plant."""
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n))
    perm = rng.permutation(n)
    for a, b in zip(perm[:-1], perm[1:]):
        A[a, b] = A[b, a] = 1.0
    iu = np.triu_indices(n, 1)
    hit = rng.random(iu[0].size) < p
    A[iu[0][hit], iu[1][hit]] = 1.0
    return np.maximum(A, A.T)


def shortcut_seeded_bed(seed=7, n_heavy=6, heavy=6.0):
    """The planted bed with n_heavy intra edges set to `heavy`, so d(i,j) < w_ij.

    The ONLY regime in which Eq. 10 and the spec's rule are different maps, and
    therefore the only bed on which the question "do they share fixed points" has
    a trajectory to measure rather than an identity to restate.
    """
    W, labels = planted_two_community()
    rng = np.random.default_rng(seed)
    ii, jj = np.nonzero(np.triu(W, 1))
    for q in rng.choice(ii.size, n_heavy, replace=False):
        W[ii[q], jj[q]] = W[jj[q], ii[q]] = heavy
    return W, labels


# ---------------------------------------------------------------------------
# THE CUT, and the score
# ---------------------------------------------------------------------------

def adjusted_rand_index(a, b):
    """ARI from the contingency table, by the definition. Permutation-free."""
    a, b = np.asarray(a), np.asarray(b)
    comb2 = lambda x: x * (x - 1) / 2.0
    tab = np.array([[float(np.sum((a == x) & (b == y))) for y in np.unique(b)]
                    for x in np.unique(a)])
    sij, si, sj = (comb2(tab).sum(), comb2(tab.sum(axis=1)).sum(),
                   comb2(tab.sum(axis=0)).sum())
    exp = si * sj / comb2(float(a.size))
    denom = 0.5 * (si + sj) - exp
    return 1.0 if denom == 0.0 else float((sij - exp) / denom)


def _components(W, thresh):
    """Connected components of W after deleting every edge with weight > thresh."""
    n, lab = connected_components(
        csgraph_from_dense(np.where(W > thresh, 0.0, W), null_value=0),
        directed=False)
    return n, lab


def best_cut_ari(W, labels):
    """Cut edges above a weight threshold; the BEST ARI over the whole sweep.

    The threshold is swept over every distinct edge weight, and the SAME sweep is
    applied to every arm -- flowed, unflowed, either rule. A per-arm tuned
    threshold would be the arm choosing its own scoring rule; a single fixed
    threshold is meaningless across arms whose weights differ by orders of
    magnitude. The best of an identical sweep favours the CONTROL exactly as much
    as the treatment, which is the direction a conservative comparison errs in.
    """
    W, labels = np.asarray(W, dtype=float), np.asarray(labels)
    w = np.unique(W[np.triu_indices_from(W, 1)])
    w = w[w > 0]
    if w.size == 0:
        return dict(ari=0.0, thresh=float("nan"), n_components=labels.size,
                    pred=np.arange(labels.size), n_cut=0)
    best = None
    up = W[np.triu_indices_from(W, 1)]
    for t in np.concatenate([[0.0], w]):
        n_comp, lab = _components(W, t)
        ari = adjusted_rand_index(labels, lab)
        if best is None or ari > best["ari"]:
            best = dict(ari=ari, thresh=float(t), n_components=int(n_comp),
                        pred=lab, n_cut=int((up > t).sum()))
    return best


def largest_component_fraction(W, quantile=CUT_QUANTILE):
    """Cut the heaviest (1 - quantile) of edges; what fraction survives together?

    The single-community negative control. If a flow with no plant under it can
    be made to shatter by the standard cut, the cut rule is the finding.
    """
    W = np.asarray(W, dtype=float)
    up = W[np.triu_indices_from(W, 1)]
    up = up[up > 0]
    _, lab = _components(W, float(np.quantile(up, quantile)))
    return float(np.bincount(lab).max()) / W.shape[0]


def cross_edge_mass(W, labels):
    """Median weight on planted cross edges vs planted intra edges, and the ratio."""
    W, labels = np.asarray(W, dtype=float), np.asarray(labels)
    ii, jj = np.triu_indices_from(W, 1)
    m = W[ii, jj] > 0
    ii, jj = ii[m], jj[m]
    cross = labels[ii] != labels[jj]
    wc, wi = W[ii[cross], jj[cross]], W[ii[~cross], jj[~cross]]
    return dict(cross_med=float(np.median(wc)), n_cross=int(wc.size),
                intra_med=float(np.median(wi)), n_intra=int(wi.size),
                ratio=float(np.median(wc) / np.median(wi)))


# ---------------------------------------------------------------------------
# SELF-CHECK
# ---------------------------------------------------------------------------

def _print_trace(name, tr, stride=1):
    print("    %s | rule=%s eta=%s steps=%d gauge=%s measure: %s"
          % (name, tr["rule"], tr["eta"], tr["n_steps"], tr["gauge"], tr["measure"]))
    print("    %4s %12s %12s | %+9s %+9s %10s %6s | %5s %9s"
          % ("step", "resid_raw", "resid_GAUGE", "k_min", "k_max", "k_spread",
             "edges", "short", "eta_ceil"))
    for s in range(tr["n_steps"]):
        if s % stride and s != tr["n_steps"] - 1:
            continue
        print("    %4d %12.4e %12.4e | %+9.5f %+9.5f %10.3e %6d | %2d/%-2d %9.4g"
              % (s + 1, tr["residual_abs"][s], tr["residual_gauge"][s],
                 tr["kappa_min"][s], tr["kappa_max"][s], tr["kappa_spread"][s],
                 tr["n_edges"][s], tr["shortcuts"][s], tr["n_edges"][s],
                 tr["eta_ceiling"][s]))
    if stride > 1:
        print("    (column decimated to every %dth step plus the last; all %d "
              "steps were run)" % (stride, tr["n_steps"]))
    print("    VERDICT raw  : %s" % residual_verdict(tr["residual_abs"]))
    print("    VERDICT GAUGE: %s" % residual_verdict(tr["residual_gauge"]))


def demo():
    t0 = time.time()
    print("SOURCE OF Eq. 10: arXiv:1907.03993 (Ni, Lin, Luo, Gao, Sci. Rep. 9),")
    print("    w^(k+1)_ij = (1 - kappa^(k)_ij) d^(k)(i,j), d recomputed each step.")
    print("MEASURE FOR EVERY NUMBER BELOW: %s" % MEASURE)
    print("DOES IT TRANSFER TO NI'S WEIGHT-PROPORTIONAL MEASURE? Only at step 0.")
    print("DERIVED: on an UNWEIGHTED graph w_xz / sum_z w_xz = 1/deg(x) term by")
    print("term, so the two measures ARE the same measure there. The input bed is")
    print("unweighted -- and the flow leaves that regime at step 1, which is the")
    print("whole point of the flow. Measured on the planted bed:")
    W0, _ = planted_two_community()
    for tag, X in (("step 0, the input bed", W0),
                   ("after 1 Ni step", ni_flow_step(W0)["W"])):
        u = np.asarray(cv.curvature_from_graph(X, measure=cv.UNIFORM)["kappa"])
        p = np.asarray(cv.curvature_from_graph(
            X, measure=cv.WEIGHT_PROPORTIONAL)["kappa"])
        up = X[np.triu_indices_from(X, 1)]
        up = up[up > 0]
        print("    %-22s max|uniform - weight-prop| = %.3e over %d edges, %d"
              % (tag, float(np.abs(u - p).max()), u.size, np.unique(up).size))
        print("    %-22s distinct weights" % "")
    print("    So the it.2 INPUT transfers with nothing to re-measure; every number")
    print("    taken at step >= 1 is measure-dependent and does not. (c) prints the")
    print("    gap at step 20.\n")

    print("(a) THE GAUGE, first, because it decides what a residual means. kappa is")
    print("    invariant under a global rescale of the weights, and all three")
    print("    updates are 1-homogeneous in w, so the dynamics live on a ray.")
    W, labels = planted_two_community()
    base = np.asarray(cv.curvature_from_graph(W)["kappa"])
    for c in (2.0, 13.7, 1000.0):
        k = np.asarray(cv.curvature_from_graph(c * W)["kappa"])
        gap = float(np.abs(k - base).max())
        print("    c = %-8g max|kappa(cW) - kappa(W)| = %.3e   EDGES %d"
              % (c, gap, k.size))
        assert gap < 1e-10, "curvature is not gauge invariant at c = %g" % c
    homog = {}
    for nm, f in (("ni", lambda X: ni_flow_step(X)["W"]),
                  ("spec", lambda X: spec_flow_step(X, eta=ETA)["W"]),
                  ("recip", lambda X: recip_flow_step(X, eta=ETA)["W"])):
        lhs = gauge_normalise(f(W))
        rhs = gauge_normalise(f(gauge_normalise(3.7 * W)))
        homog[nm] = float(np.abs(lhs - rhs).max())
        print("    %-6s 1-homogeneous: max|G(F(W)) - G(F(G(3.7W)))| = %.3e"
              % (nm, homog[nm]))
        assert homog[nm] < 1e-10, \
            "%s does not commute with the gauge: normalising would change the run" % nm
    print("    PER RULE, and NOT one figure for all three. ni and spec sit at")
    print("    machine epsilon because their update MULTIPLIES by a gauge-invariant")
    print("    factor; recip DIVIDES by one, so it is the loosest of the three here")
    print("    (%.3e against %.3e) and is bed-dependent -- a second bed read it"
          % (homog["recip"], homog["ni"]))
    print("    orders looser again. The bound asserted for all three is 1e-10.")
    assert homog["recip"] == max(homog.values()), \
        ("recip is no longer the loosest of ALL THREE (%r): the claim is about the "
         "three together, so the assert compares it against every one of them and "
         "not against ni alone"
         % {k: "%.3e" % v for k, v in sorted(homog.items())})

    print("(b) TWO RULES, NOT ONE -- and the exact regime in which they coincide.")
    print("    DERIVED: on a SHORTCUT-FREE graph d(i,j) = w_ij on every edge, so")
    print("    Eq. 10's (1 - kappa) d and the spec's w (1 - eta kappa) at eta = 1")
    print("    are the same product. Off that regime they are not.")
    Ws = np.zeros((5, 5))
    for (i, j), w in {(0, 1): 10.0, (0, 2): 1.0, (1, 2): 1.0,
                      (2, 3): 1.0, (3, 4): 1.0, (2, 4): 1.0}.items():
        Ws[i, j] = Ws[j, i] = w
    Ds = cv.graph_metric(Ws)
    ks = float(cv.kappa_edge(Ws, Ds, 0, 1))
    sc, ne = shortcut_census(Ws)
    print("    shortcut graph (%d/%d edges are shortcuts): w_01 = %.1f, d(0,1) = %.1f,"
          % (sc, ne, Ws[0, 1], Ds[0, 1]))
    print("    kappa_01 = %+.6f -> Eq. 10 gives %.6f, spec eta=1 gives %.6f "
          "(factor %.3g)" % (ks, ni_flow_step(Ws)["W"][0, 1],
                             spec_flow_step(Ws, eta=1.0)["W"][0, 1],
                             spec_flow_step(Ws, eta=1.0)["W"][0, 1]
                             / ni_flow_step(Ws)["W"][0, 1]))
    assert abs(ni_flow_step(Ws)["W"][0, 1] - (1 - ks) * Ds[0, 1]) < 1e-12
    assert abs(spec_flow_step(Ws, eta=1.0)["W"][0, 1] - Ws[0, 1] * (1 - ks)) < 1e-12
    assert abs(ni_flow_step(Ws)["W"][0, 1]
               - spec_flow_step(Ws, eta=1.0)["W"][0, 1]) > 1e-9, \
        "the two rules agree even on a shortcut graph: one is silently the other"
    print("    UNDER FLOW, on the planted bed with %d intra edges seeded at weight"
          % 6)
    print("    6.0 so shortcuts exist at step 0. Eq. 10 multiplies by d and heals")
    print("    them; the spec multiplies by w and takes longer.")
    Wh, _ = shortcut_seeded_bed()
    A = B = Wh
    print("    %4s %14s %12s %12s" % ("step", "max|ni-spec|", "short_in_ni",
                                      "short_in_spec"))
    heal_ni = heal_sp = None
    for s in range(1, 8):
        sa, ne = shortcut_census(A)
        sb, _ = shortcut_census(B)
        if sa == 0 and heal_ni is None:
            heal_ni = s - 1
        if sb == 0 and heal_sp is None:
            heal_sp = s - 1
        A = ni_flow_step(A)["W"]
        B = spec_flow_step(B, eta=1.0, kappa_target=0.0)["W"]
        print("    %4d %14.4e %9d/%-3d %9d/%-3d"
              % (s, float(np.abs(A - B).max()), sa, ne, sb, ne))
    print("    Eq. 10 was shortcut-free after %s steps, the spec after %s."
          % (heal_ni, heal_sp))
    assert heal_ni is not None and (heal_sp is None or heal_ni < heal_sp), \
        ("Eq. 10 no longer heals shortcuts faster than the rescale (%r vs %r): the "
         "recorded mechanism must be re-measured" % (heal_ni, heal_sp))

    print("(c) T-FLOW CONVERGENCE on the planted bed (%d + %d nodes, p = %g, %d"
          % (BLOCK_N, BLOCK_N, BLOCK_P, N_CROSS))
    print("    cross edges). RESIDUAL PER STEP AS A COLUMN, raw AND gauge, because")
    print("    the raw one is partly the gauge drifting and says the opposite.")
    tr_ni = run_flow(W, "ni", N_STEPS, gauge=False)
    _print_trace("NI Eq. 10", tr_ni)
    tr_sp = run_flow(W, "spec", N_STEPS, eta=ETA, kappa_target=0.0, gauge=False)
    _print_trace("SPEC eta=%g target=0" % ETA, tr_sp)
    same = float(np.abs(tr_ni["W"] - tr_sp["W"]).max())
    shorts = sum(tr_ni["shortcuts"])
    uni20 = np.asarray(cv.curvature_from_graph(tr_ni["W"],
                                               measure=cv.UNIFORM)["kappa"])
    wp20 = np.asarray(cv.curvature_from_graph(
        tr_ni["W"], measure=cv.WEIGHT_PROPORTIONAL)["kappa"])
    print("    THE MEASURE GAP AT STEP %d, promised in the preamble: max|uniform -"
          % N_STEPS)
    print("    weight-proportional| = %.3e over %d edges, against a uniform kappa"
          % (float(np.abs(uni20 - wp20).max()), uni20.size))
    rng20 = float(uni20.max() - uni20.min())
    print("    range of %.3e on the same graph -- the disagreement is %.1fx the"
          % (rng20, float(np.abs(uni20 - wp20).max()) / rng20))
    print("    whole spectrum. The flowed graph is fully weighted, so the two")
    print("    measures are no longer the same measure and nothing here transfers.")
    assert float(np.abs(uni20 - wp20).max()) > 1e-3, \
        ("the two measures now agree on the FLOWED graph too: the measure caveat "
         "on every step >= 1 number must be re-measured")
    print("    THE TWO TRAJECTORIES: max|W_ni - W_spec| after %d steps = %.3e,"
          % (N_STEPS, same))
    print("    and the shortcut census was 0 at every step (%d shortcut-edge-steps"
          % shorts)
    print("    in total), which is exactly why. BITWISE identical: %s"
          % (tr_ni["W"].tobytes() == tr_sp["W"].tobytes()))
    assert shorts == 0 and same == 0.0, \
        ("the bed left the shortcut-free regime (%d shortcut-edge-steps, gap %.3e): "
         "the recorded identity must be re-measured" % (shorts, same))

    print("(d) THE PHASE, and the fixed point, on the small bed (%d + %d nodes,"
          % (SMALL_N, SMALL_N))
    print("    p = %g) where a run long enough to REACH one is affordable." % SMALL_P)
    print("    DERIVED: in the ray quotient a fixed point is F(W) = cW. For Eq. 10")
    print("    on a shortcut-free graph that is kappa_e = 1 - c on EVERY edge; for")
    print("    the spec it is kappa_e = (1 - c)/eta on every edge. Both name the")
    print("    CONSTANT-CURVATURE graphs. Neither names kappa = 0 unless c = 1.")
    Wsm, lab_sm = cv.two_block_bed(seed=0, n_per_block=SMALL_N, p=SMALL_P,
                                   n_cross=SMALL_CROSS)
    long_ni = run_flow(gauge_normalise(Wsm), "ni", LONG_STEPS, gauge=True,
                       tol=LONG_TOL)
    _print_trace("NI Eq. 10, long run", long_ni, stride=10)
    raw = long_ni["residual_abs"]
    on3, dr3 = plateau_onset(raw, rel=1e-3)
    on2, dr2 = plateau_onset(raw, rel=1e-2)
    att = first_attained(raw)
    print("    THE PLATEAU, ONSET MEASURED not asserted. The raw residual's last")
    print("    value is %.4e. It is within 1%% of that from step %s onward (drift"
          % (raw[-1], on2))
    print("    %.3f%%), within 0.1%% from step %s (drift %.3f%%), and first PRINTS"
          % (100 * dr2, on3, 100 * dr3))
    print("    as %.4e at step %s. Step 21 reads %.4e, which is %.2f%% above the"
          % (raw[-1], att, raw[20], 100 * (raw[20] / raw[-1] - 1.0)))
    print("    floor -- naming step 21 as the onset would be wrong by %d steps."
          % (att - 21))
    print("    THE ONSET IS BOUNDED ON BOTH SIDES: recorded %d, tolerance %d steps,"
          % (ONSET_STEP, ONSET_TOL))
    print("    measured %d, |measured - recorded| = %d. An assert reading"
          % (att, abs(att - ONSET_STEP)))
    print("    'onset > 21' is a FLOOR, not a guard -- it fires when the onset")
    print("    moves earlier and never when it moves later; forced to 90 it passes.")
    assert on2 is not None and att is not None, "the plateau onset is not defined"
    assert abs(att - ONSET_STEP) <= ONSET_TOL, \
        ("the raw plateau onset measured %d against a recorded %d, tolerance %d "
         "steps: the docstring's onset must be re-measured"
         % (att, ONSET_STEP, ONSET_TOL))
    sp = spectrum(long_ni["W"])
    print("    THE PHASE at the fixed point: kappa min %+.9f  med %+.9f  max %+.9f"
          % (sp["min"], sp["med"], sp["max"]))
    print("    spread %.3e over EDGES %d" % (sp["spread"], sp["n_edges"]))
    print("    histogram counts %s" % (sp["counts"],))
    print("    histogram edges  %s" % (["%+.9f" % e for e in sp["edges_hist"]],))
    assert sp["spread"] < 1e-6, \
        ("the fixed point is not constant-curvature: spread %.3e over %d edges -- "
         "the derivation has stopped reproducing" % (sp["spread"], sp["n_edges"]))
    assert abs(sp["med"]) > 1e-6, \
        "the constant is zero after all: the 'not flat' half of the claim is wrong"
    print("    FIRED: CONSTANT curvature %+.9f, not zero, spread %.1e over %d edges."
          % (sp["med"], sp["spread"], sp["n_edges"]))

    print("(e) DO THE TWO RULES SHARE FIXED POINTS? Measured, same bed, same gauge,")
    print("    same tolerance %.0e, every arm run to its own convergence." % LONG_TOL)
    print("    %-22s %6s %12s %16s %11s %7s"
          % ("arm", "steps", "resid_gauge", "kappa constant", "spread", "ARI"))
    arms = [("ni Eq. 10", "ni", None)]
    arms += [("spec eta=%g" % e, "spec", e) for e in ETA_SWEEP]
    arms += [("recip eta=2 (repair)", "recip", 2.0)]
    fixed, steps, fp, res_g = {}, {}, {}, {}
    for nm, rule, e in arms:
        # the ni arm is the run (d) already made on this bed, reused rather than
        # repeated: 81 more curvature passes buy nothing and cost 11 s.
        tr = (long_ni if rule == "ni" else
              run_flow(gauge_normalise(Wsm), rule, LONG_STEPS,
                       eta=(ETA if e is None else e), gauge=True, tol=LONG_TOL))
        if tr["refusal"] is not None:
            print("    %-22s %6d %12s %16s %11s %7s"
                  % (nm, tr["n_steps"], "REFUSED", "--", "--", "--"))
            print("        %s" % tr["refusal"].reason)
            continue
        s = spectrum(tr["W"])
        fixed[nm], steps[nm], fp[nm] = s, tr["n_steps"], tr["W"]
        res_g[nm] = tr["residual_gauge"][-1]
        print("    %-22s %6d %12.3e %+16.9f %11.3e %7.4f"
              % (nm, tr["n_steps"], tr["residual_gauge"][-1], s["med"],
                 s["spread"], best_cut_ari(tr["W"], lab_sm)["ari"]))
    consts = [s["med"] for s in fixed.values()]
    print("    SHARED? spread of the constant across %d converged arms: %.3e"
          % (len(consts), max(consts) - min(consts)))
    print("    -> the rules SHARE their fixed-point set; eta moves the RATE")
    print("    (%d steps at eta=2 against %d at eta=1, %d at eta=0.5) and nothing"
          % (steps.get("spec eta=2", -1), steps.get("spec eta=1", -1),
             steps.get("spec eta=0.5", -1)))
    print("    else. The sign flip, which converges to nothing at all, is measured")
    print("    in (k) on the cheap bed rather than here.")
    assert max(consts) - min(consts) < 1e-6, \
        "the arms land on different constants: %r" % consts

    print("(n) DOES 'CONVERGED' MEAN THE GEOMETRY IS RIGHT, OR ONLY THE CURVATURE?")
    print("    Every arm in (e) reached the SAME kappa constant to nine digits. The")
    print("    other half of the question is whether they reached the same WEIGHTS.")
    print("    w* is Eq. 10's OWN fixed point -- the only non-strawman target, since")
    print("    a flow has no reason to return to a point that is not fixed. Both")
    print("    sides are gauge-normalised before comparing, because kappa is")
    print("    scale-invariant and an unnormalised comparison measures the gauge.")
    w_star = fp["ni Eq. 10"]
    print("    %-22s %6s %12s %13s %11s %6s %11s"
          % ("arm", "steps", "resid_gauge", "max|w/w*-1|", "worst edge",
             "short", "|1 ni step|"))
    werr = {}
    for nm in [a[0] for a in arms if a[0] in fp]:
        e, edge = weight_error(fp[nm], w_star)
        sc, ne = shortcut_census(fp[nm])
        mv = float(np.abs(gauge_normalise(ni_flow_step(fp[nm])["W"])
                          - gauge_normalise(fp[nm])).max())
        werr[nm] = (e, sc)
        print("    %-22s %6d %12.3e %13.4e %11s %2d/%-3d %11.3e"
              % (nm, steps[nm], res_g[nm], e, "%d-%d" % edge, sc, ne, mv))
    print("    THE DISCRIMINATOR IS THE SHORTCUT CENSUS, the same quantity (b) and")
    print("    (c) turn on. Every arm whose fixed point is shortcut-free lands on")
    print("    Eq. 10's weights; the one that carries shortcuts does not, and is")
    print("    not an Eq. 10 fixed point at all -- one Eq. 10 step moves it.")
    clean = [nm for nm, (e, sc) in werr.items() if sc == 0]
    dirty = [nm for nm, (e, sc) in werr.items() if sc > 0]
    print("    shortcut-free arms %r: worst max|w/w*-1| = %.4e"
          % (clean, max(werr[nm][0] for nm in clean)))
    print("    shortcut-carrying  %r: worst max|w/w*-1| = %.4e"
          % (dirty, max(werr[nm][0] for nm in dirty) if dirty else float("nan")))
    assert dirty, ("no arm reached a shortcut-carrying fixed point on this bed, so "
                   "the discriminator has nothing to discriminate: re-measure")
    assert max(werr[nm][0] for nm in clean) < 1e-5,         "a shortcut-free arm missed Eq. 10's weights: %r" % werr
    assert min(werr[nm][0] for nm in dirty) > 1e-3,         "the shortcut-carrying arm now recovers the weights too: %r" % werr
    worst_nm = max(dirty, key=lambda nm: werr[nm][0])
    print("    IN PLAIN TERMS: %s reports a gauge residual of %.3e -- ten digits of"
          % (worst_nm, res_g[worst_nm]))
    print("    'converged' -- and the same kappa constant as Eq. 10 to nine digits,")
    print("    while being %.1f%% wrong in geometry."
          % (100 * werr[worst_nm][0]))
    print("    SO: kappa CONVERGED IS NOT GEOMETRY CONVERGED. It is geometry")
    print("    converged exactly when the shortcut census is zero, and that census")
    print("    is already in every run_flow trace -- it costs one shortest-path.")

    print("    PLANT AND RECOVER, the direct form: perturb Eq. 10's own fixed point")
    print("    by lognormal(sigma) and re-run Eq. 10 from there.")
    rec = recovery_table(rules=(("ni", None),), n_steps=LONG_STEPS, tol=LONG_TOL,
                         sigma=0.30, seed=3, W=Wsm, w_star=w_star)
    print("    perturbation planted: max|w/w*-1| = %.4e on edge %d-%d over %d edges"
          % (rec["start_err"], rec["start_edge"][0], rec["start_edge"][1],
             rec["n_edges"]))
    for r in rec["rows"]:
        print("    recovered by %-12s %3d steps, resid %.3e, max|w/w*-1| %.4e"
              % (r["name"], r["steps"], r["resid"], r["w_err"]))
        assert r["resid"] < LONG_TOL and r["w_err"] < 1e-6,             ("Eq. 10 did not return to its own fixed point: resid %.3e, weight "
             "error %.4e" % (r["resid"], r["w_err"]))
    print("    FIRED: a %.2f perturbation is undone to %.1e -- Eq. 10's fixed point"
          % (rec["start_err"], rec["rows"][0]["w_err"]))
    print("    is ATTRACTING, not merely stationary.")

    print("    WHICH MODE CARRIES THE ERROR, and would a second normalisation kill")
    print("    it? The log-coordinate Jacobian of the gauge-normalised Eq. 10 step,")
    print("    central differences at eps = %g, on a %d + %d bed."
          % (JAC_EPS, JAC_N, JAC_N))
    Wj, _ = cv.two_block_bed(seed=0, n_per_block=JAC_N, p=JAC_P, n_cross=2)
    fpj = run_flow(gauge_normalise(Wj), "ni", LONG_STEPS, gauge=True, tol=LONG_TOL)
    J, jedges = log_jacobian(fpj["W"], rule="ni")
    fm = flat_modes(J)
    print("    GATE first: J @ 1 = %.3e (must be ~0, or the Jacobian is noise and"
          % fm["gauge_residual"])
    print("    nothing below it means anything). %d edges, %d steps to the point."
          % (len(jedges), fpj["n_steps"]))
    print("    spectral radius %.6f, second %.6f, modes on the unit circle %d,"
          % (fm["spectral_radius"], fm["second"], fm["n_unit"]))
    print("    modes at zero %d. Top five |lambda|: %s"
          % (fm["n_zero"], ["%.6f" % v for v in fm["magnitudes"][:5]]))
    assert fm["gauge_residual"] < 5e-2,         "the Jacobian gate failed: J @ 1 = %.3e" % fm["gauge_residual"]
    assert fm["n_unit"] == 0 and fm["spectral_radius"] < 1.0,         ("a mode on the unit circle appeared (%d, rho %.6f): the fixed point is no "
         "longer isolated modulo gauge and the mechanism must be re-measured"
         % (fm["n_unit"], fm["spectral_radius"]))
    print("    SO THE ANSWER IS NO, a second normalisation would not help. There is")
    print("    no flat direction to normalise away: Eq. 10's fixed point is")
    print("    ISOLATED modulo gauge and attracting at rate %.6f. The weight error"
          % fm["spectral_radius"])
    print("    comes from the OTHER rules having fixed points Eq. 10 does not --")
    print("    the spec never consults d, so it can stop on a graph whose edges are")
    print("    not geodesics. The report that closes the hole is the shortcut")
    print("    census, not another gauge.")

    print("(f) THE NI BIND, with its control. Planted two-community graph, ground")
    print("    truth known; cut edges above a weight threshold, threshold swept")
    print("    IDENTICALLY in every arm, components scored by ARI.")
    ctl = best_cut_ari(W, labels)
    bni = best_cut_ari(tr_ni["W"], labels)
    bsp = best_cut_ari(tr_sp["W"], labels)
    print("    %-26s %8s %11s %6s %6s" % ("arm", "ARI", "thresh", "comps", "cut"))
    for nm, b in (("NO FLOW (control)", ctl),
                  ("NI Eq. 10, %d steps" % N_STEPS, bni),
                  ("SPEC eta=%g, %d steps" % (ETA, N_STEPS), bsp)):
        print("    %-26s %8.4f %11.3e %6d %6d"
              % (nm, b["ari"], b["thresh"], b["n_components"], b["n_cut"]))
    for nm, Wx in (("no flow", W), ("ni", tr_ni["W"]), ("spec", tr_sp["W"])):
        m = cross_edge_mass(Wx, labels)
        print("    %-8s planted cross median %.4e (%d edges) / intra median %.4e "
              "(%d edges) = %.4g" % (nm, m["cross_med"], m["n_cross"],
                                     m["intra_med"], m["n_intra"], m["ratio"]))
    assert ctl["ari"] < 0.5, "the no-flow control already separates: %.4f" % ctl["ari"]
    assert bni["ari"] > ctl["ari"] and bni["ari"] >= 0.8, \
        "Ni's separation did not reproduce: ARI %.4f against control %.4f" \
        % (bni["ari"], ctl["ari"])
    assert bsp["ari"] >= bni["ari"] - 1e-12, "the spec rule lost the separation"
    print("    THE BIND HOLDS -- by IDENTITY, not by agreement: on this bed the")
    print("    spec's zero-source rule IS Eq. 10 (bitwise, (c) above).")

    print("(g) THE SILENT FAILURE, fired on purpose: FREEZING d. Eq. 10 is defined")
    print("    against the metric of (V, E, w^(k)), so a frozen d is not the flow.")
    print("    Foreman measured the trap; here it is on this bed.")
    frz = run_flow(gauge_normalise(Wsm), "ni", 15, gauge=True, freeze_metric=True)
    live = run_flow(gauge_normalise(Wsm), "ni", 15, gauge=True)
    sf, sl = spectrum(frz["W"]), spectrum(live["W"])
    print("    frozen d : residual_gauge %.3e after %d steps, but the RECOMPUTED"
          % (frz["residual_gauge"][-1], frz["n_steps"]))
    print("               curvature spread is %.3e over %d edges"
          % (sf["spread"], sf["n_edges"]))
    print("    live   d : residual_gauge %.3e after %d steps, curvature spread"
          % (live["residual_gauge"][-1], live["n_steps"]))
    print("               %.3e over %d edges" % (sl["spread"], sl["n_edges"]))
    print("    RATIO of the two spreads: %.3g -- the frozen run is the one whose"
          % (sf["spread"] / sl["spread"]))
    print("    residual looks better and whose geometry is worse.")
    assert sf["spread"] > sl["spread"], \
        "the frozen-d trap stopped firing: it must be measurably worse geometry"

    print("(h) PLANTED NEGATIVE 1: ONE community, no plant, same n and density.")
    print("    The same cut must NOT shatter it under any rule.")
    W1 = gauge_normalise(planted_one_community())
    f0 = largest_component_fraction(W1, CUT_QUANTILE)
    neg_steps = 8
    for rule in RULES:
        out = run_flow(W1, rule, neg_steps, eta=ETA, gauge=True)
        frac = largest_component_fraction(out["W"], CUT_QUANTILE)
        print("    %-6s largest component after %d steps: %.1f%% of %d nodes  "
              "(no flow: %.1f%%)" % (rule, neg_steps, 100 * frac, W1.shape[0],
                                     100 * f0))
        assert frac >= 0.9, "%s shattered a single community: %.1f%%" \
                            % (rule, 100 * frac)
    print("    FIRED: no plant, no separation, under any of the three rules.")

    print("(i) PLANTED NEGATIVE 2: eta = 0 must be BITWISE identity.")
    z = spec_flow_step(W, eta=0.0, kappa_target=0.0)["W"]
    zr = recip_flow_step(W, eta=0.0, kappa_target=0.0)["W"]
    print("    spec  bitwise identical: %s (%d weights differ)"
          % (z.tobytes() == W.tobytes(), int((z != W).sum())))
    print("    recip bitwise identical: %s (%d weights differ)"
          % (zr.tobytes() == W.tobytes(), int((zr != W).sum())))
    assert z.tobytes() == W.tobytes() and zr.tobytes() == W.tobytes(), \
        "eta = 0 moved a weight"

    print("(j) PLANTED NEGATIVE 3: a step at target is a no-op to 1e-12.")
    got = _curvature_pass(W)
    tgt = {e: k for e, k in zip(got[0]["edges"], got[0]["kappa"])}
    n0 = spec_flow_step(W, eta=0.5, kappa_target=tgt)["W"]
    n1 = recip_flow_step(W, eta=0.5, kappa_target=tgt)["W"]
    print("    spec  max|w' - w| at kappa_target = measured kappa: %.3e"
          % np.abs(n0 - W).max())
    print("    recip max|w' - w| at kappa_target = measured kappa: %.3e"
          % np.abs(n1 - W).max())
    assert np.abs(n0 - W).max() < 1e-12 and np.abs(n1 - W).max() < 1e-12, \
        "an at-target step moved a weight"

    print("(k) REFUSALS ARE VALUES. THREE causes, not four, and the missing one is")
    print("    a finding: curvature.py's own refusal codes are UNREACHABLE from a")
    print("    flow step. A step visits EDGES only, and both endpoints of an edge")
    print("    have degree >= 1 inside one component, so self-loop,")
    print("    degenerate-neighbourhood, zero-distance and unreachable-support")
    print("    cannot arise. Two disjoint triangles therefore refuse NOTHING -- and")
    print("    must still come back finite, which is the contract that survives.")
    Wd = np.zeros((6, 6))
    for i, j in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]:
        Wd[i, j] = Wd[j, i] = 1.0
    od = ni_flow_step(Wd)
    print("    disjoint triangles : edges %d refused %d %r  finite %s"
          % (od["n_edges"], od["n_refused"], od["refusals"],
             bool(np.isfinite(od["W"]).all())))
    kap = np.asarray(_curvature_pass(W)[0]["kappa"])
    ceil = eta_ceiling(kap)
    over = spec_flow_step(W, eta=2.0 * ceil, kappa_target=0.0)
    print("    eta ceiling = 1/max kappa = %.4g; at eta = %.4g the rescale refuses"
          % (ceil, 2.0 * ceil))
    print("    %d edges (%r), min weight %.3e, negatives %d"
          % (over["n_refused"], over["refusals"],
             over["W"][over["W"] > 0].min(), int((over["W"] < 0).sum())))
    bad = run_flow(W, "not-a-rule")
    print("    unknown rule       : %r" % (bad,))
    Wb, _ = cv.two_block_bed(seed=0, n_per_block=JAC_N, p=JAC_P, n_cross=2)
    Wb = gauge_normalise(Wb)
    slow = run_flow(Wb, "spec", 250, eta=-2.0, gauge=True)
    up = slow["W"][np.triu_indices_from(slow["W"], 1)]
    up = up[up > 0]
    print("    sign flip on the LENGTH variable, %d + %d bed. At eta = -2 it does"
          % (JAC_N, JAC_N))
    print("    NOT refuse inside %d steps -- it just diverges, weight span %.3e --"
          % (slow["n_steps"], float(up.max() / up.min())))
    print("    so divergence and refusal are not the same event and the bed decides")
    print("    which one is seen. At eta = -4 the same bed trips the LP:")
    blown = run_flow(Wb, "spec", 250, eta=-4.0, gauge=True)
    assert slow["refusal"] is None,         "eta = -2 now refuses on this bed too: the pair must be re-measured"
    assert blown["refusal"] is not None,         ("the sign-flipped arm survived %d steps at eta = -4 on the %d + %d bed: "
         "the recorded blow-up must be re-measured" % (blown["n_steps"], JAC_N, JAC_N))
    print("    sign-flipped arm at eta = -4: ran %d" % blown["n_steps"])
    print("    steps, then %s -- an exception inside the transport LP arriving as a"
          % blown["refusal"].code)
    print("    VALUE, because it is a fact about the flow, not a malformed input.")
    assert np.isfinite(over["W"]).all() and (over["W"] >= 0).all()
    assert over["refusals"].get(NON_POSITIVE_WEIGHT, 0) > 0, "the eta ceiling did not fire"
    assert is_refusal(bad) and not isinstance(bad, dict)
    assert blown["refusal"] is not None, \
        "the sign-flipped arm no longer blows up: the recorded failure must be re-measured"
    causes = set(od["refusals"]) | set(over["refusals"]) | {bad.code,
                                                            blown["refusal"].code}
    print("    distinct causes seen: %s" % sorted(causes))
    assert od["n_refused"] == 0, \
        ("a disjoint graph now refuses %r: curvature.py's refusals have become "
         "reachable from a flow step and the census must be re-measured"
         % (od["refusals"],))
    assert causes == {NON_POSITIVE_WEIGHT, LP_FAILED, UNKNOWN_RULE}, \
        "the reachable refusal causes changed: %r" % sorted(causes)

    print("(l) ARI AGAINST ITS OWN CORNERS, since every number in (f) is one.")
    lab = np.array([0] * 10 + [1] * 10)
    print("    perfect %.4f   all-one-cluster %.4f   all-singletons %.4f"
          % (adjusted_rand_index(lab, lab),
             adjusted_rand_index(lab, np.zeros(20, int)),
             adjusted_rand_index(lab, np.arange(20))))
    assert adjusted_rand_index(lab, lab) == 1.0
    assert abs(adjusted_rand_index(lab, np.zeros(20, int))) < 1e-12
    assert abs(adjusted_rand_index(lab, np.arange(20))) < 1e-12

    elapsed = time.time() - t0
    print("(m) BUDGET. elapsed %.1f s on CPU (bar: 180 s)." % elapsed)
    assert elapsed < 180.0, "demo exceeded its 180 s CPU budget: %.1f s" % elapsed
    print("ALL SELF-CHECKS PASSED")


if __name__ == "__main__":
    demo()

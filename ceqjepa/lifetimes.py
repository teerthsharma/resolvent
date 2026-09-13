"""A1: every created simplex declares a lifetime, and the AXIS is part of it.

WHAT IS OCCUPIED, FIRST, BECAUSE IT DECIDES WHAT IS LEFT. The lifetime of an
individual simplex pair is Edelsbrunner-Letscher-Zomorodian 2002: a positive
simplex creates a class and a negative one merges it, and the pair carries the
life-time. What they spend it on is SIMPLIFICATION -- a change is feature or
noise by its life-time, and the short ones are removed. Barcode-as-survival-data
is Murris, Stolz and Borgwardt, arXiv 2606.11911, 10 June 2026: persistence
p = d - b as a time-to-event, S(t) = P(p > t) as the object that compares
diagrams, estimated by Kaplan-Meier with Nelson-Aalen for the cumulative hazard.
That is exactly the estimator this file uses and it is three months old; it is
cited, not claimed.

WHAT IS CLAIMED IS THE READ-SIDE GATE AND THE CENSORED CLASS, and nothing wider.
ELZ consult a lifetime to REMOVE a simplex; here a lifetime GATES a query --
T-DEAD, demo() (f) -- so a read that consults a simplex outside its own (b, d)
returns EXPIRED as a value a scorer can count, and never an answer. And the
survival above is over features that are "fully observed", which forecloses the
essential classes; here an essential class is d = infinity, enters the estimator
RIGHT-CENSORED at t_max, and is never a large finite number -- written as t_max
it silently becomes the longest FINITE bar in every aggregate that follows, and
nothing downstream can then recover which bars it was. The consequence is
visible in the estimate: with censored mass present S does not reach zero, and
demo() (e) prints S at t_max beside a FINITE cumulative hazard there.

THE OWNER'S KILL, MADE MECHANICAL. A lifetime without its axis is not a lifetime,
so Lifetime(b, d) with no axis raises and two lifetimes on different axes refuse
to be ORDERED -- equality across axes is False rather than an error, because
containers call __eq__ and a raising __eq__ breaks `in`. Construction refuses by
EXCEPTION and every read refuses by VALUE, and those are different failures: a
malformed lifetime must not exist, while a dead read is a fact a scorer counts.

THE BARCODES ARE COMPUTED, AND TWO OF THE ORACLES ARE INDEPENDENT OF THE CODE
(demo() (b), (c)). The unit square under Rips: H0 three finite bars and one
essential, H1 the single bar (1, 1.414213562373095), worst endpoint gap against
inspection 0.000e+00. A 12-ring growing edge by edge and coned off: H0 eleven
finite and one essential over 49 simplices, H1 the single bar (12, 20) on the
TIME axis, gap 0.000e+00 -- these filtration values are TIMES, so no point cloud
produces them and the axis is doing work. A two-well cubical field on the
TEMPERATURE axis merges its two components at level 1.0000 against the analytic
saddle 1.0, a gap of 0.0000 at grid step 0.166667. Then the same barcode from two
ENGINES over the same 60 points at noise 0.05: gudhi 3.12.0 against ripser
0.6.14, worst endpoint disagreement 1.216e-08 in H0 and 8.791e-09 in H1, which is
ripser's float32 accumulation against gudhi's float64 and not a difference of
definition.

PER AXIS, EVERY COUNT BESIDE THE SIMPLICES IT CAME FROM (demo() (d)):

    time         step 1         49 simplices     H0 11 finite + 1 essential
    scale        step 1e-12     36050 simplices  H0 59 finite + 1 essential
    temperature  step 0.166667  2601 cells       H0 1 finite + 1 essential

SURVIVAL, WITH THE ESSENTIAL CLASSES RIGHT-CENSORED (demo() (e)). The estimator
is KAPLAN-MEIER. Essential classes enter as right-censored observations of length
t_max - b; they are not dropped. On the scale bed's 61 observations the
product-limit and exp(-cumulative hazard) differ by at most 0.008423. The
identity S = exp(-integral h) is run on a planted exponential whose hazard is the
known constant 1.3: at n = 60 the estimator gap is 0.009280 and the quadrature
0.017056, rate recovered 1.2113; at n = 600 they are 0.000935 and 0.012834, rate
1.2209. BOTH fall with n and the rate is what separates them: the estimator
gap falls 89.92% over that decade while the quadrature falls only 24.76%,
because the estimator gap is order 1/n and the quadrature is floored by the bin
width, which no n removes. The product-limit and exp(-cumulative hazard) are the
discrete pair, not exact inverses, so nothing here asserts they are equal.

THE COST OF DROPPING THEM, measured because the owner asked for it either way:
over 61 bars of which 1 is essential the restricted mean lifetime is 0.180260
censored and 0.141598 with the essential class deleted -- a downward bias of
0.038662 in rips-radius units. The sign is not an accident, since deleting the
longest observations in a sample can only lower its mean.

T-DEAD ON THE 3x2 (demo() (f)). 366 cases built from the scale barcode's own
intervals: 122 that must answer, 244 that must refuse. The interval criterion is
exact -- sensitivity 100.00%, specificity 100.00%. Beside it, refuse-everything
scores sensitivity 100.00% at specificity 0.00% and answer-everything the mirror
of that. No pooled agreement rate is computed anywhere in this file, because on a
bed whose class balance is chosen freely a pooled rate ranks one of those two
first.

THE MARS ITEM, MEASURED RATHER THAN HIDDEN (demo() (h)). Lifetimes ARE gamed by
the filtration step. The same points at the same threshold, with the pairwise
distances quantised onto a grid, move the H1 endpoints by 0.0139 at step 0.02,
0.0339 at step 0.05, 0.0618 at step 0.1 and 0.1339 at step 0.25. A lifetime
quoted without its step is therefore not reproducible, which is why Axis carries
one and why every number above is printed beside its step.

WHAT IS NOT CLAIMED. (i) The hazard identity is an analytic tautology in
continuous time; the two numbers reported for it measure the ESTIMATOR gap
between two forms of the same estimate and the ARITHMETIC of re-integrating a
binned hazard, not the truth of any model. (ii) The T-DEAD 3x2 is exact because
its oracle IS the interval. That demonstrates the refusal is decidable and
countable, not that the intervals are the right ones -- an oracle independent of
the intervals would be a different experiment and is not run here. (iii) The
three axes are three different filtrations on three different objects; nothing
claims a bar on one is comparable to a bar on another, and the type refuses
precisely that. (iv) The survival function is over ONE barcode of ONE bed and is
not a distribution over lifetimes in general. (v) Nothing here is a claim about
prediction. (vi) This file is not compared against the repo's own barcode
producer, and could not be: ceq/certs/topological.py:401 returns
`d[np.isfinite(d[:, 1])]` from dgms[1] alone, so H0 never reaches a caller and
every non-finite death is filtered out. It is a censored-data-destroying
baseline and a comparison against it would measure the filter.

THE PIN THAT IS OWED. requirements.txt pins ripser at 0.6.14 and does not
mention gudhi at all, while the repo's policy is that every function-local TDA
import is pinned. This file does not own requirements.txt, so instead of leaving
a bare ImportError to surface somewhere downstream, _gudhi() names the debt.

RUN: python -m ceqjepa.lifetimes
"""

import math
import time
from collections import namedtuple
from functools import total_ordering

import numpy as np

from ceqjepa.curvature import Refusal, is_refusal

__all__ = [
    "Axis", "AXES", "TIME", "SCALE", "TEMPERATURE", "AxisMismatch", "Lifetime",
    "Barcode", "persistence_barcode", "rips_barcode", "ripser_barcode",
    "square_bed", "ring_bed", "scale_bed", "scale_barcode", "temperature_bed",
    "SCALE_THRESH", "STEP_SWEEP", "SQUARE_H1_BY_INSPECTION", "RING_H1_BY_INSPECTION",
    "EXPIRED", "AXIS_MISMATCH", "NULL_QUERY", "read_at",
    "DEFINED", "UNDEFINED", "NULL", "t_dead_cases", "score_3x2",
    "read_criterion", "refuse_everything", "answer_everything",
    "KM", "km", "s_at", "km_mean", "censoring_bias", "survival_report",
    "identity_check", "Refusal", "is_refusal",
]

#: Rips threshold on the scale bed. Every scale number below is at this
#: threshold; a bar quoted at another one says so beside itself.
SCALE_THRESH = 2.5

#: Points on the noisy-circle bed, and the noise that makes it a bed rather
#: than a circle.
SCALE_N = 60
SCALE_NOISE = 0.05
SCALE_SEED = 0

#: The cubical grid for the temperature bed: two wells at x = -1 and x = +1 on
#: a square of side 4, so the analytic saddle sits at the origin at level 1.
TEMP_GRID = 25
TEMP_HALF = 2.0

#: The MARS sweep: filtration steps the barcode is recomputed at. The point of
#: the sweep is that the lifetimes MOVE, and by how much.
STEP_SWEEP = (0.02, 0.05, 0.1, 0.25)

#: The unit square's H1 bar, by inspection: born when the four sides enter at
#: length 1, filled when the first diagonal enters at sqrt(2). tests/curvature/
#: test_lifetimes.py carries its own independent copy of this.
SQUARE_H1_BY_INSPECTION = (1.0, math.sqrt(2.0))

#: The growing ring's H1 bar, by inspection: the twelfth edge closes the loop at
#: t = 12, the cone fills it at t = 20.
RING_N = 12
RING_H1_BY_INSPECTION = (12.0, 20.0)


# ---------------------------------------------------------------------------
# THE AXIS, AND THE LIFETIME THAT CANNOT EXIST WITHOUT ONE
# ---------------------------------------------------------------------------

class Axis(namedtuple("Axis", "name unit step")):
    """A filtration axis: what the parameter means, and how finely it resolves.

    `step` is the resolution at which this axis's filtration values are
    reported. It is not decoration: the barcode is a function of the step (see
    demo() (h)), so a lifetime quoted without one cannot be reproduced.
    """

    __slots__ = ()

    def __repr__(self):
        return "Axis(%s in %s, step %g)" % (self.name, self.unit, self.step)


#: Insertion time. The ring bed inserts at integer times, so the step is 1.
TIME = Axis("time", "insertion step", 1.0)

#: Vietoris-Rips radius. Exact pairwise distances resolve to working precision;
#: STEP_SWEEP coarsens this deliberately and demo() (h) measures what that costs.
SCALE = Axis("scale", "rips radius", 1e-12)

#: Sublevel of a scalar field. The step is the grid spacing of the cubical bed.
TEMPERATURE = Axis("temperature", "level", 2 * TEMP_HALF / (TEMP_GRID - 1))

AXES = (TIME, SCALE, TEMPERATURE)


class AxisMismatch(TypeError):
    """Two lifetimes from different filtrations were compared.

    A TypeError subclass because that is what Python raises for an order that
    does not exist, and because a comparison must not quietly return a bool.
    """


@total_ordering
class Lifetime:
    """sigma -> (b, d) on a NAMED axis. The owner's kill lives in __init__.

    An essential class is death = infinity, never the filtration maximum and
    never a large float: written as t_max it becomes the longest FINITE bar in
    every aggregate downstream, and nothing after that point can recover which
    bars were essential.

    Construction refuses by EXCEPTION rather than by value, unlike every read in
    this module. A malformed lifetime must not exist: a Refusal returned from a
    constructor would have to be stored, and every consumer downstream would
    then have to re-check what it was handed. Refusals-as-values are for reads,
    where a scorer can count them.
    """

    __slots__ = ("birth", "death", "axis", "label")

    def __init__(self, birth, death, axis, label=""):
        if not isinstance(axis, Axis):
            raise TypeError(
                "a lifetime without its axis is not a lifetime: axis=%r" % (axis,))
        b, d = float(birth), float(death)
        if math.isnan(b) or math.isnan(d):
            raise ValueError("NaN endpoint: (%r, %r)" % (birth, death))
        if not math.isfinite(b):
            raise ValueError(
                "a birth at infinity is not a birth: (%r, %r) would give a NaN "
                "duration and enter the survival estimator silently" % (b, d))
        if d < b:
            raise ValueError("death before birth: (%r, %r)" % (b, d))
        self.birth, self.death, self.axis, self.label = b, d, axis, label

    @property
    def is_essential(self):
        """True iff the class never dies on this axis -- d = infinity."""
        return math.isinf(self.death)

    @property
    def duration(self):
        """d - b. Infinite for an essential class, by construction."""
        return self.death - self.birth

    def censored_duration(self, t_max):
        """The honest floor for an essential class: it lasted AT LEAST this long.

        This is the right-censored observation the survival estimator consumes.
        Dropping the class instead removes the longest observations from the
        sample and biases every lifetime downward -- measured in demo() (e).
        """
        return min(self.death, float(t_max)) - self.birth

    def contains(self, tau):
        """Half-open [b, d): a class is not evidence at the instant it dies."""
        return self.birth <= tau < self.death

    def __eq__(self, other):
        if not isinstance(other, Lifetime):
            return NotImplemented
        return (self.axis is other.axis
                and self.birth == other.birth and self.death == other.death)

    def __hash__(self):
        return hash((self.axis, self.birth, self.death))

    def __lt__(self, other):
        """Barcode order, and only within one axis.

        A scale bar is not shorter than a time bar; the comparison is a category
        error that would otherwise return a perfectly ordinary bool, which is
        why it raises. Equality across axes returns False rather than raising,
        because containers call __eq__ and a raising __eq__ breaks `in`.
        """
        if not isinstance(other, Lifetime):
            return NotImplemented
        if self.axis is not other.axis:
            raise AxisMismatch(
                "a lifetime on %r cannot be ordered against one on %r"
                % (self.axis.name, other.axis.name))
        return (self.birth, self.death) < (other.birth, other.death)

    def __repr__(self):
        d = "inf" if self.is_essential else "%.6g" % self.death
        tag = " %s" % self.label if self.label else ""
        return "Lifetime(%.6g, %s, axis=%s, step=%g%s)" % (
            self.birth, d, self.axis.name, self.axis.step, tag)


# ---------------------------------------------------------------------------
# BARCODES, COMPUTED PER AXIS
# ---------------------------------------------------------------------------

class Barcode(namedtuple("Barcode", "axis t_max n_simplices bars label")):
    """Every bar this filtration produced, each one a Lifetime on `axis`.

    `n_simplices` travels with the bars because a count of classes means nothing
    without the count of simplices it came from, and `t_max` because that is
    where the essential classes are right-censored.
    """

    __slots__ = ()

    def in_dimension(self, dim):
        """The bars of homological degree `dim`, in barcode order."""
        return sorted(b for d, b in self.bars if d == dim)

    def counts(self):
        """{dim: (n_finite, n_essential)} -- the per-axis report line."""
        out = {}
        for dim, bar in self.bars:
            f, e = out.get(dim, (0, 0))
            out[dim] = (f + (0 if bar.is_essential else 1),
                        e + (1 if bar.is_essential else 0))
        return out

    @property
    def n_finite(self):
        return sum(1 for _, b in self.bars if not b.is_essential)

    @property
    def n_essential(self):
        return sum(1 for _, b in self.bars if b.is_essential)


def _gudhi():
    """gudhi, or a named failure that says which pin is owed.

    requirements.txt pins ripser at 0.6.14 and does not mention gudhi, while the
    repo's policy is that every function-local TDA import is pinned. This file
    does not own requirements.txt, so the debt is named here rather than left to
    surface as a bare ImportError from whichever caller reached it first.
    """
    try:
        import gudhi
    except Exception as exc:                    # noqa: BLE001 -- named, not swallowed
        raise ImportError(
            "ceqjepa.lifetimes needs gudhi (every number in it was measured "
            "against gudhi 3.12.0) and it did not import: %r. gudhi is ABSENT "
            "from requirements.txt, which pins ripser at 0.6.14; that pin is "
            "owed and this file does not own the file that carries it."
            % (exc,)) from exc
    return gudhi


def _ripser():
    """ripser, or a named failure. Pinned at 0.6.14 in requirements.txt."""
    try:
        from ripser import ripser
    except Exception as exc:                    # noqa: BLE001 -- named, not swallowed
        raise ImportError(
            "ceqjepa.lifetimes needs ripser (pinned 0.6.14) for the two-engine "
            "cross-check and it did not import: %r" % (exc,)) from exc
    return ripser


def persistence_barcode(cx, axis, max_dim=1, t_max=None, label=""):
    """Bars out of any gudhi complex that computes persistence.

    Takes a SimplexTree or a CubicalComplex -- both expose compute_persistence,
    persistence_intervals_in_dimension and num_simplices -- so the three axes
    below run through one path and a bar cannot pick up its axis by accident.
    """
    cx.compute_persistence()
    bars, seen_max = [], 0.0
    for dim in range(max_dim + 1):
        for b, d in cx.persistence_intervals_in_dimension(dim):
            bars.append((dim, Lifetime(b, d, axis)))
            seen_max = max(seen_max, float(b))
            if math.isfinite(d):
                seen_max = max(seen_max, float(d))
    return Barcode(axis, float(seen_max if t_max is None else t_max),
                   int(cx.num_simplices()), tuple(bars), label)


def rips_barcode(X, thresh, axis, step=None, max_dim=1):
    """Vietoris-Rips on `X`, optionally with the filtration COARSENED to `step`.

    Coarsening quantises the pairwise distances down onto a grid of width step
    before the complex is built, which is exactly how a filtration step gets
    gamed: the bars move, and demo() (h) measures by how much.
    """
    gudhi = _gudhi()
    X = np.asarray(X, dtype=float)
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    if step:
        D = np.floor(D / step) * step
        np.fill_diagonal(D, 0.0)
    st = gudhi.RipsComplex(distance_matrix=D, max_edge_length=float(thresh)
                           ).create_simplex_tree(max_dimension=max_dim + 1)
    return persistence_barcode(st, axis, max_dim=max_dim, t_max=float(thresh),
                               label="rips step=%s" % (step,))


def ripser_barcode(X, thresh, axis, max_dim=1):
    """The same barcode from a different engine, for the cross-check.

    Agreement between two implementations is not proof -- both could be wrong
    the same way -- which is why square_bed() and ring_bed() carry answers known
    by inspection. This catches the other failure: a bar that is an artefact of
    one library's construction.
    """
    dgms = _ripser()(np.asarray(X, dtype=float), maxdim=max_dim,
                  thresh=float(thresh))["dgms"]
    bars = [(dim, Lifetime(b, d, axis))
            for dim in range(max_dim + 1) for b, d in dgms[dim]]
    return Barcode(axis, float(thresh), sum(len(g) for g in dgms), tuple(bars),
                   "ripser")


# ---------------------------------------------------------------------------
# THE BEDS, one per axis, plus the two whose answers are known by inspection
# ---------------------------------------------------------------------------

def square_bed():
    """The unit square under Rips. Answer known by inspection, SCALE axis."""
    X = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    return rips_barcode(X, 3.0, SCALE, max_dim=1)


def ring_bed():
    """A ring of RING_N vertices growing edge by edge, coned off. TIME axis.

    Vertices at t = 0; edge (i, i+1) at t = i+1; the closing edge at t = RING_N
    creates the loop; a hub vertex and every triangle at t = 20 fill it. Both H1
    endpoints were chosen by hand, which is what makes this a check on the axis
    rather than on Rips: these filtration values are TIMES and no point cloud
    produces them.
    """
    gudhi = _gudhi()
    st = gudhi.SimplexTree()
    for i in range(RING_N):
        st.insert([i], 0.0)
    for i in range(RING_N - 1):
        st.insert([i, i + 1], float(i + 1))
    st.insert([RING_N - 1, 0], float(RING_N))
    cone = RING_H1_BY_INSPECTION[1]
    st.insert([RING_N], cone)
    for i in range(RING_N):
        st.insert([i, RING_N], cone)
        st.insert([i, (i + 1) % RING_N, RING_N], cone)
    st.make_filtration_non_decreasing()
    return persistence_barcode(st, TIME, max_dim=1, t_max=cone, label="ring")


def scale_bed(seed=SCALE_SEED, n=SCALE_N, noise=SCALE_NOISE):
    """A noisy circle: one long H1 class over a crowd of short H0 ones."""
    rng = np.random.default_rng(seed)
    th = rng.uniform(0.0, 2.0 * math.pi, n)
    return np.c_[np.cos(th), np.sin(th)] + rng.normal(0.0, noise, (n, 2))


def scale_barcode():
    """The pinned SCALE barcode: scale_bed() under Rips at SCALE_THRESH."""
    return rips_barcode(scale_bed(), SCALE_THRESH, SCALE)


def temperature_bed():
    """Sublevel sets of a two-well field on a cubical grid. TEMPERATURE axis.

    f(x, y) = min over the two wells of the squared distance to that well, so
    the two components merge at the saddle at the origin, whose analytic level
    is 1 exactly. The measured merge sits at the grid's own resolution of that
    saddle, and demo() (b) prints the gap rather than rounding it away.
    """
    gudhi = _gudhi()
    xs = np.linspace(-TEMP_HALF, TEMP_HALF, TEMP_GRID)
    gx, gy = np.meshgrid(xs, xs)
    f = np.minimum((gx + 1.0) ** 2 + gy ** 2, (gx - 1.0) ** 2 + gy ** 2)
    cc = gudhi.CubicalComplex(top_dimensional_cells=f)
    return persistence_barcode(cc, TEMPERATURE, max_dim=1, t_max=float(f.max()),
                               label="two wells")


# ---------------------------------------------------------------------------
# T-DEAD: the read, and its refusal as a VALUE
# ---------------------------------------------------------------------------

#: The read consulted a simplex outside its own interval. A dead relation is
#: not evidence; a global one always is.
EXPIRED = "EXPIRED"

#: The query's filtration value belongs to a different axis. This is the kill
#: enforced at read time: a scale bar has no opinion about t = 8.
AXIS_MISMATCH = "AXIS_MISMATCH"

#: There is no query. NaN reaches here from an upstream refusal that was cast
#: to float somewhere, and it must not be allowed to look like a live read.
NULL_QUERY = "NULL_QUERY"


def read_at(lifetime, tau, axis, value):
    """Consult `lifetime` at filtration value `tau` on `axis`.

    Returns `value` when the simplex is alive, and a Refusal carrying its reason
    otherwise -- never an exception, never a NaN. An exception is caught by a
    caller's except clause and a NaN passes `nan < tol` silently; a Refusal is a
    value a scorer can count, and a threshold applied to one raises loudly.
    """
    if not isinstance(axis, Axis) or axis is not lifetime.axis:
        return Refusal(AXIS_MISMATCH,
                       "read on %r against a lifetime on %r"
                       % (getattr(axis, "name", axis), lifetime.axis.name))
    t = float(tau)
    if math.isnan(t):
        return Refusal(NULL_QUERY, "no filtration value on axis %r" % (axis.name,))
    if not lifetime.contains(t):
        return Refusal(EXPIRED,
                       "tau=%g is outside [%g, %s) on axis %r"
                       % (t, lifetime.birth,
                          "inf" if lifetime.is_essential else "%g" % lifetime.death,
                          axis.name))
    return value


#: The three truth classes of the 3x2. NULL is not UNDEFINED: an undefined read
#: has a well-posed question with no answer, a null read has no question.
DEFINED = "defined"
UNDEFINED = "undefined"
NULL = "null"


def t_dead_cases(barcode, seed=0, n_per_bar=2):
    """The T-DEAD bed, built from the intervals themselves.

    The oracle is the interval, so the real criterion should be exact here; what
    the bed is for is the two trivial criteria beside it. Each bar contributes
    live queries inside [b, d), dead queries outside it, and null queries that
    carry the wrong axis or no value at all.
    """
    rng = np.random.default_rng(seed)
    other = TIME if barcode.axis is not TIME else SCALE
    cases = []
    for _, bar in barcode.bars:
        hi = barcode.t_max if bar.is_essential else bar.death
        for _ in range(n_per_bar):
            if hi > bar.birth:
                cases.append((bar, float(rng.uniform(bar.birth, hi)),
                              barcode.axis, DEFINED))
            if not bar.is_essential:
                cases.append((bar, float(bar.death + rng.uniform(0.0, 1.0)),
                              barcode.axis, UNDEFINED))
            if bar.birth > 0.0:
                cases.append((bar, float(rng.uniform(0.0, bar.birth)),
                              barcode.axis, UNDEFINED))
        cases.append((bar, float(bar.birth), other, NULL))
        cases.append((bar, float("nan"), barcode.axis, NULL))
    return cases


def read_criterion(lifetime, tau, axis):
    """The real criterion: True iff the module's own read refuses."""
    return is_refusal(read_at(lifetime, tau, axis, 1.0))


def refuse_everything(lifetime, tau, axis):
    """The first trivial criterion. Perfect sensitivity, zero specificity."""
    return True


def answer_everything(lifetime, tau, axis):
    """The second trivial criterion. Zero sensitivity, perfect specificity."""
    return False


def score_3x2(cases, criterion):
    """The 3x2: defined / undefined / null against answered / refused.

    Sensitivity and specificity are returned SEPARATELY and no pooled agreement
    rate is returned at all, because a pooled rate on a bed whose class balance
    is chosen freely cannot tell the real criterion from either trivial one --
    which is exactly the pair of failures the 3x2 exists to expose.
    """
    table = {cls: {"answered": 0, "refused": 0}
             for cls in (DEFINED, UNDEFINED, NULL)}
    for lifetime, tau, axis, truth in cases:
        key = "refused" if criterion(lifetime, tau, axis) else "answered"
        table[truth][key] += 1
    should_refuse = sum(table[c]["refused"] + table[c]["answered"]
                        for c in (UNDEFINED, NULL))
    got_refused = sum(table[c]["refused"] for c in (UNDEFINED, NULL))
    live = table[DEFINED]["refused"] + table[DEFINED]["answered"]
    return {
        "table": table,
        "n": len(cases),
        "sensitivity": got_refused / max(should_refuse, 1),
        "specificity": table[DEFINED]["answered"] / max(live, 1),
        "n_should_refuse": should_refuse,
        "n_should_answer": live,
    }


# ---------------------------------------------------------------------------
# SURVIVAL AND HAZARD, with the essential classes RIGHT-CENSORED
# ---------------------------------------------------------------------------

KM = namedtuple("KM", "t S H n_risk n_event n")


def km(durations, censored=()):
    """Kaplan-Meier product-limit S, with Nelson-Aalen cumulative hazard beside it.

    `censored` are right-censored observations: known to have lasted at least
    that long. They stay in the risk set up to their own time and then leave
    without an event, which is the whole difference between this and the
    empirical survival of the uncensored durations alone.

    Returns both S = prod(1 - d_i/n_i) and H = sum(d_i/n_i) so the two forms of
    the same estimate can be differenced -- that gap is the estimator error, of
    order 1/n, and demo() (e) shows it shrinking.
    """
    obs = sorted([(float(x), 1) for x in durations]
                 + [(float(x), 0) for x in censored])
    n = len(obs)
    if n == 0:
        raise ValueError("no observations")
    t, S, H, nr, ne = [], [], [], [], []
    surv, cum, at_risk = 1.0, 0.0, n
    for tick in sorted({x for x, _ in obs}):
        d = sum(1 for x, e in obs if x == tick and e == 1)
        c = sum(1 for x, e in obs if x == tick and e == 0)
        if d:
            surv *= 1.0 - d / at_risk
            cum += d / at_risk
            t.append(tick), S.append(surv), H.append(cum)
            nr.append(at_risk), ne.append(d)
        at_risk -= d + c
    return KM(np.array(t), np.array(S), np.array(H), np.array(nr),
              np.array(ne), n)


def s_at(est, tau):
    """S(tau), right-continuous: the estimate at the last event at or before tau."""
    idx = np.searchsorted(est.t, float(tau), side="right") - 1
    return 1.0 if idx < 0 else float(est.S[idx])


def km_mean(est, t_max):
    """Restricted mean lifetime: the integral of S over [0, t_max].

    Restricted because with censoring S need never reach zero, so the unbounded
    mean is not identified. The bound is the filtration's own maximum, which is
    the only horizon the data actually covers.
    """
    edges = np.concatenate(([0.0], est.t[est.t < t_max], [float(t_max)]))
    return float(sum((edges[i + 1] - edges[i]) * s_at(est, edges[i])
                     for i in range(len(edges) - 1)))


def censoring_bias(barcode):
    """What dropping the essential classes costs, in the units of the axis.

    Censored: essential classes enter as observations of length t_max - b.
    Dropped: they are deleted. Deleting the longest observations in the sample
    can only lower the mean, so the bias is one-signed and is reported rather
    than argued about.
    """
    finite = [b.duration for _, b in barcode.bars if not b.is_essential]
    cens = [b.censored_duration(barcode.t_max)
            for _, b in barcode.bars if b.is_essential]
    kept = km(finite, cens)
    dropped = km(finite)
    return {
        "censored_mean": km_mean(kept, barcode.t_max),
        "dropped_mean": km_mean(dropped, barcode.t_max),
        "bias": km_mean(kept, barcode.t_max) - km_mean(dropped, barcode.t_max),
        "n_bars": len(barcode.bars),
        "n_essential": len(cens),
        "n_finite": len(finite),
    }


def survival_report(barcode):
    """S and H on a barcode's own lifetimes, essential classes censored."""
    finite = [b.duration for _, b in barcode.bars if not b.is_essential]
    cens = [b.censored_duration(barcode.t_max)
            for _, b in barcode.bars if b.is_essential]
    est = km(finite, cens)
    gap = float(np.max(np.abs(est.S - np.exp(-est.H)))) if est.n else 0.0
    return {"km": est, "km_vs_na": gap, "n": est.n, "n_events": len(est.t)}


def identity_check(n=600, rate=1.3, seed=0, cells=40):
    """S(tau) = exp(-integral h) on a planted sample where h is a known constant.

    Two agreements, because they fail for different reasons.

    km_vs_na is the ESTIMATOR gap between the product-limit prod(1 - d/n) and
    exp(-sum d/n). It is order 1/n and must shrink when n grows; it does not
    depend on any quadrature, so it cannot be tuned away by a finer grid.

    quadrature is the ARITHMETIC gap: the hazard is re-estimated on a uniform
    grid as events-per-unit-exposure per cell, integrated back by the trapezoid
    rule, and exp of the negative integral is compared against S. On a stepped
    estimator this is the only honest way to run the identity -- differencing a
    step function gives spikes, not a hazard.

    rate_hat is the control: the planted hazard is a constant and must come back.
    """
    rng = np.random.default_rng(seed)
    x = rng.exponential(1.0 / rate, n)
    est = km(x)
    km_vs_na = float(np.max(np.abs(est.S - np.exp(-est.H))))

    hi = float(np.quantile(x, 0.95))
    edges = np.linspace(0.0, hi, cells + 1)
    width = edges[1] - edges[0]
    h = np.empty(cells)
    for i in range(cells):
        at_risk = int(np.sum(x >= edges[i]))
        events = int(np.sum((x >= edges[i]) & (x < edges[i + 1])))
        h[i] = 0.0 if at_risk == 0 else events / (at_risk * width)
    integral = np.concatenate(([0.0], np.cumsum(h * width)))
    worst = max(abs(math.exp(-integral[i + 1]) - s_at(est, edges[i + 1]))
                for i in range(cells))
    return {
        "km_vs_na": km_vs_na,
        "quadrature": float(worst),
        "rate_hat": float(n / np.sum(x)),
        "grid_hi": hi,
        "n": n,
    }


# ---------------------------------------------------------------------------
# THE DEMO: every number beside its control
# ---------------------------------------------------------------------------

def demo():
    """Every number this module claims, printed with the control beside it.

    Returns the dict of measured values so tests/curvature/test_lifetimes.py can
    pin each published number to the expression that recomputes it in the same
    run. A literal typed into a print statement satisfies a docstring guard
    trivially; it does not satisfy that.
    """
    t0 = time.time()
    stats = {}
    print("A1 lifetimes: every created simplex declares (b, d) on a NAMED axis.")
    print("OCCUPIED, and cited rather than claimed. The per-simplex-pair lifetime is")
    print("Edelsbrunner-Letscher-Zomorodian 2002, spent there on SIMPLIFICATION.")
    print("Barcode-as-survival-data with Kaplan-Meier and Nelson-Aalen is Murris,")
    print("Stolz & Borgwardt, arXiv 2606.11911, 10 June 2026, over FULLY OBSERVED")
    print("features. Kaplan & Meier, JASA 53:457-481, 1958; Nelson 1972 / Aalen 1978.")
    print("CLAIMED HERE: the read-side gate (T-DEAD, (f)) and the essential class")
    print("carried as RIGHT-CENSORED (e). Engines: gudhi 3.12.0 and ripser 0.6.14.")
    print("Seeds pinned; RUN: python -m ceqjepa.lifetimes")

    print("(a) THE AXES, AND THE KILL. A lifetime without its axis is not a")
    print("    lifetime, so the constructor refuses one -- by exception, because")
    print("    a malformed lifetime must not exist to be stored.")
    for ax in AXES:
        print("    %-12s unit %-16s step %g" % (ax.name, ax.unit, ax.step))
    for bad in (None, "scale", (1.0, 2.0)):
        try:
            Lifetime(0.0, 1.0, bad)
        except TypeError as exc:
            print("    axis=%-12r -> refused: %s" % (bad, exc))
        else:
            raise AssertionError("a lifetime was built with axis=%r" % (bad,))
    ess, fin = Lifetime(0.0, math.inf, SCALE), Lifetime(0.0, 2.5, SCALE)
    print("    essential duration %r vs the longest finite bar this bed has, %r"
          % (ess.duration, fin.duration))
    assert math.isinf(ess.duration), "an essential class became a finite number"
    try:
        ess < Lifetime(0.0, 8.0, TIME)
    except AxisMismatch as exc:
        print("    cross-axis order -> refused: %s" % exc)
    else:
        raise AssertionError("a scale bar was ordered against a time bar")

    print("(b) THE BARCODE AGAINST INSPECTION. Two complexes whose answers were")
    print("    written down before the code ran, on two different axes.")
    sq = square_bed()
    sq_h1 = sq.in_dimension(1)
    sq_gap = max(abs(sq_h1[0].birth - SQUARE_H1_BY_INSPECTION[0]),
                 abs(sq_h1[0].death - SQUARE_H1_BY_INSPECTION[1]))
    print("    unit square  H0 %d finite + %d essential, H1 %s"
          % (sq.counts()[0][0], sq.counts()[0][1], sq_h1[0]))
    print("    inspection   H1 (%g, %.16g) -> worst endpoint gap %.3e"
          % (SQUARE_H1_BY_INSPECTION[0], SQUARE_H1_BY_INSPECTION[1], sq_gap))
    assert sq_gap < 1e-12, "the square barcode left inspection: %r" % sq_gap
    assert sq.counts()[0] == (3, 1), "the square's H0 is not 3 finite + 1 essential"

    rg = ring_bed()
    rg_h1 = rg.in_dimension(1)
    rg_gap = max(abs(rg_h1[0].birth - RING_H1_BY_INSPECTION[0]),
                 abs(rg_h1[0].death - RING_H1_BY_INSPECTION[1]))
    print("    growing ring H0 %d finite + %d essential, H1 %s over %d simplices"
          % (rg.counts()[0][0], rg.counts()[0][1], rg_h1[0], rg.n_simplices))
    print("    inspection   H1 (%g, %g) -> worst endpoint gap %.3e"
          % (RING_H1_BY_INSPECTION[0], RING_H1_BY_INSPECTION[1], rg_gap))
    assert rg_gap < 1e-12, "the ring barcode left inspection: %r" % rg_gap
    assert rg.counts()[0] == (RING_N - 1, 1), "the ring's H0 count moved"

    tp = temperature_bed()
    tp_h0 = tp.in_dimension(0)
    merge = min(b.death for b in tp_h0 if not b.is_essential)
    print("    two wells    H0 %d finite + %d essential over %d cells"
          % (tp.counts()[0][0], tp.counts()[0][1], tp.n_simplices))
    print("    the two components merge at level %.4f against the analytic" % merge)
    print("    saddle 1.0, a gap of %.4f at grid step %g -- the grid's own"
          % (merge - 1.0, TEMPERATURE.step))
    print("    resolution of the saddle, reported and not rounded away.")
    assert tp.counts()[0][0] == 1 and tp.counts()[0][1] == 1, \
        "the two-well bed is not two components: %r" % (tp.counts(),)

    print("(c) THE SAME BARCODE TWO WAYS. gudhi against ripser on the scale bed:")
    print("    two engines, one definition, %d points at noise %g." % (SCALE_N, SCALE_NOISE))
    X = scale_bed()
    a, b = rips_barcode(X, SCALE_THRESH, SCALE), ripser_barcode(X, SCALE_THRESH, SCALE)
    worst = 0.0
    for dim in (0, 1):
        pa, pb = a.in_dimension(dim), b.in_dimension(dim)
        assert len(pa) == len(pb), "H%d: gudhi %d bars, ripser %d" % (dim, len(pa), len(pb))
        gap = max((max(abs(u.birth - v.birth),
                       0.0 if u.is_essential else abs(u.death - v.death))
                   for u, v in zip(pa, pb)), default=0.0)
        print("    H%d  %d bars each, worst endpoint disagreement %.3e" % (dim, len(pa), gap))
        worst = max(worst, gap)
    print("    %d of those H0 bars is ESSENTIAL -- ripser reports it and this file"
          % b.n_essential)
    print("    keeps them; ceq/certs/topological.py:401 returns only dgms[1] and")
    print("    filters non-finite deaths, so no essential class reaches a caller there.")
    print("    worst over both dimensions %.3e, which is ripser's float32" % worst)
    print("    accumulation against gudhi's float64 and not a difference of")
    print("    definition. CONTROL: the inspection barcodes in (b) agree to 0.000e+00.")
    assert worst < 1e-5, "the engines disagree by %r" % worst
    stats["engine_worst"] = worst

    print("(d) PER AXIS, EVERY COUNT BESIDE THE SIMPLICES IT CAME FROM.")
    for bc in (rg, a, tp):
        cnt = bc.counts()
        print("    %-12s step %-9g t_max %-8.4g  %d simplices  "
              "H0 %d+%dinf  H1 %d+%dinf"
              % (bc.axis.name, bc.axis.step, bc.t_max, bc.n_simplices,
                 cnt.get(0, (0, 0))[0], cnt.get(0, (0, 0))[1],
                 cnt.get(1, (0, 0))[0], cnt.get(1, (0, 0))[1]))
    stats["scale_n_finite"] = a.n_finite
    stats["scale_n_essential"] = a.n_essential
    stats["scale_n_simplices"] = a.n_simplices
    print("    scale bed: %d finite classes and %d essential over %d simplices."
          % (stats["scale_n_finite"], stats["scale_n_essential"],
             stats["scale_n_simplices"]))

    print("(e) SURVIVAL AND HAZARD. The estimator is KAPLAN-MEIER; the essential")
    print("    classes are RIGHT-CENSORED at t_max, never dropped.")
    rep = survival_report(a)
    print("    scale bed: %d observations, %d event times, prod-limit against"
          % (rep["n"], rep["n_events"]))
    print("    exp(-cumulative hazard) worst %.6f. They are the DISCRETE PAIR and"
          % rep["km_vs_na"])
    print("    not exact inverses, so the gap is reported and never asserted away.")
    surviving = s_at(rep["km"], a.t_max)
    cum_h = float(rep["km"].H[-1])
    print("    CENSORED MASS. With %d essential class right-censored at t_max %g,"
          % (a.n_essential, a.t_max))
    print("    S(t_max) = %.6f > 0 at a FINITE cumulative hazard %.4f -- which is"
          % (surviving, cum_h))
    print("    what an essential class looks like when it is not encoded as a big")
    print("    finite death. The finite-only sample instead reaches S = 0.000000.")
    dropped_km = km([b.duration for _, b in a.bars if not b.is_essential])
    assert surviving > 0.0 and math.isfinite(cum_h), "the censored mass vanished"
    assert s_at(dropped_km, a.t_max) == 0.0, "the dropped sample did not reach zero"
    stats["surviving_mass"] = surviving
    small, large = identity_check(n=60), identity_check(n=600)
    for tag, r in (("n=60 ", small), ("n=600", large)):
        print("    planted exponential %s  km-vs-exp(-H) %.6f  quadrature %.6f"
              "  rate %.4f (planted 1.3)"
              % (tag, r["km_vs_na"], r["quadrature"], r["rate_hat"]))
    fall_est = 100.0 * (1.0 - large["km_vs_na"] / small["km_vs_na"])
    fall_quad = 100.0 * (1.0 - large["quadrature"] / small["quadrature"])
    print("    BOTH fall over that decade -- the direction does not separate them,")
    print("    the RATE does: estimator gap %.2f%%, quadrature %.2f%%. The estimator"
          % (fall_est, fall_quad))
    print("    gap is order 1/n; the quadrature is floored by the bin width and no n")
    print("    removes that floor, which is why only the first is asserted below.")
    stats["fall_estimator_pct"] = fall_est
    stats["fall_quadrature_pct"] = fall_quad
    assert fall_quad > 0.0, "the quadrature did not fall at all: the claim is the rate"
    assert fall_est > 3.0 * fall_quad,         "the two gaps fell at comparable rates: the distinction claimed is not there"
    assert large["km_vs_na"] < small["km_vs_na"], "the estimator gap did not shrink"
    assert abs(large["rate_hat"] - 1.3) < 0.2, "the planted rate was not recovered"
    stats["identity_quadrature"] = large["quadrature"]
    stats["identity_rate_hat"] = large["rate_hat"]

    bias = censoring_bias(a)
    stats["censoring_bias"] = bias["bias"]
    print("    CENSORED vs DROPPED, the bias the owner asked to be shown: over")
    print("    %d bars of which %d essential, the restricted mean lifetime is"
          % (bias["n_bars"], bias["n_essential"]))
    print("    %.6f censored and %.6f with the essential classes dropped, a"
          % (bias["censored_mean"], bias["dropped_mean"]))
    print("    downward bias of %.6f in rips-radius units at step %g."
          % (bias["bias"], SCALE.step))
    assert bias["bias"] > 0.0, "dropping the essential classes did not bias downward"

    print("(f) T-DEAD ON THE 3x2, with the two trivial criteria beside it. The")
    print("    oracle is the intervals themselves; sensitivity is over the reads")
    print("    that MUST refuse (undefined + null), specificity over the reads")
    print("    that must answer. No pooled rate is computed at all.")
    cases = t_dead_cases(a, seed=0)
    stats["t_dead_n_cases"] = len(cases)
    real = score_3x2(cases, read_criterion)
    assert all(sum(real["table"][c].values()) > 0 for c in (DEFINED, UNDEFINED, NULL)),         "one truth class is empty: the 3x2 cannot discriminate on this bed"
    stats["t_dead_sensitivity"] = real["sensitivity"]
    stats["t_dead_specificity"] = real["specificity"]
    print("    %d cases: %d must answer, %d must refuse"
          % (real["n"], real["n_should_answer"], real["n_should_refuse"]))
    for label, crit in (("interval (ours)", read_criterion),
                        ("refuse everything", refuse_everything),
                        ("answer everything", answer_everything)):
        sc = score_3x2(cases, crit)
        tb = sc["table"]
        print("    %-18s sensitivity %6.2f%%  specificity %6.2f%%   "
              "(defined %d/%d ans, undefined %d/%d ref, null %d/%d ref)"
              % (label, 100 * sc["sensitivity"], 100 * sc["specificity"],
                 tb[DEFINED]["answered"], tb[DEFINED]["answered"] + tb[DEFINED]["refused"],
                 tb[UNDEFINED]["refused"], tb[UNDEFINED]["answered"] + tb[UNDEFINED]["refused"],
                 tb[NULL]["refused"], tb[NULL]["answered"] + tb[NULL]["refused"]))
    assert real["sensitivity"] == 1.0 and real["specificity"] == 1.0, \
        "the read is not exact on its own intervals: %r" % (real,)
    print("    Refuse-everything reaches 100.00% sensitivity at 0.00% specificity")
    print("    and answer-everything the mirror of it: a single pooled agreement")
    print("    rate would have ranked one of them first on this bed.")

    print("(g) PLANTED NEGATIVES, each SEEN to fire.")
    alive = Lifetime(0.0, math.inf, SCALE)
    dead = Lifetime(0.0, 1.0, SCALE)
    taus = np.linspace(0.0, 1e6, 501)
    n_alive_ref = sum(1 for t in taus if is_refusal(read_at(alive, float(t), SCALE, 1.0)))
    dtaus = np.linspace(1.0, 500.0, 500)
    n_dead_ans = sum(1 for t in dtaus
                     if not is_refusal(read_at(dead, float(t), SCALE, 1.0)))
    print("    a class alive across the filtration: refused %d of %d reads out to"
          % (n_alive_ref, len(taus)))
    print("    tau = 1e6 -- a global relation is always evidence.")
    print("    a class dead at d = 1: answered %d of %d reads on [1, 500]."
          % (n_dead_ans, len(dtaus)))
    assert n_alive_ref == 0 and n_dead_ans == 0, "a planted negative did not fire"
    stats["planted_alive_refusals"] = n_alive_ref
    stats["planted_dead_answers"] = n_dead_ans

    print("(h) THE MARS ITEM: lifetimes ARE gamed by the filtration step, and the")
    print("    sensitivity is measured rather than hidden. The same points, the")
    print("    same threshold, the distances quantised onto a grid of width step:")
    base = a.in_dimension(1)
    moved = 0.0
    for step in STEP_SWEEP:
        c = rips_barcode(X, SCALE_THRESH, SCALE, step=step)
        h1 = c.in_dimension(1)
        if len(h1) == len(base):
            shift = max(max(abs(u.birth - v.birth), abs(u.death - v.death))
                        for u, v in zip(h1, base))
            print("    step %-6g  H1 %d bars  longest %.4f  worst endpoint shift %.4f"
                  % (step, len(h1), max(b.duration for b in h1), shift))
            moved = max(moved, shift)
        else:
            print("    step %-6g  H1 %d bars (was %d): the bar COUNT changed"
                  % (step, len(h1), len(base)))
            moved = math.inf
    stats["mars_worst_shift"] = moved
    print("    Worst H1 endpoint shift across the sweep: %.4f. A lifetime quoted"
          % (moved if math.isfinite(moved) else float("nan")))
    print("    without its step is not reproducible, which is why Axis carries one.")
    assert moved > 0.01, "no coarsening moved a bar: the sweep is not biting"

    dt = time.time() - t0
    print("    demo() wall clock %.2f s against the 180 s budget, this box, CPU." % dt)
    assert dt < 180.0, "over budget at %r s" % dt
    print("ALL SELF-CHECKS PASSED")
    return stats


if __name__ == "__main__":
    demo()

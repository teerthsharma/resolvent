"""A3: is a kNN cluster a PHASE because binding beats thermal, or because a
threshold was fitted and then given thermodynamic vocabulary?

BOTH, AND THE SPLIT IS THE RESULT. The FORM of the critical line is forced and
not fitted. The CONSTANT in it is fitted, and it moves with the instrument.
Every number here is printed by one run of `python -m ceqjepa.phase_criterion`.

WHY THE FORM IS FORCED, NOT FITTED. The bed carries exactly two lengths, sep and
sigma. kNN ranking is scale-free and the hop metric discards edge length, so a
tenfold rescale at fixed seed moves the cloud by 7.105e-15 and moves the binding
and the entropy by 0, bitwise. Their ratio is the only dimensionless combination
two lengths can build, so sep_crit = c*sigma is the only form available at all;
sep proportional to T*T needs a third length this bed does not contain. The
linear fit is therefore not a discovery, it is a CHECK that bed and instrument
respect the invariance they claim, and the alternatives lose because they must.

WHY THE CONSTANT IS NOT PHYSICS. Swept over neighbour count at fixed seeds, the
critical ratio runs 2.7832, 2.4904, 2.2247 at k = 5, 8, 12: monotone, a move of
-22.4% of the k=8 value. The marginal CIs overlap, because they are dominated by
a seed spread common to every k; paired on the three shared seeds the difference
is +0.5586 with CI [+0.0920, +1.0251], excluding zero. Repeated at a SECOND bed
size, 25 points per phase against 20, the same control gives 3.0932, 2.6965,
2.4039, a move of -25.6%, paired +0.6893 with CI [+0.3688, +1.0098]: same sign,
same order, both excluding zero. The owner's k-gaming control FIRES twice: c is
an instrument property. The form survives it untouched, because the scale
invariance that forces the form holds at every k and at every bed size.

An earlier report of this work cited a corroborating percentage from a probe
that lived outside the tree. No run, module or test in this repository produced
it, so it is WITHDRAWN, and it is not restated here: a withdrawn figure quoted
in a docstring is the same unbindable number wearing an apology. The second bed
size above replaces it and every digit of it is printed by this run.

THE FOUR-FACTOR CRITERION IS NOT COMMENSURABLE. coupling x density is L^-3 and
temperature x entropy is L^+1, so as written it compares quantities that no
change of unit can make comparable. Rerouted, not retired: kappa already IS what
coupling and density jointly produce on the graph, and sigma/sep is the only
dimensionless temperature two lengths admit.

T-CRIT. Six train legs give c = 2.4978, CI [2.1262, 2.8694], RSS 1.097160, AIC
-8.1942, residual RMS 0.427621. Fitted on the IDENTICAL points, sep = c*T*T
gives RSS 5.332367 and sep = constant gives RSS 15.827721: the linear form wins
at RSS ratios 4.9 and 14.4 and AIC gaps 9.49 and 16.01. Held out on
temperatures and seeds the fit never saw, the data implies c = 2.7738, inside
the frozen interval.

LANDAU IS SECOND ORDER HERE, NOT FIRST. Two local minima do appear, but the
shallower sits 0.005631 below the global one against a seed-to-seed spread of
0.011362 in the same family, so no jump between wells is demonstrated; the
minimum SLIDES. It crosses the critical ratio at tau = 0.1857 while the
criterion's own crossing is 1/c = 0.4004, a gap of -0.2146. Those are two
different statements and are reported as two.

T-LIFE. The binding budget predicts tau_death = U/S = 1.1666, CI [0.8980,
1.4352]. The measured death is 0.4789, CI [0.3549, 0.6029]. The budget
OVERPREDICTS the survivable temperature by -0.6877, because it spends the
entropy of the cold configuration while the phase dies at a hotter one where the
entropy has grown. The owner's lineage figure is 0.60 at coupling 1/4, from HIS
run, never a source for anything here; this measurement sits -0.1211 from it.

REFUSALS ARE VALUES WITH REASONS. On the structureless bed the entropy does not
respond to the control parameter, dS/ds = -0.00053 with CI [-0.00295, +0.00188],
against -0.15533 with CI [-0.19390, -0.11676] on the plant, so the critical line
is refused rather than reported -- the bare sign change is not evidence, since
U - tau*S crosses zero on ANY bed with positive binding as tau falls. The H0
barcode route is refused too: the A-to-B single-linkage join is link 1 of 40 at
its earliest, so the second bar is born already dead.

Run: python -m ceqjepa.phase_criterion
"""

import hashlib
import json
import re as _re
import sys
import time

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from scipy.stats import t as student_t

from . import curvature as cv

# ---------------------------------------------------------------------------
# THE PINNED BED AND INSTRUMENT
# ---------------------------------------------------------------------------

N_PER = 20                  # points per phase; N = 2*N_PER + 1
DIM = 3
JITTER = 0.05               # isthmus scatter AS A FRACTION OF sep -- see the bed
K = 8                       # curvature.K_NEIGHBORS
COUPLING_COLD = 0.5         # curvature.ALPHA
COUPLING_HOT = 0.25         # the owner's "coupling 1/4"

S_GRID = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0)     # sep / sigma
TRAIN_T = (0.5, 1.0, 2.0)
TEST_T = (0.75, 1.5)
TRAIN_SEEDS = (4101, 4102)
TEST_SEEDS = (8201, 8202)
COUPLING_SEEDS = (6301, 6302, 6303)
NULL_SEEDS = (9401,)
K_SWEEP = (5, 8, 12)
K_SEEDS = (4101, 4102, 4103)   # the k control is measured PAIRED on shared seeds
K_BED_SIZES = (N_PER, 25)      # and REPEATED at a second bed size, see _k_control
KNOWN_LINE_G = 2.0

OWNER_LINEAGE_TAU = 0.60     # HIS number, from HIS run. Never a source here.
OWNER_LINEAGE_COUPLING = 0.25

Refusal = cv.Refusal
is_refusal = cv.is_refusal


# ---------------------------------------------------------------------------
# 1. THE CRITERION, DEFINED SO IT CAN FAIL
# ---------------------------------------------------------------------------

UNITS = {
    "separation": "L",
    "temperature": "L",
    "reduced_temperature": "1",
    "coupling": "1",
    "entropy": "1",
    "binding": "1",
    "thermal": "1",
}

FOUR_FACTOR_DIMENSIONS = {
    "binding": "coupling x density",
    "binding_exponent": -DIM,
    "thermal": "temperature x entropy",
    "thermal_exponent": +1,
    "verdict": (
        "coupling x density is L^-%d and temperature x entropy is L^+1, so the "
        "four-factor form as written is not commensurable and any threshold read "
        "off it is a fitted number wearing thermodynamic vocabulary. REROUTE, "
        "not retire: kappa already IS what coupling and density jointly produce "
        "on the graph, and the reduced temperature sigma/sep is the only "
        "dimensionless temperature two lengths can build, so binding > thermal "
        "survives as kappa_within > (sigma/sep) * S with both sides in units of 1."
    ) % DIM,
}


def reduced_temperature(sep, sigma):
    """tau = sigma / sep. Dimensionless, and the ONLY temperature this bed has.

    sep and sigma are the bed's only two lengths. kNN ranking is scale-free and
    the hop metric discards edge length, so every measurable is a function of the
    ratio alone. A temperature that is a length cannot multiply a dimensionless
    entropy; a temperature that is a ratio can.
    """
    return float(sigma) / float(sep)


def binding(m):
    """U = mean Ollivier-Ricci kappa over WITHIN-phase edges. Dimensionless.

    Positive kappa is a bound neighbourhood: mass moves between neighbours at
    less than the graph distance. The laziness alpha (the coupling) and the kNN
    degree (the density) both enter kappa by construction, so this single number
    is the product the four-factor form tried to write as two factors with
    incompatible units.
    """
    return float(m["U"])


def thermal(m, tau):
    """tau * S: reduced temperature times mixing entropy. Dimensionless.

    S is the binary Shannon entropy in nats of the edge partition into
    within-phase and cross-phase. Maximal when the labels are indistinguishable
    from a coin, zero when the boundary carries no edges at all.
    """
    return float(tau) * float(m["S"])


def free_energy(U, S, tau):
    """Landau F = E - T*S with E = -U. All three terms dimensionless."""
    return -float(U) - float(tau) * float(S)


def criterion(m, sep, sigma):
    """PHASE iff binding exceeds thermal. Returns (verdict, margin)."""
    tau = reduced_temperature(sep, sigma)
    margin = binding(m) - thermal(m, tau)
    return bool(margin > 0.0), float(margin)


# ---------------------------------------------------------------------------
# 2. THE BEDS
# ---------------------------------------------------------------------------

def two_phase_bed(seed, sep, sigma, n_per=N_PER, dim=DIM, jitter=JITTER):
    """Two isotropic phases and one isthmus point. EVERY length is proportional.

    The isthmus scatter is jitter*sep and there is no additive constant anywhere,
    so the bed contains exactly two lengths and X(lambda*sep, lambda*sigma) is
    lambda*X(sep, sigma) exactly, at fixed seed. That is what forces the critical
    line to be linear: sep_crit = c*sigma is the only form two lengths can make.

    curvature.two_cluster_bed cannot be used for this. It subtracts a hardcoded
    length from the isthmus span and adds a hardcoded absolute jitter, which puts
    a third scale into a two-scale problem; a critical line measured there is
    contaminated by that scale and its slope is not the slope of anything.
    """
    rng = np.random.default_rng(seed)
    off = np.zeros(dim)
    off[0] = sep / 2.0
    A = rng.normal(scale=sigma, size=(n_per, dim)) - off
    B = rng.normal(scale=sigma, size=(n_per, dim)) + off
    mid = rng.normal(scale=jitter * sep, size=(1, dim))
    X = np.vstack([A, B, mid])
    labels = np.concatenate([np.zeros(n_per, int), np.ones(n_per, int),
                             np.full(1, 2)])
    return X, labels


def null_bed(seed, sep, sigma, n_per=N_PER, dim=DIM):
    """ONE phase, split by a coin. sep redraws the cloud but plants no structure.

    The hard form of the negative: sep is not inert, so the measured values move
    from cell to cell and the entropy slope has real sampling noise to be tested
    against. A null whose data is bit-identical at every sep would make the gate
    look sharper than it is.
    """
    rng = np.random.default_rng(int(seed) * 7919 + int(round(sep * 1000)))
    X = rng.normal(scale=sigma, size=(2 * n_per + 1, dim))
    labels = np.zeros(2 * n_per + 1, int)
    labels[n_per:2 * n_per] = 1
    labels[-1] = 2
    rng.shuffle(labels)
    return X, labels


def bed_measure(X, labels, k=K, alpha=COUPLING_COLD):
    """Binding, entropy and the edge counts behind them, from one curvature run."""
    r = cv.curvature_from_points(X, k=k, alpha=alpha, weighting="hop",
                                 measure="uniform")
    kf = np.asarray(r["kappa_full"], dtype=float)
    within, n_cross = [], 0
    for pos, (i, j) in enumerate(r["all_edges"]):
        a, b = int(labels[i]), int(labels[j])
        if a == b and a != 2:
            within.append(kf[pos])
        elif a != b:
            n_cross += 1
    n_all = len(r["all_edges"])
    q = n_cross / float(n_all) if n_all else float("nan")
    S = 0.0 if not (0.0 < q < 1.0) else float(-(q * np.log(q)
                                                + (1 - q) * np.log(1 - q)))
    return dict(U=float(np.mean(within)) if within else float("nan"),
                S=S, q=float(q), n_within=len(within), n_cross=n_cross,
                n_edges=int(r["n_edges"]), n_refused=int(r["n_refused"]),
                k=int(k), alpha=float(alpha))


# ---------------------------------------------------------------------------
# 3. THE CROSSING AND THE FITS
# ---------------------------------------------------------------------------

def leg_seed(base, sigma):
    """An INDEPENDENT draw per (base, temperature).

    The bed is exactly scale invariant, so reusing one seed across temperatures
    reproduces the same graph bitwise and the temperature axis carries no
    information at all: the first run of this module fitted six "points" that
    were two measurements copied three times, and reported a CI narrowed by the
    copies. Every temperature gets its own realisation.
    """
    return int(base) * 1000 + int(round(float(sigma) * 100))


def _sweep_leg(seed, sigma, k=K, alpha=COUPLING_COLD, s_grid=S_GRID,
               n_per=N_PER):
    """One (seed, sigma) leg: the criterion margin at every point of s_grid."""
    out = []
    for s in s_grid:
        X, lab = two_phase_bed(seed=seed, sep=s * sigma, sigma=sigma,
                               n_per=n_per)
        m = bed_measure(X, lab, k=k, alpha=alpha)
        _, margin = criterion(m, s * sigma, sigma)
        m.update(s=float(s), sep=float(s * sigma), sigma=float(sigma),
                 seed=int(seed), margin=float(margin))
        out.append(m)
    return out


def critical_ratio(leg):
    """s* where the margin first crosses zero, linearly interpolated.

    A leg that never crosses inside the swept range is a Refusal and not a NaN:
    the answer is "this grid does not bracket it", which is a different fact from
    "the crossing does not exist".
    """
    for a, b in zip(leg, leg[1:]):
        if a["margin"] < 0.0 <= b["margin"]:
            span = b["s"] - a["s"]
            return float(a["s"] + span * (-a["margin"]) / (b["margin"] - a["margin"]))
    return Refusal("NO_CROSSING",
                   "the margin does not change sign across s in [%g, %g]"
                   % (leg[0]["s"], leg[-1]["s"]))


def _basis(name, T):
    T = np.asarray(T, dtype=float)
    return {"linear": T, "quadratic": T * T, "constant": np.ones_like(T)}[name]


def fit_line(name, T, sep):
    """One-parameter least squares through the origin, with CI, RSS and AIC.

    All three forms carry one parameter, so AIC reduces to the residual ordering;
    it is reported anyway because the owner's test is about which form loses and
    a reader should not have to take the reduction on trust.
    """
    x = _basis(name, T)
    y = np.asarray(sep, dtype=float)
    n = int(y.size)
    sxx = float((x * x).sum())
    c = float((x * y).sum() / sxx)
    resid = y - c * x
    rss = float((resid * resid).sum())
    se = float(np.sqrt(max(rss, 1e-300) / max(n - 1, 1) / sxx))
    tcrit = float(student_t.ppf(0.975, max(n - 1, 1)))
    return dict(model=name, c=c, se=se, ci_lo=c - tcrit * se, ci_hi=c + tcrit * se,
                rss=rss, aic=float(n * np.log(max(rss, 1e-300) / n) + 2.0),
                resid_rms=float(np.sqrt(rss / n)), n=n,
                resid=[float(v) for v in resid])


def fit_alternatives(T, sep):
    return {name: fit_line(name, T, sep)
            for name in ("linear", "quadratic", "constant")}


def slope_ci(x, y):
    """OLS slope of y on x with an intercept, and its CI. Used for dS/ds."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    sl, ic = np.polyfit(x, y, 1)
    resid = y - (sl * x + ic)
    sxx = float(((x - x.mean()) ** 2).sum())
    se = float(np.sqrt(float((resid ** 2).sum()) / (n - 2) / sxx))
    tc = float(student_t.ppf(0.975, n - 2))
    return dict(slope=float(sl), se=se, ci_lo=float(sl - tc * se),
                ci_hi=float(sl + tc * se), n=int(n))


def mean_ci(values):
    v = np.asarray([float(x) for x in values], dtype=float)
    n = v.size
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n else float("nan")
    tc = float(student_t.ppf(0.975, max(n - 1, 1)))
    return dict(mean=float(v.mean()), sd=sd, ci_lo=float(v.mean() - tc * se),
                ci_hi=float(v.mean() + tc * se), n=int(n))


# ---------------------------------------------------------------------------
# 4. PRE-REGISTRATION: A DIGEST OVER THE INPUTS
# ---------------------------------------------------------------------------

_DIGEST_KEYS = ("bed", "instrument", "points", "measured")

#: Filled from the first run and then frozen. If the bed, the seeds, the sweep
#: points or the measured values move, this stops matching and the shipped line
#: is known to have been fitted on data other than the data now shipped.
FROZEN_DIGEST = "ee43b6e57972e4d454e96c8a4394741a9c268abe2924ec36d2517e692f400487"


def freeze_digest(spec):
    """SHA-256 over the INPUTS ONLY: bed spec, instrument, sweep points, values.

    Deliberately NOT over the fitted parameters. A digest that covers the outputs
    cannot tell a freeze from an honest refit -- refit, re-digest, and the record
    verifies against itself -- so the fitted slope is excluded here and any key
    outside _DIGEST_KEYS is ignored.
    """
    payload = {key: spec[key] for key in _DIGEST_KEYS}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def perturb_one_measured_value(spec, delta):
    """A copy of spec with exactly one measured value moved by delta."""
    out = json.loads(json.dumps({k: spec[k] for k in _DIGEST_KEYS}))
    out["measured"][0][0] = out["measured"][0][0] + float(delta)
    return out


def check_freeze(spec, fitted_c):
    """Does this spec's digest match the one pre-registered in the source?"""
    got = freeze_digest(spec)
    if got != FROZEN_DIGEST:
        return dict(ok=False, digest=got, c=float(fitted_c),
                    reason=("input digest %s does not match the pre-registered "
                            "digest %s: this line was fitted on other data"
                            % (got[:16], FROZEN_DIGEST[:16])))
    return dict(ok=True, digest=got, c=float(fitted_c),
                reason="input digest matches the pre-registration")


def _spec(points, measured):
    return dict(
        bed=dict(n_per=N_PER, dim=DIM, jitter=JITTER, family="two_phase_bed"),
        instrument=dict(k=K, weighting="hop", measure="uniform",
                        s_grid=list(S_GRID)),
        points=points,
        measured=measured,
    )


def planted_refit_on_test_grid():
    """The owner's struck sin, planted so the freeze check can be seen to fire.

    Refit the line on the held-out grid and present it as though it were the
    frozen one. The digest is over the inputs, so the substituted measurements
    change it and the check catches the refit rather than taking its word.
    """
    r = report()
    h = r["tcrit"]["heldout"]
    spec = _spec(h["points"], h["measured"])
    fit = fit_line("linear", h["T"], h["sep_crit"])
    return spec, fit["c"]


# ---------------------------------------------------------------------------
# 5. THE H0 ROUTE THAT REFUSED
# ---------------------------------------------------------------------------

def h0_two_phase_death(X, labels):
    """Single-linkage height at which the two planted sides join, and the largest
    within-side link below it. This is the H0 bar of the Vietoris-Rips filtration
    restricted to the plant, computed here rather than imported: lifetimes.py is
    being written in parallel and is not this module's to depend on.
    """
    D = squareform(pdist(X))
    mst = minimum_spanning_tree(D).tocoo()
    edges = sorted(zip(mst.data.tolist(), mst.row.tolist(), mst.col.tolist()))
    parent = list(range(len(X)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    side = [{int(l)} if int(l) != 2 else set() for l in labels]
    scale = float(np.median([e[0] for e in edges]))
    for rank, (w, i, j) in enumerate(edges, start=1):
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        if (0 in side[ri] and 1 in side[rj]) or (1 in side[ri] and 0 in side[rj]):
            return dict(death=float(w), rank=rank, n_links=len(edges),
                        within_scale=scale, ratio=float(w) / scale)
        parent[ri] = rj
        side[rj] = side[rj] | side[ri]
    return dict(death=float("nan"), rank=len(edges), n_links=len(edges),
                within_scale=scale, ratio=float("nan"))


WANTED_FROM_LIFETIMES = (
    "the interface this module would want from a barcode module: "
    "h0_bars(X, labels=None) -> list of (birth, death, representative_vertex), "
    "with the bar's death in the SAME length unit as X, a companion "
    "within_component_scale so persistence can be read as a ratio rather than "
    "an absolute length, and a Refusal when the second bar's death is not "
    "separated from the within-component linkage, instead of a float that a "
    "consumer cannot tell apart from a resolved one."
)


# ---------------------------------------------------------------------------
# THE COVERAGE FIGURE IS A MEASURED NUMBER LIKE ANY OTHER
# ---------------------------------------------------------------------------

#: The same pattern the test uses: any decimal AND any bare integer, scoped by
#: __module__, no exemption list. The test RE-IMPLEMENTS this scan rather than
#: calling this helper -- two counters that must agree is a check; one shared
#: helper this module could quietly narrow is not.
_NUM = _re.compile(
    r"(?<![\w.])[+-]?\d+(?:\.\d+)*(?:[eE][+-]?\d+)?(?!\w)(?!\.\d)")
_DEC = _re.compile(
    r"(?<![\w.])[+-]?\d+\.\d+(?:\.\d+)*(?:[eE][+-]?\d+)?(?!\w)(?!\.\d)")


def docstring_numbers():
    """How many docstrings this module defines, and how many numbers are in them.

    A coverage figure is prose like any other and drifts the same way: a count
    taken from a RED run, before demo() had a docstring of its own, is how a
    guard comes to be reported as wider than it is. So the figure is printed by
    the run and checked against a scan the test writes for itself.
    """
    mod = sys.modules[__name__]
    docs = [("module", mod.__doc__ or "")]
    for name, obj in sorted(vars(mod).items()):
        if getattr(obj, "__module__", None) != __name__:
            continue
        doc = getattr(obj, "__doc__", None)
        if isinstance(doc, str) and doc.strip():
            docs.append((name, doc))
    return dict(n_docstrings=len(docs),
                n_numbers=sum(len(_NUM.findall(d)) for _, d in docs),
                n_decimals=sum(len(_DEC.findall(d)) for _, d in docs),
                names=[n for n, _ in docs])


# ---------------------------------------------------------------------------
# 6. THE REPORT -- ONE SWEEP, CACHED
# ---------------------------------------------------------------------------

_REPORT = {}


def report():
    if _REPORT:
        return _REPORT["r"]
    _REPORT["r"] = _build_report()
    return _REPORT["r"]


def _legs(seeds, temps, k=K, alpha=COUPLING_COLD, n_per=N_PER):
    out = []
    for T in temps:
        for seed in seeds:
            seed = leg_seed(seed, T)
            leg = _sweep_leg(seed=seed, sigma=T, k=k, alpha=alpha,
                             n_per=n_per)
            out.append(dict(T=float(T), seed=int(seed), alpha=float(alpha),
                            k=int(k), n_per=int(n_per), leg=leg,
                            s_star=critical_ratio(leg)))
    return out


def _grid_points(legs):
    """(points, measured, T, sep_crit) for the legs that produced a crossing."""
    points, measured, T, sep = [], [], [], []
    for g in legs:
        if is_refusal(g["s_star"]):
            continue
        points.append([g["alpha"], g["T"], g["seed"], g["k"]])
        measured.append([round(p["U"], 12) for p in g["leg"]]
                        + [round(p["S"], 12) for p in g["leg"]])
        T.append(g["T"])
        sep.append(g["s_star"] * g["T"])
    return points, measured, T, sep


def _k_control(train, n_per=N_PER):
    """The owner's k-gaming control at ONE bed size: does s* track the graph?

    Repeated at a second bed size by the caller, because a control run once is a
    number with no spread behind it. An earlier report of this work cited a
    second-bed-size figure that no run in the tree produced; this function is
    the producer that claim needed and did not have.

    PAIRED across shared seeds. The marginal CI at each k is dominated by the
    seed-to-seed spread of s*, which is common to every k and cancels in the
    difference; comparing two marginal CIs throws that cancellation away and
    calls a real trend "overlapping".
    """
    rows, by_k = [], {}
    for kk in K_SWEEP:
        legs_k = _legs(K_SEEDS, (1.0,), k=kk, n_per=n_per)
        by_k[kk] = {g_["seed"]: g_["s_star"] for g_ in legs_k}
        st = [v for v in by_k[kk].values() if not is_refusal(v)]
        ci = mean_ci(st)
        rows.append(dict(k=int(kk), s_star=ci["mean"], ci_lo=ci["ci_lo"],
                         ci_hi=ci["ci_hi"], n_seeds=ci["n"],
                         per_seed=[float(v) for v in st],
                         n_edges=int(np.mean([q_["n_edges"] for g_ in legs_k
                                              for q_ in g_["leg"]]))))
    rows.sort(key=lambda r_: r_["k"])
    k_lo, k_hi = rows[0]["k"], rows[-1]["k"]
    shared = [s_ for s_ in (leg_seed(b, 1.0) for b in K_SEEDS)
              if not is_refusal(by_k[k_lo][s_]) and not is_refusal(by_k[k_hi][s_])]
    paired = [float(by_k[k_lo][s_] - by_k[k_hi][s_]) for s_ in shared]
    pci = mean_ci(paired)
    monotone = all(a["s_star"] > b["s_star"] for a, b in zip(rows, rows[1:]))
    lo, hi = rows[0]["s_star"], rows[-1]["s_star"]
    mid = [r_ for r_ in rows if r_["k"] == K][0]["s_star"]
    frac_move = float((hi - lo) / mid)
    tracks = bool(abs(frac_move) > 0.10)
    ci_disjoint = bool(rows[-1]["ci_hi"] < rows[0]["ci_lo"])
    return dict(
        n_per=int(n_per), n_points=int(2 * n_per + 1), rows=rows,
        frac_move=frac_move, tracks_k=tracks, ci_disjoint=ci_disjoint,
        monotone=monotone, paired=paired, paired_mean=pci["mean"],
        paired_ci_lo=pci["ci_lo"], paired_ci_hi=pci["ci_hi"], paired_n=pci["n"],
        paired_excludes_zero=bool(pci["ci_lo"] > 0.0 or pci["ci_hi"] < 0.0),
        verdict=("the critical ratio moves %.1f%% of its k=%d value across "
                 "k in [%d, %d], so the CONSTANT is an instrument property and "
                 "not a fact about the geometry; the FORM sep = c*T is untouched "
                 "by k because the scale invariance that forces it holds at "
                 "every k. The end CIs are %s"
                 % (100.0 * frac_move, K, k_lo, k_hi,
                    "DISJOINT" if ci_disjoint else "OVERLAPPING; the marginal "
                    "CIs are dominated by seed spread common to every k, so the "
                    "PAIRED difference is the statistic that answers this"))
        if tracks else
        ("the critical ratio moves %.1f%% of its k=%d value across k in [%d, %d], "
         "inside the seed noise, so the line does not detectably track the "
         "construction" % (100.0 * frac_move, K, k_lo, k_hi)))


def _build_report():
    t0 = time.time()

    # -- identity probe -----------------------------------------------------
    ip_seed, ip_sep, ip_sigma = 4101, 3.0, 1.0
    X, lab = two_phase_bed(seed=ip_seed, sep=ip_sep, sigma=ip_sigma)
    ipm = bed_measure(X, lab, k=K, alpha=COUPLING_COLD)

    # -- exact scale invariance ---------------------------------------------
    Xa, la = two_phase_bed(seed=7, sep=6.0, sigma=1.0)
    Xb, lb = two_phase_bed(seed=7, sep=60.0, sigma=10.0)
    ma, mb = bed_measure(Xa, la), bed_measure(Xb, lb)
    Xc, _ = cv.two_cluster_bed(seed=7, sigma=0.8, sep=9.0)
    Xd, _ = cv.two_cluster_bed(seed=7, sigma=8.0, sep=90.0)
    scale = dict(dU=abs(ma["U"] - mb["U"]), dS=abs(ma["S"] - mb["S"]),
                 dX=float(np.abs(10.0 * Xa - Xb).max()),
                 shipped_dX=float(np.abs(10.0 * Xc - Xd).max()),
                 n_edges=ma["n_edges"], n_refused=ma["n_refused"])

    # -- T-CRIT -------------------------------------------------------------
    train = _legs(TRAIN_SEEDS, TRAIN_T)
    tr_pts, tr_meas, tr_T, tr_sep = _grid_points(train)
    fits = fit_alternatives(tr_T, tr_sep)
    winner = min(fits, key=lambda k_: fits[k_]["aic"])

    test = _legs(TEST_SEEDS, TEST_T)
    te_pts, te_meas, te_T, te_sep = _grid_points(test)
    te_T_a, te_sep_a = np.asarray(te_T, float), np.asarray(te_sep, float)
    heldout = dict(n=len(te_T), T=te_T, sep_crit=te_sep,
                   points=te_pts, measured=te_meas,
                   temperatures=sorted({float(v) for v in te_T}),
                   seeds=sorted({int(p[2]) for p in te_pts}),
                   c_implied=float((te_T_a * te_sep_a).sum() / (te_T_a ** 2).sum()))

    spec = _spec(tr_pts, tr_meas)
    freeze = dict(spec=spec, digest=freeze_digest(spec),
                  check=check_freeze(spec, fits["linear"]["c"]))

    tcrit = dict(fits=fits, winner=winner, heldout=heldout,
                 train_T=tr_T, train_sep_crit=tr_sep,
                 train_temperatures=sorted({float(v) for v in tr_T}),
                 train_seeds=sorted({int(p[2]) for p in tr_pts}),
                 n_refused_legs=sum(1 for g in train if is_refusal(g["s_star"])),
                 s_star_by_leg=[(g["T"], g["seed"],
                                 g["s_star"] if not is_refusal(g["s_star"])
                                 else str(g["s_star"].code)) for g in train])

    # -- LANDAU -------------------------------------------------------------
    fam_leg = [g for g in train if g["T"] == 1.0][0]["leg"]
    family = [dict(s=p["s"], U=p["U"], S=p["S"], q=p["q"],
                   n_edges=p["n_edges"], n_refused=p["n_refused"])
              for p in fam_leg]
    ladder = []
    for tau in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.70, 1.00):
        F = np.array([free_energy(p["U"], p["S"], tau) for p in family])
        minima = [(family[i]["s"], float(F[i])) for i in range(F.size)
                  if (i == 0 or F[i] <= F[i - 1])
                  and (i == F.size - 1 or F[i] <= F[i + 1])]
        ladder.append(dict(tau=float(tau), argmin_s=float(family[int(F.argmin())]["s"]),
                           F_min=float(F.min()), minima=minima))
    tau_crit = 1.0 / fits["linear"]["c"]
    # The transition temperature is where the free-energy minimum crosses the
    # criterion's own critical ratio, NOT the first grid step at which argmin
    # moves: a minimum that slides continuously moves at every step, and calling
    # the first of them a transition reports the grid spacing as physics.
    tau_jump, jump_from, jump_to = float("nan"), None, None
    for a, b in zip(ladder, ladder[1:]):
        if a["argmin_s"] >= fits["linear"]["c"] > b["argmin_s"]:
            span = b["tau"] - a["tau"]
            lo, hi = a["argmin_s"], b["argmin_s"]
            tau_jump = a["tau"] + span * (lo - fits["linear"]["c"]) / (lo - hi)
            jump_from, jump_to = lo, hi
            break

    #: A local minimum of a discrete sweep appears from sampling noise alone, so
    #: the well count is read against the seed-to-seed spread of the same family.
    n_wells = [len(r["minima"]) for r in ladder]
    gaps = [sorted(r["minima"], key=lambda z: z[1])[1][1]
            - sorted(r["minima"], key=lambda z: z[1])[0][1]
            for r in ladder if len(r["minima"]) >= 2]
    well_gap = float(min(gaps)) if gaps else float("nan")
    other = [g for g in train if g["T"] == 1.0][1]["leg"]
    well_noise = float(np.std([abs(p["U"] - o["U"])
                               for p, o in zip(fam_leg, other)], ddof=1))
    first_order = bool(max(n_wells) >= 2 and well_gap > well_noise)
    landau = dict(family=family, ladder=ladder, tau_jump=tau_jump,
                  jump_from=jump_from, jump_to=jump_to,
                  tau_crit_from_criterion=float(tau_crit),
                  gap_to_criterion=float(tau_jump - tau_crit),
                  n_wells_max=int(max(n_wells)), well_gap=well_gap,
                  well_noise=well_noise, wells_exceed_noise=first_order,
                  transition_order="first" if first_order else "second",
                  order_verdict=(
                      "at most %d local minima appear on the ladder, but the "
                      "shallower one sits %s below the global minimum against a "
                      "seed-to-seed spread of %s in the same family, so it is not "
                      "resolved and no jump between wells is demonstrated. What "
                      "this bed shows is a minimum that SLIDES with temperature: "
                      "the CONTINUOUS form of the same Landau statement, not the "
                      "addendum's first-order picture."
                      % (max(n_wells),
                         "nothing" if not np.isfinite(well_gap)
                         else "%.6f" % well_gap, "%.6f" % well_noise)
                      ) if not first_order else (
                      "two wells separated by %.6f, above the seed noise %.6f: "
                      "the minimum jumps and the transition is FIRST order"
                      % (well_gap, well_noise)))

    # -- T-LIFE -------------------------------------------------------------
    hot = _legs(COUPLING_SEEDS, (1.0,), alpha=COUPLING_HOT)
    s0 = 4.0
    pred, meas, at_s0, s_stars = [], [], [], []
    for g in hot:
        p0 = [p for p in g["leg"] if p["s"] == s0][0]
        pred.append(p0["U"] / p0["S"])
        at_s0.append(dict(seed=g["seed"], U=p0["U"], S=p0["S"],
                          n_edges=p0["n_edges"], n_refused=p0["n_refused"]))
        if not is_refusal(g["s_star"]):
            meas.append(1.0 / g["s_star"])
            s_stars.append(float(g["s_star"]))
    pred_ci, meas_ci = mean_ci(pred), mean_ci(meas)

    h0 = [h0_two_phase_death(*two_phase_bed(seed=COUPLING_SEEDS[0], sep=s,
                                            sigma=1.0)) for s in S_GRID]
    ranks = [h["rank"] for h in h0]
    barcode = dict(
        death=Refusal("H0_INTERPENETRATING",
                      "the A-to-B single-linkage join is the %d-th of %d links "
                      "at its earliest and the %d-th at its latest, so the "
                      "second H0 bar is born already dead and no death can be "
                      "read off it: Gaussian tails at this width overlap before "
                      "the phases separate"
                      % (min(ranks), h0[0]["n_links"], max(ranks))),
        rows=[dict(s=float(s), death=h["death"], rank=h["rank"],
                   ratio=h["ratio"], within_scale=h["within_scale"])
              for s, h in zip(S_GRID, h0)],
        rank_min=int(min(ranks)), rank_max=int(max(ranks)),
        n_links=int(h0[0]["n_links"]),
        ratio_max=float(max(h["ratio"] for h in h0)),
        ratio_min=float(min(h["ratio"] for h in h0)),
        n=len(h0), wanted_interface=WANTED_FROM_LIFETIMES)

    tlife = dict(coupling=COUPLING_HOT, s0=s0, predicted=pred_ci, measured=meas_ci,
                 at_s0=at_s0, s_stars=s_stars,
                 predicted_values=[float(v) for v in pred],
                 measured_values=[float(v) for v in meas],
                 difference=float(meas_ci["mean"] - pred_ci["mean"]),
                 owner_figure=OWNER_LINEAGE_TAU,
                 owner_coupling=OWNER_LINEAGE_COUPLING,
                 owner_figure_is_ours=False,
                 difference_from_owner=float(meas_ci["mean"] - OWNER_LINEAGE_TAU),
                 barcode=barcode)

    # -- NEGATIVES ----------------------------------------------------------
    null_leg = []
    for s in S_GRID:
        Xn, ln = null_bed(NULL_SEEDS[0], sep=s, sigma=1.0)
        mn = bed_measure(Xn, ln)
        _, margin = criterion(mn, s, 1.0)
        mn.update(s=float(s), margin=float(margin))
        null_leg.append(mn)
    null_slope = slope_ci([p["s"] for p in null_leg], [p["S"] for p in null_leg])
    null_cross = critical_ratio(null_leg)
    if not is_refusal(null_cross) and null_slope["ci_lo"] <= 0.0 <= null_slope["ci_hi"]:
        null_cross = Refusal("ENTROPY_INERT",
                             "the entropy does not respond to the control "
                             "parameter, so the sign change is the trivial one "
                             "every bed with positive binding has as tau -> 0")
    plant_slope = slope_ci([p["s"] for p in fam_leg], [p["S"] for p in fam_leg])

    g = KNOWN_LINE_G
    known = fit_line("linear", [v / g for v in tr_T], tr_sep)
    known_line = dict(g=float(g), expected_c=float(g * fits["linear"]["c"]),
                      c=known["c"], ci_lo=known["ci_lo"], ci_hi=known["ci_hi"],
                      n=known["n"])

    # PAIRED across shared seeds. The marginal CI at each k is dominated by the
    # seed-to-seed spread of s*, which is common to every k and cancels in the
    # difference; comparing two marginal CIs throws that cancellation away and
    # calls a real trend "overlapping".
    k_beds = [_k_control(train, n_per=n) for n in K_BED_SIZES]
    k_gaming = dict(k_beds[0])
    k_gaming["beds"] = k_beds
    k_gaming["second_bed"] = k_beds[1]
    k_gaming["corroborated"] = bool(
        k_beds[0]["tracks_k"] and k_beds[1]["tracks_k"]
        and (k_beds[0]["frac_move"] < 0) == (k_beds[1]["frac_move"] < 0))

    negatives = dict(
        null_bed=dict(critical_ratio=null_cross, dS_ds=null_slope["slope"],
                      dS_ds_ci_lo=null_slope["ci_lo"],
                      dS_ds_ci_hi=null_slope["ci_hi"],
                      U_mean=float(np.mean([p["U"] for p in null_leg])),
                      S_mean=float(np.mean([p["S"] for p in null_leg])),
                      n_edges=int(np.mean([p["n_edges"] for p in null_leg]))),
        planted_bed=dict(dS_ds=plant_slope["slope"],
                         dS_ds_ci_lo=plant_slope["ci_lo"],
                         dS_ds_ci_hi=plant_slope["ci_hi"]),
        known_line=known_line, k_gaming=k_gaming)

    return dict(tcrit=tcrit, freeze=freeze, landau=landau, tlife=tlife,
                negatives=negatives, scale=scale,
                identity_probe=dict(seed=ip_seed, sep=ip_sep, sigma=ip_sigma,
                                    k=K, alpha=COUPLING_COLD, U=ipm["U"],
                                    S=ipm["S"], n_edges=ipm["n_edges"],
                                    n_refused=ipm["n_refused"]),
                seconds=float(time.time() - t0))


# ---------------------------------------------------------------------------
# 7. DEMO
# ---------------------------------------------------------------------------

def demo():
    r = report()
    p = print

    p("=" * 78)
    p("A3  THE PHASE CRITERION: BINDING VS THERMAL")
    p("=" * 78)

    p("\n1. UNITS. Is binding > thermal a criterion, or a fitted threshold?")
    for name in sorted(UNITS):
        p("     %-22s [%s]" % (name, UNITS[name]))
    p("   FOUR-FACTOR FORM AS WRITTEN: binding L^%d vs thermal L^%+d"
      % (FOUR_FACTOR_DIMENSIONS["binding_exponent"],
         FOUR_FACTOR_DIMENSIONS["thermal_exponent"]))
    p("   " + FOUR_FACTOR_DIMENSIONS["verdict"])
    s = r["scale"]
    p("   SCALE INVARIANCE (why the linear form is forced, not fitted):")
    p("     rescale x10 at fixed seed: max|10*X1 - X2| = %.3e" % s["dX"])
    p("     |dU| = %.17g   |dS| = %.17g   over %d edges, %d refused"
      % (s["dU"], s["dS"], s["n_edges"], s["n_refused"]))
    p("     curvature.two_cluster_bed under the same rescale: %.4f  (NOT scale "
      "invariant -- hardcoded lengths; that bed cannot carry this line)"
      % s["shipped_dX"])

    p("\n2. T-CRIT. Critical line on the TRAIN grid, all three forms, same data.")
    p("   train legs (T, seed, s*):")
    for T, seed, st in r["tcrit"]["s_star_by_leg"]:
        p("     T=%-5.2f seed=%d  s* = %s"
          % (T, seed, ("%.4f" % st) if not isinstance(st, str) else st))
    p("   %-10s %-9s %-22s %-11s %-9s %s"
      % ("form", "c", "95% CI", "RSS", "AIC", "resid RMS"))
    for name in ("linear", "quadratic", "constant"):
        f = r["tcrit"]["fits"][name]
        p("   %-10s %-9.4f [%8.4f, %8.4f] %-11.6f %-9.4f %.6f"
          % (name, f["c"], f["ci_lo"], f["ci_hi"], f["rss"], f["aic"],
             f["resid_rms"]))
    p("   WINNER: %s   (n = %d points, %d legs refused)"
      % (r["tcrit"]["winner"], r["tcrit"]["fits"]["linear"]["n"],
         r["tcrit"]["n_refused_legs"]))
    lin = r["tcrit"]["fits"]["linear"]
    for wrong in ("quadratic", "constant"):
        f = r["tcrit"]["fits"][wrong]
        p("     sep = c*T beats sep = %-9s : RSS ratio %.1f, AIC gap %.2f"
          % (wrong, f["rss"] / lin["rss"], f["aic"] - lin["aic"]))

    p("\n3. PRE-REGISTRATION. Digest over INPUTS, then the held-out grid.")
    p("   SHA-256 over (bed, instrument, sweep points, measured values) = %s"
      % r["freeze"]["digest"])
    p("   source pre-registers                                     = %s"
      % FROZEN_DIGEST)
    p("   freeze check: ok=%s  %s"
      % (r["freeze"]["check"]["ok"], r["freeze"]["check"]["reason"]))
    h = r["tcrit"]["heldout"]
    p("   HELD OUT: T in %s, seeds %s -- disjoint from train T %s, seeds %s"
      % (h["temperatures"], h["seeds"], r["tcrit"]["train_temperatures"],
         r["tcrit"]["train_seeds"]))
    p("   held-out data implies c = %.4f; frozen CI is [%.4f, %.4f] -> %s"
      % (h["c_implied"], lin["ci_lo"], lin["ci_hi"],
         "INSIDE" if lin["ci_lo"] <= h["c_implied"] <= lin["ci_hi"] else "OUTSIDE"))

    p("\n4. LANDAU. F = E - T*S, E = -U, over the measured family at T=1.0.")
    p("   %-6s %-9s %-9s %-9s %s" % ("s", "U", "S", "q", "n_edges"))
    for q in r["landau"]["family"]:
        p("   %-6.2f %-+9.4f %-9.4f %-9.4f %d (%d refused)"
          % (q["s"], q["U"], q["S"], q["q"], q["n_edges"], q["n_refused"]))
    p("   %-7s %-10s %-11s %s" % ("tau", "argmin s", "F_min", "local minima (s, F)"))
    for row in r["landau"]["ladder"]:
        p("   %-7.2f %-10.2f %-+11.4f %s"
          % (row["tau"], row["argmin_s"], row["F_min"],
             "  ".join("(%.2f, %+.4f)" % m for m in row["minima"])))
    L = r["landau"]
    p("   minimum crosses the critical ratio between s = %.2f and s = %.2f at "
      "tau = %.4f" % (L["jump_from"], L["jump_to"], L["tau_jump"]))
    p("   most local minima at any temperature: %d;  shallowest well separation "
      "%s against seed noise %.6f"
      % (L["n_wells_max"],
         "none (single well)" if not np.isfinite(L["well_gap"])
         else "%.6f" % L["well_gap"], L["well_noise"]))
    p("   TRANSITION ORDER: %s" % L["transition_order"].upper())
    p("   " + L["order_verdict"])
    p("   criterion crossing tau = 1/c = %.4f; Landau jump tau = %.4f; gap %+.4f"
      % (L["tau_crit_from_criterion"], L["tau_jump"], L["gap_to_criterion"]))

    p("\n5. T-LIFE. Binding budget against the measured death, coupling = %g."
      % r["tlife"]["coupling"])
    t = r["tlife"]
    p("   predicted tau_death = U/S at s = %.1f : %.4f  CI [%.4f, %.4f]  n=%d"
      % (t["s0"], t["predicted"]["mean"], t["predicted"]["ci_lo"],
         t["predicted"]["ci_hi"], t["predicted"]["n"]))
    p("   measured  tau_death = 1/s*          : %.4f  CI [%.4f, %.4f]  n=%d"
      % (t["measured"]["mean"], t["measured"]["ci_lo"], t["measured"]["ci_hi"],
         t["measured"]["n"]))
    p("   difference measured - predicted     : %+.4f" % t["difference"])
    p("   owner's lineage figure (HIS run, coupling %g): %.2f -- difference %+.4f"
      % (t["owner_coupling"], t["owner_figure"], t["difference_from_owner"]))
    b = t["barcode"]
    p("   H0 barcode route: REFUSED [%s] %s" % (b["death"].code, b["death"].reason))
    p("     %-6s %-10s %-8s %-10s %s"
      % ("s", "death", "rank", "med link", "death / med link"))
    for row in b["rows"]:
        p("     %-6.2f %-10.4f %-8s %-10.4f %.4f"
          % (row["s"], row["death"], "%d/%d" % (row["rank"], b["n_links"]),
             row["within_scale"], row["ratio"]))
    p("     the join is link %d of %d at its earliest: single linkage sees ONE "
      "component, at every separation the curvature resolves cleanly"
      % (b["rank_min"], b["n_links"]))
    p("     " + b["wanted_interface"])

    p("\n6. PLANTED NEGATIVES.")
    n = r["negatives"]["null_bed"]
    p("   structureless bed: U = %.4f, S = %.4f over %d edges"
      % (n["U_mean"], n["S_mean"], n["n_edges"]))
    p("     dS/ds = %+.5f  CI [%+.5f, %+.5f]  -> contains zero, entropy inert"
      % (n["dS_ds"], n["dS_ds_ci_lo"], n["dS_ds_ci_hi"]))
    p("     critical ratio: REFUSED [%s] %s"
      % (n["critical_ratio"].code, n["critical_ratio"].reason))
    pl = r["negatives"]["planted_bed"]
    p("   planted bed:      dS/ds = %+.5f  CI [%+.5f, %+.5f]  -> excludes zero"
      % (pl["dS_ds"], pl["dS_ds_ci_lo"], pl["dS_ds_ci_hi"]))
    kl = r["negatives"]["known_line"]
    p("   known planted line (temperature scaled by %.1f): expect c = %.4f, "
      "recovered %.4f CI [%.4f, %.4f] -> %s"
      % (kl["g"], kl["expected_c"], kl["c"], kl["ci_lo"], kl["ci_hi"],
         "RECOVERED" if kl["ci_lo"] <= kl["expected_c"] <= kl["ci_hi"] else "MISSED"))
    g = r["negatives"]["k_gaming"]
    for bed in g["beds"]:
        p("   k-gaming control at %d points per phase (%d in the cloud):"
          % (bed["n_per"], bed["n_points"]))
        for row in bed["rows"]:
            p("     k=%-3d s* = %.4f  CI [%.4f, %.4f]  over %d seeds, %d edges"
              % (row["k"], row["s_star"], row["ci_lo"], row["ci_hi"],
                 row["n_seeds"], row["n_edges"]))
        p("     paired s*(k=%d) - s*(k=%d) over %d shared seeds: %+.4f "
          "CI [%+.4f, %+.4f] -> %s zero;  monotone in k: %s;  move %.1f%%"
          % (bed["rows"][0]["k"], bed["rows"][-1]["k"], bed["paired_n"],
             bed["paired_mean"], bed["paired_ci_lo"], bed["paired_ci_hi"],
             "EXCLUDES" if bed["paired_excludes_zero"] else "contains",
             bed["monotone"], 100.0 * bed["frac_move"]))
    p("   the two bed sizes %s each other: both track k, same sign"
      % ("CORROBORATE" if g["corroborated"] else "DO NOT corroborate"))
    p("   WITHDRAWN: an earlier report of this work cited a corroborating "
      "percentage from a probe that lived outside this tree. No run, module or "
      "test here produced it. It is withdrawn, and the figure is deliberately "
      "NOT restated -- a withdrawn number quoted back is the same unbindable "
      "number wearing an apology. The second bed size above replaces it and "
      "every digit of that replacement is printed here.")
    p("   " + g["verdict"])


    p("\n   identity probe: seed %d, sep %.1f, sigma %.1f, k %d, alpha %g -> "
      "U = %.10f over %d edges (%d refused)"
      % (r["identity_probe"]["seed"], r["identity_probe"]["sep"],
         r["identity_probe"]["sigma"], r["identity_probe"]["k"],
         r["identity_probe"]["alpha"], r["identity_probe"]["U"],
         r["identity_probe"]["n_edges"], r["identity_probe"]["n_refused"]))
    cov = docstring_numbers()
    p("\n   docstring coverage: %d docstrings defined in this module, %d numbers "
      "in them, %d of those decimals"
      % (cov["n_docstrings"], cov["n_numbers"], cov["n_decimals"]))
    p("   sweep wall time %.1f s" % r["seconds"])

    # -- self-checks --------------------------------------------------------
    assert UNITS["binding"] == UNITS["thermal"] == "1"
    assert s["dU"] == 0.0 and s["dS"] == 0.0, "the bed is not scale invariant"
    assert s["shipped_dX"] > 1e-6
    assert lin["rss"] < r["tcrit"]["fits"]["quadratic"]["rss"]
    assert lin["rss"] < r["tcrit"]["fits"]["constant"]["rss"]
    assert r["tcrit"]["winner"] == "linear"
    assert r["freeze"]["digest"] == freeze_digest(r["freeze"]["spec"])
    assert freeze_digest(perturb_one_measured_value(r["freeze"]["spec"], 1e-9)) \
        != r["freeze"]["digest"]
    assert is_refusal(n["critical_ratio"])
    assert n["dS_ds_ci_lo"] <= 0.0 <= n["dS_ds_ci_hi"]
    assert pl["dS_ds_ci_hi"] < 0.0
    assert kl["ci_lo"] <= kl["expected_c"] <= kl["ci_hi"]
    assert is_refusal(b["death"])
    assert np.isfinite(L["tau_jump"]) and L["jump_from"] != L["jump_to"]
    assert L["transition_order"] in ("first", "second")
    assert (L["transition_order"] == "first") == L["wells_exceed_noise"]
    assert b["rank_min"] >= 1 and b["rank_max"] <= b["n_links"]
    assert cov["n_docstrings"] >= 15 and cov["n_decimals"] >= 20, (
        "the docstring scan has shrunk to %d docstrings / %d decimals; a scope "
        "regression reports zero missing and passes"
        % (cov["n_docstrings"], cov["n_decimals"]))
    assert abs(L["tau_crit_from_criterion"] - 1.0 / lin["c"]) < 1e-12
    p("\nALL SELF-CHECKS PASSED")


if __name__ == "__main__":
    sys.exit(demo())

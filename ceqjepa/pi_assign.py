"""DR-2 R1 and R3: assign every latent coordinate a corner, or refuse it.

WHAT IS CLAIMED. For each coordinate d of an encoder's representation, fit the
scaling exponent alpha_d from a context-length sweep, ||s_d(n)|| proportional to
n^alpha_d over n in {16, 32, 64, 128}. alpha near 0 is INTENSIVE and takes the
softmax corner beta=1; alpha near 1 is EXTENSIVE and takes the linear corner
beta=0; anything else is REFUSED as a value carrying MIXED-SCALING, and a
coordinate whose norm is not a power law at all is refused under a DIFFERENT
reason, NOT-A-POWER-LAW, because an exponent fitted to something that is not a
power law is a number with no referent. The mask is frozen after the sweep
against a digest over the sweep INPUTS, and is emitted as a vector along a NAMED
axis. Only dimensionless arguments enter exp.

WHAT IS NOT CLAIMED. Nothing here is a statement about a trained model. The
encoder is synthetic and every coordinate's class is planted by construction,
because R2, the JEPA build, is a later iteration and no trained encoder exists to
read. The numbers below are properties of this bed and of this instrument.
Dimensional consistency itself is not claimed as new -- see PRIOR ART.

WHERE THE RULE COMES FROM, AND IT IS NOT A HEURISTIC. The containment theorem in
this repository (READ lean/CEQ/V16Domain.lean:433 three_corners_containment, and
:445 corners_are_distinct) puts softmax and linear attention at two settings of
one operator, W = e^l / Z^beta. Carry the units through it. A sum over n tokens
carries the token count N, so Z = sum_k e^(l_k) carries N^1 and W carries
N^(-beta); a read sum_j W_j v_j therefore carries N^(1-beta), which is a MEAN at
beta=1 and a TOTAL at beta=0. The exponent of the count in the output IS 1-beta,
and the exponent of the count in the input IS alpha, so the assignment
beta = 1 - alpha is FORCED by the bookkeeping, and the corners are the only two
settings at which the output is a mean or a total rather than a fractional power
of a count, which is neither. That is why MIXED-SCALING is refused rather than
rounded. Measured over a beta sweep on this bed, alpha_out - (1-beta) is at worst
0.0093 across beta in {0.00, 0.25, 0.50, 0.75, 1.00}.

PRIOR ART, AND THE DELTA STATED IN ITS TERMS. The nearest work is Bakarji,
Callaham, Brunton and Kutz, "Dimensionally Consistent Learning with Buckingham
Pi", arXiv:2202.04643, Nature Computational Science 2022. BuckiNet learns one
exponent per input coordinate and enforces dimensionlessness by a hard null-space
constraint, a soft null-space loss, or a pre-generated candidate set. TWO THINGS
ARE DIFFERENT HERE, and neither is dimensional consistency. First, its exponent
matrix is ONE SHARED FIRST LAYER: every coordinate passes through the same
component, whereas the rule here hands each coordinate a DIFFERENT corner of the
operator family, so the model component is per coordinate. Second, those
exponents are FITTED against a null space, whereas alpha here is MEASURED from a
context-length sweep, and the coordinates fitting neither corner are REFUSED
rather than assigned. Searched and not found across that paper and Barbero et al.
arXiv:2406.04267 and arXiv:2511.20038: a per-coordinate scaling exponent measured
from a context-length sweep, such an exponent used to pick between attention
variants per coordinate, and a refusal for an unassignable coordinate. That
characterisation of the three papers is RELAYED from this round's prior-art pass
and is NOT verified in this module; it is stated so a reader checks it rather
than takes it. Bakarji's own stated shortcoming bears directly on the refusal
here: Buckingham pi does not give a UNIQUE set of dimensionless groups, so "make
it dimensionless" presumes a choice the theorem does not determine. THE CHOICE
HERE IS MINE AND IT IS THIS: the base dimensions are V, U and N -- value, second
value, and token COUNT -- and the two logits are a z-score of the value channel,
V over V, and a log ratio of the second channel to its own mean, U over U.
Another basis gives other groups and this module does not claim otherwise.

R-SQUARED IS THE WRONG TEST AND THIS BED SHOWS IT. R^2 measures residual against
the SPREAD of the data, and an intensive coordinate has no spread by definition,
so a flat power law scores badly for being flat: the three planted intensive
coordinates read R^2 of 0.9567, 0.9104 and 0.9416 while being exactly the power
laws they are planted as. A design refusing at R^2 < 0.95 would refuse 2 of the 8
planted power laws on this bed. The test used instead is chi-square against the
MEASURED sampling error of each norm, 2 degrees of freedom, critical value 20.0,
which is p = 4.540e-05 per coordinate and 4.085e-04 over the 9 coordinates
tested. The planted non-power-law reads chi-square 7.827e+06 against the eight
power laws' worst of 4.5849.

THE FLOOR, IN CLOSED FORM, BECAUSE FOUR POINTS IS A THIN FIT. At four log-spaced
points a power law is a straight line in log-log and the whole departure lives in
the 2 residual degrees of freedom. A quadratic departure in log2 units leaves
residuals q*(+1, -1, -1, +1) after the line is removed, so the smallest curvature
refusable at chi-square 20.0 is q_floor = sqrt(20.0/4) * se = 2.2361*se, with se
the log2 standard error of one sweep point. A planted logarithm, which is NOT a
power law, carries curvature -0.000375 against its own floor of 0.001866: it
PASSES the power-law test, is ASSIGNED the softmax corner at alpha 0.0387, and
this instrument cannot say otherwise. That is the measured detection floor and it
is the honest answer to what four points buy. The thinness shows in the intervals
too -- the Student-t interval on 2 residual degrees of freedom runs from -3.3460
to 3.3459 on the non-power-law coordinate, which is the fit saying so itself.

THE ATTACK, WHICH FIRES. RMS normalisation over a representation whose norm grows
with n subtracts that growth from every exponent at once. On this bed the total
norm reads alpha 0.9982 before and 0.0000 after; all 3 planted extensive
coordinates are SILENTLY reclassified INTENSIVE, and the 3 intensive ones are
driven to alpha near -1 and refused loudly. The rule then measures the
normalizer, not the data. The census that detects it runs inside assign() and
replaces every reachable corner with a Refusal: leg 1 is the pinned total norm,
exact to 0.00e+00 relative spread after normalisation against 8.74e-01 before;
leg 2 needs no access to the layer at all and is the one that generalises -- the
disagreement between each coordinate's DECLARED count exponent and its MEASURED
alpha, over the 8 coordinates that passed the power-law test (osc_shape
excluded, because an exponent with no referent has no offset to contribute), has
median 0.0031 with IQR 0.0314 on the clean bed and median 1.0024 with IQR 0.0271
after, and a common nonzero offset across coordinates is a property of a layer
and not of any data.

WHAT THE REFUSAL DOES, EXACTLY, BECAUSE ADVISORY AND STRUCTURAL ARE NOT THE SAME
WORD. assign() returns a dict, not a Refusal. When the census fires it sets
verdict AND replaces beta on every returned Assignment with that Refusal, so a
caller who reads assignments and never looks at verdict still cannot obtain a
corner: arithmetic on a Refusal raises TypeError rather than passing the way a
NaN does. On the attacked bed 0 of the 9 coordinates hand back a float. The
exponents stay populated because they are the diagnosis, and rule_only carries
what the rule alone said, because what the rule WOULD have returned is the
finding. Before this was structural it was a verdict beside nine populated
assignments, and a naive read returned beta 1.0 INTENSIVE for all 3 planted
extensive coordinates with no obstacle -- which is the exact wrong answer the
census exists to prevent.

A PROPOSED TIEBREAK, TREATED AS A HYPOTHESIS AND KILLED HERE. It was put that
where declared and measured disagree and the DECLARED exponent is a corner, the
declared value should be preferred -- which would turn half_walk's refusal into
the linear corner. What kills it is a coordinate declaring a corner whose
correct corner is another, and one exists: sqrt_total, a sum over the first
ceil(sqrt(n)) tokens, which is what a sparse attention on a sublinear budget
computes. It declares N^1 like any sum and measures 0.5152 against the probe's
walk at 0.4890, a difference of 0.0262, so the tiebreak reads identical inputs
for both; but sqrt_total is genuinely a total over a SUBLINEAR count, and the
linear corner the tiebreak would hand it misses by 0.4848. One rule cannot be
right for both. THE REFUSAL STANDS, and what ships instead is the diagnosis
already printed per coordinate, so a consumer with its own target decides with
the evidence rather than having this module decide blind. That a corner exists
for such a coordinate at all is in any case a statement about the coordinate
ALONE and not about it inside a model.

LIMIT: a normalizer dividing by an independent noisy estimate of n pins no norm
and is caught only by leg 2, which needs declared units; against an encoder that
declares none, this census is blind.

THE MAX-SUBTRACTION TRICK IS UNIT-SAFE ONLY AT beta=1. Subtracting the row max
from the logits multiplies the read by e^(-max), and the max of n logits grows
with n. At beta=1 that factor cancels in Z and the read is unchanged; at beta=0
there is no Z to cancel it and the read's exponent falls from 1.0089 to 0.6663,
which is neither corner and would be refused as MIXED-SCALING, while beta=1 stays
at 0.0082. The stabilisation every implementation applies is therefore a change
of measurement at every beta but one, and this module does not apply it.

THE MASK'S AXIS IS NAMED BECAUSE THE MISTAKE IS LIVE. beta in the shipped arm is
per LAYER, a 0-dim parameter (READ ceq/arm_smprime.py:527) applied to a sum over
the last axis raised to beta (READ :250) with the value contraction after it
(READ :264), and a shaped beta trips a hard assertion on the empty shape (READ
tests/gate0/test_g13_beta_learnable.py:130). None of that is this module's to
change. What IS this module's is that a beta of shape [S] would be silently
accepted and WRONG, because the exponent would then index the key rather than the
query, so the mask is emitted by frozen_mask() as a vector along the axis named
"D", of length 9, carrying its axis and its coordinate names beside it and never
a bare list. That name is pinned by a LITERAL in this module's own test and is
compared against the name written here, because comparing the emitted axis to
the constant that produced it is true under any rename, and because a consumer
in another file repeating the check is a pin that vanishes with the consumer.

RUN: python -m ceqjepa.pi_assign
"""

import hashlib
import json
import math
import platform
import re as _re
import subprocess
import sys
import time
from collections import OrderedDict, namedtuple
from pathlib import Path

import numpy as np

from . import curvature as cv

__all__ = [
    "N0", "N_GRID", "N_REPLICATES", "N_BLOCKS", "SEED", "REFIT_SEED",
    "ALPHA_TOL", "CHI2_CRIT", "R2_NAIVE_MIN", "T_CRIT_2DOF",
    "BASE_DIMS", "COORD_NAMES", "COORD_UNITS", "DECLARED_N", "PLANT",
    "Refusal", "is_refusal", "REASON_MIXED", "REASON_NOT_POWER_LAW",
    "REASON_NORMALIZER", "REASON_DIMENSIONFUL", "REASON_NO_ALPHA",
    "units_sum", "is_dimensionless", "safe_exp", "LOGITS", "DIMENSIONFUL_LOGIT",
    "draw_tokens", "encode_batch", "rmsnorm",
    "sweep_measurements", "coordinate_norms", "total_norms",
    "PowerFit", "fit_power_law", "Assignment", "assign", "strip_alpha", "struck",
    "read_at_beta", "output_alpha", "beta_sweep", "theorem_hypotheses",
    "baselines", "normalizer_census", "normalizer_attack",
    "sweep_spec", "mask_digest", "check_mask", "FROZEN_MASK_DIGEST",
    "tiebreak_probe",
    "perturb_one_measured_norm", "planted_refit", "frozen_mask", "MASK_AXIS",
    "PRIOR_ART",
    "provenance", "docstring_numbers", "report",
]

# ---------------------------------------------------------------------------
# THE PINNED BED AND INSTRUMENT
# ---------------------------------------------------------------------------

N0 = 16                          #: shortest context in the sweep
N_GRID = (16, 32, 64, 128)       #: n0, 2*n0, 4*n0, 8*n0 -- four points, by spec
N_REPLICATES = 1024              #: sequences per sweep point; the norm is an RMS over them
N_BLOCKS = 32                    #: blocks the replicates split into, for the sampling error
SEED = 3301
REFIT_SEED = 7717                #: the planted refit's seed. NEVER the frozen one.

#: Half-width of each corner. alpha within this of 0 is intensive, within this of
#: 1 is extensive. PINNED at the value the brief fixes.
ALPHA_TOL = 0.1

#: The power-law test. chi-square on 2 residual degrees of freedom against the
#: MEASURED sampling error, not against the spread of the data -- see the module
#: docstring for why R^2 cannot do this job on a flat coordinate. 20.0 is
#: p = 4.54e-05 per coordinate, 4.09e-04 over the 9 tested.
CHI2_CRIT = 20.0

#: Reported for the control ONLY: the threshold a naive R^2 design would use, and
#: the count of planted power laws it would wrongly refuse is printed beside it.
R2_NAIVE_MIN = 0.95

T_CRIT_2DOF = 4.302652729911275  #: Student t, 0.975, 2 dof. Four points, two spent on the line.
Z_NORMAL = 1.959963984540054     #: 0.975 normal quantile, for the se-based interval

VALUE_MEAN = 1.0                 #: token value channel, unit V
VALUE_SD = 0.3
U_MEAN = 2.0                     #: second channel, unit U, so a ratio has somewhere to come from
U_SD = 0.4
THRESH = 1.15                    #: the threshold the probability and the count are taken above
OSC_BASE = 2.0                   #: the non-power-law plant: a symmetric bump in log n
OSC_AMP = 1.5
LOG_BASE = 1.0                   #: the floor plant: a logarithm, which is not a power law
LOG_COEF = 0.04
JITTER = 0.02                    #: multiplicative scatter on the two planted shapes

BASE_DIMS = ("V", "U", "N")      #: value, second value, token COUNT -- count is a dimension here
                                 #: for the same reason amount of substance is one in SI: a total
                                 #: and a mean differ by exactly one power of it.

#: The nearest prior work, printed by the run so the citation is bound to it and
#: not discovered by a reviewer. The characterisation is RELAYED from this
#: round's prior-art pass and is not verified in this module.
PRIOR_ART = (
    "Bakarji, Callaham, Brunton, Kutz, Dimensionally Consistent Learning with "
    "Buckingham Pi, arXiv:2202.04643, Nature Computational Science 2022 -- one "
    "exponent per input coordinate through ONE SHARED first layer, FITTED "
    "against a null space. Here the exponent is MEASURED from a context-length "
    "sweep, the model component it selects is PER COORDINATE, and a coordinate "
    "fitting neither corner is REFUSED. Dimensional consistency itself is NOT "
    "claimed as new. Also searched: Barbero et al. arXiv:2406.04267 and "
    "arXiv:2511.20038; neither measures a per-coordinate scaling exponent from "
    "a context-length sweep nor refuses an unassignable coordinate. RELAYED, "
    "not verified here.")

Refusal = cv.Refusal
is_refusal = cv.is_refusal

REASON_MIXED = "MIXED-SCALING"
REASON_NOT_POWER_LAW = "NOT-A-POWER-LAW"
REASON_NORMALIZER = "NORMALIZER-CARRIES-N"
REASON_DIMENSIONFUL = "DIMENSIONFUL-EXPONENT"
REASON_NO_ALPHA = "ASSIGNED-WITHOUT-ALPHA"

COORD_NAMES = ("mean_value", "prob_above", "ratio_vu",
               "count_above", "sum_value", "total_u",
               "half_walk", "osc_shape", "log_shape")

#: Unit exponents of each coordinate over BASE_DIMS. A coordinate may be
#: dimensionful -- only a LOGIT may not.
COORD_UNITS = {
    "mean_value":  {"V": 1, "U": 0, "N": 0},
    "prob_above":  {"V": 0, "U": 0, "N": 0},
    "ratio_vu":    {"V": 1, "U": -1, "N": 0},
    "count_above": {"V": 0, "U": 0, "N": 1},
    "sum_value":   {"V": 1, "U": 0, "N": 1},
    "total_u":     {"V": 0, "U": 1, "N": 1},
    "half_walk":   {"V": 1, "U": 0, "N": 1},
    "osc_shape":   {"V": 1, "U": 0, "N": 0},
    "log_shape":   {"V": 1, "U": 0, "N": 0},
}
DECLARED_N = {k: float(v["N"]) for k, v in COORD_UNITS.items()}

#: (planted class, planted alpha or None). The ground truth is CONSTRUCTION, not
#: a label: a mean, a probability and a ratio cannot scale with n, a count, a sum
#: and a total must, a sum of zero-mean tokens grows like the square root, and the
#: last two are shapes that are not power laws at all.
PLANT = OrderedDict([
    ("mean_value",  ("INTENSIVE", 0.0)),
    ("prob_above",  ("INTENSIVE", 0.0)),
    ("ratio_vu",    ("INTENSIVE", 0.0)),
    ("count_above", ("EXTENSIVE", 1.0)),
    ("sum_value",   ("EXTENSIVE", 1.0)),
    ("total_u",     ("EXTENSIVE", 1.0)),
    ("half_walk",   ("MIXED", 0.5)),
    ("osc_shape",   ("NOT-A-POWER-LAW", None)),
    ("log_shape",   ("FLOOR", None)),
])


# ---------------------------------------------------------------------------
# 1. R3: UNITS, AND THE exp THAT REFUSES A DIMENSIONFUL ARGUMENT
# ---------------------------------------------------------------------------

def units_sum(units):
    """Sum of the unit exponents. Necessary to be zero, and NOT sufficient."""
    return int(sum(units.get(d, 0) for d in BASE_DIMS))


def is_dimensionless(units):
    """Every exponent zero. V^1 N^-1 SUMS to zero and is still a ratio of unlike
    things, so the sum test alone would let it into an exponential; this is the
    test that is enforced, and the sum is printed beside it."""
    return all(units.get(d, 0) == 0 for d in BASE_DIMS)


def safe_exp(x, units):
    """exp(x), or a Refusal if x is not dimensionless. The KILL, at the exp."""
    if not is_dimensionless(units):
        return Refusal(REASON_DIMENSIONFUL,
                       "exp of an argument carrying %s (exponents sum to %d): only a "
                       "pi group may enter an exponential"
                       % (" ".join("%s^%d" % (d, units.get(d, 0)) for d in BASE_DIMS),
                          units_sum(units)))
    return np.exp(x)


def _z_value(v, u):
    """Logit 1: the value channel as a z-score. V/V, a pi group."""
    return (v - v.mean(axis=1, keepdims=True)) / v.std(axis=1, keepdims=True)


def _log_ratio_u(v, u):
    """Logit 2: log of the second channel over its own mean. U/U, a pi group."""
    return np.log(u / u.mean(axis=1, keepdims=True))


#: The declared logits, with their unit exponents. Both are ratios of like
#: quantities, so every exponent is zero and the sum is zero.
LOGITS = OrderedDict([
    ("z_value",     dict(units={"V": 0, "U": 0, "N": 0}, build=_z_value)),
    ("log_ratio_u", dict(units={"V": 0, "U": 0, "N": 0}, build=_log_ratio_u)),
])

#: The struck candidate, kept so the kill is SEEN to fire: the raw value channel
#: as a logit carries V^1 and exp must refuse it.
DIMENSIONFUL_LOGIT = dict(name="raw_value", units={"V": 1, "U": 0, "N": 0})


# ---------------------------------------------------------------------------
# 2. THE SYNTHETIC ENCODER, WITH THE GROUND TRUTH PLANTED
# ---------------------------------------------------------------------------

def draw_tokens(n, replicates=N_REPLICATES, seed=SEED):
    """(value, second-channel, zero-mean) token streams, shape (replicates, n).

    One independent stream per sweep point, so no shared randomness can carry a
    spurious trend from one n to the next.
    """
    rng = np.random.default_rng([int(seed), int(n)])
    v = rng.normal(VALUE_MEAN, VALUE_SD, size=(replicates, n))
    u = rng.normal(U_MEAN, U_SD, size=(replicates, n))
    w = rng.normal(0.0, VALUE_SD, size=(replicates, n))
    j = rng.normal(0.0, JITTER, size=(replicates, 2))
    return v, u, w, j


def _osc(n):
    """The planted non-power-law: a symmetric bump in log2(n).

    Symmetric about the middle of the grid, so the OLS slope through the four
    sampled points is zero by construction and the naive rule would call it
    INTENSIVE and hand it the softmax corner. Only the power-law test saves it,
    which is why that test runs FIRST.
    """
    t = math.log2(n / N0)
    return OSC_BASE + OSC_AMP * math.cos(2.0 * math.pi * (t - 1.5) / 3.0)


def _logshape(n):
    """The planted floor: a logarithm. Not a power law, and below the curvature
    this bed can refuse."""
    return LOG_BASE + LOG_COEF * math.log(n / N0)


def encode_batch(n, replicates=N_REPLICATES, seed=SEED):
    """The representation S, shape (replicates, 9), one row per sequence."""
    v, u, w, j = draw_tokens(n, replicates, seed)
    hit = (v > THRESH)
    cols = [
        v.mean(axis=1),                                   # mean_value
        hit.mean(axis=1),                                 # prob_above
        v.mean(axis=1) / u.mean(axis=1),                  # ratio_vu
        hit.sum(axis=1).astype(float),                    # count_above
        v.sum(axis=1),                                    # sum_value
        u.sum(axis=1),                                    # total_u
        w.sum(axis=1),                                    # half_walk
        _osc(n) * (1.0 + j[:, 0]),                        # osc_shape
        _logshape(n) * (1.0 + j[:, 1]),                   # log_shape
    ]
    return np.stack(cols, axis=1)


def rmsnorm(S):
    """THE ATTACK. Divide every row by its own RMS: a normalization layer that
    hides n, because the row norm here is dominated by the extensive coordinates
    and therefore grows with n."""
    return S * math.sqrt(S.shape[1]) / np.linalg.norm(S, axis=1, keepdims=True)


def _rms(a, axis=0):
    return np.sqrt(np.mean(np.square(a), axis=axis))


def sweep_measurements(seed=SEED, n_grid=N_GRID, normalize=False):
    """The whole sweep: per-coordinate norms, their log2 sampling errors, and the
    norm of the whole representation.

    ||s_d(n)|| is the RMS of coordinate d over the replicate ensemble. The
    sampling error is estimated by splitting the replicates into N_BLOCKS blocks
    and taking the standard error of the per-block log2 norm: no distributional
    assumption, and it is what the power-law test is measured against.
    """
    norms = OrderedDict((k, np.zeros(len(n_grid))) for k in COORD_NAMES)
    ses = OrderedDict((k, np.zeros(len(n_grid))) for k in COORD_NAMES)
    totals = np.zeros(len(n_grid))
    for i, n in enumerate(n_grid):
        S = encode_batch(n, N_REPLICATES, seed)
        if normalize:
            S = rmsnorm(S)
        blocks = S.reshape(N_BLOCKS, S.shape[0] // N_BLOCKS, S.shape[1])
        per_block = np.log2(_rms(blocks, axis=1))         # (N_BLOCKS, 9)
        for d, name in enumerate(COORD_NAMES):
            norms[name][i] = float(_rms(S[:, d]))
            ses[name][i] = float(np.std(per_block[:, d], ddof=1) / math.sqrt(N_BLOCKS))
        totals[i] = float(_rms(np.linalg.norm(S, axis=1)))
    return tuple(n_grid), norms, ses, totals


def coordinate_norms(seed=SEED, n_grid=N_GRID, normalize=False):
    """(n grid, {coordinate: the four measured norms}). The two-value view the
    identity layer recomputes against."""
    grid, norms, _, _ = sweep_measurements(seed, n_grid, normalize)
    return grid, norms


def total_norms(seed=SEED, n_grid=N_GRID, normalize=False):
    """The four norms of the WHOLE representation. Leg 1 of the census reads these."""
    return sweep_measurements(seed, n_grid, normalize)[3]


# ---------------------------------------------------------------------------
# 3. THE FIT, ITS CI, ITS RESIDUAL, ITS R^2, AND THE POWER-LAW TEST
# ---------------------------------------------------------------------------

PowerFit = namedtuple(
    "PowerFit",
    "alpha ci_lo ci_hi ci_se_lo ci_se_hi resid_rms r2 chi2 curvature q_floor "
    "power_law_ok norms se")


def fit_power_law(n_grid, norms, se=None):
    """OLS of log2||s(n)|| on log2(n), with everything needed to distrust it.

    alpha is the plain OLS slope so that a caller with no error bars recomputes
    exactly the published number. ci_lo/ci_hi are the Student-t interval on 2
    residual degrees of freedom -- four points, two spent on the line -- and are
    therefore wide on purpose. ci_se_lo/ci_se_hi use the MEASURED sampling error
    instead and are tighter; chi2, the curvature and the floor need it and are
    NaN without it.
    """
    y = np.log2(np.asarray(norms, dtype=float))
    x = np.log2(np.asarray(n_grid, dtype=float))
    xc = x - x.mean()
    sxx = float(np.dot(xc, xc))
    alpha = float(np.dot(xc, y - y.mean()) / sxx)
    intercept = float(y.mean() - alpha * x.mean())
    resid = y - (alpha * x + intercept)
    dof = len(x) - 2
    resid_rms = float(np.sqrt(np.mean(np.square(resid))))
    s2 = float(np.dot(resid, resid) / dof)
    half_t = T_CRIT_2DOF * math.sqrt(s2 / sxx)
    sst = float(np.dot(y - y.mean(), y - y.mean()))
    r2 = float(1.0 - np.dot(resid, resid) / sst) if sst > 0 else float("nan")
    if se is None:
        chi2 = curvature = q_floor = float("nan")
        half_se = float("nan")
        ok = True
    else:
        sev = np.asarray(se, dtype=float)
        chi2 = float(np.sum(np.square(resid / sev)))
        # The 2 residual dof of a four-point line are exactly the quadratic and
        # the cubic; the quadratic is the departure a power law cannot have, and
        # its residual signature after the line is removed is (+1, -1, -1, +1)/4.
        curvature = float(np.dot(resid, np.array([1.0, -1.0, -1.0, 1.0])) / 4.0)
        q_floor = float(math.sqrt(CHI2_CRIT / 4.0) * float(np.mean(sev)))
        half_se = Z_NORMAL * float(math.sqrt(np.sum(np.square(sev * xc)))) / sxx
        ok = chi2 <= CHI2_CRIT
    return PowerFit(alpha=alpha, ci_lo=alpha - half_t, ci_hi=alpha + half_t,
                    ci_se_lo=alpha - half_se, ci_se_hi=alpha + half_se,
                    resid_rms=resid_rms, r2=r2, chi2=chi2, curvature=curvature,
                    q_floor=q_floor, power_law_ok=bool(ok),
                    norms=tuple(float(v) for v in norms),
                    se=tuple(float(v) for v in (se if se is not None else
                                                [float("nan")] * len(x))))


Assignment = namedtuple(
    "Assignment",
    "coord alpha ci_lo ci_hi ci_se_lo ci_se_hi resid_rms r2 chi2 curvature "
    "q_floor klass beta norms")


def _assign_one(name, fit):
    """The rule, in the order that matters.

    The power-law test runs FIRST: a coordinate whose norm is not a power law has
    no exponent to compare against a corner, and refusing it for having an
    exponent between the corners would be reporting the wrong finding about it.
    """
    if not fit.power_law_ok:
        beta = Refusal(REASON_NOT_POWER_LAW,
                       "chi-square %.4g on 2 dof against the measured sampling "
                       "error exceeds %.1f: the norm is not a power law in n and "
                       "the exponent %+.4f fitted to it has no referent"
                       % (fit.chi2, CHI2_CRIT, fit.alpha))
        klass = "REFUSED"
    elif abs(fit.alpha) <= ALPHA_TOL:
        beta, klass = 1.0, "INTENSIVE"
    elif abs(fit.alpha - 1.0) <= ALPHA_TOL:
        beta, klass = 0.0, "EXTENSIVE"
    else:
        beta = Refusal(REASON_MIXED,
                       "alpha %+.4f is %.4f from the softmax corner and %.4f from "
                       "the linear corner, both beyond %.2f: the read would carry "
                       "N^%+.4f, which is neither a mean nor a total"
                       % (fit.alpha, abs(fit.alpha), abs(fit.alpha - 1.0),
                          ALPHA_TOL, fit.alpha))
        klass = "REFUSED"
    return Assignment(coord=name, alpha=fit.alpha, ci_lo=fit.ci_lo, ci_hi=fit.ci_hi,
                      ci_se_lo=fit.ci_se_lo, ci_se_hi=fit.ci_se_hi,
                      resid_rms=fit.resid_rms, r2=fit.r2, chi2=fit.chi2,
                      curvature=fit.curvature, q_floor=fit.q_floor,
                      klass=klass, beta=beta, norms=fit.norms)


def strip_alpha(a):
    """The owner's struck sin, planted: an assignment with its exponent removed."""
    return a._replace(alpha=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"))


def struck(a):
    """A Refusal if this assignment carries no alpha, None if it is live.

    KILL: a coordinate assigned without its alpha is struck. The check runs in
    assign(), so a stripped assignment cannot reach a caller.
    """
    if not np.isfinite(a.alpha):
        return Refusal(REASON_NO_ALPHA,
                       "coordinate %s carries beta %r with alpha %r: an assignment "
                       "without its exponent is a corner nobody measured"
                       % (a.coord, a.beta, a.alpha))
    return None


def assign(seed=SEED, n_grid=N_GRID, normalize=False):
    """Fit every coordinate, assign or refuse it, and run the census AUTOMATICALLY.

    The census is not a function a reader remembers to call: assign() runs it on
    every sweep and returns a Refusal for the whole assignment when it fires,
    because on a normalized representation every individual exponent is a
    measurement of the normalizer and each one separately looks fine.
    """
    grid, norms, ses, totals = sweep_measurements(seed, n_grid, normalize)
    fits = OrderedDict((k, fit_power_law(grid, norms[k], ses[k])) for k in COORD_NAMES)
    out = OrderedDict((k, _assign_one(k, fits[k])) for k in COORD_NAMES)
    for name, a in out.items():
        dead = struck(a)
        if dead is not None:                       # unreachable unless alpha is lost
            raise AssertionError("%s: %r" % (name, dead))
    census = normalizer_census(fits, totals)
    rule_only = out
    verdict = None
    if census["fires"]:
        # STRUCTURAL, NOT ADVISORY. A verdict beside populated assignments is a
        # flag a caller can skip, and the caller who skips it gets a corner that
        # is wrong in exactly the way the census exists to catch. So the corners
        # a caller can REACH become Refusals: arithmetic on one raises TypeError
        # instead of passing silently. The exponents stay, because they are the
        # diagnosis, and rule_only keeps what the rule alone said, because what
        # the rule WOULD have returned is the finding.
        verdict = Refusal(REASON_NORMALIZER, census["verdict"])
        out = OrderedDict((k, a._replace(beta=verdict, klass="REFUSED"))
                          for k, a in rule_only.items())
    return dict(grid=grid, norms=norms, ses=ses, totals=totals, fits=fits,
                assignments=out, rule_only=rule_only, census=census,
                verdict=verdict)


# ---------------------------------------------------------------------------
# 4. THE CENSUS AGAINST A NORMALIZER THAT HIDES n
# ---------------------------------------------------------------------------

PIN_TOL = 1e-12          #: relative spread below which the total norm counts as PINNED
OFFSET_MIN = 0.25        #: a common offset smaller than this is not distinguishable from data
OFFSET_SPREAD_MAX = 0.25 #: and one with more spread than this is not COMMON


def normalizer_census(fits, totals):
    """Two legs, and the second is the one that generalises.

    Leg 1: the total norm is PINNED. An exact normalizer fixes ||s|| at every n,
    which no data does, and the relative spread over the sweep reads it directly.
    Leg 2: a COMMON OFFSET between each coordinate's declared count exponent and
    its measured alpha. A layer shifts every exponent by the same amount; data
    does not. Only coordinates that passed the power-law test contribute, because
    the others have no alpha with a referent to offset.
    LIMIT: a normalizer dividing by an independent noisy estimate of n pins no
    norm, so leg 1 misses it, and leg 2 needs an encoder that declares units.
    """
    tot = np.asarray(totals, dtype=float)
    spread = float((tot.max() - tot.min()) / tot.max()) if tot.max() > 0 else 0.0
    pinned = bool(spread <= PIN_TOL)
    offsets = [DECLARED_N[k] - f.alpha for k, f in fits.items() if f.power_law_ok]
    med = float(np.median(offsets)) if offsets else float("nan")
    iqr = float(np.percentile(offsets, 75) - np.percentile(offsets, 25)) if offsets else float("nan")
    common = bool(abs(med) > OFFSET_MIN and iqr < OFFSET_SPREAD_MAX)
    fires = bool(pinned or common)
    return dict(
        fires=fires, pinned=pinned, common_offset=common,
        total_spread=spread, offset_median=med, offset_iqr=iqr,
        n_offsets=len(offsets), reason=(REASON_NORMALIZER if fires else None),
        verdict=("total norm spread %.2e (pinned=%s) and declared-minus-measured "
                 "offset median %+.4f with IQR %.4f over %d coordinates: a common "
                 "shift of every exponent is a property of a layer, not of data"
                 % (spread, pinned, med, iqr, len(offsets))
                 if fires else
                 "total norm spread %.2e is not pinned and the declared-minus-"
                 "measured offset median %+.4f with IQR %.4f is inside +/-%.2f: "
                 "no normalizer detected over %d coordinates"
                 % (spread, med, iqr, OFFSET_MIN, len(offsets))))


def normalizer_attack(seed=SEED):
    """Insert the normalizer and report what it does to EVERY alpha."""
    clean = assign(seed=seed, normalize=False)
    dirty = assign(seed=seed, normalize=True)
    per = OrderedDict()
    silent = 0
    for name in COORD_NAMES:
        c = clean["assignments"][name]
        raw = dirty["rule_only"][name]        # what the RULE alone said
        d = dirty["assignments"][name]        # what a caller can actually reach
        if PLANT[name][0] == "EXTENSIVE" and c.klass == "EXTENSIVE" \
                and raw.klass == "INTENSIVE":
            silent += 1
        per[name] = dict(alpha_clean=c.alpha, alpha_attacked=raw.alpha,
                         class_clean=c.klass, class_attacked_rule_only=raw.klass,
                         class_attacked=d.klass)
    at = fit_power_law(clean["grid"], clean["totals"],
                       [1e-6] * len(clean["grid"]))
    ad = fit_power_law(dirty["grid"], np.maximum(dirty["totals"], 1e-300),
                       [1e-6] * len(dirty["grid"]))
    return dict(per_coordinate=per, silently_reclassified=silent,
                alpha_total_clean=at.alpha, alpha_total_attacked=ad.alpha,
                census_clean=clean["census"], census_attacked=dirty["census"],
                census_runs_inside_assign=True,
                rule_only_attacked=dirty["rule_only"],
                reachable_corners_attacked=sum(
                    1 for a in dirty["assignments"].values()
                    if not is_refusal(a.beta)),
                clean_verdict=clean["verdict"], attacked_verdict=dirty["verdict"])


# ---------------------------------------------------------------------------
# 5. THE READ, THE beta SWEEP, AND THE THEOREM'S HYPOTHESES
# ---------------------------------------------------------------------------

GATE = 0.0      #: g identically zero -- the hypothesis three_corners_containment needs

BETA_SWEEP = (0.0, 0.25, 0.5, 0.75, 1.0)


def read_at_beta(values, logits, beta, max_subtract=False):
    """sum_j (e^l_j / Z^beta) v_j, the containment family's read.

    The row max is NOT subtracted. That stabilisation multiplies the read by
    e^(-max), and max over n logits grows with n; at beta=1 the same factor
    appears in Z and cancels, at every other beta it does not, so applying it
    would change the exponent the sweep is measuring. It is available here only
    as the control that shows exactly that.
    """
    l = logits - logits.max(axis=1, keepdims=True) if max_subtract else logits
    e = safe_exp(l + GATE, LOGITS["z_value"]["units"])
    if is_refusal(e):
        return e
    num = np.sum(e * values, axis=1)
    if beta == 0.0:
        return num
    return num / np.power(np.sum(e, axis=1), beta)


def output_alpha(beta, seed=SEED, n_grid=N_GRID, max_subtract=False):
    """The scaling exponent of the READ itself, over the same sweep."""
    vals, ses = np.zeros(len(n_grid)), np.zeros(len(n_grid))
    for i, n in enumerate(n_grid):
        v, u, w, _ = draw_tokens(n, N_REPLICATES, seed)
        r = read_at_beta(v, _z_value(v, u), beta, max_subtract)
        vals[i] = float(_rms(r))
        blocks = r.reshape(N_BLOCKS, r.shape[0] // N_BLOCKS)
        per_block = np.log2(_rms(blocks, axis=1))
        ses[i] = float(np.std(per_block, ddof=1) / math.sqrt(N_BLOCKS))
    return fit_power_law(n_grid, vals, ses)


def beta_sweep(seed=SEED):
    """alpha_out against the DERIVED 1-beta, at five settings of the family."""
    rows = []
    for b in BETA_SWEEP:
        f = output_alpha(b, seed=seed)
        rows.append(dict(beta=b, predicted=1.0 - b, alpha_out=f.alpha,
                         gap=f.alpha - (1.0 - b), r2=f.r2, chi2=f.chi2,
                         ci_lo=f.ci_lo, ci_hi=f.ci_hi))
    return rows


def theorem_hypotheses(seed=SEED, n=64):
    """three_corners_containment's hypotheses, as numbers on THIS bed.

    READ lean/CEQ/V16Domain.lean:433 for the theorem and :445 for
    corners_are_distinct. The theorem is about an operator family; what it needs
    from the data is that the gate vanishes, the logits are finite, the beta=1
    corner is row-stochastic, and the two corners do not coincide here. A win
    against a theorem whose hypotheses were never checked on the bed is not a win.
    """
    v, u, w, _ = draw_tokens(n, N_REPLICATES, seed)
    l = _z_value(v, u)
    e = np.exp(l + GATE)
    z = np.sum(e, axis=1, keepdims=True)
    rows1 = np.sum(e / z, axis=1)
    rows0 = np.sum(e, axis=1)
    return dict(
        name="three_corners_containment",
        source="lean/CEQ/V16Domain.lean:433 (corners_are_distinct :445)",
        gate_is_zero=bool(GATE == 0.0),
        logits_finite=bool(np.all(np.isfinite(l))), n_logits=int(l.size),
        row_sum_beta1=float(np.mean(rows1)),
        row_sum_beta1_err=float(np.max(np.abs(rows1 - 1.0))),
        row_sum_beta0=float(np.mean(rows0)),
        corners_distinct=bool(abs(float(np.mean(rows0)) - 1.0) > ALPHA_TOL),
        n=int(n))


# ---------------------------------------------------------------------------
# 6. L-SIMPLE: ONE CORNER EVERYWHERE, AND A COIN
# ---------------------------------------------------------------------------

def _score(name, corner_of, assignments):
    """A coordinate is SERVED when the corner's output exponent 1-beta matches the
    coordinate's own alpha. That is the whole job: a read that turns a total into
    a mean has lost the count, and one that turns a mean into a total has invented
    one."""
    served = broken = refused = 0
    by_class = {}
    for coord, a in assignments.items():
        beta = corner_of(coord, a)
        cls = PLANT[coord][0]
        slot = by_class.setdefault(cls, dict(served=0, total=0))
        slot["total"] += 1
        if is_refusal(beta):
            refused += 1
        elif abs((1.0 - beta) - a.alpha) <= ALPHA_TOL:
            served += 1
            slot["served"] += 1
        else:
            broken += 1
    return dict(name=name, served=served, broken=broken, refused=refused,
                n_coords=len(assignments), by_class=by_class)


def baselines(assignments, seed=SEED):
    """The measured rule beside every simpler thing that could have been done."""
    rng = np.random.default_rng([int(seed), 99])
    coin = {k: float(rng.integers(0, 2)) for k in assignments}
    return OrderedDict([
        ("rule", _score("rule", lambda c, a: a.beta, assignments)),
        ("all_softmax", _score("all_softmax", lambda c, a: 1.0, assignments)),
        ("all_linear", _score("all_linear", lambda c, a: 0.0, assignments)),
        ("random", _score("random", lambda c, a: coin[c], assignments)),
    ])


# ---------------------------------------------------------------------------
# 7. THE FROZEN MASK: A DIGEST OVER THE INPUTS
# ---------------------------------------------------------------------------

_MASK_KEYS = ("encoder", "seed", "n_grid", "measured")

#: Filled from the first run and then frozen. Over the sweep INPUTS only -- a
#: digest covering the fitted exponents cannot tell a freeze from an honest
#: refit, because a refit re-digests and verifies against itself.
FROZEN_MASK_DIGEST = "854dc3aae168dd98e0e631e189aa30c0800cd8d1a61e69433b2b2ea21fad504a"

#: The axis the mask lives on. NOT "S". A beta of shape [S] is silently accepted
#: by a broadcast and is wrong, because the exponent then indexes the key rather
#: than the query, so the name travels with the vector.
MASK_AXIS = "D"


def sweep_spec(seed=SEED, n_grid=N_GRID, normalize=False):
    """Exactly the inputs the mask is frozen over: encoder spec, seed, n grid,
    measured norms. No fitted exponent appears anywhere in it."""
    grid, norms = coordinate_norms(seed, n_grid, normalize)
    return dict(
        encoder=dict(coords=list(COORD_NAMES), replicates=N_REPLICATES,
                     blocks=N_BLOCKS, value_mean=VALUE_MEAN, value_sd=VALUE_SD,
                     u_mean=U_MEAN, u_sd=U_SD, threshold=THRESH,
                     osc_base=OSC_BASE, osc_amp=OSC_AMP, log_base=LOG_BASE,
                     log_coef=LOG_COEF, jitter=JITTER, normalize=bool(normalize)),
        seed=int(seed),
        n_grid=[int(v) for v in grid],
        measured={k: [float(x) for x in v] for k, v in norms.items()},
    )


def mask_digest(spec):
    """A sha256 over the INPUT keys only; anything else in spec is ignored."""
    payload = {k: spec[k] for k in _MASK_KEYS}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def perturb_one_measured_norm(spec, delta):
    """A copy of spec with exactly one measured norm moved by delta."""
    out = json.loads(json.dumps({k: spec[k] for k in _MASK_KEYS}))
    first = COORD_NAMES[0]
    out["measured"][first][0] = out["measured"][first][0] + float(delta)
    return out


def check_mask(spec):
    """Does this spec's digest match the one pre-registered in the source?"""
    got = mask_digest(spec)
    if got != FROZEN_MASK_DIGEST:
        return dict(ok=False, digest=got,
                    reason=("input digest %s does not match the pre-registered %s: "
                            "this mask was frozen over other data"
                            % (got[:16], FROZEN_MASK_DIGEST[:16])))
    return dict(ok=True, digest=got, reason="input digest matches the pre-registration")


def tiebreak_probe(seed=SEED, n_grid=N_GRID):
    """THE PROPOSED TIEBREAK, TREATED AS A HYPOTHESIS AND KILLED ON THIS BED.

    The proposal: where a coordinate's DECLARED count exponent and its MEASURED
    alpha disagree and the declared value is a corner, prefer declared -- which
    would turn half_walk's MIXED-SCALING refusal into the linear corner. What
    would kill it is a coordinate whose declared exponent is a corner and whose
    correct corner is the other one, or none, because there the tiebreak assigns
    confidently and wrongly, which is worse than refusing.

    ONE EXISTS AND IT IS BUILT HERE. sqrt_total is a sum over the first
    ceil(sqrt(n)) tokens -- a top-k pool at k = sqrt(n), which is what a sparse
    attention over a sublinear budget computes. It is a SUM over the token axis,
    so it declares N^1 exactly as a zero-mean walk does, and its norm grows like
    the square root, so it measures 0.5152 against the probe's own walk at
    0.4890 (the same construction as the bed's half_walk, drawn on this probe's
    stream), a difference of 0.0262 -- well inside the 0.10 corner tolerance.
    THE TWO ARE INDISTINGUISHABLE ON EVERY INPUT THE TIEBREAK READS. But
    sqrt_total genuinely IS a total over a sublinear count: a read reproducing
    it must carry n^0.5152, which no corner provides, so refusing is the right
    answer for it, while the linear corner the tiebreak would hand it carries
    n^1 and misses by 0.4848 -- 4.8 times the corner tolerance.

    So one rule cannot be right for both, and nothing the tiebreak looks at tells
    them apart. THE REFUSAL STANDS. What ships instead is the diagnosis that was
    already there: the declared-minus-measured gap is printed per coordinate, so
    a consumer with its own target and its own loss can make that call with the
    evidence in hand rather than have this module make it blind.
    """
    rng = np.random.default_rng([int(seed), 4242])
    walk, sqrt_tot = np.zeros(len(n_grid)), np.zeros(len(n_grid))
    ses_w, ses_s = np.zeros(len(n_grid)), np.zeros(len(n_grid))
    for i, n in enumerate(n_grid):
        w = rng.normal(0.0, VALUE_SD, size=(N_REPLICATES, n))
        v = rng.normal(VALUE_MEAN, VALUE_SD, size=(N_REPLICATES, n))
        k = int(math.ceil(math.sqrt(n)))
        a, b = w.sum(axis=1), v[:, :k].sum(axis=1)
        walk[i], sqrt_tot[i] = float(_rms(a)), float(_rms(b))
        for arr, out in ((a, ses_w), (b, ses_s)):
            blk = arr.reshape(N_BLOCKS, arr.shape[0] // N_BLOCKS)
            out[i] = float(np.std(np.log2(_rms(blk, axis=1)), ddof=1)
                           / math.sqrt(N_BLOCKS))
    fw = fit_power_law(n_grid, walk, ses_w)
    fs = fit_power_law(n_grid, sqrt_tot, ses_s)
    proposed = 0.0                     # declared N^1 -> the linear corner
    return dict(
        walk_alpha=fw.alpha, sqrt_alpha=fs.alpha,
        alpha_difference=abs(fw.alpha - fs.alpha),
        declared_walk=1.0, declared_sqrt=1.0,
        indistinguishable=bool(abs(fw.alpha - fs.alpha) <= ALPHA_TOL),
        proposed_beta=proposed,
        sqrt_gap=abs((1.0 - proposed) - fs.alpha),
        gap_in_tolerances=abs((1.0 - proposed) - fs.alpha) / ALPHA_TOL,
        kills_the_tiebreak=bool(abs((1.0 - proposed) - fs.alpha) > ALPHA_TOL),
        verdict=("sqrt_total declares N^1 and measures %+.4f, within %.4f of "
                 "half_walk's %+.4f, so the tiebreak cannot tell them apart; it "
                 "would hand sqrt_total the linear corner, whose read carries "
                 "n^1 against a coordinate that scales as n^%.4f, missing by "
                 "%.4f. REFUSED as a rule; the MIXED-SCALING refusal stands and "
                 "the declared-minus-measured gap ships as a diagnosis instead."
                 % (fs.alpha, abs(fw.alpha - fs.alpha), fw.alpha, fs.alpha,
                    abs((1.0 - proposed) - fs.alpha))))


def planted_refit():
    """The struck sin, planted so the freeze is SEEN to fire: re-sweep at another
    seed and present the result as though it were the frozen mask."""
    spec = sweep_spec(seed=REFIT_SEED)
    return spec, mask_digest(spec)


def frozen_mask(assignments, digest):
    """The mask as a vector along a NAMED axis, never a bare list.

    Whoever wires R2 gets axis="D", length 9, and the coordinate names in the
    same order as beta, so a [D] mask cannot be mistaken for an [S] one. A
    refused coordinate carries None in beta and its reason in refusals: dropping
    the refusals and shipping only the assignable coordinates would silently
    change the length of the axis, which is the same mistake in a smaller coat.

    The axis name is checked against the literal "D" by this module's own test
    rather than against MASK_AXIS, because comparing the emitted axis to the
    constant that produced it is true under any rename; and the name written in
    this docstring is compared to the name the code emits, because a stale
    docstring beside renamed code otherwise ships green.
    """
    names = list(assignments)
    beta = [None if is_refusal(a.beta) else float(a.beta)
            for a in assignments.values()]
    return dict(
        axis=MASK_AXIS, axis_len=len(names), coords=names, beta=beta,
        alpha=[float(a.alpha) for a in assignments.values()],
        refusals={n: a.beta.code for n, a in assignments.items()
                  if is_refusal(a.beta)},
        n_assigned=sum(1 for b in beta if b is not None),
        digest=digest,
        note=("beta is indexed by the latent coordinate axis %r of length %d; a "
              "vector of length S would index the key position and be wrong"
              % (MASK_AXIS, len(names))))


# ---------------------------------------------------------------------------
# 8. PROVENANCE AND THE DOCSTRING SCAN
# ---------------------------------------------------------------------------

def provenance():
    """The producing commit and the machine id, COMPUTED here, never copied in.

    The machine id is the leading twelve hex characters of a sha256 over the node
    name, which is the
    form the rest of this round uses; uuid.getnode() is a different number and is
    deliberately not it.
    """
    root = Path(__file__).resolve().parents[1]
    commit = ""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"],
                                cwd=str(root), capture_output=True, text=True,
                                timeout=30).stdout.strip()
    except Exception:                                          # noqa: BLE001
        commit = ""
    if not _re.fullmatch(r"[0-9a-f]{7}", commit or ""):
        commit = "UNKNOWN"
    node = platform.node()
    return dict(commit=commit,
                machine=hashlib.sha256(node.encode("utf-8")).hexdigest()[:12],
                node=node, python=platform.python_version(), numpy=np.__version__)


_DECIMAL = _re.compile(r"(?<![\w.])[+-]?\d+(?:\.\d+)*(?:[eE][+-]?\d+)?(?!\w)(?!\.\d)")


def docstring_numbers():
    """(docstring count, number count) for this module's OWN scope.

    Scoped on __module__ so numpy's docstrings do not come along and NOT filtered
    through __all__, because demo() prints every number and is absent from it.
    The guard in tests/curvature/ writes its own counter rather than calling this
    one: two counters that must agree is a check, one shared helper is not.
    """
    docs = [__doc__ or ""]
    for obj in list(globals().values()):
        if getattr(obj, "__module__", None) != __name__:
            continue
        doc = getattr(obj, "__doc__", None)
        if isinstance(doc, str) and doc.strip():
            docs.append(doc)
    toks = {t.lstrip("+") for d in docs for t in _DECIMAL.findall(d)}
    return dict(n_docstrings=len(docs), n_numbers=len(toks),
                n_decimals=len([t for t in toks if "." in t]))


# ---------------------------------------------------------------------------
# 9. THE REPORT
# ---------------------------------------------------------------------------

_CACHE = {}


def report(seed=SEED):
    """Everything this module measures, in one pass, memoised."""
    if seed in _CACHE:
        return _CACHE[seed]
    t0 = time.time()
    res = assign(seed=seed, normalize=False)
    A = res["assignments"]

    logits = OrderedDict()
    v, u, _, _ = draw_tokens(N_GRID[-1], N_REPLICATES, seed)
    for name, spec in LOGITS.items():
        arr = spec["build"](v, u)
        got = safe_exp(arr, spec["units"])
        logits[name] = dict(units=dict(spec["units"]), units_sum=units_sum(spec["units"]),
                            exp_ok=not is_refusal(got),
                            max_abs=float(np.max(np.abs(arr))))
    killed = safe_exp(v, DIMENSIONFUL_LOGIT["units"])

    sweep = beta_sweep(seed)
    worst = max(abs(r["gap"]) for r in sweep)
    maxsub = output_alpha(0.0, seed=seed, max_subtract=True)
    maxsub1 = output_alpha(1.0, seed=seed, max_subtract=True)

    uva = OrderedDict((k, dict(declared_n=DECLARED_N[k], alpha=A[k].alpha,
                               gap=DECLARED_N[k] - A[k].alpha)) for k in COORD_NAMES)

    floor_names = [k for k, p in PLANT.items() if p[0] == "FLOOR"]
    truth = [k for k, p in PLANT.items()
             if p[0] in ("INTENSIVE", "EXTENSIVE", "MIXED", "FLOOR")]
    r2_would_refuse = [k for k in truth if A[k].r2 < R2_NAIVE_MIN]

    spec = sweep_spec(seed=seed)
    refit_spec, refit_digest = planted_refit()
    atk = normalizer_attack(seed)

    good = A[[k for k, p in PLANT.items() if p[0] == "INTENSIVE"][0]]
    out = dict(
        provenance=provenance(),
        grid=res["grid"], assignments=A, fits=res["fits"],
        census=res["census"], verdict=res["verdict"],
        logits=logits,
        kills=dict(alphaless_struck=is_refusal(struck(strip_alpha(good))),
                   dimensionful_exp_refused=is_refusal(killed),
                   dimensionful_reason=(killed.code if is_refusal(killed) else None)),
        beta_sweep=sweep, beta_sweep_worst_gap=worst,
        maxsub_alpha_beta0=maxsub.alpha, maxsub_alpha_beta1=maxsub1.alpha,
        unit_vs_alpha=uva,
        baselines=baselines(A, seed),
        theorem=theorem_hypotheses(seed),
        floor=dict(names=floor_names,
                   r2_max=max(A[k].r2 for k in floor_names),
                   curvature=A[floor_names[0]].curvature,
                   q_floor=A[floor_names[0]].q_floor,
                   chi2=A[floor_names[0]].chi2),
        r2_control=dict(threshold=R2_NAIVE_MIN, would_refuse=len(r2_would_refuse),
                        names=r2_would_refuse, n_true_power_laws=len(truth)),
        mask=dict(spec=spec, digest=mask_digest(spec), check=check_mask(spec),
                  planted_refit_digest=refit_digest,
                  planted_refit_fires=(not check_mask(refit_spec)["ok"]),
                  frozen=frozen_mask(A, mask_digest(spec))),
        tiebreak=tiebreak_probe(seed),
        prior_art=PRIOR_ART,
        chi2_p=math.exp(-CHI2_CRIT / 2.0),
        chi2_p_family=1.0 - (1.0 - math.exp(-CHI2_CRIT / 2.0)) ** len(COORD_NAMES),
        attack=atk,
        attacked_assignment_verdict=atk["attacked_verdict"],
        coverage=docstring_numbers(),
    )
    out["published"] = OrderedDict([
        ("beta_sweep_worst_gap", worst),
        ("maxsub_alpha_beta0", maxsub.alpha),
        ("maxsub_alpha_beta1", maxsub1.alpha),
        ("alpha_total_clean", atk["alpha_total_clean"]),
        ("alpha_total_attacked", atk["alpha_total_attacked"]),
        ("silently_reclassified", atk["silently_reclassified"]),
        ("offset_median_clean", atk["census_clean"]["offset_median"]),
        ("offset_median_attacked", atk["census_attacked"]["offset_median"]),
        ("offset_iqr_attacked", atk["census_attacked"]["offset_iqr"]),
        ("offset_iqr_clean", atk["census_clean"]["offset_iqr"]),
        ("reachable_corners_attacked", atk["reachable_corners_attacked"]),
        ("tiebreak_sqrt_gap", out["tiebreak"]["sqrt_gap"]),
        ("tiebreak_alpha_difference", out["tiebreak"]["alpha_difference"]),
        ("total_spread_clean", atk["census_clean"]["total_spread"]),
        ("total_spread_attacked", atk["census_attacked"]["total_spread"]),
        ("r2_would_refuse", len(r2_would_refuse)),
        ("floor_curvature", out["floor"]["curvature"]),
        ("floor_q_floor", out["floor"]["q_floor"]),
        ("row_sum_beta1", out["theorem"]["row_sum_beta1"]),
        ("row_sum_beta0", out["theorem"]["row_sum_beta0"]),
        ("rule_served", out["baselines"]["rule"]["served"]),
        ("all_softmax_served", out["baselines"]["all_softmax"]["served"]),
        ("all_linear_served", out["baselines"]["all_linear"]["served"]),
        ("random_served", out["baselines"]["random"]["served"]),
    ])
    out["seconds"] = time.time() - t0
    _CACHE[seed] = out
    return out


def demo():
    """Every number this module publishes, printed by one run."""
    t0 = time.time()
    p = provenance()
    print("PROV commit %s machine %s node %s python %s numpy %s"
          % (p["commit"], p["machine"], p["node"], p["python"], p["numpy"]))
    r = report()
    A = r["assignments"]
    say = print

    say("\n(a) R3 FIRST: only a pi group enters exp. Unit exponents, as numbers.")
    for name, row in r["logits"].items():
        say("    logit %-12s %s  sum %d  exp accepted: %s  max|l| %.4f"
            % (name, " ".join("%s^%+d" % (d, row["units"][d]) for d in BASE_DIMS),
               row["units_sum"], row["exp_ok"], row["max_abs"]))
    say("    KILL, planted: logit %s carries %s -> %s"
        % (DIMENSIONFUL_LOGIT["name"],
           " ".join("%s^%+d" % (d, DIMENSIONFUL_LOGIT["units"][d]) for d in BASE_DIMS),
           r["kills"]["dimensionful_reason"]))
    say("    and V^+1 U^+0 N^-1 SUMS to %d and is refused all the same: sum-zero is "
        "necessary, not sufficient." % units_sum({"V": 1, "U": 0, "N": -1}))

    say("\n(b) THE SWEEP. n = %s, %d replicates in %d blocks, seed %d."
        % (", ".join(str(n) for n in r["grid"]), N_REPLICATES, N_BLOCKS, SEED))
    say("    %-12s %-9s %-20s %-9s %-9s %-11s %s"
        % ("coordinate", "alpha", "CI (2 dof)", "resid", "R^2", "chi^2", "verdict"))
    for name, a in A.items():
        beta = ("beta=%.0f %s" % (a.beta, a.klass) if not is_refusal(a.beta)
                else "REFUSED %s" % a.beta.code)
        say("    %-12s %+8.4f  [%+7.4f,%+7.4f]  %.6f  %8.4f  %10.4g  %s"
            % (name, a.alpha, a.ci_lo, a.ci_hi, a.resid_rms, a.r2, a.chi2, beta))
    say("    planted truth: " + ", ".join("%s=%s" % (k, PLANT[k][0]) for k in COORD_NAMES))
    say("    tighter interval from the MEASURED sampling error, for comparison:")
    for name in ("mean_value", "sum_value", "half_walk"):
        a = A[name]
        say("      %-12s alpha %+0.4f  se-interval [%+0.4f, %+0.4f] against the "
            "2-dof [%+0.4f, %+0.4f]"
            % (name, a.alpha, a.ci_se_lo, a.ci_se_hi, a.ci_lo, a.ci_hi))

    say("\n(c) THE POWER-LAW TEST IS NOT R^2, AND THIS BED SHOWS WHY.")
    c = r["r2_control"]
    say("    a design refusing at R^2 < %.2f would refuse %d of the %d planted "
        "power laws: %s" % (c["threshold"], c["would_refuse"], c["n_true_power_laws"],
                            ", ".join(c["names"]) or "none"))
    ok = [A[k].chi2 for k, pl in PLANT.items() if pl[0] != "NOT-A-POWER-LAW"]
    bad = [A[k].chi2 for k, pl in PLANT.items() if pl[0] == "NOT-A-POWER-LAW"]
    say("    chi^2 separates them: power laws at worst %.4f, the plant at %.4g, "
        "critical %.1f" % (max(ok), max(bad), CHI2_CRIT))
    say("    critical %.1f on 2 dof is p = %.3e per coordinate and %.3e over the "
        "%d tested" % (CHI2_CRIT, r["chi2_p"], r["chi2_p_family"], len(COORD_NAMES)))

    say("\n(d) THE FLOOR, IN CLOSED FORM. Four points leave 2 dof; a quadratic "
        "departure q leaves residuals q*(+1,-1,-1,+1).")
    f = r["floor"]
    say("    smallest refusable curvature q_floor = sqrt(%.1f/4)*se = %.4f*se = "
        "%.6f (log2 units)"
        % (CHI2_CRIT, math.sqrt(CHI2_CRIT / 4.0), f["q_floor"]))
    say("    the planted logarithm carries curvature %.6f, chi^2 %.4f, R^2 %.4f -> "
        "PASSES, and is assigned beta=%.0f at alpha %+0.4f"
        % (f["curvature"], f["chi2"], f["r2_max"], A["log_shape"].beta,
           A["log_shape"].alpha))
    say("    it is NOT a power law. This instrument cannot say so. That is the floor.")

    say("\n(e) WHERE beta COMES FROM: W = e^l/Z^beta carries N^-beta, so a read "
        "carries N^(1-beta).")
    for row in r["beta_sweep"]:
        say("    beta %.2f  predicted alpha_out %+0.4f  measured %+0.4f  "
            "CI [%+0.4f,%+0.4f]  gap %+0.4f"
            % (row["beta"], row["predicted"], row["alpha_out"], row["ci_lo"],
               row["ci_hi"], row["gap"]))
    say("    worst gap over the sweep %.4f; the softmax read is a MEAN and the "
        "linear read is a TOTAL, measured, not asserted." % r["beta_sweep_worst_gap"])
    say("    CONTROL: subtracting the row max moves alpha_out at beta=0 from "
        "+1.0000 to %+0.4f (refused as MIXED-SCALING) and leaves beta=1 at %+0.4f. "
        "The stabilisation is unit-safe at one corner only."
        % (r["maxsub_alpha_beta0"], r["maxsub_alpha_beta1"]))

    say("\n(f) DECLARED UNITS AGAINST MEASURED ALPHA, per coordinate.")
    for name, row in r["unit_vs_alpha"].items():
        say("    %-12s declared N^%+0.1f   measured alpha %+0.4f   disagreement %+0.4f"
            % (name, row["declared_n"], row["alpha"], row["gap"]))
    say("    half_walk DECLARES a total and MEASURES half a count: the refusal has "
        "two independent routes, not one.")

    say("\n(g) L-SIMPLE. The simplest things that could have been done, same bed.")
    for name, b in r["baselines"].items():
        say("    %-12s served %d/%d  broken %d  refused %d   by class: %s"
            % (name, b["served"], b["n_coords"], b["broken"], b["refused"],
               "  ".join("%s %d/%d" % (k, v["served"], v["total"])
                         for k, v in sorted(b["by_class"].items()))))
    bs, rl = r["baselines"]["all_softmax"], r["baselines"]["rule"]
    say("    L-FAIL both directions: all_softmax serves %d/%d INTENSIVE (the rule "
        "%d/%d, a tie) and %d/%d EXTENSIVE (the rule %d/%d). It wins inside its "
        "class and loses outside it."
        % (bs["by_class"]["INTENSIVE"]["served"], bs["by_class"]["INTENSIVE"]["total"],
           rl["by_class"]["INTENSIVE"]["served"], rl["by_class"]["INTENSIVE"]["total"],
           bs["by_class"]["EXTENSIVE"]["served"], bs["by_class"]["EXTENSIVE"]["total"],
           rl["by_class"]["EXTENSIVE"]["served"], rl["by_class"]["EXTENSIVE"]["total"]))

    say("\n(h) L-CLASS. The theorem, and its hypotheses checked on THIS bed.")
    t = r["theorem"]
    say("    %s  READ %s" % (t["name"], t["source"]))
    say("    gate g == 0: %s;  logits finite: %s over %d entries;  n = %d"
        % (t["gate_is_zero"], t["logits_finite"], t["n_logits"], t["n"]))
    say("    beta=1 row sums %.12f, worst |sum-1| %.3e;  beta=0 row sums %.6f -> "
        "corners distinct: %s"
        % (t["row_sum_beta1"], t["row_sum_beta1_err"], t["row_sum_beta0"],
           t["corners_distinct"]))

    say("\n(i) THE FROZEN MASK: a digest over the INPUTS.")
    say("    payload keys %s -- no fitted exponent is in it" % (", ".join(_MASK_KEYS),))
    say("    digest %s  check ok=%s" % (r["mask"]["digest"], r["mask"]["check"]["ok"]))
    say("    planted refit at seed %d -> %s  fires: %s"
        % (REFIT_SEED, r["mask"]["planted_refit_digest"][:16],
           r["mask"]["planted_refit_fires"]))
    fm = r["mask"]["frozen"]
    say("    THE MASK, WITH ITS AXIS NAMED: axis %r length %d, %d assigned, "
        "%d refused" % (fm["axis"], fm["axis_len"], fm["n_assigned"],
                        len(fm["refusals"])))
    say("    beta along %s = [%s]"
        % (fm["axis"], ", ".join("None" if b is None else "%.0f" % b
                                 for b in fm["beta"])))
    say("    %s" % fm["note"])
    say("    beta in the shipped arm is per LAYER and 0-dim: READ "
        "ceq/arm_smprime.py:527, applied at :250, value contraction at :264, "
        "and the empty-shape assertion at tests/gate0/test_g13_beta_learnable."
        "py:130. None of that is this module's to change.")

    say("\n(j) THE ATTACK: a normalization layer that hides n.")
    atk = r["attack"]
    say("    total-norm exponent: clean %+0.4f  attacked %+0.4f"
        % (atk["alpha_total_clean"], atk["alpha_total_attacked"]))
    say("    %-12s %-20s %-11s %-11s %s"
        % ("coordinate", "alpha clean -> attacked", "class clean",
           "rule alone", "reachable after the census"))
    for name, row in atk["per_coordinate"].items():
        say("    %-12s %+0.4f -> %+0.4f     %-11s %-11s %s"
            % (name, row["alpha_clean"], row["alpha_attacked"],
               row["class_clean"], row["class_attacked_rule_only"],
               row["class_attacked"]))
    say("    %d of %d planted extensive coordinates SILENTLY reclassified intensive."
        % (atk["silently_reclassified"],
           sum(1 for k, pl in PLANT.items() if pl[0] == "EXTENSIVE")))
    say("    census, clean bed:    fires=%s  %s"
        % (atk["census_clean"]["fires"], atk["census_clean"]["verdict"]))
    say("    census, attacked bed: fires=%s  %s"
        % (atk["census_attacked"]["fires"], atk["census_attacked"]["verdict"]))
    say("    the RULE ALONE would have silently reclassified them; the census\n        overrides every corner, so %d of the %d coordinates hand a caller back\n        a float. assign() returns a dict whose verdict is set AND whose every\n        reachable beta is that Refusal -- structural, not advisory."
        % (atk["reachable_corners_attacked"], len(COORD_NAMES)))
    say("    assign() verdict on the attacked bed: %r" % (r["attacked_assignment_verdict"],))
    say("    LIMIT: a normalizer dividing by an independent noisy estimate of n "
        "pins no norm; only leg 2 catches it, and leg 2 needs declared units.")

    say("\n(l) THE PROPOSED TIEBREAK, TREATED AS A HYPOTHESIS AND KILLED.")
    tb = r["tiebreak"]
    say("    proposal: where declared and measured disagree and DECLARED is a\n    corner, prefer declared -- which would turn half_walk's refusal into\n    the linear corner.")
    say("    half_walk  declares N^%+0.1f  measures %+0.4f" 
        % (tb["declared_walk"], tb["walk_alpha"]))
    say("    sqrt_total declares N^%+0.1f  measures %+0.4f   (a sum over the first\n    ceil(sqrt(n)) tokens: a top-k pool at k = sqrt(n))"
        % (tb["declared_sqrt"], tb["sqrt_alpha"]))
    say("    the two differ by %.4f, inside the %.2f tolerance: the tiebreak reads\n    the SAME inputs for both and cannot tell them apart."
        % (tb["alpha_difference"], ALPHA_TOL))
    say("    it would hand both beta=%.0f, whose read carries n^1; sqrt_total is a\n    total over a SUBLINEAR count, so that misses by %.4f, which is %.1f\n    times the tolerance."
        % (tb["proposed_beta"], tb["sqrt_gap"], tb["gap_in_tolerances"]))
    say("    VERDICT: %s" % tb["verdict"])
    say("    NOTE, and it bounds what a MIXED refusal is worth: that a corner\n    exists for such a coordinate at all is a statement about the coordinate\n    ALONE. The consumer of this mask measured the same arm degrade by a\n    large factor once it shared a value net with a probability, and said\n    the factor was the trunk and not the exponent. That measurement is\n    HIS, is not reproduced here, and no number of it is quoted here.")

    say("\n(k) PRIOR ART. The nearest work, cited here rather than found later.")
    say("    %s" % r["prior_art"])

    cov = r["coverage"]
    say("\n    docstring coverage: %d docstrings in this module, %d distinct "
        "numbers, %d of them decimals"
        % (cov["n_docstrings"], cov["n_numbers"], cov["n_decimals"]))
    dt = time.time() - t0
    # NO SPACE BEFORE THE UNIT, DELIBERATELY. A wall clock printed as "0.20 s"
    # enters the token guard's "printed by a run" set, which does two bad things:
    # an absent-decimal control anywhere in the runtime range goes red at random,
    # and a fabricator quoting any plausible small decimal is handed it free.
    # "0.20s" matches no token, because the guard's trailing (?!\w) swallows it.
    # The 300 s bar keeps its space: it is a declared constant, not a reading.
    say("    elapsed %.2fs against the 300 s bar (report build %.2fs)"
        % (dt, r["seconds"]))

    # -- self-checks --------------------------------------------------------
    assert p["commit"] != "UNKNOWN" and _re.fullmatch(r"[0-9a-f]{12}", p["machine"])
    for name, row in r["logits"].items():
        assert row["units_sum"] == 0 and row["exp_ok"], name
    assert r["kills"]["dimensionful_exp_refused"] and r["kills"]["alphaless_struck"]
    for name, a in A.items():
        assert np.isfinite(a.alpha), name
        assert struck(a) is None, name
        klass, want = PLANT[name][0], PLANT[name][1]
        if klass == "INTENSIVE":
            assert a.beta == 1.0 and abs(a.alpha - want) <= ALPHA_TOL, name
        elif klass == "EXTENSIVE":
            assert a.beta == 0.0 and abs(a.alpha - want) <= ALPHA_TOL, name
        elif klass == "MIXED":
            assert is_refusal(a.beta) and a.beta.code == REASON_MIXED, name
            assert abs(a.alpha - want) <= ALPHA_TOL, name
        elif klass == "NOT-A-POWER-LAW":
            assert is_refusal(a.beta) and a.beta.code == REASON_NOT_POWER_LAW, name
        else:
            assert not is_refusal(a.beta) and a.chi2 <= CHI2_CRIT, name
    assert REASON_MIXED != REASON_NOT_POWER_LAW
    assert r["r2_control"]["would_refuse"] > 0, \
        "R^2 refuses no planted power law here: the control is vacuous"
    assert max(bad) > CHI2_CRIT > max(ok)
    assert abs(f["curvature"]) < f["q_floor"], "the floor plant is above the floor"
    assert r["beta_sweep_worst_gap"] <= ALPHA_TOL
    assert abs(r["maxsub_alpha_beta0"] - 1.0) > ALPHA_TOL
    assert abs(r["maxsub_alpha_beta1"]) <= ALPHA_TOL
    assert t["row_sum_beta1_err"] < 1e-12 and t["corners_distinct"]
    assert bs["by_class"]["INTENSIVE"]["served"] == bs["by_class"]["INTENSIVE"]["total"]
    assert bs["by_class"]["EXTENSIVE"]["served"] == 0 < rl["by_class"]["EXTENSIVE"]["served"]
    assert r["baselines"]["all_linear"]["by_class"]["EXTENSIVE"]["served"] == 3
    assert r["baselines"]["all_linear"]["by_class"]["INTENSIVE"]["served"] == 0
    assert r["mask"]["check"]["ok"] and r["mask"]["planted_refit_fires"]
    assert mask_digest(perturb_one_measured_norm(r["mask"]["spec"], 1e-9)) \
        != r["mask"]["digest"]
    assert fm["axis"] == MASK_AXIS != "S" and fm["axis_len"] == len(COORD_NAMES)
    assert len(fm["beta"]) == len(fm["coords"]) == fm["axis_len"]
    assert fm["n_assigned"] + len(fm["refusals"]) == fm["axis_len"]
    assert "2202.04643" in r["prior_art"] and "Bakarji" in r["prior_art"]
    assert tb["indistinguishable"], (
        "the two probe coordinates differ by %.4f: they no longer share the\n        tiebreak's inputs and the kill is not a kill" % tb["alpha_difference"])
    assert tb["kills_the_tiebreak"], "the tiebreak survived its own killer"
    assert tb["declared_walk"] == tb["declared_sqrt"] == 1.0
    assert is_refusal(A["half_walk"].beta), "the refusal did not stand"
    assert not atk["census_clean"]["fires"] and atk["census_attacked"]["fires"]
    assert atk["silently_reclassified"] == 3
    assert is_refusal(r["attacked_assignment_verdict"])
    assert atk["reachable_corners_attacked"] == 0, (
        "%d coordinates still hand a naive caller a corner on the attacked bed"
        % atk["reachable_corners_attacked"])
    for name, a in assign(seed=SEED, normalize=True)["assignments"].items():
        assert is_refusal(a.beta) and np.isfinite(a.alpha), name
    assert cov["n_docstrings"] >= 35 and cov["n_numbers"] >= 50, (
        "the docstring scan has shrunk to %d docstrings / %d numbers; a scope "
        "regression reports zero missing and passes"
        % (cov["n_docstrings"], cov["n_numbers"]))
    assert dt < 300.0
    say("\nALL SELF-CHECKS PASSED")
    return r["published"]


if __name__ == "__main__":
    demo()

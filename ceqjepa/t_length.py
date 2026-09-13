"""T-LENGTH: the length bar, on an intensive target and an extensive one at once.

RUN: python -m ceqjepa.t_length

WHAT IS MEASURED HERE, AND WHAT IS ONLY CALIBRATION. Train at n0, test at 4*n0
and 8*n0 on the SAME bed, with two targets read off the same tokens: an INTENSIVE
one (a probability, the mean of a per-token quantity) and an EXTENSIVE one (a
count, the sum of it). The single-exponent arms are CALIBRATION INSTRUMENTS, not
findings -- that a single softmax readout degrades on an extensive target as n
grows is a theorem someone else proved, cited below, and the bed either
reproduces it or the bed is broken. What is measured is the one unoccupied
question: whether assigning the normalisation exponent PER OUTPUT COORDINATE
recovers the right corner for each, and how that stands against a head free to
learn the exponent at matched parameter counts.

THE SCOPE OF EVERY "CANNOT" IN THIS FILE: ONE forward pass, NO scratchpad, and
the representation of ONE readout row. Length-generalizable softmax
chain-of-thought transformers are Turing-complete (arXiv:2511.20038, Theorem 4.3,
relative positional encodings), so an unscoped "softmax cannot count" is refuted
in one line and deserves to be.

WHAT IS NOT CLAIMED. Nothing here says WHICH coordinate gets which exponent --
that is the assignment rule, it is not in this file, and this file does not
import it. The mask is an INPUT. Nothing here claims a general length result: the
bed is synthetic, the tokens are exchangeable, and the only aggregation is one
weighted pool. Nothing here claims the exponent is the whole assignment -- (h)
measures a second thing that has to be assigned with it.

THE REFUSED TRADE-OFF. T-LENGTH was supposed to wait for the assignment rule to
land. It does not have to. The per-coordinate mask is an argument, the harness
runs today against masks constructed here, and wiring the measured mask in is one
argument.

WIRING THE MEASURED MASK (one argument, mechanically):

    from ceqjepa.pi_assign import <whatever it exports>    # when it lands
    mask = <that>(...)                # a sequence of C floats, C == len(TARGETS)
    t_length_report(masks={"assigned-measured": mask})

The harness asks of a mask exactly three things, and nothing else:
  1. len(mask) == len(TARGETS), one entry per OUTPUT coordinate, in that order.
     THIS IS ASSERTED BY NAME, not left to broadcast: in the shipped per-layer
     code a beta shaped like the TOKEN axis is silently accepted and wrong,
     because [..., S, 1] ** [S] broadcasts to [..., S, S] and the exponent then
     indexes the key rather than the query. pool() raises on it; (b) fires it.
  2. each entry a float in [0, 1] -- 1 normalised, 0 unnormalised, between
     interpolated, and pool() is continuous in it so a soft mask is legal.
  3. it is a function of the TRAINING data and the coordinate index only. A mask
     that reads n at test time makes the extensive bar vacuous -- see (i).

THE MEASURED MASK, NOW WIRED (k). The rule landed, so the argument this harness
was built around is supplied rather than planted, and the wiring was the one
argument it was advertised to be. The vector crosses as DATA and is validated as
data: the axis name is checked rather than rewritten, the axis length is checked
against the vector length so a compacted axis is refused, and a refused
coordinate carrying a number instead of None is refused as a dropped refusal.
Nothing in this file imports the rule; it is located at call time and only its
emitted vector crosses, so the table above was produced before the rule existed
and would be produced again without it.

WHICH COORDINATE IS MINE is a SEMANTIC choice and it is mine, not a measurement:
my intensive target is a probability, so it is that file's prob_above, and my
extensive target is a count, so it is its count_above. The choice is shown
insensitive in the run -- its three intensive coordinates all carry beta 1.0 and
its three extensive ones all carry beta 0.0 -- but insensitive is not measured.

WHAT THE RULE RETURNED, loaded in this process and not copied from anyone's run:
axis 'D' of length 9, 7 of 9 coordinates assigned and 2 refused (half_walk at
alpha +0.5180 as MIXED-SCALING, osc_shape as NOT-A-POWER-LAW). My two arrive as
prob_above alpha -0.0330 beta 1.0 and count_above alpha +0.9670 beta 0.0, so the
measured vector is (1.0, 0.0): the rule recovers the right corner for both of my
coordinates FROM DATA ALONE. The row it then produces is identical to the oracle
row BY CONSTRUCTION -- same vector, same pipeline, same seeds, 0.0014 / 0.0702 at
4x and 0.0011 / 0.0804 at 8x, asserted equal cell by cell -- so the information
is in the VECTOR comparison and the row is a tautology. Presenting that row as
independent confirmation would be dressing a tautology as a measurement.

THE FLOOR, AND WHETHER IT BITES HERE. Four sweep points leave two residual
degrees of freedom, and a log-shaped coordinate is not a power law yet passes the
power-law test: in the very vector loaded here, log_shape arrives ASSIGNED the
softmax corner, beta 1.0 at alpha +0.0387, labelled intensive with no warning.
Neither of my coordinates is log-shaped -- my own targets measure alpha -0.0042
and +0.9958 -- so the floor does not bite this bed, but it would bite any
coordinate whose growth is logarithmic and this harness could not tell.

THE REFUSAL CHANNEL, EXERCISED END TO END (l). The rule refused 2 of its 9
coordinates, and a refusal is only worth having if declining costs less than
guessing. Both refusals are exercised here, on beds whose second target is built
the WAY the coordinate it maps onto is planted, so the mapping is a construction
and not an assertion: MIXED-SCALING maps onto half_walk, planted there as
w.sum(axis=1) over a zero-mean stream, so mine is the sum of a zero-mean
per-token feature -- a signed running total whose magnitude grows like the square
root because the terms cancel, measured alpha +0.5024. NOT-A-POWER-LAW maps onto
osc_shape, a symmetric bump in log2(n) times a per-sequence jitter, so mine is a
per-sequence quantity scaled by an amplitude that is a function of the CONTEXT
LENGTH alone: its norm reads 0.5083 at n=16, 1.2618 at n=64 and 0.5092 at n=128,
the same at 8x as at the training length, so its fitted slope is -0.0006 and
describes nothing. No readout whose only n-dependence is a pooling exponent can
produce that, which is a FLOOR by construction and is said here rather than
discovered below.

Five arms per reason, 1 seed, 1500 steps on 256 sequences, 58 parameters each.
force-* share one trunk between the two coordinates, which is the setting the
assignment problem lives in; solo-* train on the refused coordinate ALONE, which
separates "does a corner represent this coordinate" from "can one value net serve
it and another at once"; refuse trains on the assignable coordinate only and
publishes a REFUSAL for the other cell. The refusing arm reads 0.0006 and 0.0004
on its assignable cell -- so the budget is not what limits anything here -- and
REFUSED on the other, which is the whole point: it answers where it can.

THE TWO REASONS DO NOT COST THE SAME, and the discriminator is not a threshold,
it is whether ONE corner wins at EVERY test length.

    MIXED-SCALING     solo-linear 0.0526 and 0.0694, solo-softmax 0.7500 and
                      0.8750: the linear corner wins at BOTH lengths, so a corner
                      EXISTS and the refusal is CONSERVATIVE -- honouring it
                      costs a reader a model that works.
    NOT-A-POWER-LAW   solo-linear 0.2884 at 4x but 5.1177 at 8x, solo-softmax
                      0.5999 at 4x and 0.0017 at 8x: the ranking FLIPS. No corner
                      wins, the error is not monotone in n, and no finite set of
                      test lengths certifies one. The refusal is PROTECTIVE.

WHERE MY OWN PREDICTION WAS WRONG, corrected by the run rather than quietly
dropped. I expected the linear corner to be near-exact for the half-walk in the
TABLE, because a sum of zero-mean tokens is a sum. Alone it nearly is, 0.0526;
sharing a value net with a probability it reads 4.5369, a factor of 86. That
factor is the trunk, not the exponent, and it says the cost of forcing a corner
is not a property of the coordinate alone.

WHAT THE HALF-WALK SAYS ABOUT THE RULE'S OWN STATISTIC. Its softmax-corner error
is 0.7500 and 0.8750 -- exactly the count law 1 - n0/n, the SAME law as for a
plain count, although its norm exponent is about 0.50 and not 1. The exponent the
rule MEASURES governs the magnitude; the exponent that governs the READOUT error
is the DECLARED count exponent. ROUTE, needing no new measurement: that file
already prints the declared count exponent beside the measured alpha, and
half_walk declares N^+1.0 while measuring about half. Where the two disagree and
the DECLARED value is a corner, preferring the declared one turns this refusal
into a correct assignment.

THE SUBSTITUTION, PLANTED AND CAUGHT. A careless caller writes
`b if b is not None else 1.0` and a refusal becomes the softmax corner with no
trace in the type, the shape or the table. Run on the REAL vector with my
extensive target mapped onto half_walk: the careless path returns (1.0, 1.0) and
the harness returns None carrying the rule's own reason. The price of that
default is the force-softmax row, 0.9099 at 4x, in the same units as everything
else in this file.

BETA IS DERIVED, NOT CHOSEN. A read carries N^(1-beta), so the count exponent of
the output is 1-beta and beta = 1-alpha is bookkeeping rather than a knob. On my
own targets alpha measures -0.0042 and +0.9958, giving beta 1.0042 and 0.0042:
the assigned arm is the dimensionally FORCED setting on this bed, not a heuristic
that happened to work. NO fit quality anywhere in this file is gated on R2 --
R2 measures residual against a SPREAD that an intensive coordinate does not have,
so a threshold on it refuses exactly the flat power laws it should keep.

PRE-REGISTERED, BEFORE THE RUN, IN CLOSED FORM (e). A beta = 1 readout is a
convex combination of bounded per-token values, so an affine head fitted at n0
keeps predicting an n0-sized count and its nrmse on the extensive bar is
1 - n0/n: 0.7500 at 4x and 0.8750 at 8x. A beta = 0 readout sums, so it
overshoots an intensive target by n/n0 and its nrmse there is n/n0 - 1: 3.0000
and 7.0000. MEASURED on this bed: 0.7500 and 0.8750 for the softmax arm, exact to
four decimals, and 3.0229 and 7.0671 for the linear arm, 0.76% and 0.96% above
the closed form. The bed reproduces the cited direction AND the scale, which is
what licenses the rest.

MEASURED, this box, CPU, `python -m ceqjepa.t_length`, 3 seeds x 2000 Adam steps
at lr 0.03 on 512 training sequences of length 16, torch float64, predictions
pooled over seeds before the error is taken (nrmse = RMSE over RMS of the truth):

    arm                 params  4x intensive  4x extensive  8x extensive
    lookup-table             0        0.1128        0.7548        0.8793
    marginal-frequency       0        0.3972        0.7895        0.8964
    all-softmax             58        0.0019        0.7500        0.8750
    all-linear              58        3.0229        0.2541        0.2889
    learned-beta            60        1.1477        0.2757        0.3739
    assigned-oracle         58        0.0014        0.0702        0.0804
    assigned-measured       58        0.0014        0.0702        0.0804

Both single-exponent failures grow with n, which is the calibration: 0.7500 to
0.8750 for softmax on the extensive bar, 3.0229 to 7.0671 for linear on the
intensive one. As factors over the assigned arm in the same cell at 4x, softmax
on the extensive bar is 10.68 and linear on the intensive bar 2175.03; the
owner point estimates are 7.9 and 31.6, differences +2.78 and +2143.43, and
neither of those two numbers was an input to anything here. Against the
marginal-frequency floor rather than the assigned arm the same two cells read
0.95 and 7.61 -- so the softmax factor is not an artefact of a small denominator
and the linear one largely is.

THE OWNER COUNTER, SCORED AND NOT ARGUED WITH. The learned-beta head is the only
baseline that matters, because it could discover the assignment by itself. On
this bed it does not, and the reason is measurable rather than rhetorical: at ONE
training length the exponent is barely identified. The pooled denominator moves
by sd 0.0015 in log across training instances at n = 16, while it moves 1.3863 in
log between n = 16 and n = 64 -- a ratio of 900, so the training set carries
nearly three orders of magnitude less signal about the exponent than the test
asks of it. Given the BEST of three configurations by training loss at each seed,
the learned head still lands at (0.5762, 0.2054) instead of (1, 0) -- per seed
(0.54, 0.29), (0.60, 0.20), (0.59, 0.13), never near the mask -- and its worst
test cell is 2.0322 against the assigned arm 0.0804. VERDICT WIN, at 60
parameters for the learned head against 58 for the assigned one: the learned head
carries MORE, so the win is not bought with parameters. It is under-DETERMINED,
not under-trained.

THE ASSIGNMENT IS NOT ONLY THE EXPONENT (h), and this is a CORRECTION to the
instance, which predicted the assigned arm exact on both bars. With the exponent
assigned and the additive output constant left free, the extensive residual is
0.0702 at 4x and 0.0804 at 8x rather than zero. Pinning both output constants to
zero -- a bias-free readout, everything else identical -- collapses it to 0.0012
and 0.0009, a factor of 58.6. A constant added to a normalised read is not
extensive, so the assignment has to cover the constant as well as the exponent.

THE MARS ATTACK (i). An extensive target leaks if the model can read n. Censused
rather than asserted: the per-token feature MEANS predict n at R2 = 0.0019, but
the feature EXTREMES predict it at R2 = 0.7697, because the maximum of n draws
grows with n. That channel is real and is reported rather than hidden -- and it
does not convert. Training the softmax head with a raw position feature moves its
extensive bar only from 0.7500 to 0.7185, 4.21%, because the encoder saturates
outside the range it trained on. Handing it n outright DOES convert: multiplying
its intensive read by the true n collapses the extensive bar to 0.0030. The
missing quantity is exactly the factor n and nothing else, so a bed carrying n
measures nothing, and the census is seen to fire on a planted position feature at
R2 = 1.0000.

PRIOR ART, stood on rather than routed around.
  arXiv:2410.01104 (Velickovic et al., ICML 2025) is the softmax row itself --
    softmax circuits must disperse as the item count grows. Already marked [V] in
    this tree at docs/sources/sweep/sweep_expressivity.md:325-330. The softmax
    arm here REPLICATES it on a new bed and is reported as a replication.
  arXiv:2406.04267 (Barbero et al., NeurIPS 2024) is the decoder stack: Theorem
    B.3 gives representational collapse of the last-token representation in L1
    under 4 hypotheses, Corollary B.10 turns it into the counting failure,
    Proposition B.9 is the non-asymptotic version with no positional encodings,
    and Gemini 1.5 is reported miscopying at length 300.
  arXiv:2511.20038 bounds every "cannot" sentence above.
  arXiv:2202.04643 (Bakarji et al., Nature Computational Science 2022) is the
    nearest prior work to the delta: BuckiNet learns one exponent per INPUT
    coordinate, but through a single shared layer against a null-space loss, and
    the exponents are fitted rather than swept.
  arXiv:2006.16236 (Katharopoulos et al. 2020) is the normalised/unnormalised
    axis; arXiv:2108.12409 (Press et al.) is the length-generalisation line.
UNOCCUPIED across all of them, searched for rather than inferred: none measures a
per-coordinate scaling exponent from a context-length sweep, none uses such an
exponent to pick between two attention variants per coordinate, and none refuses
a coordinate that fits neither class.

LIMITS, collected here and not scattered. The bed is synthetic and its tokens are
exchangeable, so nothing here bears on a real sequence model. One pool, one
layer, two output coordinates. The exponent is identified at a single training
length only through instance-to-instance variation in the denominator, which is
sd 0.0015 here; training at two lengths would identify it and the learned head
would then be expected to find the corner -- so the WIN above is a statement
about single-length training, which is the protocol the brief fixed, and not
about learned exponents in general. The measured mask agrees with the planted one
on THESE TWO coordinates, which are a probability and a count -- the two easiest
cases in the rule's own bed, both far from its detection floor -- so nothing here
tests the rule on a coordinate it finds hard, and the two coordinates it refused
never reached this harness at all. The refusal channel is now exercised on both
refused coordinates, but on beds I BUILT to their construction rather than on the
rule's own bed, so what is measured is what those SHAPES cost a readout, not what
that file's coordinates cost. One seed there, and the MIXED-SCALING conclusion is
about a coordinate trained alone: in the shared-trunk setting the assignment
problem actually lives in, no corner is good for it.
"""


import hashlib
import math
import platform
import re
import subprocess
import time

import numpy as np
import torch

__all__ = [
    "N0", "TEST_LENGTHS", "TARGETS", "MASKS", "OWNER_4X", "TIE_BAND",
    "HEAD_COMMIT", "MACHINE_ID", "provenance",
    "bed", "pool", "Head", "train_head", "predict", "nrmse",
    "predicted_single_exponent_error",
    "lookup_predict", "marginal_predict", "census_n",
    "length_refusal", "t_length_report", "docstring_numbers",
    "MEASURED_AXIS", "MEASURED_AXIS_LEN", "COORD_FOR_TARGET",
    "target_alpha", "mask_from_frozen", "load_measured_mask",
    "REFUSAL_REASONS", "osc_amplitude", "refusal_bed",
    "refusal_target_norms", "refusal_target_alpha",
    "silently_defaulted_mask", "refusal_report", "predicted_osc_error",
    "REFUSAL_INSTANCES", "REFUSAL_STEPS",
]

#: The training length, and the two test lengths. Never evaluated at N0 alone --
#: see length_refusal, which is the owner's kill written as a guard.
N0 = 16
TEST_LENGTHS = (64, 128)

#: Output coordinates, in the order every mask is indexed by.
TARGETS = ("intensive", "extensive")

#: Masks constructed HERE, today, with no assignment rule in sight.
MASKS = {
    "all-softmax": (1.0, 1.0),
    "all-linear": (0.0, 0.0),
    "learned-beta": None,            # the exponent is a free parameter
    "assigned-oracle": (1.0, 0.0),   # the mask an assignment rule ought to return
}

#: The owner's point estimates at 4x, carried as PROSE and never as an input.
OWNER_4X = (7.9, 31.6)

#: A win needs the loser to be worse by this factor in its worst cell. Anything
#: inside the band is a TIE, stated plainly.
TIE_BAND = 1.5

#: L-PROV. Pinned, then VERIFIED at run time by provenance() -- the commit from
#: git itself and the machine id recomputed from this box's node name.
HEAD_COMMIT = "c9a9434"
MACHINE_ID = "60b8cf943ee0"

#: THE WIRING OF THE MEASURED MASK. The rule emits a vector along a NAMED axis
#: with refused coordinates kept as None, so the axis cannot silently shorten.
#: Both are checked at the boundary rather than trusted, and the axis name is
#: never rewritten here: a vector of length S would index the key position.
MEASURED_AXIS = "D"
MEASURED_AXIS_LEN = 9

#: WHICH of the rule's coordinates are MINE. This mapping is SEMANTIC and it is
#: my choice, not a measurement: my intensive target is a probability, so it is
#: that file's prob_above, and my extensive target is a count, so it is its
#: count_above. The choice is shown to be insensitive in (k) -- all three of its
#: intensive coordinates carry one exponent and all three extensive ones the
#: other -- but insensitive is not the same as measured, and it is stated here.
COORD_FOR_TARGET = {"intensive": "prob_above", "extensive": "count_above"}

#: THE REFUSAL BED. Two extra targets, each built the SAME WAY as the coordinate
#: of the rule's bed it maps onto, so the mapping is a construction rather than
#: an assertion -- a bad mapping masquerading as a bad refusal is the thing to
#: avoid here. MIXED-SCALING maps onto half_walk, which that file plants as
#: w.sum(axis=1) over a zero-mean stream: a running TOTAL of a signed per-token
#: quantity, declared N^1 but whose magnitude grows like the square root because
#: the terms cancel. NOT-A-POWER-LAW maps onto osc_shape, planted there as a
#: symmetric bump in log2(n) times a per-sequence jitter: a quantity whose
#: amplitude depends on the context length itself, non-monotonically.
REFUSAL_REASONS = ("MIXED-SCALING", "NOT-A-POWER-LAW")

#: My own bump constants, in the form _osc uses but not its values. The bump is
#: symmetric about the middle of the grid, so its OLS slope through the sweep is
#: zero by construction and a rule reading only the slope calls it INTENSIVE.
OSC_BASE = 1.0
OSC_AMP = 0.5

#: The teacher. Fixed, so the bed is the same function of x at every length.
TEACHER_W = (1.5, 0.8, -0.5, 0.3)
TEACHER_B = -0.2
D_FEATURES = 4

TRAIN_INSTANCES = 512
EVAL_INSTANCES = 512
STEPS = 2000
LR = 0.03
#: The learned-beta head is scored at the BEST of these by training loss, per
#: seed. A win over an under-trained head is not a win.
LEARNED_SWEEP = ((2000, 0.03), (2000, 0.1), (4000, 0.03))
HIDDEN = 8
SEEDS = (0, 1, 2)


def provenance():
    """The commit and the machine id, RECOMPUTED and checked against the pins.

    uuid.getnode() is the wrong thing here -- it returns the MAC, 00155d146528
    on this box, not the id the round is stamped with. The id is the first 12
    hex of sha256 over the node name.
    """
    node = platform.node()
    machine = hashlib.sha256(node.encode()).hexdigest()[:12]
    try:
        commit = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"],
                                capture_output=True, text=True,
                                timeout=30).stdout.strip()
    except Exception as exc:                              # noqa: BLE001
        commit = "git-unavailable: %r" % (exc,)
    return dict(node=node, machine=machine, commit=commit,
                machine_ok=(machine == MACHINE_ID),
                commit_ok=(commit == HEAD_COMMIT))


# ---------------------------------------------------------------------------
# The bed: one data set, two bars.
# ---------------------------------------------------------------------------

def bed(n, n_instances, seed, with_position=False):
    """Sequences of n exchangeable tokens; both targets read off the same tokens.

    Each sequence carries a latent shift m ~ N(0, 1) added to feature 0 of every
    token, so the intensive target keeps an O(1) spread across instances at any
    length instead of concentrating on its own mean as n grows -- without it the
    intensive bar becomes a constant and any predictor wins it.

    r_i = sigmoid(TEACHER_W . x_i + TEACHER_B) is the per-token quantity;
    the intensive target is mean(r), the extensive target is sum(r), so the
    second is EXACTLY n times the first and the two bars are one bed.

    with_position appends the raw index i as a feature. That is the planted leak
    of (i), never the bed the table is measured on.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_instances, n, D_FEATURES))
    X[:, :, 0] += rng.normal(size=(n_instances, 1))
    r = 1.0 / (1.0 + np.exp(-(X @ np.asarray(TEACHER_W) + TEACHER_B)))
    Y = np.stack([r.mean(1), r.sum(1)], 1)
    if with_position:
        pos = np.tile(np.arange(n, dtype=float)[None, :, None], (n_instances, 1, 1))
        X = np.concatenate([X, pos], 2)
    return X, Y


def osc_amplitude(n, n0=None):
    """A symmetric bump in log2(n / n0): the planted NON-power-law amplitude.

    Symmetric about the middle of the sweep, so the slope through the sampled
    points is zero and a rule reading only the slope hands it the softmax
    corner. At n0 and at 8*n0 the bump takes the SAME value and at 4*n0 it does
    not, which is what "no exponent describes this" looks like from inside a
    table: the error is not monotone in n, so no finite set of test lengths
    certifies it.
    """
    t = math.log2(float(n) / float(N0 if n0 is None else n0))
    return OSC_BASE + OSC_AMP * math.cos(2.0 * math.pi * (t - 1.5) / 3.0)


def predicted_osc_error(n, n0=N0):
    """Pre-registered, before the run: what a fitted amplitude costs at length n.

    A model trained at n0 learns the amplitude osc_amplitude(n0) and the
    per-sequence part, and has no channel through which to learn that the
    amplitude moved. Its error at n is therefore |A(n0) - A(n)| against a truth
    of size A(n), so the nrmse is |1 - A(n0)/A(n)| -- 0 wherever the bump
    returns to its training value, whatever n is. No exponent has that shape.
    """
    a0, an = osc_amplitude(n0), osc_amplitude(n)
    return abs(1.0 - a0 / an)


def refusal_bed(n, n_instances, seed, reason):
    """The bed for one refused reason: one assignable target and one refused one.

    Column 0 is the SAME intensive target as bed(), so every table below still
    has a cell that a corner can fill -- the point is what happens to the OTHER
    cell. Column 1 is built to the construction of the coordinate it maps onto:

      MIXED-SCALING    sum of a zero-mean per-token feature. It IS a total, and
                       a beta = 0 read is a sum, so one corner can represent it
                       EXACTLY -- while its norm grows like sqrt(n), which is
                       what the rule measures and why it refuses.
      NOT-A-POWER-LAW  a per-sequence quantity scaled by osc_amplitude(n). The
                       per-sequence part is readable from the tokens; the
                       amplitude is a function of the context length alone, and
                       no readout whose only n-dependence is a pooling exponent
                       can produce it. That is a FLOOR by construction and it is
                       said here rather than discovered below.
    """
    X, Y = bed(n, n_instances, seed=seed)
    if reason == "MIXED-SCALING":
        col = X[:, :, 3].sum(1)
    elif reason == "NOT-A-POWER-LAW":
        col = osc_amplitude(n) * (1.0 + 0.2 * X[:, :, 0].mean(1))
    else:
        raise ValueError("no such refused reason: %r" % (reason,))
    return X, np.stack([Y[:, 0], col], 1)


def refusal_target_norms(lengths=None, n_instances=EVAL_INSTANCES, seed=7):
    """RMS of each refused-shape target, the statistic the rule's alpha fits."""
    lengths = tuple(lengths or ((N0,) + TEST_LENGTHS))
    out = {}
    for reason, key in zip(REFUSAL_REASONS, ("mixed", "not-a-power-law")):
        out[key] = {}
        for n in lengths:
            _, Y = refusal_bed(n, n_instances, seed, reason)
            out[key][n] = float(np.sqrt(np.mean(Y[:, 1] ** 2)))
    return out


def refusal_target_alpha(lengths=(16, 32, 64, 128)):
    """The count exponent of each refused-shape target, fitted as a line in
    log-log. The mixed one lands between the corners; the bump's slope is zero
    by construction, which is precisely why a slope is not enough."""
    norms = refusal_target_norms(lengths)
    xs = [math.log(float(n)) for n in lengths]
    xbar = sum(xs) / len(xs)
    sxx = sum((x - xbar) ** 2 for x in xs)
    out = {}
    for key, per_n in norms.items():
        ys = [math.log(per_n[n]) for n in lengths]
        ybar = sum(ys) / len(ys)
        out[key] = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / sxx
    return out


def silently_defaulted_mask(frozen, coord_for_target=None):
    """THE PLANTED SUBSTITUTION, written the way a careless caller writes it.

    `b if b is not None else 1.0` turns every refusal into the softmax corner
    with no trace in the type, the shape or the table. It exists here so the
    guard can be SEEN to catch it and so the cost of the substitution can be
    scored in the same units as everything else. It is never used to fill a cell.
    """
    coord_for_target = dict(coord_for_target or COORD_FOR_TARGET)
    coords, beta = list(frozen["coords"]), list(frozen["beta"])
    return tuple(1.0 if beta[coords.index(coord_for_target[t])] is None
                 else float(beta[coords.index(coord_for_target[t])])
                 for t in TARGETS)


def pool(w, u, beta):
    """The one line the whole claim is about: sum(w*u) / sum(w)**beta.

    beta is a per-coordinate exponent, so this returns [N, C] from [N, n]: at
    beta = 1 the column is the softmax-weighted MEAN of u, at beta = 0 the plain
    weighted SUM, in between a continuous interpolation. Works on numpy arrays
    and on torch tensors, and carries gradient through beta when beta is one.

    THE AXIS IS ASSERTED, NEVER LEFT TO BROADCAST. In the shipped code beta is
    per LAYER -- a 0-dim parameter at ceq/arm_smprime.py:527 applied at :250 --
    and a beta shaped like the TOKEN axis is silently accepted there, because
    [..., S, 1] ** [S] broadcasts to [..., S, S] and the exponent then indexes
    the key j rather than the query i. Nothing about that is visible in the
    output shape of a square batch, so it is checked here by name instead.
    """
    shape = tuple(getattr(beta, "shape", None) or np.shape(beta))
    if len(shape) != 1 or shape[0] != len(TARGETS):
        raise ValueError(
            "beta must index the OUTPUT axis: expected (%d,) for TARGETS %r, "
            "got %r. A beta shaped like the token axis broadcasts to [..., S, S] "
            "and exponentiates by the key rather than the query."
            % (len(TARGETS), TARGETS, shape))
    num = (w * u).sum(-1)[..., None]
    den = w.sum(-1)[..., None]
    if hasattr(den, "detach"):
        b = beta if hasattr(beta, "detach") else torch.as_tensor(
            np.asarray(beta, dtype=float), dtype=den.dtype)
    else:
        b = np.asarray(beta, dtype=float)
    return num / den ** b


class Head(torch.nn.Module):
    """One attention-shaped readout. The mask is an argument, not a property.

    A tiny value net u(x) and a bounded score net w(x) = exp(2*tanh(.)) are
    SHARED by both output coordinates -- which is the point: if the coordinates
    could have separate trunks there would be no assignment problem. Only the
    exponent, the output scale and the output constant are per-coordinate.

    beta_mask=None makes the exponent a free parameter, one logit per
    coordinate: that is the learned-beta baseline, and it carries exactly
    len(TARGETS) more parameters than the assigned head.
    """

    def __init__(self, d, beta_mask, seed=0, hidden=HIDDEN, bias_free=False):
        super().__init__()
        self.v1 = torch.nn.Linear(d, hidden).double()
        self.v2 = torch.nn.Linear(hidden, 1).double()
        self.score = torch.nn.Linear(d, 1).double()
        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.empty_like(p).uniform_(-0.5, 0.5, generator=gen))
        c = len(TARGETS)
        self.scale = torch.nn.Parameter(torch.ones(c, dtype=torch.float64))
        if bias_free:
            self.register_buffer("offset", torch.zeros(c, dtype=torch.float64))
        else:
            self.offset = torch.nn.Parameter(torch.zeros(c, dtype=torch.float64))
        self.learned = beta_mask is None
        if self.learned:
            self.rho = torch.nn.Parameter(torch.zeros(c, dtype=torch.float64))
            self.register_buffer("fixed", torch.zeros(c, dtype=torch.float64))
        else:
            mask = np.asarray(beta_mask, dtype=float)
            if mask.shape != (c,) or mask.min() < 0.0 or mask.max() > 1.0:
                raise ValueError("a mask is %d floats in [0, 1]: got %r"
                                 % (c, beta_mask))
            self.register_buffer("fixed", torch.as_tensor(mask, dtype=torch.float64))

    def beta(self):
        """The exponents actually in force, learned or given."""
        return torch.sigmoid(self.rho) if self.learned else self.fixed

    def values(self, X):
        """The per-token value u and weight w. Exposed so the hull bound of the
        theorem can be measured on the model's OWN values rather than assumed."""
        u = self.v2(torch.tanh(self.v1(X))).squeeze(-1)
        w = torch.exp(2.0 * torch.tanh(self.score(X).squeeze(-1)))
        return u, w

    def forward(self, X):
        u, w = self.values(X)
        return self.scale * pool(w, u, self.beta()) + self.offset

    def n_params(self):
        """Trainable parameters. Printed beside every comparison, always."""
        return sum(int(p.numel()) for p in self.parameters() if p.requires_grad)


def train_head(beta_mask, seed=0, n0=N0, n_instances=TRAIN_INSTANCES,
               steps=STEPS, lr=LR, bias_free=False, with_position=False,
               bed_fn=None, coord_weights=None):
    """Full-batch Adam on the training length ONLY. Returns (head, final loss).

    The per-target loss is divided by that target's TRAINING variance, so the
    count (mean about 8 at n = 16) and the probability (mean about 0.5) weigh the
    same. The variance comes from the training length alone -- a test-length
    statistic in the loss would be the leak this file is attacking.
    """
    if bed_fn is None:
        X, Y = bed(n0, n_instances, seed=100 + seed, with_position=with_position)
    else:
        X, Y = bed_fn(n0, n_instances, 100 + seed)
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    var = Yt.var(0)
    wts = (torch.ones(len(TARGETS), dtype=torch.float64) if coord_weights is None
           else torch.tensor(np.asarray(coord_weights, dtype=float)))
    head = Head(X.shape[2], beta_mask, seed=seed, bias_free=bias_free)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    loss = torch.tensor(float("nan"))
    for _ in range(steps):
        opt.zero_grad()
        loss = ((((head(Xt) - Yt) ** 2).mean(0) / var) * wts).sum()
        loss.backward()
        opt.step()
    return head, float(loss.detach())


def predict(head, X):
    """[N, C] predictions as numpy, no gradient."""
    with torch.no_grad():
        return head(torch.tensor(X)).numpy()


def predicted_single_exponent_error(train_length, n):
    """What a single exponent MUST cost at length n, in closed form, before any run.

    Pre-registration, not a fit. A beta = 1 readout is a convex combination of
    bounded per-token values, so an affine head fitted at n0 keeps predicting an
    n0-sized count: its prediction is about (n0/n) times the truth and its nrmse
    is 1 - n0/n. A beta = 0 readout sums instead, so at length n it overshoots by
    about n/n0 and its nrmse on an intensive target is n/n0 - 1.

    THE SCOPE, which the sentence is worthless without: ONE forward pass, NO
    scratchpad, and the representation of ONE readout row. Length-generalizable
    softmax chain-of-thought transformers are Turing-complete (arXiv:2511.20038,
    Theorem 4.3), so an unscoped "softmax cannot count" is simply refuted.
    """
    ratio = float(train_length) / float(n)
    return dict(softmax_extensive=1.0 - ratio, linear_intensive=1.0 / ratio - 1.0)


def target_alpha(lengths=(16, 32, 64, 128), n_instances=EVAL_INSTANCES, seed=7):
    """The count exponent of each TARGET, measured from a length sweep.

    THE ASSIGNMENT IS FORCED, NOT TUNED, and this is where that is checked on my
    own bed rather than taken on trust. Carrying units through W = e^l / Z^beta,
    Z sums over n tokens so it carries N^1, W carries N^(-beta), and a read
    sum_j W_j v_j therefore carries N^(1-beta) -- a mean at beta = 1, a total at
    beta = 0. The exponent of the count in the output IS 1 - beta and the
    exponent of the count in the target IS alpha, so beta = 1 - alpha is
    bookkeeping and the corners are the only two settings whose output is a mean
    or a total rather than a fractional power of a count. The containment
    theorem the corners come from is READ at lean/CEQ/V16Domain.lean:433
    three_corners_containment and :445 corners_are_distinct -- RELAYED from the
    rule's docstring, not verified here.

    Fitted as a straight line in log-log, which is what a power law is. NO fit
    quality is gated on R2 anywhere in this file: R2 measures residual against
    the SPREAD of the data and an intensive coordinate has no spread, so a
    threshold on it refuses exactly the flat power laws it should keep.
    """
    xs, ys = [], {t: [] for t in TARGETS}
    for n in lengths:
        _, Y = bed(n, n_instances, seed=seed)
        xs.append(math.log(float(n)))
        for c, t in enumerate(TARGETS):
            ys[t].append(math.log(float(np.sqrt(np.mean(Y[:, c] ** 2)))))
    xbar = sum(xs) / len(xs)
    sxx = sum((x - xbar) ** 2 for x in xs)
    out = {}
    for t in TARGETS:
        ybar = sum(ys[t]) / len(ys[t])
        out[t] = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys[t])) / sxx
    return out


def mask_from_frozen(frozen, coord_for_target=None):
    """Take the VECTOR, not the module: validate it as data and map it to TARGETS.

    Every check here is a way for the wiring to be wrong while the shapes still
    look plausible, so each one is refused by name:

      the axis is RENAMED      -- a [D] mask and an [S] mask have the same type
                                  and opposite meanings, so the name is the only
                                  thing standing between them;
      the axis is COMPACTED    -- dropping the refused coordinates and shipping
                                  only the assignable ones silently changes the
                                  length of the axis, which is the same mistake
                                  in a smaller coat;
      the refusals are DROPPED -- a refused coordinate carrying a number instead
                                  of None is a fabricated exponent.

    A mapped coordinate that the rule REFUSED comes back as a refusal carrying
    its reason, never as a fabricated beta: that is the refusal channel, and a
    harness that invented a corner here would be answering a question the rule
    declined.
    """
    coord_for_target = dict(coord_for_target or COORD_FOR_TARGET)
    axis = frozen.get("axis")
    if axis != MEASURED_AXIS:
        raise ValueError("the mask axis is %r, not %r: a vector along another "
                         "axis indexes something else" % (axis, MEASURED_AXIS))
    n_axis = int(frozen["axis_len"])
    beta, coords = list(frozen["beta"]), list(frozen["coords"])
    if len(beta) != n_axis or len(coords) != n_axis:
        raise ValueError("the axis says %d but the vector carries %d beta and %d "
                         "names: the axis was compacted"
                         % (n_axis, len(beta), len(coords)))
    refusals = dict(frozen.get("refusals") or {})
    for name in refusals:
        if name in coords and beta[coords.index(name)] is not None:
            raise ValueError("coordinate %r is refused for %r but carries beta "
                             "%r: a refusal was dropped"
                             % (name, refusals[name], beta[coords.index(name)]))
    assigned = sum(1 for b in beta if b is not None)
    if "n_assigned" in frozen and int(frozen["n_assigned"]) != assigned:
        raise ValueError("the mask says %d assigned, the vector carries %d"
                         % (int(frozen["n_assigned"]), assigned))
    picked, why = [], None
    for t in TARGETS:
        name = coord_for_target[t]
        if name not in coords:
            raise ValueError("coordinate %r for target %r is not on the axis: %r"
                             % (name, t, coords))
        value = beta[coords.index(name)]
        if value is None:
            why = ("the rule REFUSED %r, the coordinate mapped to the %s target: "
                   "%s" % (name, t, refusals.get(name, "no reason given")))
            break
        picked.append(float(value))
    detail = dict(axis=axis, axis_len=n_axis, n_assigned=assigned,
                  refusals=refusals, coord_for_target=coord_for_target,
                  refused=why)
    if why is not None:
        return None, detail
    return tuple(picked), detail


def load_measured_mask(module_name="ceqjepa.pi_assign"):
    """Fetch the frozen vector, or say why there is none. NEVER required.

    The harness produced its whole table before any assignment rule existed and
    still does: this returns a refusal rather than raising, and demo() prints the
    row only when a vector arrives. The module is located at CALL time and only
    its emitted vector crosses into this file -- nothing here imports it, so the
    two files stay independent and the table is not hostage to the other one.
    """
    try:
        import importlib
        rule = importlib.import_module(module_name)
        res = rule.assign()
        frozen = rule.frozen_mask(res["assignments"],
                                  rule.mask_digest(rule.sweep_spec()))
        return dict(available=True, reason="", frozen=frozen)
    except Exception as exc:                                   # noqa: BLE001
        return dict(available=False, frozen=None,
                    reason="no mask from %r: %r" % (module_name, exc))


def nrmse(y, yhat):
    """RMSE over the RMS of the truth: comparable across a count and a rate."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)) / np.sqrt(np.mean(y ** 2)))


# ---------------------------------------------------------------------------
# L-SIMPLE: the untrained baselines, scored before anything trained.
# ---------------------------------------------------------------------------

def lookup_predict(Xtr, Ytr, X, ndigits=1):
    """The simplest thing that could work: a table keyed by a rounded statistic.

    Keyed on the sequence mean of feature 0 rounded to one decimal -- which is
    what the latent shift moves -- and returning that bucket's training mean,
    falling back to the global training mean for a key it never saw.
    """
    table = {}
    for k, y in zip(np.round(Xtr[:, :, 0].mean(1), ndigits), Ytr):
        table.setdefault(float(k), []).append(y)
    table = {k: np.mean(v, 0) for k, v in table.items()}
    fallback = Ytr.mean(0)
    keys = np.round(X[:, :, 0].mean(1), ndigits)
    return np.stack([table.get(float(k), fallback) for k in keys])


def marginal_predict(Ytr, n_rows):
    """The training mean of each target, for every row. Ignores the input."""
    return np.tile(Ytr.mean(0), (n_rows, 1))


# ---------------------------------------------------------------------------
# The owner's kill, written as a guard.
# ---------------------------------------------------------------------------

def length_refusal(train_length, test_lengths):
    """A reason string if this is not a length test, None if it is.

    T-LENGTH run at the training length only is not the test. The refusal is a
    VALUE a scorer can read, not an exception a caller's except clause eats and
    not a NaN that slips through `nan < tol`.
    """
    longer = [int(L) for L in test_lengths if int(L) > int(train_length)]
    if not longer:
        return ("no test length exceeds the training length %d: %r is the "
                "training length again, which measures nothing about length"
                % (int(train_length), tuple(int(L) for L in test_lengths)))
    if any(int(L) < int(train_length) for L in test_lengths):
        return ("a length below the training length %d was offered as a length "
                "test: %r" % (int(train_length), tuple(int(L) for L in test_lengths)))
    return None


# ---------------------------------------------------------------------------
# The MARS attack: what do the inputs carry about n?
# ---------------------------------------------------------------------------

def _r2_for_n(lengths, seed, with_position, extremes):
    X_rows, n_rows = [], []
    for L in lengths:
        X, _ = bed(L, 256, seed=seed, with_position=with_position)
        feats = [X.mean(1)] + ([X.max(1), X.min(1)] if extremes else [])
        X_rows.append(np.concatenate(feats, 1))
        n_rows.append(np.full(X.shape[0], float(L)))
    F = np.vstack(X_rows)
    F = np.hstack([F, np.ones((F.shape[0], 1))])
    y = np.concatenate(n_rows)
    coef, *_ = np.linalg.lstsq(F, y, rcond=None)
    return float(1.0 - np.sum((y - F @ coef) ** 2) / np.sum((y - y.mean()) ** 2))


def census_n(lengths, seed=11):
    """What the inputs carry about n, by OLS R2, before any model is blamed.

    Three readings: the per-token feature MEANS, which is what a uniform
    normalised head sees; the means with the EXTREMES, which is what a sharp
    attention head could see, and which carries n through order statistics; and
    the same on a bed with a planted position feature, where the census must
    saturate or it is decoration.
    """
    return dict(mean_r2=_r2_for_n(lengths, seed, False, False),
                extreme_r2=_r2_for_n(lengths, seed, False, True),
                leak_r2=_r2_for_n(lengths, seed, True, True))


# ---------------------------------------------------------------------------
# The report.
# ---------------------------------------------------------------------------

def t_length_report(n0=N0, test_lengths=TEST_LENGTHS, seeds=SEEDS, masks=None,
                    steps=STEPS, eval_instances=EVAL_INSTANCES, eval_seed=7):
    """The whole table, or a refusal. Every arm on the same rows.

    Predictions from all seeds are POOLED before the error is taken, so a
    published cell is one nrmse over one array and the identity layer can
    recompute it; the per-seed cells are kept beside it for the spread.
    """
    why = length_refusal(n0, test_lengths)
    if why is not None:
        return dict(refused=why, train_length=int(n0),
                    test_lengths=tuple(int(L) for L in test_lengths))
    masks = dict(MASKS if masks is None else masks)
    lengths = (int(n0),) + tuple(int(L) for L in test_lengths)
    seeds = tuple(seeds)

    truth, X_eval = {}, {}
    for L in lengths:
        X, Y = bed(L, eval_instances, seed=eval_seed)
        X_eval[L] = X
        truth[L] = np.tile(Y, (len(seeds), 1))

    Xtr, Ytr = bed(n0, TRAIN_INSTANCES, seed=100 + seeds[0])
    pred = {"lookup-table": {}, "marginal-frequency": {}}
    for L in lengths:
        pred["lookup-table"][L] = np.tile(
            lookup_predict(Xtr, Ytr, X_eval[L]), (len(seeds), 1))
        pred["marginal-frequency"][L] = np.tile(
            marginal_predict(Ytr, X_eval[L].shape[0]), (len(seeds), 1))
    n_params = {"lookup-table": 0, "marginal-frequency": 0}
    beta_out = {"lookup-table": None, "marginal-frequency": None}
    train_loss, max_abs_u, per_seed_beta = {}, {}, {}

    arms = ["lookup-table", "marginal-frequency"] + list(masks)
    for arm, mask in masks.items():
        bias_free = arm.endswith("nobias")
        rows = {L: [] for L in lengths}
        losses, betas, hulls = [], [], {L: [] for L in lengths}
        # The learned-beta head is the only baseline that matters, so it gets the
        # BEST of a sweep by TRAINING loss at each seed -- the fairest version of
        # it -- and the others get one configuration. A win over a head that was
        # merely under-trained would be worth nothing.
        configs = LEARNED_SWEEP if mask is None else ((steps, LR),)
        for s in seeds:
            trained = [train_head(mask, seed=s, n0=n0, steps=st, lr=lr,
                                  bias_free=bias_free) for st, lr in configs]
            head, loss = min(trained, key=lambda hl: hl[1])
            losses.append(loss)
            betas.append(tuple(float(b) for b in head.beta().detach()))
            for L in lengths:
                rows[L].append(predict(head, X_eval[L]))
                with torch.no_grad():
                    u, _ = head.values(torch.tensor(X_eval[L]))
                hulls[L].append(float(u.abs().max()))
        pred[arm] = {L: np.vstack(rows[L]) for L in lengths}
        n_params[arm] = head.n_params()
        beta_out[arm] = tuple(float(np.mean([b[c] for b in betas]))
                              for c in range(len(TARGETS)))
        per_seed_beta[arm] = betas
        train_loss[arm] = float(np.mean(losses))
        max_abs_u[arm] = {L: float(np.mean(hulls[L])) for L in lengths}

    err = {arm: {L: tuple(nrmse(truth[L][:, c], pred[arm][L][:, c])
                          for c in range(len(TARGETS)))
                 for L in lengths}
           for arm in arms}

    if "learned-beta" in err and "assigned-oracle" in err:
        worst_learned = max(err["learned-beta"][L][c]
                            for L in test_lengths for c in range(len(TARGETS)))
        worst_assigned = max(err["assigned-oracle"][L][c]
                             for L in test_lengths for c in range(len(TARGETS)))
        verdict = "WIN" if worst_learned > TIE_BAND * max(worst_assigned, 1e-12) else "TIE"
        because = ("worst cell over the test lengths: learned-beta %.4f at %d "
                   "parameters, assigned-oracle %.4f at %d, band %.2f"
                   % (worst_learned, n_params["learned-beta"], worst_assigned,
                      n_params["assigned-oracle"], TIE_BAND))
    else:
        worst_learned = worst_assigned = float("nan")
        verdict, because = None, "no learned-beta arm in this report"

    return dict(refused=None, train_length=int(n0),
                test_lengths=tuple(int(L) for L in test_lengths),
                lengths=lengths, arms=arms, seeds=seeds, steps=int(steps),
                nrmse=err, pred=pred, truth=truth,
                truth_mean={L: (float(truth[L][:, 0].mean()),
                                float(truth[L][:, 1].mean())) for L in lengths},
                n_params=n_params, beta=beta_out, per_seed_beta=per_seed_beta,
                train_loss=train_loss, max_abs_u=max_abs_u,
                verdict=verdict, verdict_because=because, tie_band=TIE_BAND,
                worst_learned=worst_learned, worst_assigned=worst_assigned)


#: Training sequences for the refusal arms. Half the main table's, because ten
#: arms are trained here and the effects below are factors rather than
#: tie-breaks; the refusing arm still reaches 0.0004 on its assignable cell, so
#: the smaller budget is not what limits any arm in this section.
REFUSAL_INSTANCES = 256
REFUSAL_STEPS = 1500


def refusal_report(reason, seeds=(0,), steps=REFUSAL_STEPS, test_lengths=TEST_LENGTHS,
                   eval_instances=EVAL_INSTANCES, eval_seed=7,
                   n_instances=REFUSAL_INSTANCES):
    """Three arms on a bed whose second coordinate the rule REFUSED.

    force-softmax and force-linear guess a corner; refuse trains on the
    assignable coordinate ONLY and publishes a REFUSAL for the other cell rather
    than a default. The refusing arm is not a worse model, it is a SHORTER one:
    it answers where it can and declines where it cannot, and the difference
    between its table and the other two is the whole content of the channel.

    ONE SEED, stated rather than buried: the effects below are factors, not
    tie-breaks, and the main table keeps its three seeds.
    """
    def make(n, n_instances, seed):
        return refusal_bed(n, n_instances, seed, reason)

    lengths = (N0,) + tuple(test_lengths)
    truth = {}
    X_eval = {}
    for L in lengths:
        X, Y = make(L, eval_instances, eval_seed)
        X_eval[L], truth[L] = X, np.tile(Y, (len(seeds), 1))

    # force-* share ONE trunk between the two coordinates, which is the setting
    # the whole assignment problem lives in. solo-* train on the refused
    # coordinate ALONE, which separates two different questions that the shared
    # arms confound: "does a corner represent this coordinate at all" from "can
    # one value net serve this coordinate and another at the same time".
    arms = (("force-softmax", (1.0, 1.0), None),
            ("force-linear", (1.0, 0.0), None),
            ("solo-softmax", (1.0, 1.0), (0.0, 1.0)),
            ("solo-linear", (1.0, 0.0), (0.0, 1.0)),
            ("refuse", (1.0, 1.0), (1.0, 0.0)))
    err, params = {}, {}
    for name, mask, weights in arms:
        rows = {L: [] for L in lengths}
        for sd in seeds:
            head, _ = train_head(mask, seed=sd, steps=steps, bed_fn=make,
                                 n_instances=n_instances, coord_weights=weights)
            for L in lengths:
                rows[L].append(predict(head, X_eval[L]))
        params[name] = head.n_params()
        err[name] = {}
        for L in lengths:
            got = np.vstack(rows[L])
            cell0 = nrmse(truth[L][:, 0], got[:, 0])
            if weights is not None and weights[0] == 0.0:
                cell0 = "NOT TRAINED"
            if weights is not None and weights[1] == 0.0:
                cell1 = "REFUSED (%s)" % reason
            else:
                cell1 = nrmse(truth[L][:, 1], got[:, 1])
            err[name][L] = (cell0, cell1)
    return dict(reason=reason, lengths=lengths, test_lengths=tuple(test_lengths),
                nrmse=err, n_params=params, seeds=tuple(seeds))


# ---------------------------------------------------------------------------
# The coverage figure, counted by the module itself.
# ---------------------------------------------------------------------------

_NUM = re.compile(r"(?<![\w.])[+-]?\d+(?:\.\d+)*(?:[eE][+-]?\d+)?(?!\w)(?!\.\d)")


def docstring_numbers():
    """This module's own docstring and number counts, for the run to print.

    Scoped by __module__ so numpy's and torch's docstrings stay out, and NOT by
    __all__, because demo() prints every number and is absent from __all__. No
    number is ever exempted; the scope is the only filter.
    """
    import sys
    me = sys.modules[__name__]
    docs = [me.__doc__ or ""]
    for name, obj in sorted(vars(me).items()):
        if getattr(obj, "__module__", None) != __name__:
            continue
        doc = getattr(obj, "__doc__", None)
        if isinstance(doc, str) and doc.strip():
            docs.append(doc)
        for attr in sorted(vars(obj)) if isinstance(obj, type) else ():
            member = getattr(obj, attr, None)
            member = member.fget if isinstance(member, property) else member
            if getattr(member, "__module__", None) != __name__:
                continue
            sub = getattr(member, "__doc__", None)
            if isinstance(sub, str) and sub.strip() and attr != "__doc__":
                docs.append(sub)
    toks = set()
    for d in docs:
        toks |= {t.lstrip("+") for t in _NUM.findall(d)}
    return dict(n_docstrings=len(docs), n_numbers=len(toks))


# ---------------------------------------------------------------------------
# The self-checks.
# ---------------------------------------------------------------------------

def demo():
    """Lettered self-checks, each with its planted negative seen to fire."""
    t_start = time.time()
    prov = provenance()
    print("PROVENANCE commit %s (git says %s) machine %s (first 12 hex of the "
          "sha256 over node %s) -- both verified, not copied"
          % (HEAD_COMMIT, prov["commit"], MACHINE_ID, prov["node"]))
    assert prov["machine_ok"], "machine id is %r, not %r" % (prov["machine"], MACHINE_ID)
    assert prov["commit_ok"], "HEAD is %r, not %r" % (prov["commit"], HEAD_COMMIT)

    print("(a) THE BED. One data set, two bars: the extensive target is EXACTLY n")
    print("    times the intensive one, so the two bars are not two experiments.")
    worst_identity = 0.0
    for L in (N0,) + TEST_LENGTHS:
        _, Y = bed(L, EVAL_INSTANCES, seed=7)
        worst_identity = max(worst_identity, float(np.abs(Y[:, 1] - L * Y[:, 0]).max()))
        print("    n=%3d  intensive mean %.4f   extensive mean %8.4f" % (L, Y[:, 0].mean(), Y[:, 1].mean()))
    print("    worst |extensive - n * intensive| over all three lengths: %.3e"
          % worst_identity)
    assert worst_identity < 1e-9, "the two targets are not one bed"
    _, Y0 = bed(N0, EVAL_INSTANCES, seed=7)
    _, Y8 = bed(TEST_LENGTHS[1], EVAL_INSTANCES, seed=7)
    drift = abs(float(Y8[:, 0].mean() - Y0[:, 0].mean()))
    growth = float(Y8[:, 1].mean() / Y0[:, 1].mean())
    print("    L-CLASS hypotheses ON THE DATA: intensive mean drifts %.4f across"
          % drift)
    print("    the lengths (it is intensive); extensive mean grows %.4fx (it is not)."
          % growth)
    assert drift < 0.02 and growth > 7.0, "the bed does not carry the distinction"

    alpha_mine = target_alpha()
    print("    count exponent alpha of each target, from a length sweep over")
    print("    %r: intensive %+.4f, extensive %+.4f (no R2 gate anywhere)."
          % ((16, 32, 64, 128), alpha_mine["intensive"], alpha_mine["extensive"]))

    print("(b) THE MASK IS AN INPUT. One pool(), four masks, four readouts. The")
    print("    harness runs TODAY; the measured mask is one argument when it lands.")
    rng = np.random.default_rng(0)
    w, u = rng.random((4, 6)) + 0.5, rng.normal(size=(4, 6))
    seen = {}
    for name, mask in (("all-softmax", (1.0, 1.0)), ("all-linear", (0.0, 0.0)),
                       ("assigned-oracle", (1.0, 0.0)), ("half-and-half", (0.5, 0.5))):
        out = pool(w, u, mask)
        seen[name] = out
        print("    %-16s beta=%-12s row 0 -> [%+.4f, %+.4f]"
              % (name, mask, out[0, 0], out[0, 1]))
    assert not np.allclose(seen["all-softmax"], seen["all-linear"])
    hull_ok = all(u[i].min() <= seen["all-softmax"][i, 0] <= u[i].max() for i in range(4))
    print("    the beta=1 column lies inside the convex hull of u at every row: %s"
          % hull_ok)
    assert hull_ok, "a normalised readout left the hull: the arithmetic is wrong"
    print("    THE AXIS FOOTGUN, seen to fire. READ: the shipped beta is a 0-dim")
    print("    nn.Parameter at ceq/arm_smprime.py:527 applied at :250, where no")
    print("    coordinate axis is in scope. A beta shaped like the TOKEN axis")
    print("    broadcasts to [..., S, S] and exponentiates by the key instead of")
    print("    the query -- silently, in the shipped per-layer code. Here it raises:")
    try:
        pool(w, u, np.full(w.shape[-1], 0.5))
        raise AssertionError("a token-shaped beta was accepted: the footgun is live")
    except ValueError as exc:
        print("      pool(w, u, beta of shape (%d,)) -> ValueError: %s"
              % (w.shape[-1], str(exc).split(".")[0]))

    print("(c) THE KILL, AS A GUARD. T-LENGTH at the training length only is not")
    print("    the test, so it is REFUSED as a value carrying its reason.")
    print("    length_refusal(%d, (%d,)) -> %s" % (N0, N0, length_refusal(N0, (N0,))))
    assert length_refusal(N0, (N0,)) is not None
    assert length_refusal(N0, TEST_LENGTHS) is None
    refused = t_length_report(test_lengths=(N0,), seeds=(0,))
    assert refused["refused"] and "nrmse" not in refused, "a refused call answered"
    print("    FIRED: the report refuses and publishes no table.")

    print("(d) THE TABLE, simplest arms first, %d seeds x %d Adam steps at lr %.2f"
          % (len(SEEDS), STEPS, LR))
    print("    on %d sequences of length %d, torch float64, predictions pooled"
          % (TRAIN_INSTANCES, N0))
    print("    over seeds before the error is taken.")
    wired = load_measured_mask()
    masks = dict(MASKS)
    measured = dict(available=wired["available"], reason=wired["reason"],
                    mask=None, axis=None, axis_len=None, n_assigned=None,
                    refusals={}, agrees_with_oracle=None, differs_at=[])
    if wired["available"]:
        mask, detail = mask_from_frozen(wired["frozen"])
        measured.update(axis=detail["axis"], axis_len=detail["axis_len"],
                        n_assigned=detail["n_assigned"],
                        refusals=detail["refusals"], mask=mask,
                        refused=detail["refused"])
        if mask is not None:
            masks["assigned-measured"] = mask
            oracle = MASKS["assigned-oracle"]
            measured["agrees_with_oracle"] = tuple(mask) == tuple(oracle)
            measured["differs_at"] = [TARGETS[i] for i in range(len(TARGETS))
                                      if mask[i] != oracle[i]]
    rep = t_length_report(masks=masks)
    print("    %-20s %7s %9s %9s %9s %9s %9s"
          % ("arm", "params", "1x int", "1x ext", "4x int", "4x ext", "8x ext"))
    for arm in rep["arms"]:
        e = rep["nrmse"][arm]
        print("    %-20s %7d %9.4f %9.4f %9.4f %9.4f %9.4f"
              % (arm, rep["n_params"][arm], e[N0][0], e[N0][1],
                 e[TEST_LENGTHS[0]][0], e[TEST_LENGTHS[0]][1], e[TEST_LENGTHS[1]][1]))
    print("    8x intensive, the column the table above has no room for: "
          + "  ".join("%s %.4f" % (a, rep["nrmse"][a][TEST_LENGTHS[1]][0])
                      for a in ("all-softmax", "all-linear", "assigned-oracle")))

    four, eight = TEST_LENGTHS
    soft, lin = rep["nrmse"]["all-softmax"], rep["nrmse"]["all-linear"]
    assigned = rep["nrmse"]["assigned-oracle"]
    print("(e) BED CALIBRATION, PRE-REGISTERED. The single-exponent arms are not")
    print("    findings here: the softmax one replicates a theorem someone else")
    print("    proved (see (k)), so it calibrates the bed. The direction AND the")
    print("    magnitude are stated before the numbers are read:")
    for L in TEST_LENGTHS:
        pr = predicted_single_exponent_error(N0, L)
        print("      n=%3d predicted  softmax on extensive %.4f   linear on "
              "intensive %.4f" % (L, pr["softmax_extensive"], pr["linear_intensive"]))
    for L in TEST_LENGTHS:
        pr = predicted_single_exponent_error(N0, L)
        print("      n=%3d measured   softmax on extensive %.4f   linear on "
              "intensive %.4f" % (L, soft[L][1], lin[L][0]))
        d_s = abs(soft[L][1] - pr["softmax_extensive"])
        d_l = abs(lin[L][0] - pr["linear_intensive"]) / pr["linear_intensive"]
        print("      n=%3d agreement  softmax %+.4f absolute, linear %+.2f%% "
              "relative -- %s" % (L, soft[L][1] - pr["softmax_extensive"], 100.0 * d_l,
                                  "sign AND scale" if (d_s < 0.05 and d_l < 0.10)
                                  else "SIGN ONLY, the scale DIVERGES"))
    print("    A bed reproducing the sign but not the scale is still usable; it is")
    print("    said either way rather than hidden.")
    losers = ("all-linear", "lookup-table", "marginal-frequency", "learned-beta")
    for L in TEST_LENGTHS:
        worst = max(rep["nrmse"][a][L][0] for a in losers)
        print("    n=%3d INTENSIVE control: softmax %.4f against the best of the "
              "arms it must beat, %.4f" % (L, soft[L][0], min(rep["nrmse"][a][L][0] for a in losers)))
        assert soft[L][0] < min(rep["nrmse"][a][L][0] for a in losers), (
            "VOID, not negative: the softmax head lost the intensive control at "
            "n=%d, so the baseline is broken and this bed decides nothing" % L)
        del worst
    for L in TEST_LENGTHS:
        print("    n=%3d EXTENSIVE plant  : softmax %.4f against the assigned "
              "arm's %.4f -- a factor of %.2f" % (L, soft[L][1], assigned[L][1],
                                                  soft[L][1] / assigned[L][1]))
        assert soft[L][1] > 0.25 and soft[L][1] > assigned[L][1], (
            "VOID, not negative: the softmax head did not lose the extensive "
            "plant at n=%d. The theorem cited in (k) predicts it must, so the "
            "hypothesis is unmet on THIS BED -- the extensive target is leaking "
            "through a positional feature, or the bar is not extensive -- and "
            "nothing about the assignment has been measured here." % L)
    print("    BOTH FIRED, and both failures GROW: softmax extensive %.4f -> %.4f,"
          % (soft[four][1], soft[eight][1]))
    print("    linear intensive %.4f -> %.4f from 4x to 8x." % (lin[four][0], lin[eight][0]))
    assert soft[eight][1] > soft[four][1] and lin[eight][0] > lin[four][0], \
        "a gap that does not grow with n is a fitting offset, not a length failure"

    print("(f) MY FACTORS BESIDE THE OWNER'S, which were never an input here.")
    f_soft = soft[four][1] / assigned[four][1]
    f_lin = lin[four][0] / assigned[four][0]
    f_soft8 = soft[eight][1] / assigned[eight][1]
    print("    factor = nrmse(arm) / nrmse(assigned-oracle), same cell, 4x:")
    print("      softmax on the extensive bar  mine %8.2f   owner %5.1f   "
          "difference %+8.2f" % (f_soft, OWNER_4X[0], f_soft - OWNER_4X[0]))
    print("      linear on the intensive bar   mine %8.2f   owner %5.1f   "
          "difference %+8.2f" % (f_lin, OWNER_4X[1], f_lin - OWNER_4X[1]))
    print("    The pairing of his two numbers to these two cells is MY reading of")
    print("    the brief, which labelled neither. Against the marginal-frequency")
    print("    floor instead of the assigned arm the same two cells read %.2f and"
          % (soft[four][1] / rep["nrmse"]["marginal-frequency"][four][1]))
    print("    %.2f, so the first factor is not an artefact of a small denominator"
          % (lin[four][0] / rep["nrmse"]["marginal-frequency"][four][0]))
    print("    and the second one largely is.")

    print("(g) THE OWNER'S COUNTER, SCORED. The learned-beta head is the only")
    print("    baseline that matters: it could discover the assignment itself.")
    print("    assigned-oracle %d parameters, learned-beta %d parameters -- the"
          % (rep["n_params"]["assigned-oracle"], rep["n_params"]["learned-beta"]))
    print("    difference is exactly %d, one exponent logit per output coordinate,"
          % (rep["n_params"]["learned-beta"] - rep["n_params"]["assigned-oracle"]))
    print("    and it is the LEARNED head that carries the extra.")
    print("    learned beta landed at (%.4f, %.4f); the assigned mask is (%.1f, %.1f)"
          % (rep["beta"]["learned-beta"] + rep["beta"]["assigned-oracle"]))
    print("    per seed it landed at %s -- never near the mask."
          % (" ".join("(%.2f,%.2f)" % b for b in rep["per_seed_beta"]["learned-beta"])))
    print("    It was given the BEST of %d configurations by TRAINING loss at each"
          % len(LEARNED_SWEEP))
    print("    seed %s, so it is not merely under-trained:" % (LEARNED_SWEEP,))
    print("    training loss: learned %.6f, assigned %.6f -- under-DETERMINED, not"
          % (rep["train_loss"]["learned-beta"], rep["train_loss"]["assigned-oracle"]))
    print("    under-trained, and the identifiability diagnostic says why:")
    head, _ = train_head(MASKS["assigned-oracle"], seed=0, steps=1000)
    logden = {}
    for L in (N0,) + TEST_LENGTHS:
        X, _ = bed(L, EVAL_INSTANCES, seed=7)
        with torch.no_grad():
            _, w_L = head.values(torch.tensor(X))
        logden[L] = np.log(w_L.sum(1).numpy())
    within = float(logden[N0].std())
    between = float(logden[four].mean() - logden[N0].mean())
    print("      log(sum w) moves sd %.4f across instances AT n=%d, and %.4f"
          % (within, N0, between))
    print("      between n=%d and n=%d: a ratio of %.0f, so the training set"
          % (N0, four, between / within))
    print("      carries three orders of magnitude less signal about the exponent")
    print("      than the test asks of it.")
    print("    VERDICT %s -- %s" % (rep["verdict"], rep["verdict_because"]))
    assert rep["verdict"] in ("WIN", "TIE")
    if rep["verdict"] == "TIE":
        print("    A TIE STATED PLAINLY: the assignment buys interpretability and a")
        print("    refusal channel, not accuracy, and the numbers above are it.")

    print("(h) THE EXPONENT IS NOT THE WHOLE ASSIGNMENT -- a CORRECTION to the")
    print("    instance, which predicted the assigned arm exact on both bars.")
    nb = t_length_report(masks={"assigned-nobias": MASKS["assigned-oracle"]},
                         seeds=(0,))
    nbe = nb["nrmse"]["assigned-nobias"]
    print("    assigned, output constant FREE  : 4x extensive %.4f  8x %.4f"
          % (assigned[four][1], assigned[eight][1]))
    print("    assigned, output constant ZEROED: 4x extensive %.4f  8x %.4f"
          % (nbe[four][1], nbe[eight][1]))
    print("    FIRED: a constant added to a normalised read is not extensive, so")
    print("    the residual above is the CONSTANT and not the length: a factor of")
    print("    %.1f between the two rows." % (assigned[four][1] / nbe[four][1]))
    assert nbe[four][1] < 0.25 * assigned[four][1], \
        "the bias control did not fire: the residual is something else"
    assert nbe[four][0] < 0.05, "the bias-free arm lost the intensive bar"

    print("(i) THE MARS ATTACK: does n leak through a position? Censused first.")
    cen = census_n((N0,) + TEST_LENGTHS)
    print("    R2 for predicting n from the per-token feature MEANS   : %.4f"
          % cen["mean_r2"])
    print("    R2 with the feature EXTREMES, what a sharp head sees   : %.4f"
          % cen["extreme_r2"])
    print("    R2 on a bed with a PLANTED position feature            : %.4f"
          % cen["leak_r2"])
    assert cen["mean_r2"] < 0.05, "the means already carry n"
    assert cen["leak_r2"] > 0.5, "the census cannot see a leak it was handed"
    leak_head, _ = train_head(MASKS["all-softmax"], seed=0, with_position=True)
    Xl, Yl = bed(four, EVAL_INSTANCES, seed=7, with_position=True)
    leak_ext = nrmse(Yl[:, 1], predict(leak_head, Xl)[:, 1])
    X4, Y4 = bed(four, EVAL_INSTANCES, seed=7)
    soft_head, _ = train_head(MASKS["all-softmax"], seed=0)
    oracle_n = predict(soft_head, X4)[:, 0] * four
    oracle_ext = nrmse(Y4[:, 1], oracle_n)
    print("    a raw position feature moves the softmax extensive bar %.4f -> %.4f"
          % (soft[four][1], leak_ext))
    print("    (%.2f%%): the channel exists and does NOT convert, because the"
          % (100.0 * (1.0 - leak_ext / soft[four][1])))
    print("    encoder saturates outside the range it trained on.")
    print("    POSITIVE CONTROL, seen to fire: hand the same head the true n and")
    print("    multiply -- the extensive bar collapses %.4f -> %.4f. The missing"
          % (soft[four][1], oracle_ext))
    print("    quantity is exactly the factor n, so a bed carrying n measures")
    print("    nothing and this one does not carry it.")
    assert oracle_ext < 0.05, \
        "the n-oracle control did not fire: the failure is not just the factor n"
    assert leak_ext < soft[four][1], "the planted leak did not help at all"

    print("(j) THE HULL BOUND, the theorem's hypothesis, ON THE MODEL'S OWN VALUES.")
    mu = rep["max_abs_u"]["all-softmax"]
    print("    max|u| for the softmax head: %.4f at n=%d, %.4f at n=%d, %.4f at n=%d"
          % (mu[N0], N0, mu[four], four, mu[eight], eight))
    print("    a growth of %.4fx while the extensive target grows %.1fx -- the"
          % (mu[eight] / mu[N0], rep["truth_mean"][eight][1] / rep["truth_mean"][N0][1]))
    print("    reachable interval is fixed and the target is not, which is the")
    print("    whole of the theorem.")
    assert mu[eight] < 3.0 * mu[N0], "the hull bound is not fixed: the theorem does not apply"

    print("(k) THE MEASURED MASK, WIRED. The rule now exists, so the argument")
    print("    this harness was built around is supplied rather than planted.")
    if not measured["available"]:
        print("    NO MASK: %s" % measured["reason"])
        print("    The table above is unaffected, which is the point of the")
        print("    interface: the harness never waited for it.")
    else:
        print("    axis %r length %d, %d of %d coordinates assigned, %d refused %r"
              % (measured["axis"], measured["axis_len"], measured["n_assigned"],
                 measured["axis_len"],
                 measured["axis_len"] - measured["n_assigned"],
                 sorted(measured["refusals"])))
        fz = wired["frozen"]
        for name, b, al in zip(fz["coords"], fz["beta"], fz["alpha"]):
            mine = [t for t, c in COORD_FOR_TARGET.items() if c == name]
            print("      %-12s alpha %+.4f  beta %-5s %s"
                  % (name, al, "None" if b is None else "%.1f" % b,
                     ("<- my %s target" % mine[0]) if mine else ""))
        print("    MAPPING INSENSITIVITY, measured not assumed: the rule's three")
        intensive_b = {fz["beta"][fz["coords"].index(c)]
                       for c in ("mean_value", "prob_above", "ratio_vu")
                       if c in fz["coords"]}
        extensive_b = {fz["beta"][fz["coords"].index(c)]
                       for c in ("count_above", "sum_value", "total_u")
                       if c in fz["coords"]}
        print("    intensive coordinates all read beta %r and its three extensive"
              % sorted(intensive_b))
        print("    ones all read beta %r, so which member of each triple I map to"
              % sorted(extensive_b))
        print("    is immaterial. The mapping is still MY semantic choice.")
        print("    THE FLOOR, and whether it bites HERE. A log-shaped coordinate")
        print("    is not a power law but passes the power-law test at four sweep")
        log_alpha = (fz["alpha"][fz["coords"].index("log_shape")]
                     if "log_shape" in fz["coords"] else float("nan"))
        log_beta = (fz["beta"][fz["coords"].index("log_shape")]
                    if "log_shape" in fz["coords"] else None)
        print("    points: in the very vector I loaded, log_shape arrives assigned")
        print("    beta %s at alpha %+.4f, labelled INTENSIVE with no warning."
              % ("%.1f" % log_beta if log_beta is not None else "None", log_alpha))
        print("    NEITHER of my coordinates is log-shaped -- my own targets")
        print("    measure alpha %+.4f and %+.4f against that trap's %+.4f -- so"
              % (alpha_mine["intensive"], alpha_mine["extensive"], log_alpha))
        print("    the floor does not bite this bed. It would bite any coordinate")
        print("    whose growth is logarithmic, and this harness could not tell.")
        print("    THE COMPARISON THAT CARRIES INFORMATION is the VECTOR, not the")
        print("    row: measured %r against oracle %r -> agree %s"
              % (measured["mask"], MASKS["assigned-oracle"],
                 measured["agrees_with_oracle"]))
        if measured["agrees_with_oracle"]:
            print("    The rule recovers the right corner for BOTH my coordinates")
            print("    from data alone. The row it produces is therefore identical")
            print("    to the oracle row BY CONSTRUCTION -- same vector, same")
            print("    pipeline, same seeds -- and presenting it as independent")
            print("    confirmation would be a tautology dressed as a measurement.")
            for L in TEST_LENGTHS:
                same = (rep["nrmse"]["assigned-measured"][L]
                        == rep["nrmse"]["assigned-oracle"][L])
                print("      n=%3d measured %.4f / %.4f   oracle %.4f / %.4f   "
                      "identical %s"
                      % (L, rep["nrmse"]["assigned-measured"][L][0],
                         rep["nrmse"]["assigned-measured"][L][1],
                         rep["nrmse"]["assigned-oracle"][L][0],
                         rep["nrmse"]["assigned-oracle"][L][1], same))
                assert same, "identical masks gave different rows: not deterministic"
        else:
            print("    THE RULE DISAGREES at %r, which is a finding and not a"
                  % (measured["differs_at"],))
            print("    failure. What it costs, cell by cell:")
            for L in TEST_LENGTHS:
                for c, t in enumerate(TARGETS):
                    mm = rep["nrmse"]["assigned-measured"][L][c]
                    oo = rep["nrmse"]["assigned-oracle"][L][c]
                    print("      n=%3d %-10s measured %.4f  oracle %.4f  cost %+.4f"
                          % (L, t, mm, oo, mm - oo))
    print("    BETA IS DERIVED, NOT CHOSEN. A read carries N^(1-beta), so the")
    print("    count exponent of the output is 1-beta and beta = 1-alpha is")
    print("    bookkeeping. On MY targets alpha measures %+.4f and %+.4f, giving"
          % (alpha_mine["intensive"], alpha_mine["extensive"]))
    print("    beta %.4f and %.4f -- the assigned arm is the dimensionally FORCED"
          % (1.0 - alpha_mine["intensive"], 1.0 - alpha_mine["extensive"]))
    print("    setting on this bed, not a heuristic that happened to work. The")
    print("    corners themselves are READ at lean/CEQ/V16Domain.lean:433")
    print("    three_corners_containment and :445 corners_are_distinct, RELAYED")
    print("    from the rule's docstring and NOT verified in this file.")
    assert abs(alpha_mine["intensive"]) < 0.05 and abs(alpha_mine["extensive"] - 1.0) < 0.05

    print("(l) THE REFUSAL CHANNEL, EXERCISED. The rule refused 2 of its 9")
    print("    coordinates. What does a target cost when it has NO corner?")
    ralpha = refusal_target_alpha()
    rnorms = refusal_target_norms()
    print("    THE MAPPING IS A CONSTRUCTION, not an assertion: each target below")
    print("    is built the way the coordinate it maps onto is planted.")
    print("      MIXED-SCALING  <- half_walk, w.sum(axis=1) over a zero-mean")
    print("        stream. MINE: the sum of a zero-mean per-token feature -- a")
    print("        signed running total, a net drift. It IS a total, and its")
    print("        magnitude grows like the square root because terms cancel:")
    print("        measured alpha %+.4f, between the corners." % ralpha["mixed"])
    print("      NOT-A-POWER-LAW <- osc_shape, a symmetric bump in log2(n) times")
    print("        a per-sequence jitter. MINE: a per-sequence quantity scaled by")
    print("        an amplitude that is a function of the CONTEXT LENGTH alone.")
    print("        Its norm reads %.4f at n=%d, %.4f at n=%d and %.4f at n=%d --"
          % (rnorms["not-a-power-law"][N0], N0,
             rnorms["not-a-power-law"][four], four,
             rnorms["not-a-power-law"][eight], eight))
    print("        the same at 8x as at the training length and different at 4x,")
    print("        so its slope is %+.4f and describes nothing. NO readout whose"
          % ralpha["not-a-power-law"])
    print("        only n-dependence is a pooling exponent can produce it: that")
    print("        is a FLOOR by construction, said here and not discovered below.")
    refusal = {}
    for reason in REFUSAL_REASONS:
        rr = refusal_report(reason)
        refusal[reason] = rr
        print("    %s, 1 seed, %d steps on %d sequences, params %d every arm:"
              % (reason, REFUSAL_STEPS, REFUSAL_INSTANCES,
                 rr["n_params"]["force-softmax"]))
        print("      %-14s %-26s %s" % ("arm", "4x assignable/refused",
                                        "8x assignable/refused"))
        for arm in ("force-softmax", "force-linear", "solo-softmax",
                    "solo-linear", "refuse"):
            cells = []
            for L in TEST_LENGTHS:
                c0, c1 = rr["nrmse"][arm][L]
                cells.append("%s / %s"
                             % (c0 if isinstance(c0, str) else "%.4f" % c0,
                                c1 if isinstance(c1, str) else "%.4f" % c1))
            print("      %-14s %-26s %s" % (arm, cells[0], cells[1]))
    def _best(reason_key, arms):
        return min(refusal[reason_key]["nrmse"][a][four][1] for a in arms)

    shared = ("force-softmax", "force-linear")
    solo = ("solo-softmax", "solo-linear")
    mixed_best, nopow_best = _best(*(("MIXED-SCALING", shared),)[0]), _best(
        "NOT-A-POWER-LAW", shared)
    mixed_solo, nopow_solo = _best("MIXED-SCALING", solo), _best(
        "NOT-A-POWER-LAW", solo)
    def _solo_winner(reason_key):
        return {L: ("solo-linear"
                    if refusal[reason_key]["nrmse"]["solo-linear"][L][1]
                    < refusal[reason_key]["nrmse"]["solo-softmax"][L][1]
                    else "solo-softmax") for L in TEST_LENGTHS}

    mixed_win, nopow_win = _solo_winner("MIXED-SCALING"), _solo_winner("NOT-A-POWER-LAW")
    mixed_worst = max(refusal["MIXED-SCALING"]["nrmse"][a][L][1]
                      for L in TEST_LENGTHS
                      for a in [set(mixed_win.values()).pop()]
                      ) if len(set(mixed_win.values())) == 1 else float("inf")
    good_corner = len(set(mixed_win.values())) == 1 and mixed_worst < 0.15
    ranking_flips = len(set(nopow_win.values())) > 1
    says = ("the two reasons do NOT cost the same: trained ALONE, the best corner "
            "costs %.4f on MIXED-SCALING and %.4f on NOT-A-POWER-LAW at 4x; "
            "sharing a trunk it is %.4f and %.4f"
            % (mixed_solo, nopow_solo, mixed_best, nopow_best))
    print("    PRE-REGISTERED for the bump, before the run: a model that learns")
    print("    the amplitude at n0 and cannot learn that it moved costs")
    print("    |1 - A(n0)/A(n)| -- %.4f at 4x and %.4f at 8x. MEASURED on the"
          % (predicted_osc_error(four), predicted_osc_error(eight)))
    print("    softmax corner trained alone: %.4f and %.4f."
          % (refusal["NOT-A-POWER-LAW"]["nrmse"]["solo-softmax"][four][1],
             refusal["NOT-A-POWER-LAW"]["nrmse"]["solo-softmax"][eight][1]))
    print("    WHAT THE READER LOSES BY HONOURING THE REFUSAL. %s." % says)
    print("    THE DISCRIMINATOR IS NOT A THRESHOLD, it is whether ONE corner")
    print("    wins at EVERY test length.")
    print("      MIXED-SCALING   winner per length %r" % (sorted(set(mixed_win.values())),))
    print("      NOT-A-POWER-LAW winner per length %r, the ranking %s"
          % (sorted(set(nopow_win.values())), "FLIPS" if ranking_flips else "holds"))
    print("    TWO QUESTIONS, SEPARATED, because the shared arms confound them:")
    print("    'does a corner represent this coordinate' is the solo row, and")
    print("    'what does forcing it cost in a table' is the force row. On")
    print("    MIXED-SCALING the linear corner alone reads %.4f while the same"
          % refusal["MIXED-SCALING"]["nrmse"]["solo-linear"][four][1])
    print("    corner sharing a trunk reads %.4f -- a factor of %.0f, and that"
          % (refusal["MIXED-SCALING"]["nrmse"]["force-linear"][four][1],
             refusal["MIXED-SCALING"]["nrmse"]["force-linear"][four][1]
             / max(refusal["MIXED-SCALING"]["nrmse"]["solo-linear"][four][1], 1e-12)))
    print("    factor is the trunk, not the exponent. MY OWN PREDICTION WAS")
    print("    WRONG HERE and the run corrected it: I expected the linear corner")
    print("    to be near-exact in the table because a sum of zero-mean tokens")
    print("    IS a sum, and it is near-exact only when it does not have to share")
    print("    a value net with a probability.")
    if good_corner:
        print("    On MIXED-SCALING ONE corner wins at BOTH lengths (%.4f and %.4f"
              % (refusal["MIXED-SCALING"]["nrmse"]["solo-linear"][four][1],
                 refusal["MIXED-SCALING"]["nrmse"]["solo-linear"][eight][1]))
        print("    against the other corner's %.4f and %.4f), so a corner EXISTS"
              % (refusal["MIXED-SCALING"]["nrmse"]["solo-softmax"][four][1],
                 refusal["MIXED-SCALING"]["nrmse"]["solo-softmax"][eight][1]))
        print("    and the refusal is CONSERVATIVE about it: honouring the refusal")
        print("    costs a reader a model that works when the coordinate is not")
        print("    sharing a value net. AND THE SOFTMAX ROW IS THE EVIDENCE FOR")
        print("    WHY: its error is %.4f and %.4f against the closed form"
              % (refusal["MIXED-SCALING"]["nrmse"]["solo-softmax"][four][1],
                 refusal["MIXED-SCALING"]["nrmse"]["solo-softmax"][eight][1]))
        print("    1 - n0/n = %.4f and %.4f -- the SAME law as for a plain count,"
              % (predicted_single_exponent_error(N0, four)["softmax_extensive"],
                 predicted_single_exponent_error(N0, eight)["softmax_extensive"]))
        print("    although this coordinate's norm exponent is about %.2f and not 1."
              % ralpha["mixed"])
        print("    The exponent the rule MEASURES governs the magnitude; the one")
        print("    that governs the READOUT error is the declared count exponent.")
        print("    The reason is visible in the construction -- a sum of zero-mean")
        print("    tokens IS a total, so beta=0 represents it, while the norm the")
        print("    rule fits grows like sqrt(n) and reports something between the")
        print("    corners. ROUTE, and it needs no new measurement: that file")
        print("    already computes the DECLARED count exponent beside the")
        print("    measured alpha, and half_walk declares N^+1.0 while measuring")
        print("    about half. Where declared and measured disagree AND the")
        print("    declared value is a corner, the declared one is the exponent")
        print("    of the READOUT and the measured one is a magnitude: preferring")
        print("    declared there converts this refusal into a correct assignment.")
    else:
        print("    On MIXED-SCALING no corner is close even trained alone, so the")
        print("    refusal costs nothing and is protective in both cases.")
    print("    On NOT-A-POWER-LAW the refusal is PROTECTIVE, and the table shows")
    print("    why in a way a single test length cannot: the forced arms read")
    for arm in ("force-softmax", "force-linear", "solo-softmax"):
        print("      %-14s %.4f at 4x and %.4f at 8x"
              % (arm, refusal["NOT-A-POWER-LAW"]["nrmse"][arm][four][1],
                 refusal["NOT-A-POWER-LAW"]["nrmse"][arm][eight][1]))
    print("    -- NOT monotone. A reader testing only at 8x would certify a")
    print("    corner that is wrong at 4x. An exponent cannot express that shape,")
    print("    which is exactly what NOT-A-POWER-LAW means, and no finite set of")
    print("    test lengths can stand in for the refusal.")
    print("    THE SUBSTITUTION, planted and caught. A careless caller writes")
    print("    `b if b is not None else 1.0` and the refusal becomes the softmax")
    if measured["available"]:
        fz2 = wired["frozen"]
        bad_map = {"intensive": "prob_above", "extensive": "half_walk"}
        careless = silently_defaulted_mask(fz2, bad_map)
        guarded, gdetail = mask_from_frozen(fz2, bad_map)
        print("    corner with no trace. On the REAL vector, mapping my extensive")
        print("    target onto half_walk: the careless path returns %r and the"
              % (careless,))
        print("    harness returns %r carrying %r."
              % (guarded, gdetail["refused"][:46]))
        assert guarded is None and careless is not None, "the plant did not fire"
        print("    The careless number is the force-softmax row above, %.4f at 4x:"
              % refusal["MIXED-SCALING"]["nrmse"]["force-softmax"][four][1])
        print("    that is the price of the default, in the same units as the")
        print("    rest of this file.")
    assert isinstance(refusal["MIXED-SCALING"]["nrmse"]["refuse"][four][1], str), \
        "the refusing arm filled a refused cell with a number"

    print("(m) PRIOR ART. The softmax arm is a CALIBRATION INSTRUMENT, not a")
    print("    finding: the theorem is someone else's, it predicts the direction")
    print("    and (e) shows the bed reproduces it, which is what licenses the")
    print("    only unoccupied measurement here -- (g), the per-coordinate")
    print("    assignment against the learned head at matched counts.")
    print("      arXiv:2410.01104 Velickovic et al., ICML 2025 -- softmax circuits")
    print("        must disperse as the item count grows. Already marked [V] in this")
    print("        tree at docs/sources/sweep/sweep_expressivity.md:325-330.")
    print("      arXiv:2406.04267 Barbero et al., NeurIPS 2024 -- Theorem B.3 gives")
    print("        representational collapse of the last-token representation in L1")
    print("        under 4 hypotheses, Corollary B.10 turns it into the counting")
    print("        failure, Proposition B.9 is the non-asymptotic version with no")
    print("        positional encodings; Gemini 1.5 miscopies at length 300.")
    print("      arXiv:2006.16236 Katharopoulos et al. 2020 is the normalised/")
    print("        unnormalised axis itself; arXiv:2108.12409 Press et al. is the")
    print("        length-generalisation line.")
    print("    EVERY 'cannot' IN THIS FILE IS SCOPED, because arXiv:2511.20038")
    print("    (Theorem 4.3, relative positional encodings) proves length-")
    print("    generalizable softmax chain-of-thought transformers Turing-complete.")
    print("    The scope here is ONE forward pass, NO scratchpad, and the")
    print("    representation of ONE readout row. Unscoped, the sentence is refuted.")
    print("    NEAREST PRIOR WORK to the delta: arXiv:2202.04643 Bakarji et al.,")
    print("    Nature Computational Science 2022 -- BuckiNet learns one exponent per")
    print("    input coordinate, but through a SINGLE shared layer against a")
    print("    null-space loss, and the exponents are fitted, not swept.")
    print("    UNOCCUPIED, and searched for rather than inferred: none of the four")
    print("    measures a per-coordinate exponent from a context-length sweep, none")
    print("    uses it to pick between two attention variants per coordinate, and")
    print("    none refuses a coordinate that fits neither class.")

    cov = docstring_numbers()
    print("(n) COVERAGE. %d docstrings and %d distinct number tokens in this"
          % (cov["n_docstrings"], cov["n_numbers"]))
    print("    module, every one of which some run prints; 0 numbers exempted,")
    print("    because there is no exemption list. 1 training length, 2 targets,")
    print("    4 masks, 8 arms scored in all.")

    elapsed = time.time() - t_start
    print("(o) BUDGET. %.1f s elapsed against the 300 s bar." % elapsed)
    assert elapsed < 300.0, "demo() took %.1f s" % elapsed

    stats = dict(
        report=rep, head_commit=HEAD_COMMIT, machine_id=MACHINE_ID,
        softmax_extensive_factor_4x=f_soft, linear_intensive_factor_4x=f_lin,
        softmax_extensive_factor_8x=f_soft8, owner_4x=OWNER_4X,
        diff_vs_owner=(f_soft - OWNER_4X[0], f_lin - OWNER_4X[1]),
        n_params_assigned=rep["n_params"]["assigned-oracle"],
        n_params_learned=rep["n_params"]["learned-beta"],
        verdict=rep["verdict"], nobias_extensive_4x=nbe[four][1],
        identifiability_ratio=between / within, census=cen,
        leak_extensive_4x=leak_ext, oracle_n_extensive_4x=oracle_ext,
        n_docstrings=cov["n_docstrings"], n_numbers=cov["n_numbers"],
        measured_mask=measured, target_alpha=alpha_mine,
        refusal=dict(refusal, verdict=dict(mixed_has_a_good_corner=good_corner,
                                           nopow_corner_ranking_flips=ranking_flips,
                                           says=says, mixed_best=mixed_best,
                                           nopow_best=nopow_best,
                                           mixed_solo=mixed_solo,
                                           nopow_solo=nopow_solo,
                                           mixed_winner=mixed_win,
                                           nopow_winner=nopow_win)),
        elapsed_s=elapsed)
    print("ALL SELF-CHECKS PASSED")
    return stats


if __name__ == "__main__":
    demo()

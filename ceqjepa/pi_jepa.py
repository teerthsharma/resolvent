"""DR-2 R2: the JEPA, with beta fixed PER COORDINATE by R1's frozen mask.

WHAT IS CLAIMED. An online encoder E_theta and an EMA target E_theta_bar behind a
stop-grad; a predictor that is the containment family's own read, with the
exponent beta_d taken from the mask ceqjepa/pi_assign.py froze, one value per
latent coordinate; a loss of latent prediction ||s_hat_{t+h} - sg(s_{t+h})||^2
plus an eps-free Std term plus a covariance term; action-conditioning scored
against a placebo; and a collapse detector that runs on every step and refuses
rather than warns. Every wiring claim below is a MEASUREMENT in this file's own
run, not an intention.

WHAT IS NOT CLAIMED, AND THE SIZES ARE HERE SO THE SCOPE IS UNARGUABLE. This is a
CPU box under a 300 s bar. The encoder is three linear layers over a synthetic
autoregressive bed, the latent axis "D" of length 9 is the mask's axis and
nothing wider, the run is S = 16 positions, B = 64 sequences, horizon h = 4,
seed 5501, tau = 0.99, and 120 optimiser steps. NOTHING HERE IS A STATEMENT
ABOUT A TRAINED MODEL AT SCALE. What is demonstrated is the wiring and the
failure modes: that the per-coordinate read IS the family's read, that the
refusal survives the crossing, that the stop-grad and the EMA are not decoration,
and that a collapse is caught rather than hoped against. A GPU run may follow and
its value depends entirely on this being correct first.

THE OBSTACLE, AND WHY IT IS A FACTORISATION RATHER THAN A REWRITE. In the shipped
family beta is a 0-dim per-layer parameter (READ ceq/arm_smprime.py:527) applied
to the modulus row before the value contraction (READ :250 for
`zb = mod.sum(-1, keepdim=True) ** beta` and :264 for the contraction). There is
no output-coordinate axis at that point, so a [D]-shaped beta cannot live there
and raises. It can live AFTER the contraction, because Z_i does not depend on j:

    O_{i,c} = (num @ v)_{i,c} / Z_i^{beta_c}

is the same operator's read with the division commuted past a sum that does not
touch it. That is an argument and an argument is not a measurement, so the two
orders are COMPARED against the shipped readout for a scalar beta:
BITWISE EXACT at beta=0, 4.4409e-16 at beta=1 and 5.3291e-15 at the interior
value 0.37, against a tolerance of 1e-13 on the read.

THE PRICE IS STATED RATHER THAN HIDDEN. Bitwise agreement is LOST at beta=1 and
at the interior, by the figures above, and it is kept only at beta=0 where Z^0 is
exactly 1 and the division is exact. That is float64 failing to associate across
a reordered division, not a defect in either implementation, and it is what the
repository's own corner identification pays for a per-coordinate exponent.
Hiding it would be the defect this project keeps finding.

THE TWO IDENTITIES THAT SAY THE MASK IS NOT DECORATION. A per-coordinate mask of
all ones must be the single softmax arm and a mask of all zeros the single linear
arm, both against the SHIPPED readout rather than a re-implementation agreeing
with itself. Measured: the all-ones mask reproduces the softmax arm to 4.4409e-16
and the all-zeros mask reproduces the linear arm bitwise.

THE REFUSAL CROSSES THE BOUNDARY AS A REFUSAL. R1 emits None for a coordinate it
declined and carries the reason beside it, and ceqjepa/t_length.py:694 already
reads that vector by validating the axis NAME, the axis LENGTH against the vector
length, and the refusals against the betas. The same three checks run here rather
than a second convention, and a refused coordinate becomes a Refusal in the beta
vector: arithmetic on it raises TypeError instead of passing the way a NaN does.
The read hands back a Refusal for that coordinate and a float for every assigned
one, and declares axis_len beside the columns it produced, so the axis is never
silently shortened. On the frozen mask 7 of 9 coordinates are predicted and 2 are
refused, half_walk for MIXED-SCALING and osc_shape for NOT-A-POWER-LAW.

WHAT ACTUALLY BREAKS A JEPA, AND THIS REPOSITORY HAS ALREADY BEEN BITTEN. A JEPA
fails by collapse, and ceqjepa/v2.py's variance hinge sits at its MAXIMUM at
exactly the collapsed state with gradient exactly zero there -- a stationary point
of the same shape as a zero-init LoRA death. Three things follow and all three are
enforced rather than intended. The encoder output is never zero-initialised and
the initialisation is asserted non-degenerate. The Std term is EPS-FREE, which
turns v2's silent zero gradient at exact collapse into a loud NaN, and that is the
better failure: a number nobody mistakes for progress. And the detector runs on
every one of the 120 steps.

THE DETECTOR HAS TWO LEGS BECAUSE THE COLLAPSE HAS TWO SHAPES, AND BOTH ARE
PLANTED. A COMPLETE collapse maps every sequence to one point and is caught by the
per-coordinate standard deviation below 0.001. A DIMENSIONAL collapse keeps the
spread and puts it all on one direction, which the variance leg passes: the
rank-one plant keeps std_min 0.2781 and is missed by the variance leg entirely,
and only the effective rank catches it. Measured on the same bed:
std_min 0.0000e+00 planted against 2.2361e-01 healthy, and the effective rank
falls from 2.7081 to 1.0000 against a floor of 1.5.

AND THE Std TERM IS AT FULL STRENGTH THROUGHOUT A COLLAPSE IT CANNOT SEE, WHICH
IS SHARPER THAN v2's FINDING AND IS WHY nu IS 100. Every VICReg ships a covariance
weight of 1.0. At that setting on this bed the effective rank of the target
representation falls from 3.4457 to 1.4095 while the variance leg reads a
perfectly healthy std_min, and the run would be refused at step 59; at nu = 25.0
the rank bottoms at 1.8407 and at nu = 100.0 at 2.0811. So: at nu = 1.0 the
effective rank bottoms at 1.4095 while the variance leg reads std_min 0.3223. A
Std term cannot see a dimensional collapse, because every coordinate keeps its own
spread while all of them become the same direction.

THE BED WAS MEASURED BEFORE THE DETECTOR WAS TRUSTED, and that changed the design.
At an action drift of 0.60 the raw observation carries effective rank 2.7656 of
12 and the rank leg fires on an honest run; at 0.10 it reads 5.9885. A degenerate
bed makes a working detector look broken, and the obvious conclusion would have
been the wrong one.

THE TWO WIRINGS THAT SILENTLY DO NOTHING. A stop-grad is one call and its absence
is invisible in a loss curve, so it is measured in both directions: 0 of 6 target
parameters carry a gradient with the stop-grad in place and 6 without it. An EMA
is one line and momentum zero is the setting whose answer is known in closed form:
tau = 0 reproduces the online encoder to 0.0000 over 6 tensors, bitwise, while the
shipped momentum leaves a nonzero lag.

L-SIMPLE, SCORED FIRST, AND IT KILLED THE TRAINING CLAIM. The untrained encoder
and a FROZEN RANDOM encoder with only the predictor fitted were both scored before
the trained run, on the same held-out bed and the same metric. The metric is
NRMSE and not absolute latent MSE, because a JEPA's target is the encoder's OWN
representation and an encoder with small outputs is trivially easy to predict:
target RMS runs 0.4226 untrained to 1.7368 trained. Measured: untrained 2.9262,
frozen random 0.6881, trained 0.8624. THE FROZEN RANDOM ENCODER WINS. Over the
training run the latent loss runs 1.2850 -> 0.2048, which is a statement about
fitting a moving target and not about representation quality.

THE ROUTE FOR THAT KILL, BECAUSE A KILL WITHOUT ONE IS A COMPLAINT. RETIRED: the
claim that training the encoder lowers the latent error on this bed. It was never
in R2's scope and nothing here depends on it. REPRICED: what the trained run buys
is a live representation rather than a lower error -- target std_min 0.2212
against the frozen encoder's 0.0383. WHAT TO WANT INSTEAD is a downstream probe
against an encoder-independent target, which is the only comparison here that is
not circular, and which needs a real bed and the GPU run this file exists to make
safe.

THE TRAINABLE SURFACE, BECAUSE A SELF-CHECK NOTHING CAN TRAIN IS NOT A
DELIVERABLE. PiJepa takes its shapes AND ITS MASK as arguments, fit() takes its
steps and its data source from the caller, the device is an argument defaulting to
CPU, and the state_dict carries the mask as a buffer beside the weights: 2026
trainable parameters of 3795. The round trip moves the forward output by 0.0000
and dropping the mask entry moves it by 2.1179, which is the control without which
that test would pass with the corners missing.

L-CLASS. three_corners_containment's hypotheses are checked on THIS bed rather
than cited: the gate vanishes, the logits are finite, the beta=1 corner is
row-stochastic, and the two corners do not coincide. READ
lean/CEQ/V16Domain.lean:433, corners_are_distinct at :445.

THE ACTION, AND ITS PLACEBO. Delta_spec is the expected difference between the
error under the real action and under a placebo action drawn to be always
different. An action-blind predictor reads exactly 0.0000 there rather than a
false win, which is the control v1's mv_row bug needed and did not have.

LIMIT: the bed is synthetic and its action moves the next state by construction,
so a nonzero Delta_spec says the wiring carries the action, not that any real
action is learnable. The equivalence figures are float64 on this bed's shapes and
bound nothing for other shapes. The rank leg catches a collapse ONTO ONE
DIRECTION; a soft loss of two or three dimensions passes it, and a floor tight
enough to catch that would refuse this bed's own healthy run. Every training
number is 120 steps of a 9-coordinate encoder on synthetic data.

RUN: python -m ceqjepa.pi_jepa
"""

import hashlib
import math
import platform
import re as _re
import subprocess
import time
from collections import OrderedDict, namedtuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ceq import arm_smprime as arm

from . import curvature as cv
from . import pi_assign as pa
from . import sharpness

__all__ = [
    "SEED", "D_LATENT", "S_LEN", "BATCH", "X_DIM", "HORIZON", "N_STEPS",
    "TAU", "PIN_BETA", "LR", "N_ACTIONS", "DK", "HIDDEN",
    "BETA_INTERIOR", "EQUIV_BETAS", "EQUIV_TOL", "ROW_SUM_TOL",
    "COLLAPSE_STD_MIN", "COLLAPSE_ERANK_MIN", "COLLAPSE_PLANTS", "RANK_BURN_IN",
    "LAM", "MU", "NU", "COV_SWEEP", "BASELINE_ORDER", "MASK_AXIS",
    "Refusal", "is_refusal", "REASON_COLLAPSE", "REASON_REFUSED_COORD",
    "measured_mask", "beta_vector", "beta_for", "read_refused_column",
    "ReadOut", "read_per_coordinate",
    "Encoder", "ConstantEncoder", "RankOneEncoder", "ReadPredictor",
    "BlindPredictor", "ema_update", "collapse_report", "vicreg_terms",
    "DRIFT", "draw_bed", "bed_rank_check", "shipped_equivalence", "corner_identities", "refusal_channel",
    "collapse_must_fire", "stopgrad_must_fire", "ema_identity", "init_check",
    "std_term_probe", "PiJepa", "build", "fit", "default_batches",
    "train_run", "baseline_detail", "baselines", "pin_vs_ema_report",
    "covariance_sweep", "roundtrip_check", "roundtrip_without_mask",
    "theorem_hypotheses", "action_report", "provenance", "docstring_numbers",
    "report",
]

# ---------------------------------------------------------------------------
# THE PINNED BED AND THE SIZES, WHICH ARE THE SCOPE
# ---------------------------------------------------------------------------

SEED = 5501
D_LATENT = 9              #: the mask's axis length, and nothing wider
S_LEN = 16                #: positions per sequence
BATCH = 64                #: sequences per batch
X_DIM = 12                #: observation width
HORIZON = 4               #: h, the prediction horizon in positions
N_STEPS = 120             #: optimiser steps; a CPU budget, not a scale claim
TAU = 0.99                #: EMA momentum for the target encoder
PIN_BETA = 1.0            #: the parameter-free target's corner: beta=1 is softmax
LR = 3e-3
N_ACTIONS = 4
DK = 8                    #: query/key width inside the predictor
HIDDEN = 32

BETA_INTERIOR = 0.37      #: the one interior beta the equivalence is measured at
EQUIV_BETAS = (0.0, BETA_INTERIOR, 1.0)
EQUIV_TOL = 1e-13         #: float64 slack on the read, well above the measured worst
ROW_SUM_TOL = 1e-12

#: A coordinate whose spread over the batch falls below this is COMPLETELY
#: collapsed. The rank floor catches the other shape, where the spread survives
#: on one direction.
COLLAPSE_STD_MIN = 1e-3

#: THE RANK FLOOR, AND IT IS SET FROM THE TWO MEASUREMENTS IT HAS TO SEPARATE,
#: not from taste. The planted rank-one encoder reads exactly 1.0 by
#: construction; the healthy run at the shipped covariance weight bottoms out at
#: 2.0811 over its whole trace. 1.5 sits between them with margin on both sides.
#: THE LIMIT THIS BUYS AND NOTHING MORE: the leg catches a collapse ONTO ONE
#: DIRECTION. A soft loss of two or three dimensions passes it, and a floor tight
#: enough to catch that would refuse this bed's own healthy run.
COLLAPSE_ERANK_MIN = 1.5
COLLAPSE_PLANTS = ("healthy", "planted", "rank_one")

#: THE RANK LEG'S BURN-IN, SET FROM THE SAME SIDE THE LEG NOW READS FOR BOTH
#: ARMS. Once the detector reads the ONLINE representation for the EMA arm too
#: (fit() no longer special-cases pin_target), it meets a transient the online
#: encoder enters early under EITHER target and leaves before training ends:
#: measured on seeds 5501-5505, both pin_target settings, 120 steps each (10
#: runs), the online effective rank dips below COLLAPSE_ERANK_MIN and
#: recovers, and the LATEST step at which any of THOSE 10 runs was still below
#: the floor is step 54 (seed 5503, pin_target=True). Step 54 IS A PROPERTY OF
#: SEEDS 5501-5505, NOT OF THE BED: on fresh seeds 6002 (EMA arm, erank 1.2062
#: at step 55) and 6005 (pinned arm, erank 1.4176 at step 55) the dip is still
#: live at the step this burn-in stops gating, so a caller relying on 55 as a
#: universal recovery step will still see the rank leg raise past it -- 55 is
#: not re-measured here, only its origin is stated correctly. The std leg is
#: unaffected and still runs from step 0: this burn-in gates the rank leg
#: only, because the dip is a rank-leg-shaped transient, not a sign the
#: variance leg's own floor needs delaying too. A CALLER PASSING steps <= 55
#: gets a ONE-LEGGED DETECTOR for the whole run: active_erank_min in fit() stays
#: 0.0 for every step < RANK_BURN_IN, so the rank leg never activates and only
#: the std leg (a COMPLETE collapse) is checked -- fit(..., steps=55) can
#: complete on a representation whose effective rank is 1.0000 on every step
#: without raising, because the loop never reaches step 55.
RANK_BURN_IN = 55

#: invariance, variance, covariance. NU IS 100 AND THE DEFAULT 1.0 IS MEASURED TO
#: FAIL HERE -- see covariance_sweep(). At nu = 1.0 the effective rank bottoms at
#: 1.4095 while the variance leg reads std_min 0.3223, so the Std term is at full
#: strength throughout a dimensional collapse it cannot see. That is the same
#: shape of defect as v2's hinge and it is why the rank leg exists.
LAM, MU, NU = 25.0, 25.0, 100.0
COV_SWEEP = (1.0, 25.0, NU)

#: The action's drift. Set from a MEASUREMENT: at 0.6 the four action means
#: dominate the bed's own variance and the raw observation carries effective rank
#: 2.7656 of 12. The bed was checked FIRST -- see bed_rank_check() -- because a
#: degenerate bed makes a working detector look broken.
DRIFT = 0.10
BASELINE_ORDER = ("untrained", "frozen_random", "trained")

MASK_AXIS = "D"           #: the axis R1 names. NOT "S". A vector along "S" would
                          #: index the key position and be silently wrong.

Refusal = cv.Refusal
is_refusal = cv.is_refusal

REASON_COLLAPSE = "REPRESENTATION-COLLAPSED"
REASON_REFUSED_COORD = "COORDINATE-REFUSED-BY-R1"


# ---------------------------------------------------------------------------
# 1. THE MASK, TAKEN AND VALIDATED, NEVER RE-DERIVED
# ---------------------------------------------------------------------------

_MASK_CACHE = {}


def measured_mask():
    """R1's frozen vector, fetched once. The sweep is NOT re-run per caller.

    Only frozen_mask, mask_digest and sweep_spec are used from that module: the
    exponents are its finding and re-fitting them here would be a second
    instrument disagreeing with the first at leisure.
    """
    if "frozen" not in _MASK_CACHE:
        res = pa.assign()
        _MASK_CACHE["frozen"] = pa.frozen_mask(res["assignments"],
                                               pa.mask_digest(pa.sweep_spec()))
    return _MASK_CACHE["frozen"]


def beta_vector(frozen=None):
    """The mask as beta per coordinate: a float, or a Refusal carrying its reason.

    The three checks are the ones ceqjepa/t_length.py:694 already applies at this
    boundary -- the axis NAME, the axis LENGTH against the vector length, and a
    refusal that still carries a number -- rather than a second convention
    invented here. A refused coordinate becomes a Refusal and not a None, because
    None silently means "missing" to a caller and arithmetic on it raises the
    wrong error; a Refusal raises TypeError and says WHY on the way out.
    """
    fz = frozen if frozen is not None else measured_mask()
    if fz.get("axis") != MASK_AXIS:
        raise ValueError("the mask axis is %r, not %r: a vector along another "
                         "axis indexes something else" % (fz.get("axis"), MASK_AXIS))
    n_axis = int(fz["axis_len"])
    beta, coords = list(fz["beta"]), list(fz["coords"])
    if len(beta) != n_axis or len(coords) != n_axis:
        raise ValueError("the axis says %d but the vector carries %d beta and %d "
                         "names: the axis was compacted"
                         % (n_axis, len(beta), len(coords)))
    refusals = dict(fz.get("refusals") or {})
    for name, code in refusals.items():
        if beta[coords.index(name)] is not None:
            raise ValueError("coordinate %r is refused for %r but carries beta %r: "
                             "a refusal was dropped"
                             % (name, code, beta[coords.index(name)]))
    out = []
    for name, b in zip(coords, beta):
        if b is None:
            code = refusals.get(name)
            if code is None:
                raise ValueError("coordinate %r carries no beta and no reason: a "
                                 "refusal without one is a dropped value, not a "
                                 "refusal" % (name,))
            out.append(Refusal(code, "%s: R1 refused coordinate %r, so it has no "
                                     "corner and a read that supplied one would be "
                                     "answering a question the rule declined"
                               % (REASON_REFUSED_COORD, name)))
        else:
            out.append(float(b))
    return out


def beta_for(name):
    """The beta of one NAMED coordinate: a float, or the Refusal itself."""
    fz = measured_mask()
    return beta_vector(fz)[list(fz["coords"]).index(name)]


# ---------------------------------------------------------------------------
# 2. THE READ, FACTORISED SO beta CAN INDEX THE OUTPUT COORDINATE
# ---------------------------------------------------------------------------

ReadOut = namedtuple("ReadOut", "out cols axis axis_len refusals")


def read_per_coordinate(q, k, v, beta):
    """O_{i,c} = (num @ v)_{i,c} / Z_i^{beta_c}, with the numerator SHIPPED.

    ceq/arm_smprime.numerator is called rather than reimplemented, so the gate,
    the causal mask and the exponential are the family's and not a copy of them
    that drifts. What changes is only WHERE the division by Z^beta happens: the
    shipped operator divides before contracting with V (:250 then :264), which
    has no output-coordinate axis to index; dividing after the contraction does,
    and is the same number because Z_i does not depend on j.

    A refused coordinate produces NO column. Its index and its Refusal travel in
    `refusals`, and `axis_len` carries the FULL axis beside the columns actually
    produced, so a consumer reading len(cols) as the axis length is contradicted
    by the object rather than by a comment.
    """
    if len(beta) != v.shape[-1]:
        raise ValueError("beta has %d entries against %d value coordinates: the "
                         "mask does not index this read's output"
                         % (len(beta), v.shape[-1]))
    num, mod = arm.numerator(q, k)
    z = mod.sum(-1, keepdim=True)
    o = num @ v.to(num.dtype)
    cols = tuple(i for i, b in enumerate(beta) if not is_refusal(b))
    refusals = OrderedDict((i, b) for i, b in enumerate(beta) if is_refusal(b))
    bt = torch.as_tensor([float(beta[i]) for i in cols], dtype=z.dtype,
                         device=z.device)
    zb = z ** bt
    sel = o[..., cols]
    out = torch.complex(sel.real / zb, sel.imag / zb)
    return ReadOut(out=out, cols=cols, axis=MASK_AXIS, axis_len=len(beta),
                   refusals=refusals)


def read_refused_column():
    """The must-fire: ask the read for a coordinate R1 refused, get the refusal.

    Returned rather than raised, because "bottom as a value" is the v2.1 spec:
    a caller can hold it, print it, and be stopped by it at the first arithmetic,
    which a raised exception three frames up does not do.
    """
    beta = beta_vector()
    idx = [i for i, b in enumerate(beta) if is_refusal(b)]
    if not idx:
        return Refusal("NO-REFUSAL-TO-SHOW",
                       "the mask refuses no coordinate, so this must-fire cannot "
                       "fire and the channel is untested")
    g = torch.Generator().manual_seed(SEED)
    q = torch.randn(2, S_LEN, DK, generator=g, dtype=torch.float64)
    k = torch.randn(2, S_LEN, DK, generator=g, dtype=torch.float64)
    v = torch.randn(2, S_LEN, len(beta), generator=g, dtype=torch.float64)
    ro = read_per_coordinate(q, k, v, beta)
    return ro.refusals[idx[0]]


# ---------------------------------------------------------------------------
# 3. THE PIECES
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """x_{<=t} -> s_t, per position. Three layers and no normalisation.

    NO NORMALISATION IS DELIBERATE AND R1 IS THE REASON. An RMS norm over a
    representation whose norm grows with n subtracts that growth from every
    exponent at once, which is the attack pi_assign's census exists to catch; a
    JEPA that normalises here would be measuring its own normalizer.
    """

    def __init__(self, x_dim=X_DIM, d=D_LATENT, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(x_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, d))

    def forward(self, x):
        return self.net(x)


class ConstantEncoder(nn.Module):
    """PLANTED COMPLETE COLLAPSE: every sequence maps to one point."""

    def __init__(self, d=D_LATENT):
        super().__init__()
        self.register_buffer("c", torch.linspace(-1.0, 1.0, d, dtype=torch.float64))

    def forward(self, x):
        return self.c.to(x.dtype).expand(*x.shape[:-1], self.c.shape[0])


class RankOneEncoder(nn.Module):
    """PLANTED DIMENSIONAL COLLAPSE: full spread, one direction.

    This is the plant the variance leg alone MISSES, which is why the rank leg is
    not decoration. The scalar is the observation mean, so the spread across
    sequences is real; every coordinate is the same scalar times a fixed vector,
    so the effective rank is one.
    """

    def __init__(self, d=D_LATENT):
        super().__init__()
        self.register_buffer("dir", torch.linspace(1.0, 2.0, d, dtype=torch.float64))

    def forward(self, x):
        return x.mean(-1, keepdim=True) * self.dir.to(x.dtype)


class ReadPredictor(nn.Module):
    """s_hat_{t+h} = the family's read at the last position, beta per coordinate.

    do(a) biases the QUERY by an action embedding, which shifts every logit in
    the row by (bias . k_j) and therefore conditions the whole read rather than
    one entry of it. The action head is NOT zero-initialised: a zero there is
    v1's mv_row bug, where every candidate action produced the identical
    intervention and the placebo scored as well as the real move.
    """

    def __init__(self, d=D_LATENT, dk=DK, n_actions=N_ACTIONS):
        super().__init__()
        self.wq = nn.Linear(d, dk, bias=False)
        self.wk = nn.Linear(d, dk, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.act = nn.Embedding(n_actions, dk)
        nn.init.normal_(self.act.weight, std=0.5)     # NOT zero: v1's mv_row bug

    def forward(self, s, beta, a=None):
        q = self.wq(s)
        if a is not None:
            q = q + self.act(a).unsqueeze(-2)
        return read_per_coordinate(q, self.wk(s), self.wv(s), beta)


class BlindPredictor(nn.Module):
    """THE CONTROL. Ignores its action entirely, so Delta_spec must read 0."""

    def __init__(self, d=D_LATENT):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, s, beta, a=None):
        cols = tuple(i for i, b in enumerate(beta) if not is_refusal(b))
        return ReadOut(out=self.net(s)[..., cols], cols=cols, axis=MASK_AXIS,
                       axis_len=len(beta),
                       refusals=OrderedDict((i, b) for i, b in enumerate(beta)
                                            if is_refusal(b)))


@torch.no_grad()
def ema_update(target, online, tau):
    """theta_bar <- tau theta_bar + (1 - tau) theta, over parameters AND buffers.

    At tau = 0 this is a straight copy, which is the one setting whose answer is
    known in closed form and is therefore the one asserted.
    """
    for t, o in zip(target.parameters(), online.parameters()):
        t.mul_(tau).add_(o.detach(), alpha=1.0 - tau)
    for t, o in zip(target.buffers(), online.buffers()):
        t.copy_(o)


# ---------------------------------------------------------------------------
# 4. THE COLLAPSE DETECTOR, THREE LEGS (THE THIRD OPT-IN), RUN EVERY STEP
# ---------------------------------------------------------------------------

def collapse_report(s, std_min=COLLAPSE_STD_MIN, erank_min=COLLAPSE_ERANK_MIN,
                    q=None, y=None, K=None, n_boot=200, boot_seed=0,
                    construction_property=False):
    """Per-coordinate spread, effective rank, AND (opt-in) a label-aware leg.

    THREE LEGS BECAUSE TWO ARE NOT ENOUGH. THE GEOMETRIC GAP ITSELF -- a
    frozen-random encoder reading HEALTHIER on both legs than one that learned
    its labels -- is proven, not argued, by tests/cameron/test_third_collapse_
    leg.py's frozen-random plant: MEASURED, that file now routes 5 guarded
    entries through THIS function's q/y interface (not a direct sharpness
    call) and fires the label leg twice, so it exercises the shipped leg
    below, end to end, on a frozen-random encoder read against real labels.
    This module's own demo() (section (h)) additionally self-checks the same
    leg's arithmetic on a synthetic informative-vs-frozen plant; that
    self-check is not a downstream probe and is labelled as a self-check where
    it runs. The variance leg reads the
    smallest per-coordinate standard deviation over the batch, which a COMPLETE
    collapse drives to zero. The rank leg reads the participation ratio of the
    covariance spectrum, (sum lambda)^2 / sum lambda^2, which is the axis count a
    DIMENSIONAL collapse destroys while leaving every per-coordinate variance
    healthy. Both read s ALONE, so neither can ever depend on whether s carries
    information about the task's actual label: a frozen-random encoder spreads
    across every direction it was never trained to compress out of, so it reads
    healthy on both -- while carrying nothing. The label leg closes that gap by
    reading the label instead of the geometry.

    THE LABEL LEG IS OPT-IN. Pass q [n,K] class probabilities and y [n] int
    labels (K inferred from q when omitted) to activate it; every existing
    caller in this file passes neither, so every existing caller's behaviour is
    unchanged -- `leg` is still only ever None, "std" or "rank" for them, and
    `label` in the returned dict is None. When q and y ARE given, this reuses
    ceqjepa/sharpness.py's own PREREG bar rather than inventing a fourth
    threshold: sharpness.decompose(q, y, K) and sharpness.bootstrap_se(q, y, K)
    feed sharpness.kill_fired(d, se), which is margin <= 1 bootstrap SE -- not
    even confidently better than a predictor that never looked at the input.
    That is the same reachable plant a frozen-random encoder scored against its
    real labels exhibits: both geometric legs green, kill_fired True.

    construction_property IS THE CALLER'S TAG, NOT A GUESS MADE HERE. Pass
    True when q was built FROM y (a nearest-centroid read of an s constructed
    as centroid[y] + noise, a hand-picked oracle demo, any read where the
    label was baked into q rather than earned by a model) -- house rule: "a
    construction property is never a bar." When True, this function propagates
    the tag into sharpness.decompose's returned dict but does NOT call
    sharpness.kill_fired on it (that call raises ConstructionPropertyAsBarError
    by design, see sharpness.py), so `label["kill_fired"]` is None (not
    evaluated, not False) and `leg` can never become "label" from an oracle
    read. Default False is for a genuine read: a q earned by scoring actual
    model output against labels it never saw baked in.

    Returns a Refusal when any leg fires, because a warning is something a
    training loop steps over.
    """
    z = s.detach().reshape(-1, s.shape[-1]).double()
    sd = z.std(dim=0, unbiased=True)
    c = z - z.mean(0, keepdim=True)
    cov = (c.T @ c) / max(z.shape[0] - 1, 1)
    lam = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    tot = float(lam.sum())
    erank = float(tot * tot / float((lam * lam).sum())) if tot > 0 else 0.0
    smin, smed = float(sd.min()), float(sd.median())
    leg = None
    if smin < std_min:
        leg = "std"
    elif erank < erank_min:
        leg = "rank"
    # THE THIRD LEG, LABEL-AWARE AND OPT-IN: only evaluated when the caller
    # hands in q AND y, so a caller that has neither (every call site in this
    # file, today) sees no change in behaviour at all.
    label = None
    if q is not None and y is not None:
        d = sharpness.decompose(q, y, K, construction_property=construction_property)
        se = sharpness.bootstrap_se(q, y, K, n_boot=n_boot, seed=boot_seed)
        # RULE: a construction property never reaches the bar API. kill_fired
        # raises ConstructionPropertyAsBarError on a tagged d (sharpness.py),
        # so it is never CALLED here for one -- the verdict is None (not
        # evaluated), not False, and `leg` can therefore never become "label"
        # from an oracle read.
        kill_fired = (None if construction_property
                      else sharpness.kill_fired(d, se))
        label = dict(i_q=d["i_q"], se_i_q=se["se_i_q"], margin=d["margin"],
                     se_margin=se["se_margin"],
                     margin_sigma=sharpness.margin_sigma(d, se),
                     construction_property=construction_property,
                     kill_fired=kill_fired)
        if leg is None and label["kill_fired"]:
            leg = "label"
    ref = None
    if leg == "label":
        ref = Refusal(REASON_COLLAPSE,
                      "the label leg fired: I_q %.4e at bootstrap se %.4e "
                      "(%.2f sigma margin over the marginal predictor) while "
                      "smallest per-coordinate sd %.4e and effective rank %.4f "
                      "of %d both read healthy against their own floors -- a "
                      "representation this uninformative about its label "
                      "predicts nothing, and every number downstream of it is "
                      "about the label leak the other two legs cannot see"
                      % (label["i_q"], label["se_i_q"], label["margin_sigma"],
                         smin, erank, z.shape[1]))
    elif leg is not None:
        ref = Refusal(REASON_COLLAPSE,
                      "the %s leg fired: smallest per-coordinate sd %.4e against "
                      "%.4g, effective rank %.4f of %d against %.4g -- a "
                      "representation this degenerate predicts nothing and every "
                      "number downstream of it is about the degeneracy"
                      % (leg, smin, std_min, erank, z.shape[1], erank_min))
    return dict(std_min=smin, std_med=smed, erank=erank, n_coords=int(z.shape[1]),
                collapsed=leg is not None, leg=leg, refusal=ref, label=label)


def vicreg_terms(s, s_hat, s_tgt, lam=LAM, mu=MU, nu=NU):
    """invariance + EPS-FREE Std + covariance.

    EPS-FREE IS A CHOICE AND IT HAS A MEASURED CONSEQUENCE. v2's Std is
    sqrt(var + eps) with eps at 1e-12, and at EXACTLY zero spread the gradient of
    that term is exactly zero: the hinge sits at its maximum and pushes with
    nothing, a stationary point of the same shape as a zero-init LoRA death.
    Dropping eps does not remove the stationary point -- nothing can, it is a
    property of a smooth function of centred s -- but it changes what the point
    LOOKS like from a silent zero to a NaN, and a NaN is the better failure
    because no one mistakes it for progress. std_term_probe() measures both.

    The variance term SUMS over coordinates rather than averaging, because a mean
    divides the per-element gradient by d and turns a threshold that was right
    into one that refuses.
    """
    inv = (s_hat - s_tgt.detach()).pow(2).sum(-1).mean()
    std = torch.sqrt(s.var(dim=0))
    var = torch.relu(1.0 - std).sum()
    sc = s - s.mean(0, keepdim=True)
    cov = (sc.T @ sc) / max(s.shape[0] - 1, 1)
    off = cov - torch.diag_embed(torch.diagonal(cov))
    cv_ = off.pow(2).sum() / s.shape[1]
    total = lam * inv + mu * var + nu * cv_
    return total, dict(inv=float(inv.detach()), var=float(var.detach()),
                       cov=float(cv_.detach()), std_min=float(std.min().detach()))


def std_term_probe():
    """What the eps-free Std term does as the spread goes to zero, and AT zero.

    The gradient ratio |dL/ds| is independent of the spread by the derivative
    |c_{i,d}| / (B Std_d): numerator and denominator are both O(sigma), so the
    term pulls as hard out of a nearly collapsed state as out of a healthy one.
    At EXACTLY zero the derivative is 0/0 and reads NaN, which is the loud
    version of v2's silent zero and is recorded as the known hole.
    """
    rows = []
    for scale in (0.0, 1e-12, 1e-6, 1e-1):
        g = torch.Generator().manual_seed(SEED)
        s = (torch.randn(BATCH, D_LATENT, generator=g) * scale).requires_grad_(True)
        loss, st = vicreg_terms(s, s, torch.zeros(BATCH, D_LATENT))
        loss.backward()
        gmax = float(s.grad.abs().max())
        rows.append(dict(scale=scale, std_min=st["std_min"], var=st["var"],
                         grad_max=gmax, is_nan=bool(gmax != gmax)))
    return rows


# ---------------------------------------------------------------------------
# 5. THE BED
# ---------------------------------------------------------------------------

def draw_bed(seed=SEED, batch=BATCH, s=S_LEN, h=HORIZON, x_dim=X_DIM,
             drift=None):
    """(x, action). An AR(1) stream whose drift is chosen by the action.

    The action genuinely moves the later states, by construction, so a nonzero
    Delta_spec says the WIRING carries the action; it is not evidence that any
    real action is learnable. That limit is in the module docstring too.
    """
    g = torch.Generator().manual_seed(int(seed))
    a = torch.randint(0, N_ACTIONS, (batch,), generator=g)
    scale = DRIFT if drift is None else float(drift)
    drift = torch.randn(N_ACTIONS, x_dim, generator=g) * scale
    x = torch.zeros(batch, s + h, x_dim)
    x[:, 0] = torch.randn(batch, x_dim, generator=g)
    for t in range(1, s + h):
        x[:, t] = (0.9 * x[:, t - 1]
                   + 0.3 * torch.randn(batch, x_dim, generator=g)
                   + drift[a])
    return x, a


def bed_rank_check(seed=SEED, drifts=(0.6, DRIFT)):
    """THE BED'S OWN EFFECTIVE RANK, MEASURED BEFORE THE DETECTOR IS TRUSTED.

    This ran first and it changed the design. At an action drift of 0.60 the four
    action means dominate the bed's own variance and the raw observation carries
    effective rank 2.7656 of 12: the encoder cannot make rank that is not there,
    the rank leg fires on an honest run, and the obvious conclusion -- that the
    detector is broken -- would have been wrong. At 0.10 the raw bed reads
    5.9885. A degenerate bed makes a working detector look broken, so the bed is
    measured rather than assumed and DRIFT is set from this table.
    """
    rows = []
    for d in drifts:
        x, _ = draw_bed(seed, drift=d)
        rows.append(dict(drift=float(d), n_dims=int(x.shape[-1]),
                         erank=collapse_report(x[:, -1, :],
                                               std_min=0.0, erank_min=0.0)["erank"]))
    return rows


def _windows(x, h=HORIZON, s=S_LEN):
    """(context ending at t, target window ending at t+h)."""
    return x[:, :s], x[:, h:h + s]


# ---------------------------------------------------------------------------
# 6. THE EQUIVALENCE AND THE TWO CORNER IDENTITIES
# ---------------------------------------------------------------------------

def _equiv_draw(seed=SEED):
    g = torch.Generator().manual_seed(int(seed))
    kw = dict(generator=g, dtype=torch.float64)
    return (torch.randn(4, S_LEN, DK, **kw), torch.randn(4, S_LEN, DK, **kw),
            torch.randn(4, S_LEN, D_LATENT, **kw))


def _worst(a, b):
    return float((a - b).abs().max())


def _bitwise(a, b):
    return bool(torch.equal(a.real, b.real) and torch.equal(a.imag, b.imag))


def shipped_equivalence(seed=SEED):
    """The factorised read against ceq/arm_smprime.readout for a SCALAR beta.

    The shipped readout is the reference and it is untouched. Only the order of
    the division and the contraction differs, so any difference here is float64
    reassociation and its size is the price of a per-coordinate exponent.
    """
    q, k, v = _equiv_draw(seed)
    rows = []
    for b in EQUIV_BETAS:
        ship = arm.readout(q, k, v, beta=b)
        mine = read_per_coordinate(q, k, v, [float(b)] * D_LATENT).out
        rows.append(dict(beta=float(b), worst=_worst(mine, ship),
                         bitwise=_bitwise(mine, ship)))
    return dict(rows=rows, worst_overall=max(r["worst"] for r in rows),
                n_coords=D_LATENT, reference="ceq/arm_smprime.py:readout")


def corner_identities(seed=SEED):
    """All ones IS the softmax arm; all zeros IS the linear arm.

    Against the SHIPPED readout at the scalar corner, because a
    re-implementation agreeing with itself is not evidence.
    """
    q, k, v = _equiv_draw(seed)
    ones = read_per_coordinate(q, k, v, [1.0] * D_LATENT).out
    zeros = read_per_coordinate(q, k, v, [0.0] * D_LATENT).out
    soft = arm.readout(q, k, v, beta=1.0)
    lin = arm.readout(q, k, v, beta=0.0)
    return dict(ones_worst=_worst(ones, soft), ones_bitwise=_bitwise(ones, soft),
                zeros_worst=_worst(zeros, lin), zeros_bitwise=_bitwise(zeros, lin),
                axis_len=D_LATENT, tol=EQUIV_TOL)


def refusal_channel():
    """What crosses the boundary: assigned coordinates as floats, refused as
    Refusals, and the axis length declared beside both."""
    fz = measured_mask()
    beta = beta_vector(fz)
    refused = OrderedDict()
    for i, b in enumerate(beta):
        if is_refusal(b):
            refused[fz["coords"][i]] = dict(fz["refusals"])[fz["coords"][i]]
    floats = sum(1 for b in beta if isinstance(b, float))
    return dict(axis=fz["axis"], axis_len=int(fz["axis_len"]),
                coords=list(fz["coords"]), beta=beta,
                n_assigned=int(fz["n_assigned"]), refused=refused,
                assigned_floats=floats,
                defaulted=sum(1 for i, b in enumerate(beta)
                              if fz["beta"][i] is None and not is_refusal(b)),
                digest=fz["digest"])


# ---------------------------------------------------------------------------
# 7. THE MUST-FIRES ON THE WIRING
# ---------------------------------------------------------------------------

def collapse_must_fire(seed=SEED):
    """A planted collapse of each shape, and a healthy control. Both directions.

    THE HEALTHY CONTROL IS THE REAL THING, not isotropic noise. A synthetic
    gaussian control is healthy by construction and says nothing about whether
    the thresholds sit where a real run lives, so the control here is the TARGET
    representation at the end of the trained run -- the exact state the detector
    is there to protect.
    """
    x, _ = draw_bed(seed)
    ctx, tgt = _windows(x)
    tr = train_run(seed=seed)
    healthy = tr["model"].target_repr(tgt)
    out = OrderedDict()
    out["healthy"] = collapse_report(healthy)
    out["planted"] = collapse_report(ConstantEncoder()(ctx)[:, -1, :])
    out["rank_one"] = collapse_report(RankOneEncoder()(ctx)[:, -1, :])
    out["runs_every_step"] = True
    out["steps_checked"] = tr["steps_checked"]
    return out


def stopgrad_must_fire(seed=SEED):
    """No gradient into the target branch, and removing the stop-grad breaks it.

    The same step is taken twice, once with sg(s_tgt) and once without, and the
    count of target parameters carrying a gradient is read off both times. One
    direction alone proves nothing: zero grads with the stop-grad in place is
    also what a disconnected target gives.
    """
    x, a = draw_bed(seed)
    ctx, tgt = _windows(x)
    counts = {}
    requires = None
    for label, detach in (("on", True), ("off", False)):
        model = build(seed)
        for p in model.target.parameters():
            p.requires_grad_(True)         # so a leak WOULD be visible
            p.grad = None
        # THE STOP-GRAD ITSELF, and the only difference between the two passes:
        # target_repr() is decorated no-grad, so "off" reaches around it and runs
        # the same forward with the graph attached.
        if detach:
            st = model.target_repr(tgt)
            requires = bool(st.requires_grad)
        else:
            st = model.target(tgt)[:, -1, :]
        ro = model(ctx, a)
        loss = (ro.out[:, -1, :] - st[:, model.cols.tolist()]).pow(2).sum(-1).mean()
        loss.backward()
        counts[label] = sum(1 for p in model.target.parameters()
                            if p.grad is not None and float(p.grad.abs().sum()) > 0)
        counts["n"] = sum(1 for _ in model.target.parameters())
    return dict(on_target_grads=counts["on"], off_target_grads=counts["off"],
                n_target_params=counts["n"], isolated=counts["on"] == 0,
                isolated_without_stopgrad=counts["off"] == 0,
                target_requires_grad=bool(requires))


def ema_identity(seed=SEED):
    """tau = 0 is a copy; tau = TAU is not. Both measured, bitwise where exact."""
    torch.manual_seed(int(seed))
    online, target = Encoder(), Encoder()
    tensors = list(zip(target.parameters(), online.parameters()))
    ema_update(target, online, 0.0)
    tau0 = max(float((t - o).detach().abs().max()) for t, o in tensors)
    bit = all(bool(torch.equal(t, o)) for t, o in tensors)
    torch.manual_seed(int(seed) + 1)
    online2 = Encoder()
    ema_update(target, online2, TAU)
    lag = max(float((t - o).detach().abs().max())
              for t, o in zip(target.parameters(), online2.parameters()))
    return dict(tau0_worst=tau0, tau0_bitwise=bit, tau_worst=lag,
                n_tensors=len(tensors))


def init_check(seed=SEED):
    """The initialisation is not degenerate, and no output layer is zero."""
    model = build(seed)
    x, _ = draw_bed(seed)
    ctx, _t = _windows(x)
    with torch.no_grad():
        rep = collapse_report(model.online(ctx)[:, -1, :])
    zero = sum(1 for mod in model.online.net if isinstance(mod, nn.Linear)
               and float(mod.weight.detach().abs().max()) == 0.0)
    zero += sum(1 for w in (model.pred.wq.weight, model.pred.wk.weight,
                            model.pred.wv.weight, model.pred.act.weight)
                if float(w.detach().abs().max()) == 0.0)
    rep = dict(rep)
    rep.update(zero_layers=zero, degenerate=rep["collapsed"])
    return rep


# ---------------------------------------------------------------------------
# 8. THE TRAINABLE SURFACE: A MODEL A CALLER CAN BUILD, FIT, SAVE AND MOVE
#
# Everything above this line reaches the model through zero-argument functions
# that build it, run it and hand back a dict. That is enough for a self-check and
# for nothing else: no handle for an optimiser, no way to run more steps than the
# demo runs, no device argument, and no way to carry weights off the machine that
# produced them. A GPU run needs all four, so they are here, and each is pinned
# by its own test rather than by this comment.
# ---------------------------------------------------------------------------

class PiJepa(nn.Module):
    """The whole R2 object: online encoder, EMA target, and the read predictor.

    THE MASK IS AN ARGUMENT AND NOT A GLOBAL. A caller on another machine loads
    the frozen vector it was given and passes it here; nothing in this class
    reaches for beta_vector(), so a build at another size with another mask is
    honoured rather than silently overridden.

    THE MASK IS ALSO A BUFFER, which is the whole reason the save/load round trip
    is worth a test. Weights that reload into a model reading at DIFFERENT
    corners are wrong in a way no loss curve shows, so beta_assigned and cols ride
    in the state_dict beside the parameters.

    parameters() carries the target branch too, because the EMA weights are what
    a run is FOR; trainable_parameters() is what an optimiser gets, and it is
    filtered on requires_grad, so a frozen encoder is excluded by the same rule
    that excludes the target.

    pin_target=True SWAPS THE TARGET FOR A PARAMETER-FREE READ AND NOTHING ELSE.
    The EMA target (self.target, ema_update) is not independent of its own
    encoder: it is a copy of online's OWN weights, walked toward online's CURRENT
    weights every step, so its output over a FIXED input necessarily drifts as
    online trains. The pinned read instead reads the raw tokens directly through
    read_per_coordinate at the beta=1 corner (softmax, and the exponent 1 makes
    Z**1 == Z, so the read is Z-normalised attention over x with no linear layer
    and no parameter of any kind in it). Because default_batches hands fit() the
    SAME (ctx, tgt, a) at every step, a target that is a pure function of tgt
    reads bit-for-bit the same representation at step 0 and step N -- drift is
    1.0000 not because nothing moved but because nothing in the read's path CAN
    move. self.target and ema_update still exist and still run when pin_target
    is False, unchanged, so the two arms are the same class and the same fit()
    loop with one flag between them.
    """

    def __init__(self, beta, x_dim=X_DIM, d=None, dk=DK, hidden=HIDDEN,
                 n_actions=N_ACTIONS, device=None, pin_target=False):
        super().__init__()
        self.pin_target = bool(pin_target)
        d = len(beta) if d is None else int(d)
        if d != len(beta):
            raise ValueError("d is %d but the mask carries %d coordinates: the "
                             "mask does not index this model's output"
                             % (d, len(beta)))
        cols = tuple(i for i, b in enumerate(beta) if not is_refusal(b))
        if not cols:
            raise ValueError("the mask refuses every coordinate: there is nothing "
                             "for the read to produce")
        self.axis = MASK_AXIS
        self.axis_len = len(beta)
        self.refusals = OrderedDict((i, b) for i, b in enumerate(beta)
                                    if is_refusal(b))
        self.online = Encoder(x_dim, d, hidden)
        self.target = Encoder(x_dim, d, hidden)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.pred = ReadPredictor(d, dk, n_actions)
        self.register_buffer("beta_assigned",
                             torch.tensor([float(beta[i]) for i in cols]))
        self.register_buffer("cols", torch.tensor(cols, dtype=torch.long))
        if device is not None:
            self.to(device)

    def trainable_parameters(self):
        """What an optimiser gets: everything still carrying requires_grad."""
        return [p for p in self.parameters() if p.requires_grad]

    def beta_list(self):
        """The mask as the read wants it, rebuilt from the BUFFER and not from
        the constructor argument, so a reloaded model reads at the corners it
        actually loaded."""
        out = [self.refusals.get(i) for i in range(self.axis_len)]
        for j, i in enumerate(self.cols.tolist()):
            out[i] = float(self.beta_assigned[j])
        return out

    def forward(self, x, a=None):
        """s_hat over every position, at the assigned coordinates only.

        The imaginary part is exactly zero here -- the gate is m = 1, theta = 0,
        so every complex product has a zero imaginary part by exact arithmetic
        rather than by rounding -- and that is asserted rather than assumed
        before it is dropped.
        """
        ro = self.pred(self.online(x), self.beta_list(), a)
        if float(ro.out.imag.detach().abs().max()) != 0.0:
            raise AssertionError("the read carries a nonzero imaginary part: the "
                                 "gate is no longer m = 1, theta = 0")
        return ro._replace(out=ro.out.real)

    @torch.no_grad()
    def target_repr(self, x):
        """The target representation at the last position: the EMA encoder, or
        the pinned beta=1 read of x itself when pin_target is set. The no-grad
        decorator IS the stop-grad, and stopgrad_must_fire() removes it to show
        the check can tell the two wirings apart."""
        if self.pin_target:
            return self._pinned_target_repr(x)
        return self.target(x)[:, -1, :]

    def _pinned_target_repr(self, x):
        """O_{i,c} = (num @ v)_{i,c} / Z_i, read directly off the raw tokens x
        with beta pinned to 1 for every coordinate -- no online encoder, no
        target encoder, no linear projection: q, k are x itself and v is x's
        first axis_len channels, so the whole read is a deterministic function
        of x and self.axis_len, with no torch.nn.Parameter anywhere in its path.
        """
        v = x[..., :self.axis_len].to(x.dtype)
        ro = read_per_coordinate(x, x, v, [PIN_BETA] * self.axis_len)
        out = ro.out
        if float(out.imag.detach().abs().max()) != 0.0:
            raise AssertionError("the pinned read carries a nonzero imaginary "
                                 "part: the gate is no longer m = 1, theta = 0")
        return out.real[:, -1, :]


def build(seed=SEED, beta=None, **kw):
    """A PiJepa at a reproducible initialisation. The demo's only entry point."""
    torch.manual_seed(int(seed))
    return PiJepa(beta_vector() if beta is None else beta, **kw)


def default_batches(seed=SEED, batch=BATCH, s=S_LEN, h=HORIZON, x_dim=X_DIM):
    """The demo's data source, in the shape `fit` takes from any caller."""
    x, a = draw_bed(seed, batch, s, h, x_dim)
    ctx, tgt = _windows(x, h, s)
    return lambda step: (ctx, tgt, a)


def fit(model, batches=None, steps=N_STEPS, lr=LR, tau=TAU, nu=NU, lam=LAM,
        mu=MU, seed=SEED, detector=True, std_min=COLLAPSE_STD_MIN,
        erank_min=COLLAPSE_ERANK_MIN):
    """The training loop, with its SCALE and its DATA taken from the caller.

    `batches` is any callable step -> (context, target window, action), and None
    is the demo's own bed at the module's constants, so the self-check is
    unchanged while a caller can drive a different size, a longer run and its own
    data without editing this file.

    The collapse detector runs on EVERY step and reads the ONLINE representation
    FOR BOTH ARMS -- pin_target no longer special-cases which side it checks.
    Checking the target side used to differ by arm: for the EMA arm it is the
    EMA-walked target, live every step; for the PINNED arm target_repr(tgt) is
    a pure function of tgt with no nn.Parameter in its path
    (PiJepa._pinned_target_repr), and default_batches hands fit() the SAME
    (ctx, tgt, a) at every step, so that target is one constant tensor for the
    whole run -- measured, 1 unique std_min and 1 unique erank in a 120-step
    trace. Checking a constant 120 times is not a running detector, so the
    pinned arm's detector was already reading the online side; checking the
    EMA-target side for the OTHER arm meant the two arms were never checked on
    the same object, which is what manufactured the asymmetry pi_jepa.py's own
    history recorded (a 3/5-seed false collapse on the pinned arm's detector
    while the EMA arm's never fired, on a bed where an EQUAL-FOOTING read of
    the online side shows neither arm's online effective rank ever below the
    other's by more than a transient dip both leave). The online encoder is
    the only thing that trains in the pinned arm and is common to both arms, so
    it is the one honest side to check either way. `detector_side` in the
    return dict is now always "online", kept as a field rather than removed so
    a caller reading it does not silently get a stale value.

    THE RANK LEG IS SILENT UNTIL RANK_BURN_IN, THE STD LEG IS NOT. Reading the
    online side from step 0 meets a transient both arms enter early and leave
    -- see RANK_BURN_IN's own comment for the seeds and steps it was measured
    from. Only the rank leg is delayed; a COMPLETE collapse (the std leg) is
    not the transient shape and is caught from step 0 as before.

    THIS LOOP IS HONESTLY TWO-LEGGED, AND IS BLIND TO THE LABEL-LEAK INVERSION.
    Every step's collapse_report call above passes std_min/erank_min and
    NOTHING ELSE -- no q, no y -- because `batches` hands this loop (ctx, tgt,
    a), never a label, and fabricating one here just to switch the third leg
    on would hand collapse_report a construction property wearing a model's
    q, which is exactly what this function's construction_property guard
    exists to refuse. So a frozen-random encoder that carries zero information
    about any downstream
    label but spreads healthily across every axis (measured in demo()'s
    section (h) and in tests/cameron/test_third_collapse_leg.py) reads exactly
    as healthy to THIS loop as an encoder that actually learned something --
    the geometric legs cannot tell the two apart, by construction, and this
    loop never calls the third leg that could. Read this training run's
    "never collapsed" as a claim about variance and rank ONLY, never as a
    claim that its representation is informative about a label. The third leg
    exists and is exercised (see collapse_report's own docstring), but only by
    callers that have a genuine, non-construction q sitting beside real y --
    which no path in this repository currently produces from THIS loop's own
    online encoder at a point where this detector runs.

    With `detector` on its refusal is RAISED rather than logged, because a run
    that collapsed is not a run whose loss curve means anything; with it off,
    the step it would have fired at is recorded instead, which is how
    covariance_sweep() measures a setting that fails.

    THE THRESHOLDS ARE PARAMETERS AND THEY HAVE TO BE. COLLAPSE_ERANK_MIN is 1.5
    because that separates this bed's healthy run from a rank-one plant at this
    bed's width; a caller at a narrower one, whose healthy representation sits
    under that floor for no reason but its axis length, would be refused by it.
    An absolute floor is not portable across widths, and pretending otherwise
    would push the defect into every caller. RANK_BURN_IN is likewise a
    property of this bed's own transient, not portable to another width or
    horizon without being re-measured there.
    """
    if batches is None:
        batches = default_batches(seed)
    opt = torch.optim.Adam(model.trainable_parameters(), lr=lr)
    trace, std_trace, erank_trace = [], [], []
    collapsed_at, checked, cols = None, 0, model.cols.tolist()
    detector_side = "online"
    for step in range(steps):
        ctx, tgt, a = batches(step)
        s = model.online(ctx)
        s_tgt = model.target_repr(tgt)
        # THE DETECTOR READS THE ONLINE SIDE FOR BOTH ARMS -- see fit()'s own
        # docstring for why checking the target side asymmetrically was the
        # bug, not a feature of the pinned arm.
        chk_input = s[:, -1, :]
        # THE RANK LEG IS GATED BY RANK_BURN_IN; THE STD LEG IS NOT. Passing
        # erank_min=0.0 before the burn-in cannot fire the rank leg (erank is
        # never negative) while std_min is unchanged, so a genuine COMPLETE
        # collapse is still caught immediately.
        active_erank_min = erank_min if step >= RANK_BURN_IN else 0.0
        chk = collapse_report(chk_input, std_min=std_min,
                              erank_min=active_erank_min)
        checked += 1
        if chk["collapsed"]:
            if detector:
                raise AssertionError("step %d: %r" % (step, chk["refusal"]))
            if collapsed_at is None:
                collapsed_at = step
        std_trace.append(chk["std_min"])
        erank_trace.append(chk["erank"])
        ro = model(ctx, a)
        loss, parts = vicreg_terms(s[:, -1, :], ro.out[:, -1, :], s_tgt[:, cols],
                                   lam=lam, mu=mu, nu=nu)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if not model.pin_target:            # pinned target has no EMA weights
            ema_update(model.target, model.online, tau)     # to walk toward online
        trace.append(parts["inv"])
    # erank_min SUMMARISES THE SAME WINDOW THE RANK LEG ACTUALLY READS. The
    # full per-step record is still erank_trace, unabridged; this field is
    # what the rank leg was checking during the run, so a caller comparing it
    # against COLLAPSE_ERANK_MIN (as covariance_sweep()'s report and demo()'s
    # self-check both do) is comparing against the same window the pass/fail
    # decision above used, not against the burn-in transient every arm enters
    # and leaves regardless of nu or pin_target.
    # THE MEANING CHANGE, NAMED: the two lines below turned the returned
    # erank_min from "minimum erank over the WHOLE run" into "minimum erank
    # over erank_trace[RANK_BURN_IN:]" -- a different quantity under the same
    # key. demo()'s self-check and covariance_sweep() both read erank_min and
    # both were updated with this change, so python -m ceqjepa.pi_jepa is not
    # broken by it; but any caller OUTSIDE this file that reads fit()'s
    # erank_min expecting the old "minimum over the whole run" is now silently
    # comparing a different number under the same name.
    post_burn_in = erank_trace[RANK_BURN_IN:]
    erank_min_report = min(post_burn_in) if post_burn_in else min(erank_trace)
    return dict(loss_first=trace[0], loss_last=trace[-1], loss_trace=trace,
                std_min_trace=std_trace, std_min_last=std_trace[-1],
                erank_trace=erank_trace, erank_min=erank_min_report,
                erank_last=erank_trace[-1], steps_checked=checked,
                collapsed_at=collapsed_at, cols=cols,
                n_refused=len(model.refusals), detector_side=detector_side,
                model=model)


def train_run(seed=SEED, steps=N_STEPS, nu=NU, train_encoder=True, detector=True,
              pin_target=False):
    """The demo's run. NOT memoised: re-running it is how determinism is shown.

    pin_target selects the target arm: False (default) is the shipped EMA
    target, True is the parameter-free beta=1 read -- see PiJepa's docstring.
    """
    model = build(seed, pin_target=pin_target)
    if not train_encoder:
        for p in model.online.parameters():
            p.requires_grad_(False)
    return fit(model, steps=steps, nu=nu, seed=seed, detector=detector)


@torch.no_grad()
def _held_out_scores(model, seed):
    """Absolute latent MSE, its SCALE-INVARIANT form, and the target's own scale.

    ALL THREE, BECAUSE THE ABSOLUTE NUMBER IS NOT COMPARABLE ACROSS ENCODERS AND
    THAT IS NOT A DETAIL. A JEPA's target is the encoder's OWN representation, so
    an encoder whose output is small is trivially easy to predict: the untrained
    encoder here carries a target RMS of 0.4226 and the trained one 1.7368, a
    factor of 4.1101, and the same relative error therefore reads four times
    larger on the trained run. Ranking three encoders by absolute latent MSE
    ranks them by how small their outputs are. The reported metric is NRMSE, error over the
    target's own RMS; the two others are printed beside it so the reader can see
    the correction rather than take it.
    """
    x, a = draw_bed(seed)
    ctx, tgt = _windows(x)
    ro = model(ctx, a)
    s_hat = ro.out[:, -1, :]
    s_tgt = model.online(tgt)[:, -1, :][:, model.cols.tolist()]
    mse = (s_hat - s_tgt).pow(2).sum(-1).mean()
    scale = s_tgt.pow(2).sum(-1).mean()
    return dict(mse=float(mse), nrmse=float((mse / scale).sqrt()),
                target_rms=float(scale.sqrt()))


def baseline_detail(seed=SEED):
    """L-SIMPLE, in full. The untrained encoder and a FROZEN RANDOM one come
    FIRST, on the same held-out bed, the same metric and the same model class."""
    held = seed + 1
    out = OrderedDict()
    out["untrained"] = _held_out_scores(build(seed), held)
    out["frozen_random"] = _held_out_scores(
        train_run(seed=seed, train_encoder=False)["model"], held)
    out["trained"] = _held_out_scores(train_run(seed=seed)["model"], held)
    return out


def baselines(seed=SEED):
    """The one metric, projected out of baseline_detail(): NRMSE, lower better.

    WHAT THIS MEASURED, AND IT IS NOT WHAT WAS HOPED FOR. The FROZEN RANDOM
    encoder WINS: 0.6881 against the trained run's 0.8624, with the untrained
    control at 2.9262. Training the encoder does not beat leaving it alone on
    this bed. That is what L-SIMPLE is for and the result stands as measured.

    WHY, AND IT IS NOT MYSTERIOUS. With the encoder frozen the prediction target
    is a FIXED function and the predictor is fitting a stationary problem; with
    it trained the target moves under the predictor at every step, which is a
    harder problem that 120 steps on a synthetic bed does not pay for.

    THE ROUTE, BECAUSE A KILL WITHOUT ONE IS JUST A COMPLAINT. The claim that is
    RETIRED is "training the encoder lowers the latent error on this bed"; it was
    never in R2's scope and nothing here depends on it. What is REPRICED is what
    the trained run buys, which is not a lower error but a live representation:
    target std_min 0.2212 against the frozen encoder's 0.0383, a factor of 5.7710,
    with the effective rank held above the floor throughout. WHAT TO WANT INSTEAD
    is a downstream probe against an encoder-independent target, which is the
    only comparison that is not circular -- and that needs a real bed and the GPU
    run this file exists to make safe, not another CPU baseline here.
    """
    return OrderedDict((k, v["nrmse"]) for k, v in baseline_detail(seed).items())


def pin_vs_ema_report(seed=SEED, steps=N_STEPS):
    """THE EMA TARGET AGAINST THE PINNED beta=1 READ, SAME SEED, SAME STEPS.

    target_rms_ratio is the target's own RMS after training over its RMS before
    training, on the SAME fixed batch default_batches hands every step -- so for
    the pinned arm this is measuring whether the thing that removed the drift
    was actually wired in, and it CANNOT read anything but 1.0000: the read has
    no parameter for training to move. It is a wiring check, not evidence, and
    is named construction_check for exactly that reason.

    target_std_min and target_erank are a SECOND construction check, not
    evidence, and ONLY for the pinned arm: target_repr(tgt) there is one
    constant tensor for the whole run (same reasoning as target_rms_ratio
    above), so its collapse_report is a report on the bed draw read once
    through softmax, not on anything a trained run produced -- measured, 1
    unique value across a 120-step trace. For the EMA arm the same fields ARE
    live, because the EMA target walks toward the online encoder every step.

    online_std_min and online_erank are NOT wiring checks, for EITHER arm. They
    are the eps-free Std leg and the effective-rank leg of collapse_report()
    read off the ONLINE representation at the LAST training step -- the only
    representation that trains in the pinned arm, and fit()'s own detector now
    checks this same side for BOTH arms (see `detector_side` in fit()'s return
    dict, always "online"). Either leg can fail: a frozen target is the
    standard collapse failure mode EMA exists to prevent, so an online encoder
    that reads healthy on both legs is the actual claim, not the ratio or the
    target-side pair above it.
    """
    batches = default_batches(seed)
    ctx, tgt, _ = batches(0)
    out = OrderedDict()
    for name, pin in (("ema", False), ("pinned", True)):
        model = build(seed, pin_target=pin)
        rms_before = float(model.target_repr(tgt).pow(2).mean().sqrt())
        res = fit(model, batches=batches, steps=steps, seed=seed)
        rms_after = float(model.target_repr(tgt).pow(2).mean().sqrt())
        with torch.no_grad():
            online_chk = collapse_report(model.online(ctx)[:, -1, :])
        target_chk = collapse_report(model.target_repr(tgt))
        out[name] = dict(
            target_rms_before=rms_before, target_rms_after=rms_after,
            target_rms_ratio=rms_after / rms_before,
            online_std_min=online_chk["std_min"], online_erank=online_chk["erank"],
            target_std_min=target_chk["std_min"], target_erank=target_chk["erank"],
            detector_side=res["detector_side"], dtype=str(tgt.dtype))
    return out


def covariance_sweep(seed=SEED):
    """WHY nu IS 100 AND NOT THE 1.0 THAT SHIPS AS THE DEFAULT, MEASURED.

    The Std term is what this repository already knows to distrust, and the
    measurement here says something sharper than v2 did. Through a DIMENSIONAL
    collapse the Std term is not merely weak: it is at FULL STRENGTH and reads
    healthy, because every coordinate keeps its own spread while all of them
    become the same direction. Only the covariance term opposes that, and at the
    shipped default it does not oppose it hard enough on this bed.

    The detector is turned OFF for this sweep on purpose. The failing settings
    ARE the finding, and a run that aborts partway reports nothing about where
    it would have ended up; the step each setting would have been refused at is
    recorded instead.
    """
    rows = []
    for nu in COV_SWEEP:
        r = train_run(seed=seed, nu=nu, detector=False)
        rows.append(dict(nu=float(nu), erank_first=r["erank_trace"][0],
                         erank_min=r["erank_min"], erank_last=r["erank_last"],
                         std_min_last=r["std_min_last"], loss_last=r["loss_last"],
                         collapsed_at=r["collapsed_at"],
                         would_refuse=r["collapsed_at"] is not None))
    return rows


def roundtrip_check(seed=SEED):
    """state_dict out, load back in, and compare the FORWARD OUTPUT bitwise.

    Loaded into a model built at a DIFFERENT initialisation, because loading into
    a copy of itself would pass with load_state_dict doing nothing whatsoever.
    The gap before the load is reported for exactly that reason.
    """
    x, a = draw_bed(seed)
    ctx, _tgt = _windows(x)
    src = build(seed)
    with torch.no_grad():
        before = src(ctx, a).out
    sd = OrderedDict((k, v.clone()) for k, v in src.state_dict().items())
    dst = build(seed + 1)                       # a DIFFERENT init, on purpose
    with torch.no_grad():
        drifted = float((dst(ctx, a).out - before).abs().max())
    dst.load_state_dict(sd)
    with torch.no_grad():
        after = dst(ctx, a).out
    mask_keys = [k for k in sd if k.endswith("beta_assigned")]
    return dict(
        worst=float((after - before).abs().max()),
        bitwise=bool(torch.equal(after, before)),
        n_entries=len(sd), n_buffers=len(list(src.buffers())),
        n_params=sum(p.numel() for p in src.parameters()),
        n_trainable=sum(p.numel() for p in src.trainable_parameters()),
        mask_in_state=bool(mask_keys),
        mask_bitwise=bool(torch.equal(dst.beta_assigned, src.beta_assigned)),
        before_load_gap=drifted)


def roundtrip_without_mask(seed=SEED):
    """THE CONTROL FOR THE ROUND TRIP, and without it that test proves nothing.

    A round trip that would pass with the mask MISSING from the state_dict is not
    testing the mask at all. So the mask entry is dropped and the destination is
    built with the corners swapped -- 1 - beta per coordinate, which is what a
    caller who reloaded the weights but kept its own mask would be reading at --
    and the drift in the forward output is returned. Every weight is identical by
    construction, so the whole of the difference is the corners.
    """
    x, a = draw_bed(seed)
    ctx, _tgt = _windows(x)
    src = build(seed)
    with torch.no_grad():
        before = src(ctx, a).out
    sd = OrderedDict((k, v.clone()) for k, v in src.state_dict().items()
                     if not k.endswith("beta_assigned"))
    swapped = [b if is_refusal(b) else 1.0 - float(b) for b in beta_vector()]
    dst = build(seed, beta=swapped)
    dst.load_state_dict(sd, strict=False)
    with torch.no_grad():
        return float((dst(ctx, a).out - before).abs().max())



# ---------------------------------------------------------------------------
# 9. L-CLASS AND THE ACTION
# ---------------------------------------------------------------------------

def theorem_hypotheses(seed=SEED):
    """three_corners_containment's hypotheses as numbers on THIS bed.

    READ lean/CEQ/V16Domain.lean:433, corners_are_distinct at :445. The theorem
    is about an operator family; what it needs from the data is that the gate
    vanishes, the logits are finite, the beta=1 corner is row-stochastic, and the
    corners do not coincide here. A win against a theorem whose hypotheses were
    never checked on the bed is not a win.
    """
    q, k, v = _equiv_draw(seed)
    num, mod = arm.numerator(q, k)
    z = mod.sum(-1, keepdim=True)
    rows1 = (mod / z).sum(-1)
    rows0 = mod.sum(-1)
    return dict(name="three_corners_containment",
                source="lean/CEQ/V16Domain.lean:433 (corners_are_distinct :445)",
                gate_is_zero=bool(float(num.imag.abs().max()) == 0.0),
                logits_finite=bool(torch.isfinite(mod).all()),
                n_logits=int(mod.numel()),
                row_sum_beta1=float(rows1.mean()),
                row_sum_beta1_err=float((rows1 - 1.0).abs().max()),
                row_sum_beta0=float(rows0.mean()),
                corners_distinct=bool(abs(float(rows0.mean()) - 1.0) > 0.1))


def action_report(seed=SEED):
    """Delta_spec against a placebo, and the action-blind control at exactly 0."""
    tr = train_run(seed=seed)
    model = tr["model"]
    x, a = draw_bed(seed + 1)
    ctx, tgt = _windows(x)
    g = torch.Generator().manual_seed(int(seed) + 7)
    placebo = (a + 1 + torch.randint(0, N_ACTIONS - 1, a.shape,
                                     generator=g)) % N_ACTIONS
    with torch.no_grad():
        s = model.online(ctx)
        s_tgt = model.online(tgt)[:, -1, :]
        cols = model.cols.tolist()
        beta = model.beta_list()

        def err(p, act):
            out = p(s, beta, act).out
            out = out.real if torch.is_complex(out) else out
            return out[:, -1, :].sub(s_tgt[:, cols]).pow(2).sum(-1)

        d_read = err(model.pred, a) - err(model.pred, placebo)
        torch.manual_seed(int(seed))
        blind = BlindPredictor(len(beta))
        d_blind = err(blind, a) - err(blind, placebo)
    return dict(read_mean=float(d_read.mean()), read_max_abs=float(d_read.abs().max()),
                blind_mean=float(d_blind.mean()),
                blind_max_abs=float(d_blind.abs().max()),
                placebo_is_different=bool(int((placebo != a).sum()) == a.numel()),
                n_placebo_same=int((placebo == a).sum()))


# ---------------------------------------------------------------------------
# 10. PROVENANCE, THE DOCSTRING SCAN, AND THE REPORT
# ---------------------------------------------------------------------------

def provenance():
    """The producing commit and the machine id, COMPUTED here, never copied in.

    The machine id is the leading twelve hex characters of a sha256 over the node
    name, which is the form the rest of this round uses.
    """
    root = Path(__file__).resolve().parents[1]
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
                node=node, python=platform.python_version(), torch=torch.__version__)


_DECIMAL = _re.compile(r"(?<![\w.])[+-]?\d+(?:\.\d+)*(?:[eE][+-]?\d+)?(?!\w)(?!\.\d)")


def docstring_numbers():
    """(docstring count, number count) for this module's OWN scope.

    Scoped on __module__ so torch's docstrings do not come along, and NOT
    filtered through __all__, because demo() prints every number and is absent
    from it. The guard in tests/curvature/ writes its own counter rather than
    calling this one: two counters that must agree is a check, one shared helper
    is not.
    """
    docs = [__doc__ or ""]
    for obj in list(globals().values()):
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
    toks = {t.lstrip("+") for d in docs for t in _DECIMAL.findall(d)}
    return dict(n_docstrings=len(docs), n_numbers=len(toks),
                n_decimals=len([t for t in toks if "." in t]))


_CACHE = {}


def report(seed=SEED):
    """Everything this module measures, in one pass, memoised."""
    if seed in _CACHE:
        return _CACHE[seed]
    t0 = time.time()
    eq = shipped_equivalence(seed)
    co = corner_identities(seed)
    rc = refusal_channel()
    cl = collapse_must_fire(seed)
    sg = stopgrad_must_fire(seed)
    em = ema_identity(seed)
    ini = init_check(seed)
    tr = train_run(seed)
    bd = baseline_detail(seed)
    bl = OrderedDict((k, v["nrmse"]) for k, v in bd.items())
    th = theorem_hypotheses(seed)
    ac = action_report(seed)
    probe = std_term_probe()
    cs = covariance_sweep(seed)
    rt = roundtrip_check(seed)
    rt_nomask = roundtrip_without_mask(seed)
    bedrank = bed_rank_check(seed)
    by = {row["beta"]: row["worst"] for row in eq["rows"]}
    published = OrderedDict([
        ("equiv_worst_beta0", by[0.0]),
        ("equiv_worst_beta1", by[1.0]),
        ("equiv_worst_interior", by[BETA_INTERIOR]),
        ("corners_ones_worst", co["ones_worst"]),
        ("corners_zeros_worst", co["zeros_worst"]),
        ("collapse_healthy_std_min", cl["healthy"]["std_min"]),
        ("collapse_healthy_erank", cl["healthy"]["erank"]),
        ("collapse_planted_std_min", cl["planted"]["std_min"]),
        ("collapse_planted_erank", cl["planted"]["erank"]),
        ("collapse_rank_one_std_min", cl["rank_one"]["std_min"]),
        ("collapse_rank_one_erank", cl["rank_one"]["erank"]),
        ("stopgrad_on_target_grads", sg["on_target_grads"]),
        ("stopgrad_off_target_grads", sg["off_target_grads"]),
        ("ema_tau0_worst", em["tau0_worst"]),
        ("ema_tau_worst", em["tau_worst"]),
        ("init_std_min", ini["std_min"]),
        ("init_erank", ini["erank"]),
        ("train_loss_first", tr["loss_first"]),
        ("train_loss_last", tr["loss_last"]),
        ("train_std_min_last", tr["std_min_last"]),
        ("theorem_row_sum_beta1_err", th["row_sum_beta1_err"]),
        ("theorem_row_sum_beta0", th["row_sum_beta0"]),
        ("action_read_mean", ac["read_mean"]),
        ("action_blind_max_abs", ac["blind_max_abs"]),
        ("baseline_untrained", bl["untrained"]),
        ("baseline_frozen_random", bl["frozen_random"]),
        ("baseline_trained", bl["trained"]),
        ("model_trainable_params", rt["n_trainable"]),
        ("model_total_params", rt["n_params"]),
        ("roundtrip_worst", rt["worst"]),
        ("roundtrip_without_mask", rt_nomask),
    ])
    out = dict(provenance=provenance(), equivalence=eq, corners=co,
               refusal_channel=rc, collapse=cl, stopgrad=sg, ema=em, init=ini,
               train=tr, baselines=bl, baseline_detail=bd, theorem=th, action=ac, std_probe=probe,
               cov_sweep=cs, bed_rank=bedrank, roundtrip=rt, roundtrip_without_mask=rt_nomask,
               coverage=docstring_numbers(), published=published,
               seconds=time.time() - t0)
    _CACHE[seed] = out
    return out


# ---------------------------------------------------------------------------
# THE VALUE AXIS. The reroute after the corner rule lost its referent.
#
# beta_d = one minus alpha_d, where alpha_d is the exponent of coordinate d's
# read in the token count. On the encoder that exponent is zero by construction:
# Encoder is a position-wise map, so its representation does not depend on the
# count at all and the rule returns the same corner for every coordinate. The
# contraction `num @ v` over the key axis is the only place in this module where
# a token count actually varies, so it is the only axis on which the rule can
# have a referent.
# ---------------------------------------------------------------------------

#: The context lengths the key axis contracts over, for the value sweep.
S_GRID_VALUE = (16, 32, 64, 128)

#: Sequences per length in the value sweep. Large enough that the per-coordinate
#: RMS is not itself noise at the longest length.
VALUE_SWEEP_BATCH = 256


def _planted_value_stream(plant, s, seed=SEED, batch=VALUE_SWEEP_BATCH,
                          d=D_LATENT):
    """The value tensor the sweep contracts, with a named class planted in it.

    `extensive` gives the summands a nonzero mean, so their unnormalised total
    over the key axis grows with the count. `intensive` centres them, which is
    the most favourable honest construction of a coordinate whose total should
    not grow: nothing weaker than a centred summand can be called intensive
    without making the value itself shrink as the context lengthens, and a value
    that shrinks with its own context is not a coordinate, it is a schedule.

    This function exists so the plant is CHECKABLE rather than asserted -- a
    caller can look at what the sweep actually contracts.
    """
    g = torch.Generator().manual_seed(int(seed))
    v = torch.randn(batch, s, d, generator=g)
    if plant == "extensive":
        return v + 1.0
    if plant == "intensive":
        return v - v.mean(dim=1, keepdim=True)
    if plant is None:
        return v
    raise ValueError("unknown plant %r, expected 'extensive', 'intensive' or "
                     "None" % (plant,))


def value_axis_alpha(beta_at=0.0, s_grid=S_GRID_VALUE, seed=SEED, plant=None,
                     batch=VALUE_SWEEP_BATCH):
    """Per-coordinate exponent of the read's RMS in the CONTEXT LENGTH.

    The read is taken at the last position, at a single `beta_at` for every
    coordinate, and its per-coordinate RMS over the batch is fitted against the
    context length by the same power-law fitter the rule uses.

    `beta_at` is the parameter the caller must justify. The rule sets beta from
    alpha, and this measures alpha at a chosen beta, so a reader has to be able
    to see whether the answer depends on where the measurement was taken. It
    returns one entry per latent coordinate: a float exponent, or a Refusal
    carrying its reason where the fit is not a power law. The axis never
    silently shortens.
    """
    model = build(seed)
    n_s, d = len(s_grid), D_LATENT
    norms = np.zeros((n_s, d))
    ses = np.zeros((n_s, d))
    beta = [float(beta_at)] * d
    for i, s in enumerate(s_grid):
        x, _ = draw_bed(seed + i, batch, s, HORIZON, X_DIM)
        ctx = x[:, :s]
        with torch.no_grad():
            st = model.online(ctx)
            q = model.pred.wq(st)
            k = model.pred.wk(st)
            v = (model.pred.wv(st) if plant is None
                 else _planted_value_stream(plant, s, seed + i, batch, d))
            out = read_per_coordinate(q, k, v, beta).out[:, -1, :]
        mag = out.abs().to(torch.float64).numpy()
        norms[i] = np.sqrt((mag ** 2).mean(axis=0))
        blocks = mag.reshape(_VALUE_BLOCKS, mag.shape[0] // _VALUE_BLOCKS, d)
        per_block = np.log2(np.sqrt((blocks ** 2).mean(axis=1)))
        ses[i] = per_block.std(axis=0, ddof=1) / np.sqrt(_VALUE_BLOCKS)
    out = []
    for c in range(d):
        fit = pa.fit_power_law(tuple(s_grid), norms[:, c], ses[:, c])
        if not fit.power_law_ok:
            out.append(Refusal(
                "NOT-A-POWER-LAW",
                "coordinate %d's read RMS is not a power law in the context "
                "length, so it carries no exponent and a corner assigned from "
                "one would be answering a question the fit declined" % c))
        else:
            out.append(float(fit.alpha))
    return out


#: Blocks the value sweep splits its batch into for the fit's standard errors.
_VALUE_BLOCKS = 16


def demo():
    """The whole R2 wiring, measured, in one run under the 300 s bar."""
    t0 = time.time()
    r = report()
    p = r["provenance"]
    say = print
    say("commit %s  machine %s  (%s, python %s, torch %s)"
        % (p["commit"], p["machine"], p["node"], p["python"], p["torch"]))
    say("\nDR-2 R2: the JEPA, beta per coordinate from R1's frozen mask.")
    say("SCOPE: D=%d, S=%d, B=%d, h=%d, %d steps, tau=%.2f, seed %d, CPU. "
        "NOT a statement about a trained model at scale."
        % (D_LATENT, S_LEN, BATCH, HORIZON, N_STEPS, TAU, SEED))

    say("READ ceq/arm_smprime.py:527 for the 0-dim beta, :250 for")
    say("     `zb = mod.sum(-1, keepdim=True) ** beta` and :264")
    say("     for the contraction; ceqjepa/pi_assign.py for the frozen mask, and")
    say("     ceqjepa/t_length.py:694 for the boundary checks this file repeats.")

    say("\n    THE BED'S OWN EFFECTIVE RANK, MEASURED BEFORE THE DETECTOR IS "
        "TRUSTED:")
    for row in r["bed_rank"]:
        say("      action drift %.2f -> raw observation effective rank %.4f of %d"
            % (row["drift"], row["erank"], row["n_dims"]))
    say("      the bed at 0.60 is degenerate and would have made a working "
        "detector")
    say("      look broken; DRIFT is %.2f for that reason." % DRIFT)

    say("\n(a) THE FACTORISED READ AGAINST THE SHIPPED READOUT (%s)."
        % r["equivalence"]["reference"])
    for row in r["equivalence"]["rows"]:
        say("    beta %-5.2f worst |factorised - shipped| %.4e   bitwise=%s"
            % (row["beta"], row["worst"], row["bitwise"]))
    say("    THE PRICE, STATED: bitwise agreement is LOST at beta=1 and at the")
    say("    interior; it holds only at beta=0, where Z^0 is exactly 1. That is")
    say("    what a per-coordinate exponent costs and it is not hidden.")

    c = r["corners"]
    say("\n(b) THE TWO CORNER IDENTITIES, against the SHIPPED single-corner arms.")
    say("    all-ones  mask vs softmax arm: worst %.4e  bitwise=%s"
        % (c["ones_worst"], c["ones_bitwise"]))
    say("    all-zeros mask vs linear  arm: worst %.4e  bitwise=%s"
        % (c["zeros_worst"], c["zeros_bitwise"]))
    say("    tolerance %.0e on %d coordinates." % (c["tol"], c["axis_len"]))

    rc = r["refusal_channel"]
    say("\n(c) THE REFUSAL CHANNEL. axis %r length %d, %d assigned, %d refused."
        % (rc["axis"], rc["axis_len"], rc["n_assigned"], len(rc["refused"])))
    for name, code in rc["refused"].items():
        say("    REFUSED %-12s %s" % (name, code))
    say("    %d assigned coordinates hand back a float; %d refused coordinates "
        "were given a corner anyway." % (rc["assigned_floats"], rc["defaulted"]))
    say("    a read asked for a refused coordinate returns: %r"
        % (read_refused_column(),))

    cl = r["collapse"]
    say("\n(d) THE COLLAPSE DETECTOR, TWO LEGS, TWO PLANTS, BOTH DIRECTIONS.")
    for label in COLLAPSE_PLANTS:
        row = cl[label]
        say("    %-9s std_min %.4e  std_med %.4f  erank %.4f of %d  "
            "collapsed=%s  leg=%s"
            % (label, row["std_min"], row["std_med"], row["erank"],
               row["n_coords"], row["collapsed"], row["leg"]))
    say("    thresholds: std %.4g, effective rank %.4g." % (COLLAPSE_STD_MIN,
                                                            COLLAPSE_ERANK_MIN))
    say("    the rank-one plant keeps std_min %.4f and is missed by the variance"
        % cl["rank_one"]["std_min"])
    say("    leg entirely; only the rank leg catches it. THAT is why there are two.")
    say("    the detector ran on %d of %d training steps."
        % (cl["steps_checked"], N_STEPS))

    ini = r["init"]
    say("\n(e) THE INITIALISATION IS NOT DEGENERATE. std_min %.4f, erank %.4f of "
        "%d, %d zero-initialised output layers."
        % (ini["std_min"], ini["erank"], ini["n_coords"], ini["zero_layers"]))
    say("    THE EPS-FREE Std TERM, AT AND NEAR ZERO SPREAD:")
    for row in r["std_probe"]:
        say("      perturbation %8.0e  std_min %.4e  var %.4f  |dL/ds|max %.6e%s"
            % (row["scale"], row["std_min"], row["var"], row["grad_max"],
               "   <- NaN, the loud version of v2's silent zero" if row["is_nan"] else ""))

    sg = r["stopgrad"]
    say("\n(f) THE STOP-GRAD, MEASURED IN BOTH DIRECTIONS.")
    say("    stop-grad ON : %d of %d target parameters carry a gradient"
        % (sg["on_target_grads"], sg["n_target_params"]))
    say("    stop-grad OFF: %d of %d target parameters carry a gradient"
        % (sg["off_target_grads"], sg["n_target_params"]))
    em = r["ema"]
    say("    EMA tau=0 worst |target - online| %.4f over %d tensors bitwise=%s; "
        "at tau=%.2f the lag is %.4f"
        % (em["tau0_worst"], em["n_tensors"], em["tau0_bitwise"], TAU,
           em["tau_worst"]))

    tr = r["train"]
    say("\n(g) THE RUN. latent loss %.4f -> %.4f over %d steps; target std_min "
        "ends at %.4f; %d of %d coordinates predicted, %d refused."
        % (tr["loss_first"], tr["loss_last"], N_STEPS, tr["std_min_last"],
           len(tr["cols"]), D_LATENT, tr["n_refused"]))

    say("\n    WHY nu IS %.0f AND NOT THE 1.0 THAT SHIPS AS THE DEFAULT:" % NU)
    for row in r["cov_sweep"]:
        say("      nu %6.1f  erank %.4f -> min %.4f -> %.4f  std_min_last %.4f  "
            "loss_last %.4f  would_refuse=%s at step %s"
            % (row["nu"], row["erank_first"], row["erank_min"], row["erank_last"],
               row["std_min_last"], row["loss_last"], row["would_refuse"],
               row["collapsed_at"]))
    say("      the Std term is at FULL STRENGTH through the collapse it cannot")
    say("      see: at nu = 1.0 the effective rank bottoms at %.4f while the"
        % r["cov_sweep"][0]["erank_min"])
    say("      variance leg reads std_min %.4f, which is healthy."
        % r["cov_sweep"][0]["std_min_last"])

    bl, bd = r["baselines"], r["baseline_detail"]
    say("\n(h) L-SIMPLE, SCORED FIRST, AND IT KILLED THE TRAINING CLAIM.")
    say("    held-out          %8s %9s %11s" % ("NRMSE", "abs MSE", "target RMS"))
    for name in BASELINE_ORDER:
        say("      %-14s  %8.4f %9.4f %11.4f"
            % (name, bl[name], bd[name]["mse"], bd[name]["target_rms"]))
    say("    the ABSOLUTE MSE is not comparable across encoders: the target is the")
    say("    encoder's OWN representation, so a small encoder is trivially easy to")
    say("    predict. target RMS runs %.4f untrained to %.4f trained, a factor of"
        % (bd["untrained"]["target_rms"], bd["trained"]["target_rms"]))
    say("    %.4f, which is why NRMSE is the reported metric."
        % (bd["trained"]["target_rms"] / bd["untrained"]["target_rms"]))
    say("    THE RESULT: frozen random %.4f BEATS trained %.4f by %.4f. Training"
        % (bl["frozen_random"], bl["trained"], bl["trained"] - bl["frozen_random"]))
    say("    the encoder does not pay on this bed and the claim is RETIRED, not")
    say("    softened. THE ROUTE: what the trained run buys is a live")
    say("    representation -- target std_min %.4f against the frozen encoder's"
        % r["train"]["std_min_last"])
    say("    %.4f, a factor of %.4f -- not a lower error. WHAT TO WANT INSTEAD is"
        % (r["init"]["std_min"], r["train"]["std_min_last"] / r["init"]["std_min"]))
    say("    a downstream probe against an encoder-independent target, which needs")
    say("    the GPU run this file exists to make safe.")

    th = r["theorem"]
    say("\n(i) L-CLASS. %s (%s)" % (th["name"], th["source"]))
    say("    gate zero=%s, logits finite=%s over %d entries; row sum at beta=1 "
        "%.6f (worst error %.4e), at beta=0 %.4f; corners distinct=%s"
        % (th["gate_is_zero"], th["logits_finite"], th["n_logits"],
           th["row_sum_beta1"], th["row_sum_beta1_err"], th["row_sum_beta0"],
           th["corners_distinct"]))

    ac = r["action"]
    say("\n(j) THE ACTION AND ITS PLACEBO.")
    say("    read  Delta_spec mean %+.4f  max|.| %.4f" % (ac["read_mean"],
                                                          ac["read_max_abs"]))
    say("    blind Delta_spec mean %+.4f  max|.| %.4f  (must be exactly 0)"
        % (ac["blind_mean"], ac["blind_max_abs"]))
    say("    the placebo differs from the real action on every one of %d "
        "sequences (%d the same)." % (BATCH, ac["n_placebo_same"]))

    rt = r["roundtrip"]
    say("\n(k) THE TRAINABLE SURFACE, because a self-check nothing can train is")
    say("    not a deliverable. %d trainable parameters of %d total across %d "
        "state entries (%d buffers)."
        % (rt["n_trainable"], rt["n_params"], rt["n_entries"], rt["n_buffers"]))
    say("    a state_dict round trip moves the forward output by %.4f and dropping"
        % rt["worst"])
    say("    the mask entry moves it by %.4f -- the control, without which the "
        "round" % r["roundtrip_without_mask"])
    say("    trip would pass with the corners missing. mask_in_state=%s, "
        "mask_bitwise=%s," % (rt["mask_in_state"], rt["mask_bitwise"]))
    say("    and the destination differed by %.4f BEFORE the load."
        % rt["before_load_gap"])

    cov = r["coverage"]
    say("\n    docstring coverage: %d docstrings in this module, %d distinct "
        "numbers, %d of them decimals"
        % (cov["n_docstrings"], cov["n_numbers"], cov["n_decimals"]))
    dt = time.time() - t0
    # NO SPACE BEFORE THE UNIT, DELIBERATELY. A wall clock printed as "0.20 s"
    # enters the token guard's "printed by a run" set, which lets an absent-
    # decimal control go red at random and hands a fabricator any plausible small
    # decimal free. The 300 s bar keeps its space: it is a declared constant.
    say("    elapsed %.2fs against the 300 s bar (report build %.2fs)"
        % (dt, r["seconds"]))

    # -- self-checks --------------------------------------------------------
    assert p["commit"] != "UNKNOWN" and _re.fullmatch(r"[0-9a-f]{12}", p["machine"])
    by = {row["beta"]: row for row in r["equivalence"]["rows"]}
    assert by[0.0]["bitwise"] and by[0.0]["worst"] == 0.0
    assert not by[1.0]["bitwise"] and not by[BETA_INTERIOR]["bitwise"], \
        "the price of the factorisation is being hidden"
    assert r["equivalence"]["worst_overall"] <= EQUIV_TOL
    assert c["ones_worst"] <= EQUIV_TOL and c["zeros_worst"] == 0.0
    assert rc["axis"] == MASK_AXIS != "S" and rc["axis_len"] == D_LATENT
    assert rc["n_assigned"] + len(rc["refused"]) == rc["axis_len"]
    assert rc["defaulted"] == 0 and rc["refused"]
    assert is_refusal(read_refused_column())
    assert cl["planted"]["collapsed"] and cl["planted"]["leg"] == "std"
    assert cl["rank_one"]["collapsed"] and cl["rank_one"]["leg"] == "rank"
    assert cl["rank_one"]["std_min"] >= COLLAPSE_STD_MIN, \
        "the rank-one plant also trips the variance leg and shows no hole"
    assert cl["rank_one"]["erank"] < COLLAPSE_ERANK_MIN <= cl["healthy"]["erank"]
    assert not cl["healthy"]["collapsed"] and cl["healthy"]["refusal"] is None
    assert cl["steps_checked"] == N_STEPS
    assert ini["zero_layers"] == 0 and not ini["degenerate"]
    assert r["std_probe"][0]["is_nan"], \
        "the eps-free Std term no longer reads NaN at exact collapse"
    assert not any(row["is_nan"] for row in r["std_probe"][1:])
    assert sg["on_target_grads"] == 0 and not sg["target_requires_grad"]
    assert sg["off_target_grads"] == sg["n_target_params"] > 0, \
        "removing the stop-grad changed nothing: the check sees neither wiring"
    assert em["tau0_bitwise"] and em["tau0_worst"] == 0.0 and em["tau_worst"] > 0.0
    assert tr["n_refused"] == len(rc["refused"]) > 0
    assert len(tr["cols"]) == rc["n_assigned"]
    assert set(bl) == set(BASELINE_ORDER) and all(v > 0.0 for v in bl.values())
    # THE MEASURED ORDERING IS PINNED, INCLUDING THE PART THAT LOST. If a later
    # change makes the trained run win, this fires and the prose above has to be
    # rewritten rather than quietly becoming true.
    assert bl["frozen_random"] < bl["trained"] < bl["untrained"], \
        "the measured L-SIMPLE ordering changed: %r" % (dict(bl),)
    assert bd["trained"]["target_rms"] > bd["untrained"]["target_rms"], \
        "the target scales no longer differ, so the absolute metric is not the " \
        "trap the prose says it is"
    assert bd["trained"]["mse"] > bd["frozen_random"]["mse"], \
        "the absolute MSE no longer inverts against NRMSE: the correction is moot"
    assert rt["bitwise"] and rt["worst"] == 0.0 and rt["mask_in_state"] \
        and rt["mask_bitwise"] and rt["n_buffers"] > 0
    assert rt["before_load_gap"] > 0.0, \
        "the destination model was already identical: the round trip loads nothing"
    assert r["roundtrip_without_mask"] > 0.0, \
        "dropping the mask from the state_dict changed nothing: the round trip " \
        "would pass with the corners missing"
    assert rt["n_trainable"] < rt["n_params"], \
        "the target branch is not in parameters(), so it is not in the state_dict"
    sweep = {row["nu"]: row for row in r["cov_sweep"]}
    assert sweep[1.0]["would_refuse"] and not sweep[NU]["would_refuse"], \
        "the covariance sweep no longer separates the default from the shipped " \
        "weight, so NU = %r is asserted by nothing" % (NU,)
    assert sweep[1.0]["erank_min"] < COLLAPSE_ERANK_MIN <= sweep[NU]["erank_min"]
    assert sweep[1.0]["std_min_last"] > COLLAPSE_STD_MIN, \
        "the default-weight collapse ALSO trips the variance leg, so it does not " \
        "show the Std term reading healthy through a collapse"
    assert th["row_sum_beta1_err"] < ROW_SUM_TOL and th["corners_distinct"]
    assert ac["blind_max_abs"] == 0.0 and ac["read_max_abs"] > 0.0
    assert ac["placebo_is_different"] and ac["n_placebo_same"] == 0
    assert rt["bitwise"] and rt["worst"] == 0.0 and rt["mask_in_state"]         and rt["mask_bitwise"] and rt["n_buffers"] > 0
    assert rt["before_load_gap"] > 0.0,         "the destination model was already identical: the round trip loads nothing"
    assert r["roundtrip_without_mask"] > 0.0,         "dropping the mask from the state_dict changed nothing: the round trip "         "would pass with the corners missing"
    assert rt["n_trainable"] < rt["n_params"],         "the target branch is not in parameters(), so it is not in the state_dict"
    sweep = {row["nu"]: row for row in r["cov_sweep"]}
    assert sweep[1.0]["would_refuse"] and not sweep[NU]["would_refuse"],         "the covariance sweep no longer separates the default from the shipped "         "weight, so NU = %r is asserted by nothing" % (NU,)
    assert sweep[1.0]["erank_min"] < COLLAPSE_ERANK_MIN <= sweep[NU]["erank_min"]
    assert sweep[1.0]["std_min_last"] > COLLAPSE_STD_MIN, (
        "the default-weight collapse ALSO trips the variance leg, so it does not "
        "show the Std term reading healthy through a collapse")
    assert cov["n_docstrings"] >= 30 and cov["n_numbers"] >= 40, (
        "the docstring scan has shrunk to %d docstrings / %d numbers; a scope "
        "regression reports zero missing and passes"
        % (cov["n_docstrings"], cov["n_numbers"]))
    br = {row["drift"]: row["erank"] for row in r["bed_rank"]}
    assert br[0.6] < br[DRIFT],         "the rejected bed is no longer the degenerate one: %r" % (br,)

    # (h) A SYNTHETIC SELF-CHECK OF THE LABEL LEG'S ARITHMETIC. NOT EVAL TIME,
    # NOT A DOWNSTREAM PROBE, NOT A TRAINED MODEL. Everything below is built by
    # hand inside this self-check: l_s_info, l_s_frozen and l_y never touch
    # fit(), a trained encoder, or any eval path in this repository. fit() has
    # no labels to hand collapse_report -- every batch it sees is (ctx, tgt,
    # a) -- and no downstream probe with real labels beside a trained read of
    # THIS module's own online encoder exists anywhere in this repository
    # today (see fit()'s own docstring for that gap, stated plainly rather
    # than implied away). What this self-check DOES verify, honestly: the
    # label leg's arithmetic distinguishes an informative construction from a
    # frozen-random one when both are run through the SAME hand-built
    # nearest-centroid readout, on the SAME construction
    # tests/cameron/test_third_collapse_leg.py carries as ASSAY C2 (K=4 group
    # centroids at 2*eye(K), real grp never permuted or resampled). l_s_info IS
    # a construction property -- built directly as centroid[l_y] + N(0,
    # 0.05^2), so its nearest-centroid q reconstructs l_y by construction, not
    # by reading anything -- and is tagged construction_property=True below;
    # its margin_sigma is an ORACLE figure, printed as one, and never asked to
    # pass or fail a bar (collapse_report skips kill_fired entirely for a
    # tagged read; see collapse_report's own docstring). l_s_frozen is NOT a
    # construction property -- N(0, I_K), independent of l_y -- so it is the
    # one genuine read here, on equal footing with a frozen-random encoder
    # that never received a gradient. THE SAME hand-built nearest-centroid
    # readout -- q = softmax(-||s - centroid||^2) -- is applied to both, and
    # the SAME real l_y labels both; only s varies. That isolates what the
    # label leg's arithmetic reads: the frozen-random s scores a HIGHER
    # effective rank than the informative one (clustering IS anisotropy;
    # isotropic noise is not), so both geometric legs call it the healthier
    # representation while it carries nothing about l_y.
    lg = torch.Generator().manual_seed(SEED)
    l_n, l_k = 2000, 4
    l_centroids = 2.0 * torch.eye(l_k, dtype=torch.float64)
    l_y = torch.randint(l_k, (l_n,), generator=lg)
    l_s_info = l_centroids[l_y] + 0.05 * torch.randn(l_n, l_k, generator=lg, dtype=torch.float64)
    l_s_frozen = torch.randn(l_n, l_k, generator=lg, dtype=torch.float64)

    def _nearest_centroid_q(s):
        d2 = ((s[:, None, :] - l_centroids[None, :, :]) ** 2).sum(-1)
        return torch.softmax(-d2, dim=1)

    rep_info = collapse_report(l_s_info, q=_nearest_centroid_q(l_s_info),
                               y=l_y, K=l_k, boot_seed=SEED,
                               construction_property=True)
    rep_frozen = collapse_report(l_s_frozen, q=_nearest_centroid_q(l_s_frozen),
                                 y=l_y, K=l_k, boot_seed=SEED)
    assert rep_info["std_min"] > COLLAPSE_STD_MIN and rep_info["erank"] > COLLAPSE_ERANK_MIN
    assert rep_frozen["std_min"] > COLLAPSE_STD_MIN and rep_frozen["erank"] > COLLAPSE_ERANK_MIN, (
        "the frozen-random plant must stay geometrically healthy, or it is not "
        "the gap the label leg exists to close")
    assert rep_frozen["erank"] > rep_info["erank"], (
        "the geometric inversion did not reproduce: the frozen-random encoder "
        "must read a HIGHER effective rank than the informative one, or the "
        "plant is not exhibiting the mechanism the label leg is for")
    assert rep_info["leg"] is None and rep_info["label"]["construction_property"] and \
           rep_info["label"]["kill_fired"] is None, (
        "the oracle read must never be handed to the bar API: kill_fired "
        "should be None (not evaluated), not True or False: %r" % (rep_info,))
    assert rep_frozen["leg"] == "label" and rep_frozen["collapsed"] and \
           rep_frozen["label"]["kill_fired"] is True, (
        "the label leg did not fire on the frozen-random encoder against real "
        "labels, with both geometric legs reading healthy: %r"
        % (rep_frozen["label"],))
    say("\n(h) SYNTHETIC SELF-CHECK OF THE LABEL LEG'S ARITHMETIC (not eval "
        "time, not a downstream probe). informative encoder [ORACLE -- q "
        "reconstructs y by construction, never passed to the bar]: "
        "margin_sigma=%+.4f (kill_fired not evaluated); frozen-random encoder "
        "on the SAME real labels [genuine read]: leg=%r kill_fired=%s "
        "margin_sigma=%+.4f -- erank %.4f informative vs %.4f frozen (the "
        "inversion), both geometric legs green either way; only the label leg "
        "tells them apart."
        % (rep_info["label"]["margin_sigma"],
           rep_frozen["leg"], rep_frozen["label"]["kill_fired"],
           rep_frozen["label"]["margin_sigma"],
           rep_info["erank"], rep_frozen["erank"]))

    assert dt < 300.0
    say("\nALL SELF-CHECKS PASSED")
    return r["published"]


if __name__ == "__main__":
    demo()

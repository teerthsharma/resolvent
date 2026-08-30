"""M3 — long-range sign capability, with an ABSOLUTE bar.

WHY THIS IS THE ITEM THAT DECIDES THE PROJECT. M1 (negative influence exists) is
a PRECONDITION -- SignGT, SDA and Cog Attention all pass it, so it is not
novelty. M2 (flat sign-flip rate) is a STATISTIC, and CHECKLIST.md's preamble
says statistics count for nothing. M3 is the only MANDATORY item that is a
CAPABILITY, measured against softmax's own number in the same table.

THE TASK. One flipper token at distance d from the query decides the SIGN of a
payload carried by a different token:

    y = payload_value * flipper_sign

payload and flipper sit at DIFFERENT positions, so the model cannot solve it by
retrieving one negative value -- it must combine two positions multiplicatively.
That is the shape of "token B suppresses token A's contribution", which is what
a non-negative operator provably cannot represent in its influence Jacobian and
must instead push into the MLP.

BASELINE IMPOSSIBILITY IS MEASURED, NOT ASSUMED. Softmax plus an MLP CAN solve
this given enough capacity; the question is whether it still can at MATCHED
PARAMETERS and at d >= 256. So the softmax arm runs FIRST and its failure
distance is recorded before any pivot number exists, with results/ timestamps
proving the order. CHECKLIST M3 requires exactly this.

THE ORACLE IS EXECUTABLE AND THE CORPUS HOLDS NO ANSWER KEY. `make_batch`
returns (x, y) computed from the sampled data by `oracle()`. Nothing is written
to disk with its label attached. The W4 death was a corpus split by literal
value, which measured embedding coverage rather than generalization; a stored
answer key is the same failure with a shorter fuse.

THE ABSOLUTE BAR. NRMSE = RMSE / std(y). NRMSE = 1.0 is exactly the
predict-the-mean predictor. EVERY arm in the W4 round died ABOVE 1.0 -- worse
than a constant -- while still producing a pretty ordering between arms, and
that ordering was reported as a result. So the bar is absolute and is checked
before any arm is credited with beating it:

    KILL (CHECKLIST M3, pre-registered): pivot arm above NRMSE 1.0, OR CIs
    overlap softmax at every d >= 256, OR the unsigned-pivot ablation matches it
    (then the routing is the contribution, G4 fires, and the claim sentence must
    be rewritten before work continues).
"""
from __future__ import annotations

import argparse
import functools
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# T-FAMILY 2 IMPACT imports (symbolic, no real-market data) [READ scale/impact.py:1]
try:
    from scale.impact import (
        make_impact_batch as _make_impact_batch,
        make_impact_hetero_batch as _make_impact_hetero_batch,
        impact_oracle as _impact_oracle,
        impact_features as _impact_features,
        impact_flipper_dependence as _impact_flipper_dependence,
    )
except Exception:
    _make_impact_batch = _make_impact_hetero_batch = _impact_oracle = _impact_features = _impact_flipper_dependence = None  # type: ignore

#: channel layout of the input. Kept explicit so no arm can accidentally read a
#: label channel that should not exist.
CH_FLIP = 0      # +-1 at the flipper position, 0 everywhere else
CH_PAYLOAD = 1   # the value, at the payload position, 0 everywhere else
CH_NOISE = 2     # distractor channels start here


def oracle(x: torch.Tensor, f: int, p: int) -> torch.Tensor:
    """THE EXECUTABLE ORACLE. y = payload * flipper_sign, recomputed from x.

    Deliberately a function of `x` rather than of a stored label: a corpus file
    containing y is a corpus file that can be memorised, and the split-by-value
    failure (W4) is exactly what that produces.
    """
    return x[:, p, CH_PAYLOAD] * x[:, f, CH_FLIP]


def make_batch(n: int, s: int, d: int, *, d_model: int = 16, seed: int = 0,
               device=None):
    """(x, y, f, p). The flipper sits d positions before the query at s-1.

    The payload sits adjacent to the query so that RETRIEVING it is easy and the
    only hard part is the long-range SIGN. Otherwise a failure could be a
    retrieval failure wearing a sign failure's name.
    """
    if not (1 <= d < s - 1):
        raise ValueError(f"distance d={d} does not fit in s={s}")
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = device or torch.device("cpu")
    x = torch.randn(n, s, d_model, generator=g).to(dev) * 0.1
    f = s - 1 - d
    p = s - 2
    if f == p:
        raise ValueError("flipper and payload collide; increase s or reduce d")
    x[:, :, CH_FLIP] = 0.0
    x[:, :, CH_PAYLOAD] = 0.0
    signs = (torch.randint(0, 2, (n,), generator=g).to(dev).float() * 2 - 1)
    x[:, f, CH_FLIP] = signs
    x[:, p, CH_PAYLOAD] = torch.randn(n, generator=g).to(dev)
    return x, oracle(x, f, p), f, p


# ==========================================================================
# SECOND M3 TASK: counter_squared, the Hankel-gap task (S2)
#
# The M3 corpus is not a word corpus -- an example is a [s, d_model] float
# tensor, not a string. `counter_squared`, f(w) = ((#a) - (#b))**2, is expressed
# in it by letting CH_FLIP carry the LETTERS at every position rather than a
# single sign at one position: +1 is 'a', -1 is 'b'. Nothing else about the
# encoding changes, so the same arms, the same operator and the same training
# loop run unmodified.
#
# WHY THIS TASK AND NOT THE PLAIN COUNTER. The plain counter's Hankel is
# additive, H[u,v] = c_u + c_v, and `ceq/hankel.py::additive_nonneg_certificate`
# pins its rank_+ at exactly 2 with an explicit nonnegative factorisation -- no
# gap at any block size. Squaring is the smallest departure from the same
# counter that breaks additivity: H[u,v] = (c_u + c_v)**2 has real rank 3 and a
# rectangle-covering lower bound on rank_+ that GROWS with the number of count
# levels.
#
# THE PAYLOAD CHANNEL IS KEPT. It carries an independent N(0,1) draw that the
# label does not depend on, so `calibrate_bar`'s payload_only clause stays a
# live control rather than a comparison against a channel of zeros.
# ==========================================================================

def counter_squared_oracle(x: torch.Tensor, f: int, p: int) -> torch.Tensor:
    """((#a) - (#b))**2, read off CH_FLIP. Same signature as `oracle`.

    `f` and `p` are accepted and ignored: this task has no single flipper
    position, which is exactly the property that gives it the Hankel gap. The
    signature is kept so the function is drop-in wherever `oracle` is, including
    `calibrate_bar(oracle_fn=...)` and `m3_synthetic_settled.SyntheticSettledArm`.
    """
    return x[:, :, CH_FLIP].sum(dim=1) ** 2


def counter_squared_features(x: torch.Tensor, f: int, p: int) -> torch.Tensor:
    """The oracle features for `calibrate_bar`'s trained positive control.

    For `y = payload * sign` those are (flipper, payload). For this task the
    only sufficient statistic is the SUM of the letters, so handing the control
    a single letter would make the control fail for the right reason on the
    wrong features. The payload rides along as a distractor the control must
    learn to ignore, which also keeps the feature width at 2 and therefore the
    control's own initialisation draw identical to the shipped one.
    """
    return torch.stack([x[:, :, CH_FLIP].sum(dim=1), x[:, p, CH_PAYLOAD]], dim=-1)


def make_counter_batch(n: int, s: int, d: int, *, d_model: int = 16,
                       seed: int = 0, device=None):
    """(x, y, f, p) for `counter_squared`, in the M3 tensor format.

    `make_batch` builds the tensor, so the noise channels, the dtype, the
    payload draw and the (f, p) positions are the shipped ones and cannot drift.
    CH_FLIP is then overwritten at EVERY position with a drawn letter, and the
    label is recomputed from the tensor -- no answer key is stored, same rule as
    `oracle`.

    `d` is still required and still range-checked by `make_batch`, but it only
    positions the distractor payload here: this task has no distance parameter
    because it has no single decisive token.

    USE AN EVEN `s`. The task is well defined at any length, but the GAP is only
    DETECTABLE at even `s`. The fixed-length Hankel block at split `k` is
    ``H[u,v] = (c_u + c_v)**2``, whose zero entries sit where ``c_u = -c_v``;
    that needs the two count parities to agree, i.e. ``k = s - k (mod 2)``, i.e.
    `s` even. At odd `s` the block has NO zero entry, the rectangle-covering
    number collapses to 1, and `ceq.hankel.rank_plus_lower` returns nothing
    above the trivial `rank_R = 3` at any split. This is not guarded here
    because the task is still the task; it is measured, at every split, by
    `tests/cameron/test_m3_counter_squared.py`. M3's default s=64 is even.
    """
    x, _y, f, p = make_batch(n, s, d, d_model=d_model, seed=seed, device=device)
    g = torch.Generator(device="cpu").manual_seed(seed + 31337)
    letters = torch.randint(0, 2, (n, s), generator=g).float() * 2 - 1
    x[:, :, CH_FLIP] = letters.to(x.device)
    return x, counter_squared_oracle(x, f, p), f, p


def counter_squared_flipper_dependence(s: int) -> float:
    """The EXACT value `calibrate_bar`'s flipper_dependence must read, in closed
    form -- no constant is fitted and no threshold is chosen.

    Negating the letter at position `f` sends c = sigma_f + r to c - 2*sigma_f,
    so the label moves by |(c - 2*sigma_f)**2 - c**2| = |4*sigma_f*r| = 4*|r|
    exactly, where r is the sum of the OTHER s-1 letters. The denominator is
    E|c**2| = E[c**2] = s. Hence

        flipper_dependence = 4 * E|S_{s-1}| / s,   S_m a sum of m Rademachers,

    with E|S_m| = 2**-m * sum_k C(m,k) |2k - m|, computed here exactly.

    This is 2.0 for the negation-scope task and falls with s here, because no
    single letter dominates a global aggregate. That is not a defect of the
    task; it is the same property as the Hankel gap, seen through the bar.
    """
    m = s - 1
    tot = sum(math.comb(m, k) * abs(2 * k - m) for k in range(m + 1))
    return 4.0 * (tot / 2.0 ** m) / s


# ==========================================================================
# THE E-TASK FAMILY -- THE EQUILIBRIUM BECOMES THE LABEL (LOOP_PROMPT.md 1.7)
#
# WHAT THIS SECTION IS FOR. The two tasks above are STATIC EXPRESSIONS of their
# input -- a product of two entries, and a sum squared. Neither has a fixed
# point and neither needs an iteration, so an arm that runs a settling loop to
# convergence and then predicts one of them has been asked to show a settling
# advantage on a task where settling has nothing to compute. The project's goal
# sentence names the prediction TARGET as an equilibrium; until this section
# nothing in the corpus put one in a label.
#
# TWO FAMILIES, AND THEY ARE DIFFERENT OBJECTS ON PURPOSE.
#
#   the CHAIN family (E1, E3)  -- z* = (I - A)^{-1} b for a strictly lower
#       triangular signed A carried in the tokens. Well-posedness is NILPOTENCY:
#       (I - A) is invertible for every draw and the resolvent is a TERMINATING
#       sum, so the stabilisation time t* is EXACT rather than asymptotic. That
#       exactness is the whole point of E3 -- a dose-response dial that is a
#       hop count, not a tolerance.
#
#   the CONSEQUENCE family (E2) -- the coordinate of the new fixed point of a
#       damped best-response game after a one-token do()-shock. Well-posedness
#       is CONTRACTION, and the temperature comes from `ceq.nash.safe_tau`,
#       which already computes the threshold; nothing is invented here.
#
# THE ENCODING. An M3 example is a [s, d_model] float tensor, not a string, so
# `make_batch` builds the tensor and each family overwrites the channels it
# owns. The noise channels, the dtype, the payload draw and the (f, p) geometry
# stay the shipped ones and cannot drift. The payload channel is KEPT in both
# families, carrying an independent draw the label does not depend on, so
# `calibrate_bar`'s payload_only clause stays a live control.
# ==========================================================================
from ceq.nash import safe_tau                                     # noqa: E402

#: the chain's sub-diagonal coefficient, at every position. The first noise
#: channel, repurposed -- the same move `counter_squared` made with CH_FLIP.
CH_DRIVE = CH_NOISE

#: E2's game: `E_GAME` players, one row of the raw game matrix per token in
#: channels CH_GAME .. CH_GAME + E_GAME - 1, the bias in CH_GAME_BIAS. Six
#: players fit d_model=16 with seven noise channels left over.
E_GAME = 6
CH_GAME = CH_NOISE
CH_GAME_BIAS = CH_NOISE + E_GAME

#: `safe_tau`'s SHIPPED default margin. The Lipschitz constant of
#: `s -> sigmoid((M s + b) / tau)` is `||M||_2 / (4 tau)`, and `safe_tau`
#: returns `margin * ||M||_2 / 4`, so the constant is exactly `1 / margin` for
#: every game it is asked about -- a contraction for every drawn instance, by
#: construction rather than by rejection sampling.
E_TAU_MARGIN = 1.25
E_LIPSCHITZ = 1.0 / E_TAU_MARGIN

E2_ITERS = 200                 # best-response sweeps taken as the fixed point
E2_RESIDUAL_TOL = 1e-5         # the builder raises above this
E2_DIAL_TOL = 1e-3             # the tolerance E2's printed t* is quoted at

#: E2's difficulty dial. The chain family terminates, so its t* is a hop count.
#: E2 CONTRACTS, so its dial is the sweep count at which the closed-form bound
#: `E_LIPSCHITZ ** k` first falls below `E2_DIAL_TOL`. That bound is
#: conservative and is labelled as such: the MEASURED crossing of 1e-3 is at
#: k = 16 (n=2048, s=64, d=24, seed=0, this machine), against the bound's 31.
E2_T_STAR = math.ceil(math.log(E2_DIAL_TOL) / math.log(E_LIPSCHITZ))

#: MEASURED step budget for E2's trained positive control (`calibrate_bar`
#: clause 5). The clause trains its control on the RAW label while `run_arm`
#: trains every arm on the STANDARDISED one, so a small-scale label makes the
#: control strictly harder than the arms' own task. E2's label has std about
#: 0.062, and at the shipped 150 steps the control reads 2.446646 -- above the
#: bar, so the gate would report BROKEN for a reason that is about the control's
#: optimisation and not about the task. At 600 it reads 0.922725, at 2000
#: 0.525985 (n=2048, s=64, d=24, seed=0, lr=0.02, this machine). 600 is the
#: smallest of those three that passes, and it is a measurement, not a
#: preference. E2 must be run with `--steps 600` or more.
E2_STEPS = 600


def equilibrium_oracle(x: torch.Tensor, f: int, p: int) -> torch.Tensor:
    """z*_{s-1}, the last coordinate of `(I - A)^{-1} b`. Same signature as
    `oracle`, recomputed from `x`, nothing stored.

    `A` is the strictly lower triangular matrix whose only nonzero band is the
    sub-diagonal `A[i, i-1] = x[i, CH_DRIVE]`, and `b[i] = x[i, CH_FLIP]`. The
    resolvent of a strictly lower triangular matrix is a terminating sum, so
    `z* = sum_m A^m b` and its last coordinate is the forward scan below --
    equivalently the signed path sum `sum_j (prod_{k>j} a_k) b_j`.

    `f` and `p` are accepted and ignored: like `counter_squared_oracle`, this
    task has no single decisive position, and the signature is kept so the
    function is drop-in wherever `oracle` is.
    """
    a, b = x[:, :, CH_DRIVE], x[:, :, CH_FLIP]
    z = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    for i in range(x.shape[1]):
        z = a[:, i] * z + b[:, i]
    return z


def equilibrium_hop_reading(x: torch.Tensor, k: int) -> torch.Tensor:
    """The label as a `k`-HOP TRUNCATION: `sum_{m=0..k} (A^m b)_{s-1}`.

    This is the object that decides whether the label needs an iteration at all.
    Only the last `k + 1` tokens enter, so a model with a hop budget of `k`
    cannot do better than this reading. Because the discarded tail is a sum of
    `t* - k` independent unit-variance terms and the label is a sum of `t*` of
    them, the truncation's NRMSE against the label is

        sqrt((t* - k) / t*)   for k <= t*,   and exactly 0 for k >= t*

    -- EXACTLY 1.0 at k = 0, bounded away from the label at every `k < t*`,
    tightening with `k`, and EXACT at the full budget. The k = 0 end matters as
    much as the k = t* end: it says a zero-hop reading of this label is
    precisely the predict-the-mean predictor, so no part of the label is
    legible without hops. If this read 0 at k = 1 the family would be a third
    static task; `tests/cameron/test_m3_etasks.py` measures it at every rung.
    """
    a, b = x[:, :, CH_DRIVE], x[:, :, CH_FLIP]
    s = x.shape[1]
    z = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    for i in range(max(0, s - 1 - k), s):
        z = a[:, i] * z + b[:, i]
    return z


def equilibrium_features(x: torch.Tensor, f: int, p: int) -> torch.Tensor:
    """The oracle features for `calibrate_bar`'s trained positive control.

    The query token carries no driver (see `make_equilibrium_batch`), so the
    label's last recursion step is exactly `z*_{s-1} = a_{s-1} z*_{s-2}` and the
    control is handed those two numbers and must learn the multiply. That is the
    SAME width and the SAME difficulty class as the shipped control's
    (flipper, payload) product -- width 2, so this control's own initialisation
    draw is identical to the shipped one -- and it is given the oracle's INPUTS,
    never its output. Handing it `z*_{s-2}` alone does not work, and that is
    measurable rather than argued: `a_{s-1}` is Rademacher, so predicting
    `z*_{s-2}` gives NRMSE exactly `sqrt(2)`, above the bar at every `t*`.
    """
    a, b = x[:, :, CH_DRIVE], x[:, :, CH_FLIP]
    s = x.shape[1]
    z = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    for i in range(s - 1):
        z = a[:, i] * z + b[:, i]
    return torch.stack([z, a[:, s - 1]], dim=-1)


def make_equilibrium_batch(n: int, s: int, d: int, *, t_star: int | None = None,
                           d_model: int = 16, seed: int = 0, device=None):
    """(x, y, f, p) for the chain family. `t_star=None` means the full length.

    THE STABILISATION TIME IS EXACT AND IT IS THE ONLY DIAL. The sub-diagonal is
    set to zero at and before position `head = s - 1 - t_star`, so the chain
    reaching the query at `s - 1` is exactly `t_star` hops long and `A` is
    nilpotent of index `t_star + 1` on the coordinate that is read. `A^m b` at
    coordinate `s-1` is identically zero for `m > t_star`, so the resolvent
    terminates -- that is the well-posedness, and it is structural, not sampled.

    THE COEFFICIENTS ARE RADEMACHER, NOT GAUSSIAN, AND THE DRIVERS ARE GAUSSIAN.
    `|a_i| = 1` exactly inside the band, so every path weight has modulus 1 and
    the label is `N(0, t_star + 1)` EXACTLY -- which is what puts the truncation
    error and the flipper dependence in closed form with no constant fitted. The
    operator is therefore NILPOTENT rather than a norm contraction, and that is
    deliberate: a contraction would make `t*` a tolerance, and E3 needs it to be
    a hop count. Making the drivers Rademacher too would collapse the label to a
    parity times a magnitude, and this project already recorded parity as the
    wrong target (`tests/cameron/test_parity_is_the_wrong_target.py`).

    THE QUERY TOKEN CARRIES NO DRIVER: `b[s-1] = 0`. It was not zero in the
    first version of this builder and that cost the family the harness's own
    RED gate. `run_arm` requires the UNTRAINED arm to sit at or above NRMSE 1.0,
    and every arm here computes `z = x + A@x` at row `s-1`, so the query token's
    own channels reach the readout with ZERO hops. With a driver there, a
    `1 / (t*+1)` share of the label's variance was legible before a single
    training step and the untrained arm read BELOW the bar -- 0.993760 train /
    0.993600 eval at t*=1, 0.999442 / 0.998051 at t*=8, n_train=2048,
    n_eval=4096 -- so `m3_capability.py` aborted with INSTRUMENT BROKEN on three
    of the five rungs. Zeroing it makes the zero-hop reading EXACTLY the
    predict-the-mean predictor, which is the strongest form of the property this
    family is for: no part of an equilibrium label should be readable without
    hops.

    `f` IS RETURNED AS THE HEAD OF THE CHAIN, not as `make_batch`'s flipper.
    `make_batch` places its flipper at `s - 1 - d`, which for `t_star < d` sits
    OUTSIDE the live band, where negating it moves the label by exactly zero and
    the calibration's flipper clause would reject the correct task. The head is
    the position whose driver the label depends on most, and reporting it keeps
    `t_star` and the distance `d` independent knobs. `d` still positions the
    distractor payload and is still range-checked by `make_batch`.
    """
    x, _y, _f, p = make_batch(n, s, d, d_model=d_model, seed=seed, device=device)
    t = (s - 1) if t_star is None else int(t_star)
    if not (1 <= t <= s - 1):
        raise ValueError(f"t_star={t} does not fit in s={s}")
    head = s - 1 - t
    g = torch.Generator(device="cpu").manual_seed(seed + 777)
    b = torch.randn(n, s, generator=g).to(x.device)
    a = (torch.randint(0, 2, (n, s), generator=g).float() * 2 - 1).to(x.device)
    a[:, :head + 1] = 0.0
    b[:, s - 1] = 0.0
    x[:, :, CH_FLIP] = b.to(x.dtype)
    x[:, :, CH_DRIVE] = a.to(x.dtype)
    return x, equilibrium_oracle(x, head, p), head, p


def chain_flipper_dependence(s: int, *, t_star: int | None = None) -> float:
    """The EXACT value `calibrate_bar`'s flipper_dependence must read, in closed
    form -- no constant is fitted and no threshold is chosen.

    Negating the driver at the head of the chain sends `b_head -> -b_head`, and
    the head's path weight to the query has modulus 1, so the label moves by
    exactly `2 |b_head|`. The label is a sum of the `t*` drivers at positions
    `head .. s-2` with unit-modulus weights, so it is `N(0, t*)` exactly. Hence

        flipper_dependence = 2 E|N(0,1)| / E|N(0, t*)| = 2 / sqrt(t*).

    This is 2.0 for the negation-scope task and falls with `t*` here, because a
    single driver is a smaller part of a longer path sum. That is not a defect;
    it is the difficulty dial, seen through the bar.

    THE BAND HAS A MINIMUM SAMPLE SIZE, MEASURED. The clause estimates a ratio
    of two sample means, so the band is only a gate above the sampling spread.
    Measured on this machine (torch 2.5.1+cu121, 2 threads), 16 seeds per cell,
    s=64, d=24, as max |deviation from the exact value|:

        t*    exact       n=256     n=512     n=2048    n=4096    n=8192
         1  2.000000000  0.000000  0.000000  0.000000  0.000000  0.000000
         2  1.414213562  0.149327  0.071979  0.050166  0.021904  0.025629
         8  0.707106781  0.123496  0.085576  0.022130  0.014020  0.016892
        32  0.353553391  0.043141  0.044478  0.021788  0.015151  0.010677
        63  0.251976315  0.033327  0.030540  0.013626  0.007196  0.005730

    The `t* = 1` row is exact at every n and is not a fluke of seeding: with one
    driver the label is `+-b_head`, so the moved quantity is `2 |b_head|` and the
    scale is `|b_head|` PER EXAMPLE and the ratio has no sampling spread at all
    -- the same exact 2.0 the shipped negation-scope task has, which is what
    makes `t* = 1` the retrieval-regime control of this ladder rather than a new
    task. At the shipped tolerance 0.05 the floor for the rest is n >= 4096 for
    t* = 2 and n >= 2048 for t* in {8, 32}; t* = 63 clears it from n = 256.

    The distance to the NEAREST WRONG RUNG is what the band has to cover and it
    does: t*=2 against t*=8 is 0.707107, t*=8 against t*=32 is 0.353553, t*=32
    against t*=63 is 0.101577, all above 0.05 at every n at or above the floor.
    Both ends are measured; neither was chosen to fit.
    """
    t = (s - 1) if t_star is None else int(t_star)
    return 2.0 / math.sqrt(t)


def consequence_game(x: torch.Tensor):
    """(M, bias, tau) for E2's game, read off the tokens.

    `M` is symmetrised, exactly as `ceq.nash.nash_operator` does, so the game is
    a POTENTIAL game and its logit equilibrium is a stationary point of the
    potential -- symmetry is enforced by the reader rather than stored, so no
    draw can be asymmetric. `tau` is `ceq.nash.safe_tau` PER EXAMPLE. Calling
    `safe_tau` on the whole batch would return the max over it, which makes the
    label depend on the batch size; the per-example loop keeps the label a
    function of the example alone, and it calls the SHIPPED function rather than
    re-deriving `||M||_2 / 4` here.
    """
    g = x[:, :E_GAME, CH_GAME:CH_GAME + E_GAME]
    m = (g + g.transpose(-2, -1)) / 2.0
    bias = x[:, :E_GAME, CH_GAME_BIAS]
    tau = torch.tensor([safe_tau(m[i], margin=E_TAU_MARGIN)
                        for i in range(m.shape[0])],
                       dtype=m.dtype, device=m.device)
    return m, bias, tau


def _br_sweeps(m, bias, tau, v, iters):
    """`iters` damped best-response sweeps with player 0 CLAMPED to `v`.

    Clamping is graph surgery, not a bias nudge: player 0's incoming edges are
    removed and its stance is held, which is what `do(s_0 := v)` means. The
    start is the barycentre `1/2`, the maximum-entropy point that privileges no
    player -- the same start `qre_stance` uses.
    """
    st = torch.full((m.shape[0], E_GAME), 0.5, dtype=m.dtype, device=m.device)
    st[:, 0] = v
    for _ in range(iters):
        st = torch.sigmoid(((m @ st.unsqueeze(-1)).squeeze(-1) + bias)
                           / tau.unsqueeze(-1))
        st[:, 0] = v
    return st


def consequence_oracle(x: torch.Tensor, f: int, p: int, *,
                       k: int | None = None) -> torch.Tensor:
    """The CONSEQUENCE of one token's intervention on a downstream equilibrium.

        v  = sigmoid(x[:, f, CH_FLIP])            the intervening token's dose
        y  = z*_{m-1} under do(s_0 := v)  -  z*_{m-1} under do(s_0 := 1 - v)

    Both terms are coordinates of the NEW fixed point of the same contraction
    system after a one-token do()-shock; the label is their contrast, and `k`
    truncates the sweep count so the truncation ladder can be read.

    WHY THE CONTRAST AND NOT THE RAW COORDINATE. The raw new-fixed-point
    coordinate was built first and its own control killed it: an arm that
    ignores the intervening token entirely and predicts the UN-INTERVENED
    equilibrium reads NRMSE 0.194150 against it (n=2048, s=64, d=24, seed=0), so
    96% of that label is the game draw and the intervention is a rounding error
    -- the same "near zero BY CONSTRUCTION" shape this whole section exists to
    remove, in a new dress. Against the contrast the shock-blind reading is the
    zero predictor, whose NRMSE is `sqrt(1 + mean(y)**2 / var(y)) >= 1.0`
    IDENTICALLY, for every game draw. The control cannot be passed by ignoring
    the intervention.

    THE MIRROR ALSO MAKES THE CALIBRATION EXACT. Negating the token sends
    `v -> 1 - v`, which sends the label to MINUS ITSELF, so
    `flipper_dependence` is exactly 2.0 PER EXAMPLE with no sampling spread at
    any n -- the same exact 2.0 the negation-scope task has, reached by
    antisymmetry rather than by a product.
    """
    m, bias, tau = consequence_game(x)
    return _consequence_from_game(m, bias, tau,
                                  torch.sigmoid(x[:, f, CH_FLIP]),
                                  E2_ITERS if k is None else int(k))


def _consequence_from_game(m, bias, tau, v, iters):
    return (_br_sweeps(m, bias, tau, v, iters)[:, E_GAME - 1]
            - _br_sweeps(m, bias, tau, 1.0 - v, iters)[:, E_GAME - 1])


def unshocked_equilibrium(x: torch.Tensor, f: int, p: int) -> torch.Tensor:
    """The label E2 REPLACED, kept as a wrong-task control.

    No player is clamped, so nothing in this reading depends on the intervening
    token and its flipper dependence is exactly 0.0. A band that accepted it
    would be a band that cannot tell a consequence from a state.
    """
    m, bias, tau = consequence_game(x)
    st = torch.full((m.shape[0], E_GAME), 0.5, dtype=m.dtype, device=m.device)
    for _ in range(E2_ITERS):
        st = torch.sigmoid(((m @ st.unsqueeze(-1)).squeeze(-1) + bias)
                           / tau.unsqueeze(-1))
    return st[:, E_GAME - 1]


def consequence_features(x: torch.Tensor, f: int, p: int) -> torch.Tensor:
    """The oracle features for `calibrate_bar`'s trained positive control: the
    game's free entries, the bias, and the dose. 28 numbers, which is the
    COMPLETE determining set with nothing precomputed -- the control is handed
    the oracle's inputs and must solve the fixed point implicitly, never handed
    a partial answer. Handing it the two clamped equilibria would make the
    clause a subtraction and therefore vacuous.
    """
    m, bias, _tau = consequence_game(x)
    iu = torch.triu_indices(E_GAME, E_GAME)
    return torch.cat([m[:, iu[0], iu[1]], bias,
                      torch.sigmoid(x[:, f, CH_FLIP]).unsqueeze(-1)], dim=-1)


def make_consequence_batch(n: int, s: int, d: int, *, d_model: int = 16,
                           seed: int = 0, device=None,
                           settle_iters: int | None = None):
    """(x, y, f, p) for E2. WELL-POSEDNESS IS ENFORCED HERE, NOT AT READ TIME.

    Two things are checked on the drawn batch and BOTH raise rather than return:
    the best-response map's Lipschitz constant must be `E_LIPSCHITZ` for every
    example -- which it is by construction, because `safe_tau` sets the
    temperature to `margin * ||M||_2 / 4` -- and the residual after the sweeps
    must be below `E2_RESIDUAL_TOL`. A batch that did not settle is not a batch
    of equilibria, and a caller that silently accepts one is reading a fixed
    point that was published unconditionally after an iteration that did not
    converge.

    `settle_iters` exists ONLY so that check can be seen to fire: at 0 the
    residual is the full step and the builder raises. The label always uses
    `E2_ITERS`.

    THE INTERVENING TOKEN IS `make_batch`'s FLIPPER, at `s - 1 - d`, so `d` is
    still the retrieval distance between the intervention and the query while
    the game itself sits on tokens 0 .. E_GAME-1.
    """
    x, _y, f, p = make_batch(n, s, d, d_model=d_model, seed=seed, device=device)
    g = torch.Generator(device="cpu").manual_seed(seed + 5150)
    x[:, :E_GAME, CH_GAME:CH_GAME + E_GAME] = torch.randn(
        n, E_GAME, E_GAME, generator=g).to(x.device).to(x.dtype)
    x[:, :E_GAME, CH_GAME_BIAS] = torch.randn(
        n, E_GAME, generator=g).to(x.device).to(x.dtype)
    x[:, :, CH_FLIP] = 0.0
    x[:, f, CH_FLIP] = torch.randn(n, generator=g).to(x.device).to(x.dtype)

    m, bias, tau = consequence_game(x)
    lip = torch.linalg.matrix_norm(m, ord=2) / (4.0 * tau)
    worst = float((lip - E_LIPSCHITZ).abs().max())
    if worst > 1e-6:
        raise ValueError(f"drawn game does not contract at safe_tau's margin: "
                         f"max |L - {E_LIPSCHITZ}| = {worst:.3e}")
    v = torch.sigmoid(x[:, f, CH_FLIP])
    iters = E2_ITERS if settle_iters is None else int(settle_iters)
    st = _br_sweeps(m, bias, tau, v, iters)
    nxt = torch.sigmoid(((m @ st.unsqueeze(-1)).squeeze(-1) + bias)
                        / tau.unsqueeze(-1))
    nxt[:, 0] = v
    resid = float((nxt - st).abs().max())
    if resid > E2_RESIDUAL_TOL:
        raise ValueError(f"consequence batch did not settle: residual "
                         f"{resid:.3e} > {E2_RESIDUAL_TOL} after {iters} sweeps")
    return x, _consequence_from_game(m, bias, tau, v, E2_ITERS), f, p


def consequence_flipper_dependence(s: int) -> float:
    """Exactly 2.0, by antisymmetry, at every `s` and every `n`. See
    `consequence_oracle`."""
    return 2.0


# ==========================================================================
# FOURTH E-FAMILY: E4' -- RIPS CONNECTIVITY ON THE LARGEST-JOIN SUBSTRATE
# (LOOP_PROMPT.md 1.7d as rerouted by CHECKLIST.md RULE 5; STATE.md item 2)
#
# E4 AS SPECIFIED IS STRUCK AND THIS IS THE MEASURED REPLACEMENT. The struck
# task joins the two NEAREST components with the single bridge edge, which on
# S^2 is always a speck against the giant, so component membership collapses
# into "is my own ball small" and a static local-degree decoder reads the
# label at criticality. The reroute joins the two LARGEST instead. Measured,
# on record, and the numbers below are cited rather than re-derived:
#
#     results/e4_gate.txt:47-54  LargestJoin_S2Rips_1024 [REROUTE]
#         merge 30 x 32; gate (a) closes at k=32 (0.0000); gate (b)
#         degree-only 1.0001, +3hop 0.9951, +5hop 0.8220;
#         PLANTED degree-sum 0.0000, PLANTED degree-median 0.5530.
#     results/e4_gate.txt:38-45  LargestJoin_S2Rips_64 [REROUTE] still leaks
#         at radius 5 (0.0055): NOT admissible. THE REROUTE REQUIRES n=1024,
#         so the builder refuses to draw anything smaller.
#     STATE.md item 2: leak closes 0.1565 -> 0.9951 at n=1024.
#
# THE ENCODING. One M3 example is a [s, d_model] float tensor, so token =
# node and the substrate rides in the tokens: three channels carry each
# node's unit-sphere coordinates (the graph definition itself, from which the
# oracle rebuilds the Rips graph -- nothing about the label is stored), one
# channel carries the PER-EXAMPLE do()-bit at the two bridge endpoints
# (+1 = edge present), and two channels mark the queried node pair, drawn
# left x right across the bridge exactly as `draw_do_paired` draws them. The
# label varies because the TENSOR varies: it is carried entirely by the one
# edge the do()-bit adds or removes, which is the definition of the paired
# draw this substrate was admitted on.
#
# CHANNEL OWNERSHIP. Like every family, this one overwrites the channels it
# owns and leaves the shipped noise/dtype/payload machinery alone. Its four
# channels sit above E2's game block, which nothing else touches.
# ==========================================================================
from ceq.rips import (_add_critical_bridge,                     # noqa: E402
                      components as _rips_components,
                      rips_edges as _rips_edges,
                      REROUTED_CASES, sample_sphere as _sample_sphere)
from scale.rips_gate import adjacency as _rg_adjacency          # noqa: E402
from scale.rips_gate import ball as _rg_ball                    # noqa: E402
from scale.rips_gate import local_features as _rg_local_features  # noqa: E402

#: THE SHIPPED ADMISSIBLE SPEC, verbatim from ceq.rips.REROUTED_CASES so no
#: seed or degree expression is copied by hand. LargestJoin_S2Rips_1024.
E4P_SPEC = next(spec for spec in REROUTED_CASES if spec[1] == 1024)
E4P_NODES = E4P_SPEC[1]
E4P_TARGET_DEGREE = E4P_SPEC[2]
E4P_SEED = E4P_SPEC[3]

#: Admissibility floor, measured: at n=64 the same reroute leaks 0.0055 to a
#: radius-5 static decoder (results/e4_gate.txt:44). Below this many nodes the
#: builder raises rather than register an inadmissible task under an
#: admissible name.
E4P_MIN_NODES = 1024

CH_COORD = CH_GAME_BIAS + 1     # ..+2: unit-sphere xyz of each node (shared)
CH_BRIDGE = CH_COORD + 3        # per-example do()-bit at both bridge endpoints
CH_QA = CH_BRIDGE + 1           # one-hot mark of the queried node in `left`
CH_QB = CH_BRIDGE + 2           # one-hot mark of the queried node in `right`


@functools.lru_cache(maxsize=4)
def _e4prime_points(nodes: int, seed: int):
    """The reference point set, cached: the sampler is deterministic in its
    seed and rebuilding it is pure waste."""
    return tuple(tuple(pt) for pt in _sample_sphere(nodes, seed))


@functools.lru_cache(maxsize=4)
def _e4prime_graph(points, target_degree: float):
    """(pre_edges, bridge_edge) for a point set. Cached because the oracle,
    the hop reading and the feature reader all walk the SAME graph and the
    O(s^2) edge rule must not be paid per call."""
    pre_edges = _rips_edges(list(points), target_degree)
    edge = _add_critical_bridge(list(points), pre_edges, len(points),
                                join="largest")
    if edge is None:
        raise ValueError("the drawn substrate has fewer than two components; "
                         "no largest-join bridge exists")
    return tuple(pre_edges), edge


def _e4prime_adjacency(points, target_degree: float, bridged: bool):
    pre_edges, edge = _e4prime_graph(points, target_degree)
    edges = pre_edges + (edge,) if bridged else pre_edges
    return _rg_adjacency(len(points), edges)


def _e4prime_marks(x):
    """Per example: (query_i, query_j, bridged?) read back out of the tensor.
    The markers are the instance; nothing about them is stored elsewhere."""
    n = x.shape[0]
    out = []
    for bidx in range(n):
        qa = (x[bidx, :, CH_QA] != 0).nonzero().flatten()
        qb = (x[bidx, :, CH_QB] != 0).nonzero().flatten()
        br = (x[bidx, :, CH_BRIDGE] != 0).nonzero().flatten()
        if qa.numel() != 1 or qb.numel() != 1 or br.numel() != 2:
            raise ValueError(f"example {bidx}: malformed query/bridge marks")
        bit = float(x[bidx, br[0], CH_BRIDGE])
        if float(x[bidx, br[1], CH_BRIDGE]) != bit:
            raise ValueError(f"example {bidx}: bridge endpoints disagree")
        out.append((int(qa[0]), int(qb[0]), bit > 0))
    return out


def make_e4prime_batch(n: int, s: int, d: int, *, d_model: int = 16,
                       seed: int = E4P_SEED, device=None):
    """(x, y, f, p) for E4'. WELL-POSEDNESS IS ENFORCED HERE, NOT HOPED FOR.

    THE SCALE CLAUSE IS THE ADMISSION. `s` is the node count, and the gates
    were measured on the n=1024 substrate only: at n=64 the same reroute
    leaks 0.0055 to a static radius-5 decoder (results/e4_gate.txt:44). The
    builder raises below `E4P_MIN_NODES`, so the registered task cannot be
    quietly instantiated at a scale where its admissibility was refuted.

    THE SUBSTRATE IS SHARED, THE INTERVENTION IS PER EXAMPLE. All examples of
    a batch carry the same drawn graph in their coordinate channels -- the
    corpus object is the substrate, as for every E-task -- and each example
    carries its own do()-bit at the two bridge endpoints plus its own queried
    left-x-right pair, so the batch is exactly the paired instance set the
    gates were read on (`draw_do_paired`), with the intervention living in
    the tensor where an arm -- and the executable oracle -- can read it.
    Bits are balanced EXACTLY half on / half off before shuffling, the same
    by-construction balance the paired draw has, so no sampling knob decides
    the label's base rate.

    `f` AND `p` ARE THE BRIDGE ENDPOINTS, the pair whose connection the
    intervention IS -- the analogue of `make_equilibrium_batch` returning the
    chain head instead of `make_batch`'s flipper. They are also derivable
    from the tensor: they are the two nodes whose CH_BRIDGE channel is ever
    nonzero. The distractor payload stays at `make_batch`'s payload position,
    an independent draw the label does not depend on, keeping the
    `payload_only` clause live; `d` remains required and range-checked for
    signature parity but positions nothing decisive, because no SINGLE
    position is decisive in this family -- that absence is the property being
    registered (see `e4prime_flipper_dependence`).
    """
    if s < E4P_MIN_NODES:
        raise ValueError(
            f"s={s} nodes is below the measured admissibility floor "
            f"{E4P_MIN_NODES}: LargestJoin_S2Rips_64 leaks 0.0055 at radius 5 "
            f"(results/e4_gate.txt:44); the reroute requires n=1024")
    if d_model < CH_QB + 1:
        raise ValueError(f"d_model={d_model} cannot hold the E4' channels")
    x, _y, _f, p = make_batch(n, s, d, d_model=d_model, seed=seed,
                              device=device)
    points = _e4prime_points(s, seed)
    pre_edges, edge = _e4prime_graph(points, E4P_TARGET_DEGREE)
    a, b = edge
    lab = _rips_components(s, list(pre_edges))
    left = [v for v in range(s) if lab[v] == lab[a]]
    right = [v for v in range(s) if lab[v] == lab[b]]

    pts = torch.tensor(points, dtype=torch.float32).to(x.device)
    x[:, :, CH_COORD:CH_COORD + 3] = pts.to(x.dtype)
    x[:, :, CH_FLIP] = 0.0
    for ch in (CH_BRIDGE, CH_QA, CH_QB):
        x[:, :, ch] = 0.0

    g = torch.Generator(device="cpu").manual_seed(seed + 104729)
    bits = torch.cat([torch.ones(n // 2), -torch.ones(n - n //2)])
    bits = bits[torch.randperm(n, generator=g)]
    x[:, a, CH_BRIDGE] = bits.to(x.device)
    x[:, b, CH_BRIDGE] = bits.to(x.device)
    li = torch.randint(0, len(left), (n,), generator=g)
    ri = torch.randint(0, len(right), (n,), generator=g)
    qi = torch.tensor(left)[li]
    qj = torch.tensor(right)[ri]
    rows = torch.arange(n)
    x[rows, qi, CH_QA] = 1.0
    x[rows, qj, CH_QB] = 1.0

    #: the label comes from the EXECUTABLE ORACLE, never from the draw bookkeeping
    #: above -- one source of truth, same rule as every sibling builder.
    f = a
    return x, e4prime_oracle(x, f, b), f, b


def e4prime_oracle(x: torch.Tensor, f: int, p: int) -> torch.Tensor:
    """+1 if the marked pair shares a component under the example's own
    do()-bit, -1 otherwise. Recomputed ENTIRELY from `x`: the coordinate
    channels rebuild the Rips graph, the bridge-bit channels decide whether
    the single bridge edge is present, the marker channels name the queried
    pair. Nothing is stored and no answer key exists.

    `f` and `p` are accepted for signature parity with every sibling oracle
    and are ignored: the queried pair is PER EXAMPLE here, which is what lets
    one shared substrate carry a varying label.
    """
    points = tuple(tuple(pt) for pt in
                   x[0, :, CH_COORD:CH_COORD + 3].tolist())
    marks = _e4prime_marks(x)
    pre_adj = None
    post_adj = None
    out = torch.empty(x.shape[0], dtype=x.dtype, device=x.device)
    for bidx, (qi, qj, bridged) in enumerate(marks):
        if bridged:
            if post_adj is None:
                post_adj = _e4prime_adjacency(points, E4P_TARGET_DEGREE, True)
            adj = post_adj
        else:
            if pre_adj is None:
                pre_adj = _e4prime_adjacency(points, E4P_TARGET_DEGREE, False)
            adj = pre_adj
        seen = {qi}
        frontier = [qi]
        while frontier and qj not in seen:
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v)
                        nxt.append(v)
            frontier = nxt
        out[bidx] = 1.0 if qj in seen else -1.0
    return out


def e4prime_hop_reading(x: torch.Tensor, f: int, p: int, k: int) -> torch.Tensor:
    """The `k`-hop reachability reading of the label: +1 iff the marked pair
    is connected by a path of length at most `k`. This is exactly what a
    model with a hop budget of `k` can know, and it is the reading gate (a)
    was measured on (`truncation_ladder`). Bounded away from the label at
    every sub-diameter budget and EXACT once the budget covers the pair --
    the property that makes this an equilibrium-family task and not a third
    static one.

    `f` and `p` are accepted for signature parity and ignored, as in
    `e4prime_oracle`.
    """
    points = tuple(tuple(pt) for pt in
                   x[0, :, CH_COORD:CH_COORD + 3].tolist())
    marks = _e4prime_marks(x)
    pre_adj = None
    post_adj = None
    out = torch.empty(x.shape[0], dtype=x.dtype, device=x.device)
    for bidx, (qi, qj, bridged) in enumerate(marks):
        if bridged:
            if post_adj is None:
                post_adj = _e4prime_adjacency(points, E4P_TARGET_DEGREE, True)
            adj = post_adj
        else:
            if pre_adj is None:
                pre_adj = _e4prime_adjacency(points, E4P_TARGET_DEGREE, False)
            adj = pre_adj
        out[bidx] = 1.0 if qj in _rg_ball(adj, qi, k) else -1.0
    return out


def e4prime_features(x: torch.Tensor, f: int, p: int) -> torch.Tensor:
    """Gate (b)'s OWN static local feature matrix for this task: degrees and
    ball sizes out to radius 5 for the marked pair, the strongest strictly-
    local features available, built by `scale.rips_gate.local_features` so
    the registered task is measured with the very decoder that measured the
    substrate. HONEST STATUS, stated rather than buried: on this label these
    features are EXPECTED TO SIT AT THE BAR -- that absence is gate (b)'s
    finding and is asserted in `tests/cameron/test_e4prime_registration.py`
    alongside the planted degree controls that must fire on the same rows.
    The positive control for this family is the planted-degree probe and the
    hop-exactness of the truncation ladder, not a two-feature net; handing
    this control the bridge-bit channel would make clause 5 vacuous by
    construction, which is the defect class this project strikes on sight.
    """
    points = tuple(tuple(pt) for pt in
                   x[0, :, CH_COORD:CH_COORD + 3].tolist())
    marks = _e4prime_marks(x)
    pairs = [((qi, qj), 1 if bridged else 0) for qi, qj, bridged in marks]
    graphs = [_e4prime_adjacency(points, E4P_TARGET_DEGREE, False),
              _e4prime_adjacency(points, E4P_TARGET_DEGREE, True)]
    return torch.from_numpy(_rg_local_features(pairs, graphs, 5)).to(x.dtype)


def e4prime_flipper_dependence(s: int) -> float:
    """Exactly 0.0, by construction, at every admissible `s`.

    CH_FLIP is identically zero in this family: NO single token decides the
    label, because the label is a component-membership fact about a PAIR
    under a graph intervention. That exact zero is not the defect
    `unshocked_equilibrium` was replaced for -- there the zero meant the
    intervention did not reach the label; here it is CHECKLIST RULE 5's whole
    point, the measured globality of the rerouted substrate, and it is why
    the registration supplies the closed form instead of leaving the
    one-sided clause to reject the correct task."""
    return 0.0


# ==========================================================================
# U-LAYER, CONTRACT v10.1 (U1) -- THE RAG-MULTIHOP TWIN REGISTRATION
#
# THE CONTRACT CLAUSE. The e3-harmonic task family registers TWICE in
# `M3_TASKS`: once under its equilibrium-prediction name (`e3_t*`) and once
# as `rag_multihop_t*` under DOCUMENT-GRAPH NAMING. The twin's batch function
# returns EXACTLY the e3 tensors -- same builder call, same draw, bitwise --
# because the rag reading is a NAMING of the same corpus object, not a second
# corpus: positions are CHUNKS, the chain head `f` is the ANCHOR DOCUMENT the
# derivation starts from, each live-band position is a RETRIEVED DOCUMENT
# whose driver `b_i` is its claim and whose coefficient `a_i` gates how the
# claim propagates toward the query, and position `s-1` is the QUERY CHUNK.
# The label being a product of per-hop coefficients along a path of length
# `t*` is what makes the task multihop in the retrieval sense: no single
# document contains the answer.
#
# IDENTICAL TENSORS IS TESTED, NOT STATED: with a fixed seed the two names
# must satisfy torch.equal on x and y (`tests/cameron/test_u1_rag_registration.py`).
#
# THE METADATA CHANNEL. Every channel of the [s, d_model] example is owned by
# convention up to E4' (`CH_QB`). At the shipped d_model=16 exactly one
# channel is unowned, and it carries the rag naming ON REQUEST only: by
# default the batch stays bitwise the e3 batch, so marking can never leak
# into the registered tensors. With `mark_documents=True`, CH_DOC carries
# per-position document ids: -1 for the unretrieved pre-context, 0 for the
# anchor document at the chain head, i - head for the i-th retrieved
# document, and t* for the query chunk. The executable-oracle property is
# untouched -- `equilibrium_oracle` reads only CH_FLIP and CH_DRIVE, so
# relabelling documents cannot move any label, which the registration test
# checks bitwise rather than argues.
# ==========================================================================

#: the one channel above the E4' block. Owned by the rag naming at d_model
#: >= CH_DOC + 1; the builder refuses to mark a narrower tensor.
CH_DOC = CH_QB + 1

#: the sibling rungs mirrored from the e3 ladder (`e3_t*`), same dial values.
RAG_T_SIBLINGS = (1, 2, 8, 32)


def rag_document_ids(s: int, t_star: int) -> torch.Tensor:
    """The document-graph naming of an `s`-position chain instance at rung
    `t_star`, as float ids for the CH_DOC channel: -1 unretrieved context,
    0 the anchor document at the chain head `s-1-t*`, `1..t*-1` the retrieved
    documents, `t*` the query chunk."""
    t = (s - 1) if t_star is None else int(t_star)
    if not (1 <= t <= s - 1):
        raise ValueError(f"t_star={t} does not fit in s={s}")
    head = s - 1 - t
    ids = torch.full((s,), -1.0)
    ids[head] = 0.0
    for i in range(head + 1, s - 1):
        ids[i] = float(i - head)
    ids[s - 1] = float(t)
    return ids


def make_rag_multihop_batch(n: int, s: int, d: int, *, t_star: int | None = None,
                            d_model: int = 16, seed: int = 0, device=None,
                            mark_documents: bool = False):
    """(x, y, f, p), BITWISE the `make_equilibrium_batch` batch, named as
    multihop retrieval over documents.

    Default behaviour is pure forwarding: the returned tuple satisfies
    torch.equal against the e3 sibling's at the same arguments, which is the
    contract clause and is tested. `mark_documents=True` additionally writes
    `rag_document_ids` into CH_DOC (refusing a d_model that cannot hold it);
    channels below CH_DOC and both labels stay bitwise what the e3 builder
    produced, so a marked twin is still the same corpus object with its name
    attached.
    """
    x, y, f, p = make_equilibrium_batch(n, s, d, t_star=t_star,
                                        d_model=d_model, seed=seed,
                                        device=device)
    if mark_documents:
        if d_model < CH_DOC + 1:
            raise ValueError(f"d_model={d_model} cannot hold the rag "
                             f"document-metadata channel CH_DOC={CH_DOC}")
        t = (s - 1) if t_star is None else int(t_star)
        x[:, :, CH_DOC] = rag_document_ids(s, t).to(x.device)
    return x, y, f, p


def e_hop_reading(task: str, x: torch.Tensor, f: int, p: int, k: int):
    """The `k`-budget reading of an E-task's label, for the truncation ladder.

    ONE entry point, because the two families truncate different things and the
    caller must not have to know which: the chain family truncates HOPS (only
    the last `k+1` tokens enter), the consequence family truncates BEST-RESPONSE
    SWEEPS. Both are "what a model with a budget of `k` could compute", which is
    the quantity the ladder is about.
    """
    if task == "e2_consequence":
        return consequence_oracle(x, f, p, k=k)
    return equilibrium_hop_reading(x, k)


def e_ladder_ks(t_star: int) -> list[int]:
    """The rungs the ladder is printed at: powers of two up to the dial, plus
    the dial itself and one past it, so the reading shows both the bound
    tightening and where it stops."""
    ks = {0, 1, 2, 4, 8, 16, 32, t_star, t_star + 1}
    return sorted(k for k in ks if 0 <= k <= t_star + 1)


#: task name -> callable(s) -> the difficulty dial `t*`. S2 asks for it printed
#: per task, and it is a property of the TASK, so it lives beside the
#: registration rather than in whichever caller happens to print it. The chain
#: family's dial is an EXACT hop count; E2's is the sweep count at which the
#: closed-form contraction bound reaches `E2_DIAL_TOL`, which is conservative
#: and labelled so in `E2_T_STAR`.
E_T_STAR = {
    "e1_anchor": lambda s: s - 1,
    "e2_consequence": lambda s: E2_T_STAR,
    "e3_t1": lambda s: 1,
    "e3_t2": lambda s: 2,
    "e3_t8": lambda s: 8,
    "e3_t32": lambda s: 32,
    **{f"rag_multihop_t{t}": (lambda t: lambda s: t)(t) for t in RAG_T_SIBLINGS},
}


def e_t_star(task: str, s: int):
    """The difficulty dial for `task`, or None where the task declares none.

    THE ABSENCE IS THE POINT. `impact`, `impact_hetero` and `e4prime` are
    registered in `M3_TASKS` and carry no derived `t*`: IMPACT's label is a full
    resolvent over a graph whose diameter is a property of the draw, not a hop
    count fixed by the task name. Readers indexed `E_T_STAR` directly and raised
    `KeyError('impact')` on all three. Returning None rather than inventing a
    dial keeps a wrong difficulty number out of the tables -- a wrong dial is
    worse than an absent one, and the reading is pre-registered against the dial.
    """
    fn = E_T_STAR.get(task)
    return None if fn is None else fn(s)


#: name -> (batch_fn, oracle_fn, feature_fn, expected flipper_dependence or None).
#: The registration surface. `None` for the last field means "use the shipped
#: one-sided clause"; a callable means the task supplies its own EXACT value and
#: the clause becomes two-sided.
#:
#: E1 IS CREDITED NOTHING AND IT IS A RIGGED DEMO BY DESIGN. Its label is the
#: signed path sum, which is the object the ceq resolvent computes, so an arm
#: built on that resolvent is being asked to reproduce its own forward. It is
#: here as a MUST-FIRE: an arm that cannot beat NRMSE 1.0 on the task its own
#: construction computes tells us the harness cannot read an equilibrium label
#: at all, and nothing downstream of it may be read (K-5E). No number taken from
#: E1 is a capability claim about any construction.
M3_TASKS = {
    "negation_scope": (make_batch, oracle, None, None),
    "counter_squared": (make_counter_batch, counter_squared_oracle,
                        counter_squared_features,
                        counter_squared_flipper_dependence),
    "e1_anchor": (make_equilibrium_batch, equilibrium_oracle,
                  equilibrium_features, chain_flipper_dependence),
    "e2_consequence": (make_consequence_batch, consequence_oracle,
                       consequence_features, consequence_flipper_dependence),
    **{f"e3_t{t}": (functools.partial(make_equilibrium_batch, t_star=t),
                    equilibrium_oracle, equilibrium_features,
                    functools.partial(chain_flipper_dependence, t_star=t))
       for t in (1, 2, 8, 32)},

    #: E4', the RULE 5 reroute (STATE.md item 2): admissible at n=1024 ONLY,
    #: which `make_e4prime_batch` enforces. Its flipper dependence is the
    #: exact 0.0 of a label no single token decides.
    "e4prime": (make_e4prime_batch, e4prime_oracle, e4prime_features,
                e4prime_flipper_dependence),

     #: U1 (contract v10.1 U-layer): the rag-multihop TWINS of the e3 rungs.
     #: Identical tensors by construction (same builder call); the oracle and
     #: feature slots are the chain family's own objects, not copies, so the
     #: twin cannot drift from its sibling. The naming, not the corpus,
     #: is what is new -- see the U-layer section above.
     **{f"rag_multihop_t{t}": (functools.partial(make_rag_multihop_batch, t_star=t),
                               equilibrium_oracle, equilibrium_features,
                               functools.partial(chain_flipper_dependence, t_star=t))
        for t in RAG_T_SIBLINGS},

     #: T-FAMILY 2 IMPACT — planted news→asset propagation [U4a].
     #: Graph: MP-cleaned [V] GARCH-noised [V] symbolic, no real-market data.
     #: Label r = (I - rho A)^{-1} B n, oracle exact via K = (I - rho A)^{-1}.
     #: Tensor batch [n,s,d_model] bitwise deterministic, E4′-style admission s>=1024.
     #: Gates: decoder FAIL at cross-asset rows, truncation at shipped hops, sign-scrambled degrade, covariates per instance.
     #: Attribution: recovered B_hat vs planted B, rank correlation with CI, N3 probe checked against exact kernel.
     #: Heterogeneous plant for X21 via impact_hetero (two-block degree heterogeneity).
     "impact": (__import__("scale.impact", fromlist=["make_impact_batch"]).make_impact_batch,
                __import__("scale.impact", fromlist=["impact_oracle"]).impact_oracle,
                __import__("scale.impact", fromlist=["impact_features"]).impact_features,
                __import__("scale.impact", fromlist=["impact_flipper_dependence"]).impact_flipper_dependence),
     #: A1 REPAIR. This entry bound `make_impact_batch` -- the SAME builder as
     #: `impact` -- so both keys returned a byte-identical 4-tuple and a
     #: heterogeneous-plant arm drew the homogeneous corpus. It was its own
     #: baseline by construction (MISTAKES.md V-1, and the LIVE one -- the
     #: struck count is fourteen, `STATE.md:53`, and two separate later items
     #: were each already labelled "the fifteenth", so no number is claimed
     #: here). The oracle and feature hooks are shared with
     #: `impact` on purpose: they rebuild the plant from the tensor's CH_HET
     #: bit rather than from the registry, so no entry can pair a builder with
     #: the wrong oracle. Bound by
     #: `tests/cameron/test_impact_hetero_is_not_its_own_baseline.py`.
     "impact_hetero": (__import__("scale.impact", fromlist=["make_impact_hetero_batch"]).make_impact_hetero_batch,
                       __import__("scale.impact", fromlist=["impact_oracle"]).impact_oracle,
                       __import__("scale.impact", fromlist=["impact_features"]).impact_features,
                       __import__("scale.impact", fromlist=["impact_flipper_dependence"]).impact_flipper_dependence),
}


def nrmse(pred: torch.Tensor, y: torch.Tensor) -> float:
    """RMSE normalised by the std of y. Exactly 1.0 for the mean predictor."""
    sd = float(y.std(unbiased=False))
    if sd == 0.0:
        return float("nan")
    return float(((pred - y) ** 2).mean().sqrt()) / sd


def bootstrap_ci(pred, y, *, n_boot: int = 400, seed: int = 0, alpha=0.05):
    g = torch.Generator().manual_seed(seed)
    n = y.shape[0]
    vals = []
    for _ in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=g)
        vals.append(nrmse(pred[idx], y[idx]))
    vals = sorted(v for v in vals if v == v)
    if not vals:
        return float("nan"), float("nan")
    lo = vals[int(alpha / 2 * len(vals))]
    hi = vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))]
    return lo, hi


# ==========================================================================
# CALIBRATION OF THE BAR -- run before any arm is credited with beating it
# ==========================================================================

def calibrate_bar(n: int = 2048, s: int = 512, d: int = 256, *,
                  oracle_fn=None, batch_fn=None, feature_fn=None,
                  steps: int = 150, lr: float = 0.02,
                  seed: int = 0, standardise: bool = True) -> dict:
    """Five reference points pin the scale AND prove the task and budget are real.

    THE THREE ORIGINAL CHECKS WERE NOT ALL MEASUREMENTS. Two were algebraic
    identities:

        predict_the_mean = nrmse(y.mean(), y)          == 1.0 by definition of nrmse
        oracle           = nrmse(oracle(x,f,p), y)     == 0.0 because `make_batch`
                                                          RETURNS oracle(x,f,p) as y

    so `nrmse(t, t)` is being compared to zero. Only `payload_only` read the task
    at all, and it only requires the label to differ from the payload. A label
    with NO dependence on the flipper -- the entire premise of this task removed --
    passed the gate and printed BAR CALIBRATED. That is this project's
    "zero BY CONSTRUCTION mapped to GREEN" defect, sitting in the gate that
    decides whether any arm is credited.

    Two checks are added, and both compare VALUES with known answers at both ends.

    4. FLIPPER DEPENDENCE. Negate the flipper and require the label to move.
       For `y = payload * sign` the label negates, so the relative movement is
       exactly 2.0; for a flipper-blind label it is exactly 0.0. This is the
       check that refuses a task which is not this task.

    5. TRAINED POSITIVE CONTROL. A small model, trained at the harness's own
       budget, given ONLY the two oracle features. It must beat the bar. Without
       it "this arm failed" and "this harness cannot produce a pass" are the same
       printout -- and in this repository no arm has ever passed, while the
       `oracle` entry is an identity rather than a trained model.

       IT TRAINS ON THE LABEL THE ARMS TRAIN ON, which it did not always do.
       `run_arm` standardises `y` and un-standardises its prediction before
       scoring; this clause optimised against the raw label and scored there.
       Both were compared to the same bar of 1.0, so on a label whose scale is
       far from the initialisation's the control was solving a strictly harder
       problem than the arms and its failure was read as the task's. See the
       note above the training loop.
    """
    ofn = oracle_fn or oracle
    #: A SECOND task needs a second tensor builder, not just a second label:
    #: `make_batch` writes CH_FLIP at ONE position, so `counter_squared` over
    #: that tensor is identically 1.0 and every clause below reads NaN. The hook
    #: defaults to the shipped builder, so the shipped path is unchanged.
    x, y, f, p = (batch_fn or make_batch)(n, s, d, seed=seed)
    if oracle_fn is not None:
        y = ofn(x, f, p)
    out = {}

    # 1. predict-the-mean. Identically 1.0 -- kept because it DEFINES the bar,
    #    and labelled so nobody reads it as evidence about the task.
    out["predict_the_mean"] = nrmse(y.mean().expand_as(y), y)

    # 2. flipper-blind: sees the payload, cannot see the sign.
    out["payload_only"] = nrmse(x[:, p, CH_PAYLOAD], y)

    # 3. the oracle, recomputed. Identically 0.0 when `y` came from `ofn`.
    out["oracle"] = nrmse(ofn(x, f, p), y)

    # 4. FLIPPER DEPENDENCE -- the check that a flipper-blind task cannot pass.
    xf = x.clone()
    xf[:, f, CH_FLIP] = -xf[:, f, CH_FLIP]
    moved = float((ofn(xf, f, p) - y).abs().mean())
    scale = float(y.abs().mean())
    out["flipper_dependence"] = moved / scale if scale > 0 else float("nan")

    # 5. TRAINED POSITIVE CONTROL -- model-level, at the harness's own budget.
    g = torch.Generator().manual_seed(seed)
    #: WHICH features are the oracle features is a property of the TASK. For
    #: `counter_squared` the sufficient statistic is the sum of every letter, and
    #: handing this control (flipper, payload) would make it fail for the right
    #: reason on the wrong features -- reading "no arm can pass" when what was
    #: measured is "these two numbers do not determine the label".
    feats = (feature_fn(x, f, p) if feature_fn is not None else
             torch.stack([x[:, f, CH_FLIP], x[:, p, CH_PAYLOAD]], dim=-1))
    net = torch.nn.Sequential(torch.nn.Linear(feats.shape[-1], 32),
                              torch.nn.GELU(), torch.nn.Linear(32, 1))
    for layer in net:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.normal_(layer.weight, 0.0, 0.5, generator=g)
            torch.nn.init.zeros_(layer.bias)
    #: THE CONTROL TRAINS ON THE LABEL THE ARMS TRAIN ON. `run_arm`
    #: (`scale/m3_capability.py:196`) optimises against `(y - mu) / sigma` and
    #: un-standardises before it scores (`:187`); this clause used to optimise
    #: against RAW `y` and score there too. Both readings were then compared to
    #: the same bar of 1.0, so the control was solving a harder version of the
    #: arms' problem whenever the label's scale was far from the initialisation's.
    #: Adam's step is bounded by `lr` almost regardless of the gradient, so the
    #: distance from `normal_(w, 0, 0.5)` to the label's scale is spent out of
    #: the step budget: at `e2_consequence`'s sd 0.061984 the control read
    #: `trained_two_feature = 2.446645` at the shipped 150 steps and the bar
    #: printed BROKEN -- which reads as "no arm can pass this task" while what
    #: was measured is "the control was handed the label in the wrong units".
    #: `e2_consequence` has never been trained (FINDINGS A7, STATE.md:73-76).
    #:
    #: `standardise=False` runs the pre-repair path unchanged. It exists so the
    #: strike record stays MEASURABLE in-process rather than only describable in
    #: a document -- see
    #: `tests/cameron/test_bar_control_sees_the_arms_preprocessing.py` and
    #: `tests/cameron/test_m3_etasks.py::test_the_consequence_bar_is_broken_at_the_shipped_step_budget`.
    #: Nothing in the shipped path passes it.
    mu = float(y.mean()) if standardise else 0.0
    sigma = float(y.std(unbiased=False)) if standardise else 1.0
    if sigma == 0.0:
        sigma = 1.0
    target = (y - mu) / sigma
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((net(feats).squeeze(-1) - target) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        #: Scored on the RAW label, exactly as `run_arm`'s `raw_pred` scores.
        #: NRMSE is scale-free, so this makes the clause read the same number
        #: for `y` and for `c*y` -- the property the arms' readings already had.
        out["trained_two_feature"] = nrmse(net(feats).squeeze(-1) * sigma + mu, y)
    return out


#: The gate, in ONE place. `m3_capability.py` used to hold a private copy of the
#: pass condition; round 2 shipped a verdict whose tested copy was correct while
#: the copy that executed was not, and two copies of one rule is that defect
#: waiting to happen.
def bar_verdict(cal: dict, *, flipper_dependence: float | None = None,
                flipper_tol: float = 0.05) -> tuple[bool, str]:
    """(ok, reason). Every clause must be EVALUABLE and must be able to fail.

    `flipper_dependence` is the EXACT value the task predicts in closed form.
    When it is None the shipped one-sided clause (`> 0.5`) runs unchanged; when
    it is supplied the clause becomes TWO-SIDED, which is strictly stronger --
    a one-sided threshold accepts every label that moves more than enough, while
    the band accepts only labels that move by the predicted amount.

    THE BAND HAS A MINIMUM SAMPLE SIZE, MEASURED. `flipper_dependence` is a
    ratio of two sample means, so the band is only a gate if it is wider than
    the sampling spread and narrower than the distance to a wrong task. Measured
    on this machine at s=64, 12 seeds each, `counter_squared`:

        n= 256   sd 0.023352   max|dev from exact| 0.083739
        n= 512   sd 0.010428   max|dev from exact| 0.029482
        n=2048   sd 0.005960   max|dev from exact| 0.010315
        n=4096   sd 0.003289   max|dev from exact| 0.005910

    The default tolerance 0.05 therefore REQUIRES n >= 512: at n=256 the spread
    alone exceeds it and the gate would reject the correct task. The distance to
    the nearest wrong task -- the plain counter, at 0.31455481755627596 against
    0.39738701499186757 -- is 0.08283, so 0.05 still rejects it at every n above
    the floor. Both ends are measured; neither was chosen to fit.

    THE ONE-SIDED THRESHOLD IS TASK-SPECIFIC AND WAS NOT WEAKENED. 0.5 is
    calibrated for `y = payload * sign`, where negating the flipper negates the
    label and the ratio is exactly 2.0. `counter_squared` is a global aggregate:
    its exact ratio is `4 * E|S_{s-1}| / s`, which reads 0.39738701499186757 at
    s=64 and would be REJECTED by the shipped clause. Replacing it with a looser
    one-sided threshold would have made the clause unable to reject the plain
    counter (0.31455481755627596 at s=64), whose Hankel has no gap at all. The
    band rejects it; `tests/cameron/test_m3_counter_squared.py` runs that
    rejection.
    """
    if abs(cal["predict_the_mean"] - 1.0) > 1e-6:
        return False, f"predict_the_mean={cal['predict_the_mean']:.6f}, not 1.0 -- nrmse is mis-implemented"
    if not (cal["payload_only"] >= 1.0):
        return False, f"payload_only={cal['payload_only']:.6f} BEATS the bar -- the label is the payload"
    if not (cal["oracle"] < 1e-6):
        return False, f"oracle={cal['oracle']:.6f}, not ~0 -- the task is unsolvable"
    if flipper_dependence is None:
        if not (cal["flipper_dependence"] > 0.5):
            return False, (f"flipper_dependence={cal['flipper_dependence']:.6f} -- the label barely "
                           f"moves when the flipper is negated, so this is NOT the negation-scope task")
    elif abs(cal["flipper_dependence"] - flipper_dependence) > flipper_tol:
        return False, (f"flipper_dependence={cal['flipper_dependence']:.6f} is not the value this "
                       f"task predicts in closed form, {flipper_dependence:.6f} "
                       f"(tol {flipper_tol}) -- this is NOT the task")
    if not (cal["trained_two_feature"] < 1.0):
        return False, (f"trained_two_feature={cal['trained_two_feature']:.6f} >= 1.0 -- a model "
                       f"given the ORACLE FEATURES cannot beat the bar at this budget, so no arm "
                       f"can, and every arm reading is uninterpretable")
    return True, "BAR CALIBRATED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--s", type=int, default=512)
    ap.add_argument("--distances", nargs="+", type=int, default=[256])
    a = ap.parse_args()

    print("M3 negation-scope. ORACLE IS EXECUTABLE; no answer key on disk.")
    print("ABSOLUTE BAR: NRMSE = 1.0 is predict-the-mean. Every W4 arm died "
          "ABOVE it while still producing a pretty ordering.\n")
    print("=== BAR CALIBRATION (RED-first: the bar must be seen to fire) ===")
    ok = True
    for d in a.distances:
        c = calibrate_bar(n=a.n, s=a.s, d=d)
        print(f"\n  s={a.s} d={d}  (flipper at {a.s-1-d}, payload at {a.s-2}, "
              f"query at {a.s-1})")
        for name, v in c.items():
            print(f"    {name:>18} NRMSE {v:.6f}")
        checks = [
            ("predict_the_mean is exactly 1.0", abs(c["predict_the_mean"] - 1.0) < 1e-6),
            ("payload_only FAILS the bar (>= 1.0)", c["payload_only"] >= 1.0),
            ("oracle passes (~0)", c["oracle"] < 1e-6),
        ]
        for label, passed in checks:
            print(f"    [{'OK ' if passed else 'BAD'}] {label}")
            ok &= passed
    print(f"\n  BAR {'CALIBRATED' if ok else 'BROKEN -- do not credit any arm'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

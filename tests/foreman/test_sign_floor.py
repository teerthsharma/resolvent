"""The sign floor: a certified threshold below which `sgate` cannot be signed.

WHAT THIS FILE DECIDES. The S2 chain spends +3 on "find a Hankel task with
`rank_+ > rank`, put it in M3, show the signed arm beats the non-negative arm."
The final step compares `pivot_signed` against `pivot_unsigned`. That comparison
is sign evidence only if `pivot_signed`'s operator actually carries a negative
entry at the geometry it is scored on. F16 recorded one measurement saying it
does not (min entry exactly `0.000e+00`). One measurement at one seed is not a
property of the operator; this file turns it into one.

THE IDENTITY THE INSTRUMENT IS BUILT ON. With

    A = rho * (softmax(w) - lam * softmax(-w)) / (1 + lam),   rho > 0, lam > 0

an entry on the causal support is negative exactly when

    softmax(w)_ij < lam * softmax(-w)_ij

    <=>  log softmax(w)_ij - log softmax(-w)_ij  <  log lam
    <=>  2 w_ij + logsumexp_l(-w_il) - logsumexp_l(w_il)  <  log lam

so the whole sign question for a row is the single scalar

    R_i = min_j [ 2 w_ij + logsumexp(-w_i.) - logsumexp(w_i.) ]      (log ratio)

and the row is entrywise non-negative iff `R_i >= log lam`. No division and no
multiplication enters the decision, which is what G8 asks of a sign test.

THE CERTIFICATE. Let `L_i` be the range `max_j w_ij - min_j w_ij` over the row's
support. Bounding each logsumexp by its extreme term gives

    R_i >= -2 L_i        so       L_i <= (1/2) log(1/lam)  =>  row i is >= 0.

`L_star = (1/2) log(1/lam)` is `1.151292546497023` at the shipped `lam = 0.10`.
This is a *sufficient* condition on the logit geometry alone: no seed, no draw,
no batch at a smaller range can produce a negative entry. Test 2 shows it is a
bound and not a tautology by exhibiting drawn instances above `L_star` that are
still non-negative.

RED FIRST. `scale/foreman_signfloor.py` does not exist when this file is
written, so every test here fails at import. Tests 1, 2 and 4 turn green when
the instrument lands. Test 3 is a KILL and stays red against the code as it
stands; it is the finding, not a defect to be patched away.
"""
from __future__ import annotations

import math
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scale.foreman_signfloor import (  # noqa: E402
    ARM_GEOMETRIES, harness_operator, sign_margin,
)

torch.set_num_threads(2)

LAM, RHO = 0.10, 1.5
#: (1/2) * log(1/lam) at lam = 0.10. Written out so a change to LAM that
#: forgets this constant fails loudly instead of quietly relaxing the bound.
L_STAR = 1.151292546497023

#: Logit gains swept for the must-fire. Geometric, DRAWN, never hand-built: the
#: operator is evaluated on `randn` q/k at each gain and every point in the grid
#: is reported, so no instance is chosen for its answer. `q` and `k` are both
#: scaled, so the logits scale as `gain**2` and the grid is finer than it looks
#: -- deliberately dense between 0.3 and 1.0, because that is where the row range
#: crosses `L_STAR` and a grid that steps over the crossing cannot show that the
#: certificate is a bound rather than an iff.
GAINS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85,
         1.0, 2.0, 5.0, 10.0, 30.0)
SEEDS = (0, 1, 2)
S_SWEEP, D_SWEEP = 32, 16


def _drawn(gain: float, seed: int):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(S_SWEEP, D_SWEEP, generator=g) * gain
    k = torch.randn(S_SWEEP, D_SWEEP, generator=g) * gain
    return q, k


def _sweep():
    for gain in GAINS:
        for seed in SEEDS:
            q, k = _drawn(gain, seed)
            yield gain, seed, sign_margin(q, k, rho=RHO, lam=LAM)


# ====================================================================== #
# 1. THE CRITERION IS EXACT, AND IT IS SEEN FIRING BOTH WAYS.
# ====================================================================== #
def test_the_log_ratio_criterion_matches_the_operator_entrywise():
    seen_neg = seen_pos = 0
    for gain, seed, m in _sweep():
        predicted = m["min_log_ratio"] < math.log(LAM)
        actual = m["min_entry"] < 0.0
        assert predicted == actual, (
            f"gain={gain} seed={seed}: criterion says negative={predicted} but "
            f"the operator's min entry is {m['min_entry']:.6e}. "
            f"min_log_ratio={m['min_log_ratio']:.6f} log(lam)={math.log(LAM):.6f}")
        seen_neg += bool(actual)
        seen_pos += bool(not actual)
    assert seen_neg >= 1, "no drawn instance was ever signed: the test cannot fail"
    assert seen_pos >= 1, "no drawn instance was ever non-negative: ditto"


# ====================================================================== #
# 2. THE RANGE CERTIFICATE IS SOUND AND NOT A TAUTOLOGY.
# ====================================================================== #
def test_the_range_certificate_is_sound_and_not_vacuous():
    below = [(g, s, m) for g, s, m in _sweep() if m["max_range"] <= L_STAR]
    above_but_positive = [(g, s, m) for g, s, m in _sweep()
                          if m["max_range"] > L_STAR and m["min_entry"] >= 0.0]
    assert below, "no drawn instance landed under L_STAR: the bound is untested"
    for g, s, m in below:
        assert m["min_entry"] >= 0.0, (
            f"CERTIFICATE VIOLATED at gain={g} seed={s}: max_range="
            f"{m['max_range']:.6f} <= L_STAR={L_STAR} yet min entry is "
            f"{m['min_entry']:.6e}")
    assert above_but_positive, (
        "every instance above L_STAR was signed, so the certificate is an "
        "iff and this file is asserting a tautology rather than a bound")


# ====================================================================== #
# 3. THE KILL. Pre-registered, RED against the code as it stands.
# ====================================================================== #
@pytest.mark.parametrize("arm", sorted(ARM_GEOMETRIES))
def test_the_arm_operator_carries_a_negative_entry_at_its_own_geometry(arm):
    m = harness_operator(arm)
    assert m["min_entry"] < 0.0, (
        f"KILL: arm `{arm}` has NO negative entry at the geometry it is scored "
        f"on. min_entry={m['min_entry']:.6e}, max|w|={m['max_abs_logit']:.6e}, "
        f"max_range={m['max_range']:.6e} against L_STAR={L_STAR} "
        f"(certified safe below it), min_log_ratio={m['min_log_ratio']:.6f} "
        f"against log(lam)={m['log_lam']:.6f}. "
        f"A signed-vs-unsigned comparison on this arm is not sign evidence.")


# ====================================================================== #
# 4. THE ROUTE EXISTS AND ITS PRICE IS A NUMBER. Rule 5, measured.
# ====================================================================== #
#: The arms whose operator is `sgate` and therefore has a `lam` to move.
#: Selected on the OPERATOR, not on the name: `"signed" in "pivot_unsigned"` is
#: True, which would have put a softmax arm in this list.
SGATE_ARMS = sorted(a for a, (kind, _lam, _w) in ARM_GEOMETRIES.items()
                    if kind in ("pivot_signed", "windowed_signed"))


@pytest.mark.parametrize("arm", SGATE_ARMS)
def test_a_lambda_that_restores_sign_exists_below_the_dc_ceiling(arm):
    """`lam > exp(min_log_ratio)` buys a negative entry; `lam < 1` keeps a
    nonzero row sum. The window is non-empty because the two softmaxes both sum
    to 1, so unless they are equal everywhere some entry has ratio < 1."""
    m = harness_operator(arm)
    lam_needed = math.exp(m["min_log_ratio"])
    assert lam_needed < 1.0, (
        f"arm `{arm}`: the smallest lam that puts a negative entry in the "
        f"operator is {lam_needed:.6f}, which is not below the lam=1 ceiling "
        f"where every row sums to exactly zero. No lam gives sign and DC "
        f"together at this geometry.")


# ====================================================================== #
# 5. THE INIT READING IS NOT THE TRAINED READING.
# ====================================================================== #
#
# Every sign measurement on record in this project -- `CHECKLIST.md:333`'s
# G4 VOID included -- reads the operator at RANDOM INIT. `wq` and `wk` are the
# only parameters that move the logit geometry and Adam at `m3.LR` moves them.
# Test 3 above is therefore a statement about initialisation and nothing else,
# and this test is the other half: after the shipped training loop the same arm
# must be checked again, because a `lam` that is a threshold at initialisation
# is not necessarily a threshold at the weights the arm is scored with.


@pytest.mark.slow
@pytest.mark.parametrize("arm", SGATE_ARMS)
def test_the_trained_sgate_arm_is_checked_for_sign_at_its_trained_geometry(arm):
    """GREEN or RED, either is a result -- what is NOT admissible is reading the
    init geometry and calling it the arm's sign structure."""
    from scale.foreman_signfloor import trained_report

    kind = ARM_GEOMETRIES[arm][0]
    r = trained_report(kind, steps=150, n_train=2048)
    assert r["min_entry"] < 0.0, (
        f"arm `{arm}` is STILL entrywise non-negative after 150 steps at lr="
        f"{0.02}: min_entry={r['min_entry']:.6e}, max_range={r['max_range']:.6f} "
        f"against L_STAR={L_STAR}, eval_nrmse={r['eval_nrmse']:.6f}. The sign "
        f"axis is empty for this arm's whole life, not only at init.")


# ====================================================================== #
# 6. THE VALUE PATH: can a THIRD token move the sign of an influence?
# ====================================================================== #
#
# The S2 chain reads: `rank_+ > rank` on the task => a NON-NEGATIVE weighted
# automaton needs more states => the non-negative ARM loses at matched
# parameters. The middle step is a statement about an automaton with a LINEAR
# value path -- the hypothesis `ceq/bench.py:118-122` leans on when it says
# `W_v` and `W_o` "are constants and cannot" depend on a third token.
#
# These three tests read the property the theorem is actually about, on the M3
# arm, at the geometry the arm is scored at. The control is the SAME arm with
# the GELU replaced by an identity: one object changed, nothing else.
#
# `scale/s2_units.py` records the reading this reproduces: "pivot_unsigned
# wrt=v 0.000000 (0/512) <- structurally pinned".

#: The shipped geometry, and one two decades of logit scale above it. Both are
#: measured; neither is chosen after seeing an answer.
SHIPPED_GAIN, OPEN_GAIN = 1.0, 10.0


def test_a_linear_value_path_under_a_nonnegative_operator_cannot_flip():
    """THE CONTROL, and it must be seen reading exactly zero.

    With `M = I + A + hop2` non-negative and the readout linear, the gradient is
    `M[s-1,j]` times a constant vector, so no third token can move its sign, and
    the influence itself cannot be negative. A nonzero reading here means the
    probe is broken and nothing else in this section is admissible."""
    from scale.foreman_signfloor import value_path_flip_rate

    for gain in (SHIPPED_GAIN, OPEN_GAIN):
        r = value_path_flip_rate("pivot_unsigned", nonlinear=False, gain=gain)
        assert r["used"] >= 64, (
            f"gain={gain}: only {r['used']}/{r['n']} draws cleared the floor "
            f"{r['floor']}; the probe has no opportunity and proves nothing")
        assert r["rate"] == 0.0 and r["neg_influence_frac"] == 0.0, (
            f"PROBE BROKEN at gain={gain}: a non-negative operator with a "
            f"LINEAR value path read flip {r['rate']:.6f} / negative-influence "
            f"{r['neg_influence_frac']:.6f}. The structural zero "
            f"`scale/s2_units.py` records cannot be reproduced.")


def test_the_value_path_probe_is_seen_reading_a_nonzero_sign_channel():
    """MUST-FIRE. A probe that reads zero everywhere has not measured anything.
    Raising the input gain raises the logit range past `L_STAR`, the operator
    acquires negative entries, and the signed arm's influence must follow."""
    from scale.foreman_signfloor import value_path_flip_rate

    r = value_path_flip_rate("pivot_signed", nonlinear=True, gain=OPEN_GAIN)
    assert r["neg_influence_frac"] > 0.0, (
        f"at gain={OPEN_GAIN} the signed arm's influence at j={r['j']} is never "
        f"negative over {r['used']} live draws (min {r['min_influence']:.4e}). "
        f"This probe has never been seen firing, so every zero it reports is "
        f"uninformative.")


def test_the_two_arms_separate_on_the_sign_channel_at_the_shipped_geometry():
    """THE KILL, pre-registered. The ladder's whole discriminating axis is that
    the signed arm can express a content-conditional sign and the non-negative
    arm cannot. At the geometry the arms are scored at, that axis must have
    nonzero width."""
    from scale.foreman_signfloor import value_path_flip_rate

    sig = value_path_flip_rate("pivot_signed", nonlinear=True, gain=SHIPPED_GAIN)
    uns = value_path_flip_rate("pivot_unsigned", nonlinear=True, gain=SHIPPED_GAIN)
    assert sig["neg_influence_frac"] > uns["neg_influence_frac"], (
        f"KILL: at the shipped geometry the signed arm reads negative-influence "
        f"{sig['neg_influence_frac']:.6f} (flip {sig['rate']:.6f}) and the "
        f"non-negative arm reads {uns['neg_influence_frac']:.6f} (flip "
        f"{uns['rate']:.6f}) over {sig['used']} and {uns['used']} live draws. "
        f"The two arms read the SAME value on the axis the ladder is supposed "
        f"to discriminate on, so a win by either is not attributable to sign.")


# ====================================================================== #
# 7. THE JOINT NOBODY NAMED, at the weights the arms are scored with.
# ====================================================================== #
#
# `rank_+(H_f) > rank(H_f)` bounds the state count of a NONNEGATIVE WEIGHTED
# AUTOMATON. That object has a LINEAR value path -- it is the hypothesis
# `ceq/bench.py:118-122` leans on when it says `W_v` and `W_o` "are constants
# and cannot" depend on a third token. `scale/m3_capability.Arm` reads
# `readout(MLP(z))` with `MLP = Linear -> GELU -> Linear`, so the readout
# Jacobian at position `s-1` is a function of `z[s-1]`, which is a function of
# the whole prefix. Whether that is enough to give a non-negative operator a
# content-conditional sign is a MEASUREMENT, and this is it.
#
# The control is the SAME TRAINED WEIGHTS with the GELU replaced by an identity.
# One object changed. It must read the structural zero on the non-negative arm
# AND a nonzero value on the signed arm, or it is not a control.


@pytest.mark.slow
def test_the_trained_nonnegative_arm_cannot_express_a_conditional_sign():
    """THE KILL. If the trained NON-NEGATIVE arm can put a negative sign on a
    token's influence, then "non-negative operator" does not imply "cannot
    express content-conditional negation" for this architecture, and there is no
    joint through which a Hankel `rank_+` gap reaches this arm's capability."""
    import copy
    from scale.foreman_signfloor import train_arm, value_path_flip_rate

    uns, _mu, _sd = train_arm("pivot_unsigned", steps=150, n_train=2048)
    sig, _mu2, _sd2 = train_arm("pivot_signed", steps=150, n_train=2048)

    lin_u = value_path_flip_rate("pivot_unsigned", nonlinear=False,
                                 arm=copy.deepcopy(uns))
    lin_s = value_path_flip_rate("pivot_signed", nonlinear=False,
                                 arm=copy.deepcopy(sig))
    gel_u = value_path_flip_rate("pivot_unsigned", nonlinear=True,
                                 arm=copy.deepcopy(uns))

    # --- the control, both directions, on the SAME trained weights ---
    assert lin_u["used"] >= 64 and lin_s["used"] >= 64, (
        f"too few live draws ({lin_u['used']}, {lin_s['used']}); no opportunity")
    assert lin_u["neg_influence_frac"] == 0.0 and lin_u["rate"] == 0.0, (
        f"CONTROL BROKEN: trained non-negative operator with a LINEAR value "
        f"path read negative-influence {lin_u['neg_influence_frac']:.6f} / flip "
        f"{lin_u['rate']:.6f}; the structural zero cannot be reproduced and "
        f"nothing below is admissible.")
    assert lin_s["neg_influence_frac"] > 0.0, (
        f"CONTROL VACUOUS: the linear-readout probe reads "
        f"{lin_s['neg_influence_frac']:.6f} on the SIGNED arm too, so its zero "
        f"on the non-negative arm is a property of the probe, not of the arm.")

    # --- the kill ---
    assert gel_u["neg_influence_frac"] == 0.0 and gel_u["rate"] == 0.0, (
        f"KILL: the trained SHIPPED `pivot_unsigned` arm -- entrywise "
        f"non-negative operator, min entry >= 0 at every step -- carries a "
        f"NEGATIVE influence on {gel_u['neg_influence_frac']:.6f} of drawn "
        f"third-token interventions and FLIPS that sign on "
        f"{gel_u['k']}/{gel_u['used']} = {gel_u['rate']:.6f}, against "
        f"{lin_u['neg_influence_frac']:.6f} / {lin_u['rate']:.6f} for the "
        f"IDENTICAL trained weights with the GELU replaced by an identity. "
        f"The only difference between the two readings is the nonlinearity, so "
        f"this arm is not a non-negative weighted automaton and "
        f"`rank_+(H_f) > rank(H_f)` bounds nothing about it.")

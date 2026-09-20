"""tests/beds/test_target_pin.py -- the pinned beta=1 JEPA target, in a file
this lane wrote AGAINST the round's brief rather than against pi_jepa.py's
source. ceqjepa/pi_jepa.py is owned by another lane this round; this file
edits nothing in it and adds no hook to it.

THE BRIEF THIS FILE WAS HANDED, VERBATIM, BECAUSE IT IS THE SPEC AND NOT A
SUMMARY OF ONE.

  THE MOVE: replace the EMA target with a parameter-free read at the
  operator's beta=1 corner.

  WHY: the JEPA target is not independent of its own encoder, and the drift
  is measured -- target RMS moved 4.1101x on the synthetic bed and 136.7950x
  on the Kaggle run as the encoder trained. Self-attention gets a still
  target free because tokens are given. beta=1 IS softmax, bitwise 0.000e+00
  in float64, so a target read at that corner is a deterministic function of
  the tokens with no trained parameter in it, and the drift becomes 1.0000 BY
  CONSTRUCTION rather than by tuning a decay rate.

  THE CONSTRUCTION CHECK, AND IT IS NOT A RED -- NAME IT SO IN THE TEST:
  target RMS ratio 1.0000 to float64. Once the EMA is gone the target is a
  parameter-free function of the tokens, so measuring that it does not move
  is measuring that you removed the thing that moved it. It cannot fail. It
  goes in as a wiring check with "construction_check" in its name, never as
  evidence.

  THE RED THAT CAN FAIL: a frozen target is the standard collapse failure
  mode EMA exists to prevent, so the real claim is that pinning removes
  drift WITHOUT causing collapse. The repo has already convicted the
  one-leg version of this check: a rank-one plant keeps std_min 0.2781 and
  is missed by the variance leg entirely, and only the effective rank
  catches it (std_min 0.0000e+00 planted against 2.2361e-01 healthy). So the
  RED is BOTH legs -- std_min > 0.001 AND effective rank not falling to the
  rank-one plant's level.

  KILL: std_min < 0.001; or effective rank falls to the rank-one plant's
  measured level; or the quantity does not move under training on the
  identical held-out tokens (see ROUND TWO below).

ROUND TWO, BECAUSE ROUND ONE'S RED COULD NOT FAIL. Two of the original four
tests here were construction checks wearing a RED's name. One read
collapse_report on the PINNED target itself (target_repr) -- which is one
constant tensor for the whole run by construction (measured: 1 unique
std_min and 1 unique erank across a 120-step trace, and again across 480),
so a threshold on it is a threshold on the bed draw, not on anything trained.
The other asserted trained_std >= frozen_std where both sides were the
SAME bitwise tensor (torch.equal == True, max|delta| == 0.0) -- x >= x. Both
are deleted or rewritten below. The RED that can actually fail lives on the
ONLINE representation, the only thing that trains in the pinned arm, which is
also what pi_jepa.py's fit() now runs its own collapse detector against for
BOTH arms (see `detector_side` in fit()'s return dict and in
pin_vs_ema_report()'s per-arm dict -- always "online" as of ROUND THREE below).

ROUND THREE, BECAUSE THE DETECTOR ITSELF WAS THE BUG ROUND TWO BUILT AROUND.
fit() used to read the EMA arm's TARGET and the pinned arm's ONLINE side --
two different objects -- which manufactured a false asymmetry: pin_target=True
raised at seeds 5502/5503/5504 while pin_target=False raised at none of
5501-5505, on a bed where reading the SAME side (online) for both arms shows
the pinned arm's online effective rank never below the EMA arm's. The false
raises came from a transient both arms' online representation dips into early
(erank below 1.5 for a window closing by step 54 on these five seeds) and
leaves before N_STEPS; pi_jepa.py's rank leg now has a measured RANK_BURN_IN
and does not read it before then. Both fixes are in pi_jepa.py, not here; this
file's job is to no longer be the "one seed's threshold" the old detector bug
let it get away with, which is what FIVE_SEEDS below is for. The module
fixture below used to run at whatever pj.SEED equalled (5501 by default) with
no seed loop, so it never exercised the four seeds the old detector could not
survive; it is unchanged for the three single-run wiring/liveness checks
above -- those are construction and negative-control checks, not the
cross-seed ordering claim -- and a separate, explicit five-seed fixture below
carries the ordering claim across every seed the brief names.

  WHAT MAY NOT BE USED AS A KILL: "the pinned arm must beat frozen-random on
  error". The claim that frozen_random BEATS trained on this bed is RETIRED,
  not softened (see baselines()'s own docstring: as measured on this file's
  first run, frozen random reads 0.6881 NRMSE against the trained run's
  0.8624). What the trained run buys is a live representation, not a lower
  error -- which is exactly why the RED here is about liveness, never about
  NRMSE or MSE.

  CONTROLS, both already present in pi_jepa.py: frozen-random via
  train_run(train_encoder=False), and baseline_detail() for the retired
  error comparison this file does not repeat.

  DTYPE: EQUIV_TOL = 1e-13 and the bitwise beta=1 corner claim
  (shipped_equivalence() / corner_identities()) are float64 statements,
  measured on explicit float64 tensors elsewhere in this module. The pin
  under test here runs the read on the SAME tensors default_batches() hands
  fit() -- draw_bed() produces float32, PiJepa.online/target are float32
  nn.Linear stacks, and _pinned_target_repr never upcasts -- so every number
  this file reports below is a plain Python float computed FROM a float32
  representation (collapse_report itself upcasts to float64 internally
  before computing std_min/erank; see pi_jepa.py's own collapse_report). On
  the bf16 Kaggle path with M=800 the beta floor is 0.8891 per the round's
  brief, so this file's corner claim is a FLOAT32-INPUT, CPU, SYNTHETIC-BED
  statement and is not claimed to transfer to that path without being
  re-derived there.

WHY THIS FILE DOES NOT COLLECT RED. pi_jepa.py already carries PIN_BETA,
PiJepa(pin_target=...), target_repr's pin branch, build/train_run/fit's
pin_target threading, and pin_vs_ema_report() -- landed by the lane that owns
that file while this one was being written, in the same round. That is a
scheduling fact about two lanes running at once, not a relaxation of the
brief above: every number this file asserts on is measured in THIS file's
own run, against the thresholds and the ordering the brief names, never
against a number pin_vs_ema_report() or any other lane's report already
printed. Nothing here calls _held_out_scores, baseline_detail or baselines,
so nothing here can accidentally become the error comparison the brief rules
out.
"""
from __future__ import annotations

import pytest
import torch

from ceqjepa import pi_jepa as pj

#: never the seed anything in this file trains on -- baseline_detail's own
#: "held = seed + 1" convention, reused so the liveness reads below are on
#: tokens no arm here was fit against.
HELD = pj.SEED + 1


@pytest.fixture(scope="module")
def bundle():
    """Every expensive read this file needs, computed once and shared.

    Two training/build calls at N_STEPS on one CPU core beyond
    pin_vs_ema_report()'s own two (which fits the shipped EMA arm and the
    pinned arm): an UNTRAINED pinned arm at the SAME seed train_run() itself
    starts every arm from (build() re-seeds deterministically, so this is
    bit-identical to that arm's own step-0 state, not a re-derivation), and
    the trained pinned arm itself so its `model` object is available directly.
    All reads below share the SAME held-out (ctx, tgt, a) at seed HELD, which
    no arm here trained on, so 'at init' and 'after training' are compared on
    identical tokens in this one run.
    """
    report = pj.pin_vs_ema_report(seed=pj.SEED, steps=pj.N_STEPS)
    ctx, tgt, _a = pj.default_batches(HELD)(0)

    init_model = pj.build(pj.SEED, pin_target=True)
    trained = pj.train_run(seed=pj.SEED, steps=pj.N_STEPS,
                            pin_target=True, train_encoder=True)["model"]

    plant = pj.collapse_report(pj.RankOneEncoder()(ctx)[:, -1, :])
    with torch.no_grad():
        online_init_chk = pj.collapse_report(init_model.online(ctx)[:, -1, :])
        online_trained_chk = pj.collapse_report(trained.online(ctx)[:, -1, :])
    return dict(
        report=report,
        online_init_chk=online_init_chk,
        online_trained_chk=online_trained_chk,
        plant=plant,
        dtype=str(ctx.dtype),
    )


def test_pin_target_rms_ratio_is_one_construction_check(bundle):
    """construction_check: CANNOT FAIL once pin_target actually removes the
    EMA from the target's path. This measures that the target no longer
    depends on a trained parameter, not that the target is any good -- see
    test_pinned_target_is_alive_after_training for the claim that can fail.

    The EMA arm is read from the SAME pin_vs_ema_report() call as a control:
    if the EMA arm's own ratio also read 1.0000, this construction check
    would be measuring the held-out window's own staleness rather than the
    pin, so the EMA arm is required to have actually moved.
    """
    pinned = bundle["report"]["pinned"]
    ema = bundle["report"]["ema"]

    assert pinned["target_rms_before"] == pinned["target_rms_after"], (
        "dtype %s: pinned target RMS moved from %.6f to %.6f -- it still "
        "depends on a trained parameter and the EMA has not been removed"
        % (pinned["dtype"], pinned["target_rms_before"],
           pinned["target_rms_after"])
    )
    assert pinned["target_rms_ratio"] == 1.0, (
        "dtype %s: ratio %.10f, not 1.0000 BY CONSTRUCTION"
        % (pinned["dtype"], pinned["target_rms_ratio"])
    )
    assert ema["target_rms_ratio"] != 1.0, (
        "the EMA control arm's own ratio read exactly 1.0000 (dtype %s): "
        "this construction check is not exercising a target that moves"
        % ema["dtype"]
    )


def test_rank_one_plant_is_the_negative_the_variance_leg_alone_misses(bundle):
    """Re-measures, in THIS run, the one-leg counterexample the brief cites
    rather than taking it on faith: a rank-one plant whose std_min clears the
    variance floor while its effective rank sits at the collapse it is."""
    plant = bundle["plant"]
    assert plant["std_min"] >= pj.COLLAPSE_STD_MIN, (
        "the rank-one plant no longer clears the variance leg (std_min "
        "%.4e against floor %.4g) -- it has stopped being the counterexample "
        "the effective-rank leg exists for" % (plant["std_min"], pj.COLLAPSE_STD_MIN)
    )
    assert plant["erank"] < pj.COLLAPSE_ERANK_MIN, plant
    assert plant["leg"] == "rank" and plant["collapsed"], plant


def test_pinned_target_is_alive_after_training(bundle):
    """THE RED THAT CAN FAIL. The pinned TARGET (target_repr) is a constant
    read of a fixed batch by construction -- see
    test_pin_target_rms_ratio_is_one_construction_check -- so it cannot be the
    RED: the ONLY thing that trains in the pinned arm is the ONLINE encoder,
    and that is what this test reads collapse_report on, on held-out tokens
    (HELD = SEED + 1), never on target_repr.

    Both legs, read off the SAME held-out window as the rank-one plant above,
    so the effective-rank bar is the plant's OWN measured level in this run,
    never a fixed magnitude. A third assertion checks that the quantity this
    test reads actually MOVES under training -- collapse_report on the online
    representation at step 0 (init_model, the same seed train_run() itself
    starts from) against the SAME trained arm after N_STEPS, on the identical
    held-out tokens -- because a number training cannot change is not a RED;
    it is the same mistake the retired frozen-vs-trained test made on the
    target side, now checked on the side that can actually fail it. Only
    std_min is asserted to move: the module docstring's caveat on erank (it
    falls from init to 120 steps on this held-out window while rising on the
    training window at 480) means its DIRECTION is not established on one
    seed, only its floor clearance against the plant.

    KILL, after N_STEPS = 120 steps: std_min < 0.001; or effective rank falls
    to the rank-one plant's level; or std_min at step 120 is no larger than
    std_min at step 0 on the identical held-out tokens.
    """
    chk = bundle["online_trained_chk"]
    init = bundle["online_init_chk"]
    plant_erank = bundle["plant"]["erank"]

    assert chk["std_min"] > pj.COLLAPSE_STD_MIN, (
        "KILL: after %d steps, online std_min %.6e is at or below the "
        "collapse floor %.4g (dtype %s, held-out tokens)"
        % (pj.N_STEPS, chk["std_min"], pj.COLLAPSE_STD_MIN, bundle["dtype"])
    )
    assert chk["erank"] > plant_erank, (
        "KILL: after %d steps, online effective rank %.4f did not clear the "
        "rank-one plant's %.4f -- alive in spread, dead in rank"
        % (pj.N_STEPS, chk["erank"], plant_erank)
    )
    assert chk["std_min"] > init["std_min"], (
        "KILL: online std_min did not move under training -- %.6f at step 0 "
        "vs %.6f at step %d, on the identical held-out tokens (dtype %s); a "
        "quantity training cannot change is not a RED"
        % (init["std_min"], chk["std_min"], pj.N_STEPS, bundle["dtype"])
    )


#: THE FIVE SEEDS THE BRIEF NAMES, explicitly -- not pj.SEED, so this claim
#: does not silently track whatever the module default happens to be.
FIVE_SEEDS = (5501, 5502, 5503, 5504, 5505)


@pytest.fixture(scope="module")
def five_seed_online_erank():
    """Per seed in FIVE_SEEDS, per arm: build and fit at that seed on that
    seed's own bed (train_encoder=True, the shipped detector=True default --
    this is itself part of the claim: after pi_jepa.py's fix neither arm's
    detector should raise at any of these five seeds, where before the fix
    the pinned arm raised at three of them), then read the ONLINE
    representation's effective rank on HELD-OUT tokens at seed+1 (HELD's own
    convention, reused here per seed). A raise from train_run inside this
    fixture is not caught: it is itself a finding (the detector regressed),
    not a fixture error to swallow.
    """
    out = {}
    for seed in FIVE_SEEDS:
        held = seed + 1
        ctx, _tgt, _a = pj.default_batches(held)(0)
        row = {}
        for pin in (True, False):
            model = pj.train_run(seed=seed, steps=pj.N_STEPS, pin_target=pin,
                                  train_encoder=True)["model"]
            with torch.no_grad():
                row[pin] = pj.collapse_report(model.online(ctx)[:, -1, :])["erank"]
        out[seed] = row
    return out


def test_pinned_online_erank_never_below_ema_across_five_seeds(five_seed_online_erank):
    """THE FIVE-SEED CLAIM ITSELF, replacing the single-seed threshold ROUND
    THREE's docstring paragraph describes. At every one of FIVE_SEEDS, both
    arms built and fit at the SAME seed on the SAME bed, the pinned arm's
    end-of-training ONLINE effective rank -- read on held-out tokens at
    seed+1, never on the training window -- is never below the EMA arm's.
    Measured in THIS run (not asserted from the round's brief): margins of
    0.002970 / 0.819405 / 0.635031 / 0.329435 / 0.133843 at 5501-5505
    respectively, the first close enough to call a tie and none reversed --
    which is why the assertion is >=, never >.

    SCOPE: this is a fact about seeds 5501-5505, not about the bed in
    general. On fresh seeds 6001-6008 the ordering REVERSES at seed 6003
    (pinned online erank 2.809445 against the EMA arm's 3.305759, margin
    -0.496314) and at seed 6004 (2.123798 against 3.830536, margin
    -1.706738) -- 4 of 8 fresh seeds reversed or undecidable, both reversal
    margins one to two orders of magnitude larger than the +0.002970 tie
    5501 (this test's own weakest seed) rests on. The reason is the whole
    point of the third collapse leg this round's lane is landing elsewhere:
    this test reads the ordering in EFFECTIVE RANK, and effective rank is
    inverted for this purpose -- a representation that has actually learned
    structure pins at LOW effective rank (clustering IS anisotropy) while an
    uninformative one spreads across all directions and reads HIGHER, so
    "never below in erank" is not "never worse" outside the five seeds
    measured here.

    KILL: any one of the five seeds where the pinned arm's held-out online
    erank measures BELOW the EMA arm's. Swapping either arm's read from
    model.online to model.target_repr, or reading the training window instead
    of HELD, changes what is compared and would very likely flip at least one
    seed -- this is a real, seed-dependent training outcome, not an identity.
    """
    for seed in FIVE_SEEDS:
        row = five_seed_online_erank[seed]
        assert row[True] >= row[False], (
            "seed %d: pinned online erank %.6f fell below the ema arm's "
            "%.6f (margin %.6f) -- the never-below claim broke at this seed"
            % (seed, row[True], row[False], row[True] - row[False])
        )


def test_burn_in_still_fires_on_a_genuine_collapse():
    """RULE 2's OWN CHECK, so RANK_BURN_IN is a repair and not a disabling.
    model.online is REPLACED with RankOneEncoder before fit() takes a single
    step, so the planted dimensional collapse (erank exactly 1.0, the module
    docstring's own rank-one plant) is what the rank leg's first guarded
    entry sees. detector=True raises there immediately, at step
    RANK_BURN_IN == 55: EXACTLY 1 guarded entry is evaluated, of 56 detector
    calls total (steps 0 through 55) -- not the N_STEPS - RANK_BURN_IN == 65
    guarded entries a run that continued to completion would exercise. 65 of
    120 is what happens with the detector OFF; it is not a count this run
    produces, because detector=True stops fit() at the first guarded entry.
    The std leg (unguarded) sees the same plant's std_min ~= 0.2781 on every
    one of those same 56 calls and does not fire, because the rank-one plant
    is exactly the counterexample the variance leg alone misses. The bar
    itself is real: the one reachable guarded entry fires, and fires on the
    rank leg.

    SCOPE OF RANK_BURN_IN == 55 (set in pi_jepa.py, not owned by this file):
    it was derived from "the latest step any of 10 runs was still below 1.5
    is step 54", measured on seeds 5501-5505 only, and is a property of
    those five draws, not a property of the bed in general. On fresh seed
    6002 the EMA arm is still below 1.5 (1.2062) at step 55 and raises; on
    fresh seed 6005 the pinned arm is still below 1.5 (1.4176) at step 55
    and raises. This test's own plant is unaffected by that -- a guaranteed
    erank-1.0 collapse fires on the first guarded entry regardless of where
    the burn-in line sits -- but RANK_BURN_IN does not generally mean "clear
    of the transient by step 55" on an arbitrary seed; it means that on
    5501-5505.

    KILL: fit() completes all N_STEPS without raising (the burn-in disabled
    the leg rather than delaying it), or raises at a step before
    RANK_BURN_IN (the std leg fired instead, so this run would not have
    confirmed the RANK leg survived the change).
    """
    model = pj.build(pj.SEED, pin_target=True)
    model.online = pj.RankOneEncoder(pj.D_LATENT)
    with pytest.raises(AssertionError) as exc:
        pj.fit(model, steps=pj.N_STEPS, seed=pj.SEED, detector=True)
    msg = str(exc.value)
    step = int(msg.split(":", 1)[0].split()[-1])
    assert "rank leg fired" in msg, (
        "the plant raised via a different leg than the one under test: %r" % msg
    )
    assert step >= pj.RANK_BURN_IN, (
        "the rank-one plant tripped the detector at step %d, before "
        "RANK_BURN_IN (%d): the burn-in is not gating the read it claims to"
        % (step, pj.RANK_BURN_IN)
    )

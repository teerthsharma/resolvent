"""X35a -- the residual instrument, and the two must-fires that decide it.

CEQ_V15_1_DELTA.md, X35a: `r = z_obs - z_model(visible)`, onset by a MEMORYLESS
comparator on residual energy (Shewhart), threshold calibrated on NO-PLANT runs.

    MUST-FIRE 1. planted hidden cause => onset within +/-1 of the true index.
    MUST-FIRE 2. NO plant => no onset called, at the calibrated false-alarm rate.

Must-fire 2 is the SCOPE control and it is the whole node: "a bad model's
approximation error masquerades as a hidden cause everywhere, so the no-plant
residual must be flat or every onset is void." A detector validated only on
planted data is MISTAKES.md V-8 -- the PASS half's label is constant. CEQ_V15_2_
DELTA.md restates the same gate from the other side: the author's own KK probe
read 0.000 on a planted anticipating kernel and is VOID, because a test that
cannot fire measures nothing. The mirror defect is a detector that fires on
planted AND unplanted data alike, which is what the false-alarm rate below is
reported with its run count to exclude.

WHAT z_model IS HERE, AND WHAT THAT BOUNDS. Nothing is trained (L-LEAN), so the
visible model is the ORACLE: the generator's own analytic forward pass over the
VISIBLE variables only (`bed_k.rebuild(bed, bed["b"])` -- `b` and `pos`, never
`bed["plant"]`). With an exact visible model the no-plant residual is EXACTLY
the observation noise, so "flat under no plant" is checkable exactly and the
bar is set before any arm's residual is ever read. A trained arm carries
approximation error the oracle does not; the flatness reading below is
therefore a PRECONDITION that must be RE-MEASURED when a real arm exists, not
a property inherited by every future z_model.

That is also why `test_flatness_gate_has_a_rejection_region` exists. A gate
whose comparison holds at the extremes of its own quantity's range is
MISTAKES.md V-10, vacuous before it executes; the oracle's residual is flat by
construction, so the flatness gate is only worth reading if a DELIBERATELY
imperfect visible model -- a finite-window truncation of the same kernel, which
is exactly the error a limited-context arm makes -- is shown to fail it on the
same code path.

CALIBRATION HYGIENE. MISTAKES.md M-2: a threshold refitted to the data it
judges is not a threshold. The threshold is the (1-alpha) quantile of the
run-max residual energy over the CALIBRATION seed block, and every reported
false-alarm rate is measured on the EVALUATION seed block, which is disjoint
from it by construction and asserted disjoint below.

NO CRB IS QUOTED ANYWHERE IN THIS FILE. CEQ_V15_2_DELTA.md (d) asks for
distance-to-CRB beside every onset CI and kills any CRB number quoted before
its own noise-sweep must-fire. The onset here is a DISCRETE index and enters
the likelihood only through it, so the Fisher information for it does not
exist and there is no CRB to be at a distance from -- see V15_X35A_RESIDUAL.md.
What is reported instead is the MEASURED empirical interval of onset offsets,
which is a CI and not a floor.

Run order: this file is authored and shown RED before `ceq/x35/residual.py`
exists. V15_X35A_RESIDUAL.md carries that transcript.
"""
from __future__ import annotations

import numpy as np
import pytest

from ceq.beds import bed_k
from ceq.x35 import residual as rx

# --- pre-registered configuration (frozen before any number was read) -------
N = 128
KIND, PARAMS = "delay", dict(d=5)
PLANT_INDEX = 36          # the delta's own reference run: "36 vs 37 at sd 0.05"
PLANT_MAG = 1.0
SECOND_INDEX = 80         # v15.2 must-fire 2: superposition, two planted sources
ALPHA = 0.01              # target per-run false-alarm rate
N_CAL, CAL_SEED0 = 2000, 0
N_EVAL, EVAL_SEED0 = 2000, 1_000_000
SD_CLEAN, SD_NOISY = 0.0, 0.05

#: MUST-FIRE 1's bar, pre-registered from the analytic prediction, NOT from the
#: realised rate. At sd=0.05 the threshold is the 0.99 quantile of the run-max
#: of (0.05*g)^2 over N=128 iid normals: P(max|g| <= x) = (2*Phi(x)-1)^128 =
#: 0.99 gives x ~ 3.94, so tau ~ (0.05*3.94)^2 ~ 0.0388. The plant's drive at
#: its own index is m*u with u~N(0,1), so the detector misses index t* only if
#: |u| < sqrt(0.0388)/1.0 ~ 0.197, probability ~0.156; it misses t* AND t*+1
#: with probability ~0.156^2 ~ 0.024, i.e. onset within +/-1 is predicted at
#: ~0.976. The bar is set at 0.95 -- below the prediction with margin, and far
#: above the ~0 a broken detector reads. Realised rate is printed beside it.
LOCALIZATION_BAR = 0.95
N_LOCALIZATION = 500

#: Flatness bars. `corr` is the Pearson correlation of POOLED per-index mean
#: residual energy against index; under a flat residual its null sd is
#: 1/sqrt(N-1) = 1/sqrt(127) = 0.089 no matter how many runs are pooled (more
#: runs shrink the deviations but the correlation is scale-invariant), so 0.35
#: is ~4 null sd. `half_ratio` pools 200 runs x 64 indices per half, relative
#: sd ~ sqrt(2/12800) = 1.25%, so 0.9/1.1 is ~8 sd.
FLAT_CORR_BAR = 0.35
N_FLAT = 200


def _cfg(noise_sd, model=None, kind=KIND, params=None):
    return dict(n=N, kind=kind, params=dict(PARAMS if params is None else params),
                noise_sd=noise_sd, model=model)


@pytest.fixture(scope="module")
def cal_noisy():
    return rx.calibrate(range(CAL_SEED0, CAL_SEED0 + N_CAL), ALPHA, **_cfg(SD_NOISY))


@pytest.fixture(scope="module")
def cal_clean():
    return rx.calibrate(range(CAL_SEED0, CAL_SEED0 + N_CAL), ALPHA, **_cfg(SD_CLEAN))


# ---------------------------------------------------------------------------
# MUST-FIRE 1 -- planted hidden cause => onset within +/-1 of the true index
# ---------------------------------------------------------------------------


def test_must_fire_1_named_run_onset_within_one_at_sd_0_and_sd_005(cal_clean, cal_noisy):
    """The delta's `[RUN: exact at 0 noise, 36 vs 37 at sd 0.05]`, reproduced
    on one named seed so the +/-1 claim is checkable against a printed true
    index. Seed 0 of the EVALUATION block -- disjoint from every calibration
    seed, so the threshold this run is judged by never saw it.
    """
    plant = dict(index=PLANT_INDEX, magnitude=PLANT_MAG)
    seed = EVAL_SEED0
    out0 = rx.run_detect(seed, cal_clean["threshold"], plant=plant, **_cfg(SD_CLEAN))
    out5 = rx.run_detect(seed, cal_noisy["threshold"], plant=plant, **_cfg(SD_NOISY))
    print(f"\n[MUST-FIRE 1] true={PLANT_INDEX} onset(sd=0)={out0['onset']} "
          f"onset(sd=0.05)={out5['onset']} mag_hat={out5['magnitude']:.4f} "
          f"(m_true={PLANT_MAG}) tau(sd=0)={cal_clean['threshold']:.6g} "
          f"tau(sd=0.05)={cal_noisy['threshold']:.6g}")

    assert out0["onset"] == PLANT_INDEX, (
        f"at sd=0 the oracle residual is exactly the plant, so onset must be "
        f"exact: got {out0['onset']} vs true {PLANT_INDEX}")
    assert out5["onset"] is not None and abs(out5["onset"] - PLANT_INDEX) <= 1, (
        f"onset {out5['onset']} is not within +/-1 of {PLANT_INDEX} at sd=0.05")

    # `gate_estimates` is what X35b's posited latent node has to carry: the
    # per-index drive, zero upstream of the called onset. The full `residual`
    # field is returned beside it so CEQ_V15_2_DELTA.md's competing estimators
    # (exact source solve, time-reversal) score the SAME field this onset was
    # read from, rather than a separately generated one.
    g = out5["gate_estimates"]
    assert np.all(g[: out5["onset"]] == 0.0)
    assert np.array_equal(g[out5["onset"]:], out5["residual"][out5["onset"]:])
    assert out5["residual"].shape == (N,) and out5["energy"].shape == (N,)


def test_must_fire_1_localization_rate_meets_the_preregistered_bar(cal_noisy):
    """The delta's second KILL: "localization below the pre-set bar on planted
    beds => the residual is uninformative AT THAT NOISE, stated with the noise
    level." One named seed is one draw (MISTAKES.md M-4); the bar is scored
    over N_LOCALIZATION planted seeds, all from the evaluation block. The
    printed interval is the MEASURED 2.5/97.5 percentile of onset offsets --
    a CI, not a floor; no CRB is quoted (see module docstring).
    """
    plant = dict(index=PLANT_INDEX, magnitude=PLANT_MAG)
    offs = [rx.run_detect(s, cal_noisy["threshold"], plant=plant, **_cfg(SD_NOISY))["onset"]
            for s in range(EVAL_SEED0, EVAL_SEED0 + N_LOCALIZATION)]
    offs = [None if o is None else o - PLANT_INDEX for o in offs]
    hit = [o for o in offs if o is not None]
    within = sum(abs(o) <= 1 for o in hit) / len(offs)
    lo, hi = np.percentile(hit, [2.5, 97.5])
    hist = {k: offs.count(k) for k in sorted(set(hit))}
    print(f"\n[MUST-FIRE 1 / {N_LOCALIZATION} seeds @ sd={SD_NOISY}] within+/-1="
          f"{within:.4f} (bar {LOCALIZATION_BAR}, predicted ~0.976) "
          f"offset 95% CI=[{lo:g}, {hi:g}] offsets={hist} misses={offs.count(None)}")
    assert within >= LOCALIZATION_BAR, (
        f"localization {within} below the pre-set bar {LOCALIZATION_BAR} at "
        f"noise sd={SD_NOISY}: the residual is uninformative at that noise")


def test_must_fire_1_superposition_two_planted_sources(cal_noisy):
    """CEQ_V15_2_DELTA.md must-fire 2: two planted sources, superposed. The
    onset detector answers a strictly weaker question than the wave node's
    source solve -- it can only call the EARLIER onset, because the second
    source switches on inside a residual that is already above threshold. That
    limit is the finding, stated here rather than left for the wave node to
    discover: onset localization does not separate superposed sources, and
    `bed["plant"]["indices"]` carries both so the wave node can score against
    the right ground truth.
    """
    plant = [dict(index=PLANT_INDEX, magnitude=PLANT_MAG),
             dict(index=SECOND_INDEX, magnitude=PLANT_MAG)]
    offs = []
    for s in range(EVAL_SEED0, EVAL_SEED0 + 200):
        o = rx.run_detect(s, cal_noisy["threshold"], plant=plant, **_cfg(SD_NOISY))["onset"]
        offs.append(None if o is None else o - PLANT_INDEX)
    within = sum(o is not None and abs(o) <= 1 for o in offs) / len(offs)
    print(f"\n[SUPERPOSITION / 200 seeds] sources at {PLANT_INDEX} and "
          f"{SECOND_INDEX}; first-onset within+/-1 of {PLANT_INDEX}: {within:.4f}")
    assert within >= LOCALIZATION_BAR


# ---------------------------------------------------------------------------
# MUST-FIRE 2 -- the SCOPE control. No plant => no onset called.
# ---------------------------------------------------------------------------


def test_must_fire_2_no_plant_false_alarm_rate_matches_the_calibrated_target(cal_noisy):
    """Scored on a FRESH draw (MISTAKES.md M-2): the threshold is the 0.99
    quantile of run-max energy over seeds [0, 2000); the rate below is measured
    over seeds [1e6, 1e6+2000), no plant, same noise. Tolerance is the binomial
    envelope of alpha at N_EVAL runs (sd = sqrt(a(1-a)/n) = 0.0022 at a=0.01,
    n=2000); +/-0.01 absolute is ~4.5 sd and still two orders of magnitude
    below the ~1.0 a non-flat residual reads.
    """
    far = rx.false_alarm_rate(range(EVAL_SEED0, EVAL_SEED0 + N_EVAL),
                              cal_noisy["threshold"], **_cfg(SD_NOISY))
    print(f"\n[MUST-FIRE 2] measured FAR={far['rate']:.4f} ({far['alarms']}/{far['runs']}) "
          f"vs calibrated alpha={ALPHA} | per-index rate={far['per_index_rate']:.3e} "
          f"ARL0={far['arl0']:.4g} indices")
    assert abs(far["rate"] - ALPHA) <= 0.01, (
        f"no-plant false-alarm rate {far['rate']} is not the calibrated {ALPHA} "
        f"on a fresh draw of {far['runs']} runs")


def test_must_fire_2_at_zero_noise_the_no_plant_residual_is_bitwise_zero(cal_clean):
    """The exact-visible-model half of must-fire 2. At sd=0 with an exact
    z_model the residual is bitwise 0.0 at every index, the calibrated
    threshold is therefore 0.0, and the strict `>` comparator calls nothing.
    Stated plainly because it is the reason the bar is exact HERE and not
    inherited: a trained arm's residual at sd=0 is its approximation error,
    which is not 0.0.
    """
    far = rx.false_alarm_rate(range(EVAL_SEED0, EVAL_SEED0 + 200),
                              cal_clean["threshold"], **_cfg(SD_CLEAN))
    _, r = rx.run_residual(EVAL_SEED0, **_cfg(SD_CLEAN))
    print(f"\n[MUST-FIRE 2 @ sd=0] tau={cal_clean['threshold']!r} "
          f"max|r|={float(np.max(np.abs(r)))!r} "
          f"FAR={far['rate']} ({far['alarms']}/{far['runs']})")
    assert np.all(r == 0.0)
    assert far["alarms"] == 0


# ---------------------------------------------------------------------------
# the flatness gate must have a rejection region (MISTAKES.md V-10)
# ---------------------------------------------------------------------------


def test_no_plant_residual_is_flat_under_the_oracle():
    """Flatness, measured rather than asserted: Pearson correlation of pooled
    per-index residual energy against index, and the second-half / first-half
    mean-energy ratio. Under the oracle r IS the iid observation noise, so
    both are 0 and 1 in population; bars are the null sds computed in
    FLAT_CORR_BAR's note."""
    flat = rx.flatness(range(EVAL_SEED0, EVAL_SEED0 + N_FLAT), **_cfg(SD_NOISY))
    print(f"\n[FLATNESS / oracle] energy-vs-index corr={flat['corr']:+.4f} "
          f"half_ratio={flat['half_ratio']:.4f} mean_energy={flat['mean_energy']:.6g} "
          f"(noise variance {SD_NOISY ** 2:g})")
    assert abs(flat["corr"]) < FLAT_CORR_BAR
    assert 0.9 < flat["half_ratio"] < 1.1


@pytest.mark.parametrize("L", [8, 64])
def test_flatness_gate_has_a_rejection_region(L):
    """V-10's rule applied to this node's own kill. The oracle is flat BY
    CONSTRUCTION, so the flatness gate is vacuous unless an imperfect visible
    model is shown to fail it on the same code path. The control is a
    finite-window truncation of the SAME power-law kernel at lag L -- the
    approximation a limited-context arm actually makes -- with no plant
    anywhere. Its residual is the long-memory tail the window dropped: not
    flat, and it calls onsets far above the calibrated rate. That is the
    delta's VOID condition firing, deliberately.

    BAR, AND ITS AMENDMENT. The bar first written here was `FAR > 0.5`, which
    was a guess with no derivation behind it, and L=64 measured 0.2667 against
    it (L=8 measured 1.0000) -- so the first-written test failed. The bar is
    amended to `FAR > 10 * alpha`, which is derived from the quantity the gate
    is supposed to hold rather than from the realised number: the gate claims
    a per-run false-alarm rate of alpha under a good visible model, so an
    order of magnitude above alpha is a rejection, and both realised rates are
    printed beside it. The amendment is recorded because 0.5 -> 0.1 after
    seeing 0.2667 is the shape of MISTAKES.md M-2 if it is done silently; what
    makes it not M-2 is that the new bar is a function of alpha alone and 0.2667
    played no part in its value. V15_X35A_RESIDUAL.md carries the failing run.

    The two L values are kept because the rejection is GRADED in the model's
    error -- L=8 drops most of the tail and reads FAR 1.0, L=64 drops only the
    far tail and still reads 27x the calibrated rate -- which is the shape a
    trained arm's residual will land somewhere inside.
    """
    base = _cfg(SD_NOISY, kind="powerlaw", params=dict(H=0.75))
    cfg = _cfg(SD_NOISY, model=rx.truncated_visible(L), kind="powerlaw",
               params=dict(H=0.75))
    cal = rx.calibrate(range(CAL_SEED0, CAL_SEED0 + 300), ALPHA, **base)
    far = rx.false_alarm_rate(range(EVAL_SEED0, EVAL_SEED0 + 300), cal["threshold"], **cfg)
    flat = rx.flatness(range(EVAL_SEED0, EVAL_SEED0 + 300), **cfg)
    print(f"\n[FLATNESS / truncated L={L}] FAR={far['rate']:.4f} "
          f"corr={flat['corr']:+.4f} half_ratio={flat['half_ratio']:.4g} "
          f"mean_energy={flat['mean_energy']:.4g}")
    assert far["rate"] > 10 * ALPHA, (
        f"a visible model truncated at lag {L} still passes the no-plant "
        f"false-alarm gate (FAR={far['rate']} vs calibrated {ALPHA}): the gate "
        "has no rejection region and every onset it licenses is uninterpretable")


# ---------------------------------------------------------------------------
# a plant of magnitude ZERO is not a plant
# ---------------------------------------------------------------------------


def test_zero_magnitude_plant_leaves_z_bitwise_unchanged():
    """Separates "detects a plant" from "detects the plant FIELD being set".
    The manifest must say a plant is present at PLANT_INDEX while the label is
    bitwise identical to the unplanted one."""
    a = bed_k.build(KIND, N, seed=3, plant=dict(index=PLANT_INDEX, magnitude=0.0), **PARAMS)
    b = bed_k.build(KIND, N, seed=3, **PARAMS)
    assert a["plant"]["present"] is True and a["plant"]["index"] == PLANT_INDEX
    assert b["plant"]["present"] is False and b["plant"]["index"] is None
    assert np.array_equal(a["b"], b["b"])
    assert np.array_equal(a["z"], b["z"])


def test_zero_magnitude_plant_is_not_detected(cal_noisy):
    """Detection rate on a zero-magnitude plant must be the false-alarm rate,
    not ~1. Bar 0.03 = 3*alpha, ~4.5 binomial sd above alpha at 500 runs."""
    plant = dict(index=PLANT_INDEX, magnitude=0.0)
    far = rx.false_alarm_rate(range(EVAL_SEED0, EVAL_SEED0 + 500),
                              cal_noisy["threshold"], plant=plant, **_cfg(SD_NOISY))
    print(f"\n[ZERO-MAGNITUDE PLANT] detection rate={far['rate']:.4f} "
          f"({far['alarms']}/{far['runs']}) vs alpha={ALPHA}")
    assert far["rate"] <= 0.03


# ---------------------------------------------------------------------------
# calibration hygiene, determinism, and the manifest's ground truth
# ---------------------------------------------------------------------------


def test_threshold_is_not_scored_on_the_block_it_was_calibrated_on(cal_noisy):
    """MISTAKES.md M-2, made structural rather than promised: the two seed
    blocks are asserted disjoint, and the in-sample rate is printed BESIDE the
    fresh-draw rate rather than substituted for it. The in-sample rate is
    guaranteed <= alpha by the quantile's own construction, which is exactly
    why it is not evidence and the fresh draw is."""
    cal_seeds = set(range(CAL_SEED0, CAL_SEED0 + N_CAL))
    eval_seeds = set(range(EVAL_SEED0, EVAL_SEED0 + N_EVAL))
    assert not (cal_seeds & eval_seeds)
    in_sample = float(np.mean(cal_noisy["maxima"] > cal_noisy["threshold"]))
    fresh = rx.false_alarm_rate(sorted(eval_seeds)[:500], cal_noisy["threshold"],
                                **_cfg(SD_NOISY))
    print(f"\n[M-2] in-sample FAR={in_sample:.4f} (n={cal_noisy['runs']}) "
          f"fresh-draw FAR={fresh['rate']:.4f} (n={fresh['runs']}) alpha={ALPHA}")
    assert in_sample <= ALPHA + 1e-12


def test_determinism_same_seed_bitwise_same_onset(cal_noisy):
    plant = dict(index=PLANT_INDEX, magnitude=PLANT_MAG)
    a = rx.run_detect(EVAL_SEED0 + 7, cal_noisy["threshold"], plant=plant, **_cfg(SD_NOISY))
    b = rx.run_detect(EVAL_SEED0 + 7, cal_noisy["threshold"], plant=plant, **_cfg(SD_NOISY))
    assert a["onset"] == b["onset"]
    assert a["magnitude"] == b["magnitude"]
    assert np.array_equal(a["residual"], b["residual"])
    c = rx.run_detect(EVAL_SEED0 + 8, cal_noisy["threshold"], plant=plant, **_cfg(SD_NOISY))
    assert not np.array_equal(a["residual"], c["residual"])


def test_manifest_records_plant_ground_truth_on_both_kinds():
    """"A run can never be scored against the wrong ground truth": the plant
    field is present on EVERY manifest, planted or not, on both bed kinds, and
    the latent it records influences z while being absent from the visible
    inputs `b`/`pos`."""
    for kind, params in [("delay", dict(d=5)), ("powerlaw", dict(H=0.75))]:
        p = bed_k.build(kind, N, seed=1, plant=dict(index=PLANT_INDEX, magnitude=2.0), **params)
        assert p["plant"]["present"] and p["plant"]["index"] == PLANT_INDEX
        assert p["plant"]["magnitude"] == 2.0
        assert p["plant"]["indices"] == [PLANT_INDEX]
        u = p["plant"]["u"]
        assert np.all(u[:PLANT_INDEX] == 0.0) and np.any(u[PLANT_INDEX:] != 0.0)
        # The latent influences z and is not any visible input. To float64
        # rounding, not bitwise: `(K@b + u) - K@b` is not `u` exactly for
        # nonzero u. It IS exact where u is exactly 0.0 (adding +0.0 returns
        # the operand bitwise), which is what makes the sd=0 onset exact.
        assert np.allclose(p["z"] - bed_k.rebuild(p, p["b"]), u, rtol=0, atol=1e-12)
        assert np.all((p["z"] - bed_k.rebuild(p, p["b"]))[:PLANT_INDEX] == 0.0)
        assert len(p["b"]) == N and len(p["pos"]) == N

        two = bed_k.build(kind, N, seed=1, **params, plant=[
            dict(index=PLANT_INDEX, magnitude=1.0), dict(index=SECOND_INDEX, magnitude=3.0)])
        assert two["plant"]["indices"] == [PLANT_INDEX, SECOND_INDEX]
        assert two["plant"]["magnitudes"] == [1.0, 3.0]
        assert two["plant"]["index"] == PLANT_INDEX      # the earliest onset
        assert np.all(two["plant"]["u"][:PLANT_INDEX] == 0.0)


def test_oracle_residual_does_not_depend_on_the_kernel():
    """Scope limit, stated as a test rather than a footnote. With an EXACT
    visible model r = noise + plant identically, so the same seed on the delay
    and power-law beds gives the SAME residual. The instrument's bar is
    therefore a statement about the observation model, not about either
    kernel -- which is precisely why it must be re-measured once z_model is a
    trained arm whose error IS kernel-dependent."""
    plant = dict(index=PLANT_INDEX, magnitude=PLANT_MAG)
    _, r_d = rx.run_residual(42, plant=plant, **_cfg(SD_NOISY, kind="delay", params=dict(d=5)))
    _, r_p = rx.run_residual(42, plant=plant, **_cfg(SD_NOISY, kind="powerlaw",
                                                    params=dict(H=0.75)))
    assert np.allclose(r_d, r_p, rtol=0, atol=1e-12)

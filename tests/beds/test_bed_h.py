"""BED-H -- the belief bed. Written RED, before `ceqjepa/beds/bed_h.py` exists.

Contract: `docs/BED_H_PREREGISTRATION.md` at commit 58c197c. That page is not
editable by this file and nothing here relaxes it.

WHY THIS FILE IS WRITTEN FIRST. The pre-registration makes one demand that only
an already-written test can honour: the `L` grid and the decay criterion for
gate 1 must be fixed BEFORE a bed instance is drawn, or "window-L error decays
with L" is a conclusion chosen after seeing the numbers. Those constants live at
the top of this file, this file was run against a non-existent module first, and
that RED transcript is the record that they were pinned before any draw.

PINNED (L-NULL: what is pinned, what varies).

  PINNED, identical on every instance of both forms:
    S = 8, T_LEN = 256, PREFIX_LEN = 128, BURN_IN = 32,
    HIT_SET = {0,1,2}, ABSORB_HORIZON = 8,
    L_GRID = (1, 2, 4, 8, 16, 32), DECAY_SLACK = 0.005, DECAY_MARGIN = 0.05,
    SPREAD_FLOOR = 0.1, N_GRID = (32, 64, 128, 256),
    the belief metric (total variation = half L1),
    the window-L prior (the stationary distribution of P).

  VARIES, per instance, from that instance's seed alone:
    the transition matrix P (STRONG form) or fixed once for the pool (WEAK
    form), lambda_2 ~ U[0.90, 0.97], snr ~ U[0.5, 0.7], the hidden path, the
    observation path, and the prefix.

WHAT EACH TEST WOULD CATCH.

  * `test_oracle_matches_brute_force` is the only thing standing between this bed
    and DCM-1's failure mode, which was an oracle nobody checked. It enumerates
    every hidden path for a 3-state 5-step chain and compares the forward
    algorithm's posterior against the exhaustive one. A forward algorithm that
    transposes P, or normalises in the wrong place, survives every other test in
    this file and dies here.
  * `test_gate1_refuses_a_markov_bed` is the gate firing on the exact thing the
    pre-registration says voids the bed. At snr = 1 the emission is a bijection,
    the belief is a delta on the last observation, and window-1 is already exact.
    A gate that cannot refuse that bed is decoration.
  * `test_gate2_refuses_a_dead_label` is the same for gate 2, at snr = 1/S where
    the observation carries nothing and the belief never leaves stationarity.
  * `test_trap_*` are the pre-registration's trap. They are written against
    `inputs_for_consumer`, not against the instance, because the trap is a claim
    about what a CONSUMER is handed and the exact labels are by construction a
    function of P -- that is what "exact oracle" means.
  * `test_label_scaling_exponents` is what makes the corner rule have a referent
    here rather than an index. It measures the exponent, and assigns no corner.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from ceqjepa.beds import bed_h


# ---- pinned before any bed instance is drawn -------------------------------
L_GRID = (1, 2, 4, 8, 16, 32)
DECAY_SLACK = 0.005      # Monte-Carlo slack on monotonicity, one instance
DECAY_MARGIN = 0.05      # err(1) - err(32) must clear this for a REAL decay
SPREAD_FLOOR = 0.1
N_GRID = (32, 64, 128, 256)
SEED = 7401

# ---- added 2026-09-14 by the prior-art corrections, pinned before the redraw --
# Gate 3, the conditioning gate. Mossel and Roch (cs/0502076): without a
# non-singularity condition the learning problem is at least as hard as noisy
# parity, so a near-singular P makes the bed unlearnable for EVERY arm and the
# instance is worth nothing. The floor is DERIVED from the construction, not
# fitted to the draw: P = (1-eps)I + eps*B with eps <= 1 - 0.90 = 0.10 gives
# sigma_min(P) >= 1 - eps*(1 + sigma_max(B)) >= 1 - 0.10*(1 + sqrt(8)) = 0.62,
# so 0.10 sits far below anything the generator can reach and fires only if the
# construction is broken.
SIGMA_MIN_FLOOR = 0.10

# Recurrent widths. Svete and Cotterell (arXiv:2310.05161, EMNLP 2023) lower-bound
# the neurons for an N-state deterministic PFSA over alphabet Sigma at
# Omega(N |Sigma|) -- but VERIFIED 2026-09-14 that bound is stated for HEAVISIDE
# ELMAN RNNs under weak equivalence, with full-rank and |Sigma| >= |Q| side
# conditions, and the paper's own Limitations section declines to extend it to
# LSTMs. It therefore does NOT prove S = 8 units is too small for a GRU. It is
# used here only as a SIZING HEURISTIC: S*S = 64. The S-unit width is kept and
# reported as a FLOOR and the measurement, not the bound, decides.
FLOOR_HIDDEN = 8
SKYLINE_HIDDEN = 64


def test_pinned_constants_match_this_file():
    """The module may not quietly widen the criterion this file pinned."""
    assert bed_h.L_GRID == L_GRID
    assert bed_h.DECAY_SLACK == DECAY_SLACK
    assert bed_h.DECAY_MARGIN == DECAY_MARGIN
    assert bed_h.SPREAD_FLOOR == SPREAD_FLOOR
    assert bed_h.N_GRID == N_GRID
    assert bed_h.S == 8


def test_lambda2_in_band_and_P_non_symmetric():
    for seed in range(5):
        hmm = bed_h.HMM.draw(np.random.default_rng(SEED + seed))
        assert 0.90 <= hmm.lambda2 <= 0.97, hmm.lambda2
        assert 0.5 <= hmm.snr <= 0.7, hmm.snr
        asym = np.abs(hmm.P - hmm.P.T).max()
        assert asym > 1e-3, "P is symmetric to %.3e" % asym
        assert np.allclose(hmm.P.sum(1), 1.0)
        assert np.allclose(hmm.pi @ hmm.P, hmm.pi)


def test_oracle_matches_brute_force():
    """Forward algorithm vs exhaustive enumeration of every hidden path."""
    rng = np.random.default_rng(11)
    n, T = 3, 5
    P = rng.random((n, n)) + 0.1
    P /= P.sum(1, keepdims=True)
    E = rng.random((n, n)) + 0.1
    E /= E.sum(1, keepdims=True)
    pi = np.full(n, 1.0 / n)
    obs = rng.integers(0, n, size=T)

    exact = np.zeros((T, n))
    for t in range(T):
        joint = np.zeros(n)
        for path in np.ndindex(*([n] * (t + 1))):
            p = pi[path[0]] * E[path[0], obs[0]]
            for k in range(1, t + 1):
                p *= P[path[k - 1], path[k]] * E[path[k], obs[k]]
            joint[path[t]] += p
        exact[t] = joint / joint.sum()

    got = bed_h.forward_belief(P, E, pi, obs)
    assert np.abs(got - exact).max() < 1e-12, np.abs(got - exact).max()


def test_window_forward_equals_exact_when_window_covers_the_prefix():
    hmm = bed_h.HMM.draw(np.random.default_rng(SEED))
    obs = hmm.sample(np.random.default_rng(SEED + 1), 64)[1]
    exact = hmm.forward(obs)
    wide = hmm.window_forward(obs, 64)
    assert np.abs(exact - wide).max() < 1e-12


def test_gate1_window_error_decays_on_a_drawn_instance():
    inst, hmm = bed_h.draw_instance(SEED, form="strong")
    g = bed_h.gate_report(inst, hmm)
    errs = [g["window_err"][L] for L in L_GRID]
    assert g["gate1_pass"], errs
    assert errs[0] - errs[-1] >= DECAY_MARGIN, errs


def test_gate1_refuses_a_markov_bed():
    """snr = 1 makes the emission a bijection: the bed is Markov in y_t."""
    hmm = bed_h.HMM.draw(np.random.default_rng(SEED), snr=1.0)
    inst = bed_h.instance_from(hmm, np.random.default_rng(SEED + 2), form="weak")
    g = bed_h.gate_report(inst, hmm)
    assert not g["gate1_pass"], g["window_err"]
    assert "gate1" in g["refusals"]


def test_gate2_refuses_a_dead_label():
    """snr = 1/S: the observation carries nothing, the belief stays at pi."""
    hmm = bed_h.HMM.draw(np.random.default_rng(SEED), snr=1.0 / bed_h.S)
    inst = bed_h.instance_from(hmm, np.random.default_rng(SEED + 3), form="weak")
    g = bed_h.gate_report(inst, hmm)
    assert not g["gate2_pass"], (g["spread_intensive"], g["spread_extensive"])
    assert "gate2" in g["refusals"]
    assert min(g["spread_intensive"], g["spread_extensive"]) < SPREAD_FLOOR


def test_trap_consumer_inputs_are_exactly_the_whitelist():
    inst, _ = bed_h.draw_instance(SEED, form="strong")
    assert set(bed_h.inputs_for_consumer(inst)) == set(bed_h.CONSUMER_KEYS)
    weak, _ = bed_h.draw_instance(SEED, form="weak")
    assert bed_h.inputs_for_consumer(weak)["prefix_obs"] is None


def test_trap_P_is_not_reachable_from_consumer_inputs():
    """Fails if P -- or P transposed, rescaled, or with its entries reordered --
    sits anywhere in the float content a consumer is handed."""
    inst, hmm = bed_h.draw_instance(SEED, form="strong")
    leaks = bed_h.scan_for_matrix(bed_h.inputs_for_consumer(inst), hmm.P)
    assert leaks == [], leaks
    # and the emission matrix is no better a door
    assert bed_h.scan_for_matrix(bed_h.inputs_for_consumer(inst), hmm.E) == []


def test_trap_a_planted_leak_is_caught():
    """The scanner is useless unless it fires. Plant P and require a hit."""
    inst, hmm = bed_h.draw_instance(SEED, form="strong")
    payload = bed_h.inputs_for_consumer(inst)
    payload["obs"] = np.concatenate(
        [payload["obs"].astype(float), hmm.P.ravel()[::-1] * 3.0 + 1.0])
    assert bed_h.scan_for_matrix(payload, hmm.P) != []


def test_label_scaling_exponents_separate_intensive_from_extensive():
    """The corner rule's referent. Exponents measured; no corner assigned."""
    exps = bed_h.label_exponents(n_inst=24, form="weak", seed=SEED)
    assert abs(exps["alpha_intensive"]) < 0.10, exps
    assert abs(exps["alpha_extensive"] - 1.0) < 0.10, exps
    assert exps["alpha_extensive"] - exps["alpha_intensive"] > 0.5, exps


def test_ema_and_window_are_worse_than_the_exact_oracle():
    inst, hmm = bed_h.draw_instance(SEED, form="weak")
    exact = inst.belief
    w4 = bed_h.tv(hmm.window_forward(inst.obs, 4), exact)
    ema = bed_h.tv(bed_h.ema_belief(inst.obs, 0.4, bed_h.S), exact)
    assert w4 > 0.0 and ema > 0.0
    assert bed_h.tv(exact, exact) == pytest.approx(0.0, abs=1e-15)


# ============================================================================
# Added 2026-09-14 after the prior-art corrections. These were run RED against
# the module as it stood before `RecurrentBaseline`, gate 3 and the extensive
# baselines existed; that transcript is the record.
# ============================================================================

def test_pinned_constants_from_the_corrections():
    assert bed_h.SIGMA_MIN_FLOOR == SIGMA_MIN_FLOOR
    assert bed_h.FLOOR_HIDDEN == FLOOR_HIDDEN == bed_h.S
    assert bed_h.SKYLINE_HIDDEN == SKYLINE_HIDDEN
    assert bed_h.SKYLINE_HIDDEN >= bed_h.S * bed_h.S, "below the Omega(N|Sigma|) bound"


def test_gate3_refuses_a_near_singular_P():
    """Mossel-Roch: a rank-deficient P makes the bed unlearnable for every arm.
    Two identical rows is exactly that, and the gate must refuse it."""
    rng = np.random.default_rng(SEED)
    P = rng.dirichlet(np.ones(bed_h.S), size=bed_h.S)
    P[1] = P[0]                                  # rank S-1
    hmm = bed_h.HMM.from_matrix(P, snr=0.6)
    assert hmm.sigma_min < SIGMA_MIN_FLOOR, hmm.sigma_min
    inst = bed_h.instance_from(hmm, np.random.default_rng(SEED + 4), form="weak")
    g = bed_h.gate_report(inst, hmm)
    assert not g["gate3_pass"]
    assert "gate3" in g["refusals"]


def test_gate3_passes_on_drawn_instances_and_sigma_min_is_measured():
    for k in range(8):
        inst, hmm = bed_h.draw_instance(SEED + 1000 * k, form="strong")
        g = bed_h.gate_report(inst, hmm)
        assert g["sigma_min"] == pytest.approx(hmm.sigma_min)
        assert g["gate3_pass"], hmm.sigma_min


def test_gate1_is_reported_as_a_sanity_check_not_as_evidence():
    """Le Gland and Mevel (MCSS 13:63-93, 2000): exponential forgetting holds by
    theorem, so gate 1 cannot fail unless the generator is broken. The module
    must say that in its own report rather than sell the gate as discrimination."""
    inst, hmm = bed_h.draw_instance(SEED, form="strong")
    g = bed_h.gate_report(inst, hmm)
    assert g["gate1_kind"] == "sanity-check"


def test_decay_rate_is_measured_against_the_lambda2_prediction():
    """arXiv:1710.06078 gives the window length for error eps as
    log(eps)/(lambda_2 - lambda_1), i.e. a per-step decay factor of lambda_2.
    The module must report the MEASURED factor beside that prediction."""
    inst, hmm = bed_h.draw_instance(SEED, form="strong")
    g = bed_h.gate_report(inst, hmm)
    assert 0.0 < g["decay_per_step_measured"] < 1.0
    assert g["decay_per_step_predicted"] == pytest.approx(hmm.lambda2)
    # the DERIVED LEVEL was removed by the audit: the theory gives a slope only.
    assert "w4_derived_from_lambda2" not in g
    assert 0.0 < g["decay_per_step_quenched"] < 1.0


def test_extensive_consumer_baseline_exists_and_is_beaten_by_the_oracle():
    """The extensive corner needs its own baselines or the counting claim is
    untested. The naive running count of hits in the observation stream is the
    consumer floor; the window-L oracle's cumulated belief is the oracle line."""
    inst, hmm = bed_h.draw_instance(SEED, form="weak")
    true = inst.cum_hits
    naive = bed_h.count_obs(inst.obs)
    oracle4 = np.cumsum(hmm.window_forward(inst.obs, 4) @ inst.hit_vector)
    assert bed_h.count_mae(oracle4, true) < bed_h.count_mae(naive, true)
    assert bed_h.count_mae(np.cumsum(inst.belief @ inst.hit_vector), true) < 1e-12


def test_recurrent_baseline_has_both_heads_and_both_cells():
    for cell in ("gru", "lstm"):
        m = bed_h.RecurrentBaseline(bed_h.S, hidden=SKYLINE_HIDDEN, cell=cell)
        logb, cnt = m(torch.zeros(2, 7, bed_h.S))
        assert logb.shape == (2, 7, bed_h.S)
        assert cnt.shape == (2, 7)
        assert torch.allclose(logb.exp().sum(-1), torch.ones(2, 7), atol=1e-5)
    with pytest.raises(ValueError):
        bed_h.RecurrentBaseline(bed_h.S, hidden=8, cell="transformer")


def test_trap_consumers_have_no_door_for_P_after_the_corrections():
    for fn in (bed_h.ema_belief, bed_h.count_obs,
               bed_h.RecurrentBaseline.__init__, bed_h.fit_rnn):
        names = set(inspect.signature(fn).parameters)
        assert not (names & {"P", "E", "pi", "hmm"}), (fn, names)


def test_lyapunov_gap_recovers_lambda2_when_the_observations_carry_nothing():
    """The one case with a closed form, and the reason the estimator is trusted.

    Ye, Ma and Qian (arXiv:1710.06078, Physica D 460:134053, 2024) put the memory
    decay at the gap of the top two LYAPUNOV EXPONENTS of the random product
    M D_1 M D_2 ..., where D_n is the diagonal of emission likelihoods -- NOT at
    an eigenvalue of M, and they warn that per-matrix eigenvalues "have little to
    do with this asymptotic behavior". At snr = 1/S every D_n is (1/S) I, the
    product collapses to (1/S)^n (M^T)^n, and the gap is exactly log lambda_2(M).
    So this test pins the estimator against the only closed form available AND
    shows why lambda_2 is the right predictor only when the data says nothing."""
    hmm = bed_h.HMM.draw(np.random.default_rng(SEED), snr=1.0 / bed_h.S)
    obs = hmm.sample(np.random.default_rng(SEED + 5), 8000)[1]
    got = np.exp(bed_h.lyapunov_gap(hmm, obs))
    # 1e-3 relative, not exact: the QR estimator averages log|R_ii| over a finite
    # run and the transient decays as O(1/n). The next assertion is what rules out
    # a BIAS -- the error must shrink when the run gets longer.
    assert got == pytest.approx(hmm.lambda2, rel=1e-3), (got, hmm.lambda2)
    short = np.exp(bed_h.lyapunov_gap(hmm, obs[:500]))
    assert abs(got - hmm.lambda2) < abs(short - hmm.lambda2)


def test_informative_emissions_forget_faster_than_lambda2_predicts():
    """The finding the bed is built to expose: with real observations the filter
    forgets far faster than the chain's own second eigenvalue, so lambda_2
    OVER-predicts the window error and must not be used as the bar."""
    inst, hmm = bed_h.draw_instance(SEED, form="strong")
    g = bed_h.gate_report(inst, hmm)
    lyap = np.exp(bed_h.lyapunov_gap(hmm, inst.obs))
    assert lyap < hmm.lambda2, (lyap, hmm.lambda2)
    measured = g["decay_per_step_measured"]
    assert abs(measured - lyap) < abs(measured - hmm.lambda2)
    assert g["decay_per_step_lyapunov"] == pytest.approx(lyap, rel=0.2)
    # the QUENCHED slope is the one the theory predicts, and it is the one that
    # must match -- the annealed slope is above it by Jensen, 12 of 12 measured.
    assert g["decay_per_step_quenched"] <= g["decay_per_step_measured"] + 1e-9
    assert "w4_derived_from_lyapunov" not in g


# ============================================================================
# THE REFUSAL LEG. Added 2026-09-14 after docs/BED_H_PREREGISTRATION.md was
# rewritten at e7f7704, which struck three quarters of this bed's delta and left
# exactly one sentence standing:
#
#   "A protocol that scores a sequence architecture's BOTTOM emission against an
#    exactly-known filter posterior, with the abstain threshold anchored at
#    ln S - delta rather than TUNED FOR COVERAGE."
#
# So the load-bearing property is not that the refusal works. It is that delta is
# ANCHORED. These tests exist to make "not tuned for coverage" a checkable
# property of the code rather than a promise in prose: delta is a closed-form
# function of S and snr, both fixed at generation time, and the tests below fail
# if it ever becomes a function of the drawn data or of a measured coverage.
#
# NO GATE IS ADDED FOR THE REFUSAL BASE RATE, deliberately. A gate that refused
# instances whose abstain-class came out inconveniently sized would BE coverage
# tuning, wearing a gate's clothes. The base rate is reported instead.
#
# Belief entropy as the per-position uncertainty statistic is NOT claimed here:
# that is Durand and Guedon, arXiv:1202.6545 (2012). Only the scoring protocol is.
# ============================================================================

def test_refuse_delta_is_anchored_at_generation_time_not_tuned():
    """delta is a closed form in S and snr. It may not see the data."""
    s_ = bed_h.S
    for snr in (0.5, 0.6, 0.7):
        q = np.full(s_, (1.0 - snr) / (s_ - 1))
        q[0] = snr
        expected = float(np.log(s_) + (q * np.log(q)).sum())   # ln S - H(one look)
        assert bed_h.refuse_delta_for(snr) == pytest.approx(expected, rel=1e-12)
    # two different draws at the same snr must give the same delta
    a = bed_h.HMM.draw(np.random.default_rng(SEED), snr=0.6)
    b = bed_h.HMM.draw(np.random.default_rng(SEED + 99), snr=0.6)
    assert bed_h.refuse_delta_for(a.snr) == bed_h.refuse_delta_for(b.snr)
    assert set(inspect.signature(bed_h.refuse_delta_for).parameters) <= {"snr", "s"}


def test_refusal_label_fires_at_both_extremes():
    """Anti-vacuity for the LABELLER, holding delta fixed so the anchor cannot
    move underneath it: a uniform belief must be refused, a one-hot belief must
    not. A labeller that always answers the same way dies here."""
    uni = np.full((5, bed_h.S), 1.0 / bed_h.S)
    hot = np.eye(bed_h.S)[np.arange(5) % bed_h.S]
    assert bed_h.refusal_label(uni, 0.25).all()
    assert not bed_h.refusal_label(hot, 0.25).any()


def test_the_anchor_self_normalises_and_that_is_why_it_cannot_be_tuned():
    """delta = ln S - H(one look) MOVES WITH snr, and the two degenerate ends
    show why that is the point rather than a defect.

      snr = 1/S : one look is worthless, H(one look) = ln S, the threshold is
                  ln S, and a belief that never leaves uniform refuses -- correct.
      snr = 1   : one look is perfect, H(one look) = 0, the threshold is 0, and
                  EVERY position refuses, because a history that merely matches a
                  perfect glance has told you nothing extra -- also correct.

    Neither end is reachable from SNR_BAND, and the point is that delta is never
    free to be chosen: it is pinned by snr, which is fixed before any trajectory."""
    assert bed_h.refuse_delta_for(1.0 / bed_h.S) == pytest.approx(0.0, abs=1e-12)
    assert bed_h.refuse_delta_for(1.0) == pytest.approx(np.log(bed_h.S), abs=1e-12)
    d = [bed_h.refuse_delta_for(x) for x in (0.5, 0.6, 0.7)]
    assert d[0] < d[1] < d[2], d


def test_the_refusal_class_is_alive_across_the_pre_registered_band():
    """MEASURED, and deliberately NOT gated. A gate on the base rate would be
    coverage tuning in a gate's clothes; this test only records that the class
    is neither empty nor everything, so the protocol has something to score."""
    rates = [bed_h.draw_instance(SEED + 1000 * k, "strong")[0].meta["refuse_base_rate"]
             for k in range(16)]
    assert 0.0 < min(rates) and max(rates) < 1.0, rates
    assert 0.005 < float(np.mean(rates)) < 0.5, np.mean(rates)


def test_instance_carries_the_refusal_target_and_it_is_exact():
    inst, hmm = bed_h.draw_instance(SEED, form="strong")
    delta = bed_h.refuse_delta_for(hmm.snr)
    assert inst.refuse.dtype == bool and inst.refuse.shape == (bed_h.T_LEN,)
    assert np.array_equal(inst.refuse, bed_h.refusal_label(inst.belief, delta))
    h = bed_h.belief_entropy(inst.belief)
    assert np.array_equal(inst.refuse, h >= np.log(bed_h.S) - delta)
    assert "refuse" not in bed_h.CONSUMER_KEYS      # it is a TARGET, not an input


def test_auroc_is_one_for_the_exact_belief_and_half_for_noise():
    """The scorer is useless unless it separates. The exact belief's own entropy
    DEFINES the label, so it must score 1.0; a random score must score ~0.5."""
    inst, hmm = bed_h.draw_instance(SEED, form="weak")
    sc = bed_h.refusal_scores(inst.belief, inst, bed_h.refuse_delta_for(hmm.snr))
    assert sc["auroc"] == pytest.approx(1.0, abs=1e-12), sc
    rng = np.random.default_rng(3)
    noise = rng.random((bed_h.T_LEN, bed_h.S))
    noise /= noise.sum(1, keepdims=True)
    sn = bed_h.refusal_scores(noise, inst, bed_h.refuse_delta_for(hmm.snr))
    assert abs(sn["auroc"] - 0.5) < 0.2, sn


def test_abstention_buys_accuracy_on_a_real_baseline():
    """The claim of the leg: dropping the positions a consumer is least sure of
    lowers its risk on what remains. Measured on window-4, which is a consumer
    of its own entropy and nothing else."""
    inst, hmm = bed_h.draw_instance(SEED, form="weak")
    w4 = hmm.window_forward(inst.obs, 4)
    sc = bed_h.refusal_scores(w4, inst, bed_h.refuse_delta_for(hmm.snr))
    assert sc["risk_at_0.8"] < sc["risk_at_1"], sc
    assert sc["risk_at_0.5"] < sc["risk_at_0.8"], sc
    assert 0.0 <= sc["base_rate"] <= 1.0


def test_prior_mean_P_is_the_strong_forms_no_identification_floor():
    """MEASURED 2026-09-14 and not in the pre-registration: scoring a STRONG-form
    arm against a random draw from the prior is too easy a bar.

    An arm that does NO in-context identification at all can still use the prior
    MEAN of P, and that is strictly better than any single other draw -- 0.0802 TV
    against 0.1118 on 32 instances. So the headroom an in-context identifier can
    actually win is 0.0802, not 0.1118, and the floor a later round must beat is
    the prior mean. This test pins the ORDER, which is what makes the floor the
    floor; the magnitudes are reported rather than asserted, because they are
    properties of the draw and not of the code."""
    Ps = [bed_h.HMM.draw(np.random.default_rng(SEED + 1000 * k)).P for k in range(16)]
    Pbar = bed_h.prior_mean_P(Ps)
    assert np.allclose(Pbar.sum(1), 1.0)
    mean_tv, draw_tv = [], []
    for k in range(12):
        inst, hmm = bed_h.draw_instance(SEED + 1000 * k, "strong")
        other = bed_h.HMM.draw(np.random.default_rng(900_000 + k))
        ex = inst.belief[bed_h.BURN_IN:]
        mean_tv.append(bed_h.tv(
            bed_h.forward_belief(Pbar, hmm.E, hmm.pi, inst.obs)[bed_h.BURN_IN:], ex))
        draw_tv.append(bed_h.tv(
            bed_h.forward_belief(other.P, hmm.E, hmm.pi, inst.obs)[bed_h.BURN_IN:], ex))
    assert np.mean(mean_tv) < np.mean(draw_tv), (np.mean(mean_tv), np.mean(draw_tv))
    assert np.mean(mean_tv) > 0.0, "the prior mean is not the true P"


def test_fit_rnn_returns_the_best_checkpoint_not_the_last():
    """FAIRNESS DEFECT FOUND 2026-09-14 AND FIXED HERE.

    The STRONG-form GRU-64 in the first full run OVERFIT: val_total bottomed at
    0.0870 around step 250 and rose to 0.1072 by step 2000, a last-quarter gain
    of -0.0414. `fit_rnn` returned the LAST model, so the skyline was scored on a
    checkpoint it had already beaten. A skyline that loses because it was allowed
    to overfit is worth exactly as little as one that was too small, and this
    round's whole brief was that the skyline must be fair.

    The fix keeps the best-val checkpoint. This test fails if it regresses: the
    returned model must score the BEST val loss the trace recorded, not the last.
    """
    tr = [bed_h.draw_instance(SEED + 1000 * k, "weak")[0] for k in range(8)]
    va = [bed_h.draw_instance(SEED + 500_000 + 1000 * k, "weak")[0] for k in range(4)]
    m, ev = bed_h.fit_rnn(tr, va, "weak", hidden=64, cell="gru", steps=120,
                          lr=0.05, batch=8, seed=1, log_every=20)
    assert "best_step" in ev
    x, y, c, n_pre = bed_h._tensors(va, "weak")
    with torch.no_grad():
        got = float(bed_h._losses(m, x, y, c, n_pre)[2])
    assert got == pytest.approx(ev["val_total_best"], rel=1e-5), (got, ev)
    assert ev["val_total_best"] <= ev["val_total_final"] + 1e-12


# ============================================================================
# PHASE 2. THE ABSTENTION IS THE CLAIM; THE BED IS NOT.
#
# docs/BED_H_PREREGISTRATION.md at e7f7704 struck the bed from the delta. The
# substrate is owned by arXiv:2512.22471 (Bayesian wind tunnels: a fresh
# Dirichlet HMM per sequence, supervised on the filtered posterior with
# forward-recursion ground truth) and arXiv:2605.20824 (six HMM families, three
# seeds, forced-state counterfactuals). Exact oracle labels are Rabiner 1989.
# Belief entropy as the per-position uncertainty statistic is Durand and Guedon,
# arXiv:1202.6545. NONE of that is claimed here. What is scored below is the
# ABSTENTION, against a posterior that is exactly known.
#
# PRE-REGISTERED BEFORE ANY MEASUREMENT, and not added to afterwards:
#   DELTA_SWEEP = (0.05, 0.1, 0.2)      the abstain thresholds
#   DEADBAND    = 0.02 nats             the "undefined" class, see below
#   PLANT_LEN   = 24, scored from step 8; 3 uninformative + 3 informative windows
#   PLANT_SNR_UNINF = 1/S  (pure noise)   PLANT_SNR_INF = 0.95
#
# WHY A DEADBAND. The spec names three definedness classes -- informative,
# uninformative, undefined -- and defines the first two by the oracle's H(b_t)
# against ln S - delta. It does not define the third, so it is defined here and
# pinned before measuring: UNDEFINED is the band |H(b_t) - (ln S - delta)| <
# DEADBAND, where the oracle's own call sits inside its own resolution. Rows in
# that band are excluded from precision and recall rather than scored either way.
# A later round may disagree with the width; it may not disagree silently.
# ============================================================================

DELTA_SWEEP = (0.05, 0.1, 0.2)
DEADBAND = 0.02

# PLANT GEOMETRY -- FIRST PIN FAILED, AND THE FAILURE IS RECORDED RATHER THAN
# OVERWRITTEN. The first pin was PLANT_LEN = 24, PLANT_SCORE_FROM = 8, chosen
# without deriving either from the bed's own relaxation time. Must-fire 1 came
# back 0.078 / 0.378 / 0.749 refused at delta = 0.05 / 0.10 / 0.20, against the
# required 0.95. The cause was measured, not guessed:
#
#   During an uninformative stretch the likelihood is flat across states, so it
#   cancels in the normalisation and the belief propagates as b_t = b_{t-1} P.
#   It therefore relaxes to the STATIONARY distribution at rate lambda_2, and its
#   entropy ceiling is H(pi) = 2.0400, not ln S = 2.0794. Measured trajectory
#   inside a 24-step stretch: H climbs 1.022 -> 2.006 and is still climbing at
#   the end. 1/(1 - lambda_2) = 11.7 steps, so 8 steps of settle is under one
#   relaxation time.
#
# The replacement is DERIVED from that measurement, not fitted to the pass line:
# a stretch must run several relaxation times before it can be called forgotten.
# With 1/(1 - lambda_2) in [10, 33] over the pre-registered lambda_2 band, 64
# steps of settle is 2 to 6 relaxation times, and the last 32 are scored.
PLANT_LEN = 96
PLANT_SCORE_FROM = 64
PLANT_SNR_UNINF = 1.0 / 8
PLANT_SNR_INF = 0.95
N_PLANTS_EACH = 1          # one of each fits once the stretches are this long


def test_phase2_constants_are_pinned_in_the_module():
    assert bed_h.DELTA_SWEEP == DELTA_SWEEP
    assert bed_h.DEADBAND == DEADBAND
    assert bed_h.PLANT_LEN == PLANT_LEN
    assert bed_h.PLANT_SCORE_FROM == PLANT_SCORE_FROM
    assert bed_h.PLANT_SNR_UNINF == pytest.approx(PLANT_SNR_UNINF)
    assert bed_h.PLANT_SNR_INF == PLANT_SNR_INF
    assert bed_h.N_PLANTS_EACH == N_PLANTS_EACH


def test_forward_belief_with_per_step_likelihood_matches_the_fixed_E_oracle():
    """Planting needs a time-varying emission, so the oracle must accept one.
    With the SAME E at every step it must reproduce the fixed-E answer exactly,
    or the planted labels are not the labels the rest of the bed reports."""
    hmm = bed_h.HMM.draw(np.random.default_rng(SEED))
    obs = hmm.sample(np.random.default_rng(SEED + 1), 64)[1]
    lik = hmm.E[:, obs].T                      # (T, S)
    got = bed_h.forward_belief_lik(hmm.P, hmm.pi, lik)
    assert np.abs(got - hmm.forward(obs)).max() < 1e-14


def test_planted_instance_keeps_the_label_exact_and_records_its_plants():
    inst, hmm = bed_h.draw_planted(SEED, form="weak")
    assert inst.belief.shape == (bed_h.T_LEN, bed_h.S)
    assert np.allclose(inst.belief.sum(1), 1.0)
    kinds = {p["kind"] for p in inst.meta["plants"]}
    assert kinds == {"uninformative", "informative"}
    assert len(inst.meta["plants"]) == 2 * N_PLANTS_EACH
    for p in inst.meta["plants"]:
        assert p["end"] - p["start"] == PLANT_LEN
        assert p["start"] >= bed_h.BURN_IN


def test_mustfire_1_planted_uninformative_stretch_is_refused_at_least_95pct():
    """MUST-FIRE 1, and it must be able to fail -- it DID fail on the first plant
    geometry, see the header. It is scored only where the anchor is REACHABLE:
    an instance whose H(pi) < ln S - delta can never have an uninformative belief
    at that delta, however long the stretch, because the belief cannot be less
    informative than the stationary distribution. That reachability is a property
    of the bed and is reported per delta, not quietly excluded."""
    for delta in DELTA_SWEEP:
        r = bed_h.plant_refusal_rate(n=24, form="weak", delta=delta, seed=SEED)
        if r["reachable_frac"] == 0.0:
            continue                      # unreachable at this delta; reported
        assert r["uninformative_reachable"] >= 0.95, (delta, r)


def test_mustfire_2_planted_informative_stretch_is_refused_at_most_5pct():
    for delta in DELTA_SWEEP:
        r = bed_h.plant_refusal_rate(n=24, form="weak", delta=delta, seed=SEED)
        assert r["informative"] <= 0.05, (delta, r)


def test_mustfire_3_always_answer_scores_zero_on_the_refused_uninformative_cell():
    """MUST-FIRE 3. Asserted, not stated. This is the control that proves the
    cell measures something: a model that never refuses cannot score in it."""
    inst, hmm = bed_h.draw_planted(SEED, form="weak")
    resolvable = 0
    for delta in DELTA_SWEEP:
        t = bed_h.abstention_table(inst.belief, inst, delta, always_answer=True)
        # the control, by construction: a model that never refuses cannot score
        # in ANY refused cell. This is what makes the cell mean something.
        assert t["uninformative"]["refused"] == 0
        assert t["informative"]["refused"] == 0
        assert t["undefined"]["refused"] == 0
        t2 = bed_h.abstention_table(inst.belief, inst, delta, always_answer=False)
        if not t2["resolvable"]:
            continue          # anchor inside the deadband of H(pi); reported
        resolvable += 1
        assert t2["uninformative"]["refused"] > 0, (delta, t2)
    assert resolvable >= 1, "no delta in the sweep is resolvable on this bed"


def test_abstention_table_is_3x2_and_totals_every_scored_position():
    inst, hmm = bed_h.draw_planted(SEED, form="weak")
    t = bed_h.abstention_table(inst.belief, inst, 0.1)
    assert t["resolvable"]
    assert set(t) >= {"informative", "uninformative", "undefined",
                      "precision", "recall", "n_scored"}
    tot = sum(t[k][a] for k in ("informative", "uninformative", "undefined")
              for a in ("answered", "refused"))
    assert tot == t["n_scored"] == bed_h.T_LEN - bed_h.BURN_IN
    # the oracle's own entropy defines the label, so the oracle is perfect on it
    assert t["precision"] == pytest.approx(1.0)
    assert t["recall"] == pytest.approx(1.0)


def test_selective_risk_dominance_is_measured_against_always_answer():
    """Abstention earns only if the curve beats the flat always-answer line at
    MATCHED coverage. A predictor that abstains at random must NOT dominate."""
    inst, hmm = bed_h.draw_planted(SEED, form="weak")
    good = hmm.window_forward(inst.obs, 4)
    d = bed_h.selective_risk(good, inst)
    assert d["risk_at_1"] > 0
    assert d["dominates_at_0.8"] is True
    rng = np.random.default_rng(5)
    noise = rng.random((bed_h.T_LEN, bed_h.S))
    noise /= noise.sum(1, keepdims=True)
    dn = bed_h.selective_risk(noise, inst)
    assert dn["dominates_at_0.8"] is False, dn


# ============================================================================
# THE DERIVED-LEVEL DEFECT, FOUND BY A THEORY AUDIT AND REPRODUCED HERE.
#
# `tv()` is the bed's pinned metric and it is an ARITHMETIC mean.
# `decay_per_step_lyapunov` is a GEOMETRIC-mean rate. The committed
# `w4_derived_from_lyapunov` multiplied the quenched rate cubed onto an annealed
# level, mixing the two. Reproduced on 6 strong draws with the unmodified
# `gate_report()`:
#
#   w4_measured / w4_derived_from_lyapunov : mean 7.22, min 2.52, max 17.11
#   rate_measured / rate_lyapunov          : mean 1.626, min 1.401, max 2.191, 6/6 > 1
#   w4_measured / w4_derived_from_lambda2  : mean 0.300 -- lambda_2 too HIGH by 3.3x
#
# BOTH named predictors miss, so a derived window-4 LEVEL is unsatisfiable as
# coded. The deeper reason decides the fix: Ye-Ma-Qian's eq. (10) introduces its
# constant C as "some constant" and never estimates it, conceding the explicit
# gap estimate is "either too loose or still difficult to find". THE THEORY
# SUPPLIES A SLOPE AND NO LEVEL. Deriving a level sets that unbounded constant
# silently to err[1]. So the level is DROPPED and the slope is kept, fitted on
# L >= 4 where the geometric law holds, against the geometric-mean error.
# ============================================================================

RATE_FIT_MIN_L = 4


def test_the_derived_w4_level_is_gone():
    """The level the mathematics does not provide must not be reported."""
    inst, hmm = bed_h.draw_instance(SEED, form="strong")
    g = bed_h.gate_report(inst, hmm)
    assert "w4_derived_from_lyapunov" not in g
    assert "w4_derived_from_lambda2" not in g
    assert "w4_measured" in g                      # the measurement stays


def test_geometric_window_error_is_reported_beside_the_arithmetic_one():
    """Jensen fixes the order, so this is a check on the code, not on the bed."""
    inst, hmm = bed_h.draw_instance(SEED, form="strong")
    g = bed_h.gate_report(inst, hmm)
    assert set(g["geo_window_err"]) == set(bed_h.L_GRID)
    for L in bed_h.L_GRID:
        assert g["geo_window_err"][L] <= g["window_err"][L] + 1e-12, L


def test_the_rate_is_fitted_on_L_at_least_4_and_on_the_geometric_error():
    inst, hmm = bed_h.draw_instance(SEED, form="strong")
    g = bed_h.gate_report(inst, hmm)
    assert g["rate_fit_L"], g
    assert min(g["rate_fit_L"]) >= RATE_FIT_MIN_L, g["rate_fit_L"]
    assert 0.0 < g["decay_per_step_quenched"] < 1.0


def test_the_quenched_rate_matches_the_lyapunov_gap_and_lambda2_does_not():
    """The slope IS what the theory supports, and it holds. lambda_2 of P does
    not, which is the falsification this bed is in a position to make."""
    ratios_q, ratios_l = [], []
    for k in range(8):
        inst, hmm = bed_h.draw_instance(SEED + 1000 * k, form="strong")
        g = bed_h.gate_report(inst, hmm)
        ratios_q.append(g["decay_per_step_quenched"] / g["decay_per_step_lyapunov"])
        ratios_l.append(g["decay_per_step_quenched"] / hmm.lambda2)
    assert abs(float(np.mean(ratios_q)) - 1.0) < 0.15, ratios_q
    assert float(np.mean(ratios_l)) < 0.7, ratios_l


def test_entropy_rate_comes_from_the_filter_normalisers():
    """h(Y) = -mean(log c_t) straight out of the forward recursion, three lines,
    no extra pass. Jurgens and Crutchfield, arXiv:2102.10487."""
    hmm = bed_h.HMM.draw(np.random.default_rng(SEED))
    obs = hmm.sample(np.random.default_rng(SEED + 1), 4000)[1]
    h = bed_h.entropy_rate(hmm, obs)
    assert 0.0 < h < np.log(bed_h.S)              # between certainty and uniform
    # it must equal the cross-entropy of the filter's own one-step predictions
    b = hmm.forward(obs)
    ce = -np.mean([np.log(((b[t - 1] @ hmm.P) @ hmm.E)[obs[t]])
                   for t in range(1, len(obs))])
    assert h == pytest.approx(ce, rel=1e-3), (h, ce)


def test_lambda1_equals_minus_the_entropy_rate_as_an_exact_identity():
    """A FREE correctness check on BOTH the filter and the QR Lyapunov code:
    the top Lyapunov exponent of the un-normalised forward recursion is exactly
    minus the observed-process entropy rate. If either side is wrong this breaks,
    and nothing else in the suite would have caught it."""
    for seed in (SEED, SEED + 1000, SEED + 2000):
        hmm = bed_h.HMM.draw(np.random.default_rng(seed))
        obs = hmm.sample(np.random.default_rng(seed + 7), 6000)[1]
        lam1, lam2 = bed_h.lyapunov_exponents(hmm, obs)
        h = bed_h.entropy_rate(hmm, obs)
        assert abs(lam1 + h) < 1e-3, (lam1, h, abs(lam1 + h))


def test_dimension_upper_bound_is_below_the_ambient_simplex():
    """d <= h / chi. Reported as an UPPER BOUND, never as a measured dimension."""
    hmm = bed_h.HMM.draw(np.random.default_rng(SEED))
    obs = hmm.sample(np.random.default_rng(SEED + 3), 4000)[1]
    d = bed_h.dimension_upper_bound(hmm, obs)
    assert 0.0 < d["d_ub"] < bed_h.S - 1, d
    assert d["chi"] > 0.0
    assert d["h"] == pytest.approx(bed_h.entropy_rate(hmm, obs), rel=1e-9)


def test_selective_risk_breaks_ties_at_random_not_by_index():
    """REGRESSION. `np.argsort(kind="mergesort")` is STABLE, so a predictor with
    CONSTANT entropy -- the uniform negative control -- had its "most confident c"
    silently become "the first c of the timeline". On a planted instance that
    correlates with where the stretches sit, and the control came back dominating
    at 1.00 of instances, which is a bug in the scorer and not a result.

    With ties broken at random the control's risk curve must be FLAT: abstaining
    on a coin flip cannot lower the expected risk on what remains."""
    inst, hmm = bed_h.draw_planted(SEED, form="weak")
    flat = np.full((bed_h.T_LEN, bed_h.S), 1.0 / bed_h.S)
    d = bed_h.selective_risk(flat, inst)
    assert d["dominates_at_0.8"] is False
    assert d["dominates_at_0.9"] is False
    # flat within a few percent of the always-answer risk, in BOTH directions
    for c in (0.9, 0.8, 0.5):
        assert abs(d["risk_at_%g" % c] - d["always_answer"]) < 0.10 * d["always_answer"], (c, d)

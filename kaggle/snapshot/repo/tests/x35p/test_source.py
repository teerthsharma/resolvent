"""X35' (a) the exact source solve and (b) time-reversal localization, and the
must-fires that decide them.

CEQ_V15_2_DELTA.md:

    (a) `h_hat = (I - A) r`.  `[RUN 8.9e-16, two sources]`
    (b) `W^T r`, with Wiener deconvolution when the noise is colored.
        `[RUN: exact at sd 0.3]`

    MUST-FIRES (L-SCOPE, production-path residuals)
      1. planted SINGLE source
      2. planted TWO sources -- superposition
      3. NO plant -- flat residual at the calibrated false-alarm rate

    KILLS
      - Any KK or CRB number quoted before its must-fire => STRUCK.
      - Detector worse than the adjoint at high noise => the exact inverse is
        retired to the noiseless regime, stated.

WHERE THE PROPAGATOR COMES FROM, AND WHY THIS IS NOT A REIMPLEMENTATION.
`ceq/beds/bed_k.py` adds the planted latent `u` to `z` DIRECTLY, after the
kernel -- V15_X35A_RESIDUAL.md §9 states this as a limit in those words. With
the oracle visible model the production-path residual is therefore

    r  =  u + observation noise,

i.e. the forward map from the hidden source to the residual field is the
IDENTITY and the resolvent's `A` is exactly 0. That is measured below
(`test_production_path_propagator_is_the_identity`) rather than assumed, and it
has a consequence worth stating before any number is read: on the production
path the exact solve `(I - A) r` reduces to `r` and must-fires 1 and 2 are
recoveries at float64 rounding -- which is what `8.9e-16` is, and why the delta
calls it "not a convergence tolerance".

An inverse problem with an identity forward map has no inverse to test, and two
estimators that are both the identity cannot be compared. So the sweep runs on
the PROPAGATED field: the SAME production-path source `bed["plant"]["u"]` and
the SAME production-path observation stream (`ceq.x35.residual.observe`, same
salt, same seed), with the architecture's own carrier between them --
`W = occupancy(A, n) = sum_k A^k` built from the production-path kernel
`bed_k.kernel_matrix`. Every component is the shipped one; the composition step
is the only new line and it is named (`ceq/x35p/source.py::wave_field`).

MATCHED TO X35a ON PURPOSE. Bed `delay d=5, n=128`, one latent at true index 36
with magnitude 1.0, calibration seeds `[0, 2000)`, evaluation seeds from `1e6`
-- the same configuration and the same disjoint blocks
V15_X35A_RESIDUAL.md reports, so the comparison against that node's onset
detector is direct rather than approximate. `run_detect` returns the whole
residual field for exactly this reason and
`test_scores_the_field_the_x35_detector_read` asserts the field is bitwise the
same one.

NO CRB NUMBER APPEARS IN THIS FILE. CEQ_V15_2_DELTA.md strikes any CRB or KK
number quoted before its own must-fire, and both instruments belong to a
parallel node. What is reported here is RAW localization error -- a hit rate and
a mean absolute index error -- with no floor beside it and no claim of
optimality, which would be a CRB claim by implication.

Run order: this file is authored and shown RED before `ceq/x35p/source.py`
exists. V15_X35P_SOURCE.md carries that transcript.
"""
from __future__ import annotations

import numpy as np
import pytest

from ceq.beds import bed_k
from ceq.x35 import residual as rx
from ceq.x35p import source as sx

# The X35a configuration, verbatim.
CFG = dict(n=128, kind="delay", params=dict(d=5))
T_STAR, MAG = 36, 1.0
CAL = range(0, 2000)          # calibration block, X35a's
EVAL = 1_000_000              # evaluation block start, X35a's
ALPHA = 0.01


def _plant(*indices):
    return [dict(index=i, magnitude=MAG) for i in indices]


# --------------------------------------------------------------------------
# the carrier: what (I - A) and W are, and what makes the inverse cheap
# --------------------------------------------------------------------------


def test_occupancy_is_the_exact_two_sided_inverse():
    """`CEQ.Occupancy.occupancy_eq_inverse_of_nilpotent` and
    `CEQ.Nilpotent.occupancy_is_exact_inverse`, measured. A is strictly lower
    triangular, so A^n = 0 and the finite sum IS the inverse -- no truncation
    error to bound, which is the hypothesis Lean #15 composes with."""
    c = sx.carrier(n=128, kind="delay", params=dict(d=5))
    A, W, I = c["A"], c["W"], np.eye(128)
    assert np.all(np.triu(A) == 0.0), "A must be strictly lower triangular"
    assert np.max(np.abs(np.linalg.matrix_power(A, 128))) == 0.0
    left = np.max(np.abs((I - A) @ W - I))
    right = np.max(np.abs(W @ (I - A) - I))
    print(f"[CARRIER] ||(I-A)W - I||_max={left:.3e} ||W(I-A) - I||_max={right:.3e} "
          f"cond(W)={np.linalg.cond(W):.4g}")
    assert left <= 1e-12 and right <= 1e-12


def test_the_inverse_is_a_first_order_difference_and_the_forward_map_is_not():
    """LEAN #15's content, as a count.

    The claim is NOT `M * M^-1 = I`, which every invertible matrix satisfies.
    It is that the inverse of THIS forward map is `I - A` -- one application of
    A, no powers -- so it is as sparse as A plus a diagonal, while the forward
    map `sum_k A^k` fills in every path of every length. A solve is O(n^3) (or
    an iteration with a conditioning question); this is O(nnz)."""
    c = sx.carrier(n=128, kind="delay", params=dict(d=5))
    A, W = c["A"], c["W"]
    nnz_A = int((A != 0).sum())
    nnz_inv = int(((np.eye(128) - A) != 0).sum())
    nnz_fwd = int((W != 0).sum())
    print(f"[FIRST-ORDER] nnz(A)={nnz_A} nnz(I-A)={nnz_inv} nnz(W=sum A^k)={nnz_fwd} "
          f"fill-in={nnz_fwd - nnz_inv:+d}")
    assert nnz_inv <= nnz_A + 128, "the inverse may add a diagonal and nothing else"
    assert nnz_fwd > nnz_inv, "the forward map must fill in, or 'first order' says nothing"


def test_exact_solve_never_forms_the_forward_map():
    """`exact_source` takes A, not W. A source solve that needed W would have
    already paid for the thing the closed form is supposed to avoid."""
    c = sx.carrier(n=128, kind="delay", params=dict(d=5))
    r = np.arange(128, dtype=np.float64)
    assert np.allclose(sx.exact_source(c["A"], r), r - c["A"] @ r, rtol=0, atol=0)


# --------------------------------------------------------------------------
# MUST-FIRE 1 and 2 -- on the production-path residual
# --------------------------------------------------------------------------


def test_production_path_propagator_is_the_identity():
    """The scope statement, measured before anything is read off it: with the
    oracle visible model the production-path residual IS the source plus the
    observation noise, so `A = 0` on that path and `(I - A) r = r`."""
    bed, r = rx.run_residual(EVAL, noise_sd=0.0, plant=_plant(T_STAR), **CFG)
    err = float(np.max(np.abs(r - bed["plant"]["u"])))
    print(f"[SCOPE] production-path ||r - u||_inf = {err:.3e} (propagator = I, A = 0)")
    assert err <= 1e-14


def test_exact_solve_recovers_one_planted_source():
    """MUST-FIRE 1, both readings: on the production-path residual (A = 0) and
    on the propagated field (A = the carrier's own hop)."""
    bed, r = rx.run_residual(EVAL, noise_sd=0.0, plant=_plant(T_STAR), **CFG)
    u = bed["plant"]["u"]
    flat = float(np.max(np.abs(sx.exact_source(np.zeros((128, 128)), r) - u)))

    f = sx.wave_field(EVAL, noise_sd=0.0, plant=_plant(T_STAR), **CFG)
    prop = float(np.max(np.abs(sx.exact_source(f["A"], f["y"]) - f["h"])))
    print(f"[MUST-FIRE 1] one source at {T_STAR}: production-path err={flat:.3e} "
          f"propagated-field err={prop:.3e}")
    assert flat <= 1e-14 and prop <= 1e-14


def test_exact_solve_recovers_two_planted_sources_superposition():
    """MUST-FIRE 2. The delta's `[RUN 8.9e-16, two sources]`.

    Both sources are recovered, not merely their sum: the solve is exact
    index-by-index, so the per-source onsets 36 and 80 are both readable off
    `h_hat` -- which is the thing X35a's Shewhart onset structurally cannot do
    (it calls the earliest source only, V15_X35A_RESIDUAL.md §4)."""
    bed, r = rx.run_residual(EVAL, noise_sd=0.0, plant=_plant(36, 80), **CFG)
    u = bed["plant"]["u"]
    flat = float(np.max(np.abs(sx.exact_source(np.zeros((128, 128)), r) - u)))

    f = sx.wave_field(EVAL, noise_sd=0.0, plant=_plant(36, 80), **CFG)
    h_hat = sx.exact_source(f["A"], f["y"])
    prop = float(np.max(np.abs(h_hat - f["h"])))
    assert bed["plant"]["indices"] == [36, 80]
    # both switch-ons are visible in the recovered source, exactly.
    assert np.all(h_hat[:36] == 0.0) or np.max(np.abs(h_hat[:36])) <= 1e-14
    print(f"[MUST-FIRE 2 / superposition] sources at {bed['plant']['indices']}: "
          f"production-path err={flat:.3e} propagated-field err={prop:.3e}")
    assert flat <= 1e-14 and prop <= 1e-14


def test_scores_the_field_the_x35_detector_read():
    """L-SCOPE. The field this node's estimators consume is bitwise the field
    X35a's `run_detect` returned, not a separately generated draw."""
    cfg = dict(CFG, noise_sd=0.05)
    det = rx.run_detect(EVAL, threshold=0.0382564, plant=_plant(T_STAR), **cfg)
    _, r = rx.run_residual(EVAL, plant=_plant(T_STAR), **cfg)
    assert np.array_equal(det["residual"], r)


# --------------------------------------------------------------------------
# MUST-FIRE 3 -- NO plant => flat residual. THE X35 KILL, AND IT BINDS FIRST.
# --------------------------------------------------------------------------


def test_no_plant_residual_is_flat():
    """MUST-FIRE 3, on the production-path residual. If this fails the correct
    output is X35' VOID with the measured non-flatness -- not a tuned
    threshold. Bars are the null sds `flatness` documents: `1/sqrt(n-1)` for
    the correlation, `sqrt(2/(runs*n/2))` for the half-ratio; 3 sd each."""
    flat = rx.flatness(range(EVAL, EVAL + 200), noise_sd=0.05, **CFG)
    corr_sd = 1.0 / np.sqrt(128 - 1)
    ratio_sd = np.sqrt(2.0 / (200 * 128 / 2))
    print(f"[MUST-FIRE 3 / flatness] corr={flat['corr']:+.4f} (3sd={3*corr_sd:.4f}) "
          f"half_ratio={flat['half_ratio']:.4f} (3sd={3*ratio_sd:.4f}) "
          f"mean_energy={flat['mean_energy']:.6g} (sd^2=0.0025)")
    assert abs(flat["corr"]) < 3 * corr_sd
    assert abs(flat["half_ratio"] - 1.0) < 3 * ratio_sd


def test_no_plant_false_alarm_rate_matches_the_calibrated_alpha():
    """MUST-FIRE 3's other half. Threshold from the calibration block, rate
    from a DISJOINT evaluation block (MISTAKES.md M-2)."""
    cfg = dict(CFG, noise_sd=0.05)
    cal = rx.calibrate(CAL, ALPHA, **cfg)
    ev = range(EVAL, EVAL + 2000)
    assert set(CAL).isdisjoint(ev)
    far = rx.false_alarm_rate(ev, cal["threshold"], **cfg)
    print(f"[MUST-FIRE 3 / FAR] tau={cal['threshold']:.7g} measured FAR={far['rate']:.4f} "
          f"({far['alarms']}/{far['runs']}) vs alpha={ALPHA}")
    assert far["rate"] < 3 * ALPHA


def test_the_raw_adjoint_is_structurally_not_flat_under_no_plant():
    """Why `W^T r` cannot be read by an onset comparator, measured rather than
    asserted: `W^T` is a REVERSE cumulative sum along the carrier's hop
    lattice, so under no plant its output energy DECREASES monotonically with
    index. A first-crossing rule on that field calls index 0 at every seed and
    every noise level. This is a property of the operator, not of the noise,
    and it is the reason instrument (b) ships normalized (§ the column-norm
    correction) rather than raw."""
    c = sx.carrier(**CFG)
    E, En = [], []
    for s in range(EVAL, EVAL + 200):
        y = sx.wave_field(s, noise_sd=0.05, plant=None, **CFG)["y"]
        E.append(sx.adjoint_source(c["W"], y, normalize=False) ** 2)
        En.append(sx.adjoint_source(c["W"], y, normalize=True) ** 2)
    idx = np.arange(128, dtype=np.float64)
    raw = float(np.corrcoef(idx, np.mean(E, axis=0))[0, 1])
    norm = float(np.corrcoef(idx, np.mean(En, axis=0))[0, 1])
    print(f"[ADJOINT FLATNESS] raw W^T corr={raw:+.4f}  column-normalized corr={norm:+.4f}")
    assert raw < -0.9, "the raw adjoint must be measurably non-flat, or the claim is unbacked"
    assert abs(norm) < 3.0 / np.sqrt(127), "normalization must restore flatness"


# --------------------------------------------------------------------------
# THE COMPARISON, AND ITS KILL
# --------------------------------------------------------------------------


def _point_hits(sd, seeds, phi=0.0):
    """Point-source localization by peak of the recovered source -- the task
    time reversal is FOR (Fink): a source at a single index, localized by where
    the back-propagated field focuses. Returns the hit rate of each estimator.

    The step-onset plant `bed_k` writes is an EXTENDED source, and an extended
    source has no focus to find; the onset sweep below is the reading on that
    one. Both are reported because they answer different questions and both
    decide the same kill.
    """
    c = sx.carrier(**CFG)
    W, A = c["W"], c["A"]
    cov = sx.ar1_cov(128, sd, phi)
    hits = dict(exact=0, adjoint=0, wiener_white=0, wiener_colored=0)
    n_seeds = 0
    for s in seeds:
        p = int(np.random.default_rng([sx.POINT_SALT, s]).integers(0, 128))
        h = np.zeros(128)
        h[p] = 1.0
        y = W @ h + sx.ar1_noise(s, 128, sd, phi)
        hits["exact"] += int(np.argmax(np.abs(sx.exact_source(A, y))) == p)
        hits["adjoint"] += int(np.argmax(np.abs(sx.adjoint_source(W, y))) == p)
        hits["wiener_white"] += int(
            np.argmax(np.abs(sx.wiener_source(W, y, sd * sd, 1.0))) == p)
        hits["wiener_colored"] += int(
            np.argmax(np.abs(sx.wiener_source(W, y, sd * sd, 1.0, noise_cov=cov))) == p)
        n_seeds += 1
    return {k: v / n_seeds for k, v in hits.items()}


def test_exact_solve_is_the_winner_at_zero_noise():
    """The half of the kill that is NOT a retirement. At sd = 0 the closed form
    is exact to float64 and localizes every seed; it is a subtraction, so it
    involves no linear solve and has no conditioning question."""
    hits = _point_hits(0.0, range(EVAL, EVAL + 200))
    print(f"[SWEEP sd=0.00] " + " ".join(f"{k}={v:.3f}" for k, v in hits.items()))
    assert hits["exact"] == 1.0


@pytest.mark.parametrize("sd", [0.2, 0.3])
def test_adjoint_beats_the_exact_solve_at_high_noise(sd):
    """The delta's kill condition, as a test. `[RUN: exact at sd 0.3]` for (b).

    A test that can only pass is worth nothing, so note what would falsify
    this: if the exact inverse held up at sd 0.3 the assertion below fails and
    the retirement is not called. It is asserted in the direction the delta
    predicts because the mechanism is known -- `(I - A)` DIFFERENCES the
    observation noise (variance x(1+gamma^2)) while `W^T` AVERAGES it over the
    carrier's reachable set -- but the numbers are measured, not assumed."""
    hits = _point_hits(sd, range(EVAL, EVAL + 200))
    print(f"[SWEEP sd={sd:.2f}] " + " ".join(f"{k}={v:.3f}" for k, v in hits.items()))
    assert hits["adjoint"] > hits["exact"] + 0.25


def test_wiener_deconvolution_needs_the_colored_covariance():
    """Instrument (b)'s qualifier: "with Wiener deconvolution when the noise is
    colored". AR(1) noise at phi = 0.7, marginal sd 0.3. The estimator handed
    the CORRECT covariance must beat the same estimator handed `sigma^2 I`, and
    at phi = 0 the two must coincide exactly -- otherwise the covariance
    argument is decoration."""
    col = _point_hits(0.3, range(EVAL, EVAL + 200), phi=0.7)
    white = _point_hits(0.3, range(EVAL, EVAL + 200), phi=0.0)
    print(f"[COLORED phi=0.7 sd=0.3] " + " ".join(f"{k}={v:.3f}" for k, v in col.items()))
    print(f"[WHITE   phi=0.0 sd=0.3] " + " ".join(f"{k}={v:.3f}" for k, v in white.items()))
    assert col["wiener_colored"] > col["wiener_white"]
    assert white["wiener_colored"] == white["wiener_white"]


def test_onset_sweep_on_the_production_path_source_favours_wiener_above_zero_noise():
    """The same kill on the extended source `bed_k` actually plants, localized
    by the SAME memoryless comparator X35a uses (`residual.detect`), with each
    estimator's threshold calibrated on its own no-plant runs. Paired on seeds,
    so the comparison is not swamped by the draw."""
    rows = sx.onset_sweep([0.0, 0.3], t_star=T_STAR, magnitude=MAG,
                          cal_seeds=range(0, 400), eval_seeds=range(EVAL, EVAL + 200),
                          alpha=ALPHA, **CFG)
    for r in rows:
        print(f"[ONSET SWEEP sd={r['noise_sd']:.2f}] exact_mae={r['exact_mae']:.3f} "
              f"wiener_mae={r['wiener_mae']:.3f} paired={r['paired_diff']:+.3f} "
              f"exact_wins={r['exact_wins']} ties={r['ties']} wiener_wins={r['wiener_wins']}")
    zero = next(r for r in rows if r["noise_sd"] == 0.0)
    high = next(r for r in rows if r["noise_sd"] == 0.3)
    assert zero["exact_mae"] == 0.0
    assert high["wiener_mae"] < high["exact_mae"]

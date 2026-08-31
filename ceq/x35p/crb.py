"""X35-prime (d) -- the onset localization floor, and the finding that it is
not a Cramer-Rao bound.

CEQ_V15_2_DELTA.md (d) asks for a "Cramer-Rao localization floor, per bed,
printed beside every onset CI". THIS MODULE DOES NOT PRODUCE ONE, and the
reason is a property of the bed rather than of the budget.

WHAT THE BED'S RESIDUAL IS. With the oracle visible model, `ceq.x35.residual`
gives `r = observation noise + planted latent` exactly, and `bed_k._plant_field`
drives the plant with `u_i = magnitude * g_i` for `i >= index`. So

    r_i ~ N(0, sigma^2)                    i <  t
    r_i ~ N(0, sigma^2 + m^2)              i >= t

independent across i, with `t` the onset. This is a VARIANCE change-point at a
DISCRETE index. `measure_residual_variance_profile` checks that against the
production path rather than assuming it (MISTAKES.md M-18: a bound registered
against a quantity whose distribution in the corpus was never checked).

WHY THE CLASSICAL CRB DOES NOT EXIST HERE, in two independent ways.

1. `t` is an integer. The Cramer-Rao inequality needs `d/d theta log p` to
   exist; a discrete parameter has no score and no Fisher information, so
   there is nothing to invert. Embedding `t` in the reals as a continuous
   `tau` with the jump kept abrupt does not help: `v_i(tau)` is then a STEP in
   `tau`, its derivative is 0 almost everywhere, the Fisher information is
   identically 0 and the bound is `+inf`. A bound of `+inf` is satisfied by
   every estimator, which is MISTAKES.md V-10 -- a gate with no rejection
   region -- reached from the bound side.

2. The cited prior art does not transfer. The delay-estimation CRB
   (`PRIOR_ART` in the delta) is derived for a KNOWN waveform `s(t - tau)`
   observed in noise, where the Fisher information is the mean-square
   bandwidth of `s`. This bed's plant is an iid Gaussian DRIVE, not a known
   waveform: `E[r] = 0` under every onset, the whole signal lives in the
   second moment, and there is no `s'` to integrate. The bandwidth form is
   not merely hard to evaluate here, it has no argument to take.

WHAT SMOOTHING BUYS, AND WHY IT IS NOT AN ANSWER. Replace the step by a ramp
of width `w` and the Fisher information becomes finite (`fisher_information_
ramp`). But `crb(w) -> 0` as `w -> 0` while the abrupt reading is `+inf`: the
two natural regularizations of the same object converge to the two opposite
vacuous answers, so no limit defines the bound. Worse, at `w << 1` the value
depends on where `tau` falls BETWEEN samples -- measured at over 1e5 spread in
`tests/x35p`. A distance-to-CRB computed this way would be a statement about
`w` and about sub-sample phase, not about the detector.

WHAT SHIPS INSTEAD: the ZIV-ZAKAI bound (`zzb`). It needs no differentiability
because it is built from binary hypothesis tests rather than from a score, it
is valid for a discrete parameter, and for this bed the pairwise test has a
closed form -- deciding `chi^2_h` at variance `v0` against `v1`. Its own
calibration is that at `m = 0` no test beats a coin and the bound must equal
the prior variance `(n^2-1)/12` exactly; that is asserted, not hoped for.

float64 throughout. Nothing trains.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import chi2

from ceq.x35 import residual

__all__ = [
    "measure_residual_variance_profile", "abrupt_loglik",
    "fisher_information_ramp", "pmin_binary", "zzb",
    "posterior_mean_onset", "noise_sweep",
]

#: The sweep's bed. The oracle residual does not depend on the kernel at all
#: (`ceq/x35/residual.py`, asserted in tests/x35), so the kind is fixed rather
#: than swept -- sweeping it would produce identical rows and read as coverage.
BED = dict(kind="delay", params=dict(d=4))


# --------------------------------------------------------------------------
# the model the bound is derived for, measured on the production path
# --------------------------------------------------------------------------


def measure_residual_variance_profile(*, onset: int, sigma: float,
                                      magnitude: float, n: int, runs: int,
                                      seed0: int) -> dict:
    """Pooled `E[r_i^2]` before and after the onset, over `runs` production
    residuals. The bound below is derived for `sigma^2` and `sigma^2 + m^2`;
    this is the check that the corpus supplies them."""
    pre, post = [], []
    for k in range(runs):
        _, r = residual.run_residual(seed0 + k, n=n, noise_sd=sigma,
                                     plant=dict(index=onset, magnitude=magnitude),
                                     **BED)
        pre.append(r[:onset] ** 2)
        post.append(r[onset:] ** 2)
    return dict(pre_var=float(np.mean(pre)), post_var=float(np.mean(post)),
                model_pre=sigma ** 2, model_post=sigma ** 2 + magnitude ** 2,
                runs=int(runs), samples_pre=int(runs * onset),
                samples_post=int(runs * (n - onset)))


# --------------------------------------------------------------------------
# the two ways the classical CRB fails
# --------------------------------------------------------------------------


def abrupt_loglik(r: np.ndarray, tau: float, *, sigma: float,
                  magnitude: float) -> float:
    """Log-likelihood at a CONTINUOUS onset `tau`, with the jump kept abrupt:
    index `i` carries the plant iff `i > tau`. Piecewise constant in `tau` by
    construction, which is the point -- the score does not exist."""
    v0 = sigma ** 2
    v1 = sigma ** 2 + magnitude ** 2
    t = int(math.floor(tau)) + 1
    v = np.where(np.arange(r.size) >= t, v1, v0)
    return float(np.sum(-0.5 * np.log(2.0 * np.pi * v) - r ** 2 / (2.0 * v)))


def fisher_information_ramp(*, tau: float, width: float, sigma: float,
                            magnitude: float, n: int) -> dict:
    """Fisher information for a continuous onset under a LOGISTIC ramp of
    width `width`, and the CRB it implies.

    `v_i(tau) = sigma^2 + m^2 * phi((i - tau)/w)`, zero-mean independent
    Gaussians, so `I(tau) = sum_i (1/2) (dv_i/dtau)^2 / v_i^2`.

    `width = 0` is the bed's own abrupt onset: `dv/dtau = 0` almost
    everywhere, `I = 0`, `crb = +inf`. That value is returned rather than
    raising, because "+inf" IS the finding -- a bound no estimator can cross.
    """
    i = np.arange(n, dtype=np.float64)
    if width == 0.0:
        v = np.where(i > tau, sigma ** 2 + magnitude ** 2, sigma ** 2)
        return dict(fisher=0.0, crb=math.inf, width=0.0, variance=v)
    u = (i - tau) / width
    phi = 1.0 / (1.0 + np.exp(-u))
    dphi = phi * (1.0 - phi)
    v = sigma ** 2 + magnitude ** 2 * phi
    dv = -(magnitude ** 2 / width) * dphi
    fisher = float(np.sum(0.5 * dv ** 2 / v ** 2))
    return dict(fisher=fisher, crb=(1.0 / fisher if fisher > 0 else math.inf),
                width=float(width), variance=v)


# --------------------------------------------------------------------------
# the bound that does exist: Ziv-Zakai
# --------------------------------------------------------------------------


def pmin_binary(h: np.ndarray, *, sigma: float, magnitude: float) -> np.ndarray:
    """Minimum error probability of the equal-prior binary test between onset
    `a` and onset `a + h`.

    Only the `h` samples in `[a, a+h)` differ between the hypotheses -- they
    carry variance `v1` under the earlier onset and `v0` under the later --
    so the test reduces to `S = sum r_i^2` against a threshold, `S/v ~
    chi^2_h`, and the answer is exact rather than simulated:

        c        = h log(v1/v0) v0 v1 / (v1 - v0)
        P_min(h) = (1/2)[ 1 - F_h(c/v0) + F_h(c/v1) ]

    It does not depend on `a`, which is what makes the ZZB's stationary form
    apply. At `magnitude = 0` the two hypotheses are identical and it is
    exactly 1/2.
    """
    h = np.asarray(h, dtype=np.float64)
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0; at sigma=0 the onset is exact")
    v0 = sigma ** 2
    v1 = sigma ** 2 + magnitude ** 2
    if v1 <= v0:
        return np.full(h.shape, 0.5)
    c = h * math.log(v1 / v0) * v0 * v1 / (v1 - v0)
    return 0.5 * (1.0 - chi2.cdf(c / v0, h) + chi2.cdf(c / v1, h))


def zzb(*, n: int, sigma: float, magnitude: float) -> dict:
    """Ziv-Zakai lower bound on the MSE of ANY onset estimator, under a
    uniform prior on `{0, ..., n-1}`:

        MSE >= (1/n) sum_{h=1}^{n-1} h (n - h) P_min(h)

    The `(n - h)` factor counts the pairs `(a, a+h)` that fit in the prior's
    support; the `1/n` is the prior mass. NOT valley-filled -- the valley-
    filling refinement only tightens the bound, so omitting it leaves a valid
    bound and errs in the safe direction for a must-fire that asks whether an
    estimator ever falls BELOW it.

    CALIBRATION, and it is exact, not asymptotic: at `magnitude = 0` the data
    are uninformative, `P_min = 1/2`, and the sum telescopes to `(n^2 - 1)/12`
    -- the variance of the prior, which is the true MMSE when there is nothing
    to learn. A bound that missed this case would be wrong by a constant
    everywhere else too.
    """
    h = np.arange(1, n, dtype=np.float64)
    p = pmin_binary(h, sigma=sigma, magnitude=magnitude)
    bound = float(np.sum(h * (n - h) * p) / n)
    return dict(bound=bound, rms=math.sqrt(bound),
                p_min=np.concatenate(([0.5], p)), n=int(n),
                sigma=float(sigma), magnitude=float(magnitude),
                prior_var=(n * n - 1) / 12.0)


# --------------------------------------------------------------------------
# the estimator the bound is run against, and the sweep
# --------------------------------------------------------------------------


def posterior_mean_onset(r: np.ndarray, *, sigma: float,
                         magnitude: float) -> float:
    """The Bayes-optimal onset estimator under the SAME uniform prior the ZZB
    assumes, with `sigma` and `m` known. Its Monte-Carlo MSE is the MMSE, and
    the MMSE is the tightest thing the bound can be tested against: if this
    estimator dips below the bound, the bound is wrong.

    `logL(t) = const + sum_{i<t} (c0_i - c1_i)`, so one cumulative sum does it.
    """
    v0 = sigma ** 2
    v1 = sigma ** 2 + magnitude ** 2
    d = (-0.5 * math.log(v0) - r ** 2 / (2.0 * v0)) - \
        (-0.5 * math.log(v1) - r ** 2 / (2.0 * v1))
    logL = np.concatenate(([0.0], np.cumsum(d)))[:r.size]
    p = np.exp(logL - logL.max())
    p /= p.sum()
    return float(np.dot(np.arange(r.size, dtype=np.float64), p))


def noise_sweep(*, n: int, magnitude: float, sigmas, runs: int, seed0: int,
                alpha: float = 0.01, calib_runs: int = 400) -> list[dict]:
    """THE MUST-FIRE for (d). At each noise level: the ZZB curve, the MSE of
    the Bayes-optimal estimator, and the MSE of the SHIPPED Shewhart detector
    from `ceq.x35.residual`.

    The onset is drawn uniformly from `{0, ..., n-1}` per run, because that is
    the prior the bound is stated under. Scoring at one fixed onset while
    bounding under a uniform prior is the mismatch that manufactures a
    spurious sub-bound point, so it is avoided by construction rather than
    checked for afterwards.

    The Shewhart arm needs a value on runs where nothing exceeds threshold.
    It reports the prior mean `(n-1)/2` there -- the least-bad constant under
    the prior. That choice is stated because it moves the arm's MSE and no
    other; it cannot create a sub-bound point, only a larger distance.

    `se_bayes` is the Monte-Carlo standard error of the MSE, so the "never
    below the bound" test has a stated tolerance instead of an eyeballed one.
    """
    rows = []
    for sigma in sigmas:
        bound = zzb(n=n, sigma=sigma, magnitude=magnitude)
        cal = residual.calibrate(range(seed0 - calib_runs, seed0), alpha,
                                 n=n, noise_sd=sigma, **BED)
        rng = np.random.default_rng([0x35D, int(round(sigma * 1e6))])
        e_bayes, e_shew, no_call = [], [], 0
        for k in range(runs):
            t = int(rng.integers(0, n))
            _, r = residual.run_residual(seed0 + k, n=n, noise_sd=sigma,
                                         plant=dict(index=t, magnitude=magnitude),
                                         **BED)
            e_bayes.append((posterior_mean_onset(r, sigma=sigma,
                                                 magnitude=magnitude) - t) ** 2)
            hit = residual.detect(r, cal["threshold"])["onset"]
            if hit is None:
                hit, no_call = (n - 1) / 2.0, no_call + 1
            e_shew.append((hit - t) ** 2)
        eb = np.asarray(e_bayes)
        es = np.asarray(e_shew)
        rows.append(dict(
            sigma=float(sigma), snr=float(magnitude ** 2 / sigma ** 2),
            bound=bound["bound"], bound_rms=bound["rms"],
            mse_bayes=float(eb.mean()), se_bayes=float(eb.std(ddof=1) / math.sqrt(runs)),
            rms_bayes=float(math.sqrt(eb.mean())),
            mse_shewhart=float(es.mean()), se_shewhart=float(es.std(ddof=1) / math.sqrt(runs)),
            rms_shewhart=float(math.sqrt(es.mean())),
            distance_bayes=float(math.sqrt(eb.mean()) / bound["rms"]),
            distance_shewhart=float(math.sqrt(es.mean()) / bound["rms"]),
            threshold=cal["threshold"], no_call=int(no_call), runs=int(runs)))
    return rows

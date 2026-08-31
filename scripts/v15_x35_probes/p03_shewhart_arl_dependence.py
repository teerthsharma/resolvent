"""[V-eq] probe 3 -- Shewhart ARL0, and what dependence does to it, versus CUSUM.

Sources transcribed:
  Basseville & Nikiforov, "Detection of Abrupt Changes: Theory and Application",
  Prentice-Hall 1993, free PDF at people.irisa.fr/Michele.Basseville/kniga/.
    Sec. 5.1.1, Eq (5.1.1)-(5.1.6): a Shewhart control chart is a repeated
      Neyman-Pearson test on samples of fixed size N; t_a = N K*; "the number of
      samples K has a geometrical distribution P(K=k) = (1-alpha_0)^k alpha_0
      where alpha_0 is the probability of false alarms of this Neyman-Pearson
      test", hence
          L(theta_0) = E_0(t_a) = N / alpha_0                     (5.1.4)
          L(theta)   = E_theta(t_a) = N / beta(theta)             (5.1.6)
      The geometric step is where INDEPENDENCE enters, and it is the only place.
  Mikosch & Wintenberger, "Some variations on the extremal index", arXiv:2106.05117.
      Leadbetter's extremal index: if n Fbar(u_n(tau)) -> tau and
          P(M_n <= u_n(tau)) -> exp(-tau theta_X),  theta_X in [0,1],
      then theta_X exists; "if (X_t) is a Gaussian stationary sequence whose
      autocovariance function satisfies cov(X_0, X_h) = o(1/log h) as h -> inf,
      then theta_X = 1" (Berman's condition, Berman 1964 Thm 3.1).
      Consequence for a memoryless chart: P(T_u > n) = P(M_n <= u), so
      E[T_u] ~ 1/(theta_X alpha) -- and theta_X = 1 for Gaussian AR(1)
      (cov = phi^h decays geometrically, so o(1/log h) holds), i.e. the i.i.d.
      value 1/alpha is the asymptote under dependence.
  Johnson & Bagshaw 1974, Technometrics 16(1):103-112 [V, abstract only]:
      CUSUM run length under serial correlation via weak convergence of the
      cumulative sums to a Wiener process. The scale of that Wiener process is
      the LONG-RUN variance sum_k gamma(k) = sigma^2 (1+phi)/(1-phi) for AR(1),
      not the marginal variance -- which is why a cumulative statistic cannot be
      calibrated on the marginal alone.

Measured here, all at the SAME marginal N(0,1) (so the per-point false alarm
probability alpha is identical by construction):
  1. i.i.d.: ARL0 vs the (5.1.4) prediction N/alpha_0 with N=1.
  2. AR(1), phi = 0.5 and 0.8: ARL0 of the same memoryless chart.
  3. threshold sweep: ARL0(AR)/ARL0(iid) -> 1 as the threshold rises (theta = 1).
  4. CUSUM calibrated to the SAME i.i.d. ARL0, then run on AR(1).
CONTROL: the one-sided tail 1 - Phi(L) substituted for the two-sided
  2(1 - Phi(L)) in (5.1.4) -- an O(1) mis-transcription that predicts 2x the ARL.
"""
import numpy as np
from scipy.stats import norm

SEED = 3535


def run_lengths(B, Tmax, phi, detector, seed):
    """Mean first-alarm time from a stationary start, over B replications.
    phi = 0 is i.i.d.; otherwise x_t = phi x_{t-1} + sqrt(1-phi^2) e_t, so the
    marginal is N(0,1) in both cases. detector(state, x) -> (state, alarm_mask)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(B)                     # stationary start
    state = np.zeros(B)
    rl = np.zeros(B, dtype=np.int64)
    live = np.ones(B, dtype=bool)
    for t in range(1, Tmax + 1):
        if t > 1:
            x = phi * x + np.sqrt(1 - phi ** 2) * rng.standard_normal(B) if phi else \
                rng.standard_normal(B)
        state, alarm = detector(state, x)
        newly = live & alarm
        rl[newly] = t
        live &= ~newly
        if not live.any():
            break
    censored = live.sum()
    rl[live] = Tmax
    return rl.mean(), rl.std() / rl.mean(), censored / B


def shewhart(L):
    return lambda s, x: (s, np.abs(x) >= L)


def cusum(k, h):
    def step(s, x):
        s = np.maximum(0.0, s + x - k)
        return s, s >= h
    return step


print("== 1. Shewhart, i.i.d. Gaussian, Basseville Eq (5.1.4) with N = 1 ==")
L = 2.5
alpha = 2 * (1 - norm.cdf(L))
pred = 1.0 / alpha
B, Tmax = 40000, 1600
arl_iid, cv_iid, cens = run_lengths(B, Tmax, 0.0, shewhart(L), SEED)
se = arl_iid / np.sqrt(B)
print("L = %.2f  alpha_0 = 2(1-Phi(L)) = %.6f" % (L, alpha))
print("predicted ARL0 = N/alpha_0        : %.3f" % pred)
print("measured  ARL0 (B=%d)          : %.3f  +/- %.3f (1 se), censored %.4f"
      % (B, arl_iid, se, cens))
print("ratio measured/predicted          : %.4f" % (arl_iid / pred))
pred_ctrl = 1.0 / (1 - norm.cdf(L))
print("CONTROL one-sided tail in (5.1.4) : %.3f  -> ratio %.4f  (O(1) wrong)"
      % (pred_ctrl, arl_iid / pred_ctrl))
print("run-length sd/mean            : %.4f   (geometric law: sqrt(1-alpha) = %.4f)"
      % (cv_iid, np.sqrt(1 - alpha)))

print()
print("== 2. Shewhart, AR(1), SAME marginal N(0,1), same threshold ==")
for phi in (0.5, 0.8):
    a, cv, c = run_lengths(B, Tmax, phi, shewhart(L), SEED + 1)
    print("phi = %.1f : ARL0 = %.3f +/- %.3f | ratio to i.i.d. = %.4f | sd/mean = %.4f | cens %.4f"
          % (phi, a, a / np.sqrt(B), a / arl_iid, cv, c))

print()
print("== 3. threshold sweep at phi = 0.8: ratio -> 1 as the threshold rises ==")
print("(Leadbetter/Berman: theta_X = 1 for a Gaussian sequence, so ARL0 -> 1/alpha)")
Bs = 8000
for Lx in (2.0, 2.5, 3.0, 3.5):
    ax = 2 * (1 - norm.cdf(Lx))
    Tm = int(12 / ax)
    a0, _, _ = run_lengths(Bs, Tm, 0.0, shewhart(Lx), SEED + 7)
    a1, _, _ = run_lengths(Bs, Tm, 0.8, shewhart(Lx), SEED + 7)
    print("L = %.1f  1/alpha = %8.1f  ARL0(iid) = %8.1f  ARL0(AR .8) = %8.1f  ratio = %.4f"
          % (Lx, 1 / ax, a0, a1, a1 / a0))

print()
print("== 4. CUSUM calibrated on i.i.d. to the SAME ARL0, then run on AR(1) ==")
k = 0.5
lo, hi = 1.0, 8.0
for _ in range(14):                                  # bisection on h
    mid = 0.5 * (lo + hi)
    a, _, _ = run_lengths(12000, 4000, 0.0, cusum(k, mid), SEED + 3)
    if a < arl_iid:
        lo = mid
    else:
        hi = mid
h = 0.5 * (lo + hi)
a_iid, _, c0 = run_lengths(B, 4000, 0.0, cusum(k, h), SEED + 4)
print("k = %.2f, calibrated h = %.4f -> i.i.d. ARL0 = %.2f (target %.2f), censored %.4f"
      % (k, h, a_iid, arl_iid, c0))
for phi in (0.5, 0.8):
    a, _, c = run_lengths(B, 4000, phi, cusum(k, h), SEED + 5)
    lrv = (1 + phi) / (1 - phi)
    print("phi = %.1f : CUSUM ARL0 = %.2f | collapse factor %.2fx | long-run var ratio %.1f | censored %.4f"
          % (phi, a, a_iid / a, lrv, c))
    a_s, _, _ = run_lengths(B, Tmax, phi, shewhart(L), SEED + 5)
    print("           memoryless chart at the same phi: ARL0 = %.2f | factor %.2fx"
          % (a_s, arl_iid / a_s))

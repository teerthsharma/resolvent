"""[V-eq] probe 7 -- Grunwald-Letnikov weights / ARFIMA fractional differencing.

Mlaiki 2026, arXiv:2605.08966 ("VORT"), Eq (2):
    (I^alpha_GL f)(t) = sum_j w_j^alpha f_{t-j},
    w_j^alpha := Gamma(j+alpha) / (Gamma(alpha) Gamma(j+1))
  Eq (3): w_j ~ j^{alpha-1} / Gamma(alpha) as j -> infinity
  Eq (4): sum_j w_j z^j = (1-z)^{-alpha},  |z| < 1
  Hypotheses: alpha in (0,1); the kernel is heavy-tailed, sum_j w_j = infinity.

Contreras-Reyes & Palma 2012, arXiv:1208.1728:
  Eq (1)  Phi(B) y_t = Theta(B) (1-B)^{-d} eps_t
  Eq (2)  (1-B)^{-d} = sum_j [Gamma(j+d)/(Gamma(j+1)Gamma(d))] B^j
  Eq (4)  eta_j ~ j^{d-1}/Gamma(d)
  Eq (18) gamma(h) ~ c_gamma |h|^{2d-1}
  Thm 2.1 hypotheses: d in (-1, 1/2), roots of Phi outside the unit circle.

The CEQ v15 contract writes the same kernel as w_k = (-1)^k C(-alpha, k).
This probe checks the two forms are the SAME sequence and checks the two limits.
"""
import numpy as np
from scipy.special import binom, gammaln

a, K = 0.37, 40
w_ceq = np.array([(-1.0) ** k * binom(-a, k) for k in range(K)])          # contract form
w_gl = np.exp(gammaln(np.arange(K) + a) - gammaln(a) - gammaln(np.arange(K) + 1.0))
print("contract (-1)^k C(-a,k) vs VORT Eq(2) Gamma form:", np.abs(w_ceq - w_gl).max())

z = 0.4
print("Eq(4): sum w_k z^k  vs  (1-z)^-a                :",
      abs(np.sum(w_gl * z ** np.arange(K)) - (1 - z) ** (-a)))

k = 10000.0
tail = np.exp(gammaln(k + a) - gammaln(a) - gammaln(k + 1.0))
print("Eq(3): w_k / (k^{a-1}/Gamma(a)) at k=1e4        :",
      tail / (k ** (a - 1) / np.exp(gammaln(a))), "(-> 1)")

for aa, name in [(1e-9, "alpha->0 (identity)"), (1 - 1e-9, "alpha->1 (cumulative sum)")]:
    ww = np.exp(gammaln(np.arange(5) + aa) - gammaln(aa) - gammaln(np.arange(5) + 1.0))
    print("%-26s w_0..w_4 = %s" % (name, np.round(ww, 12)))

# fractional differencing (1-B)^d convolved with (1-B)^{-d} must be the identity filter
d = 0.37
wd = np.array([(-1.0) ** k * binom(d, k) for k in range(K)])              # (1-B)^d weights
conv = np.convolve(wd, w_gl)[:K]
print("(1-B)^d * (1-B)^-d, first 4 taps               :", np.round(conv[:4], 12))

# the arithmetic behind BED-K's "H-hat = alpha-hat + 1/2"
for dd in [0.2, 0.45, 0.8, 1.0]:
    print("  d=%.2f -> H=d+1/2=%.2f | stationary (Thm 2.1, d<1/2)=%-5s | Hurst in (0,1)=%s"
          % (dd, dd + 0.5, dd < 0.5, 0 < dd + 0.5 < 1))

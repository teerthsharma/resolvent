"""[V-eq] probe 4 -- missing-mass inference in astronomy. MOTIVATION ONLY, no method import.

Source transcribed:
  Bertone, Hooper, Silk, "Particle dark matter: evidence, candidates and
  constraints", Phys. Rept. 405:279-390 (2005), arXiv:hep-ph/0404175, Sec. 2.1:
      "In Newtonian dynamics the circular velocity is expected to be
           v(r) = sqrt( G M(r) / r ),                                     (37)
       where, as usual, M(r) = 4 pi Int rho(r) r^2 dr, and rho(r) is the mass
       density profile, and should be falling ~ 1/sqrt(r) beyond the optical
       disc. The fact that v(r) is approximately constant implies the existence
       of an halo with M(r) ~ r and rho ~ 1/r^2."

Hypotheses as the source states them: Newtonian dynamics, circular orbits,
spherical enclosed mass M(r). The inference is a RESIDUAL inference -- the
hidden cause is whatever mass the luminous profile does not account for -- and
that is the only structural point this lineage contributes to X35. It supplies
no estimator, no onset statistic, and no localization procedure.

Checked here:
  (a) Eq (37) with rho ~ 1/r^2 gives v(r) constant (the paper's stated
      implication), to floating point;
  (b) d log M / d log r = 1 for that halo, i.e. M(r) ~ r;
  (c) the discrepancy: mass implied by a flat curve at 20 kpc versus a luminous
      exponential disc.
  CONTROLS: (i) M(r) mis-transcribed as 4 pi Int rho r dr, one power of r short
      of the source's definition -- the flat-curve solution stops being flat and
      v varies by a factor over the same radii; (ii) Eq (37) mis-transcribed
      without the square root, which misprices the enclosed mass by ~1e2.
"""
import numpy as np
from scipy.integrate import quad

G = 4.300917270e-6          # kpc (km/s)^2 / Msun
r = np.linspace(2.0, 30.0, 15)          # kpc

# isothermal halo rho(r) = rho0 (r0/r)^2
rho0, r0 = 1.0e7, 1.0                   # Msun/kpc^3 at r0 = 1 kpc


def rho(x):
    return rho0 * (r0 / x) ** 2


M = np.array([4 * np.pi * quad(lambda s: rho(s) * s ** 2, 0.0, x)[0] for x in r])
v = np.sqrt(G * M / r)                              # Eq (37)
# control (i): one power of r short in M(r) = 4 pi Int rho r^2 dr
M_ctrl = np.array([4 * np.pi * quad(lambda s: rho(s) * s, 1e-6, x)[0] for x in r])
v_ctrl = np.sqrt(G * M_ctrl / r)

print("(a) Eq (37) with rho ~ 1/r^2 : v(r) over 2-30 kpc")
print("    v mean = %.4f km/s, max|v - mean|/mean = %.3e" % (v.mean(), np.abs(v - v.mean()).max() / v.mean()))
print("    CONTROL (M ~ Int rho r dr): max|v-mean|/mean = %.3f, max/min = %.3f"
      % (np.abs(v_ctrl - v_ctrl.mean()).max() / v_ctrl.mean(), v_ctrl.max() / v_ctrl.min()))

slope = np.polyfit(np.log(r), np.log(M), 1)[0]
print("(b) d log M / d log r        : %.12f   (source says M(r) ~ r)" % slope)

# (c) the discrepancy, at NGC-6503-like numbers
v_flat, r_out = 116.0, 22.0                          # km/s, kpc
M_dyn = v_flat ** 2 * r_out / G                      # Eq (37) inverted
Rd, Sigma0 = 1.7, 6.4e8                              # exponential disc: kpc, Msun/kpc^2
M_lum = 2 * np.pi * Sigma0 * Rd ** 2 * (1 - np.exp(-r_out / Rd) * (1 + r_out / Rd))
print("(c) M_dyn(<%.0f kpc) from Eq (37) = %.3e Msun" % (r_out, M_dyn))
print("    M_lum(<%.0f kpc), exponential disc = %.3e Msun" % (r_out, M_lum))
print("    missing-mass ratio M_dyn / M_lum  = %.2f" % (M_dyn / M_lum))
M_dyn_ctrl = v_flat * r_out / G                      # control: Eq (37) without the sqrt
print("    CONTROL (no sqrt) M_dyn          = %.3e Msun  -> off by %.3e"
      % (M_dyn_ctrl, M_dyn / M_dyn_ctrl))

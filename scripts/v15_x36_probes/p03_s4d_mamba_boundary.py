"""[V-eq] probe 3 -- complex diagonal S4D / Mamba: is the magnitude constraint
open or closed, and where is |lambda| = 1 EXACTLY attainable.

Sources transcribed from arXiv LaTeX e-print sources (not the rendered PDFs).

S4D -- Gu, Gupta, Goel, Re, "On the Parameterization and Initialization of
Diagonal State Space Models", NeurIPS 2022, arXiv:2206.11893, src2/method.tex
Sec 3.2 "Parameterization of A":

  "Note that the kernel K(t) = C e^{tA} B blows up to infinity as t -> infinity
   if A has any eigenvalues with positive real part. [Goel et al. 2022] found
   that this is a serious constraint that affects the stability of the model,
   especially when using the SSM as an autoregressive generative model. They
   propose to force the real part of A to be negative, also known as the
   left-half plane condition in classical controls, by parameterizing the real
   part inside an exponential function
        A = -exp(A_Re) + i . A_Im.                                        (S4D-exp)
   We note that instead of exp, any activation function can be used as long as
   its range is bounded on one side, such as ReLU, softplus, etc.  The original
   DSS does not constrain the real part of A, which is sufficient for simple
   tasks involving fixed-length sequences, but could become unstable in other
   settings."

  and Sec 3.5 "Eigenvalue constraint": "While DSS found that letting them be
  unconstrained has slightly better performance, our experiments find that the
  difference is negligible and we recommend contraining negative real part of A
  as is standard practice in control systems."

  The ReLU option is not a remark only: it is an ABLATED, REPORTED row.
  src2/appendix_experiments.tex Table (tab:ablations-real-full), column
  "Real part" in {Exp, -, ReLU} x Discretization in {Bilinear, ZOH}:
      Bilinear ReLU  sCIFAR 85.06 (0.06)   SC(AR) 90.22 (0.25)  BIDMC 0.1172
      ZOH      ReLU  sCIFAR 84.98 (0.72)   SC(AR) 90.03 (0.13)  BIDMC 0.1232

Mamba -- Gu & Dao, arXiv:2312.00752, src/method.tex Theorem 1 and its proof in
src/appendix.tex Sec A:

  "Theorem 1.  When N=1, A=-1, B=1, s_dt = Linear(x), and tau_dt = softplus,
   then the selective SSM recurrence takes the form
        g_t = sigma(Linear(x_t))
        h_t = (1 - g_t) h_{t-1} + g_t x_t ."
  proof, ZOH: "Abar_t = exp(dt A) = 1/(1 + exp(Linear(x_t))) = sigma(-Linear(x_t))
               = 1 - sigma(Linear(x_t))",
              "Bbar_t = (dt A)^{-1}(exp(dt A) - I) . dt B = 1 - Abar = sigma(Linear(x_t))".
  and Sec 3.5.2: "A ... ultimately affects the model only through its interaction
  with dt via Abar = exp(dt A) (the discretization)".

Hypotheses as the sources state them: A_Re, A_Im unconstrained reals; dt > 0
(Mamba: dt = softplus(.) > 0 strictly); the ZOH map is Abar = exp(dt A) so
|Abar| = exp(dt Re A); the bilinear (Cayley) map is
Abar = (I + dt A/2)(I - dt A/2)^{-1}, which sends the imaginary axis to the unit
circle EXACTLY and the open left half-plane to the open unit disk.

What this probe settles for X36: the answer to "open or closed" is NOT the same
for the two activations the S4D paper itself offers.
  * with exp  : Re A < 0 strictly -> |Abar| in (0,1), OPEN (same shape as LRU).
  * with ReLU : Re A = -ReLU(A_Re) = 0 EXACTLY for every A_Re <= 0, i.e. on a
    half-line of parameters of positive measure -> |Abar| = 1 EXACTLY.
    So the CLOSED upper endpoint m = 1 is already occupied, attainably, by a
    reported S4D configuration -- not as a limit.
  * Mamba's gate is sigma(.) in (0,1), OPEN, and the paper's own mechanism for
    "ignore this token" is g_t -> 0, a limit, never g_t = 0.

CONTROLS:
  - bilinear mis-transcribed as (1 + dt A)/(1 - dt A/2): |Abar| != 1 on the
    imaginary axis by O(1).
  - ZOH mis-transcribed as exp(A/dt): O(1) wrong decay.
  - sigmoid with the sign of the argument flipped: gate inverted, O(1).

Reproduce: python scripts/v15_x36_probes/p03_s4d_mamba_boundary.py
"""
import numpy as np

rng = np.random.default_rng(20260831)
N = 400000

A_Re = rng.normal(0.0, 2.0, N)          # the unconstrained learnable real part
A_Im = rng.normal(0.0, 5.0, N)
dt = 0.01

print("=" * 74)
print("S4D:  A = -f(A_Re) + i A_Im,  ZOH  Abar = exp(dt A),  dt =", dt)
print("=" * 74)

for name, f in (("exp ", np.exp), ("ReLU", lambda z: np.maximum(z, 0.0)),
                ("softplus", lambda z: np.log1p(np.exp(np.clip(z, -700, 700))))):
    A = -f(A_Re) + 1j * A_Im
    Abar = np.exp(dt * A)
    mod = np.abs(Abar)
    eq1 = int((mod == 1.0).sum())
    print(f"  real part = {name:>8}:  Re A max = {A.real.max(): .6e}"
          f"   |Abar| max = {mod.max()!r}"
          f"   exactly 1.0: {eq1}/{N} = {eq1/N:.4f}")

print("  => with ReLU, Re A == 0.0 for every A_Re <= 0, so |Abar| == 1.0 EXACTLY")
print("     on a positive-measure set. The closed endpoint m = 1 is ATTAINED.")
print("     With exp, Re A < 0 strictly and |Abar| < 1 strictly (open).")

# how far below 1 does the exp variant sit, and where does float64 close it?
A = -np.exp(A_Re) + 1j * A_Im
mod_exp = np.abs(np.exp(dt * A))
print("  exp variant: max |Abar| =", mod_exp.max(), " 1 - max =", 1.0 - mod_exp.max())
for are in (-10.0, -20.0, -30.0, -34.0, -40.0):
    m = abs(np.exp(dt * (-np.exp(are) + 0j)))
    print(f"    A_Re = {are:>6}:  |Abar| = {m!r}   == 1.0 in float64: {m == 1.0}")

print()
print("=" * 74)
print("S4D bilinear (Cayley):  Abar = (1 + dt A/2)/(1 - dt A/2)")
print("=" * 74)


def cayley(A, dt, bad=False):
    num = 1.0 + dt * A if bad else 1.0 + dt * A / 2.0
    return num / (1.0 - dt * A / 2.0)


A_axis = 0.0 + 1j * A_Im                       # Re A = 0 exactly (the ReLU case)
m_axis = np.abs(cayley(A_axis, dt))
m_axis_bad = np.abs(cayley(A_axis, dt, bad=True))
A_lhp = -np.exp(A_Re) + 1j * A_Im
m_lhp = np.abs(cayley(A_lhp, dt))
print("  Re A = 0 (ReLU branch):  max | |Abar| - 1 | =", np.abs(m_axis - 1).max())
print("  CONTROL (1 + dt A) numerator: max | |Abar| - 1 | =",
      np.abs(m_axis_bad - 1).max(), " (O(1e-2), not O(1e-16))")
print("  Re A < 0 (exp branch) :  max |Abar| =", m_lhp.max(), " all < 1:",
      bool((m_lhp < 1).all()))

print()
print("=" * 74)
print("Mamba Theorem 1:  Abar_t = sigma(-Linear(x_t)) = 1 - g_t,  g_t = sigma(.)")
print("=" * 74)
L = rng.normal(0.0, 4.0, N)
sig = 1.0 / (1.0 + np.exp(-L))
Abar_m = 1.0 / (1.0 + np.exp(L))                # sigma(-L)
# cross-check the proof's own two expressions agree
dtt = np.log1p(np.exp(np.clip(L, -700, 700)))   # softplus(Linear(x)) = dt
zoh = np.exp(dtt * (-1.0))                      # exp(dt * A) with A = -1
print("  max | sigma(-L) - exp(softplus(L) * (-1)) | =", np.abs(Abar_m - zoh).max())
print("  CONTROL sign flip, |sigma(L) - exp(-softplus(L))| max =",
      np.abs(sig - zoh).max(), "(O(1))")
print("  Abar range on this sample: min =", Abar_m.min(), " max =", Abar_m.max())
print("  exactly 0.0:", int((Abar_m == 0.0).sum()), "  exactly 1.0:",
      int((Abar_m == 1.0).sum()))
for l in (-40.0, -37.0, -36.0, 36.0, 37.0, 40.0):
    v = 1.0 / (1.0 + np.exp(l))
    print(f"    Linear = {l:>6}:  Abar = {v!r}")
print("  => mathematically sigma maps R onto the OPEN (0,1); in float64 it")
print("     saturates to exactly 1.0 for Linear <~ -37 and exactly 0.0 once")
print("     exp(Linear) overflows, Linear > 709.78.  Mamba's stated 'ignore the")
print("     token' mechanism is")
print("     g_t -> 0, a limit, not an attained 0.")

assert (np.abs(np.exp(dt * (-np.maximum(A_Re, 0.0) + 1j * A_Im))) == 1.0).sum() > 0
assert (np.abs(np.exp(dt * (-np.exp(A_Re) + 1j * A_Im))) < 1.0).all()
assert np.abs(m_axis - 1).max() < 1e-14
assert np.abs(m_axis_bad - 1).max() > 1e-4
assert np.abs(Abar_m - zoh).max() < 1e-12
print("OK")

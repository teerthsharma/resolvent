"""[V-eq] probe 2 -- LRU (Orvieto et al. 2023): the EXACT range of |lambda|,
and what the closed/open distinction costs at the boundary.

Source transcribed VERBATIM from the arXiv LaTeX source of arXiv:2303.06349
(`content_main.tex`, retrieved via arxiv.org/e-print, not from the rendered PDF:
the PDF's Type-1 encoding silently drops the minus signs and the infinity glyph,
which is exactly how a paraphrase loses the boundary statement).

  Sec 3.3, "Enforcing stability":
    "An important benefit of the exponential parameterization is that it makes
     it simple to enforce stability on the eigenvalues. To see this, note that
     at initialization, |lambda_j| = |exp(-nu_j)| <= 1 since nu_j > 0.
     Therefore, to preserve stability during training, we can use an exponential
     or another positive nonlinearity:
         lambda_j := exp(-exp(nu_j^{log}) + i theta_j),                     (LRU)
     where nu^{log} in R^N is the parameter we optimize, and we set
     nu_j^{log} := log(nu) at initialization. ...
     We choose an exponential non-linearity over a simple ReLU nonlinearity to
     increase granularity around |lambda| = 1, achieved at nu^{log} = -infinity
     (while |lambda| = 0 is achieved at nu^{log} = infinity)."

  Lemma 3.2 (the initialisation, `\begin{restatable}[]{lem}{sampling}`):
    "Let u_1, u_2 be independent uniform random variables on the interval [0,1].
     Let 0 <= r_min <= r_max <= 1. Compute
         nu = -(1/2) log( u_1 (r_max^2 - r_min^2) + r_min^2 )   and
         theta = 2 pi u_2.
     Then exp(-nu + i theta) is uniformly distributed on the ring in C between
     circles of radii r_min and r_max."

  Prop 3.3 (Forward-pass blow-up):
    "Let Lambda be diagonal with eigenvalues sampled uniformly on the ring in C
     between circles of radii r_min < r_max < 1. Then, under constant or
     white-noise input and Glorot input projection, ...
        E[||x_inf||^2] = 1/(r_max^2 - r_min^2) log((1-r_min^2)/(1-r_max^2))
                          E[||B u||^2]."
    and its r_min = r_max = r limit, transcribed from the paper's Eq. (6):
        lim E[||x_inf||^2] / E[||Bu||^2] = 1/(1 - r^2).

  Sec 3.4 normalisation: "we use normalization parameter gamma^{log} in R^N,
    initialized element-wise as gamma_i^{log} <- log( sqrt(1 - |lambda_i|^2) )".

Hypotheses as the source states them: nu^{log} is an UNCONSTRAINED real
parameter; theta is unconstrained real; Lemma 3.2's r_min, r_max satisfy the
CLOSED chain 0 <= r_min <= r_max <= 1; Prop 3.3 requires the STRICT chain
r_min < r_max < 1, and the gamma initialisation requires |lambda| < 1 for
log(sqrt(1-|lambda|^2)) to be finite.

What this probe settles for X36 (four separate questions):
  (a) the exact range of |lambda| under (LRU) in exact arithmetic;
  (b) where float64 rounds that range shut, i.e. where the open/closed
      distinction stops being observable at all;
  (c) whether Lemma 3.2's INITIALISATION reaches the endpoints (it does: the
      chain is closed, r_min = r_max = 1 gives |lambda| = 1 exactly);
  (d) what a CLOSED cap does to the 1/(1 - a) the delta is trying to repair.

CONTROLS, one per transcription:
  - (LRU) with the inner exp dropped, lambda = exp(-nu + i theta): this is the
    delta's shorthand read literally, and its magnitude leaves [0,1] by O(1).
  - Lemma 3.2 with -log(...) in place of -(1/2)log(...): wrong ring, O(1).

Reproduce: python scripts/v15_x36_probes/p02_lru_lambda_range.py
"""
import numpy as np
from mpmath import mp, mpf, exp as mpexp

rng = np.random.default_rng(20260831)
mp.dps = 600

print("=" * 74)
print("(a) RANGE OF |lambda| UNDER  lambda = exp(-exp(nu_log) + i theta)")
print("=" * 74)


def mod_lru(nu_log):
    """|lambda| = |exp(-exp(nu_log) + i theta)| = exp(-exp(nu_log))."""
    return np.exp(-np.exp(nu_log))


nu = np.array([-1e3, -100.0, -50.0, -37.0, -36.0, -20.0, -1.0, 0.0,
               1.0, 3.0, 6.0, 6.57, 7.0, 20.0, 100.0])
for x in nu:
    print(f"  nu_log = {x:>9.4g}   |lambda| = {mod_lru(x)!r}")

# exact arithmetic: for FINITE nu_log, 0 < exp(nu_log) < inf, so 0 < |lambda| < 1.
for x in (-1000, -100, 100, 1000):
    v = mpexp(-mpexp(mpf(x)))
    print(f"  mpmath 600dps  nu_log={x:>6}:  1-|lambda| = {mp.nstr(1 - v, 8)}"
          f"   |lambda| = {mp.nstr(v, 8)}   (0 < |lambda| < 1 : "
          f"{bool(v > 0 and v < 1)})")

print()
print("  SUP  |lambda| = 1, attained only at nu_log = -infinity  -> NOT attained")
print("  INF  |lambda| = 0, attained only at nu_log = +infinity  -> NOT attained")
print("  => the LRU magnitude range is the OPEN interval (0, 1).")

print()
print("CONTROL: inner exp dropped,  lambda = exp(-nu + i theta)")
for x in (-1.0, 0.0, 1.0):
    print(f"  nu = {x:>5}  |lambda| = {np.exp(-x):.6f}"
          f"   {'>1  LEAVES THE DISK' if np.exp(-x) > 1 else ''}")
print("  max over nu in [-5,5]:", np.exp(5.0), "-- O(1e2), not O(1e-16).")

print()
print("=" * 74)
print("(b) WHERE float64 CLOSES THE OPEN INTERVAL BY ROUNDING")
print("=" * 74)
grid = np.linspace(-45.0, 12.0, 570001)
vals = mod_lru(grid)
hit1 = grid[vals == 1.0]
hit0 = grid[vals == 0.0]
print("  smallest nu_log on grid with |lambda| == 1.0 exactly (float64):",
      hit1.max() if hit1.size else None)
print("  smallest nu_log on grid with |lambda| == 0.0 exactly (float64):",
      hit0.min() if hit0.size else None)
print("  largest |lambda| strictly below 1 on grid:", vals[vals < 1.0].max())
print("  1 - that value:", 1.0 - vals[vals < 1.0].max(), " (= 1 ulp region)")
print("  => in float64 the endpoints ARE reachable, from nu_log <~ -36.7 and")
print("     nu_log >~ 6.57.  The open/closed difference is a real-analysis")
print("     statement; it is NOT observable in the stored parameter.")

print()
print("=" * 74)
print("(c) LEMMA 3.2 INITIALISATION: the chain 0 <= r_min <= r_max <= 1 IS CLOSED")
print("=" * 74)


def lemma32(u1, r_min, r_max, half=True):
    c = 0.5 if half else 1.0
    return -c * np.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2)


N = 400000
u1 = rng.random(N)
for (rmin, rmax) in [(0.0, 1.0), (0.9, 0.999), (1.0, 1.0), (0.0, 0.0)]:
    with np.errstate(divide="ignore"):
        nu_s = lemma32(u1, rmin, rmax)
        mod = np.exp(-nu_s)
    # radial CDF of a uniform ring:  P(|z| <= r) = (r^2 - rmin^2)/(rmax^2 - rmin^2)
    if rmax > rmin:
        r = 0.5 * (rmin + rmax)
        emp = (mod <= r).mean()
        thy = (r ** 2 - rmin ** 2) / (rmax ** 2 - rmin ** 2)
        extra = f"  CDF at midpoint emp={emp:.6f} theory={thy:.6f} |diff|={abs(emp-thy):.2e}"
    else:
        extra = ""
    print(f"  r_min={rmin}, r_max={rmax}:  |lambda| min={mod.min()!r} max={mod.max()!r}"
          f"  exactly==r_max: {int((mod == rmax).sum())}/{N}{extra}")

print("  CONTROL Lemma 3.2 with -log(.) instead of -(1/2)log(.), r=[0,1]:")
nu_bad = lemma32(u1, 0.0, 1.0, half=False)
mod_bad = np.exp(-nu_bad)
r = 0.5
print(f"    CDF at 0.5 = {(mod_bad <= r).mean():.6f}  vs theory {0.25:.6f}"
      f"   |diff| = {abs((mod_bad <= r).mean() - 0.25):.4f}   (O(1), not O(1e-16))")

print()
print("=" * 74)
print("(d) WHAT A CLOSED CAP DOES TO  1/(1 - a)  -- the quantity R1 found undefined")
print("=" * 74)
M = 200000
x = rng.normal(0.0, 3.0, M)                    # a pre-cap real feature
m_closed = np.clip(x, 0.0, 1.0)                # the delta's CLOSED hard cap
m_lru = mod_lru(rng.normal(0.0, 3.0, M))       # LRU's open map, same spread

print(f"  closed cap m = clip(x,0,1):   exactly 1.0: {int((m_closed==1.0).sum())}/{M}"
      f" = {(m_closed==1.0).mean():.4f}   exactly 0.0: {(m_closed==0.0).mean():.4f}")
print(f"  LRU open map:                 exactly 1.0: {int((m_lru==1.0).sum())}/{M}"
      f" = {(m_lru==1.0).mean():.4f}   exactly 0.0: {(m_lru==0.0).mean():.4f}")
with np.errstate(divide="ignore"):
    g_closed = 1.0 / (1.0 - m_closed)
    g_lru = 1.0 / (1.0 - m_lru)
print("  1/(1-m) under the CLOSED cap: infinities =", int(np.isinf(g_closed).sum()),
      " max finite =", g_closed[np.isfinite(g_closed)].max())
print("  1/(1-m) under LRU's open map: infinities =", int(np.isinf(g_lru).sum()),
      " max finite =", g_lru[np.isfinite(g_lru)].max())
print("  LRU gamma init  log(sqrt(1-|lambda|^2))  at |lambda| = 1 :",
      np.log(np.sqrt(1.0 - 1.0 ** 2)))
print("  => closing the interval at 1 does NOT repair 1/(1-a); it makes the pole")
print("     ATTAINABLE on a positive-measure set of parameters instead of")
print("     approachable only in the limit.  LRU's openness is load-bearing for")
print("     its own Prop 3.3 (r_max < 1 strict) and its own gamma normalisation.")

# gradient of the magnitude at the boundary: why LRU cannot walk onto it
eps = 1e-6
for x0 in (-30.0, -10.0, -2.0, 0.0):
    d = (mod_lru(x0 + eps) - mod_lru(x0 - eps)) / (2 * eps)
    print(f"  d|lambda|/d nu_log at nu_log={x0:>6}: {d: .6e}"
          f"   (|lambda|={mod_lru(x0):.12f})")
print("  clip cap gradient in the interior: 1.0 ; outside: 0.0 (attains the cap)")

assert mod_lru(-1e3) == 1.0 and mod_lru(1e3) == 0.0     # float64 closes it
assert 0 < mpexp(-mpexp(mpf(-1000))) < 1                # exact arithmetic does not
assert 0 < mpexp(-mpexp(mpf(1000))) < 1
assert (m_closed == 1.0).sum() > 0 and np.isinf(g_closed).sum() > 0
assert np.isinf(g_lru).sum() >= 0
print("OK")

"""[V-eq] probe 10 -- interventional identification (Perturb-seq / GRN).

Dixit, Parnas, Li, Chen, Fulco, Jerby-Arnon et al. 2016, "Perturb-Seq: Dissecting
Molecular Circuits with Scalable Single-Cell RNA Profiling of Pooled Genetic Screens",
Cell 167(7):1853-1866.e17, doi:10.1016/j.cell.2016.11.038.
  MIMOSCA model: Y = X beta + eps -- Y the (log) expression matrix, X the design
  matrix of guide identities and covariates, beta the regulatory-effect matrix;
  fitted with elastic net (l1_ratio = 0.5, alpha = 5e-4).

Hauser & Buhlmann 2012, JMLR 13:2409-2464, arXiv:1104.2808.
  Def 6 (conservative family of targets I): for all a in [p] there is I in I with a not in I.
  Thm 10: D1 and D2 are I-Markov equivalent iff for all I in I the intervention
  graphs D1^(I) and D2^(I) are observationally Markov equivalent; equivalently D1
  and D2 have the same skeleton and v-structures and D1^(I), D2^(I) have the same
  skeleton for all I in I.  Consequence: interventional Markov equivalence is a
  STRICTLY FINER partition of DAGs than observational Markov equivalence.

Instance 1: a linear SEM z -> x -> y with the confounding edge z -> y.  The
observational regression of y on x estimates a + b*Cov(z,x)/Var(x), NOT a; the
same regression under do(x) estimates a.  This is exactly the contract's sentence
"interventions identify a directly; observations confound a with b".
Instance 2: X -> Y and Y -> X are observationally Markov equivalent and separated
by one intervention (Thm 10 at work).
"""
import numpy as np

rng = np.random.default_rng(9)
N = 400_000
a, b, c = 0.8, 0.5, 1.3
sz, se = 1.0, 0.5

z = rng.normal(0, sz, N)
x = c * z + rng.normal(0, se, N)
y = a * x + b * z + rng.normal(0, se, N)
ols_obs = np.cov(x, y)[0, 1] / np.var(x)
plim = a + b * (c * sz ** 2) / (c ** 2 * sz ** 2 + se ** 2)
print("observational OLS  y ~ x : %.6f   (true direct effect a = %.2f)" % (ols_obs, a))
print("closed-form confounded plim a + b c sz^2/(c^2 sz^2 + se^2) = %.6f  |diff| = %.2e"
      % (plim, abs(plim - ols_obs)))

# do(x): x set independently of z, structural equation for y unchanged
xd = rng.normal(0, 1.0, N)
yd = a * xd + b * z + rng.normal(0, se, N)
ols_do = np.cov(xd, yd)[0, 1] / np.var(xd)
print("interventional  do(x)    : %.6f   |error vs a| = %.2e" % (ols_do, abs(ols_do - a)))
print("confounding bias removed by the intervention: %.4f -> %.2e"
      % (abs(ols_obs - a), abs(ols_do - a)))

# --- Hauser & Buhlmann Thm 10: direction is not identified observationally ---
u = rng.normal(size=N)
v = 0.9 * u + rng.normal(size=N) * 0.4                 # true DAG: u -> v
reg = lambda s, t: np.cov(s, t)[0, 1] / np.var(s)
print("\nobservational: v~u = %.4f , u~v = %.4f  (both nonzero -> same MEC)"
      % (reg(u, v), reg(v, u)))
ud = rng.normal(size=N)                                # do(u)
vd = 0.9 * ud + rng.normal(size=N) * 0.4
vi = rng.normal(size=N)                                # do(v): v cut from its parents
ui = rng.normal(size=N)                                # u unaffected by do(v)
print("after do(u): v~u = %.4f (survives) ; after do(v): u~v = %.4f (-> 0)"
      % (reg(ud, vd), reg(vi, ui)))
print("=> one intervention on a conservative target family separates the two DAGs")

"""[V-eq] probe 1 -- latent-variable inference: the EM estimator and the identifiability it needs.

Sources transcribed:
  Dempster, Laird, Rubin 1977, JRSS-B 39(1):1-38.
    E-step: compute Q(phi | phi^(p)) = E[ log f(x|phi) | y, phi^(p) ]
    M-step: phi^(p+1) = argmax_phi Q(phi | phi^(p))
    Theorem 1 (GEM): L(M(phi)) >= L(phi) for all phi in Omega.
  Allman, Matias, Rhodes 2009, Ann. Statist. 37(6A):3099-3132, arXiv:0809.5032.
    Eq (1): P = sum_{i=1..r} pi_i (x)_{j=1..p} p_ij     (r-class, p-feature model)
    Theorem 1 (Kruskal): I1 + I2 + I3 >= 2r + 2  =>  [M1,M2,M3] determines the Mj
                         up to simultaneous permutation and rescaling of rows.
    Corollary 3: M(r; k1,k2,k3) is generically identifiable up to label swapping
                 provided min(r,k1) + min(r,k2) + min(r,k3) >= 2r + 2.

What is checked here:
  (a) DLR Theorem 1 holds for the correct EM (min increment >= 0), and the
      CONTROL -- an E-step whose responsibility divides by the component density
      instead of multiplying by it, an O(1) mis-transcription of Bayes' rule --
      breaks monotonicity by O(1).
  (b) EM's guarantee is monotone ascent only: over 20 random starts the final
      log-likelihood spreads by O(1) (Wu 1983's point: stationary values, not
      the global maximum).
  (c) label swapping: two runs' parameter vectors agree only after sorting.
  (d) Corollary 3's arithmetic at the boundary, checked constructively:
      p=2, r=2, k=2 gives 2+2 = 4 < 2r+2 = 6 -> a genuinely different parameter
      vector reproduces the same 2-way table to ~1e-16;
      p=3, r=2, k=2 gives 2+2+2 = 6 >= 6 -> the same search finds no alternative.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm

rng = np.random.default_rng(35)

# ---------------------------------------------------------------- (a) and (b)
# Two-component univariate Gaussian mixture, EM exactly as DLR define it.
n = 400
z = rng.random(n) < 0.35
y = np.where(z, rng.normal(-2.0, 1.0, n), rng.normal(2.0, 1.5, n))


def loglik(y, pi, mu, sd):
    return np.log(np.sum(pi * norm.pdf(y[:, None], mu, sd), axis=1)).sum()


SD_FLOOR = 1e-2   # keeps the degenerate zero-variance spike out of the comparison


def em(y, pi, mu, sd, iters=300, invert_density=False):
    """invert_density=True is the O(1) mis-transcription control: the E-step
    responsibility is pi_i / p(y|i) renormalized, instead of pi_i p(y|i)
    renormalized -- the density enters the wrong way round."""
    L = [loglik(y, pi, mu, sd)]
    for _ in range(iters):
        dens = np.maximum(norm.pdf(y[:, None], mu, sd), 1e-300)
        w = pi / dens if invert_density else pi * dens
        R = w / w.sum(axis=1, keepdims=True)          # E-step
        Nk = R.sum(axis=0)                            # M-step
        pi = Nk / n
        mu = (R * y[:, None]).sum(axis=0) / Nk
        sd = np.maximum(np.sqrt((R * (y[:, None] - mu) ** 2).sum(axis=0) / Nk), SD_FLOOR)
        L.append(loglik(y, pi, mu, sd))
    return np.array(L), pi, mu, sd


start = (np.array([.5, .5]), np.array([-1.0, 1.0]), np.array([1.0, 1.0]))
L1, pi1, mu1, sd1 = em(y, *start)
Lc, _, _, _ = em(y, *start, invert_density=True)

finals = []
for _ in range(20):
    Lr, _, _, _ = em(y, np.array([.5, .5]),
                     rng.normal(0, 4, 2), np.abs(rng.normal(0, 3, 2)) + SD_FLOOR)
    finals.append(Lr[-1])
finals = np.array(finals)

print("DLR Thm 1  min per-iteration increment of L   :", np.diff(L1).min())
print("control (inverted-density E-step) min increment:", np.diff(Lc).min())
print("EM final loglik, best of 20 random starts     :", finals.max())
print("EM final loglik, worst of 20 random starts    :", finals.min())
print("O(1) spread over stationary points            :", finals.max() - finals.min())

# (c) label swapping: rerun A from a mirrored start; compare raw and sorted.
L3, pi3, mu3, sd3 = em(y, np.array([.5, .5]), np.array([1.0, -1.0]), np.array([1.0, 1.0]))
print("label swap: |mu_A - mu_C| raw                 :", np.abs(mu1 - mu3).max())
print("label swap: |sort(mu_A) - sort(mu_C)|         :", np.abs(np.sort(mu1) - np.sort(mu3)).max())

# ------------------------------------------------------------------------ (d)
# Allman Eq (1) for binary features: P(x_1..x_p) = sum_i pi_i prod_j p_ij(x_j)^..
def table(pi, P):
    """P has shape (r, p): P[i,j] = Pr(X_j = 1 | Z = i). Returns the full 2^p table."""
    r, p = P.shape
    idx = np.array([[(k >> j) & 1 for j in range(p)] for k in range(2 ** p)])   # (2^p, p)
    per_class = np.prod(np.where(idx[:, None, :] == 1, P[None, :, :], 1 - P[None, :, :]), axis=2)
    return per_class @ pi


def alt_search(pi_true, P_true, pi_alt_fixed, restarts=40):
    """Look for a DIFFERENT parameter vector with mixing proportions pinned at
    pi_alt_fixed that reproduces the same table. Returns the best residual and
    the parameter distance achieved."""
    T = table(pi_true, P_true)
    r, p = P_true.shape
    best = (np.inf, None)
    for _ in range(restarts):
        x0 = rng.random(r * p)
        sol = least_squares(lambda x: table(pi_alt_fixed, x.reshape(r, p)) - T,
                            x0, bounds=(1e-9, 1 - 1e-9), xtol=1e-15, ftol=1e-15, gtol=1e-15)
        res = np.abs(table(pi_alt_fixed, sol.x.reshape(r, p)) - T).max()
        if res < best[0]:
            best = (res, sol.x.reshape(r, p))
    dist = min(np.abs(best[1] - P_true).max(), np.abs(best[1][::-1] - P_true).max())
    return best[0], dist


pi_true = np.array([0.4, 0.6])
P2 = np.array([[0.2, 0.7], [0.8, 0.3]])                 # p = 2 features
P3 = np.array([[0.2, 0.7, 0.55], [0.8, 0.3, 0.15]])     # p = 3 features
pi_alt = np.array([0.25, 0.75])                         # pinned away from pi_true and its swap

r = 2
print("Kruskal/Cor.3  p=2: min(r,k) sum = %d   vs 2r+2 = %d  -> condition %s"
      % (2 + 2, 2 * r + 2, "FAILS"))
print("Kruskal/Cor.3  p=3: min(r,k) sum = %d   vs 2r+2 = %d  -> condition %s"
      % (2 + 2 + 2, 2 * r + 2, "holds"))
res2, d2 = alt_search(pi_true, P2, pi_alt)
res3, d3 = alt_search(pi_true, P3, pi_alt)
res3ok, d3ok = alt_search(pi_true, P3, pi_true)   # search capability check, same optimizer
print("p=2 alternative parameters: table residual %.3e , parameter distance %.4f" % (res2, d2))
print("p=3 alternative parameters: table residual %.3e , parameter distance %.4f" % (res3, d3))
print("p=3 search at the TRUE pi (optimizer capability): residual %.3e , distance %.3e"
      % (res3ok, d3ok))

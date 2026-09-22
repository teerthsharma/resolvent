"""Addendum M, M1 DRIFT-EPI: can its registered null pass?

Structure tensor J = G_sigma * (grad V grad V^T) on a (p, t) slice; coherence
(l1 - l2) / (l1 + l2). The contract's null: shuffle the t-axis, coherence must
fall below 0.2 everywhere. Measured on three slices: a planted drift, the same
slice with its t-axis shuffled (the contract's null), and i.i.d. noise.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, sobel

def coherence(V, sigma):
    gp, gt = sobel(V, axis=0), sobel(V, axis=1)
    Jpp = gaussian_filter(gp * gp, sigma); Jtt = gaussian_filter(gt * gt, sigma)
    Jpt = gaussian_filter(gp * gt, sigma)
    tr = Jpp + Jtt; disc = np.sqrt((Jpp - Jtt) ** 2 + 4 * Jpt ** 2)
    c = np.where(tr > 1e-12, disc / tr, 0.0)          # (l1-l2)/(l1+l2)
    theta = 0.5 * np.arctan2(2 * Jpt, Jpp - Jtt)       # dominant gradient angle
    return c, theta

rng = np.random.default_rng(0)
P, T, v = 256, 256, 0.5
f = gaussian_filter(rng.standard_normal(4 * P), 3.0)
p = np.arange(P)[:, None]; t = np.arange(T)[None, :]
drift = f[(p - v * t + 2 * P).astype(int) % (4 * P)]  # pattern moving at velocity v along p
shuffled = drift[:, rng.permutation(T)]
noise = rng.standard_normal((P, T))
crop = (slice(8, -8), slice(8, -8))
for sigma in (1.0, 2.0, 4.0):
    print(f"sigma {sigma}")
    for name, V in (("planted drift", drift), ("t-shuffled (contract null)", shuffled), ("iid noise", noise)):
        c, th = coherence(V, sigma); c = c[crop]
        print(f"  {name:28s} median {np.median(c):.3f}  p95 {np.percentile(c,95):.3f}  max {c.max():.3f}  frac>0.2 {np.mean(c>0.2):.3f}")
    c, th = coherence(drift, sigma)
    slope = -np.tan(np.median(th[crop]))              # gradient angle -> iso-line slope dp/dt
    print(f"  recovered dp/dt {slope:.3f} vs planted v {v}")


# Candidate replacement null: keep every frame intact, destroy the motion by a
# random circular shift of each t-column along p; score the SLOPE statistic
# (fraction of pixels whose local slope is within 20% of the dominant slope,
# coherence-weighted), not raw coherence. Bar: data > 99th pct of 200 surrogates.
def slope_consistency(V, sigma):
    c, th = coherence(V, sigma); c, th = c[crop], th[crop]
    s_loc = -np.tan(th); s0 = np.median(s_loc[c > 0.5]) if np.any(c > 0.5) else 0.0
    ok = np.abs(s_loc - s0) <= 0.2 * max(abs(s0), 1e-3)
    return float(np.sum(c * ok) / np.sum(c))
sigma = 2.0
stat = slope_consistency(drift, sigma)
null = []
for k in range(200):
    r = np.random.default_rng(1000 + k)
    null.append(slope_consistency(np.stack([np.roll(drift[:, j], r.integers(P)) for j in range(T)], 1), sigma))
null = np.array(null)
print(f"replacement null (circular shift per column, sigma 2): data {stat:.3f}  null median {np.median(null):.3f}  null p99 {np.percentile(null,99):.3f}  -> {'PASS' if stat > np.percentile(null,99) else 'FAIL'}")
print(f"  same statistic on the contract's t-shuffle: {slope_consistency(shuffled, sigma):.3f}; on iid noise: {slope_consistency(noise, sigma):.3f}")

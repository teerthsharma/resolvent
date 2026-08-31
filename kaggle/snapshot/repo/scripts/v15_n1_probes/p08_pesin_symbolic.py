"""[V-eq] probe 8 -- Pesin identity and the generating-partition deficit.

Contractor 2023 (U. Chicago REU, following Mane), "The Pesin Entropy Formula":
  Eq (7.1)  Ruelle/Margulis inequality: h_mu(f) <= int_M sum_i lambda_i^+ m_i dmu
  Thm 7.15, Eq (7.16): h_mu(f) = int_M chi dmu, chi = sum_i lambda_i^+ m_i
  HYPOTHESES (stated in Sec 7): f a diffeomorphism of a compact manifold M,
    f' Holder continuous (C^{1+alpha}), mu f-invariant and ABSOLUTELY CONTINUOUS
    with respect to Lebesgue.  Thm 8.1: equality holds iff mu has absolutely
    continuous conditional measures on unstable manifolds, i.e. mu is SRB.
  Def 2.10 (Kolmogorov-Sinai): h(T; alpha) = lim_n (1/n) H(vee_{i<n} T^{-i} alpha),
    h(T) = sup over partitions.

Bollt, Stanford, Lai & Zyczkowski, Physica D 154 (2001) 259-286:
  "The generating partition is defined as a partition for which the topological
   entropy achieves its supremum ... the misplacement of the partition leads to
   diminishing of the computed entropy of the system."  (non-monotone in the
   amount of misplacement -- devil's-staircase-like).

Instance: the tent map, |f'| = 2 everywhere, so lambda = log 2 EXACTLY and the
generating partition is the threshold at x = 1/2.  A misplaced threshold must
lower h_sym, so DEFICIT = lambda-hat - h_sym > 0.

The orbit is iterated in EXACT INTEGER arithmetic on x = s/q, q an odd prime:
  s -> 2s          if 2s <= q
  s -> 2(q - s)    otherwise
A float64 tent orbit loses one bit per step and collapses to a fixed point after
~52 steps, which reports h_sym = 0 at the generating partition -- the integer
orbit is required for the probe to mean anything.
"""
import numpy as np

q = 15_485_863          # prime; x = s/q stays exactly representable in int64
N = 4_000_000
s0 = 4_242_421
s = s0
ss = np.empty(N, dtype=np.int64)
for i in range(N):
    s = 2 * s if 2 * s <= q else 2 * (q - s)
    ss[i] = s
period_hit = int(np.any(ss == s0))
xs = ss / q
lam_hat = np.log(2.0)   # |f'| = 2 a.e. for the tent map, so lambda = log 2 exactly
print("orbit revisited its seed within N steps (0 = no short period):", period_hit)
print("tent map lambda = %.12f   log 2 = %.12f" % (lam_hat, np.log(2.0)))


def h_sym(sym, L=13):
    """entropy rate as the block-entropy slope H(L) - H(L-1), in nats"""
    H = []
    for Li in (L - 1, L):
        w = np.lib.stride_tricks.sliding_window_view(sym, Li)
        codes = w @ (1 << np.arange(Li))
        _, c = np.unique(codes, return_counts=True)
        p = c / c.sum()
        H.append(-(p * np.log(p)).sum())
    return H[1] - H[0]


for thr, tag in [(0.5, "generating  x=0.5 "), (0.4, "misplaced   x=0.4 "),
                 (0.25, "misplaced   x=0.25"), (0.6, "misplaced   x=0.6 "),
                 (0.75, "misplaced   x=0.75")]:
    sym = (xs > thr).astype(np.int64)
    hs = h_sym(sym)
    print("%s  h_sym = %.4f   DEFICIT = lambda - h_sym = %.4f" % (tag, hs, lam_hat - hs))

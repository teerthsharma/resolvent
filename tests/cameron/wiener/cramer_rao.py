"""
W2 - Cramer-Rao (1945) floor for a Monte-Carlo committor/probability
estimate. No unbiased estimator of p from n iid Bernoulli draws beats
variance p(1-p)/n (equality: the sample mean, which is also what every
Monte-Carlo committor estimate in this project actually is). This sharpens
the two-sided Hoeffding bound this project has been using, which assumes
the worst case p=0.5 and is therefore loose everywhere else.

CPU only.
"""
import math

Z95 = 1.959963984540054  # two-sided 95% normal quantile
DELTA = 0.05             # matches the z=1.96 convention above (2*Phi(-z)=delta)


def cramer_rao_n(p, eps, z=Z95):
    """n needed so the CLT/Cramer-Rao-achieving estimator (sample mean) has
    half-width eps at ~95% confidence: z*sqrt(p(1-p)/n) <= eps."""
    return math.ceil((z ** 2) * p * (1 - p) / (eps ** 2))


def hoeffding_n(eps, delta=DELTA):
    """Distribution-free two-sided Hoeffding bound: P(|phat-p|>eps) <= 2 exp(-2 n eps^2)."""
    return math.ceil(math.log(2 / delta) / (2 * eps ** 2))


def looseness_ratio(p, delta=DELTA, z=Z95):
    """hoeffding_n(eps) / cramer_rao_n(p, eps) -- eps cancels, ratio depends
    only on p (and the confidence level). >1 means Hoeffding over-samples
    relative to the Cramer-Rao floor at that p."""
    return math.log(2 / delta) / (2 * (z ** 2) * p * (1 - p))


if __name__ == "__main__":
    print("=== W2 sanity: the two spec points ===")
    for p, eps in [(0.5, 0.02), (0.1, 0.02)]:
        n_cr = cramer_rao_n(p, eps)
        n_hf = hoeffding_n(eps)
        ratio = n_hf / n_cr
        print(f"p={p:<4} eps={eps}: Cramer-Rao/CLT n={n_cr:>6}  "
              f"Hoeffding n={n_hf:>6}  ratio={ratio:.2f}x")

    print("\n=== sweep: looseness_ratio(p) = hoeffding_n / cramer_rao_n (eps cancels) ===")
    ps = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
    rows = [(p, looseness_ratio(p)) for p in ps]
    for p, r in rows:
        print(f"  p={p:<5} ratio={r:6.2f}x")
    worst = max(rows, key=lambda t: t[1])
    best = min(rows, key=lambda t: t[1])
    print(f"\nloosest corner : p={worst[0]} -> {worst[1]:.2f}x over the true floor")
    print(f"tightest point : p={best[0]} -> {best[1]:.2f}x (Hoeffding's own worst case, by design)")
    print("Ratio is symmetric in p<->1-p and diverges as p->0 or p->1: "
          "Hoeffding is calibrated for p=0.5 and gets arbitrarily loose at "
          "extreme (rare-event) probabilities -- exactly where this project's "
          "committor/resolution estimates tend to sit (draws, wins, rare outcomes).")

    # self-check
    n_cr_50 = cramer_rao_n(0.5, 0.02)
    n_hf_50 = hoeffding_n(0.02)
    assert n_cr_50 == 2401, n_cr_50
    assert n_hf_50 == 4612, n_hf_50
    n_cr_10 = cramer_rao_n(0.1, 0.02)
    assert n_cr_10 == 865, n_cr_10
    assert abs(looseness_ratio(0.5) - 1.92) < 0.01
    assert abs(looseness_ratio(0.1) - 5.3) < 0.05
    print("\nself-check OK: n=2401/4612 at p=0.5, n=865 at p=0.1, ratios 1.92x/5.3x reproduced exactly")

"""IS A_8 = 2.187500 A VALID CERTIFICATE? Test the sign-independence assumption.

A_k = E|sum of k Rademacher| was enumerated over 2^k sign patterns ASSUMING the
selected signs are independent and symmetric. Under top-k salience selection the
survivors are CONCOMITANTS of order statistics: the value attached to a token
picked for its magnitude. For a plain Gaussian row that is harmless -- sign and
|value| are independent, and selection touches only magnitude.

The shipped operator is not a plain Gaussian row. `sgate` is

    A = rho * (softmax(w) - lam * softmax(-w)) / (1 + lam)

so an entry's SIGN and its MAGNITUDE are both functions of the same logit w.
Selecting the top-k by |A| can therefore bias the sign law it selects, and if it
does, A_8 is the wrong constant.

MEASURED, not assumed. Three readings, and a must-fire control that a known
correlated sign pattern is DETECTED -- without it a null result means the probe
is blind, not that the signs are independent.
"""
import itertools, math, sys
import torch
sys.path.insert(0, r"C:\Users\seal\Desktop\New folder (32)")
from ceq import bench

torch.set_num_threads(2)
K, D, DRAWS = 8, 16, 4000
SIZES = (16, 128, 512)

A_K = sum(abs(sum(e)) for e in itertools.product((-1, 1), repeat=K)) / 2 ** K


def measure(s, seed=0, operator="sgate", lam=0.10):
    """Return (E|sum of MEASURED signs|, P(+), mean pairwise sign corr, n).

    `lam` IS THE DIAL, and it is threaded here rather than into a second copy of
    this loop. q and k are drawn from `torch.Generator().manual_seed(seed)`
    BEFORE the operator is built and the operator consumes no randomness, so two
    calls at the same `s` and `seed` and different `lam` see BIT-IDENTICAL draws.
    That is what makes the two lam rows comparable at all: a lam=1.00 table on
    fresh draws would confound the dial with the sample.
    """
    g = torch.Generator().manual_seed(seed)
    tot, pos, corr, n = 0.0, 0, 0.0, 0
    for _ in range(DRAWS):
        q = torch.randn(s, D, generator=g)
        k = torch.randn(s, D, generator=g)
        if operator == "sgate":
            a = bench._causal_sgate_operator(q, k, rho=1.5, lam=lam)
        else:                                   # plain Gaussian control row
            a = torch.randn(s, s, generator=g).tril(-1)
        i = s - 1
        row = a[i, :i]                          # causal row of the last query
        if row.numel() < K:
            continue
        idx = torch.topk(row.abs(), K).indices  # SALIENCE selection, by |A|
        eps = torch.sign(row[idx])
        eps = torch.where(eps == 0, torch.ones_like(eps), eps)
        tot += abs(float(eps.sum()))
        pos += int((eps > 0).sum())
        m = float(eps.mean())
        # mean pairwise product = (sum^2 - k) / (k(k-1)); 0 under independence
        corr += (float(eps.sum()) ** 2 - K) / (K * (K - 1))
        n += 1
    return tot / n, pos / (n * K), corr / n, n


print(f"A_8 (independent Rademacher, 2^{K} enumeration) = {A_K:.6f}\n")
print(f"{'operator':>10} {'lam':>6} {'s':>5} {'E|sum eps|':>12} {'vs A_8':>9} "
      f"{'P(+)':>8} {'mean pair corr':>15} {'draws':>7}")
for op, lam in (("gauss", None), ("sgate", 0.10), ("sgate", 1.00)):
    for s in SIZES:
        e, p, c, n = measure(s, operator=op, lam=(0.10 if lam is None else lam))
        print(f"{op:>10} {'--' if lam is None else f'{lam:.2f}':>6} {s:>5} "
              f"{e:>12.6f} {e/A_K:>8.4f}x {p:>8.4f} {c:>15.6f} {n:>7}")

print("\n=== DRAW-IDENTITY BIND: the two lam rows must see the SAME q, k ===")
for s in SIZES:
    ga = torch.Generator().manual_seed(0)
    gb = torch.Generator().manual_seed(0)
    qa, ka = torch.randn(s, D, generator=ga), torch.randn(s, D, generator=ga)
    qb, kb = torch.randn(s, D, generator=gb), torch.randn(s, D, generator=gb)
    same = torch.equal(qa, qb) and torch.equal(ka, kb)
    a10 = bench._causal_sgate_operator(qa, ka, rho=1.5, lam=0.10)
    a100 = bench._causal_sgate_operator(qb, kb, rho=1.5, lam=1.00)
    print(f"  s={s:>4}  torch.equal(q,q')={torch.equal(qa, qb)} "
          f"torch.equal(k,k')={torch.equal(ka, kb)}  -> draws identical: {same}   "
          f"and the OPERATORS differ: {not torch.equal(a10, a100)}")

print("\n=== MUST-FIRE CONTROL: a known-correlated sign law must be DETECTED ===")
g = torch.Generator().manual_seed(1)
for name, frac in (("independent", 0.0), ("30% aligned", 0.30), ("all aligned", 1.0)):
    tot, corr = 0.0, 0.0
    for _ in range(DRAWS):
        eps = torch.where(torch.rand(K, generator=g) < 0.5, -1.0, 1.0)
        if frac > 0:                            # force a fraction to agree
            m = int(round(frac * K))
            if m:
                eps[:m] = eps[0]
        tot += abs(float(eps.sum()))
        corr += (float(eps.sum()) ** 2 - K) / (K * (K - 1))
    print(f"  {name:>14}  E|sum eps| = {tot/DRAWS:>8.6f}  "
          f"({tot/DRAWS/A_K:>6.4f}x A_8)   mean pair corr = {corr/DRAWS:>9.6f}")

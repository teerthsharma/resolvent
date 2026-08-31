"""V15 JUPITER-2 -- the fork probe. Settles whether ONE operator family carries
both (L) label reproduction and (P) parity with self-attention.

    python scripts/v15_jupiter2_fork_probe.py

float64 identities only. Nothing is fitted, nothing is trained (L-LEAN). Every
number printed is an identity residual or a bitwise comparison; no statistic
appears anywhere in this file.

INDEXING, fixed once. Positions 0..S. Position 0 is BOS: it has no gate and no
drive. Gates a_1..a_S, drives b_1..b_S, label

    y_0 = 0,   y_i = a_i * y_{i-1} + b_i,
    y_i = sum_{j=1}^{i} P_ij b_j,   P_ij = prod_{k=j+1}^{i} a_k,   P_ii = 1.

Log-gates g_0 = 0, g_j = log a_j; C = cumsum(g), so C_0 = 0 and
P_ij = exp(C_i - C_j). This matches lean/CEQ/V15.lean's `scan` (inclusive
prefix sum over range(i+1)) entry for entry.

THE SEVEN SECTIONS
  1  the two-branch bind has an empty rejection region  (the ruling)
  2  additive-logit-only computes label / rowsum, exactly (Q1)
  3  no row-stochastic operator with the DRIVES as values reaches the label (Q2)
  4  one that reaches it with rescaled values, and its parity bind (Q3)
  5  the value-scale charge, and how tight it is
  6  the signed variant, and where it fails the rule
  7  the per-row-scalar amendment (L-AMEND), and the dichotomy that kills it

Every bind in sections 1 and 4 is run against a PLANTED NEGATIVE (MISTAKES.md
rule 8): section 1's plant is meant to pass and does, which is the finding;
section 4's plants are meant to fail and do, which is what makes that bind a
gate rather than an assertion.
"""
import numpy as np

FMT = "%.17g"
S = 8
NEG_INF = -np.inf


# ---------------------------------------------------------------- primitives

def softmax_causal(L):
    """Row softmax of a causal logit matrix. L[i, j] must be -inf for j > i."""
    m = np.max(L, axis=1, keepdims=True)
    E = np.exp(L - m)
    return E / np.sum(E, axis=1, keepdims=True)


def std_attention(q, V):
    """Standard causal self-attention read-out. Written independently of
    `head` below so that comparing the two is a check and not an identity
    (MISTAKES.md V-3)."""
    n = q.shape[0]
    L = np.where(np.tril(np.ones((n, n), dtype=bool)), q, NEG_INF)
    return softmax_causal(L) @ V


def head(q, g, s, V):
    """O(theta): ONE causal softmax head whose logit carries the scan.

        l_ij = q_ij + (C_i - C_j) + s_j,     C = cumsum(g)

    (P) at g == 0, s == 0 this is l_ij = q_ij, i.e. std_attention.
    (L) at the oracle setting of section 4 this is the chain label.
    At EVERY theta it is a causal softmax read-out -- the family never leaves
    the class, which is the whole content of the ruling in section 1."""
    n = q.shape[0]
    C = np.cumsum(g)
    L = q + (C[:, None] - C[None, :]) + s[None, :]
    L = np.where(np.tril(np.ones((n, n), dtype=bool)), L, NEG_INF)
    return softmax_causal(L) @ V, L


def chain(a, b):
    """y_0 = 0, y_i = a_i y_{i-1} + b_i. The label, by its own recurrence."""
    y = np.zeros_like(b)
    for i in range(1, len(b)):
        y[i] = a[i] * y[i - 1] + b[i]
    return y


def path_matrix(a, n):
    """P_ij = prod_{k=j+1}^{i} a_k for j <= i, else 0. Built from the products
    directly, never from cumsum(log a), so it is an independent route to the
    same object the prefix-logit hop produces."""
    P = np.zeros((n, n))
    for i in range(n):
        acc = 1.0
        P[i, i] = 1.0
        for j in range(i - 1, -1, -1):
            acc *= a[j + 1]
            P[i, j] = acc
    return P


def draw(seed=15, lo=0.15, hi=0.85, n=S + 1):
    rng = np.random.default_rng(seed)
    a = np.empty(n)
    a[0] = np.nan                       # BOS has no gate; poisons any misuse
    a[1:] = rng.uniform(lo, hi, size=n - 1)
    b = np.zeros(n)
    b[1:] = rng.normal(size=n - 1)
    g = np.zeros(n)                     # g_0 = 0 by the BOS convention
    g[1:] = np.log(a[1:])
    return a, b, g, rng


def bitwise(x, y):
    return np.asarray(x).tobytes() == np.asarray(y).tobytes()


def line(k, v):
    print(f"  {k:<52s} {v}")


a, b, g, rng = draw()
C = np.cumsum(g)
y = chain(a, b)
P = path_matrix(a, S + 1)
q_trained = rng.normal(size=(S + 1, S + 1))      # a head's own QK logits
V_trained = rng.normal(size=S + 1)               # a head's own values

print(f"S = {S}, positions 0..{S}, gates a_1..a_{S} in (0.15, 0.85), float64")
print(f"label y (from the recurrence)      : {np.array2string(y, precision=6)}")
print(f"|y - P @ b| (two routes to the label): "
      f"{FMT % np.abs(y - P @ b).max()}")
assert np.abs(y - P @ b).max() < 1e-15


# ------------------------------------------------------- 1. THE TWO-BRANCH BIND
print("\n" + "=" * 78)
print("1  THE TWO-BRANCH BIND -- out = softmax(q) @ V1 + lambda * X")
print("=" * 78)
print("""  Claim under test: "lambda = 0 gives standard attention, so (P) holds;
  V1 = 0, lambda = 1 gives the label, so (L) holds; one family, both binds."
  The gate is run against three X, the third of which is the answer key.""")


def two_branch(q, V1, lam, X):
    return std_attention(q, V1) + lam * X


X_gate = P[:, :] @ b                              # the honest gated hop
X_noise = rng.normal(size=S + 1)                  # carries nothing
X_oracle = y.copy()                               # IS the label

ref = std_attention(q_trained, V_trained)
for name, X in (("Wc(g) @ V2, the honest gate", X_gate),
                ("i.i.d. Gaussian noise", X_noise),
                ("the LABEL y itself (answer key)", X_oracle)):
    out0 = two_branch(q_trained, V_trained, 0.0, X)
    p_ok = bitwise(out0, ref)
    outL = two_branch(q_trained, np.zeros(S + 1), 1.0, X)
    l_err = np.abs(outL - y).max()
    line(f"X = {name}", f"(P) bitwise: {p_ok}   (L) max|out - y|: {FMT % l_err}")
    assert p_ok, "the lambda=0 bind failed, which would be a coding error"

print("""
  RULING: VACUOUS. The (P) half passed bitwise for all three X, including one
  that is literally the label. The proof of the lambda=0 bind uses exactly one
  fact -- that lambda multiplies the second summand -- and no property of X.
  Its rejection region over candidate mechanisms is EMPTY, so its expected
  value is PASS before it runs: MISTAKES.md V-10 verbatim. And at the (L)
  setting (V1 = 0) standard attention is absent from the object, so the two
  binds are witnessed by two disjoint objects: MISTAKES.md V-9 read forward.""")


# ---------------------------------------------- 2. ADDITIVE-LOGIT ALONE  (Q1)
print("\n" + "=" * 78)
print("2  ADDITIVE-LOGIT ONLY -- l_ij = q_ij + (C_i - C_j), values = the drives")
print("=" * 78)

out_add, L_add = head(np.zeros((S + 1, S + 1)), g, np.zeros(S + 1), b)
R = np.array([P[i, :i + 1].sum() for i in range(S + 1)])     # the hop's row sum

print("""  ALGEBRA. Any logit of the form (i-only) + (j-only) gives, after softmax,
      A_ij = w_j / W_i,     w_j = exp(s_j - C_j),   W_i = sum_{j<=i} w_j,
  because exp(C_i) is constant across j in row i and cancels between numerator
  and denominator. The target-side half of the path product exp(C_i - C_j) is
  therefore ANNIHILATED, and the row's only i-dependence is the range of the
  running normalizer W_i. With s == 0, q == 0, V = b that is A_ij = P_ij / R_i,
  so out_i = y_i / R_i.\n""")

line("R_i = sum_{j<=i} P_ij (the hop's own row sum)",
     np.array2string(R, precision=6))
line("max |out_i * R_i - y_i|", FMT % np.abs(out_add * R - y).max())
line("max |out_i - y_i|  (the actual error)", FMT % np.abs(out_add - y).max())
assert np.abs(out_add * R - y).max() < 1e-14
assert np.abs(out_add - y).max() > 1e-3

# exp(C_i) is DEAD CODE in this form -- witness, not argument.
# Delete the C_i term entirely and recompute; the output does not move.
mask = np.tril(np.ones((S + 1, S + 1), dtype=bool))
L_noC = np.where(mask, -C[None, :], NEG_INF)      # C_i term simply removed
out_noC = softmax_causal(L_noC) @ b
line("max |out_i - out_i with the C_i term DELETED|",
     FMT % np.abs(out_add - out_noC).max())
assert np.abs(out_add - out_noC).max() < 1e-14

# what the row actually depends on: the running normalizer W_i = sum_{j<=i} w_j
w0 = np.exp(-C)                                   # w_j at s == 0
W0 = np.cumsum(w0)
line("W_i = sum_{j<=i} exp(-C_j)   (s == 0)", np.array2string(W0, precision=6))
line("exp(-C_i)   (what (L) would need W_i to be)",
     np.array2string(np.exp(-C), precision=6))
line("max |W_i - exp(-C_i)|", FMT % np.abs(W0 - np.exp(-C)).max())
assert np.abs(W0 - np.exp(-C)).max() > 1e-3

print("""
  VERDICT ON Q1: NO. The additive-logit form computes the label divided by the
  MULTIPLICATIVE form's own row sum, exactly. And R_i is the same scalar Lean's
  `gate_zero_row_sum` evaluates to i+1 at g == 0. One number is simultaneously
  the obstruction to (P) for the multiplicative hop (a softmax row needs R_i=1)
  and the obstruction to (L) for the additive hop (the label needs R_i=1).
  No choice of q rescues it: exactness for all b forces A_ij = P_ij entrywise,
  hence row sum R_i, hence R_i = 1, and R_i >= 1 + a_i > 1 for i >= 2.""")


# ------------------------------ 3. THE VALUE-FIXED IMPOSSIBILITY THEOREM  (Q2)
print("\n" + "=" * 78)
print("3  ROW-STOCHASTIC WITH THE DRIVES AS VALUES -- the theorem")
print("=" * 78)
print("""  THEOREM. a_k > 0; A causal row-stochastic; values V_j = b_j. If
  sum_{j<=i} A_ij b_j = y_i for every b in R^S, then A_ij = P_ij (test on the
  standard basis), so sum_j A_ij = R_i, and row-stochasticity forces R_i = 1.
  But R_i = 1 + sum_{j<i} P_ij >= 1 + a_i > 1 for i >= 2. Contradiction.\n""")

line("forced row sums R_i (must be 1.0 to be stochastic)",
     np.array2string(R, precision=6))
line("min_i>=2 (R_i - 1)", FMT % (R[2:] - 1.0).min())
assert (R[2:] - 1.0).min() > 0

print("""
  COROLLARY, allowing A to read the data. Take a == 1, b == 1: then y_i = i,
  while every row-stochastic row over V == 1 outputs exactly 1. The failure is
  a RANGE violation, y_i not in conv{V_j}, not a lack of expressiveness.\n""")
a1 = np.ones(S + 1)
b1 = np.ones(S + 1)
b1[0] = 0.0
y1 = chain(a1, b1)
line("a == 1, b == 1: label y_i", np.array2string(y1))
line("any row-stochastic row over V == 1 gives", "1.0 at every i")
line("max_i (y_i - 1)", FMT % (y1 - 1.0).max())
assert (y1 - 1.0).max() == float(S - 1)


# ----------------------------------------------- 4. THE CONSTRUCTION      (Q3)
print("\n" + "=" * 78)
print("4  THE CONSTRUCTION -- one softmax head carrying BOTH binds")
print("=" * 78)
print("""  logit  l_ij = q_ij + (C_i - C_j) + s_j          j = 0..i
  value  V_j

  (P) identity setting : g == 0, s == 0            -> l_ij = q_ij, bitwise attention
  (L) oracle   setting : q == 0, g_j = log a_j, g_0 = 0,
                         s_0 = 0, s_j = log(1 - a_j),
                         V_0 = 0, V_j = b_j / (1 - a_j)

  WHY (L) WORKS. exp(l_i0) = exp(C_i); exp(l_ij) = (1 - a_j) exp(C_i - C_j).
  exp(-C_j)(1 - a_j) = exp(-C_j) - exp(-C_j + g_j) = exp(-C_j) - exp(-C_{j-1}),
  which telescopes over j = 1..i to exp(-C_i) - exp(-C_0) = exp(-C_i) - 1, so
      Z_i = exp(C_i) [ 1 + exp(-C_i) - 1 ] = 1
  EXACTLY: the softmax normalizer is transparent, not because it was divided
  out but because the running sum of the reweighted source terms telescopes to
  the reciprocal of the target factor. The normalizer REPRODUCES exp(C_i)
  instead of cancelling it, which is what section 2's form could not do.\n""")

s_bias = np.zeros(S + 1)
s_bias[1:] = np.log1p(-a[1:])
V_oracle = np.zeros(S + 1)
V_oracle[1:] = b[1:] / (1.0 - a[1:])
q_zero = np.zeros((S + 1, S + 1))

out_L, L_or = head(q_zero, g, s_bias, V_oracle)
Erow = np.exp(np.where(np.isfinite(L_or), L_or, NEG_INF))
Z = Erow.sum(axis=1)
A_or = softmax_causal(np.where(np.tril(np.ones((S + 1, S + 1), dtype=bool)),
                               L_or, NEG_INF))

w_or = np.exp(s_bias - C)
W_or = np.cumsum(w_or)
line("W_i = sum_{j<=i} exp(s_j - C_j)  (s = log(1-a))",
     np.array2string(W_or, precision=6))
line("exp(-C_i)", np.array2string(np.exp(-C), precision=6))
rel_tel = np.abs(W_or - np.exp(-C)).max() / np.exp(-C).max()
line("max |W_i - exp(-C_i)| / max exp(-C_i)   THE TELESCOPING", FMT % rel_tel)
assert rel_tel < 1e-15

line("(L) softmax normalizer Z_i", np.array2string(Z, precision=17))
line("(L) max |Z_i - 1|", FMT % np.abs(Z - 1.0).max())
line("(L) min entry of the attention matrix", FMT % A_or[np.tril_indices(S + 1)].min())
line("(L) max |row sum - 1|", FMT % np.abs(A_or.sum(axis=1) - 1.0).max())
line("(L) max |out_i - y_i|", FMT % np.abs(out_L - y).max())
assert np.abs(Z - 1.0).max() < 1e-15
assert A_or[np.tril_indices(S + 1)].min() > 0.0
assert np.abs(out_L - y).max() < 1e-15

out_P, _ = head(q_trained, np.zeros(S + 1), np.zeros(S + 1), V_trained)
line("(P) bitwise equal to std_attention(q, V)", bitwise(out_P, ref))
assert bitwise(out_P, ref)

print("""
  THE QUERY-SIDE SCAN TERM IS REDUNDANT. Section 2's cancellation applies here
  too: A_ij = w_j / W_i has no exp(C_i) in it. The target factor is recovered
  from the RUNNING NORMALIZER W_i = exp(-C_i), not from the C_i term in the
  logit. So the same construction runs with a KEY-ONLY logit bias:""")
L_key = np.where(mask, (-C + s_bias)[None, :], NEG_INF)
out_key = softmax_causal(L_key) @ V_oracle
line("key-only logit l_ij = -C_j + log(1 - a_j): max|out - y|",
     FMT % np.abs(out_key - y).max())
line("  vs the C_i - C_j version: max|difference|",
     FMT % np.abs(out_key - out_L).max())
assert np.abs(out_key - y).max() < 1e-15
assert np.abs(out_key - out_L).max() < 1e-15

print("\n  PLANTED NEGATIVES -- this bind has a rejection region and it is occupied.")
for name, gg, ss, VV in (
        ("drop the key bias s (s == 0)", g, np.zeros(S + 1), V_oracle),
        ("drop the value rescale (V_j = b_j)", g, s_bias, b),
        ("drop the BOS sink (V_0 = 1 not 0)", g, s_bias,
         np.concatenate([[1.0], V_oracle[1:]])),
        ("wrong bias s_j = log(1 - a_j) / 2", g, s_bias / 2.0, V_oracle)):
    o, _ = head(q_zero, gg, ss, VV)
    line(f"  {name}", f"max|out - y| = {FMT % np.abs(o - y).max()}")
    assert np.abs(o - y).max() > 1e-6, f"plant '{name}' did not fire"

print("\n  CLASS CLOSURE -- the family never leaves the softmax class.")
for k in range(3):
    gk = rng.normal(scale=0.7, size=S + 1)
    sk = rng.normal(scale=0.7, size=S + 1)
    _, Lk = head(q_trained, gk, sk, V_trained)
    Ak = softmax_causal(np.where(np.tril(np.ones((S + 1, S + 1), dtype=bool)),
                                 Lk, NEG_INF))
    line(f"  random theta #{k}: max|rowsum - 1|, min entry",
         f"{FMT % np.abs(Ak.sum(1) - 1).max()}, {FMT % Ak[np.tril_indices(S+1)].min()}")
    assert np.abs(Ak.sum(1) - 1).max() < 1e-15 and Ak[np.tril_indices(S + 1)].min() > 0

print("""
  VERDICT ON Q3: A CONSTRUCTION EXISTS, and therefore the impossibility the
  node hoped for is FALSE. Softmax's normalizer is not the obstruction to
  forming path products. The obstruction was only ever to forming them with
  the DRIVES as values; rescaling the values by the position-local factor
  1/(1 - a_j) and giving BOS a value-zero slot dissolves it.""")


# -------------------------------------------- 5. WHAT THE NORMALIZER CHARGES
print("\n" + "=" * 78)
print("5  THE VALUE-SCALE CHARGE -- what survives of the impossibility")
print("=" * 78)
print("""  THEOREM. For any causal row-stochastic A and any V, out_i lies in
  conv{V_j : j <= i}, so max_i |out_i| <= max_j |V_j|. Reproducing the label
  therefore costs value dynamic range max_j |V_j| >= max_i |y_i|. That is the
  entire residue of the impossibility: a change of units, not a barrier.\n""")
print(f"  {'a':>10s} {'max|y_i| (needed)':>20s} {'max|V_j| (used)':>18s} "
      f"{'ratio':>10s} {'1/(1-a)':>12s}")
for aa in (0.5, 0.9, 0.99, 0.999, 1 - 1e-6):
    av = np.full(S + 1, aa)
    bv = np.ones(S + 1)
    bv[0] = 0.0
    yv = chain(av, bv)
    Vv = np.zeros(S + 1)
    Vv[1:] = bv[1:] / (1.0 - av[1:])
    print(f"  {aa:>10.6f} {np.abs(yv).max():>20.10g} {np.abs(Vv).max():>18.10g} "
          f"{np.abs(Vv).max() / np.abs(yv).max():>10.4g} {1/(1-aa):>12.6g}")
print("""
  The bound is attained as a^S -> 0 (memory shorter than the context) and is
  loose by 1/(1 - a^S) when a^S -> 1. The construction pays 1/(1 - a), the
  S -> infinity optimum. A constant rescale gamma = max_i R_i would pay less at
  finite S, but its sink logit log(1 - R_i/gamma) is an arbitrary function of i
  and is NOT a scan difference -- so the architecture cannot compute it. The
  (1 - a_j) reweighting is the unique choice whose sink logit is the scan's own
  output, which is why it is the one to build.""")


# ------------------------------------------------------ 6. THE SIGNED VARIANT
print("\n" + "=" * 78)
print("6  THE SIGNED VARIANT -- where the ruling does not save it")
print("=" * 78)
print("""  Lean #3 gives chi(P_i - P_j) = chi(P_i) * chi(P_j) in ZMod 2, so the sign
  FACTORS into a source half and a target half. The source half rides in V_j.
  The target half chi(P_i) is constant across j in row i, so a row-stochastic
  (hence non-negative) weight cannot carry it: it has to be a per-position
  multiplier on the OUTPUT.\n""")
sgn = rng.integers(0, 2, size=S + 1)
sgn[0] = 0
a_s = a.copy()
a_s[1:] = np.where(sgn[1:] == 1, -a[1:], a[1:])
y_s = chain(a_s, b)
Pxor = np.cumsum(sgn) % 2
chi = np.where(Pxor == 0, 1.0, -1.0)
V_sgn = np.zeros(S + 1)
V_sgn[1:] = chi[1:] * b[1:] / (1.0 - a[1:])
out_sgn, _ = head(q_zero, g, s_bias, V_sgn)
line("max |chi_i * out_i - y_signed_i|", FMT % np.abs(chi * out_sgn - y_s).max())
line("max |out_i - y_signed_i| (no output gate)",
     FMT % np.abs(out_sgn - y_s).max())
assert np.abs(chi * out_sgn - y_s).max() < 1e-14

print("""
  So the signed carrier IS exact -- but only as O_i = chi(P_i) * (head read-out),
  and `chi(P_i) == 1 at zero sign bits` is a MULTIPLICATIVE off switch that any
  O = gamma(theta) * A passes. By the rule stated in section 1 that bind is
  vacuous unless chi(P_i) is applied by machinery already inside the transformer
  class (a following MLP reading chi(P_i) off position i's own residual), not by
  an output gate bolted to the head. This node does not settle which; it is the
  one open item handed back.""")

# ---------------------------- 7. THE PER-ROW-SCALAR AMENDMENT (L-AMEND route)
print("\n" + "=" * 78)
print("7  THE PER-ROW SCALAR -- 'let the readout divide Z_i back out'")
print("=" * 78)
print("""  The candidate route: keep the additive-logit form (which has (P) by
  `gate_zero_logit_identity`), accept that it produces the path product only up
  to a per-row scalar, and have the readout supply that scalar rho_i.

  WHAT rho_i HAS TO BE. With q == 0, s == 0 the row is A_ij = exp(-C_j) / W_i,
  so recovering P_ij = exp(C_i - C_j) needs exactly

      rho_i = exp(C_i) * W_i        (W_i = the head's own softmax denominator)

  and then rho_i * A_ij = exp(C_i) W_i exp(-C_j) / W_i = exp(C_i - C_j) = P_ij.
  The scalar is not a free constant. It is the normalizer, put back.\n""")

rho_tied = np.exp(C) * W0                          # W0 = cumsum(exp(-C)) from sec 2
A_add = np.zeros((S + 1, S + 1))
for i in range(S + 1):
    A_add[i, :i + 1] = np.exp(-C[:i + 1]) / W0[i]
line("(L) max |rho_i * A_ij - P_ij|  entrywise",
     FMT % np.abs(rho_tied[:, None] * A_add - P).max())
line("(L) max |rho_i * out_i - y_i|", FMT % np.abs(rho_tied * out_add - y).max())
assert np.abs(rho_tied[:, None] * A_add - P).max() < 1e-13
assert np.abs(rho_tied * out_add - y).max() < 1e-14

print("""
  So (L) is recovered -- and the effective operator rho_i * A_ij IS S-M's
  unnormalized hop, entry for entry. An operator that divides its own
  normalizer back out is the unnormalized operator wearing a softmax, and
  `gate_zero_not_stochastic` applies to it verbatim. Measured at g == 0:\n""")

C0 = np.zeros(S + 1)
for qname, qq in (("q == 0", q_zero), ("q = a trained head's logits", q_trained)):
    Lq = np.where(mask, qq, NEG_INF)
    Zq = np.exp(np.where(mask, qq, NEG_INF)).sum(axis=1)
    rho0 = np.exp(C0) * Zq                          # tied rho at g == 0
    out_tied0 = rho0 * (softmax_causal(Lq) @ V_trained)
    out_std0 = softmax_causal(Lq) @ V_trained
    line(f"  {qname}: rho_i at g == 0", np.array2string(rho0, precision=4))
    line("    max |tied output - standard attention|",
         FMT % np.abs(out_tied0 - out_std0).max())
    assert np.abs(rho0 - 1.0).max() > 0.5
    assert np.abs(out_tied0 - out_std0).max() > 1e-3

print("""  (P) FAILS for the tied scalar: at g == 0 it is forced to Z_i, which is i+1
  at q == 0 -- Lean's `gate_zero_row_sum` exactly -- and is the head's own
  denominator in general. Never 1.

  THE ONLY WAY OUT is to make rho a FREE coordinate switched off at parity.
  That is O = gamma(theta) * A with gamma(theta_0) = 1, and it has section 1's
  defect in multiplicative form:\n""")

for name, rr in (("rho = R_i, which solves (L)", R),
                 ("rho = i.i.d. positive noise", np.abs(rng.normal(size=S + 1)) + 0.5),
                 ("rho = the answer key y_i / out_i", y / np.where(out_add == 0, 1, out_add))):
    out_free_P = 1.0 * ref                          # rho == 1, the identity setting
    line(f"  {name}", f"(P) at rho == 1 bitwise: {bitwise(out_free_P, ref)}")
    assert bitwise(out_free_P, ref)

print("""
  VERDICT ON THE PER-ROW SCALAR: A DICHOTOMY, AND NEITHER BRANCH IS ADMISSIBLE.
  Tie rho to the gate mechanism and (P) fails, because g == 0 forces rho = Z_i
  and Z_i is never 1. Leave rho free with an off switch and (P) holds and is
  VACUOUS, passed by every gamma * A including the answer key. There is no
  third option: rho_i must equal exp(C_i) W_i for (L), and that quantity equals
  1 at g == 0 only if W_i = 1, which happens only at i = 0.

  The label is not defined up to a per-position scale. R_i runs 1.00 to 2.18
  across positions in this draw, so absorbing it is a per-position magnitude
  error, not a gauge freedom. Section 4's construction is the amendment that
  avoids the dichotomy: it does not divide the normalizer out, it arranges for
  the normalizer to telescope to the target factor on its own.""")

print("\n" + "=" * 78)
print("ALL ASSERTS PASSED.")
print("=" * 78)

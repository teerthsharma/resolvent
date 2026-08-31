"""[V-eq] probe 4 -- negative-eigenvalue linear RNNs / the signed parity mask.
Grazzi, Siems, Zela, Franke, Hutter, Pontil 2024/25, arXiv:2411.12537.
Eq (1): H_i = A(x_i) H_{i-1} + B(x_i); Thm 1: a finite-precision LRNN can solve
parity only if some layer has A(x) with an eigenvalue lambda not in {x >= 0}.
Reparametrisation for diagonal (Mamba): A^{diag-}(x) := Diag(2 s(x) - 1), s in [0,1].
Checked here: the CEQ v15 "parity mask" (-1)^{P_i XOR P_j}, P = prefix-XOR of sign
bits, IS the sign of the semiseparable product prod_{k=i+1..j} a_k.
"""
import numpy as np
rng = np.random.default_rng(3)
T = 12
s = rng.uniform(0, 1, T)
a = 2 * s - 1                                  # eigenvalues in [-1,1]  (Grazzi et al.)
signbit = (a < 0).astype(int)
Pxor = np.cumsum(signbit) % 2                  # prefix-XOR of sign bits

n_ = np.arange(T)[:, None]; m_ = np.arange(T)[None, :]
mask_parity = np.where(n_ >= m_, (-1.0) ** ((Pxor[:, None] ^ Pxor[None, :])), 0.0)
mask_prod = np.zeros((T, T))
for j in range(T):
    for i in range(j + 1):
        mask_prod[j, i] = np.sign(np.prod(a[i + 1:j + 1])) if j > i else 1.0
print("sign(prod a) vs (-1)^(P_i XOR P_j):", np.abs(mask_parity - mask_prod).max())

# parity is computable with a_t in [-1,1] and NOT with a_t in [0,1]
bits = rng.integers(0, 2, 64)
a_signed = np.where(bits == 1, -1.0, 1.0)
h = 1.0
for t in range(64):
    h = a_signed[t] * h                        # h_t = A(x_t) h_{t-1}
parity = bits.sum() % 2
print("signed LRNN state           :", h, " true parity:", parity,
      " decoded:", int(h < 0), " match:", int(h < 0) == parity)

# nonnegative-eigenvalue LRNN: state is a monotone function of the count of 1s only
def run_nonneg(bits, a0=0.7, a1=0.3, b0=0.1, b1=0.9):
    h = 0.0
    for b in bits:
        h = (a1 if b else a0) * h + (b1 if b else b0)
    return h
import itertools
xs = list(itertools.product([0, 1], repeat=8))
even = [run_nonneg(x) for x in xs if sum(x) % 2 == 0]
odd = [run_nonneg(x) for x in xs if sum(x) % 2 == 1]
ov = min(max(even), max(odd)) - max(min(even), min(odd))
print("nonneg-eig LRNN: even/odd state ranges overlap by:", round(ov, 6),
      "-> no threshold readout separates parity (Thm 1)")

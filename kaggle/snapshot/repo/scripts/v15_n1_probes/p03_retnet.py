"""[V-eq] probe 3 -- RetNet retention decay mask.
Sun, Dong, Huang, Ma, Wang, Ma, Wei 2023, arXiv:2307.08621.
Eq (1): s_n = A s_{n-1} + K_n^T v_n ; o_n = Q_n s_n = sum_{m<=n} gamma^{n-m} (Q_n K_m^T) v_m
Eq (5): Retention(X) = (Q K^T . D) V, D_nm = gamma^{n-m} (n>=m), 0 otherwise
Eq (8): gamma = 1 - 2^{-5-arange(0,h)}
"""
import numpy as np
rng = np.random.default_rng(2)
T, d, dv = 8, 4, 3
gamma = 0.9
Q = rng.normal(size=(T, d)); K = rng.normal(size=(T, d)); V = rng.normal(size=(T, dv))

s = np.zeros((d, dv)); O_rec = np.zeros((T, dv))
for n in range(T):
    s = gamma * s + np.outer(K[n], V[n])
    O_rec[n] = Q[n] @ s

n_ = np.arange(T)[:, None]; m_ = np.arange(T)[None, :]
D = np.where(n_ >= m_, gamma ** (n_ - m_), 0.0)
O_par = ((Q @ K.T) * D) @ V

Dw = np.where(n_ >= m_, gamma ** (n_ - m_ + 1), 0.0)   # control: off-by-one exponent
O_wrong = ((Q @ K.T) * Dw) @ V

print("recurrent vs parallel (Eq 5) :", np.abs(O_rec - O_par).max())
print("off-by-one exponent control  :", np.abs(O_rec - O_wrong).max())
h = 4
print("Eq(8) gamma schedule h=4     :", (1 - 2.0 ** (-5 - np.arange(0, h))))
print("gamma==1 -> causal all-ones  :", np.allclose(np.where(n_ >= m_, 1.0, 0.0),
                                                    np.tril(np.ones((T, T)))))

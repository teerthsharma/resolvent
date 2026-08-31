"""[V-eq] probe 1 -- SSD / Mamba-2 semiseparable mask.
Dao & Gu 2024, arXiv:2405.21060, Def 3.2 / Eq (3): M_ji = C_j^T A_{j:i} B_i,
A_{j:i} = A_j A_{j-1} ... A_{i+1}; scalar-identity case A_t = a_t I gives the
1-semiseparable mask L_ji = prod_{k=i+1..j} a_k and dual form Y = (L . C B^T) X.
Fails loudly if the product range is transcribed off by one.
"""
import numpy as np
rng = np.random.default_rng(0)
T, N, P = 7, 3, 2
a = rng.uniform(0.3, 0.95, T)                 # scalar gates a_t in (0,1)
B = rng.normal(size=(T, N)); C = rng.normal(size=(T, N)); X = rng.normal(size=(T, P))

# (i) recurrent scan: h_t = a_t h_{t-1} + B_t x_t^T ; y_t = C_t^T h_t
h = np.zeros((N, P)); Y_scan = np.zeros((T, P))
for t in range(T):
    h = a[t] * h + np.outer(B[t], X[t]); Y_scan[t] = C[t] @ h

# (ii) matrix form with M_ji = C_j^T A_{j:i} B_i, A_{j:i} = prod_{k=i+1}^{j} a_k
M = np.zeros((T, T))
for j in range(T):
    for i in range(j + 1):
        M[j, i] = np.prod(a[i + 1:j + 1]) * (C[j] @ B[i])
Y_mat = M @ X

# (iii) dual "masked kernel attention" form (L . C B^T) X, L = 1-semiseparable
L = np.tril(np.array([[np.prod(a[i + 1:j + 1]) if j >= i else 0.0
                       for i in range(T)] for j in range(T)]))
Y_dual = (L * (C @ B.T)) @ X

# control: off-by-one product prod_{k=i}^{j} (WRONG transcription)
Lw = np.tril(np.array([[np.prod(a[i:j + 1]) if j >= i else 0.0
                        for i in range(T)] for j in range(T)]))
Y_wrong = (Lw * (C @ B.T)) @ X

print("scan vs M_ji form        :", np.abs(Y_scan - Y_mat).max())
print("scan vs dual (L o CB^T)  :", np.abs(Y_scan - Y_dual).max())
print("off-by-one control       :", np.abs(Y_scan - Y_wrong).max())
print("a_t == 1 -> L is all-ones causal:", np.allclose(
    np.tril(np.ones((T, T))),
    np.tril(np.array([[np.prod(np.ones(T)[i+1:j+1]) if j >= i else 0.0
                       for i in range(T)] for j in range(T)]))))

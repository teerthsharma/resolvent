"""[V-eq] probe 2 -- Gated Linear Attention.
Yang, Wang, Shen, Panda, Kim 2024, arXiv:2312.06635.
Eq (3): S_t = (alpha_t^T 1) . S_{t-1} + k_t^T v_t = Diag(alpha_t) S_{t-1} + k_t^T v_t
Eq (4): P_ij = sum_k Q_ik K_jk exp(log B_ik - log B_jk), i>=j ; O = (P . M) V,
        B_t = prod_{j<=t} alpha_j (cumulative product of gates).
"""
import numpy as np
rng = np.random.default_rng(1)
T, dk, dv = 6, 4, 3
Q = rng.normal(size=(T, dk)); K = rng.normal(size=(T, dk)); V = rng.normal(size=(T, dv))
alpha = rng.uniform(0.2, 0.99, size=(T, dk))        # alpha_t in (0,1)^{d_k}

S = np.zeros((dk, dv)); O_rec = np.zeros((T, dv))
for t in range(T):
    S = alpha[t][:, None] * S + np.outer(K[t], V[t])
    O_rec[t] = Q[t] @ S

Bm = np.cumprod(alpha, axis=0)                       # B_t
P = np.zeros((T, T))
for i in range(T):
    for j in range(i + 1):
        P[i, j] = np.sum(Q[i] * K[j] * np.exp(np.log(Bm[i]) - np.log(Bm[j])))
O_par = (P * np.tril(np.ones((T, T)))) @ V

# control: forget the cumulative product (use alpha_t itself as B_t)
Pw = np.zeros((T, T))
for i in range(T):
    for j in range(i + 1):
        Pw[i, j] = np.sum(Q[i] * K[j] * np.exp(np.log(alpha[i]) - np.log(alpha[j])))
O_wrong = (Pw * np.tril(np.ones((T, T)))) @ V

print("GLA recurrent vs parallel Eq(4):", np.abs(O_rec - O_par).max())
print("no-cumprod control            :", np.abs(O_rec - O_wrong).max())

# alpha == 1 collapses GLA to ungated linear attention (Eq 1)
S = np.zeros((dk, dv)); O1 = np.zeros((T, dv))
for t in range(T):
    S = S + np.outer(K[t], V[t]); O1[t] = Q[t] @ S
print("alpha==1 vs linear attention  :", np.abs(
    O1 - (np.tril(Q @ K.T)) @ V).max())

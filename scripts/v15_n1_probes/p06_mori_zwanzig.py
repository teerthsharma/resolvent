"""[V-eq] probe 6 -- Mori-Zwanzig / Dyson identity, the projection theorem.
Lin & Lu, "Data-driven model reduction, Wiener projections, and the Koopman-
Mori-Zwanzig formalism", JCP 2021, arXiv:1908.07725.
  Eq (2.5) Dyson: M^{n+1} = sum_{k=0}^{n} M^{n-k} P M (Q M)^k + (Q M)^{n+1},
      with M the Koopman operator, P an orthogonal projection, Q = I - P.
  Eq (2.2): x_{n+1} = P F(x_n) + sum_{k=1}^{n} Gamma_k(x_{n-k}) + xi_{n+1}
      = Markov term + memory kernel + orthogonal-dynamics (noise) term.
Ma, Wang & E 2018, arXiv:1808.04258 Eq (11): the same split in GLE form.
The probe verifies Dyson's identity as an operator identity at finite dimension:
if the identity were transcribed wrong the residual would not be ~1e-13.
"""
import numpy as np
rng = np.random.default_rng(5)
dim, r = 9, 3
M = rng.normal(size=(dim, dim)) / np.sqrt(dim)     # Koopman operator (finite section)
U, _ = np.linalg.qr(rng.normal(size=(dim, r)))
P = U @ U.T                                        # orthogonal projection, P^2 = P
Q = np.eye(dim) - P
print("P idempotent, P=P^T          :", np.abs(P @ P - P).max(), np.abs(P - P.T).max())
for n in [0, 1, 2, 5]:
    lhs = np.linalg.matrix_power(M, n + 1)
    rhs = sum(np.linalg.matrix_power(M, n - k) @ P @ M @ np.linalg.matrix_power(Q @ M, k)
              for k in range(n + 1)) + np.linalg.matrix_power(Q @ M, n + 1)
    print("Dyson (2.5) residual n=%d     : %.3e" % (n, np.abs(lhs - rhs).max()))
# control: drop the memory sum -> Markov + noise only
n = 5
lhs = np.linalg.matrix_power(M, n + 1)
markov_only = np.linalg.matrix_power(M, n) @ P @ M + np.linalg.matrix_power(Q @ M, n + 1)
print("Markov+noise WITHOUT memory  : %.3e  (this is the size of the memory kernel)"
      % np.abs(lhs - markov_only).max())

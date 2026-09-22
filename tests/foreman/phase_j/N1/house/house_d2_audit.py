"""House's adversarial audit of Addendum N's D2 (contract §7): where does
'path gates are similarity transforms' break? float64/complex128, default_rng(0)."""
import numpy as np
rng = np.random.default_rng(0)
n, d = 48, 8
q, k = rng.standard_normal((n, d)), rng.standard_normal((n, d))
s = q @ k.T / np.sqrt(d)
tri = np.tril(np.ones((n, n), bool))
theta = rng.uniform(-np.pi, np.pi, n)
Phi = np.cumsum(theta)                       # Pi_i = e^{i Phi_i}; G_ij = e^{i(Phi_i - Phi_j)}
Gphase = np.exp(1j * (Phi[:, None] - Phi[None, :])) * tri

def op(G, beta, zmode="abs"):
    e = np.exp(s) * tri
    Z = (np.abs(G) * e).sum(1) if zmode == "abs" else (G * e).sum(1)
    return (G * e) / (Z[:, None] ** beta)

D = np.diag(np.exp(1j * Phi))
out = {}
for beta in (1.0, 1.0906, 0.8996):
    W0 = op(tri.astype(complex), beta)
    WG = op(Gphase, beta)
    out[f"a_similarity_resid_beta{beta}"] = float(np.abs(WG - D @ W0 @ np.linalg.inv(D)).max())
# (b) the head reads Re(W_G V): the real operator is A = Re(W_G) = W0 o cos(Phi_i - Phi_j)
W0 = op(tri.astype(complex), 1.0).real
A = op(Gphase, 1.0).real
out["b_hadamard_identity_resid"] = float(np.abs(A - W0 * np.cos(Phi[:, None] - Phi[None, :]) * tri).max())
out["b_eig_minus_diag_A"] = float(np.abs(np.sort(np.linalg.eigvals(A).real) - np.sort(np.diag(A))).max())
out["b_diag_A_minus_diag_W0"] = float(np.abs(np.diag(A) - np.diag(W0)).max())
def gain(M):            # right eigenvector for eigenvalue 1 with u_0 = 1 (absorber at 0)
    u = np.zeros(n); u[0] = 1.0
    for i in range(1, n):
        u[i] = (M[i, :i] @ u[:i]) / (1 - M[i, i])
    return u
uW, uA = gain(W0), gain(A)
out["b_equilibrium_gain_W0_max_abs_minus_1"] = float(np.abs(uW - 1).max())
out["b_equilibrium_gain_A_range"] = [float(uA.min()), float(uA.max())]
out["b_negative_entries_in_A"] = int((A < -1e-15).sum())
# same A reached by any diagonal similarity of W0? |entries| must match: they do not
out["b_max_abs_entry_gap_A_vs_W0"] = float(np.abs(np.abs(A) - np.abs(W0)).max())
# (c) a Z that sums G instead of |G|
out["c_diag_shift_Z_sums_G"] = float(np.abs(np.diag(op(Gphase, 1.0, "signed")) - np.diag(op(tri.astype(complex), 1.0))).max())
for kk, vv in out.items(): print(f"{kk}: {vv}")

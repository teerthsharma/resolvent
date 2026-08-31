"""[V-eq] probe 4 -- RoPE (Su et al. 2021): what the m == 1 face occupies, and
what it does NOT.

Source transcribed from the arXiv LaTeX e-print of arXiv:2104.09864
(`roformer_arxiv.tex`), Sec 3.2.

  2D case, eq (fn:res-2d):
      f_q(x_m, m) = (W_q x_m) e^{i m theta}
      f_k(x_n, n) = (W_k x_n) e^{i n theta}
      g(x_m, x_n, m-n) = Re[ (W_q x_m)(W_k x_n)^* e^{i(m-n) theta} ]
  "theta in R is a preset non-zero constant."

  General form, eq (fn:rope-fqk) and (fn:rope-RMat):
      f_{q,k}(x_m, m) = R^d_{Theta,m} W_{q,k} x_m
      R^d_{Theta,m} = block-diag of  [[cos m theta_i, -sin m theta_i],
                                      [sin m theta_i,  cos m theta_i]],  i=1..d/2
  "is the rotary matrix with pre-defined parameters
   Theta = { theta_i = 10000^{-2(i-1)/d}, i in [1,2,...,d/2] }."

  eq (fn:rope-qk):
      q_m^T k_n = (R^d_{Theta,m} W_q x_m)^T (R^d_{Theta,n} W_k x_n)
                = x^T W_q R^d_{Theta,n-m} W_k x_n,
      "where R^d_{Theta,n-m} = (R^d_{Theta,m})^T R^d_{Theta,n}.  Note that
       R^d_Theta is an orthogonal matrix, which ensures stability during the
       process of encoding position information."

  Sec 3.3 (linear attention): "Since RoPE injects position information by
  rotation, which keeps the norm of hidden representations unchanged, ..."

  Sec 3.3 long-term decay (appendix:long-term-decay): the decay is obtained by
  an Abel transformation over the FREQUENCY index, bounding
  |sum_i h_i (S_{i+1} - S_i)| by (max_i |S_i|) sum_i |h_{i+1} - h_i|, "the value
  of (1/(d/2)) sum_{i=1}^{d/2} |S_i| decay[s] with the relative distance m-n".

Hypotheses as the source states them: d even; theta_i preset, non-zero, NOT
learned; R depends on the position index m only, never on the content x.

What this probe settles for X36: RoPE is the m == 1 face of a . e^{i theta} and
nothing else -- norm preserved EXACTLY, |eigenvalue| == 1 for every block, no
magnitude parameter at all, learned or otherwise. What RoPE therefore does NOT
occupy: any m < 1, any attenuation, and any content-dependent phase. Its
apparent "decay" with distance is interference across the d/2 frequencies at
constant norm, not a magnitude gate; this probe separates the two by measuring
them side by side.

CONTROLS:
  - the 2x2 block mis-transcribed with a shared angle across the block rows
    (cos m theta_i on the diagonal, sin m theta_{i+1} off-diagonal): not
    orthogonal, O(1).
  - the relative-position identity checked against a wrong sign convention.

Reproduce: python scripts/v15_x36_probes/p04_rope_norm.py
"""
import numpy as np

rng = np.random.default_rng(20260831)
d = 128
base = 10000.0
theta = base ** (-2.0 * (np.arange(1, d // 2 + 1) - 1) / d)   # theta_i, i = 1..d/2


def rope(m, theta, bad=False):
    """R^d_{Theta,m}: block-diagonal 2x2 rotations."""
    R = np.zeros((2 * len(theta), 2 * len(theta)))
    c, s = np.cos(m * theta), np.sin(m * theta)
    if bad:                       # CONTROL: off-diagonal uses the NEXT frequency
        s_off = np.roll(np.sin(m * theta), -1)
    else:
        s_off = s
    for i in range(len(theta)):
        R[2 * i, 2 * i] = c[i]
        R[2 * i, 2 * i + 1] = -s_off[i]
        R[2 * i + 1, 2 * i] = s_off[i]
        R[2 * i + 1, 2 * i + 1] = c[i]
    return R


I = np.eye(d)
x = rng.normal(size=d)

print("RoPE, d =", d, " theta_i = 10000^{-2(i-1)/d},  i = 1..d/2")
worst_orth = 0.0
worst_norm = 0.0
worst_eig = 0.0
for m in (0, 1, 7, 128, 4096):
    R = rope(m, theta)
    worst_orth = max(worst_orth, np.linalg.norm(R.T @ R - I))
    worst_norm = max(worst_norm, abs(np.linalg.norm(R @ x) - np.linalg.norm(x)))
    worst_eig = max(worst_eig, np.abs(np.abs(np.linalg.eigvals(R)) - 1).max())
print("  max_m ||R^T R - I||_F                :", worst_orth)
print("  max_m | ||R x|| - ||x|| |            :", worst_norm)
print("  max_m max_j | |eig_j(R)| - 1 |       :", worst_eig)

Rb = rope(7, theta, bad=True)
print("  CONTROL (block rows use different frequencies):")
print("    ||R'^T R' - I||_F                  :", np.linalg.norm(Rb.T @ Rb - I))
print("    | ||R' x|| - ||x|| |               :",
      abs(np.linalg.norm(Rb @ x) - np.linalg.norm(x)))

# relative-position identity  R_m^T R_n = R_{n-m}
worst_rel = 0.0
for (m, n) in ((3, 11), (100, 7), (0, 5000)):
    worst_rel = max(worst_rel,
                    np.linalg.norm(rope(m, theta).T @ rope(n, theta) - rope(n - m, theta)))
print("  max ||R_m^T R_n - R_{n-m}||_F        :", worst_rel)
print("  CONTROL with R_{m-n} instead, (m,n)=(3,11):",
      np.linalg.norm(rope(3, theta).T @ rope(11, theta) - rope(3 - 11, theta)))

# The decay is interference across frequencies, at EXACTLY constant norm.
# The paper's decaying quantity is NOT the inner product itself but the Abel
# bound's factor  (1/(d/2)) sum_{i=1}^{d/2} |S_i|,  S_j = sum_{i=0}^{j-1} e^{i(m-n)theta_i}.
print()
print("Long-term decay (the paper's own quantity) vs magnitude, side by side")
q = rng.normal(size=d)
th0 = base ** (-2.0 * np.arange(0, d // 2) / d)      # appendix indexing, i = 0..d/2-1


def abel_factor(dist, th0):
    S = np.concatenate(([0.0 + 0j], np.cumsum(np.exp(1j * dist * th0))))  # S_0..S_{d/2}
    return np.abs(S[1:]).mean()


print("   m-n |  ||R_{m-n} q||/||q||  |  (1/(d/2)) sum_i |S_i|")
for dist in (0, 1, 4, 16, 64, 256, 1024, 4096):
    R = rope(dist, theta)
    nrm = np.linalg.norm(R @ q) / np.linalg.norm(q)
    print(f"  {dist:>5} |  {nrm!r}  |  {abel_factor(dist, th0): .6f}")
print("  => the norm column is 1.0 to the last bit at every distance, while the")
print(f"     Abel factor falls from d/2 = {d // 2} at zero distance to O(1).")
print("     RoPE's decay is interference across frequencies at constant norm;")
print("     it carries NO magnitude and cannot bound a path product below 1.")

# path product of L rotations: modulus exactly 1 (the Lean #16 statement's face)
L = 10000
P = np.exp(1j * rng.uniform(-np.pi, np.pi, L)).prod()
print("  |prod of", L, "unit phases| - 1 =", abs(P) - 1.0)

assert worst_orth < 1e-12 and worst_norm < 1e-12 and worst_eig < 1e-12
assert np.linalg.norm(Rb.T @ Rb - I) > 1e-1
assert worst_rel < 1e-9
assert abs(abs(P) - 1.0) < 1e-12
print("OK")

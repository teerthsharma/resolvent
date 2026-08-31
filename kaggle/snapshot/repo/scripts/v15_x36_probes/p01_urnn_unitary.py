"""[V-eq] probe 1 -- Arjovsky/Shah/Bengio 2016 uRNN: what is EXACTLY constrained.

Source transcribed (arXiv:1511.06464, LaTeX source `uRNN.tex` from arXiv e-print):

  Lemma (sec.4, cited to Horn & Johnson): "A complex square matrix W is unitary
    if and only if it has an eigendecomposition of the form W = V D V*, where *
    denotes the conjugate transpose. Here V, D in C^{nxn} are complex matrices,
    where V is unitary, and D is a diagonal such that |D_{j,j}| = 1."

  The four unitary building blocks (sec.4):
    D    a diagonal matrix with D_{j,j} = e^{i w_j}, with parameters w_j in R,
    R    = I - 2 v v* / ||v||^2, a reflection matrix in the complex vector v in C^n,
    Pi   a fixed random index permutation matrix,
    F, F^{-1}  the Fourier and inverse Fourier transforms.

  The composition actually used (eq. after "we settled on the following composition"):
    W = D_3 R_2 F^{-1} D_2 Pi R_1 F D_1                                     (*)

  modReLU (sec.5.2), the ONLY magnitude-touching map in the architecture:
    sigma_modReLU(z) = (|z|+b) z/|z|   if |z| + b >= 0
                     = 0               if |z| + b <  0
    equivalently  sigma_modReLU(z) = sigma_ReLU(|z| + b) * z/|z|.

Hypotheses as the source states them: the four blocks are each unitary; products
of unitary matrices are unitary; F is the unitary DFT (the paper uses the
norm-preserving normalisation, else the composition is not unitary); b in R is
learned, one per hidden dimension.

What this probe settles for X36: the uRNN's TRANSITION eigenvalues are on the
unit circle EXACTLY -- |lambda| == 1 identically, a single point, not an
interval. The delta's family a = m e^{i theta} restricted to m == 1 is exactly
this. uRNN therefore occupies the m == 1 FACE and nothing else. The magnitude
degree of freedom in the uRNN lives in modReLU, whose per-unit magnitude gain
attains 0 EXACTLY (closed below) and is UNBOUNDED above (no cap at all).

CONTROL: the reflection with the factor 2 dropped, R' = I - v v*/||v||^2, which
is the single most likely mis-transcription of a Householder. It is not unitary
and the deviation is O(1), not O(1e-16).

Reproduce: python scripts/v15_x36_probes/p01_urnn_unitary.py
"""
import numpy as np

rng = np.random.default_rng(20260831)
n = 64


def dft(n):
    """Unitary DFT matrix (norm-preserving normalisation)."""
    j = np.arange(n)
    return np.exp(-2j * np.pi * np.outer(j, j) / n) / np.sqrt(n)


def diag_phase(n, rng):
    """D_{j,j} = e^{i w_j}, w_j ~ U[-pi, pi] (the paper's initialisation)."""
    w = rng.uniform(-np.pi, np.pi, n)
    return np.diag(np.exp(1j * w))


def reflection(v, factor=2.0):
    """R = I - factor * v v* / ||v||^2.  factor=2 is the source; factor=1 is the control."""
    v = v.reshape(-1, 1)
    return np.eye(len(v)) - factor * (v @ v.conj().T) / (v.conj().T @ v).real


def perm(n, rng):
    P = np.eye(n)
    return P[rng.permutation(n)]


F = dft(n)
Finv = F.conj().T
D1, D2, D3 = (diag_phase(n, rng) for _ in range(3))
v1 = rng.normal(size=n) + 1j * rng.normal(size=n)
v2 = rng.normal(size=n) + 1j * rng.normal(size=n)
Pi = perm(n, rng)

R1, R2 = reflection(v1), reflection(v2)
W = D3 @ R2 @ Finv @ D2 @ Pi @ R1 @ F @ D1

R1b, R2b = reflection(v1, 1.0), reflection(v2, 1.0)
Wbad = D3 @ R2b @ Finv @ D2 @ Pi @ R1b @ F @ D1

I = np.eye(n)
h = rng.normal(size=n) + 1j * rng.normal(size=n)

ev = np.linalg.eigvals(W)
evb = np.linalg.eigvals(Wbad)

print("uRNN W = D3 R2 F^-1 D2 Pi R1 F D1, n =", n)
print("  ||W* W - I||_F                       :", np.linalg.norm(W.conj().T @ W - I))
print("  CONTROL ||W'* W' - I||_F (R w/o the 2):", np.linalg.norm(Wbad.conj().T @ Wbad - I))
print("  max_j | |lambda_j| - 1 |              :", np.abs(np.abs(ev) - 1).max())
print("  CONTROL max_j | |lambda_j| - 1 |      :", np.abs(np.abs(evb) - 1).max())
print("  |lambda| min / max                    :", np.abs(ev).min(), np.abs(ev).max())
print("  | ||W h|| - ||h|| | / ||h||           :",
      abs(np.linalg.norm(W @ h) - np.linalg.norm(h)) / np.linalg.norm(h))
print("  CONTROL same ratio                    :",
      abs(np.linalg.norm(Wbad @ h) - np.linalg.norm(h)) / np.linalg.norm(h))

# Powers: the whole point of a unitary transition is that ||W^k h|| does not move.
p = h.copy()
for _ in range(1000):
    p = W @ p
print("  ||W^1000 h|| / ||h||                  :", np.linalg.norm(p) / np.linalg.norm(h))

# ---- modReLU: the magnitude channel of the uRNN, and its endpoints -------------
m = 200000
z = rng.normal(size=m) + 1j * rng.normal(size=m)
b = rng.normal(size=m)                      # learned bias, one per hidden dim
absz = np.abs(z)
gain = np.maximum(absz + b, 0.0) / absz     # |sigma_modReLU(z)| / |z|
print("modReLU magnitude gain  |sigma(z)|/|z|,  m =", m, "samples")
print("  exactly 0.0 (closed lower endpoint)   :", int((gain == 0.0).sum()),
      "of", m, "=", (gain == 0.0).mean())
print("  max gain observed (NO upper cap)      :", gain.max())
print("  fraction with gain > 1                :", (gain > 1.0).mean())
print("  CONTROL modReLU without the ReLU, (|z|+b)/|z| -> min:",
      ((absz + b) / absz).min(), "(negative: magnitudes flip sign, O(1) wrong)")

assert np.linalg.norm(W.conj().T @ W - I) < 1e-12
assert np.abs(np.abs(ev) - 1).max() < 1e-12
assert np.linalg.norm(Wbad.conj().T @ Wbad - I) > 1e-1     # control must be O(1) off
assert (gain == 0.0).sum() > 0                              # 0 is ATTAINED
assert gain.max() > 1.0                                     # no cap above
print("OK")

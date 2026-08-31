"""[V-eq] probe 5 -- X37's four certificates at equation level: Berry phase,
Kuramoto order parameter, Euler-Poincare, Poincare-Hopf.

MISTAKES.md P-10 exists because a source's intro was cited as its theorem. Each
block below carries the theorem's own statement WITH ITS HYPOTHESES and one
numeric instance with an O(1) mis-transcription control.

--------------------------------------------------------------------------
(a) BERRY PHASE.  M. V. Berry, "Quantal Phase Factors Accompanying Adiabatic
Changes", Proc. R. Soc. Lond. A 392:45-57 (1984).  The JSTOR scan has no text
layer; pages were rendered at 200 dpi with PyMuPDF and read directly.  Sec 2,
pp.46-47, transcribed:

  "Let the Hamiltonian H be changed by varying parameters R = (X, Y, ...) on
   which it depends. Then the excursion of the system between times t = 0 and
   t = T can be pictured as transport round a closed path R(t) in parameter
   space, with Hamiltonian H(R(t)) and such that R(T) = R(0). ... For the
   adiabatic approximation to apply, T must be large."
      H(R(t))|psi(t)> = i hbar |psi'(t)>                                    (1)
      H(R)|n(R)> = E_n(R)|n(R)>                                             (2)
  "At any instant, the natural basis consists of the eigenstates |n(R)>
   (assumed discrete) ... For present purposes any (differentiable) choice of
   phases can be made, provided |n(R)> is single-valued in a parameter domain
   that includes the circuit C."
      |psi(t)> = exp{(-i/hbar) int_0^t dt' E_n(R(t'))} exp(i gamma_n(t))|n(R(t))>  (3)
      gamma_n'(t) = i <n(R(t))| grad_R n(R(t))> . R'(t)                     (4)
      |psi(T)> = exp(i gamma_n(C)) exp{(-i/hbar) int_0^T dt E_n(R(t))} |psi(0)>  (5)
      gamma_n(C) = i oint_C <n(R)| grad_R n(R)> . dR                        (6)
  "The normalization of |n> implies that <n|grad_R n> is imaginary, which
   guarantees that gamma_n is real."

  Sec 4, pp.48-49, the two-state instance:
      H(R) = (1/2) [[Z, X - iY], [X + iY, -Z]]                             (12)
      E_+(R) = -E_-(R) = (1/2)(X^2+Y^2+Z^2)^{1/2} = R/2                    (13)
      exp{i gamma_+-(C)} = exp{-+ (1/2) i Omega(C)}                        (18)
  "where Omega(C) is the solid angle that C subtends at the degeneracy."

  Hypotheses as Berry states them: closed circuit R(T) = R(0); T large
  (adiabatic); eigenvalues discrete and, for (12)-(18), the level NON-DEGENERATE
  away from the isolated degeneracy at R = 0 (eq 8's denominator E_n - E_m and
  eq 10's (E_m - E_n)^2 both require it).  Eq (18) is exact only for the
  standard form (12).

--------------------------------------------------------------------------
(b) KURAMOTO.  Rodrigues, Peron, Ji, Kurths, "The Kuramoto model in complex
networks", Physics Reports 610:1-98 (2016), arXiv:1511.07139, Sec 2, verbatim:

      theta_i' = omega_i + (lambda/N) sum_{j=1}^N sin(theta_j - theta_i)     (1)
      R e^{i psi(t)} = (1/N) sum_{j=1}^N e^{i theta_j(t)}                    (2)
      theta_i' = omega_i + lambda R sin(psi - theta_i)                       (3)
      R = lambda R int_{-pi/2}^{pi/2} cos^2(theta) g(lambda R sin theta) dtheta (8)
      lambda_c^{KM} = 2 / (pi g(0))                                          (9)

  Hypotheses as the source states them: g(omega) unimodal and symmetric about
  its mean, shifted so mean = 0 and g(omega) = g(-omega); all-to-all coupling
  with the 1/N normalisation; (9) is the N -> infinity, t -> infinity onset,
  obtained "by letting R -> 0+ in Eq. 8".

--------------------------------------------------------------------------
(c) EULER-POINCARE.  A. Hatcher, "Algebraic Topology", CUP 2002, Sec 2.2,
p.146-147, verbatim:

  "For a finite CW complex X, the Euler characteristic chi(X) is defined to be
   the alternating sum sum_n (-1)^n c_n where c_n is the number of n-cells of X"
      Theorem 2.44.  chi(X) = sum_n (-1)^n rank H_n(X).

  Hypothesis as Hatcher states it: X a FINITE CW complex.  Nothing about
  manifolds, vector fields, or equilibria enters.

--------------------------------------------------------------------------
(d) POINCARE-HOPF.  J. Milnor, "Topology from the Differentiable Viewpoint",
Sec 6, p.35, verbatim:

  "Let M be a compact manifold and w a smooth vector field on M with isolated
   zeros. If M has a boundary, then w is required to point outward at all
   boundary points.
   Poincare-Hopf Theorem. The sum sum(iota) of the indices at the zeros of such
   a vector field is equal to the Euler number
       chi(M) = sum_{i=0}^m (-1)^i rank H_i(M).
   In particular this index sum is a topological invariant of M: it does not
   depend on the particular choice of vector field."

  Hypotheses, all four load-bearing: M COMPACT, M a MANIFOLD, the zeros
  ISOLATED, and w OUTWARD-POINTING on any boundary.

WHAT THIS PROBE SETTLES FOR X37(c): the delta writes "Sigma(-1)^k beta_k from
the toolkit MUST EQUAL the Poincare-Hopf index sum of the equilibrium census.
Mismatch => census defect."  Block (d) below exhibits a mismatch with NO census
defect: a Hopf-normal-form field on the closed disk whose census is complete and
whose index sum is exactly chi(D^2) = 1, while the Rips carrier of its own
trajectory has Sigma(-1)^k beta_k = 0.  The two numbers are invariants of two
different spaces (the domain M, and the omega-limit set the trajectory samples),
and Milnor's hypotheses bind only the first.  A registered "must equal" would
here fire on a correct census.

Reproduce: python scripts/v15_x36_probes/p05_certificates.py
"""
import numpy as np

rng = np.random.default_rng(20260831)

# ============================================================================
print("=" * 74)
print("(a) BERRY 1984, eq (12)-(18): spin-1/2 round a cone, discrete Wilson loop")
print("=" * 74)

sx = np.array([[0, 1], [1, 0]], complex)
sy = np.array([[0, -1j], [1j, 0]], complex)
sz = np.array([[1, 0], [0, -1]], complex)


def H(R):
    """Berry eq (12): H(R) = (1/2)[[Z, X-iY],[X+iY, -Z]] = (1/2) R . sigma."""
    X, Y, Z = R
    return 0.5 * (X * sx + Y * sy + Z * sz)


def upper_state(R):
    """|+>: eigenvector of H(R) for E_+ = R/2  (Berry eq 13)."""
    w, v = np.linalg.eigh(H(R))
    return v[:, np.argmax(w)]


def wilson_loop(theta0, n, gauge_noise=0.0, close=True):
    """Discrete form of eq (6): gamma = -arg prod_k <n_k|n_{k+1}>, circuit closed."""
    phis = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    states = []
    for p in phis:
        R = np.array([np.sin(theta0) * np.cos(p),
                      np.sin(theta0) * np.sin(p),
                      np.cos(theta0)])
        s = upper_state(R)
        if gauge_noise:                       # arbitrary smooth-free gauge choice
            s = s * np.exp(1j * rng.uniform(-np.pi, np.pi))
        states.append(s)
    prod = 1.0 + 0j
    m = n if close else n - 1                 # CONTROL: drop the closing overlap
    for k in range(m):
        prod *= np.vdot(states[k], states[(k + 1) % n])
    return -np.angle(prod)


for theta0 in (0.3, np.pi / 3, 1.9):
    Omega = 2.0 * np.pi * (1.0 - np.cos(theta0))          # solid angle of the cone
    pred = -0.5 * Omega                                    # eq (18), upper branch
    got = wilson_loop(theta0, 4000)
    # phases are defined mod 2pi
    err = (got - pred + np.pi) % (2 * np.pi) - np.pi
    got_g = wilson_loop(theta0, 4000, gauge_noise=1.0)     # gauge invariance
    err_g = (got_g - pred + np.pi) % (2 * np.pi) - np.pi
    # CONTROL: drop the closing overlap <n_{N-1}|n_0>. Eq (6) is a CIRCUIT
    # integral; without the closure the result is a gauge artefact, O(1).
    bad = wilson_loop(theta0, 4000, gauge_noise=1.0, close=False)
    err_bad = (bad - pred + np.pi) % (2 * np.pi) - np.pi
    print(f"  theta0={theta0:.4f}  Omega={Omega:.9f}  -Omega/2={pred: .9f}"
          f"  Wilson={got: .9f}  err={err: .3e}")
    print(f"      gauge-randomised (must be invariant) err={err_g: .3e}"
          f"   CONTROL open circuit err={err_bad: .3e}")
    assert abs(err) < 1e-5 and abs(err_g) < 1e-5
    assert abs(err_bad) > 1e-2
print("  CONTROL eq (18) mis-transcribed WITHOUT the 1/2:  -Omega vs -Omega/2 at")
print("    theta0=pi/3 differs by", 0.5 * 2 * np.pi * (1 - np.cos(np.pi / 3)), "rad (O(1)).")

# ============================================================================
print()
print("=" * 74)
print("(b) KURAMOTO eq (1)(2)(3)(9)")
print("=" * 74)

N = 4000
gamma_l = 0.5                                   # Lorentzian half-width
omega = gamma_l * np.tan(np.pi * (rng.random(N) - 0.5))   # Cauchy(0, gamma)
g0 = 1.0 / (np.pi * gamma_l)                    # g(0) for the Lorentzian
lam_c = 2.0 / (np.pi * g0)                      # eq (9)
print(f"  Lorentzian g(0) = 1/(pi*gamma) = {g0:.9f}   eq(9) lambda_c = {lam_c:.9f}"
      f"   (= 2*gamma = {2*gamma_l})")


def drift(th, lam, flip=False):
    """eq (1). flip=True is the CONTROL: sin(theta_i - theta_j)."""
    d = th[None, :] - th[:, None]
    if flip:
        d = -d
    return omega + lam * np.sin(d).mean(axis=1)


def order(th):
    """eq (2): R e^{i psi} = (1/N) sum_j e^{i theta_j}."""
    z = np.exp(1j * th).mean()
    return abs(z), np.angle(z)


# The long runs below integrate eq (3); the identity check further down shows
# eq (3) reproduces eq (1) to 1e-16 on this configuration, so this is the same
# dynamics at O(N) instead of O(N^2) per step.
def drift3(th, lam):
    z = np.exp(1j * th).mean()
    return omega + lam * abs(z) * np.sin(np.angle(z) - th)


for lam in (0.4, 0.9, 1.5, 3.0):
    th = rng.uniform(-np.pi, np.pi, N)
    dt, T = 0.01, 20000
    for _ in range(T):
        th = th + dt * drift3(th, lam)
    R, psi = order(th)
    Rth = np.sqrt(max(0.0, 1.0 - lam_c / lam)) if lam > lam_c else 0.0
    print(f"  lambda={lam:>4}  R(sim)={R:.6f}   sqrt(1-lambda_c/lambda)={Rth:.6f}"
          f"   |diff|={abs(R-Rth):.4f}")

# small O(N^2) cross-check that integrating eq (1) directly lands in the same place
th_a = rng.uniform(-np.pi, np.pi, 400)
th_b = th_a.copy()
om_full = omega
omega = omega[:400]
for _ in range(4000):
    th_a = th_a + 0.01 * drift(th_a, 3.0)
    th_b = th_b + 0.01 * drift3(th_b, 3.0)
print(f"  eq(1)-integrated R = {order(th_a)[0]:.6f} vs eq(3)-integrated"
      f" R = {order(th_b)[0]:.6f}  (N=400, lambda=3)")
omega = om_full

# eq (1) == eq (3) identity, exactly, at a PARTIALLY SYNCHRONISED configuration
# (a uniform-random one has R ~ 1/sqrt(N), which would make the sign-flip control
# look small for a reason that has nothing to do with the transcription).
th = 0.6 * rng.normal(size=N)
lam = 1.7
R, psi = order(th)
lhs = drift(th, lam)
rhs = omega + lam * R * np.sin(psi - th)
bad = omega + lam * R * np.sin(th - psi)
print("  max | eq(1) - eq(3) |            :", np.abs(lhs - rhs).max())
print("  CONTROL sin(theta_i - psi)       :", np.abs(lhs - bad).max(), "(O(1))")
print("  CONTROL eq(1) with sin(th_i-th_j):", np.abs(drift(th, lam, flip=True) - rhs).max())
assert np.abs(lhs - rhs).max() < 1e-12
assert np.abs(lhs - bad).max() > 1e-2

# ============================================================================
print()
print("=" * 74)
print("(c) EULER-POINCARE, Hatcher Thm 2.44, on the 7-vertex minimal torus")
print("=" * 74)

# Csaszar / Moebius 7-vertex triangulation of T^2: V=7, E=21, F=14.
tris = [(i, (i + 1) % 7, (i + 3) % 7) for i in range(7)] + \
       [(i, (i + 1) % 7, (i + 5) % 7) for i in range(7)]
tris = [tuple(sorted(t)) for t in tris]
edges = sorted({tuple(sorted(e)) for t in tris for e in
                ((t[0], t[1]), (t[0], t[2]), (t[1], t[2]))})
verts = sorted({v for t in tris for v in t})
V, E, F = len(verts), len(edges), len(tris)


def gf2_rank(M):
    M = M.copy() % 2
    r = 0
    rows, cols = M.shape
    for c in range(cols):
        piv = np.nonzero(M[r:, c])[0]
        if piv.size == 0:
            continue
        p = r + piv[0]
        M[[r, p]] = M[[p, r]]
        hit = np.nonzero(M[:, c])[0]
        for h in hit:
            if h != r:
                M[h] ^= M[r]
        r += 1
        if r == rows:
            break
    return r


ei = {e: k for k, e in enumerate(edges)}
d1 = np.zeros((V, E), np.uint8)                 # edges -> vertices
for e, k in ei.items():
    d1[e[0], k] ^= 1
    d1[e[1], k] ^= 1
d2 = np.zeros((E, F), np.uint8)                 # triangles -> edges
for j, t in enumerate(tris):
    for e in ((t[0], t[1]), (t[0], t[2]), (t[1], t[2])):
        d2[ei[e], j] ^= 1
r1, r2 = gf2_rank(d1), gf2_rank(d2)
b0 = V - r1
b1 = (E - r1) - r2
b2 = F - r2
chi_cells = V - E + F
chi_betti = b0 - b1 + b2
print(f"  V={V} E={E} F={F}   chi = V - E + F = {chi_cells}")
print(f"  beta over GF(2): b0={b0} b1={b1} b2={b2}   sum (-1)^k b_k = {chi_betti}")
print(f"  Thm 2.44 residual |chi_cells - chi_betti| = {abs(chi_cells - chi_betti)}")
print(f"  CONTROL chi mis-transcribed as V + E + F = {V + E + F}"
      f"   (differs by {abs(V + E + F - chi_betti)}, O(1))")
assert (b0, b1, b2) == (1, 2, 1) and chi_cells == chi_betti == 0

# ============================================================================
print()
print("=" * 74)
print("(d) POINCARE-HOPF vs a RIPS CARRIER: a mismatch that is NOT a census defect")
print("=" * 74)


def field(p):
    """Hopf normal form, w(x,y): r' = r(1-r^2), theta' = 1.  Sole zero: origin."""
    x, y = p[..., 0], p[..., 1]
    r2 = x * x + y * y
    return np.stack([x * (1 - r2) - y, y * (1 - r2) + x], -1)


# index of the zero at the origin = winding number of w/|w| on a small circle
t = np.linspace(0, 2 * np.pi, 20001)[:-1]
c = 1e-3 * np.stack([np.cos(t), np.sin(t)], -1)
w = field(c)
ang = np.unwrap(np.arctan2(w[:, 1], w[:, 0]))
idx = (ang[-1] - ang[0] + (ang[1] - ang[0])) / (2 * np.pi)
print(f"  index at the origin (winding of w/|w|) = {idx:.9f}  -> {int(round(idx))}")

# Milnor's boundary hypothesis on the closed disk of radius Rb > 1
Rb = 2.0
b = Rb * np.stack([np.cos(t), np.sin(t)], -1)
outward = (field(b) * b).sum(-1)
print(f"  on |p| = {Rb}: w . n has sign(min)={np.sign(outward.min()):.0f}"
      f" sign(max)={np.sign(outward.max()):.0f}  -> w points INWARD;")
print("    Milnor's hypothesis holds for -w, whose zero set and index are the same.")
print(f"  Sigma(iota) = {int(round(idx))}   chi(D^2) = 1   -> Poincare-Hopf HOLDS.")

# now the Rips carrier of a trajectory of the SAME field
p = np.array([0.05, 0.0])
dt = 0.01
traj = []
for k in range(120000):
    p = p + dt * field(p[None, :])[0]
    if k >= 40000 and k % 40 == 0:            # drop the transient, subsample
        traj.append(p.copy())
traj = np.array(traj)
try:
    import gudhi
    st = gudhi.RipsComplex(points=traj, max_edge_length=0.6).create_simplex_tree(max_dimension=2)
    st.compute_persistence()
    bnums = st.betti_numbers()
except Exception as exc:                       # pragma: no cover
    bnums = None
    print("  gudhi unavailable:", exc)
if bnums is not None:
    bnums = list(bnums) + [0] * (3 - len(bnums))
    chi_carrier = bnums[0] - bnums[1] + bnums[2]
    print(f"  Rips carrier of the trajectory ({len(traj)} pts): beta = {bnums[:3]}"
          f"   Sigma(-1)^k beta_k = {chi_carrier}")
    print(f"  X37(c) as written would compare {chi_carrier} against {int(round(idx))}"
          f" and declare a CENSUS DEFECT.")
    print("  The census is complete (one zero, index +1) and Poincare-Hopf holds on")
    print("  D^2. The carrier is the omega-limit set, a circle: chi = 0. Two spaces,")
    print("  two correct invariants. Milnor's hypotheses bind M, never the carrier.")
    assert bnums[0] == 1 and bnums[1] == 1
    assert chi_carrier != int(round(idx))

assert int(round(idx)) == 1
print("OK")

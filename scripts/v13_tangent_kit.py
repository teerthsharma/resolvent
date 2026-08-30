"""V13 X28b -- the tangent kit: forward sensitivity, Lyapunov spectrum, adjoint.

    python scripts/v13_tangent_kit.py

Companion to `V13_X28B_TANGENT_KIT.md`. Every assertion carries an equation
number `(K*)` from that file. numpy and torch, float64 throughout, fixed seeds,
no test framework and no fixtures. A failing assertion is recorded as a finding,
never repaired by widening the tolerance.

    I1   forward sensitivity by `torch.func.jvp` against a central difference
         (K1)-(K4)
    I2   Lyapunov spectrum by Benettin QR on the variational equation
         (K5)-(K8)
    I3   adjoint gradients for a continuous-depth field, gradchecked at
         rtol = 1e-4  (K12)-(K16)
    MF   the two must-fires: logistic r=4 (lambda = ln 2, (K9)) and logistic
         r=3.2 (lambda = ln 0.4, (K10)); Henon as the 2-D closed form (K11)
    CT   the consistency triangle kappa = lambda*(1-d) on a planted open map
         (K17)-(K22), and whether it is three instruments or one rearrangement

Nothing here writes to `results/`; the script prints and asserts only.
"""

from __future__ import annotations

import math
import os
import sys
import time

# Pinned to one thread BEFORE numpy/torch import.  The Benettin loop factorises
# a batch of 1x1 and 2x2 matrices per iteration; multithreaded LAPACK spends
# more time in thread handoff than in arithmetic on operands that small, and a
# pinned run is also reproducible and does not compete with other measurement
# lanes on the box.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

torch.set_num_threads(1)

torch.set_default_dtype(torch.float64)

EPS = float(np.finfo(np.float64).eps)          # 2.220446049250313e-16
CBRT_EPS = EPS ** (1.0 / 3.0)                  # 6.055454452393343e-06

# ---- run parameters, all provenance for the printed numbers ---------------
SEED = 28
LOG4_B, LOG4_N, LOG4_BURN = 512, 200_000, 10_000
LOG4_CHECK = (200, 2_000, 20_000, 200_000)
LOG32_B, LOG32_N, LOG32_BURN = 8, 20_000, 200_000
HENON_B, HENON_N, HENON_BURN = 32, 100_000, 10_000
ADJ_D, ADJ_T, ADJ_STEPS = 4, 1.0, 128
ADJ_DT_SWEEP = (8, 16, 32, 64, 128)
TRI_A, TRI_B = 3.0, 5.0
TRI_N0, TRI_T = 4_000_000, 15
TRI_LEVEL = 22
TRI_JMIN, TRI_JMAX = 8, 20

LN2 = math.log(2.0)                            # 0.6931471805599453


def note(msg=""):
    print(msg, flush=True)


def rule(title):
    note()
    note("=" * 78)
    note(title)
    note("=" * 78)


# ==========================================================================
# I1 -- FORWARD SENSITIVITY (JVP) AGAINST A CENTRAL DIFFERENCE
# ==========================================================================
#
# (K1)  x_{n+1} = f(x_n),   v_{n+1} = Df(x_n) v_n
# (K2)  D_h f(x)[v] := (f(x + h v) - f(x - h v)) / (2h)
#                    = Df(x)v + (h^2/6) D^3f(x)[v,v,v] + O(h^4)
# (K3)  total error E(h) = C3 h^2 / 6 + eps_m M0 / h, minimised at
#       h* = (3 eps_m M0 / C3)^(1/3);  with M0 ~ C3 ~ O(1) this is
#       h* ~ eps_m^(1/3) = 6.055e-06 and E(h*) ~ eps_m^(2/3) ~ 3.7e-11.
# (K4)  Df^n(x_0) = Df(x_{n-1}) ... Df(x_0)  -- the composed tangent equals the
#       iterated single-step tangent, which is (K1) unrolled.


def fd_directional(f, x, v, h):
    """(K2) central difference; h is chosen by (K3), never by trial."""
    return (f(x + h * v) - f(x - h * v)) / (2.0 * h)


def fd_step(x):
    """(K3) h* = eps_m^(1/3) scaled by the state magnitude."""
    return CBRT_EPS * max(1.0, float(x.abs().max()))


def jvp_vs_fd(label, f, x, v, analytic=None):
    """Report the achieved relative error; assert only a loose sanity floor."""
    _, tan = torch.func.jvp(f, (x,), (v,))
    h = fd_step(x)
    fd = fd_directional(f, x, v, h)
    rel_fd = float((tan - fd).norm() / tan.norm())
    line = f"  {label:<34s} h = {h:.6e}   rel|jvp - fd| = {rel_fd:.4e}"
    if analytic is not None:
        rel_an = float((tan - analytic).norm() / tan.norm())
        line += f"   rel|jvp - closed form| = {rel_an:.4e}"
        assert rel_an < 1e-14, (
            f"K1 {label}: torch.func.jvp disagrees with the closed-form "
            f"Jacobian-vector product at {rel_an:.3e}, which is above roundoff")
    note(line)
    assert rel_fd < 1e-7, (
        f"K2/K3 {label}: jvp and the central difference at the derived step "
        f"h = {h:.3e} disagree at {rel_fd:.3e}; the predicted floor is "
        f"~eps^(2/3) = {EPS ** (2/3):.1e}, so this is a real disagreement")
    return rel_fd


def logistic_t(r):
    return lambda x: r * x * (1.0 - x)


def henon_t(a, b):
    def f(x):
        return torch.stack((1.0 - a * x[0] ** 2 + x[1], b * x[0]))
    return f


def tanh_field(W, b):
    return lambda z: torch.tanh(W @ z + b)


def check_i1():
    rule("I1  FORWARD SENSITIVITY -- torch.func.jvp vs a central difference")
    g = torch.Generator().manual_seed(SEED)
    note(f"  float64 eps = {EPS:.16e},  eps^(1/3) = {CBRT_EPS:.15e},  "
         f"eps^(2/3) = {EPS ** (2/3):.4e}")
    note()

    # ---- (K1) single-step tangents on three beds -------------------------
    x = torch.tensor([0.37])
    v = torch.tensor([1.0])
    jvp_vs_fd("logistic r=4, single step", logistic_t(4.0), x, v,
              analytic=(4.0 - 8.0 * x) * v)

    xh = torch.tensor([0.31, -0.17])
    vh = torch.tensor([0.6, -0.8])
    jac_h = torch.tensor([[-2 * 1.4 * float(xh[0]), 1.0], [0.3, 0.0]])
    jvp_vs_fd("henon a=1.4 b=0.3, single step", henon_t(1.4, 0.3), xh, vh,
              analytic=jac_h @ vh)

    W = torch.randn(ADJ_D, ADJ_D, generator=g) / math.sqrt(ADJ_D)
    bb = torch.randn(ADJ_D, generator=g) * 0.1
    z = torch.randn(ADJ_D, generator=g)
    vz = torch.randn(ADJ_D, generator=g)
    s = 1.0 - torch.tanh(W @ z + bb) ** 2
    jvp_vs_fd("tanh(Wz+b) field, state tangent", tanh_field(W, bb), z, vz,
              analytic=s * (W @ vz))

    # tangent in the parameters as well as the state -- the jvp the adjoint
    # in I3 is the transpose of.
    def f_all(zz, WW, bbv):
        return torch.tanh(WW @ zz + bbv)

    dz, dW, db = (torch.randn(ADJ_D, generator=g),
                  torch.randn(ADJ_D, ADJ_D, generator=g),
                  torch.randn(ADJ_D, generator=g))
    _, tan = torch.func.jvp(f_all, (z, W, bb), (dz, dW, db))
    an = s * (W @ dz + dW @ z + db)
    rel = float((tan - an).norm() / tan.norm())
    note(f"  {'tanh field, (z,W,b) tangent':<34s} "
         f"rel|jvp - closed form| = {rel:.4e}")
    assert rel < 1e-14, (
        f"K1 joint tangent: jvp disagrees with s*(W dz + dW z + db) at "
        f"{rel:.3e}")

    # ---- (K3) the step-size V-curve, measured rather than assumed --------
    #
    # FINDING, recorded rather than tuned around: the logistic map is a
    # QUADRATIC, so D^3 f = 0 identically and the (K2) truncation term
    # vanishes.  Its central difference is exact up to roundoff and its error
    # falls monotonically in h -- there is no V and no interior optimum.  The
    # V-curve is therefore measured on tanh(Wz+b), where D^3 f != 0.
    f4 = logistic_t(4.0)
    exact4 = float((4.0 - 8.0 * x)[0])
    e_small = abs(float(fd_directional(f4, x, v, 1e-8)[0]) - exact4) / abs(exact4)
    e_big = abs(float(fd_directional(f4, x, v, 1e-2)[0]) - exact4) / abs(exact4)
    note()
    note("  (K2) FINDING -- the logistic map is quadratic, so D^3f = 0 and the")
    note("       central difference carries no truncation term at all:")
    note(f"         h = 1e-08  rel err = {e_small:.4e}   "
         f"h = 1e-02  rel err = {e_big:.4e}   (error falls WITH h, no optimum)")
    note("       The (K3) step is derived for D^3f != 0; on a quadratic the "
         "achieved error is")
    note(f"       pure roundoff, eps_m/h.  At h = {fd_step(x):.3e} that "
         f"predicts {EPS / (2 * fd_step(x)) / abs(exact4):.2e}.")
    assert e_big < e_small, (
        "K2 the logistic central difference does not improve with larger h; "
        "the quadratic-exactness claim is wrong")

    note()
    note("  (K3) step-size sweep on tanh(Wz+b), where D^3f != 0 -- the "
         "measured minimum")
    note("       against the derived h*:")
    ftanh = tanh_field(W, bb)
    exact_t = s * (W @ vz)
    best_h, best_e = None, math.inf
    for k in range(-12, -1):
        h = 10.0 ** k
        e = float((fd_directional(ftanh, z, vz, h) - exact_t).norm()
                  / exact_t.norm())
        if e < best_e:
            best_h, best_e = h, e
        note(f"        h = 1e{k:<4d}  rel err = {e:.4e}")
    note(f"        measured minimum at h = {best_h:.0e} with rel err = "
         f"{best_e:.4e};  derived h* = {CBRT_EPS:.3e}, "
         f"derived floor ~ {EPS ** (2/3):.1e}")
    assert 1e-7 <= best_h <= 1e-4, (
        f"K3 the measured optimal step {best_h:.0e} is not within a decade of "
        f"the derived h* = {CBRT_EPS:.3e}; the error model is wrong")
    assert best_e < 1e-9, (
        f"K3 the achieved minimum {best_e:.3e} is far above the derived floor "
        f"eps^(2/3) = {EPS ** (2/3):.1e}")

    # ---- (K4) composed tangent == iterated single-step tangent -----------
    note()
    n_comp = 8

    def f4n(xx):
        for _ in range(n_comp):
            xx = 4.0 * xx * (1.0 - xx)
        return xx

    _, tan_comp = torch.func.jvp(f4n, (x,), (v,))
    xi, vi = x.clone(), v.clone()
    for _ in range(n_comp):
        oi, vi = torch.func.jvp(f4, (xi,), (vi,))
        xi = oi
    rel_chain = float((tan_comp - vi).norm() / tan_comp.norm())
    h = fd_step(x)
    fd_comp = fd_directional(f4n, x, v, h)
    rel_fdc = float((tan_comp - fd_comp).norm() / tan_comp.norm())
    note(f"  (K4) n = {n_comp} composed logistic map: "
         f"|Df^n v| = {float(tan_comp.abs()):.6e}")
    note(f"       rel|composed jvp - iterated (K1) tangent| = {rel_chain:.4e}")
    note(f"       rel|composed jvp - central difference|    = {rel_fdc:.4e}   "
         f"(amplification 2^{n_comp} = {2 ** n_comp} inflates the truncation "
         f"term)")
    assert rel_chain < 1e-13, (
        f"K4 the composed tangent and the iterated single-step tangent differ "
        f"at {rel_chain:.3e}; the tangent recursion is not the chain rule")
    assert rel_fdc < 1e-5, (
        f"K4 composed jvp vs central difference at {rel_fdc:.3e}")


# ==========================================================================
# I2 -- LYAPUNOV SPECTRUM BY BENETTIN QR
# ==========================================================================
#
# (K5)  M_n = J_n Q_{n-1},   M_n = Q_n R_n  with diag(R_n) > 0 enforced by
#       S = diag(sign(diag(R))):  M = (Q S)(S R),  S^2 = I.
# (K6)  A_N Q_0 = Q_N R_N R_{N-1} ... R_1  with A_N = J_N ... J_1, so the
#       diagonal of the triangular product is prod_n (R_n)_ii.
# (K7)  lambda_i = lim_{N->inf} (1/N) sum_{n=1}^{N} ln (R_n)_ii   (unit step;
#       divide by N*dt for a flow).
# (K8)  ln (R_n)_ii is an ergodic average of a finite-variance observable, so
#       |lambda_hat_i(N) - lambda_i| ~ sigma_i N^(-1/2).


def benettin(step, jac, x, n_iter, n_burn, checkpoints=()):
    """(K5)-(K7). `x` is a (B, d) ensemble; returns (B, d) exponents.

    `jac(x)` must return (B, d, d).  Checkpoints record the running estimate
    so the (K8) convergence rate costs no extra iterations.
    """
    for _ in range(n_burn):
        x = step(x)
    b, d = x.shape
    q = np.broadcast_to(np.eye(d), (b, d, d)).copy()
    acc = np.zeros((b, d))
    hist = {}
    collapse = 0
    for n in range(1, n_iter + 1):
        m = jac(x) @ q
        q, r = np.linalg.qr(m)
        diag = np.diagonal(r, axis1=-2, axis2=-1)
        s = np.where(diag < 0.0, -1.0, 1.0)
        q = q * s[..., None, :]
        diag = diag * s
        if not np.all(diag > 0.0):
            collapse += 1
            diag = np.where(diag > 0.0, diag, np.finfo(float).tiny)
        acc += np.log(diag)
        x = step(x)
        if n in checkpoints:
            hist[n] = acc / n
    return acc / n_iter, hist, collapse


def logistic_np(r):
    return (lambda x: r * x * (1.0 - x),
            lambda x: (r - 2.0 * r * x).reshape(x.shape[0], 1, 1))


def henon_np(a, b):
    def step(x):
        return np.stack((1.0 - a * x[:, 0] ** 2 + x[:, 1], b * x[:, 0]), axis=1)

    def jac(x):
        j = np.zeros((x.shape[0], 2, 2))
        j[:, 0, 0] = -2.0 * a * x[:, 0]
        j[:, 0, 1] = 1.0
        j[:, 1, 0] = b
        return j
    return step, jac


def seed_logistic(b, rng):
    """(K9 pitfall) seed off every exact preimage of the fixed point 0.

    x = sin^2(pi u) conjugates the r=4 map to u -> 2u mod 1, under which the
    preimages of 0 are exactly the dyadic rationals.  Drawing u away from
    k/2^m and rejecting u whose first 60 doubling images come within 2^-40 of
    an integer removes every seed that would land on 0.5 -> 1 -> 0.
    """
    out = []
    while len(out) < b:
        u = float(rng.uniform(0.05, 0.95))
        v, ok = u, True
        for _ in range(60):
            v = (2.0 * v) % 1.0
            if min(v, 1.0 - v) < 2.0 ** -40:
                ok = False
                break
        if ok:
            out.append(math.sin(math.pi * u) ** 2)
    return np.array(out).reshape(b, 1)


def check_i2_positive():
    rule("MF-1  POSITIVE CONTROL -- logistic r=4, lambda = ln 2 exactly (K9)")
    note("  (K9) x = sin^2(pi u) conjugates x -> 4x(1-x) to u -> 2u mod 1.")
    note("       |du'/du| = 2 everywhere, so lambda = ln 2 = "
         f"{LN2:.16f} exactly.")
    rng = np.random.default_rng(SEED)
    x0 = seed_logistic(LOG4_B, rng)
    note(f"  seeds: {LOG4_B} draws of x0 = sin^2(pi u), u ~ U(0.05,0.95), "
         f"rng seed {SEED}, dyadic-preimage rejection over 60 doublings")
    note(f"  transient discarded: {LOG4_BURN}   measured iterations: {LOG4_N}")
    step, jac = logistic_np(4.0)
    t0 = time.perf_counter()
    lam, hist, collapse = benettin(step, jac, x0, LOG4_N, LOG4_BURN,
                                   LOG4_CHECK)
    dt = time.perf_counter() - t0
    note(f"  wall clock {dt:.1f} s;  non-positive R-diagonals encountered: "
         f"{collapse}  (an orbit landing exactly on x = 0.5 would produce one)")

    note()
    note("  (K8) convergence of the ensemble mean and of the per-seed RMS "
         "error:")
    ns, rms = [], []
    for n in LOG4_CHECK:
        e = hist[n][:, 0] - LN2
        r = float(np.sqrt(np.mean(e ** 2)))
        ns.append(n)
        rms.append(r)
        note(f"        N = {n:>9d}   mean lambda_hat = "
             f"{float(hist[n][:, 0].mean()):.9f}   "
             f"|mean - ln2| = {abs(float(hist[n][:, 0].mean()) - LN2):.3e}   "
             f"per-seed RMS err = {r:.4e}")
    slope = float(np.polyfit(np.log(ns), np.log(rms), 1)[0])
    note(f"        fitted slope of log(RMS err) vs log(N): {slope:+.4f}   "
         f"(K8 predicts -1/2)")
    assert -0.62 < slope < -0.38, (
        f"K8 the measured convergence exponent {slope:+.4f} is not the "
        f"-1/2 the central limit theorem predicts for a finite-variance "
        f"ergodic average")

    mean = float(lam[:, 0].mean())
    sd = float(lam[:, 0].std(ddof=1))
    se = sd / math.sqrt(LOG4_B)
    err = abs(mean - LN2)
    note()
    note(f"  MEASURED  lambda_hat = {mean:.12f}")
    note(f"  PREDICTED lambda     = {LN2:.12f}   (ln 2, closed form)")
    note(f"  absolute error       = {err:.4e}")
    note(f"  ensemble s.d. {sd:.4e}, standard error of the mean {se:.4e}, "
         f"error / s.e. = {err / se:.2f}")
    assert err < 5.0 * se, (
        f"MF-1 logistic r=4 reads {mean:.9f} against ln 2 = {LN2:.9f}; the "
        f"error {err:.3e} is {err / se:.1f} standard errors, so the positive "
        f"control does not fire and every number downstream of this kit is "
        f"void")
    assert mean > 0.0
    return mean, err, se


def check_i2_negative():
    rule("MF-2  NEGATIVE CONTROL -- logistic r=3.2, lambda = ln 0.4 (K10)")
    r = 3.2
    mult = 4.0 + 2.0 * r - r * r
    lam_true = 0.5 * math.log(abs(mult))
    note("  (K10) the period-2 orbit of x -> r x(1-x) has multiplier")
    note("        f'(p1) f'(p2) = 4 + 2r - r^2, so lambda = (1/2) ln|4+2r-r^2|.")
    note(f"        r = {r}:  4 + 2r - r^2 = {mult:.16f} = 0.4^2, "
         f"lambda = ln 0.4 = {lam_true:.16f}")
    note("        |4+2r-r^2| < 1, so the 2-cycle is attracting and the "
         "exponent is negative by construction.")
    rng = np.random.default_rng(SEED + 1)
    x0 = rng.uniform(0.2, 0.8, size=(LOG32_B, 1))
    step, jac = logistic_np(r)
    lam, _, collapse = benettin(step, jac, x0, LOG32_N, LOG32_BURN)
    mean = float(lam[:, 0].mean())
    spread = float(lam[:, 0].max() - lam[:, 0].min())
    err = abs(mean - lam_true)
    note(f"  seeds: {LOG32_B} draws of x0 ~ U(0.2,0.8), rng seed {SEED + 1}; "
         f"transient {LOG32_BURN}, measured {LOG32_N}")
    note(f"  MEASURED  lambda_hat = {mean:.15f}   (spread across seeds "
         f"{spread:.3e}, non-positive R-diagonals {collapse})")
    note(f"  PREDICTED lambda     = {lam_true:.15f}")
    note(f"  absolute error       = {err:.4e}")
    assert mean < 0.0, (
        f"MF-2 the negative control reads {mean:.9f} >= 0; the instrument "
        f"cannot return 'no chaos here' and is not validated")
    assert err < 1e-9, (
        f"MF-2 logistic r=3.2 reads {mean:.12f} against ln 0.4 = "
        f"{lam_true:.12f}, error {err:.3e}")
    return mean, err, lam_true


def check_i2_henon():
    rule("MF-3  2-D CLOSED FORM -- Henon a=1.4 b=0.3, sum of exponents (K11)")
    a, b = 1.4, 0.3
    note("  (K11) J = [[-2a x, 1], [b, 0]] has det J = -b at every point, so")
    note(f"        lambda_1 + lambda_2 = ln|det J| = ln {b} = "
         f"{math.log(b):.16f} exactly.")
    note("        This is the only closed form here that exercises the QR "
         "frame in d > 1;")
    note("        in d = 1 the QR step is trivial and (K5) is untested.")
    rng = np.random.default_rng(SEED + 2)
    x0 = np.stack((rng.uniform(-0.1, 0.1, HENON_B),
                   rng.uniform(-0.1, 0.1, HENON_B)), axis=1)
    step, jac = henon_np(a, b)
    lam, _, collapse = benettin(step, jac, x0, HENON_N, HENON_BURN)
    assert np.isfinite(lam).all(), "MF-3 a Henon seed escaped to infinity"
    l1 = float(lam[:, 0].mean())
    l2 = float(lam[:, 1].mean())
    ssum = l1 + l2
    err = abs(ssum - math.log(b))
    note(f"  seeds {HENON_B}, rng seed {SEED + 2}, transient {HENON_BURN}, "
         f"measured {HENON_N}, non-positive R-diagonals {collapse}")
    note(f"  MEASURED  lambda_1 = {l1:+.9f}   lambda_2 = {l2:+.9f}   "
         f"sum = {ssum:+.12f}")
    note(f"  PREDICTED sum      = {math.log(b):+.12f}   "
         f"absolute error = {err:.4e}")
    note(f"  lambda_2 < 0: {l2 < 0.0}  (a second reading in the 'no chaos "
         f"along this direction' direction)")
    note(f"  cross-check against the figure carried in ceq/nonnormal.py and "
         f"NOTES.md, lambda_1 = +0.42084: deviation {abs(l1 - 0.42084):.5f}")
    assert err < 1e-8, (
        f"MF-3 lambda_1 + lambda_2 = {ssum:.12f} against ln(b) = "
        f"{math.log(b):.12f}; the QR frame does not conserve the determinant "
        f"and the d > 1 path of (K5) is wrong")
    assert l1 > 0.0 > l2, (
        f"MF-3 Henon spectrum is not (+,-): {l1:+.6f}, {l2:+.6f}")
    return l1, l2, err


# ==========================================================================
# I3 -- ADJOINT GRADIENTS FOR A CONTINUOUS-DEPTH FIELD
# ==========================================================================
#
# Field:  dz/dt = f(z, theta) = tanh(W z + b),  z(0) = z0,  t in [0, T].
#
# (K12) first variation:  d(dz)/dt = (df/dz) dz + (df/dtheta) dtheta,
#       dz(0) = 0.
# (K13) adjoint equation: dlam/dt = -(df/dz)^T lam,  lam(T) = dL/dz(T).
# (K14) with (K12) and (K13),
#       d/dt [ lam^T dz ] = -lam^T (df/dz) dz + lam^T (df/dz) dz
#                           + lam^T (df/dtheta) dtheta
#                         = lam^T (df/dtheta) dtheta,
#       i.e. (K13) is exactly the choice that cancels the state term.
# (K15) integrating (K14) over [0, T] with dz(0) = 0:
#       dL = lam(T)^T dz(T) = ( int_0^T lam^T (df/dtheta) dt ) dtheta,
#       so  dL/dtheta = int_0^T lam(t)^T (df/dtheta) dt   and   dL/dz0 = lam(0).
# (K16) the augmented system is integrated by the same RK4 as the forward
#       solve, so the discrete adjoint and the discretised continuous adjoint
#       differ by O(dt^4); the gradcheck residual must fall as dt^4 or the gap
#       is a bug rather than a discretisation.
#
# Closed forms used in the backward pass, s := sech^2(W z + b) = 1 - tanh^2:
#       (df/dz)^T lam      = W^T (s * lam)
#       (df/dW)^T lam      = outer(s * lam, z)
#       (df/db)^T lam      = s * lam


def _f(z, W, b):
    return torch.tanh(W @ z + b)


def _rk4(field, y, t0, t1, n):
    h = (t1 - t0) / n
    for _ in range(n):
        k1 = field(y)
        k2 = field(y + 0.5 * h * k1)
        k3 = field(y + 0.5 * h * k2)
        k4 = field(y + h * k3)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return y


def _fwd(z0, W, b, T, n):
    return _rk4(lambda z: _f(z, W, b), z0, 0.0, T, n)


def _aug_field(y, W, b, d):
    z = y[:d]
    lam = y[d:2 * d]
    fz = torch.tanh(W @ z + b)
    s = 1.0 - fz ** 2
    sl = s * lam
    return torch.cat((fz,                       # dz/dt   = f
                      -(W.transpose(0, 1) @ sl),  # (K13)
                      torch.outer(sl, z).reshape(-1),   # (K15) integrand, W
                      sl))                      # (K15) integrand, b


class ODEAdjoint(torch.autograd.Function):
    """Forward RK4; backward by (K13)/(K15) on the augmented system.

    The backward pass reconstructs z(t) by integrating the field backwards
    from z(T) rather than storing the forward trajectory -- that is what makes
    it the continuous adjoint and not backpropagation through the solver. The
    reconstruction error is measured, not assumed away.
    """

    @staticmethod
    def forward(ctx, z0, W, b, n_steps, T):
        with torch.no_grad():
            zT = _fwd(z0, W, b, T, n_steps)
        ctx.save_for_backward(zT, W, b, z0)
        ctx.n_steps, ctx.T = n_steps, T
        return zT

    @staticmethod
    def backward(ctx, grad_out):
        zT, W, b, z0 = ctx.saved_tensors
        d = zT.numel()
        with torch.no_grad():
            y = torch.cat((zT, grad_out.reshape(-1),
                           torch.zeros(d * d + d, dtype=zT.dtype)))
            y = _rk4(lambda yy: _aug_field(yy, W, b, d), y,
                     ctx.T, 0.0, ctx.n_steps)
        ODEAdjoint.last_recon_err = float((y[:d] - z0).norm())
        # g(T) = 0 and dg/dt = +lam^T df/dtheta in forward time, so the
        # backward integration returns g(0) = -dL/dtheta.  Negate.
        return (y[d:2 * d],
                -y[2 * d:2 * d + d * d].reshape(d, d),
                -y[2 * d + d * d:],
                None, None)


ODEAdjoint.last_recon_err = float("nan")


def _numeric_jac(fn, tensors, eps):
    """Central-difference Jacobian of a vector output w.r.t. flattened inputs."""
    with torch.no_grad():
        base = fn(*tensors)
    cols = []
    for ti, t in enumerate(tensors):
        flat = t.reshape(-1)
        for i in range(flat.numel()):
            pert = [x.clone() for x in tensors]
            pert[ti].reshape(-1)[i] += eps
            with torch.no_grad():
                hi = fn(*pert)
            pert = [x.clone() for x in tensors]
            pert[ti].reshape(-1)[i] -= eps
            with torch.no_grad():
                lo = fn(*pert)
            cols.append((hi - lo) / (2.0 * eps))
    return torch.stack(cols, dim=1), base


def _analytic_jac(fn, tensors):
    for t in tensors:
        t.requires_grad_(True)
        if t.grad is not None:
            t.grad = None
    out = fn(*tensors)
    rows = []
    for i in range(out.numel()):
        gs = torch.autograd.grad(out[i], tensors, retain_graph=True)
        rows.append(torch.cat([g.reshape(-1) for g in gs]))
    for t in tensors:
        t.requires_grad_(False)
    return torch.stack(rows, dim=0)


def check_i3():
    rule("I3  ADJOINT GRADIENTS -- derivation (K12)-(K15), gradcheck at "
         "rtol = 1e-4")
    g = torch.Generator().manual_seed(SEED + 3)
    d = ADJ_D
    W = torch.randn(d, d, generator=g) / math.sqrt(d)
    b = torch.randn(d, generator=g) * 0.3
    z0 = torch.randn(d, generator=g)
    note(f"  field dz/dt = tanh(W z + b), d = {d}, T = {ADJ_T}, "
         f"RK4 with {ADJ_STEPS} steps, rng seed {SEED + 3}")
    note(f"  parameters: {d * d} in W plus {d} in b, state {d}; "
         f"{d * d + 2 * d} scalar inputs, {d} outputs")

    # closed-form (df/dz)^T cross-checked against autodiff, since the backward
    # pass uses it by hand.
    zc = torch.randn(d, generator=g)
    jz = torch.func.jacrev(lambda zz: _f(zz, W, b))(zc)
    s = 1.0 - torch.tanh(W @ zc + b) ** 2
    rel_j = float((jz - torch.diag(s) @ W).norm() / jz.norm())
    note(f"  closed-form df/dz = diag(sech^2) W vs torch.func.jacrev: "
         f"rel dev = {rel_j:.3e}")
    assert rel_j < 1e-14, f"K13 hand Jacobian wrong at {rel_j:.3e}"

    def fn(a, b_, c):
        return ODEAdjoint.apply(a, b_, c, ADJ_STEPS, ADJ_T)

    # ---- raw max relative deviation, computed here rather than inferred ----
    fd_eps = 1e-6
    num, _ = _numeric_jac(fn, (z0, W, b), fd_eps)
    ana = _analytic_jac(fn, (z0.clone(), W.clone(), b.clone()))
    dev = (ana - num).abs()
    scale = num.abs().clamp_min(1e-12)
    note(f"  backward-in-time state reconstruction error |z_rec(0) - z0| = "
         f"{ODEAdjoint.last_recon_err:.4e}")
    note(f"  Jacobian entries compared: {ana.numel()}   "
         f"max|analytic| = {float(ana.abs().max()):.6f}")
    note(f"  max absolute deviation  = {float(dev.max()):.4e}")
    note(f"  max relative deviation  = {float((dev / scale).max()):.4e}   "
         f"(central difference, eps = {fd_eps:.0e})")

    # ---- the declared kill -----------------------------------------------
    note()
    note("  DECLARED KILL: torch.autograd.gradcheck at rtol = 1e-4 "
         "(atol = 1e-6, eps = 1e-6).")
    ok = torch.autograd.gradcheck(
        fn, (z0.clone().requires_grad_(True),
             W.clone().requires_grad_(True),
             b.clone().requires_grad_(True)),
        eps=fd_eps, atol=1e-6, rtol=1e-4, raise_exception=False)
    note(f"  gradcheck result: {'PASS' if ok else 'FAIL'}")
    assert ok, (
        "I3 gradcheck FAILED at rtol = 1e-4. The tolerance is not loosened. "
        "No training through the dynamics is licensed this round.")

    # ---- (K16) the residual must fall as dt^4 ----------------------------
    note()
    note("  (K16) dt-sweep -- if the residual is a discretisation gap it falls "
         "as dt^4;")
    note("        if it is a bug it does not move.")
    prev = None
    devs = []
    for n in ADJ_DT_SWEEP:
        def fn_n(a, b_, c, n=n):
            return ODEAdjoint.apply(a, b_, c, n, ADJ_T)
        nm, _ = _numeric_jac(fn_n, (z0, W, b), fd_eps)
        an = _analytic_jac(fn_n, (z0.clone(), W.clone(), b.clone()))
        mx = float((an - nm).abs().max())
        devs.append((n, mx))
        ratio = "" if prev is None else f"   ratio vs previous = {prev / mx:6.1f}"
        note(f"        steps = {n:4d}  dt = {ADJ_T / n:.6f}  "
             f"max abs dev = {mx:.4e}{ratio}")
        prev = mx
    floor = min(m for _, m in devs)
    live = [(n, m) for n, m in devs if m > 10.0 * floor]
    slope = float(np.polyfit(np.log([n for n, _ in live]),
                             np.log([m for _, m in live]), 1)[0])
    note(f"        finite-difference floor reached at max dev = "
         f"{floor:.3e}; the {len(live)} points above 10x that floor fit a "
         f"slope of {slope:+.3f}")
    note(f"        (K16 predicts -4 for RK4.  The per-halving ratios are the "
         f"direct reading: 2^4 = 16.)")
    assert slope < -3.5, (
        f"K16 the gradcheck residual falls as dt^{-slope:.2f}, not dt^4. A "
        f"residual that does not carry the solver's order is a bug in the "
        f"adjoint, not a discretisation gap")
    return ok, float((dev / scale).max()), slope


# ==========================================================================
# CT -- THE CONSISTENCY TRIANGLE, AND WHETHER IT IS CIRCULAR
# ==========================================================================
#
# Planted open map on [0, 1], two expanding branches with different slopes:
#
#   f(x) = a x         for x <= 1/2         (survives iff x <= 1/a)
#   f(x) = b (1 - x)   for x >  1/2         (survives iff x >= 1 - 1/b)
#
# with a, b > 2 so the two surviving pre-images are disjoint.
#
# (K17) Lebesgue measure is EXACTLY conditionally invariant here: one step maps
#       each surviving interval onto [0,1] uniformly, so the surviving measure
#       is sigma^t with sigma = 1/a + 1/b, and
#           kappa = -ln sigma            (no transient, exact from t = 0).
# (K18) the natural measure on the saddle is Bernoulli with
#           p_1 = (1/a)/sigma,  p_2 = (1/b)/sigma
#       (the Gibbs measure for the potential -ln|f'|).
# (K19) lambda = p_1 ln a + p_2 ln b.
# (K20) H := -(p_1 ln p_1 + p_2 ln p_2) = lambda + ln sigma = lambda - kappa,
#       and the information dimension of a self-similar measure is D_1 = H /
#       lambda, hence
#           D_1 = (lambda - kappa) / lambda = 1 - kappa / lambda
#       -- Kantz-Grassberger, recovered exactly on this bed by construction.
# (K21) the BOX-COUNTING dimension is the Moran root of a^-s + b^-s = 1 and is
#       NOT D_1 whenever a != b.  Substituting D_0 for D_1 in the identity is a
#       measurable, systematic error.
# (K22) if d is obtained as d := 1 - kappa/lambda, then
#           kappa - lambda (1 - d) = 0
#       identically, for ANY (kappa, lambda) whatsoever.  The residual is
#       roundoff and carries no information about the dynamics.


def tri_closed_forms(a, b):
    sigma = 1.0 / a + 1.0 / b
    kappa = -math.log(sigma)
    p1, p2 = (1.0 / a) / sigma, (1.0 / b) / sigma
    lam = p1 * math.log(a) + p2 * math.log(b)
    ent = -(p1 * math.log(p1) + p2 * math.log(p2))
    d1 = ent / lam
    lo, hi = 0.0, 1.0
    for _ in range(200):                     # Moran root, bisection (K21)
        mid = 0.5 * (lo + hi)
        if a ** -mid + b ** -mid > 1.0:
            lo = mid
        else:
            hi = mid
    return sigma, kappa, p1, p2, lam, ent, d1, 0.5 * (lo + hi)


def tri_measure_kappa_lambda(a, b, n0, tmax, rng):
    """Routes 1 and 2, each using only its own raw data.

    kappa: survivor counts n(t), fitted to ln n(t) = ln n0 - kappa t.  Uses no
    Jacobian and no dimension.
    lambda: mean of ln|f'(x_t)| over every point that survives step t.  By
    (K17)/(K18) the points alive at t are Lebesgue-distributed, so conditioning
    on one further survival draws the branch with probability p_i exactly and
    the estimator is unbiased with no transient.  Uses no survival rate and no
    dimension.
    """
    x = rng.uniform(0.0, 1.0, size=n0)
    counts = [n0]
    lsum, lsq, lcnt = 0.0, 0.0, 0
    for _ in range(tmax):
        left = x <= 0.5
        slope = np.where(left, a, b)
        y = np.where(left, a * x, b * (1.0 - x))
        alive = (y >= 0.0) & (y <= 1.0)
        ln_s = np.log(slope[alive])
        lsum += float(ln_s.sum())
        lsq += float((ln_s ** 2).sum())
        lcnt += int(alive.sum())
        x = y[alive]
        counts.append(x.size)
        if x.size < 200:
            break
    ts = np.arange(len(counts))
    keep = np.array(counts) >= 200
    t_fit, n_fit = ts[keep], np.array(counts)[keep]
    slope_fit, _ = np.polyfit(t_fit, np.log(n_fit), 1)
    kappa_hat = -float(slope_fit)
    # standard error of the fitted slope from the binomial variance of ln n(t)
    var = (1.0 - n_fit / n0) / n_fit
    sxx = float(((t_fit - t_fit.mean()) ** 2).sum())
    kappa_se = math.sqrt(float((((t_fit - t_fit.mean()) ** 2) * var).sum())) / sxx
    lam_hat = lsum / lcnt
    lam_se = math.sqrt(max(lsq / lcnt - lam_hat ** 2, 0.0) / lcnt)
    return (kappa_hat, kappa_se, lam_hat, lam_se, lcnt, counts,
            int(t_fit.max()))


def tri_measure_dimension(a, b, p1, p2, level, jmin, jmax):
    """Route 3: the epsilon-scaling of the saddle's own measure.

    Enumerates every level-`level` cylinder of the invariant set by inverse
    iteration (g1(x) = x/a, g2(x) = 1 - x/b) with its natural-measure weight,
    then bins at epsilon = 2^-j and fits both

        I(eps) = -sum P ln P   ->  D_1     (information dimension)
        ln N(eps)              ->  D_0     (box-counting dimension)

    Neither fit reads kappa or lambda.  Cylinder lengths at this level are
    below a^-level, orders under the smallest epsilon, so no box straddles the
    resolution of the construction.

    The estimator's own error bar comes from its window dependence: a
    self-similar measure with incommensurable contraction ratios carries
    log-periodic oscillations on I(eps), so the fitted slope moves with the
    epsilon window.  The half-spread over sliding 8-octave sub-windows is the
    systematic, and it is computed WITHOUT reference to the closed form.
    """
    mid = np.array([0.5])
    wgt = np.array([1.0])
    for _ in range(level):
        mid = np.concatenate((mid / a, 1.0 - mid / b))
        wgt = np.concatenate((wgt * p1, wgt * p2))
    rows = []
    for j in range(jmin, jmax + 1):
        eps = 2.0 ** -j
        idx = np.minimum((mid / eps).astype(np.int64), (1 << j) - 1)
        pr = np.bincount(idx, weights=wgt)
        pr = pr[pr > 0.0]
        pr = pr / pr.sum()
        rows.append((j, eps, float(-(pr * np.log(pr)).sum()), int(pr.size)))

    def fit(lo, hi):
        sel = [r for r in rows if lo <= r[0] <= hi]
        inv = np.array([math.log(1.0 / e) for _, e, _, _ in sel])
        return (float(np.polyfit(inv, np.array([i for _, _, i, _ in sel]),
                                 1)[0]),
                float(np.polyfit(inv,
                                 np.array([math.log(n) for _, _, _, n in sel]),
                                 1)[0]))

    win = []
    for lo in range(jmin, jmax - 7):
        f1, f0 = fit(lo, lo + 8)
        win.append((lo, lo + 8, f1, f0))
    d1_full, d0_full = fit(jmin, jmax)
    sys1 = 0.5 * (max(w[2] for w in win) - min(w[2] for w in win))
    sys0 = 0.5 * (max(w[3] for w in win) - min(w[3] for w in win))
    return (d1_full, d0_full, sys1, sys0, rows, win, mid.size, a ** -level)


def check_triangle():
    rule("CT  THE CONSISTENCY TRIANGLE kappa = lambda (1 - d) -- independent "
         "or circular?")
    a, b = TRI_A, TRI_B
    sigma, kap, p1, p2, lam, ent, d1, d0 = tri_closed_forms(a, b)
    note(f"  planted open map: f(x) = {a:g}x on [0,1/2], "
         f"f(x) = {b:g}(1-x) on (1/2,1]; everything outside escapes.")
    note(f"  (K17) sigma = 1/a + 1/b = {sigma:.16f}   "
         f"kappa = -ln sigma = {kap:.16f}")
    note(f"  (K18) p1 = {p1:.16f}   p2 = {p2:.16f}")
    note(f"  (K19) lambda = p1 ln a + p2 ln b = {lam:.16f}")
    note(f"  (K20) H = {ent:.16f} = lambda - kappa = {lam - kap:.16f}   "
         f"(deviation {abs(ent - (lam - kap)):.3e})")
    note(f"        D_1 = H / lambda = {d1:.16f} = 1 - kappa/lambda = "
         f"{1.0 - kap / lam:.16f}")
    note(f"  (K21) D_0 = Moran root of a^-s + b^-s = 1 = {d0:.16f}")
    note(f"        D_0 - D_1 = {d0 - d1:+.6e}  -- the saddle is multifractal, "
         f"and the identity is about D_1 only.")
    assert abs(ent - (lam - kap)) < 1e-14, "K20 closed form does not close"
    assert abs(d0 - d1) > 1e-4, (
        "K21 the planted bed is not multifractal, so it cannot show the "
        "D_0/D_1 substitution error")

    note()
    note("  ---- Route 1: kappa from survivor counts alone ----")
    rng = np.random.default_rng(SEED + 4)
    (kh, kse, lh, lse, lcnt, counts,
     tmaxfit) = tri_measure_kappa_lambda(a, b, TRI_N0, TRI_T, rng)
    note(f"  N0 = {TRI_N0} uniform seeds, rng seed {SEED + 4}; survivor counts "
         f"n(t) for t = 0..{tmaxfit} (fit floor n >= 200)")
    note("        t   n(t)        n(t)/N0        predicted sigma^t")
    for t, c in enumerate(counts[:tmaxfit + 1]):
        note(f"      {t:3d}   {c:>9d}   {c / TRI_N0:.6e}   {sigma ** t:.6e}")
    note(f"  MEASURED  kappa_hat = {kh:.6f} +/- {kse:.6f}   "
         f"PREDICTED {kap:.6f}   deviation {abs(kh - kap):.3e} "
         f"({abs(kh - kap) / kse:.2f} s.e.)")
    assert abs(kh - kap) < 5.0 * kse, (
        f"CT route 1: kappa_hat = {kh:.6f} against {kap:.6f}, "
        f"{abs(kh - kap) / kse:.1f} standard errors")

    note()
    note("  ---- Route 2: lambda from ln|f'| along surviving orbits alone ----")
    note(f"  {lcnt} surviving transitions pooled; Benettin in d = 1 is exactly "
         f"the running sum of ln|f'| (K5 with a 1x1 QR).")
    note(f"  MEASURED  lambda_hat = {lh:.6f} +/- {lse:.6f}   "
         f"PREDICTED {lam:.6f}   deviation {abs(lh - lam):.3e} "
         f"({abs(lh - lam) / lse:.2f} s.e.)")
    assert abs(lh - lam) < 5.0 * lse, (
        f"CT route 2: lambda_hat = {lh:.6f} against {lam:.6f}, "
        f"{abs(lh - lam) / lse:.1f} standard errors")

    note()
    note("  ---- Route 3: d from the epsilon-scaling of the saddle measure "
         "alone ----")
    (d1h, d0h, sys1, sys0, rows, win, ncyl,
     cyl) = tri_measure_dimension(a, b, p1, p2, TRI_LEVEL, TRI_JMIN, TRI_JMAX)
    note(f"  level-{TRI_LEVEL} cylinder enumeration: {ncyl} intervals, longest "
         f"{cyl:.3e}, smallest epsilon {2.0 ** -TRI_JMAX:.3e}")
    note("        j    eps          I(eps)      occupied boxes")
    for j, eps, info, nb in rows:
        note(f"      {j:3d}   {eps:.6e}   {info:9.5f}   {nb:>8d}")
    note("  window dependence -- sliding 8-octave fits, the estimator's own "
         "error bar:")
    for lo, hi, f1, f0 in win:
        note(f"        j = {lo:2d}..{hi:2d}   D_1 = {f1:.6f}   D_0 = {f0:.6f}")
    note(f"  MEASURED  D_1_hat = {d1h:.6f} +/- {sys1:.6f} (window systematic)"
         f"   PREDICTED {d1:.6f}   deviation {abs(d1h - d1):.3e}")
    note(f"  MEASURED  D_0_hat = {d0h:.6f} +/- {sys0:.6f}"
         f"   PREDICTED {d0:.6f}   deviation {abs(d0h - d0):.3e}")
    assert abs(d1h - d1) < 3.0 * sys1, (
        f"CT route 3: D_1_hat = {d1h:.6f} against {d1:.6f}, deviation "
        f"{abs(d1h - d1):.3e} exceeds 3x its own window systematic "
        f"{sys1:.3e}")
    note(f"  NOTE: this leg is {sys1 / lse:.0f}x less precise than lambda and "
         f"{sys1 / kse:.0f}x less precise than kappa,")
    note("        and it is the only leg whose error is a fit systematic "
         "rather than a countable statistic.")

    note()
    note("  ---- The identity, checked on three numbers none of which used "
         "the other two ----")
    resid = kh - lh * (1.0 - d1h)
    unc = math.sqrt(kse ** 2 + ((1.0 - d1h) * lse) ** 2 + (lh * sys1) ** 2)
    share = (lh * sys1) ** 2 / unc ** 2
    note(f"  kappa_hat - lambda_hat (1 - D_1_hat) = {resid:+.6e}")
    note(f"  combined uncertainty (kappa s.e. {kse:.2e}, lambda s.e. "
         f"{lse:.2e}, D_1 systematic {sys1:.2e}) = {unc:.6e}")
    note(f"  the dimension leg supplies {share:.1%} of that variance.")
    note(f"  residual / uncertainty = {abs(resid) / unc:.2f}")
    note(f"  DISCRIMINATING POWER: the check can only reject a violation "
         f"larger than {unc / kh:.2%} of kappa.")
    assert abs(resid) < 5.0 * unc, (
        f"CT the identity fails on independently measured quantities: "
        f"residual {resid:.3e} against uncertainty {unc:.3e}")
    assert share > 0.9, (
        f"CT the dimension leg supplies only {share:.1%} of the error budget; "
        f"the claim that it is the binding constraint is wrong")

    note()
    note("  ---- Now the substitution that is actually made in practice ----")
    resid_d0 = kh - lh * (1.0 - d0h)
    note(f"  (K21) D_0 and D_1 differ in closed form by "
         f"{abs(d0 - d1):.3e}, a real multifractal gap.")
    note(f"        Feeding the BOX-COUNTING dimension to the identity: "
         f"kappa_hat - lambda_hat (1 - D_0_hat) = {resid_d0:+.6e}")
    note(f"        = {abs(resid_d0) / unc:.2f}x the uncertainty, "
         f"{abs(resid_d0) / kh:.2%} of kappa.")
    note(f"  FINDING: the closed-form D_0/D_1 gap ({abs(d0 - d1):.2e}) is "
         f"SMALLER than the direct estimator's own")
    note(f"        window systematic ({sys1:.2e}), so the triangle cannot "
         f"tell which dimension it was")
    note("        handed.  A check that does not notice the wrong dimension "
         "is not certifying the right one.")
    assert abs(resid_d0) < 3.0 * unc, (
        f"K21 the D_0-for-D_1 substitution WAS detected at "
        f"{abs(resid_d0) / unc:.1f} sigma; the report's claim that the "
        f"triangle cannot resolve it is wrong and must be rewritten")

    note()
    note("  ---- (K22) the circularity demonstration ----")
    note("  If d is obtained from the relation rather than measured, the "
         "'consistency check' is")
    note("  kappa - lambda (1 - (1 - kappa/lambda)) = 0, an algebraic "
         "identity.  Fed 10^5 random")
    note("  pairs -- including negative exponents, which describe no chaotic "
         "saddle at all:")
    rng2 = np.random.default_rng(SEED + 5)
    lam_r = rng2.uniform(-3.0, 3.0, 100_000)
    lam_r[np.abs(lam_r) < 1e-2] += 1.0
    kap_r = rng2.uniform(-3.0, 3.0, 100_000)
    d_r = 1.0 - kap_r / lam_r
    res_r = np.abs(kap_r - lam_r * (1.0 - d_r))
    rel_r = res_r / np.maximum(np.abs(kap_r), 1e-12)
    note(f"        max absolute residual = {float(res_r.max()):.3e}")
    note(f"        max relative residual = {float(rel_r.max()):.3e}")
    note(f"        fraction of nonsense pairs the check rejects at 1e-6 "
         f"relative: {float((rel_r > 1e-6).mean()):.6f}")
    assert float(rel_r.max()) < 1e-9, (
        "K22 the rearranged identity does not close to roundoff, which would "
        "mean the algebra above is wrong")
    assert float((rel_r > 1e-6).mean()) == 0.0, (
        "K22 the rearranged check rejected something; it is supposed to have "
        "zero discriminating power")
    note("        VERDICT: zero rejections out of 100000. A triangle whose "
         "third leg is")
    note("        computed from the other two certifies nothing.")
    return kh, kse, lh, lse, d1h, d0h, resid, unc, resid_d0


# ==========================================================================


def main():
    t_start = time.perf_counter()
    note("V13 X28b tangent kit -- self-check")
    note(f"numpy {np.__version__}, torch {torch.__version__}, "
         f"python {sys.version.split()[0]}, float64, seed {SEED}")
    check_i1()
    mf1 = check_i2_positive()
    mf2 = check_i2_negative()
    mf3 = check_i2_henon()
    i3 = check_i3()
    ct = check_triangle()

    rule("SUMMARY")
    note(f"  MF-1 logistic r=4    lambda_hat = {mf1[0]:.9f}  vs ln 2 = "
         f"{LN2:.9f}   |err| = {mf1[1]:.3e}   PASS")
    note(f"  MF-2 logistic r=3.2  lambda_hat = {mf2[0]:.9f}  vs ln 0.4 = "
         f"{mf2[2]:.9f}  |err| = {mf2[1]:.3e}   PASS (negative)")
    note(f"  MF-3 henon           l1+l2 = {mf3[0] + mf3[1]:+.9f}  vs ln 0.3 "
         f"= {math.log(0.3):+.9f}  |err| = {mf3[2]:.3e}   PASS")
    note(f"  I3   adjoint gradcheck at rtol = 1e-4: "
         f"{'PASS' if i3[0] else 'FAIL'}   max rel dev = {i3[1]:.3e}   "
         f"dt-slope = {i3[2]:+.2f}")
    note(f"  CT   kappa_hat = {ct[0]:.6f}+/-{ct[1]:.6f}  "
         f"lambda_hat = {ct[2]:.6f}+/-{ct[3]:.6f}  D1_hat = {ct[4]:.6f}")
    note(f"       identity residual {ct[6]:+.3e} against uncertainty "
         f"{ct[7]:.3e}")
    note(f"       same check with d taken from the relation: identically 0 "
         f"for every input, 0/100000 rejections")
    note()
    note(f"ALL ASSERTIONS PASSED   ({time.perf_counter() - t_start:.1f} s "
         f"wall clock)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

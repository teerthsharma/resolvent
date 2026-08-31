"""[V-eq] probe 5 -- the X35 COMPOSITION, run from 1993 textbook equations.

This probe exists because X35's claim is a composition claim: residual of a model
+ onset statistic + a latent node whose effect is re-propagated through the
model. The composition is written down, in closed form, in:

  Basseville & Nikiforov, "Detection of Abrupt Changes: Theory and Application",
  Prentice-Hall 1993 (free PDF, people.irisa.fr/Michele.Basseville/kniga/),
  Sec. 7.2.4 "State-Space Models".

Equations transcribed:
  (7.2.94)  X_{k+1} = F X_k + G U_k + W_k ;  Y_k = H X_k + J U_k + V_k
  (7.2.97)  X_{k+1} = F X_k + G U_k + W_k + Gamma Upsilon_x(k,t0)
            Y_k     = H X_k + J U_k + V_k +       Upsilon_y(k,t0)
            "Neither the gain matrices nor the profiles are necessarily
             completely known a priori. ... The instant t0 is again the unknown
             change time, so that Upsilon_x(k,t0) = Upsilon_y(k,t0) = 0 for k < t0."
  (7.2.100) the scalar-magnitude case: ... + nu Upsilon_x(k,t0), nu unknown.
  (7.2.109) X_k = X_k^0 + alpha(k,t0);  Xhat_k = Xhat_k^0 + beta(k,t0);
            eps_k = eps_k^0 + rho(k,t0)
  (7.2.110) alpha(k,t0) = F alpha(k-1,t0) + Gamma Upsilon_x(k-1,t0)
            beta(k,t0)  = (I - K H) F beta(k-1,t0) + K [H alpha(k,t0) + Upsilon_y(k,t0)]
            rho(k,t0)   = H [alpha(k,t0) - F beta(k-1,t0)] + Upsilon_y(k,t0)
            with alpha(t0,t0) = 0, beta(t0-1,t0) = 0.
  (7.2.126) nuhat_k(j) = [sum_{i=j..k} rho^T(i,j) Sigma_i^{-1} eps_i]
                       / [sum_{i=j..k} rho^T(i,j) Sigma_i^{-1} rho(i,j)]
  (7.2.125) sup_nu S_j^k = (1/2) [sum rho^T Sigma^{-1} eps]^2 / [sum rho^T Sigma^{-1} rho]
  (7.2.123) t0hat = argmax_{1<=j<=t_a} S_j^{t_a}
  Sec. 7.2.4: "the basic computation on which the detection is based is the
  correlation between the innovations eps of the Kalman filter and the
  signatures of the changes in (7.2.97) on these innovations."

Hypotheses as the source states them: linear time-invariant (F,G,H,J); W, V
independent Gaussian white noises with covariances Q, R; the innovation is the
output of the Kalman filter for the UNCHANGED model (7.2.94), so it is the
innovation before t0 and the residual after; the closed forms (7.2.111)-(7.2.112)
assume the steady-state Kalman filter, in which the signature depends only on
k - t0.

Measured here, on a model whose planted cause enters a state coordinate that is
never observed directly:
  1. planted cause -> onset recovered within +/- 1 (X35a MUST-FIRE 1) and
     magnitude recovered, over 400 seeds at three plant magnitudes;
  2. no plant -> no onset called at a threshold calibrated on no-plant runs
     (X35a MUST-FIRE 2, the SCOPE control);
  3. CONTROL: (7.2.110)'s third line mis-transcribed as rho(k,t0) = H alpha(k,t0),
     dropping the -F beta(k-1,t0) feedback term -- an O(1) mis-transcription of
     the signature that costs both the onset and the magnitude.
"""
import numpy as np

F = np.array([[0.90, 0.20], [0.00, 0.85]])
H = np.array([[1.0, 0.0]])
Q = np.diag([0.01, 0.01])
R = np.array([[0.05]])
Gam = np.array([[0.0], [1.0]])          # the hidden cause enters the UNOBSERVED coordinate
T, T0, NU = 120, 60, 0.15
JMAX = T - 20                            # candidate onsets need >= 20 samples of evidence


def steady_state():
    P = np.eye(2)
    for _ in range(5000):
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        P_upd = P - K @ H @ P
        P = F @ P_upd @ F.T + Q
    return K, float((H @ P @ H.T + R)[0, 0])


K, SIG = steady_state()


def signature(dmax, drop_feedback=False):
    """rho(k, t0) from (7.2.110) at steady state, as a function of d = k - t0.
    drop_feedback=True is the CONTROL: rho = H alpha, without the -F beta term."""
    rho = np.zeros(dmax + 1)
    alpha = np.zeros((2, 1))
    beta_prev = np.zeros((2, 1))
    for d in range(1, dmax + 1):
        alpha = F @ alpha + Gam * 1.0            # step profile Upsilon_x = 1 for k >= t0
        r = float((H @ (alpha if drop_feedback else alpha - F @ beta_prev))[0, 0])
        beta = F @ beta_prev + K * float((H @ (alpha - F @ beta_prev))[0, 0])
        rho[d] = r
        beta_prev = beta
    return rho


def innovations(seed, nu):
    rng = np.random.default_rng(seed)
    x = np.zeros((2, 1))
    xhat = np.zeros((2, 1))
    eps = np.zeros(T)
    for k in range(T):
        y = float((H @ x)[0, 0]) + rng.normal(0, np.sqrt(R[0, 0]))
        e = y - float((H @ xhat)[0, 0])
        eps[k] = e
        xhat = F @ (xhat + K * e)
        w = rng.multivariate_normal(np.zeros(2), Q).reshape(2, 1)
        x = F @ x + w + (nu * Gam if k >= T0 else 0.0)
    return eps


def glr(eps, rho):
    """(7.2.125)/(7.2.126)/(7.2.123): returns (g, t0hat, nuhat)."""
    best = (-np.inf, 0, 0.0)
    for j in range(1, JMAX + 1):
        d = np.arange(0, T - j)
        r = rho[d]
        e = eps[j:]
        num = float(r @ e) / SIG
        den = float(r @ r) / SIG
        if den <= 0:
            continue
        S = 0.5 * num ** 2 / den
        if S > best[0]:
            best = (S, j, num / den)
    return best


def report(rho, label):
    # threshold from no-plant runs (X35a: "Threshold calibrated on no-plant runs")
    g0 = np.array([glr(innovations(10_000 + s, 0.0), rho)[0] for s in range(400)])
    thr = np.quantile(g0, 0.95)
    print("== %s ==" % label)
    print("  no-plant threshold at 5%% FAR : g* = %.3f" % thr)
    print("  no plant  -> onset called    : %.3f   (X35a MUST-FIRE 2, target 0.05)"
          % (g0 > thr).mean())
    for nu in (0.15, 0.30, 0.60):
        res = [glr(innovations(s, nu), rho) for s in range(400)]
        g1 = np.array([r[0] for r in res])
        t0h = np.array([r[1] for r in res])
        nuh = np.array([r[2] for r in res])
        print("  nu = %.2f : detect %.3f | onset within +/-1 %.3f | median |t0hat-t0| %.1f"
              " | median nuhat %.4f" % (nu, (g1 > thr).mean(), (np.abs(t0h - T0) <= 1).mean(),
                                        np.median(np.abs(t0h - T0)), np.median(nuh)))


report(signature(T), "(7.2.110) signature, (7.2.123)/(7.2.126) GLR")
print()
report(signature(T, drop_feedback=True), "CONTROL: rho = H alpha, feedback term dropped")

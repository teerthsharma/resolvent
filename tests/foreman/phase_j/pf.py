"""PF-GAMMA: phase-field penalty P_eps(m) = sum_k [eps*(m_{k+1}-m_k)^2 + W(m_k)/eps],
W(m)=m^2(1-m)^2. Continuum analytic constant + discrete/refined-lattice minimisation."""
import math
import numpy as np
from scipy.optimize import minimize


def W(m):
    return m ** 2 * (1 - m) ** 2


def continuum_min_energy():
    """Continuum functional eps*m'^2 + W(m)/eps, profile 0->1.
    Equipartition (AM-GM equality): eps*m'^2 == W(m)/eps at the minimiser,
    so E_min = int [eps m'^2 + W/eps] dx = int 2*sqrt(eps*m'^2 * W/eps) dx
             = 2*int_0^1 sqrt(W(m)) dm = 2*int_0^1 m(1-m) dm = 2*(1/6) = 1/3.
    This is eps-independent, and is NOT sqrt(2)/6: the literal P_eps has
    coefficient `eps` (not eps/2) on the gradient term, so it differs from the
    Modica-Mortola convention that produces c_W by exactly a factor sqrt(2).
    """
    E_min = 2.0 * (1.0 / 6.0)  # = 1/3
    formula = "E_min = 2*int_0^1 sqrt(W(m)) dm = 2*int_0^1 m(1-m) dm = 1/3"
    return E_min, formula


def _objective_unit(x, n, eps):
    m = np.empty(n + 1)
    m[0] = 0.0
    m[-1] = 1.0
    m[1:-1] = x
    diffs = m[1:] - m[:-1]
    grad_term = eps * np.sum(diffs ** 2)
    w_term = np.sum(W(m[:-1])) / eps  # k = 0..n-1
    return grad_term + w_term


def discrete_min(eps, n=400):
    x0 = np.linspace(0, 1, n + 1)[1:-1]
    res = minimize(_objective_unit, x0, args=(n, eps), method="L-BFGS-B",
                    bounds=[(0, 1)] * (n - 1))
    m = np.empty(n + 1)
    m[0], m[-1] = 0.0, 1.0
    m[1:-1] = res.x
    count_mid = int(np.sum((m > 0.05) & (m < 0.95)))
    return float(res.fun), count_mid


def _objective_refined(x, n, eps, h):
    m = np.empty(n + 1)
    m[0] = 0.0
    m[-1] = 1.0
    m[1:-1] = x
    diffs = (m[1:] - m[:-1]) / h
    grad_term = eps * np.sum(diffs ** 2) * h
    w_term = np.sum(W(m[:-1])) * h / eps
    return grad_term + w_term


def refined_min(eps, n=400):
    h = eps / 20.0
    x0 = np.linspace(0, 1, n + 1)[1:-1]
    res = minimize(_objective_refined, x0, args=(n, eps, h), method="L-BFGS-B",
                    bounds=[(0, 1)] * (n - 1))
    return float(res.fun)


if __name__ == "__main__":
    E_min, formula = continuum_min_energy()
    c_W = math.sqrt(2) / 6
    print(f"(1) {formula}")
    print(f"    E_min = {E_min:.6f}  c_W = {c_W:.6f}  equal? {abs(E_min - c_W) < 1e-9}")
    print()
    print("(2) unit-spacing discrete, n=400:")
    for eps in (1, 0.3, 0.1, 0.03):
        P_min, count_mid = discrete_min(eps, n=400)
        print(f"    eps={eps:<5} P_min={P_min:.6f}  n_mid(0.05<m<0.95)={count_mid}")
    print()
    print("(3) refined lattice h=eps/20, n=400:")
    for eps in (1, 0.3, 0.1, 0.03):
        P_ref = refined_min(eps, n=400)
        print(f"    eps={eps:<5} h={eps/20:.5f}  P_min={P_ref:.6f}")

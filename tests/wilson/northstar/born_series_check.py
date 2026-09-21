"""
Verification, not trust: checks the Born-series / Neumann-series identity and the
worked-instance numbers claimed on docs/NORTH_STAR.md before Wilson signs off on them.

T = (I - V G0)^-1 V   (Lippmann-Schwinger, closed form)
Born series: T ~= sum_{k=0}^{N} (V G0)^k V, converges iff rho(G0 V) < 1.

demo() is the runnable self-check: asserts the convergent case matches the page's
claimed rho and truncation errors, and that the divergent case (E inside the H0 band)
diverges rather than converges.
"""
import numpy as np


def make_instance(n, e_real, seed=0):
    rng = np.random.default_rng(seed)
    h0_diag = rng.uniform(1, 3, n).astype(np.complex128)
    H0 = np.diag(h0_diag)
    A = rng.normal(0, 0.1, (n, n))
    V = ((A + A.T) / 2).astype(np.complex128)
    E = e_real + 0.05j
    G0 = np.linalg.inv(E * np.eye(n) - H0)
    return H0, V, G0


def spectral_radius(M):
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def born_truncation_error(V, G0, n_hops, T_exact):
    n = V.shape[0]
    term = V.copy()
    T_approx = V.copy()
    for _ in range(n_hops):
        term = V @ G0 @ term
        T_approx = T_approx + term
    return float(np.linalg.norm(T_approx - T_exact, ord='fro'))


def run_case(n, e_real, seed, label):
    H0, V, G0 = make_instance(n, e_real, seed)
    rho = spectral_radius(G0 @ V)
    T_exact = np.linalg.inv(np.eye(n) - V @ G0) @ V
    errs = {k: born_truncation_error(V, G0, k, T_exact) for k in (5, 10, 20)}
    print(f"{label}: rho(G0V)={rho:.4f}  err5={errs[5]:.2e}  err10={errs[10]:.2e}  err20={errs[20]:.2e}")
    return rho, errs


def demo():
    rho_c, errs_c = run_case(12, 4.0, 0, "convergent (E=4+0.05i)")
    assert abs(rho_c - 0.2596) < 5e-4, f"rho mismatch: {rho_c}"
    assert rho_c < 1.0
    # errors must fall geometrically (each ~3-4 orders of magnitude tighter every 10 hops)
    assert errs_c[20] < errs_c[10] < errs_c[5]
    assert errs_c[5] < 5e-4 and errs_c[10] < 1e-6 and errs_c[20] < 1e-11

    rho_d, errs_d = run_case(12, 2.0, 0, "divergent (E=2+0.05i, E inside H0 band)")
    assert rho_d > 1.0, f"expected divergence, got rho={rho_d}"
    assert abs(rho_d - 1.6688) < 5e-3, f"rho mismatch: {rho_d}"
    # truncation error must GROW with more hops when rho > 1
    assert errs_d[20] > errs_d[10] > errs_d[5]
    print("demo(): all assertions passed.")


if __name__ == "__main__":
    demo()

"""Horn's closed-form absolute orientation by unit quaternions.

Given two corresponded, centred point sets P, Q (N x 3, row-index correspondence
P[i] <-> Q[i]), recover the rotation R minimizing sum_i ||Q[i] - R @ P[i]||^2 via
the top eigenvector of the 4x4 symmetric matrix built from the cross-covariance
sum_i outer(P[i], Q[i]) (Horn 1987, J. Opt. Soc. Am. A).

Correspondence convention (limit, stated per contract): row i of P corresponds to
row i of Q; centroids are removed internally before building the cross-covariance.
"""
import numpy as np


def horn_align(P, Q):
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)

    H = Pc.T @ Qc  # 3x3 cross-covariance, sum_i outer(P_i, Q_i)
    Sxx, Sxy, Sxz = H[0]
    Syx, Syy, Syz = H[1]
    Szx, Szy, Szz = H[2]

    N = np.array([
        [Sxx + Syy + Szz, Syz - Szy,       Szx - Sxz,       Sxy - Syx],
        [Syz - Szy,       Sxx - Syy - Szz, Sxy + Syx,       Szx + Sxz],
        [Szx - Sxz,       Sxy + Syx,      -Sxx + Syy - Szz, Syz + Szy],
        [Sxy - Syx,       Szx + Sxz,       Syz + Szy,      -Sxx - Syy + Szz],
    ])

    eigvals, eigvecs = np.linalg.eigh(N)
    q = eigvecs[:, np.argmax(eigvals)]  # top eigenvector = optimal quaternion (w,x,y,z)
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])
    return R


def rotation_angle_deg(R):
    """Angle of rotation matrix R in degrees, via trace formula."""
    c = (np.trace(R) - 1.0) / 2.0
    c = np.clip(c, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def demo():
    rng = np.random.default_rng(42)
    P = rng.standard_normal((50, 3))
    P -= P.mean(axis=0)
    A = rng.standard_normal((3, 3))
    Qm, Rm = np.linalg.qr(A)
    Qm = Qm @ np.diag(np.sign(np.diag(Rm)))
    if np.linalg.det(Qm) < 0:
        Qm[:, 0] *= -1
    Q = P @ Qm.T
    R_est = horn_align(P, Q)
    err = rotation_angle_deg(R_est.T @ Qm)
    assert err < 1e-8, f"demo self-check failed: {err} deg"
    print("horn.py self-check OK, angle error (deg):", err)


if __name__ == "__main__":
    demo()

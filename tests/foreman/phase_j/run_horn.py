"""Runner: executes the Horn checks, prints measured numbers (no forced pass)."""
import numpy as np
from horn import horn_align, rotation_angle_deg


def random_rotation(rng):
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def case_a():
    rng = np.random.default_rng(0)
    P = rng.standard_normal((200, 3))
    P -= P.mean(axis=0)
    R_true = random_rotation(rng)
    Q = P @ R_true.T
    R_est = horn_align(P, Q)
    err_no_noise_rad = np.radians(rotation_angle_deg(R_est.T @ R_true))

    rng2 = np.random.default_rng(1)
    P2 = rng2.standard_normal((200, 3))
    P2 -= P2.mean(axis=0)
    R_true2 = random_rotation(rng2)
    Q2 = P2 @ R_true2.T + rng2.normal(scale=0.01, size=P2.shape)
    R_est2 = horn_align(P2, Q2)
    err_noise_rad = np.radians(rotation_angle_deg(R_est2.T @ R_true2))
    return err_no_noise_rad, err_noise_rad


def case_b():
    rng = np.random.default_rng(2)
    base = rng.standard_normal((200, 3))
    base -= base.mean(axis=0)
    n_seg = 12
    boundaries = {2, 4, 6, 8, 10}
    Rs = []
    cur = np.eye(3)
    for k in range(n_seg):
        if k in boundaries:
            cur = random_rotation(rng)
        Rs.append(cur)
    clouds = [base @ R.T for R in Rs]

    drift_angles = []
    for k in range(1, n_seg):
        R_k = horn_align(base, clouds[k])
        R_km1 = horn_align(base, clouds[k - 1])
        rel = R_k @ R_km1.T
        drift_angles.append(rotation_angle_deg(rel))

    boundary_idx = sorted(b - 1 for b in boundaries)
    non_boundary_idx = [i for i in range(len(drift_angles)) if i not in boundary_idx]
    min_boundary = min(drift_angles[i] for i in boundary_idx)
    max_nonboundary = max(drift_angles[i] for i in non_boundary_idx)
    hits = sum(1 for i in boundary_idx if drift_angles[i] > max_nonboundary)
    agreement = hits / len(boundary_idx)
    return drift_angles, boundary_idx, non_boundary_idx, agreement


def case_c(noise_floor_rad):
    rng = np.random.default_rng(3)
    base = rng.standard_normal((200, 3))
    base -= base.mean(axis=0)
    R = random_rotation(rng)
    cloud = base @ R.T
    half = 100
    R_a = horn_align(base[:half], cloud[:half])
    R_b = horn_align(base[half:], cloud[half:])
    rel = R_b @ R_a.T
    null_angle_rad = np.radians(rotation_angle_deg(rel))
    return null_angle_rad, null_angle_rad < noise_floor_rad


if __name__ == "__main__":
    err_no_noise, err_noise = case_a()
    print("A no-noise angle error (rad):", err_no_noise, "PASS<=1e-9:", err_no_noise <= 1e-9)
    print("A noise sigma=0.01 angle error (rad):", err_noise)

    drift_angles, boundary_idx, non_boundary_idx, agreement = case_b()
    print("B drift_angles (deg):", [round(x, 4) for x in drift_angles])
    print("B boundary_idx:", boundary_idx, "agreement:", agreement, "PASS>=0.8:", agreement >= 0.8)

    null_angle, null_pass = case_c(err_no_noise)
    print("C null split angle (rad):", null_angle, "noise_floor(rad from A):", err_no_noise,
          "PASS(below floor):", null_pass)

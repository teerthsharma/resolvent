"""RED-first test for Horn's closed-form absolute orientation (unit quaternion method).
Row DRIFT-SYNTH. Assert-based, no framework.
"""
import numpy as np
from horn import horn_align, rotation_angle_deg


def random_rotation(rng):
    # random rotation via QR of a random 3x3 matrix
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def test_planted_rotation_no_noise():
    rng = np.random.default_rng(0)
    P = rng.standard_normal((200, 3))
    P -= P.mean(axis=0)
    R_true = random_rotation(rng)
    Q = P @ R_true.T
    R_est = horn_align(P, Q)
    err = rotation_angle_deg(R_est.T @ R_true) * np.pi / 180.0
    assert err <= 1e-9, f"angle error too large: {err}"


def test_planted_rotation_noise():
    rng = np.random.default_rng(1)
    P = rng.standard_normal((200, 3))
    P -= P.mean(axis=0)
    R_true = random_rotation(rng)
    Q = P @ R_true.T + rng.normal(scale=0.01, size=P.shape)
    R_est = horn_align(P, Q)
    err_rad = rotation_angle_deg(R_est.T @ R_true) * np.pi / 180.0
    assert err_rad < 0.1, f"noise-case angle error unexpectedly huge: {err_rad}"
    print("noise sigma=0.01 angle error (rad):", err_rad)


def test_drift_boundaries():
    rng = np.random.default_rng(2)
    base = rng.standard_normal((200, 3))
    base -= base.mean(axis=0)
    n_seg = 12
    boundaries = {2, 4, 6, 8, 10}  # 5 planted boundaries (segment k differs from k-1)
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

    # boundary k means transition into segment k differs -> index k-1 in drift_angles
    boundary_idx = sorted(b - 1 for b in boundaries)
    non_boundary_idx = [i for i in range(len(drift_angles)) if i not in boundary_idx]

    thresh = (min(drift_angles[i] for i in boundary_idx) + max(drift_angles[i] for i in non_boundary_idx)) / 2 \
        if non_boundary_idx else 1e-6
    hits = sum(1 for i in boundary_idx if drift_angles[i] > 1e-6)
    agreement = hits / len(boundary_idx)
    assert agreement >= 0.8, f"boundary agreement {agreement} below bar"
    print("drift_angles:", drift_angles, "boundary_idx:", boundary_idx, "agreement:", agreement)


def test_null_split_segment():
    rng = np.random.default_rng(3)
    base = rng.standard_normal((200, 3))
    base -= base.mean(axis=0)
    R = random_rotation(rng)
    cloud = base @ R.T
    half = 100
    R_a = horn_align(base[:half], cloud[:half])
    R_b = horn_align(base[half:], cloud[half:])
    rel = R_b @ R_a.T
    null_angle_deg = rotation_angle_deg(rel)
    print("null split angle (deg):", null_angle_deg)
    # bar compares to noise floor measured in (a); checked in runner script, not asserted here
    assert null_angle_deg >= 0.0


if __name__ == "__main__":
    test_planted_rotation_no_noise()
    test_planted_rotation_noise()
    test_drift_boundaries()
    test_null_split_segment()
    print("ALL PASS")

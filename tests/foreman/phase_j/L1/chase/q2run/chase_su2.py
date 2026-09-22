"""Quaternion helpers, float64, CPU only. Own numpy version for seat Chase (row Q2/Q3).
Shared scratchpad su2.py was overwritten mid-run by another seat with an
incompatible torch API, so this module is kept private to avoid the race."""
import numpy as np

def qmul(p, q):
    w1, x1, y1, z1 = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    w2, x2, y2, z2 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.stack([w, x, y, z], axis=-1)

def qconj(q):
    q = np.asarray(q, dtype=np.float64).copy()
    q[..., 1:] *= -1
    return q

def qnorm(q):
    return np.linalg.norm(q, axis=-1)

def qinv(q):
    q = np.asarray(q, dtype=np.float64)
    n2 = np.sum(q * q, axis=-1, keepdims=True)
    return qconj(q) / n2

def qnormalize(q):
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)

def quat_from_axis_angle(axis, angle_rad):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    return np.array([np.cos(angle_rad/2), *(np.sin(angle_rad/2)*axis)], dtype=np.float64)

def qdist(p, q):
    return min(np.linalg.norm(p - q), np.linalg.norm(p + q))

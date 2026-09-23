"""Quaternion utilities for the 6-DOF simulator.

Conventions
-----------
- Scalar-first ``q = [q0, q1, q2, q3]``, Hamilton convention.
- ``q`` represents the rotation **from body frame B to inertial frame I**:
  ``v_I = R(q) v_B``.
- Kinematics: ``q_dot = 0.5 * q ⊗ [0, omega_B]`` with ``omega_B`` the body
  angular rate expressed in the body frame.
"""

from __future__ import annotations

import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Return ``q / |q|``; the zero quaternion maps to identity."""
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_multiply(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton product ``p ⊗ q`` (applies q first, then p)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    w1, x1, y1, z1 = p
    w2, x2, y2, z2 = q
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate (inverse for unit quaternions)."""
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """Direction cosine matrix R such that ``v_I = R v_B``."""
    q = quat_normalize(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def dcm_to_quat(R: np.ndarray) -> np.ndarray:
    """Shepperd's method: convert a rotation matrix to a unit quaternion."""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    i = int(np.argmax([tr, R[0, 0], R[1, 1], R[2, 2]]))
    if i == 0:
        s = np.sqrt(1.0 + tr) * 2.0
        q = np.array(
            [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
        )
    elif i == 1:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array(
            [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
        )
    elif i == 2:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = np.array(
            [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
        )
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = np.array(
            [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
        )
    if q[0] < 0:
        q = -q
    return quat_normalize(q)


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Quaternion for a rotation of ``angle`` radians about ``axis``."""
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n == 0.0:
        raise ValueError("axis must be nonzero")
    axis = axis / n
    s = np.sin(angle / 2.0)
    return np.array([np.cos(angle / 2.0), *(s * axis)])


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX (yaw-pitch-roll) Euler angles to quaternion, body→inertial."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def quat_to_euler(q: np.ndarray) -> tuple[float, float, float]:
    """Quaternion to ZYX Euler angles (roll, pitch, yaw) in radians."""
    R = quat_to_dcm(q)
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate body-frame vector ``v`` into the inertial frame: ``v_I = R(q) v_B``."""
    return quat_to_dcm(q) @ np.asarray(v, dtype=float)


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate inertial-frame vector ``v`` into the body frame: ``v_B = R(q)^T v_I``."""
    return quat_to_dcm(q).T @ np.asarray(v, dtype=float)


def quat_derivative(q: np.ndarray, omega_b: np.ndarray) -> np.ndarray:
    """Quaternion rate ``q_dot = 0.5 * q ⊗ [0, omega_B]``."""
    return 0.5 * quat_multiply(q, np.concatenate([[0.0], np.asarray(omega_b, dtype=float)]))


def quat_error(q_des: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Small-angle attitude error vector in the body frame.

    Returns ``2 * vec(q^-1 ⊗ q_des)`` with the shortest-path sign fix, i.e.
    the rotation vector that takes the current attitude to the desired one,
    expressed in the current body frame.
    """
    e = quat_multiply(quat_conjugate(quat_normalize(q)), quat_normalize(q_des))
    if e[0] < 0:
        e = -e
    return 2.0 * e[1:4]


def quat_integrate(q: np.ndarray, omega_b: np.ndarray, dt: float) -> np.ndarray:
    """Exact (exponential-map) integration of ``q`` under constant body rate ``omega_b``.

    ``q_new = q ⊗ exp(omega_B dt / 2)`` where the body-frame increment
    quaternion is ``[cos(|w|dt/2), sin(|w|dt/2) w/|w|]``.
    """
    q = quat_normalize(q)
    omega_b = np.asarray(omega_b, dtype=float)
    w = np.linalg.norm(omega_b)
    if w < 1e-12:
        return q
    half = w * dt / 2.0
    dq = np.concatenate([[np.cos(half)], np.sin(half) * omega_b / w])
    return quat_normalize(quat_multiply(q, dq))


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric cross-product matrix of a 3-vector."""
    x, y, z = np.asarray(v, dtype=float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def quat_from_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Quaternion rotating unit/non-unit vector ``a`` onto ``b`` (shortest arc)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = np.cross(a, b)
    d = float(np.dot(a, b))
    if d < -1.0 + 1e-9:
        # Antiparallel: pick any axis orthogonal to a.
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        return quat_from_axis_angle(axis, np.pi)
    q = np.concatenate([[1.0 + d], c])
    return quat_normalize(q)

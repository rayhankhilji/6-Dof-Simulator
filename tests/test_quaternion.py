import numpy as np
import pytest

from sixdof.math.quaternion import (
    dcm_to_quat,
    quat_error,
    quat_from_axis_angle,
    quat_from_euler,
    quat_from_two_vectors,
    quat_integrate,
    quat_multiply,
    quat_normalize,
    quat_rotate,
    quat_to_dcm,
    quat_to_euler,
)


def test_normalize():
    q = quat_normalize([2.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(q, [1, 0, 0, 0])
    np.testing.assert_allclose(np.linalg.norm(quat_normalize([1, 2, 3, 4])), 1.0)


def test_multiply_matches_dcm_composition():
    rng = np.random.default_rng(0)
    for _ in range(10):
        p = quat_normalize(rng.standard_normal(4))
        q = quat_normalize(rng.standard_normal(4))
        np.testing.assert_allclose(
            quat_to_dcm(quat_multiply(p, q)), quat_to_dcm(p) @ quat_to_dcm(q), atol=1e-12
        )


def test_dcm_orthonormal_det_one():
    q = quat_from_euler(0.3, -0.5, 1.2)
    R = quat_to_dcm(q)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)


def test_euler_round_trip():
    for angles in [(0.1, 0.2, 0.3), (-0.4, 0.3, -1.0), (0.0, 0.0, 0.0)]:
        r, p, y = quat_to_euler(quat_from_euler(*angles))
        np.testing.assert_allclose([r, p, y], angles, atol=1e-12)


def test_dcm_quat_round_trip():
    q = quat_from_euler(0.7, -0.4, 0.9)
    np.testing.assert_allclose(np.abs(dcm_to_quat(quat_to_dcm(q))), np.abs(q), atol=1e-12)


def test_quat_error_small_angle():
    q = quat_from_euler(0.0, 0.0, 0.0)
    q_des = quat_from_axis_angle([0, 0, 1], 0.01)
    err = quat_error(q_des, q)
    np.testing.assert_allclose(err, [0.0, 0.0, 0.01], atol=1e-6)
    # Error in own frame of a rotated attitude.
    q2 = quat_from_euler(0.5, -0.2, 0.3)
    q_des2 = quat_multiply(q2, quat_from_axis_angle([1, 0, 0], 0.02))
    np.testing.assert_allclose(quat_error(q_des2, q2), [0.02, 0, 0], atol=1e-6)


def test_quat_integrate_constant_rate():
    q0 = np.array([1.0, 0, 0, 0])
    w = np.array([0.0, 0.0, 0.5])  # 0.5 rad/s about z_B
    q1 = quat_integrate(q0, w, 1.0)
    np.testing.assert_allclose(q1, quat_from_axis_angle([0, 0, 1], 0.5), atol=1e-12)


def test_quat_from_two_vectors():
    rng = np.random.default_rng(1)
    for _ in range(10):
        a = rng.standard_normal(3)
        b = rng.standard_normal(3)
        q = quat_from_two_vectors(a, b)
        rot = quat_rotate(q, a / np.linalg.norm(a))
        np.testing.assert_allclose(rot, b / np.linalg.norm(b), atol=1e-10)


def test_rotate_inverse():
    q = quat_from_euler(0.2, -0.3, 0.5)
    v = np.array([1.0, 2.0, 3.0])
    from sixdof.math.quaternion import quat_rotate_inverse

    np.testing.assert_allclose(quat_rotate_inverse(q, quat_rotate(q, v)), v, atol=1e-12)

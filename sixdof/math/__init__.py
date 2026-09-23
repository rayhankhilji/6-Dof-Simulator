"""Math utilities: quaternions and rotations."""

from .quaternion import (
    dcm_to_quat,
    quat_conjugate,
    quat_derivative,
    quat_error,
    quat_from_axis_angle,
    quat_from_euler,
    quat_from_two_vectors,
    quat_integrate,
    quat_multiply,
    quat_normalize,
    quat_rotate,
    quat_rotate_inverse,
    quat_to_dcm,
    quat_to_euler,
    skew,
)

__all__ = [
    "dcm_to_quat",
    "quat_conjugate",
    "quat_derivative",
    "quat_error",
    "quat_from_axis_angle",
    "quat_from_euler",
    "quat_from_two_vectors",
    "quat_integrate",
    "quat_multiply",
    "quat_normalize",
    "quat_rotate",
    "quat_rotate_inverse",
    "quat_to_dcm",
    "quat_to_euler",
    "skew",
]
